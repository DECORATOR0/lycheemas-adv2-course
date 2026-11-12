import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lychee_mas.course.cli import summarize, arguments, run as run_cli
from lychee_mas.course.data import check_disjoint, score_answer, read_rows
from lychee_mas.course.io import PURPOSE, ChatClient
from lychee_mas.course.workflow import AGENTS, CourseExecutor, fixed_indicators, make_view, safe_arithmetic, run_langgraph
from lychee_mas.methods.prerun.maspo.optimizer import MASPOOptimizer
from lychee_mas.methods.prerun.maspo.textops import parse_comparison_result
from lychee_mas.methods.runtime.agentdropout_v2 import AgentDropoutV2Controller


class CourseTests(unittest.IsolatedAsyncioTestCase):
    async def test_disable_adv2_only_removes_audits_from_cli_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path, test_path = root / "train.json", root / "test.json"
            train_path.write_text(json.dumps([
                {"id": "train1", "question": "First training task?", "answer": "A", "split": "train"},
                {"id": "train2", "question": "Second training task?", "answer": "B", "split": "train"},
            ]), encoding="utf-8")
            test_path.write_text(json.dumps([
                {"id": "test1", "question": "Test task? A. first B. second", "answer": "B", "split": "test"},
            ]), encoding="utf-8")
            view = make_view("arc_easy")
            prompts = {name: spec.system_prompt for name, spec in view.specs.items()}
            runs = {}
            for disabled in (False, True):
                output = root / ("off" if disabled else "on")
                output.mkdir()
                (output / "prompts.json").write_text(json.dumps({"prompts": prompts}), encoding="utf-8")
                argv = ["--phase", "eval", "--benchmark", "arc_easy", "--train-file", str(train_path),
                        "--test-file", str(test_path), "--output", str(output), "--train-n", "2",
                        "--eval-n", "1", "--base-url", "http://127.0.0.1:1/v1", "--model", "mock"]
                args = arguments(argv + (["--disable-adv2"] if disabled else []))
                self.assertEqual(args.disable_adv2, disabled)
                sent = []
                def response(payload):
                    sent.append(payload)
                    text = ('{"feedback":"checked","violations":[]}' if "response_format" in payload
                            else "The task supports B. <answer>B</answer>")
                    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
                with mock.patch.object(ChatClient, "_request", side_effect=response):
                    await run_cli(args)
                config = json.loads((output / "config.json").read_text(encoding="utf-8"))
                self.assertNotIn("disable_adv2", config)
                self.assertEqual(config["evaluation_gate"], not disabled)
                row = json.loads((output / "results.jsonl").read_text(encoding="utf-8"))
                self.assertEqual([node["node"] for node in row["nodes"]], view.names)
                self.assertEqual(row["prediction"], "B")
                self.assertEqual(sum("response_format" in p for p in sent), 0 if disabled else 11)
                if disabled:
                    self.assertTrue(all(node["gate"] is None for node in row["nodes"]))
                    self.assertTrue(all(node["initial_output"] == node["accepted_output"] for node in row["nodes"]))
                    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                    self.assertEqual(summary["gate_attempt_decisions"], {"pass": 0, "retry": 0, "reject": 0})
                else:
                    self.assertTrue(all(node["gate"]["decision"] == "pass" for node in row["nodes"]))
                runs[disabled] = (config, [p for p in sent if "response_format" not in p])
                # The persisted config has the old single evaluation_gate field;
                # default resume must still accept it and skip the completed example.
                with mock.patch.object(ChatClient, "_request") as request:
                    await run_cli(args)
                    request.assert_not_called()
                args.disable_adv2 = not disabled
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    await run_cli(args)
            on_config, on_generation = runs[False]
            off_config, off_generation = runs[True]
            self.assertEqual(len(on_generation), 11)
            self.assertEqual(on_generation, off_generation)
            self.assertEqual({k: v for k, v in on_config.items() if k != "evaluation_gate"},
                             {k: v for k, v in off_config.items() if k != "evaluation_gate"})

    async def test_arc_challenge_uses_choice_workflow_and_preserves_numeric_labels(self):
        args = arguments(["--benchmark", "arc_challenge", "--train-file", "train.jsonl",
                          "--test-file", "test.jsonl", "--output", "new-run",
                          "--base-url", "http://127.0.0.1:1/v1", "--model", "unused"])
        self.assertEqual(args.benchmark, "arc_challenge")
        view = make_view(args.benchmark)
        self.assertEqual(fixed_indicators("arc_challenge"), fixed_indicators("arc_easy"))
        seen_prompts = []
        async def llm(prompt):
            seen_prompts.append(prompt)
            return "The evidence supports option 2. <answer>2</answer>"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "challenge.jsonl"
            path.write_text(json.dumps({"id": "challenge:test:1", "question": "Which is a mammal?",
                                        "choices": {"label": ["1", "2", "3", "4"],
                                                    "text": ["trout", "whale", "ant", "frog"]},
                                        "answerKey": "2", "split": "test"}), encoding="utf-8")
            row = read_rows(path, "arc_challenge", "test")[0]
        self.assertEqual(row["answer"], "2")
        self.assertEqual(row["benchmark"], "arc_challenge")
        self.assertIn("2. whale", row["question"])
        executor = CourseExecutor(view, llm)
        prediction, cache = await executor.run(row["question"])
        self.assertEqual(prediction, "2")
        self.assertTrue(score_answer(prediction, "2", "arc_challenge")["correct"])
        self.assertFalse(score_answer(prediction, "3", "arc_challenge")["correct"])
        self.assertTrue(score_answer("(C)", "C", "arc_challenge")["correct"])
        self.assertIn("option label exactly as shown", seen_prompts[0])
        self.assertIn("scientific facts", seen_prompts[2])
        self.assertNotIn("Arithmetic tool", cache.node_outputs_raw["C"])

    async def test_proposal_budget_is_independent_and_truncation_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            default = ChatClient("http://127.0.0.1:1/v1", "test", 1, root / "default.jsonl")
            client = ChatClient("http://127.0.0.1:1/v1", "test", 1, root / "calls.jsonl",
                                max_tokens=1024, proposal_max_tokens=3072)
            def response(payload):
                return {"choices": [{"message": {"content": "<prompt>instruction</prompt>"},
                                     "finish_reason": "stop"}], "usage": {"completion_tokens": 5}}
            with mock.patch.object(default, "_request", side_effect=response) as request:
                await default.propose("improve the instruction")
                self.assertEqual(request.call_args.args[0]["max_tokens"], 1536)
            with mock.patch.object(client, "_request", side_effect=response) as request:
                await client.generate("ordinary agent task")
                self.assertEqual(request.call_args.args[0]["max_tokens"], 1024)
                await client.propose("improve the instruction")
                self.assertEqual(request.call_args.args[0]["max_tokens"], 3072)
                self.assertEqual(client.calls[-1]["max_tokens"], 3072)
            truncated = response({})
            truncated["choices"][0]["finish_reason"] = "length"
            with mock.patch.object(client, "_request", return_value=truncated):
                with self.assertRaisesRegex(ValueError, r"max_tokens=3072.*incomplete"):
                    await client.propose("improve the instruction")
            self.assertEqual(client.statistics()["truncated_calls"], 3)

    async def test_text_truncation_retries_concisely_with_accurate_call_records(self):
        with tempfile.TemporaryDirectory() as directory:
            client = ChatClient("http://127.0.0.1:1/v1", "test", 1, Path(directory) / "calls.jsonl",
                                max_tokens=2048)
            replies = [
                {"choices": [{"message": {"content": "LOOP_MARKER " * 20}, "finish_reason": "length"}],
                 "usage": {"prompt_tokens": 20, "completion_tokens": 2048}},
                {"choices": [{"message": {"content": "Complete response."}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 65, "completion_tokens": 8}},
            ]
            with mock.patch.object(client, "_request", side_effect=replies) as request:
                result = await client.call("Original task only.", temperature=0.25)
            self.assertEqual(result, "Complete response.")
            self.assertEqual(request.call_count, 2)
            first, second = client.calls
            self.assertEqual(first["prompt"], "Original task only.")
            self.assertFalse(first["output_complete"])
            self.assertEqual(second["retry_reason"], "truncated_text")
            self.assertEqual([first["attempt"], second["attempt"]], [0, 1])
            self.assertEqual([first["format_attempt"], second["format_attempt"]], [0, 1])
            self.assertEqual(second["temperature"], 0.25)
            self.assertEqual(second["max_tokens"], 2048)
            self.assertNotIn("LOOP_MARKER", second["prompt"])
            self.assertIn("Do not repeat", second["prompt"])
            for invocation, event in zip(request.call_args_list, client.calls):
                sent = invocation.args[0]
                self.assertEqual(sent["messages"][0]["content"], event["prompt"])
                self.assertEqual(sent["temperature"], event["temperature"])
            self.assertEqual(client.statistics()["completion_tokens"], 2056)

    async def test_valid_json_at_length_limit_is_complete_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            client = ChatClient("http://127.0.0.1:1/v1", "test", 1, Path(directory) / "calls.jsonl")
            content = '{"feedback":"checked","violations":[]}' + " \n" * 100
            reply = {"choices": [{"message": {"content": content}, "finish_reason": "length"}]}
            with mock.patch.object(client, "_request", return_value=reply) as request:
                actual = await client.judge("Audit this message.")
            self.assertEqual(json.loads(actual), {"feedback": "checked", "violations": []})
            self.assertEqual(request.call_count, 1)
            self.assertEqual(client.calls[0]["finish_reason"], "length")
            self.assertTrue(client.calls[0]["output_complete"])
            self.assertEqual(client.calls[0]["completeness_check"], "complete_json_object")

    async def test_incomplete_json_is_retried_and_never_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            client = ChatClient("http://127.0.0.1:1/v1", "test", 1, Path(directory) / "calls.jsonl")
            invalid = {"choices": [{"message": {"content": '{"feedback":"unfinished' + " \n" * 50},
                                     "finish_reason": "length"}]}
            valid = {"choices": [{"message": {"content": '{"feedback":"checked","violations":[]}'},
                                   "finish_reason": "stop"}]}
            with mock.patch.object(client, "_request", side_effect=[invalid, valid]):
                actual = await client.judge("Audit this message.")
            self.assertEqual(json.loads(actual)["violations"], [])
            self.assertFalse(client.calls[0]["output_complete"])
            self.assertEqual(client.calls[1]["retry_reason"], "invalid_json_object")
            self.assertIn("complete JSON object", client.calls[1]["prompt"])
            with mock.patch.object(client, "_request", return_value=invalid) as request:
                with self.assertRaisesRegex(ValueError, "invalid after 2 format retries"):
                    await client.judge("Audit this other message.")
            self.assertEqual(request.call_count, 3)
            self.assertTrue(all(not row["output_complete"] for row in client.calls[-3:]))

    async def test_resume_rejects_same_id_changed_data_or_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = [{"id": "tr1", "question": "First?", "answer": "1", "split": "train"},
                     {"id": "tr2", "question": "Second?", "answer": "2", "split": "train"}]
            test = [{"id": "te1", "question": "Third?", "answer": "3", "split": "test"}]
            train_path, test_path = root / "train.json", root / "test.json"
            train_path.write_text(json.dumps(train), encoding="utf-8")
            test_path.write_text(json.dumps(test), encoding="utf-8")
            output = root / "run"
            args = arguments(["--phase", "eval", "--benchmark", "gsm8k", "--train-file", str(train_path),
                              "--test-file", str(test_path), "--output", str(output),
                              "--base-url", "http://127.0.0.1:1/v1", "--model", "unused",
                              "--train-n", "2", "--eval-n", "1"])
            with self.assertRaisesRegex(ValueError, "completed optimization"):
                await run_cli(args)
            config_path = output / "config.json"
            old_config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(old_config.pop("proposal_max_tokens"), 1536)
            config_path.write_text(json.dumps(old_config), encoding="utf-8")
            # The old absent option means 1536, while a real change must not resume.
            with self.assertRaisesRegex(ValueError, "completed optimization"):
                await run_cli(args)
            args.proposal_max_tokens = 3072
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                await run_cli(args)
            args.proposal_max_tokens = 1536
            test[0]["answer"] = "999"
            test_path.write_text(json.dumps(test), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "example content differs"):
                await run_cli(args)
            test[0]["answer"] = "3"
            test_path.write_text(json.dumps(test), encoding="utf-8")
            prompts = {name: spec.system_prompt for name, spec in make_view("gsm8k").specs.items()}
            (output / "prompts.json").write_text(json.dumps({"prompts": prompts}), encoding="utf-8")
            changed_prompts = {**prompts, "final": "changed"}
            (output / "evaluation_prompts.json").write_text(json.dumps(changed_prompts), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prompts changed"):
                await run_cli(args)

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "optional langgraph dependency")
    async def test_native_langgraph_and_graphview_match(self):
        async def llm(prompt):
            return f"node={PURPOSE.get()} <answer>A</answer>"
        async def judge(prompt):
            payload = json.loads(prompt.rsplit("\n", 1)[1])
            return json.dumps({"feedback": "checked", "violations":
                               ["task_fidelity"] if payload["agent"] == "B" else []})
        view = make_view("arc_easy")
        gate = AgentDropoutV2Controller(judge, max_retries=0, indicators=fixed_indicators("arc_easy"))
        direct = CourseExecutor(view, llm, gate=gate)
        native = CourseExecutor(view, llm, gate=gate)
        expected, direct_cache = await direct.run("Choose A")
        actual, native_cache = await run_langgraph(view, native, "Choose A")
        self.assertEqual(actual, expected)
        self.assertEqual(native_cache.node_inputs, direct_cache.node_inputs)
        self.assertEqual(native_cache.node_outputs_raw, direct_cache.node_outputs_raw)
        self.assertEqual(native.records, direct.records)

    def test_topology_exact_spatial_temporal(self):
        view = make_view("gsm8k")
        self.assertEqual(view.predecessors["A"], [])
        self.assertEqual(view.predecessors["C"], ["A", "B"])
        self.assertEqual(view.predecessors["A_r2"], ["A"])
        self.assertEqual(view.predecessors["C_r2"], ["A_r2", "B_r2", "A", "B", "C"])
        self.assertEqual(view.predecessors["final"], ["A_r2", "B_r2", "C_r2", "D_r2", "E_r2"])

    async def test_rejection_is_absent_in_spatial_temporal_and_final(self):
        async def llm(prompt):
            return f"output-{PURPOSE.get()} <answer>7</answer>"
        async def judge(prompt):
            payload = json.loads(prompt.rsplit("\n", 1)[1])
            blocked = payload["agent"] in ("B", "C_r2")
            return json.dumps({"feedback": "contradiction" if blocked else "checked",
                               "violations": ["task_fidelity"] if blocked else []})
        gate = AgentDropoutV2Controller(judge, max_retries=0, indicators=fixed_indicators("gsm8k"))
        executor = CourseExecutor(make_view("gsm8k"), llm, gate=gate)
        prediction, cache = await executor.run("There are seven objects. How many?")
        self.assertEqual(prediction, "7")
        self.assertEqual(cache.node_outputs_raw["B"], "")
        records = {r["node"]: r for r in executor.records}
        self.assertNotIn("B", records["C"]["visible_sources"])
        self.assertNotIn("B", records["B_r2"]["visible_sources"])
        self.assertNotIn("C_r2", records["final"]["visible_sources"])
        self.assertNotIn("output-B <", records["C"]["input_context"])

    async def test_retry_propagates_only_corrected_output(self):
        async def llm(prompt):
            if PURPOSE.get() == "B":
                return "fixed <answer>2</answer>" if "Verifier feedback:" in prompt else "bad B"
            return "valid <answer>2</answer>"
        async def judge(prompt):
            payload = json.loads(prompt.rsplit("\n", 1)[1])
            bad = payload["candidate"] == "bad B"
            return json.dumps({"feedback": "Use 2" if bad else "Checked 2", "violations": ["reasoning_validity"] if bad else []})
        gate = AgentDropoutV2Controller(judge, max_retries=1, indicators=fixed_indicators("gsm8k"))
        executor = CourseExecutor(make_view("gsm8k"), llm, gate=gate)
        _, cache = await executor.run("1 + 1?")
        self.assertEqual(cache.node_outputs_raw["B"], "fixed <answer>2</answer>")
        self.assertIn("fixed", cache.node_inputs["C"])
        self.assertNotIn("bad B", cache.node_inputs["B_r2"])
        self.assertEqual(executor.records[1]["gate"]["attempts"][0]["decision"], "retry")

    async def test_candidate_prompt_is_shared_and_reexecutes_both_rounds(self):
        seen = []
        async def llm(prompt):
            seen.append((PURPOSE.get(), prompt))
            return "<answer>3</answer>"
        executor = CourseExecutor(make_view("arc_easy"), llm)
        _, base = await executor.run("Choose A")
        self.assertEqual([node for node, _ in seen], executor.view.names)
        seen.clear()
        _, candidate = await executor.run_from_node("B", base, "CANDIDATE {question} {context}")
        self.assertIn(("B",), [(node,) for node, text in seen if text.startswith("CANDIDATE")])
        self.assertIn(("B_r2",), [(node,) for node, text in seen if text.startswith("CANDIDATE")])
        self.assertNotIn("A", [node for node, _ in seen])
        self.assertEqual(candidate.node_outputs_raw["A"], base.node_outputs_raw["A"])
        self.assertIn("final", [node for node, _ in seen])

    async def test_actual_maspo_optimizes_five_shared_roles(self):
        async def llm(prompt):
            return "<answer>A</answer>"
        async def proposer(prompt):
            old = prompt.split("Current instruction:\n", 1)[1].split("\nObserved training", 1)[0]
            return "<prompt>Verify independently. " + old + "</prompt>"
        async def compare(prompt):
            return "The first output is better. <choose>A</choose>"
        view = make_view("arc_easy")
        with tempfile.TemporaryDirectory() as directory:
            optimizer = MASPOOptimizer(mode="optimize", prompt_file=str(Path(directory) / "prompts.json"),
                trainset=["Question one", "Question two"], agent_llm=llm, evaluator_llm=compare,
                proposer_llm=proposer, max_total_depth=1, rounds_per_turn=1, eval_batch=2,
                executor_factory=CourseExecutor, optimize_nodes=list(AGENTS), verbose=False)
            prompts, statistics = await optimizer._optimize_with_limits(view)
            self.assertEqual(set(statistics["final_scores"]), set(AGENTS))
            self.assertTrue(all(prompts[a].startswith("Verify independently.") for a in AGENTS))
            executor = CourseExecutor(view, llm, prompts)
            self.assertTrue(all(executor._template(a) == executor._template(a + "_r2") for a in AGENTS))

    def test_data_leakage_rejected(self):
        train = [{"id": "train:1", "question": "How many?", "split": "train"}]
        test = [{"id": "test:1", "question": " how MANY? ", "split": "test"}]
        with self.assertRaises(ValueError):
            check_disjoint(train, test)

    def test_math_and_choice_scoring(self):
        self.assertTrue(score_answer("0.5", r"\frac{1}{2}", "math")["correct"])
        self.assertTrue(score_answer(r"\sqrt{4}", "2", "math")["correct"])
        self.assertFalse(score_answer("", "0", "gsm8k")["correct"])
        self.assertFalse(score_answer("B", "A", "arc_easy")["correct"])
        self.assertFalse(score_answer("(1,23)", "(12,3)", "math")["correct"])
        self.assertFalse(score_answer(r"\text{no solution}", r"\text{all real numbers}", "math")["correct"])
        self.assertTrue(score_answer("1,234", "1234", "math")["correct"])
        self.assertFalse(score_answer("-21", "-2,1", "math")["correct"])
        self.assertFalse(score_answer("1000000001", "1000000000", "gsm8k")["correct"])
        self.assertFalse(score_answer("1e-9", "0", "gsm8k")["correct"])
        self.assertTrue(score_answer("7*pi", r"7\pi", "math")["correct"])
        self.assertTrue(score_answer("6-5*i", "6 - 5i", "math")["correct"])
        self.assertTrue(score_answer("1,-2", "-2,1", "math",
                                     "Find all integer roots, separated by commas.")["correct"])
        self.assertFalse(score_answer("(2,1)", "(1,2)", "math",
                                      "Find the point coordinates.")["correct"])

    def test_unicode_pi_matches_latex_pi_without_changing_value(self):
        self.assertTrue(score_answer("7π", r"7\pi", "math")["correct"])
        self.assertTrue(score_answer(r"7\pi", "7π", "math")["correct"])
        self.assertFalse(score_answer("7π", r"8\pi", "math")["correct"])

    def test_malformed_comparison_cannot_win(self):
        self.assertTrue(parse_comparison_result("reason <choose>A</choose>"))
        self.assertFalse(parse_comparison_result("reason <choose>B</choose>"))
        for malformed in ("123", "A", "<choose>AB</choose>", "<choose>A</choose><choose>B</choose>"):
            with self.assertRaises(ValueError):
                parse_comparison_result(malformed)

    def test_restricted_arithmetic_runs_and_refuses_external_operations(self):
        self.assertEqual(safe_arithmetic("apples = 8\nanswer = apples * 3 - 4"), "20")
        with self.assertRaises(ValueError):
            safe_arithmetic("import os\nanswer = 1")
        with self.assertRaises(ValueError):
            safe_arithmetic("answer = open('a').read()")

    def test_full_denominator_includes_errors_and_rejections(self):
        rows = [{"id": str(i)} for i in range(4)]
        results = {"0": {"status": "ok", "score": {"correct": True}},
                   "1": {"status": "error", "score": {"correct": False}},
                   "2": {"status": "terminal_rejected", "score": {"correct": False}}}
        summary = summarize(results, rows, 1)
        self.assertEqual(summary["accuracy_full_denominator"], 0.25)
        self.assertEqual(summary["completed_examples"], 2)
        self.assertEqual(summary["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
