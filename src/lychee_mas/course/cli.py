"""Resumable optimization and held-out evaluation with explicit artifacts."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import traceback
from pathlib import Path

from ..methods.prerun.maspo.optimizer import MASPOOptimizer
from ..methods.runtime.agentdropout_v2 import AgentDropoutV2Controller
from .data import check_disjoint, choose, read_rows, score_answer
from .io import ChatClient, JSONLines, TASK, write_json
from .workflow import AGENTS, CourseExecutor, fixed_indicators, make_view, run_langgraph


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["both", "optimize", "eval"], default="both")
    parser.add_argument("--benchmark", type=lambda value: "math" if value == "math500" else value,
                        choices=["gsm8k", "math", "arc_easy", "arc_challenge", "openbookqa"], required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible URL ending in /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-n", type=int, default=4)
    parser.add_argument("--eval-n", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--proposal-max-tokens", type=int, default=1536,
                        help="MASPO proposal output budget, independent of agent generation")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--disable-adv2", action="store_true",
                        help="disable only the evaluation-time ADv2 gate; keep prompts and execution unchanged")
    parser.add_argument("--engine", choices=["graphview", "langgraph"], default="graphview")
    return parser.parse_args(argv)


def summarize(results: dict, selected: list[dict], elapsed_s: float) -> dict:
    rows = [results[r["id"]] for r in selected if r["id"] in results]
    counts = {"pass": 0, "retry": 0, "reject": 0}
    accepted_after_retry = 0
    for row in rows:
        for node in row.get("nodes", []):
            gate = node.get("gate")
            if gate:
                for attempt in gate["attempts"]:
                    counts[attempt["decision"]] += 1
                accepted_after_retry += gate["decision"] == "pass" and gate["retries"] > 0
    complete = sum(r["status"] in ("ok", "terminal_rejected") for r in rows)
    correct = sum(bool(r.get("score", {}).get("correct")) for r in rows)
    return {
        "selected_examples": len(selected), "recorded_examples": len(rows),
        "completed_examples": complete, "errors": sum(r["status"] == "error" for r in rows),
        "terminal_rejected": sum(r["status"] == "terminal_rejected" for r in rows),
        "correct": correct, "accuracy_full_denominator": correct / len(selected),
        "gate_attempt_decisions": counts, "messages_accepted_after_retry": accepted_after_retry,
        "model_calls": sum(r.get("usage", {}).get("model_calls", 0) for r in rows),
        "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in rows),
        "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in rows),
        "invocation_elapsed_s": round(elapsed_s, 3),
        "status": "complete" if complete == len(selected) else "incomplete",
    }


async def run(args):
    if args.concurrency < 1 or args.depth < 1 or args.eval_batch < 2:
        raise ValueError("concurrency/depth must be positive; eval-batch must be at least 2")
    if args.proposal_max_tokens < 1:
        raise ValueError("proposal-max-tokens must be positive")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    train = choose(read_rows(args.train_file, args.benchmark, "train"), args.train_n, args.seed)
    test = choose(read_rows(args.test_file, args.benchmark, "test"), args.eval_n, args.seed)
    if len(train) < 2:
        raise ValueError("MASPO's two proposal subsets require at least two training questions")
    check_disjoint(train, test)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key not in ("phase", "output", "base_url", "timeout", "concurrency", "disable_adv2")}
    config.update({"enable_thinking": False, "rounds": 2, "agents": list(AGENTS),
                   "train_ids": [r["id"] for r in train], "test_ids": [r["id"] for r in test],
                   "indicators": [vars(i) for i in fixed_indicators(args.benchmark)],
                   "optimization_gate": False, "evaluation_gate": not args.disable_adv2,
                   "shared_prompts_across_rounds": True})
    config_path = output / "config.json"
    if config_path.exists():
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        # Runs created before this option used the same fixed proposal budget.
        saved_config.setdefault("proposal_max_tokens", 1536)
        if saved_config != json.loads(json.dumps(config)):
            raise ValueError("existing run configuration differs; choose a fresh output directory")
    selection = {"train": train, "test": test}
    selection_path = output / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text(encoding="utf-8")) != selection:
        raise ValueError("selected example content differs from the saved run; choose a fresh output directory")
    write_json(config_path, config)
    write_json(selection_path, selection)
    client = ChatClient(args.base_url, args.model, args.concurrency, output / "calls.jsonl",
                        args.max_tokens, args.timeout, os.environ.get("MODEL_API_KEY", ""),
                        proposal_max_tokens=args.proposal_max_tokens)
    view = make_view(args.benchmark)
    prompt_file = output / "prompts.json"
    if args.phase in ("both", "optimize") and not prompt_file.exists():
        print(f"MASPO: {args.benchmark}, train={len(train)}, depth={args.depth}", flush=True)
        token = TASK.set("optimization")
        started = time.monotonic()
        optimizer = MASPOOptimizer(
            mode="optimize", prompt_file=str(prompt_file),
            trainset=[row["question"] for row in train], agent_llm=client.generate,
            evaluator_llm=client.compare, proposer_llm=client.propose,
            max_total_depth=args.depth, rounds_per_turn=1, beam_width=args.beam_width,
            eval_batch=min(args.eval_batch, len(train)), seed=args.seed,
            max_concurrency=args.concurrency, executor_factory=CourseExecutor,
            optimize_nodes=list(AGENTS))
        prompts, statistics = await optimizer._optimize_with_limits(view)
        for agent in AGENTS:
            prompts[f"{agent}_r2"] = prompts[agent]
        statistics.update({"usage": client.statistics("optimization"),
                           "elapsed_s": round(time.monotonic() - started, 3),
                           "changed_roles": [a for a in AGENTS if prompts[a] != view.specs[a].system_prompt]})
        optimizer._save(prompts, statistics)
        payload = json.loads(prompt_file.read_text(encoding="utf-8"))
        payload["meta"].update({"integration": "five_roles_two_rounds_shared_prompts",
                                "prompt_texts": "course_authored", "train_ids": config["train_ids"],
                                "optimization_gate": False})
        write_json(prompt_file, payload)
        TASK.reset(token)
    if args.phase == "optimize":
        return
    if not prompt_file.exists():
        raise ValueError("eval needs prompts.json from a completed optimization run")
    payload = json.loads(prompt_file.read_text(encoding="utf-8"))
    prompts = payload["prompts"]
    if any(prompts[a] != prompts[f"{a}_r2"] for a in AGENTS):
        raise ValueError("the two rounds must share the five optimized role prompts")
    evaluation_prompt_path = output / "evaluation_prompts.json"
    if evaluation_prompt_path.exists() and json.loads(evaluation_prompt_path.read_text(encoding="utf-8")) != prompts:
        raise ValueError("evaluation prompts changed; results cannot be resumed with new prompts")
    write_json(evaluation_prompt_path, prompts)
    gate = (None if args.disable_adv2 else
            AgentDropoutV2Controller(client.judge, max_retries=args.max_retries,
                                    indicators=fixed_indicators(args.benchmark), top_k=None))
    result_log = JSONLines(output / "results.jsonl")
    results = {}
    if result_log.path.exists():
        for line in result_log.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                results[row["id"]] = row
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.monotonic()

    async def one(row):
        if results.get(row["id"], {}).get("status") in ("ok", "terminal_rejected"):
            return
        async with semaphore:
            token = TASK.set(row["id"])
            begin = time.monotonic()
            executor = CourseExecutor(view, client.generate, prompts, gate)
            record = {"id": row["id"], "question": row["question"], "reference": row["answer"]}
            try:
                if args.engine == "langgraph":
                    prediction, cache = await run_langgraph(view, executor, row["question"])
                else:
                    prediction, cache = await executor.run(row["question"])
                record.update({"status": "ok" if cache.node_outputs_raw.get("final") else "terminal_rejected",
                               "prediction": prediction, "final_output": cache.node_outputs_raw.get("final"),
                               "score": score_answer(prediction, row["answer"], args.benchmark,
                                                     row["question"])})
            except Exception as error:
                record.update({"status": "error", "error": f"{type(error).__name__}: {error}",
                               "traceback": traceback.format_exc(), "score": {"correct": False}})
            record.update({"nodes": executor.records, "usage": client.statistics(row["id"]),
                           "elapsed_s": round(time.monotonic() - begin, 3)})
            results[row["id"]] = record
            result_log.append(record)
            summary = summarize(results, test, time.monotonic() - started)
            write_json(output / "summary.json", summary)
            print(f"{args.benchmark} {summary['recorded_examples']}/{len(test)} {record['status']} correct={record['score']['correct']}", flush=True)
            TASK.reset(token)

    await asyncio.gather(*[one(row) for row in test])
    summary = summarize(results, test, time.monotonic() - started)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if summary["status"] != "complete":
        raise RuntimeError("some examples failed; inspect results.jsonl and resume this command")


def main(argv=None):
    args = arguments(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
