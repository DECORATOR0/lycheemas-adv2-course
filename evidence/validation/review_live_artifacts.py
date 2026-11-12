"""Independent read-only audit; writes separate reviews, preserving original artifacts."""
from __future__ import annotations
import json
import re
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "repo"
if not (REPO / "src").is_dir():
    REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
from lychee_mas.course.workflow import AGENTS, make_view, seed_prompts
from lychee_mas.methods.prerun.maspo.executor import format_agent_prompt

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def usage(events):
    prompt = sum(e["usage"].get("prompt_tokens", 0) for e in events)
    completion = sum(e["usage"].get("completion_tokens", 0) for e in events)
    return {"calls": len(events), "prompt_tokens": prompt,
            "completion_tokens": completion, "total_tokens": prompt + completion,
            "finish_reasons": dict(Counter(e["finish_reason"] for e in events)),
            "purposes": dict(Counter(e["purpose"] for e in events))}

def review(dataset):
    root = REPO / "results" / dataset
    config, selection, summary = [load(root / n) for n in ("config.json", "selection.json", "summary.json")]
    payload, saved_prompts, stats = [load(root / n) for n in ("prompts.json", "evaluation_prompts.json", "prompts_stats.json")]
    prompts = payload["prompts"]
    rows, calls = lines(root / "results.jsonl"), lines(root / "calls.jsonl")
    benchmark = config["benchmark"]
    view = make_view(benchmark)
    selected = {r["id"]: r for r in selection["test"]}
    train_ids = {r["id"] for r in selection["train"]}
    normalize = lambda s: " ".join(s.casefold().split())
    train_questions = {normalize(r["question"]) for r in selection["train"]}
    test_questions = {normalize(r["question"]) for r in selection["test"]}
    problems = []
    def verify(condition, description):
        if not condition:
            problems.append(description)
    verify(len(rows) == len(selected) == len({r["id"] for r in rows}), "results incomplete or duplicate")
    verify(set(selected) == {r["id"] for r in rows}, "unexpected result ID population")
    verify(not train_ids & set(selected), "overlapping train/test IDs")
    verify(not train_questions & test_questions, "overlapping train/test questions")
    verify(all(r["split"] == "train" for r in selection["train"]), "wrong training split")
    verify(all(r["split"] == "test" for r in selection["test"]), "wrong test split")
    verify(prompts == saved_prompts, "evaluated snapshot differs from optimized prompts")
    verify(all(prompts[a] == prompts[a + "_r2"] for a in AGENTS), "cross-round prompts diverge")
    changed = [a for a in AGENTS if prompts[a] != seed_prompts(benchmark)[a]]
    verify(changed == stats["changed_roles"], "changed-role statistic mismatch")
    optimization = [c for c in calls if c["task_id"] == "optimization"]
    evaluation = [c for c in calls if c["task_id"] in selected]
    verify(len(optimization) + len(evaluation) == len(calls), "unassigned call events")
    test_leaks = [r["id"] for r in selection["test"] if any(r["question"] in c["prompt"] for c in optimization)]
    verify(not test_leaks, "test question appears in optimization prompt")
    verify(all(c["enable_thinking"] is False for c in calls), "thinking was enabled")
    verify(not any(c["finish_reason"] == "length" for c in calls), "truncated generation")
    gate_counts = Counter()
    rejection_checks = []
    generation_checks = 0
    for row in rows:
        rid = row["id"]
        question = selected[rid]["question"]
        verify(row["question"] == question and row["reference"] == selected[rid]["answer"], rid + ": question/reference mismatch")
        verify([n["node"] for n in row["nodes"]] == view.names, rid + ": unexpected two-round execution order")
        accepted = {}
        task_calls = [c for c in evaluation if c["task_id"] == rid]
        for node in row["nodes"]:
            name = node["node"]
            sources = [p for p in view.predecessors[name] if accepted.get(p)]
            context = "\n\n".join(f"[{p}]\n{accepted[p]}" for p in sources)
            verify(node["visible_sources"] == sources, rid + ": visible-source mismatch at " + name)
            verify(node["input_context"] == context, rid + ": nonaccepted content in input at " + name)
            role = view.specs[name].meta.get("prompt_source", name)
            expected_prompt = format_agent_prompt(prompts[role], question, context)
            generated = [c for c in task_calls if c["purpose"] == name]
            verify(bool(generated) and generated[0]["prompt"] == expected_prompt, rid + ": optimized prompt not used at " + name)
            generation_checks += 1
            gate = node["gate"]
            gate_counts.update(a["decision"] for a in gate["attempts"])
            if gate["decision"] == "reject":
                verify(node["accepted_output"] is None and gate["content"] is None, rid + ": rejected message retained")
                later = [n for n in row["nodes"] if name in view.predecessors[n["node"]]]
                verify(all(name not in n["visible_sources"] for n in later), rid + ": rejected source propagated")
                rejection_checks.append({"task_id": rid, "node": name,
                                         "direct_consumers_checked": [n["node"] for n in later],
                                         "all_inputs_reconstructed_from_accepted_sources": True})
            else:
                verify(node["accepted_output"] == gate["content"] == gate["attempts"][-1]["content"], rid + ": accepted version mismatch")
            accepted[name] = node["accepted_output"]
        verify(row["final_output"] == accepted["final"], rid + ": final output mismatch")
        for c in task_calls:
            if c["purpose"] == "adv2_judge":
                judge_payload = json.loads(c["prompt"].rsplit("\n", 1)[1])
                verify(set(judge_payload) == {"task", "agent", "context", "candidate", "indicators"}, rid + ": unexpected judge fields")
                verify(judge_payload["task"] == question, rid + ": wrong judge task")
    verify(dict(gate_counts) == summary["gate_attempt_decisions"], "gate statistics mismatch")
    actual_correct = sum(r["score"]["correct"] for r in rows)
    verify(actual_correct == summary["correct"], "correct count mismatch")
    for key, value in [("model_calls", len(evaluation)), ("prompt_tokens", usage(evaluation)["prompt_tokens"]),
                       ("completion_tokens", usage(evaluation)["completion_tokens"])]:
        verify(summary[key] == value, "evaluation usage mismatch: " + key)
    verify(stats["usage"]["model_calls"] == len(optimization), "optimization call count mismatch")
    if benchmark == "gsm8k":
        for row in rows:
            try:
                independent = Decimal(row["prediction"].replace(",", "")) == Decimal(row["reference"].replace(",", ""))
            except InvalidOperation:
                independent = False
            verify(independent == row["score"]["correct"], row["id"] + ": independent numeric score differs")
    if benchmark in {"arc_easy", "openbookqa"}:
        for row in rows:
            independent = row["prediction"] == row["reference"]
            verify(independent == row["score"]["correct"], row["id"] + ": independent choice score differs")
    manual = [
        {"id": "gsm8k:test:563", "reference": "12", "prediction": "Insufficient information", "correct": False,
         "reason": "12 + (12 - 4) = 20 birthday candles; 20 / 5 * 3 = 12 dollars."},
        {"id": "gsm8k:test:1309", "reference": "2280", "prediction": "2180", "correct": False,
         "reason": "430 + (430 + 320) + 300 + (300 + 400) = 2280."},
        {"id": "gsm8k:test:209", "reference": "145", "prediction": "1450", "correct": False,
         "reason": "20 dozen = 240 cups; (6 * 6000 - 1200) / 240 = 145 per cup."},
        {"id": "gsm8k:test:54", "reference": "40", "prediction": "55", "correct": False,
         "reason": "Patchy has 3 * 7 = 21 kittens; 7 + 21 + 12 = 40. Final answer wrongly applies the factor 3 to Trixie's 12."},
    ] if benchmark == "gsm8k" else []
    corrected_cases = []
    if benchmark == "arc_easy":
        manual = [
            {"id": "arc_easy_test_Mercury_415745", "reference": "C", "prediction": "B", "correct": False,
             "reason": "Electromagnetic waves can propagate in vacuum, unlike mechanical waves; energy transmission is not exclusive to electromagnetic waves. The initial A answer C was correct, and judge feedback caused a wrong revision to B.",
             "source": "https://science.nasa.gov/ems/02_anatomy/"},
            {"id": "arc_easy_test_Mercury_416637", "reference": "A", "prediction": "C", "correct": False,
             "reason": "The dataset answer is A. The final response acknowledges shared reproduction by mitosis but dismisses it because it is not unique. The question asks for a shared characteristic, so this dismissal does not follow the question. Exact-option scoring is correct."},
        ]
    if benchmark == "openbookqa":
        manual = [{"id": "openbookqa_test_9-501", "reference": "A", "prediction": "None of the options", "correct": False,
                   "reason": "The final answer is not a provided option and does not match dataset label A. NPS attributes Mount McKinley's uplift to tectonic processes along the Denali fault system. This supports a factual concern with treating this item as an unambiguous scientific error; the benchmark label and 19/20 score are preserved.",
                   "source": "https://www.nps.gov/articles/nps-geodiversity-atlas-denali-national-park-preserve-alaska.htm"}]
    if benchmark in {"arc_easy", "openbookqa"}:
        for row in rows:
            for node in row["nodes"]:
                attempts = node["gate"]["attempts"]
                if node["gate"]["decision"] == "pass" and len(attempts) > 1:
                    answers = [re.findall(r"<answer>\s*(.*?)\s*</answer>", a["content"], re.S)[-1] for a in attempts]
                    first, last = answers[0] == row["reference"], answers[-1] == row["reference"]
                    classification = "wrong_to_correct" if not first and last else "correct_to_wrong" if first and not last else "correct_to_correct" if first and last else "wrong_to_wrong"
                    corrected_cases.append({"task_id": row["id"], "node": node["node"], "reference": row["reference"],
                                            "initial_answer": answers[0], "accepted_answer": answers[-1],
                                            "classification": classification})
        verify(len(corrected_cases) == summary["messages_accepted_after_retry"], "accepted-after-retry count differs")
    if benchmark == "openbookqa":
        detail = {
            ("openbookqa_test_9-501", "A"): "Dataset match decreases A to a non-option answer. The geological explanation raises a factual issue; this classification describes the unchanged benchmark-label criterion, not a definitive factual downgrade.",
            ("openbookqa_test_9-199", "A"): "A remains A; explanation of relative liquid densities is reworded.",
            ("openbookqa_test_9-199", "D"): "A remains A; the reviewer retracts its criticism of wording in an earlier agent's already-correct conclusion.",
            ("openbookqa_test_8-260", "B"): "C remains C, but revised reasoning says 'not C' before ultimately selecting C. A pass does not establish a repaired explanation.",
            ("openbookqa_test_1698", "E"): "Genuine option correction C to A: choosing message C had been confused with answering the question about corn-derived alcohol.",
            ("openbookqa_test_1759", "A"): "A remains A; a temperature wording clarification, not an answer correction.",
            ("openbookqa_test_1759", "D"): "Genuine option correction C to A: choosing message C had been confused with answering the thermometer question.",
            ("openbookqa_test_769", "final"): "C remains C; both versions retain inconsistent opening wording about more versus less water, while the final option matches the question and label.",
        }
        for item in corrected_cases:
            item["classification_basis"] = "Exact match with original dataset answer, without changing labels"
            item["classification"] = item["classification"].replace("correct", "label_matching").replace("wrong", "label_nonmatching")
            item["manual_reason"] = detail[(item["task_id"], item["node"])]
    out = {
        "review_date": "2026-11-12", "artifact_directory": "repo/results/" + dataset,
        "review_method": "Read original artifacts; reconstruct every accepted-input context and initial generation prompt; independently sum call events; inspect incorrect items and identified retry cases.",
        "records": {"test": len(rows), "training_actually_used": len(selection["train"]), "correct": actual_correct,
                    "errors": sum(r["status"] == "error" for r in rows)},
        "model": config["model"], "engine": config["engine"],
        "changed_roles": changed, "prompt_application_checks": generation_checks,
        "gate_attempt_decisions": dict(gate_counts), "rejection_checks": rejection_checks,
        "isolation": {"shared_ids": sorted(train_ids & set(selected)), "shared_question_count": len(train_questions & test_questions),
                      "test_questions_in_optimization": test_leaks, "judge_payload_keys_verified": True},
        "usage_all_phases": usage(calls), "usage_optimization": usage(optimization), "usage_test_only": usage(evaluation),
        "manual_incorrect_item_review": manual, "accepted_after_retry_item_review": corrected_cases,
        "required_code_or_score_fixes": problems,
        "interpretation_notes": [
            "MASPO used four training examples selected from the eight-example prepared training bank.",
            "In item 54, A and A_r2 initially produced the correct answer 40; the judge wrongly rejected it and suggested an incorrect interpretation. These events demonstrate functioning intervention and isolation, not successful error detection.",
            "summary.json call/token totals cover held-out evaluation only; prompts_stats.json covers optimization. Their sum agrees with calls.jsonl.",
        ] if benchmark == "gsm8k" else [
            "MASPO used four training examples selected from the eight-example prepared training bank.",
            "The three accepted-after-retry messages contain one wrong-to-correct, one correct-to-wrong and one correct-to-correct revision; they are not three verified error corrections.",
            "All eleven rejections were isolated from graph consumers. Rejection execution does not establish correctness of the judge decision.",
            "summary.json usage covers held-out evaluation only; optimization plus evaluation agrees with the complete calls.jsonl event stream.",
        ] if benchmark == "arc_easy" else [
            "MASPO used four training examples and retained the original prompts after evaluating candidates; zero changed roles does not mean optimization was skipped.",
            "Eight accepted-after-retry messages comprise two nonmatching-to-matching, five matching-to-matching, and one matching-to-non-option transition against the original dataset labels.",
            "Item 9-501 remains incorrect under the benchmark label A; NPS geological evidence warrants separating benchmark disagreement from factual-error claims.",
            "All 27 rejections are isolated from downstream consumers; this verifies intervention execution rather than judge correctness.",
            "summary.json usage covers held-out evaluation only; full calls.jsonl includes optimization.",
        ] if benchmark == "openbookqa" else [],
    }
    (HERE / (dataset + "_live_review.json")).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": dataset, "required_fixes": problems, "all_usage": usage(calls), "records": out["records"]}, ensure_ascii=False))

if __name__ == "__main__":
    review(sys.argv[1] if len(sys.argv) > 1 else "gsm8k")
