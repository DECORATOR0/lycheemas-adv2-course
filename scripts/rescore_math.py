"""Uniformly rescore completed MATH outputs; default to a read-only full audit."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lychee_mas.course.data import score_answer

DEFAULT_REVISION = "math_notation_v2"
DEFAULT_REASON = (
    "Uniform deterministic equivalence scoring for TeX and Unicode arithmetic notation, "
    "explicit output-base numerals, degree notation in angle questions, explicitly "
    "requested root lists, and ordinal grade labels in grade questions. Predictions, "
    "references, and generation records are unchanged."
)


def rescore(run: Path, *, write: bool = False, revision: str = DEFAULT_REVISION,
            reason: str = DEFAULT_REASON) -> dict:
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    latest = {row["id"]: row for row in rows}
    changes, examples = [], []
    correct_after = 0
    for row in latest.values():
        old_score = row.get("score")
        checked = row["status"] == "ok"
        new_score = score_answer(row["prediction"], row["reference"], "math", row["question"]) if checked else old_score
        entry = {"id": row["id"], "status": row["status"], "checked": checked,
                 "question": row.get("question"), "prediction": row.get("prediction"),
                 "reference": row.get("reference"), "old_score": old_score, "new_score": new_score,
                 "correctness_changed": checked and bool(old_score["correct"]) != bool(new_score["correct"])}
        examples.append(entry)
        if entry["correctness_changed"]:
            changes.append(entry.copy())
        if checked:
            correct_after += bool(new_score["correct"])
            # Every checked example uses the same scorer, including unchanged ones.
            row.setdefault("original_score", old_score)
            row["score"] = new_score
            row["scoring_revision"] = revision
    artifact = {"scoring_revision": revision, "reason": reason,
                "checked_examples": sum(entry["checked"] for entry in examples),
                "selected_examples": len(latest), "changed_examples": changes,
                "correct_after": correct_after, "examples": examples}
    audit_path = run / "rescore_audit.json"
    if audit_path.exists():
        previous = json.loads(audit_path.read_text(encoding="utf-8"))
        # Flatten successive audits so earlier corrections and their reasons survive.
        prior = list(previous.pop("prior_audits", []))
        prior.append(previous)
        artifact["prior_audits"] = prior
    if write:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        summary["correct"] = correct_after
        summary["accuracy_full_denominator"] = correct_after / summary["selected_examples"]
        summary["scoring_review"] = audit_path.name
        summary["scoring_revision"] = revision
        (run / "results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                                  for row in rows), encoding="utf-8")
        (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        audit_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="update scores and preserve a complete audit")
    mode.add_argument("--dry-run", action="store_true", help="print the audit without writing (the default)")
    parser.add_argument("--scoring-revision", default=DEFAULT_REVISION)
    parser.add_argument("--reason", default=DEFAULT_REASON)
    args = parser.parse_args()
    print(json.dumps(rescore(args.run, write=args.write, revision=args.scoring_revision,
                             reason=args.reason), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
