"""Reaggregate archived correctness records and verify complete populations."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
for name, entry in summary.items():
    if "records_file" not in entry:
        continue
    records = [json.loads(line) for line in (root / entry["records_file"]).read_text(encoding="utf-8").splitlines() if line]
    ids = [record["id"] for record in records]
    assert len(ids) == len(set(ids)) == entry["total"], name
    assert set(ids) == set(map(str, range(entry["total"]))), name
    assert all(isinstance(record["is_correct"], bool) for record in records), name
    correct = sum(record["is_correct"] for record in records)
    assert correct == entry["verification"]["correct"] == entry["original_summary"]["num_correct"], name
    if entry.get("full_records_file"):
        full_records = [json.loads(line) for line in (root / entry["full_records_file"]).read_text(encoding="utf-8").splitlines() if line]
        full = {row["id"]: row for row in full_records}
        assert len(full_records) == len(full) == len(records), name
        assert set(full) == set(ids), name
        for row in records:
            original = full[row["id"]]
            assert all(original.get(key) for key in ("question", "answer", "raw_response")), (name, row["id"])
            assert original["is_correct"] == row["is_correct"] and original["hypothesis"] == row["hypothesis"], (name, row["id"])
    print(f"{name}: {correct}/{len(records)} = {100 * correct / len(records):.2f}%")

comparison = json.loads((root / "current_paper_comparison.json").read_text(encoding="utf-8"))
for item in comparison["comparisons"]:
    old, new = summary[item["paper_entry"]], summary[item["repeat_entry"]]
    assert old["verification"]["correct"] == item["paper_correct"]
    assert new["verification"]["correct"] == item["repeat_correct"]
    assert old["total"] == new["total"] == item["total"]
    for key in ("reasoning_model", "supervisor_model", "selector_model", "embedding_model", "metric_pool_file", "embedding_cache_file", "mode", "max_turns", "retries_times", "pass_rate", "retrieve_p", "select_q", "math_team_variant", "batch_audit_metrics"):
        assert old["config"][key] == new["config"][key], (item["benchmark"], key)
print("Current-paper references, full response excerpts, and matched method parameters verified.")

