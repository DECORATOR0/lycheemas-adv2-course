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
    print(f"{name}: {correct}/{len(records)} = {100 * correct / len(records):.2f}%")

