import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lychee_mas.course.data import score_answer

spec = importlib.util.spec_from_file_location("rescore_math", ROOT / "scripts" / "rescore_math.py")
rescore_math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rescore_math)


class MathNotationTests(unittest.TestCase):
    def check(self, prediction, reference, expected=True, question=""):
        self.assertEqual(score_answer(prediction, reference, "math", question)["correct"], expected,
                         (prediction, reference, question))

    def test_unicode_radicals_preserve_scope(self):
        self.check("2√5", r"2 \sqrt{5}")
        self.check("2√5", r"2\sqrt5")
        self.check("3√3⁄4", r"\frac{3\sqrt{3}}{4}")
        self.check("2√5", "√10", False)
        self.check("3√3⁄4", "3√(3/4)", False)
        self.check("2√5", "2√6", False)
        self.check("√(3/4)", r"\sqrt{\frac{3}{4}}")

    def test_tex_fraction_single_token_arguments(self):
        self.check("5/9", r"\frac 59")
        self.check("59", r"\frac 59", False)
        self.check("5/8", r"\frac 59", False)
        self.check("15/9", r"\frac{15}9")
        self.check("5/19", r"\frac 5{19}")
        self.check("5/9", r"\frac{5", False)

    def test_base_annotation_requires_requested_base(self):
        question = "Compute $58_9 - 18_9.$ Express your answer in base $9.$"
        self.check("40", "40_9", question=question)
        self.check("040", "40_{9}", question=question)
        self.check("40_8", "40_9", False, question)
        self.check("36", "40_9", False, question)
        self.check("49", "49_9", False, question)
        self.check("40", "40_9", False)
        self.check("25168", "2516_8", False)
        self.check("1247", "2516_8", False,
                   "The binary number $10101001110_2$ is equal to what number in base eight?")
        self.check("a_1", "a_1")
        self.check("a_2", "a_2")
        self.check("a_1", "a_2", False)

    def test_degree_suffix_in_angle_context(self):
        question = "What is the measure of angle A in the figure?"
        self.check("36", r"36^\circ", question=question)
        self.check("36°", r"36^{\circ}", question=question)
        self.check("36 rad", r"36^\circ", False, question)
        self.check("35", r"36^\circ", False, question)
        self.check("36", r"36^\circ", False)

    def test_interval_notation_keeps_endpoints_and_order(self):
        self.check("(-∞, 2) ∪ (3, ∞)", r"(-\infty, 2) \cup (3, \infty)")
        self.check("(-∞, 2] ∪ (3, ∞)", r"(-\infty, 2) \cup (3, \infty)", False)
        self.check("(3,4]", "[3,4)", False)
        self.check("(2,4)", "(4,2)", False)

    def test_plus_minus_only_expands_requested_root_lists(self):
        question = "Find all solutions. Enter all the solutions, separated by commas."
        self.check("3+2√2,3-2√2", r"3 \pm 2 \sqrt{2}", question=question)
        self.check("3-2√2,3+2√2", r"3 \pm 2 \sqrt{2}", question=question)
        self.check("3+2√2,3+2√2", r"3 \pm 2 \sqrt{2}", False, question)
        self.check("3+2√2", r"3 \pm 2 \sqrt{2}", False, question)
        self.check("3+2√2,3-2√2", r"3 \pm 2 \sqrt{2}", False)

    def test_grade_ordinals_require_explicit_grade_question(self):
        question = "For which grade is the number of male bus riders closest to 135% of female bus riders?"
        self.check("12th", "12", question=question)
        self.check("1st", "1", question="What grade is represented?")
        self.check("12th", "11", False, question)
        self.check("12st", "12", False, question)
        self.check("11st", "11", False, question)
        self.check("12th", "12", False, "What is the number?")
        self.check("12th", "12", False, "The students are in grade 5. How many students ride the bus?")
        self.check("12cm", "12", False, question)

    def test_rescore_dry_run_and_complete_preserved_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            rows = [
                {"id": "retry", "status": "error", "error": "old failure"},
                {"id": "retry", "status": "ok", "question": "Find distance.",
                 "prediction": "2√5", "reference": r"2\sqrt5",
                 "score": {"correct": False, "scorer": "old"}, "usage": {"completion_tokens": 10}},
                {"id": "unchanged", "status": "ok", "question": "Add.",
                 "prediction": "3", "reference": "3", "score": {"correct": True, "scorer": "old"}},
                {"id": "rejected", "status": "terminal_rejected", "score": {"correct": False}},
            ]
            (run / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            (run / "summary.json").write_text(json.dumps({"selected_examples": 3, "correct": 1,
                                                          "usage": {"completion_tokens": 500}}), encoding="utf-8")
            prior = {"scoring_revision": "unicode_pi_normalization", "changed_examples": ["earlier"]}
            (run / "rescore_audit.json").write_text(json.dumps(prior), encoding="utf-8")
            before = {path.name: path.read_bytes() for path in run.iterdir()}
            audit = rescore_math.rescore(run)
            self.assertEqual(before, {path.name: path.read_bytes() for path in run.iterdir()})
            self.assertEqual(audit["checked_examples"], 2)
            self.assertEqual(len(audit["examples"]), 3)
            self.assertEqual(audit["correct_after"], 2)
            self.assertEqual([row["id"] for row in audit["changed_examples"]], ["retry"])
            self.assertEqual(audit["prior_audits"], [prior])
            rescore_math.rescore(run, write=True, revision="custom_revision", reason="Uniform notation correction")
            saved = [json.loads(line) for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(saved[0], rows[0])
            self.assertEqual(saved[3], rows[3])
            for index in (1, 2):
                for key in ("prediction", "reference", "question"):
                    self.assertEqual(saved[index][key], rows[index][key])
                self.assertEqual(saved[index]["original_score"], rows[index]["score"])
                self.assertEqual(saved[index]["scoring_revision"], "custom_revision")
            self.assertEqual(saved[1]["usage"], rows[1]["usage"])
            summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["accuracy_full_denominator"], 2 / 3)
            self.assertEqual(summary["usage"], {"completion_tokens": 500})
            persisted = json.loads((run / "rescore_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["prior_audits"], [prior])
            self.assertEqual(persisted["reason"], "Uniform notation correction")
            self.assertEqual(rescore_math.rescore(run)["changed_examples"], [])


if __name__ == "__main__":
    unittest.main()
