"""Explicit train/test selection and answer scoring for small integration runs."""
from __future__ import annotations

import ast
import json
from decimal import Decimal, InvalidOperation
import random
import re
from pathlib import Path

from ..methods.prerun.maspo.textops import extract_answer


def read_rows(path: str | Path, benchmark: str, split: str) -> list[dict]:
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    raw = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()]
    rows = []
    for index, row in enumerate(raw):
        question = row.get("question", row.get("problem"))
        answer = row.get("answer", row.get("solution", row.get("answerKey")))
        if not isinstance(question, str) or answer is None:
            raise ValueError(f"{path}:{index + 1}: question and answer are required")
        if "choices" in row:
            choices = row["choices"]
            question += "\n" + "\n".join(
                f"{label}. {value}" for label, value in zip(choices["label"], choices["text"]))
        answer = str(answer)
        if "####" in answer:
            answer = answer.rsplit("####", 1)[1].strip()
        elif "\\boxed" in answer or "<answer>" in answer:
            answer = extract_answer(answer)
        rows.append({**row, "id": str(row.get("id", f"{benchmark}:{split}:{index}")),
                     "question": question, "answer": answer,
                     "benchmark": benchmark, "split": row.get("split", split)})
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"duplicate ids in {path}")
    return rows


def choose(rows: list[dict], n: int, seed: int) -> list[dict]:
    if n < 1 or n > len(rows):
        raise ValueError(f"requested {n} examples but source has {len(rows)}")
    indices = sorted(random.Random(seed).sample(range(len(rows)), n))
    return [rows[index] for index in indices]


def check_disjoint(train: list[dict], test: list[dict]) -> None:
    if {r["id"] for r in train} & {r["id"] for r in test}:
        raise ValueError("train and test share example ids")
    normalize = lambda text: " ".join(text.casefold().split())
    if {normalize(r["question"]) for r in train} & {normalize(r["question"]) for r in test}:
        raise ValueError("train and test share question text")
    if any(r["split"] != "train" for r in train):
        raise ValueError("MASPO optimization input must use the official train split")
    if any(r["split"] == "train" for r in test):
        raise ValueError("evaluation examples must not be from the train split")


def normalize_math(value: str) -> str:
    value = value.strip().replace("$", "")
    # A comma can separate coordinates or set elements; strip only an entire
    # unambiguous thousands-formatted scalar, never commas inside an expression.
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", value):
        value = value.replace(",", "")
    value = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", value)
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.replace("\\cdot", "*").replace("\\times", "*").replace("^", "**")
    value = value.replace("\\pi", "pi").replace("π", "pi").replace("\\!", "").replace("\\,", "")
    for _ in range(8):
        updated = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", value)
        updated = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", updated)
        if updated == value:
            break
        value = updated
    value = value.replace("{", "(").replace("}", ")")
    return re.sub(r"\s+", "", value).rstrip(".")


def score_answer(prediction: str, reference: str, benchmark: str, question: str = "") -> dict:
    if not prediction.strip():
        return {"correct": False, "scorer": "empty_prediction"}
    if benchmark in ("arc_easy", "openbookqa"):
        match = re.fullmatch(r"\s*\(?([A-E1-5])\)?[.\s]*", prediction)
        pred = match.group(1) if match else prediction.strip()
        return {"correct": pred == reference.strip(), "scorer": "option_exact"}
    pred, gold = normalize_math(prediction), normalize_math(reference)
    if pred == gold:
        return {"correct": True, "scorer": "normalized_exact"}
    if (benchmark == "math" and re.search(r"roots?|solutions?", question, re.I)
            and re.search(r"separated by commas", question, re.I) and "," in gold):
        predicted_parts, gold_parts = pred.split(","), gold.split(",")
        if len(predicted_parts) != len(gold_parts):
            return {"correct": False, "scorer": "unordered_roots"}
        unmatched = list(gold_parts)
        for part in predicted_parts:
            matched = next((i for i, target in enumerate(unmatched)
                            if score_answer(part, target, benchmark)["correct"]), None)
            if matched is None:
                return {"correct": False, "scorer": "unordered_roots"}
            unmatched.pop(matched)
        return {"correct": not unmatched, "scorer": "unordered_roots"}
    try:
        return {"correct": Decimal(pred) == Decimal(gold), "scorer": "decimal_exact"}
    except InvalidOperation:
        pass
    # Only a restricted arithmetic/symbolic AST reaches SymPy; no attributes or arbitrary calls.
    try:
        import sympy
        def explicit_products(expression):
            expression = re.sub(r"(?<=\d)(?=[a-zA-Z(])", "*", expression)
            return expression.replace(")(", ")*(")
        pred, gold = explicit_products(pred), explicit_products(gold)
        for expression in (pred, gold):
            if not re.fullmatch(r"[a-zA-Z0-9+*/(). -]+", expression):
                raise ValueError("unsupported notation")
            for node in ast.walk(ast.parse(expression, mode="eval")):
                if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                                         ast.Name, ast.Load, ast.Call, ast.Add, ast.Sub, ast.Mult,
                                         ast.Div, ast.Pow, ast.USub, ast.UAdd)):
                    raise ValueError("unsupported expression")
                if isinstance(node, ast.Name) and not (len(node.id) == 1 or node.id in ("sqrt", "pi")):
                    raise ValueError("unsupported name")
                if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and
                                                       node.func.id == "sqrt" and len(node.args) == 1):
                    raise ValueError("unsupported function")
        local = {letter: sympy.Symbol(letter) for letter in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"}
        local.update({"sqrt": sympy.sqrt, "pi": sympy.pi, "i": sympy.I})
        equal = sympy.simplify(sympy.sympify(pred, locals=local) - sympy.sympify(gold, locals=local)) == 0
        return {"correct": bool(equal), "scorer": "symbolic_equivalence"}
    except (ImportError, ValueError, SyntaxError, TypeError, AttributeError):
        return {"correct": False, "scorer": "unsupported_or_not_equal"}
