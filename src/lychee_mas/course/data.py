"""Explicit train/test selection and answer scoring for small integration runs."""
from __future__ import annotations

import ast
import json
from decimal import Decimal, InvalidOperation
import random
import re
from pathlib import Path

from ..methods.prerun.maspo.textops import extract_answer

CHOICE_BENCHMARKS = frozenset({"arc_easy", "arc_challenge", "openbookqa"})


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


def _expand_math_macros(value: str) -> str:
    """Expand supported TeX tokens and Unicode radicals without changing their scope."""
    def argument(start: int, unicode_radical: bool = False) -> tuple[str, int]:
        while start < len(value) and value[start].isspace():
            start += 1
        if start >= len(value):
            raise ValueError("missing math argument")
        opener = value[start]
        if opener == "{" or (unicode_radical and opener == "("):
            closer = "}" if opener == "{" else ")"
            depth, end = 1, start + 1
            while end < len(value) and depth:
                depth += (value[end] == opener) - (value[end] == closer)
                end += 1
            if depth:
                raise ValueError("unclosed math argument")
            return _expand_math_macros(value[start + 1:end - 1]), end
        if value.startswith("\\", start) or value[start] == "√":
            return macro(start)
        if unicode_radical:
            # A printed radical takes one numeric atom or one symbol. A slash
            # outside explicit parentheses is not part of its radicand.
            match = re.match(r"(?:\d+(?:\.\d+)?|\.\d+|[A-Za-zπ])", value[start:])
            if not match:
                raise ValueError("unsupported radical argument")
            return match.group(), start + len(match.group())
        # TeX's unbraced macro argument is one token: \\frac 59 is 5/9.
        if opener in "}[]()" or not (opener.isalnum() or opener in "+-."):
            raise ValueError("unsupported TeX argument")
        return opener, start + 1

    def macro(start: int) -> tuple[str, int]:
        if value[start] == "√":
            radicand, end = argument(start + 1, unicode_radical=True)
            return f"sqrt({radicand})", end
        match = re.match(r"\\[A-Za-z]+", value[start:])
        if not match:
            raise ValueError("unsupported control symbol")
        name, end = match.group(), start + len(match.group())
        if name in ("\\frac", "\\dfrac", "\\tfrac"):
            numerator, end = argument(end)
            denominator, end = argument(end)
            return f"(({numerator})/({denominator}))", end
        if name == "\\sqrt":
            radicand, end = argument(end)
            return f"sqrt({radicand})", end
        return name, end

    parts, index = [], 0
    while index < len(value):
        if value[index] in ("\\", "√"):
            try:
                part, index = macro(index)
                parts.append(part)
                continue
            except ValueError:
                # Keep malformed/unsupported syntax visible to the strict scorer.
                pass
        parts.append(value[index])
        index += 1
    return "".join(parts)


def normalize_math(value: str) -> str:
    value = value.strip().replace("$", "")
    # A comma can separate coordinates or set elements; strip only an entire
    # unambiguous thousands-formatted scalar, never commas inside an expression.
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", value):
        value = value.replace(",", "")
    value = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", value)
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("⁄", "/").replace("∞", "\\infty").replace("∪", "\\cup")
    value = _expand_math_macros(value)
    value = value.replace("\\cdot", "*").replace("\\times", "*").replace("^", "**")
    value = value.replace("\\pi", "pi").replace("π", "pi").replace("\\!", "").replace("\\,", "")
    value = value.replace("\\pm", "±")
    value = re.sub(r"\*\*(?:\{\\circ\}|\\circ)", "°", value)
    value = value.replace("{", "(").replace("}", ")")
    return re.sub(r"\s+", "", value).rstrip(".")


def _radix_score(prediction: str, reference: str, question: str) -> dict | None:
    words = {word: index for index, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen".split())}
    token = r"(\d{1,2}|" + "|".join(words) + r")\b"
    text = question.replace("$", "")
    request = re.search(
        r"(?:\b(?:express|write|give|state|represent)\b[^.!?]*\b(?:in|using)"
        r"|\b(?:what|which)\s+number\s+in|\bconvert\b[^.!?]*\bto)\s+base\s+" + token,
        text, re.I)
    requested = None
    if request:
        label = request.group(1).lower()
        requested = int(label) if label.isdigit() else words[label]
    def numeral(value):
        match = re.fullmatch(r"([+-]?[0-9A-Za-z]+)(?:_(?:\{(\d+)\}|(\d+)))?",
                             value.strip().replace("$", ""))
        if not match:
            return None
        base = match.group(2) or match.group(3)
        return match.group(1), int(base) if base else None
    pred, gold = numeral(prediction), numeral(reference)
    if requested is None and not any(
            item and item[1] is not None and 2 <= item[1] <= 36
            and re.fullmatch(r"[+-]?\d[0-9A-Za-z]*", item[0]) for item in (pred, gold)):
        # A symbolic subscript such as a_1 is not evidence of a radix numeral.
        return None
    if requested is None and not any(item and item[1] is not None for item in (pred, gold)):
        return None
    base = requested if requested is not None else (gold[1] if gold else None)
    if not pred or not gold or base is None or not 2 <= base <= 36:
        return {"correct": False, "scorer": "radix_notation"}
    if requested is None and (pred[1] is None or gold[1] is None):
        return {"correct": False, "scorer": "radix_notation", "base": base}
    if any(item[1] is not None and item[1] != base for item in (pred, gold)):
        return {"correct": False, "scorer": "radix_notation", "base": base}
    try:
        correct = int(pred[0], base) == int(gold[0], base)
    except ValueError:
        correct = False
    return {"correct": correct, "scorer": "radix_notation", "base": base}


def score_answer(prediction: str, reference: str, benchmark: str, question: str = "") -> dict:
    if not prediction.strip():
        return {"correct": False, "scorer": "empty_prediction"}
    if benchmark in CHOICE_BENCHMARKS:
        match = re.fullmatch(r"\s*\(?([A-E1-5])\)?[.\s]*", prediction)
        pred = match.group(1) if match else prediction.strip()
        return {"correct": pred == reference.strip(), "scorer": "option_exact"}
    if benchmark in ("math", "math500"):
        radix = _radix_score(prediction, reference, question)
        if radix is not None:
            return radix
    pred, gold = normalize_math(prediction), normalize_math(reference)
    if (benchmark in ("math", "math500")
            and re.search(r"\b(?:which|what)\s+grade\b", question, re.I)):
        def grade(value):
            match = re.fullmatch(r"([1-9]\d*)(st|nd|rd|th)", value)
            if not match:
                return value
            number = int(match.group(1))
            suffix = "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
            return match.group(1) if match.group(2) == suffix else value
        pred, gold = grade(pred), grade(gold)
    if (benchmark in ("math", "math500") and gold.endswith("°")
            and re.search(r"\bangles?\b|\bdegrees?\b|\\angle", question, re.I)):
        gold = gold[:-1]
        pred = pred.removesuffix("°")
    if pred == gold:
        return {"correct": True, "scorer": "normalized_exact"}
    if (benchmark in ("math", "math500") and re.search(r"roots?|solutions?", question, re.I)
            and re.search(r"separated by commas", question, re.I) and ("," in gold or "±" in gold)):
        def roots(expression):
            # Only an explicit root-list request allows this unordered expansion.
            if expression.count("±") == 1 and "," not in expression:
                return [expression.replace("±", "+"), expression.replace("±", "-")]
            return expression.split(",")
        predicted_parts, gold_parts = roots(pred), roots(gold)
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
    decimal = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    if re.fullmatch(decimal, pred) and re.fullmatch(decimal, gold):
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
