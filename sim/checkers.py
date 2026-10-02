"""Deterministic answer checkers (spec 6.5): exact | regex | json_schema | numeric | contains."""
from __future__ import annotations

import json
import re
from typing import Any, Callable

import orjson


def check_exact(expected: str, answer: str) -> bool:
    return answer.strip().lower() == str(expected).strip().lower()


def check_regex(pattern: str, answer: str) -> bool:
    try:
        return re.search(pattern, answer, re.DOTALL | re.IGNORECASE) is not None
    except re.error:
        return False


_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def check_numeric(expected: str, answer: str) -> bool:
    """True if the first number in the answer matches the expected value."""
    match = _NUM_RE.search(answer.replace(" ", ""))
    if match is None:
        return False
    try:
        value = float(match.group(0).replace(",", "."))
        return abs(value - float(expected)) <= 1e-6 * max(1.0, abs(float(expected)))
    except ValueError:
        return False


def check_contains(keywords: list[str], answer: str) -> bool:
    low = answer.lower()
    return all(kw.lower() in low for kw in keywords)


def check_json_schema(expected: Any, answer: str) -> bool:
    """The answer must be valid JSON equal (order/whitespace-insensitive) to expected."""
    try:
        parsed = orjson.loads(answer)
    except (orjson.JSONDecodeError, ValueError):
        try:
            parsed = json.loads(answer)
        except (json.JSONDecodeError, ValueError):
            return False
    return parsed == expected


def make_checker(name: str, expected: Any) -> Callable[[str], bool]:
    """Bind a checker name to its expected value; returns f(answer) -> bool."""
    if name == "exact":
        return lambda answer: check_exact(str(expected), answer)
    if name == "regex":
        return lambda answer: check_regex(str(expected), answer)
    if name == "numeric":
        return lambda answer: check_numeric(str(expected), answer)
    if name == "contains":
        return lambda answer: check_contains(list(expected), answer)
    if name == "json_schema":
        return lambda answer: check_json_schema(expected, answer)
    raise ValueError(f"unknown checker: {name}")
