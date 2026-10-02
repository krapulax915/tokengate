"""Classifier: header wins over heuristic; unknown fallback."""
from __future__ import annotations

from dietgate.core.classifier import classify


def test_header_wins_over_heuristic() -> None:
    msgs = [{"role": "user", "content": "Summarize this text please"}]
    assert classify(msgs, header_value="code") == "code"


def test_header_is_normalized() -> None:
    assert classify([], header_value=" EXTRACT ") == "extract"


def test_extract_keyword() -> None:
    assert classify([{"role": "user", "content": "Extract the invoice fields"}]) == "extract"


def test_classify_keyword() -> None:
    assert classify([{"role": "user", "content": "Classify this ticket by category"}]) == "classify"


def test_summarize_keyword() -> None:
    assert classify([{"role": "user", "content": "Summarize the article below"}]) == "summarize"


def test_reason_keyword() -> None:
    assert classify([{"role": "user", "content": "Calculate how many boxes remain"}]) == "reason"


def test_code_keyword() -> None:
    assert classify([{"role": "user", "content": "Write a function that sorts dicts"}]) == "code"


def test_unknown_fallback() -> None:
    assert classify([{"role": "user", "content": "lorem ipsum dolor sit"}]) == "unknown"


def test_priority_first_match_wins() -> None:
    # 'extract' is checked before 'code' -> extract wins on overlap
    assert (
        classify([{"role": "user", "content": "Extract fields, then write a function"}])
        == "extract"
    )
