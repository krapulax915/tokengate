"""Cheap token estimation (len/4 heuristic, spec section 7)."""
from __future__ import annotations


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def prompt_tokens_estimate(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):  # multi-part content blocks
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += estimate_tokens(part["text"])
        total += estimate_tokens(str(msg.get("name", "")))
    return total
