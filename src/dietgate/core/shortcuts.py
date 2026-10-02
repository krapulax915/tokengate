"""Local rules that answer without any model call (spec section 5.6).

A rule receives the message list and returns a ShortcutResult or None; the
first rule that fires wins. Register new rules in DEFAULT_RULES.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ShortcutResult:
    name: str
    content: str


Rule = Callable[[list[dict]], ShortcutResult | None]


def _last_user_text(messages: list[dict]) -> str | None:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
    return None


def rule_ping(messages: list[dict]) -> ShortcutResult | None:
    if (_last_user_text(messages) or "").strip().lower() == "ping":
        return ShortcutResult("ping", "pong")
    return None


def rule_reformat_json(messages: list[dict]) -> ShortcutResult | None:
    text = _last_user_text(messages) or ""
    prefix = "reformat "
    if not text.lower().startswith(prefix):
        return None
    try:
        parsed = json.loads(text[len(prefix):])
    except (json.JSONDecodeError, ValueError):
        return None
    return ShortcutResult("reformat_json", json.dumps(parsed, indent=2, ensure_ascii=False))


DEFAULT_RULES: list[Rule] = [rule_ping, rule_reformat_json]


def run_shortcuts(messages: list[dict], enabled: bool = True) -> ShortcutResult | None:
    if not enabled:
        return None
    for rule in DEFAULT_RULES:
        result = rule(messages)
        if result is not None:
            return result
    return None
