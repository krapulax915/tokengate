"""Scenario definitions for the traffic simulator."""
from __future__ import annotations

from pathlib import Path

import yaml

_DEFAULT_MIX = {
    "extract": 0.60,
    "classify": 0.20,
    "summarize": 0.10,
    "reason": 0.06,
    "code": 0.04,
}

_SCENARIOS: dict[str, dict] = {
    "default": {
        "count": 2000,
        "rate": 25.0,
        "concurrency": 20,
        "mix": _DEFAULT_MIX,
    },
    "fast": {
        "count": 300,
        "rate": 50.0,
        "concurrency": 30,
        "mix": _DEFAULT_MIX,
    },
}

_PATH = Path(__file__).resolve().parent / "scenarios.yaml"


def load_scenarios() -> dict[str, dict]:
    if _PATH.exists():
        with _PATH.open("rb") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict) and data:
            return data
    return {k: dict(v) for k, v in _SCENARIOS.items()}
