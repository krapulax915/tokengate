"""Monotonic clock helpers for hot-path timing (spec section 5)."""
from __future__ import annotations

from time import perf_counter_ns


def now_ns() -> int:
    return perf_counter_ns()


def ns_to_ms(delta_ns: int) -> float:
    return delta_ns / 1_000_000.0
