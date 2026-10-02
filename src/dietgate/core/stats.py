"""In-memory arm statistics per (task_type, model_id). Reads are hot-path-safe."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

A0 = B0 = 1.0  # Beta(successes + a0, failures + b0) prior
EMA_ALPHA = 0.2
_CI_Z = 1.96


@dataclass
class ArmStats:
    n_calls: int = 0
    n_labeled: int = 0
    successes: int = 0
    failures: int = 0
    cost_sum_usd: float = 0.0
    cost_ema_usd: float | None = None
    latency_ms_ema: float | None = None
    last_used_ts: float = 0.0

    def posterior_mean(self) -> float:
        a = self.successes + A0
        b = self.failures + B0
        return a / (a + b)

    def posterior_sd(self) -> float:
        a = self.successes + A0
        b = self.failures + B0
        var = (a * b) / ((a + b) ** 2 * (a + b + 1.0))
        return math.sqrt(var)

    def ci95(self) -> tuple[float, float]:
        mean, sd = self.posterior_mean(), self.posterior_sd()
        return (max(0.0, mean - _CI_Z * sd), min(1.0, mean + _CI_Z * sd))

    def record_call(self, cost_usd: float, latency_ms: float, ts: float) -> None:
        self.n_calls += 1
        self.cost_sum_usd += cost_usd
        self.cost_ema_usd = cost_usd if self.cost_ema_usd is None else (
            EMA_ALPHA * cost_usd + (1 - EMA_ALPHA) * self.cost_ema_usd
        )
        self.latency_ms_ema = latency_ms if self.latency_ms_ema is None else (
            EMA_ALPHA * latency_ms + (1 - EMA_ALPHA) * self.latency_ms_ema
        )
        self.last_used_ts = ts

    def record_outcome(self, success: bool) -> None:
        self.n_labeled += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1

    def adjust_outcome(self, old_success: bool, new_success: bool) -> None:
        """Idempotent label replacement (spec 6.6)."""
        if old_success == new_success:
            return
        if new_success:
            self.successes += 1
            self.failures = max(0, self.failures - 1)
        else:
            self.failures += 1
            self.successes = max(0, self.successes - 1)


@dataclass
class ArmStatsStore:
    arms: dict[tuple[str, str], ArmStats] = field(default_factory=dict)

    def get(self, task_type: str, model_id: str) -> ArmStats | None:
        return self.arms.get((task_type, model_id))

    def arm(self, task_type: str, model_id: str) -> ArmStats:
        key = (task_type, model_id)
        st = self.arms.get(key)
        if st is None:
            st = self.arms[key] = ArmStats()
        return st

    def record_call(
        self, task_type: str, model_id: str, cost_usd: float, latency_ms: float, ts: float
    ) -> None:
        self.arm(task_type, model_id).record_call(cost_usd, latency_ms, ts)

    def record_outcome(self, task_type: str, model_id: str, success: bool) -> None:
        self.arm(task_type, model_id).record_outcome(success)

    def adjust_outcome(
        self, task_type: str, model_id: str, old_success: bool, new_success: bool
    ) -> None:
        self.arm(task_type, model_id).adjust_outcome(old_success, new_success)

    def snapshot(self) -> list[dict]:
        out = []
        for (task_type, model_id), st in sorted(self.arms.items()):
            lo, hi = st.ci95()
            out.append(
                {
                    "task_type": task_type,
                    "model_id": model_id,
                    "n_calls": st.n_calls,
                    "n_labeled": st.n_labeled,
                    "successes": st.successes,
                    "failures": st.failures,
                    "success_rate": (st.successes / st.n_labeled) if st.n_labeled else None,
                    "posterior_mean": st.posterior_mean(),
                    "ci95": [lo, hi],
                    "cost_sum_usd": st.cost_sum_usd,
                    "cost_ema_usd": st.cost_ema_usd,
                    "latency_ms_ema": st.latency_ms_ema,
                    "last_used_ts": st.last_used_ts,
                }
            )
        return out

    def reset(self) -> None:
        self.arms.clear()
