"""Policy interface and the Decision value object (spec sections 6.3-6.4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from dietgate.core.config import ModelSpec, TaskSpec
from dietgate.core.context import RequestContext
from dietgate.core.cost import request_cost_usd
from dietgate.core.stats import ArmStatsStore

PRIOR_COST_TOKENS_IN = 400   # cost prior before any observation (see DECISIONS D12)
PRIOR_COST_TOKENS_OUT = 300


@dataclass
class Decision:
    model_id: str
    policy: str
    explore: bool = False
    reason: str = ""
    fallbacks: list[str] = field(default_factory=list)


@dataclass
class PolicyDeps:
    catalog: dict[str, ModelSpec]
    tasks: dict[str, TaskSpec]
    stats: ArmStatsStore

    def task_cfg(self, task_type: str) -> TaskSpec:
        return self.tasks.get(task_type) or self.tasks["unknown"]

    def available_candidates(self, task_type: str) -> list[str]:
        cfg = self.task_cfg(task_type)
        return [m for m in cfg.candidates if m in self.catalog]

    def cost_estimate(self, task_type: str, model_id: str) -> float:
        """Observed cost EMA, or a price-based prior before data exists."""
        st = self.stats.get(task_type, model_id)
        if st is not None and st.cost_ema_usd is not None:
            return st.cost_ema_usd
        return request_cost_usd(
            self.catalog, model_id, PRIOR_COST_TOKENS_IN, PRIOR_COST_TOKENS_OUT
        )

    def fallback_order(self, task_type: str, chosen: str) -> list[str]:
        """Remaining candidates, cheapest first (fallbacks run only on failure)."""
        others = [m for m in self.available_candidates(task_type) if m != chosen]
        return sorted(others, key=lambda m: self.cost_estimate(task_type, m))


class Policy(Protocol):
    name: str

    def decide(self, ctx: RequestContext) -> Decision: ...
