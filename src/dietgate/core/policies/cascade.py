"""Cascade policy: try cheapest candidate, verify, escalate on failure.

The Decision lists candidates cheapest-first; escalation itself happens in the
pipeline, which verifies each answer (ground truth from the simulator) and
falls through to the next model on mismatch. Verification needs the full
answer, so streamed cascade requests are only ordered, not verified.
"""
from __future__ import annotations

from dietgate.core.context import RequestContext
from dietgate.core.policies.base import Decision, PolicyDeps


class CascadePolicy:
    name = "cascade"

    def __init__(self, deps: PolicyDeps) -> None:
        self._deps = deps

    def decide(self, ctx: RequestContext) -> Decision:
        task_type = ctx.task_type
        by_cost = sorted(
            self._deps.available_candidates(task_type),
            key=lambda m: self._deps.cost_estimate(task_type, m),
        )
        if not by_cost:
            cfg = self._deps.task_cfg(task_type)
            return Decision(cfg.baseline, self.name, False, "cascade: no routable candidate", [])
        pick = by_cost[0]
        return Decision(
            model_id=pick,
            policy=self.name,
            explore=False,
            reason=f"cascade: cheapest first ({pick}), escalate on verify failure",
            fallbacks=by_cost[1:],
        )
