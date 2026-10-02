"""Static policy: always the task's baseline model (comparison baseline)."""
from __future__ import annotations

from dietgate.core.context import RequestContext
from dietgate.core.policies.base import Decision, PolicyDeps


class StaticPolicy:
    name = "static"

    def __init__(self, deps: PolicyDeps) -> None:
        self._deps = deps

    def decide(self, ctx: RequestContext) -> Decision:
        cfg = self._deps.task_cfg(ctx.task_type)
        model = cfg.baseline if cfg.baseline in self._deps.catalog else "auto"
        fallbacks = self._deps.fallback_order(ctx.task_type, model)
        return Decision(
            model_id=model,
            policy=self.name,
            explore=False,
            reason=f"static: baseline {model}",
            fallbacks=fallbacks,
        )
