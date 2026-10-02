"""Router: picks the policy (app.yaml default or X-Policy header) and returns a Decision."""
from __future__ import annotations

from dietgate.core.context import RequestContext
from dietgate.core.policies.base import Decision, Policy


class Router:
    def __init__(self, policies: dict[str, Policy], default_policy: str) -> None:
        if default_policy not in policies:
            raise ValueError(f"unknown default policy: {default_policy}")
        self._policies = policies
        self._default = default_policy

    @property
    def policy_names(self) -> list[str]:
        return list(self._policies.keys())

    @property
    def default_policy(self) -> str:
        return self._default

    def set_default(self, name: str) -> bool:
        """Demo control: switch the default policy at runtime."""
        name = name.strip().lower()
        if name not in self._policies:
            return False
        self._default = name
        return True

    def decide(self, ctx: RequestContext) -> Decision:
        requested = (ctx.header_policy or self._default).strip().lower()
        policy = self._policies.get(requested)
        if policy is None:
            policy = self._policies[self._default]
        decision = policy.decide(ctx)
        decision.policy = policy.name
        return decision
