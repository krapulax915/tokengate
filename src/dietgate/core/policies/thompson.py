"""Cost-aware Thompson sampling (spec section 6.3) - the main policy.

Exploit branch is the spec pseudocode: sample P(success) from each arm's Beta
posterior, keep arms whose sample meets the quality target, pick the cheapest
among them; if none is eligible pick the best-looking sample.

Three documented additions (DECISIONS.md D18, D33) keep the loop convergent in
practice:
- a bounded exploration boost during a task type's first `warmup_rounds`
  labeled outcomes, aimed at the least-tried arm, so cheap arms get enough
  data to prove themselves before the posterior judges them;
- forced exploration only targets arms that could still meet the quality target
  (upper 95% credible bound >= target), so arms the data already rules out stop
  costing quality on low-volume task types (D33);
- the demo skill margins in config/models.yaml are wide enough for the
  target to be statistically separable within the demo's traffic budget.
"""
from __future__ import annotations

import random

from dietgate.core.context import RequestContext
from dietgate.core.policies.base import Decision, PolicyDeps


class ThompsonPolicy:
    name = "thompson"

    def __init__(
        self,
        deps: PolicyDeps,
        eps: float = 0.05,
        explore_boost: float = 0.20,
        warmup_rounds: int = 300,
        seed: int = 1234,
        prune_explore: bool = True,
    ) -> None:
        self._deps = deps
        self._eps = eps
        self._boost = explore_boost
        self._warmup_rounds = warmup_rounds
        self._prune = prune_explore
        self._rng = random.Random(seed)

    def decide(self, ctx: RequestContext) -> Decision:
        task_type = ctx.task_type
        candidates = self._deps.available_candidates(task_type)
        cfg = self._deps.task_cfg(task_type)
        if not candidates:
            return Decision(cfg.baseline, self.name, False, "thompson: no routable candidate", [])

        labeled = {m: self._labeled(task_type, m) for m in candidates}
        in_warmup = sum(labeled.values()) < self._warmup_rounds
        eps_eff = self._eps + (self._boost if in_warmup else 0.0)

        # forced exploration: least-tried candidate, random tiebreak
        if self._rng.random() < eps_eff:
            pickable = self._plausible(task_type, candidates, cfg.min_success) if self._prune else candidates
            least = min(labeled[m] for m in pickable)
            pool = [m for m in pickable if labeled[m] == least]
            pick = self._rng.choice(pool)
            phase = "warmup" if in_warmup else "eps"
            return Decision(
                model_id=pick,
                policy=self.name,
                explore=True,
                reason=f"thompson: explore ({phase}) -> {pick}",
                fallbacks=self._deps.fallback_order(task_type, pick),
            )

        scored: list[tuple[str, float, float]] = []
        for model_id in candidates:
            st = self._deps.stats.get(task_type, model_id)
            successes = st.successes if st else 0
            failures = st.failures if st else 0
            p_sample = self._rng.betavariate(successes + 1.0, failures + 1.0)
            cost = self._deps.cost_estimate(task_type, model_id)
            scored.append((model_id, p_sample, cost))

        eligible = [s for s in scored if s[1] >= cfg.min_success]
        if eligible:
            pick, p, _ = min(eligible, key=lambda s: s[2])
            reason = f"thompson: p({pick})={p:.2f}>={cfg.min_success}, cheapest eligible"
        else:
            pick, p, _ = max(scored, key=lambda s: s[1])
            reason = f"thompson: none eligible (target {cfg.min_success}), best p({pick})={p:.2f}"
        return Decision(
            model_id=pick,
            policy=self.name,
            explore=False,
            reason=reason,
            fallbacks=self._deps.fallback_order(task_type, pick),
        )

    def _plausible(self, task_type: str, candidates: list[str], target: float) -> list[str]:
        """Arms that could still turn out to meet the quality target.

        Forced exploration used to sample every arm, including ones whose data already rule
        them out (upper 95% credible bound below the target); with low-volume task types that
        waste is a large share of the traffic and drags the success rate under the target.
        An arm with too little data keeps a wide interval, so it stays explorable.
        """
        keep = []
        for m in candidates:
            st = self._deps.stats.get(task_type, m)
            if st is None or st.ci95()[1] >= target:
                keep.append(m)
        return keep or list(candidates)

    def _labeled(self, task_type: str, model_id: str) -> int:
        st = self._deps.stats.get(task_type, model_id)
        return st.n_labeled if st else 0
