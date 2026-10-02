"""Policies: Thompson convergence, candidate bounds, eligibility fallback, cascade."""
from __future__ import annotations

import random

from dietgate.core.policies.base import PolicyDeps
from dietgate.core.policies.cascade import CascadePolicy
from dietgate.core.policies.static import StaticPolicy
from dietgate.core.policies.thompson import ThompsonPolicy
from dietgate.core.context import RequestContext
from dietgate.core.stats import ArmStatsStore
from tests.conftest import DEMO_API_KEY, make_app


def make_ctx(task_type: str = "extract", policy: str | None = None) -> RequestContext:
    return RequestContext(
        request_id="r", task_id="t", api_key_id="k", stream=False,
        requested_model="auto", header_policy=policy,
        messages=[{"role": "user", "content": "Extract the fields"}],
        task_type=task_type,
    )


def make_deps() -> PolicyDeps:
    rt = make_app().state.rt
    return PolicyDeps(catalog=rt.catalog, tasks=rt.tasks_cfg, stats=ArmStatsStore())


def test_thompson_converges_to_cheapest_eligible() -> None:
    """extract: target 0.90; widened skills small=0.96 medium=0.98 large=0.99."""
    deps = make_deps()
    policy = ThompsonPolicy(deps, eps=0.05, seed=7)
    rng = random.Random(99)
    skills = {"mock-small": 0.96, "mock-medium": 0.98, "mock-large": 0.99}
    picks: list[str] = []
    for round_no in range(600):
        ctx = make_ctx("extract")
        decision = policy.decide(ctx)
        assert decision.model_id in skills  # never outside the candidate list
        picks.append(decision.model_id)
        # ground-truth outcome for the chosen arm (simulated skill)
        success = rng.random() < skills[decision.model_id]
        deps.stats.record_outcome("extract", decision.model_id, success)
    recent = picks[-100:]
    assert recent.count("mock-small") >= 85, f"not converged: {recent.count('mock-small')}/100"


def test_thompson_bounded_samples_to_converge() -> None:
    deps = make_deps()
    policy = ThompsonPolicy(deps, eps=0.05, seed=11)
    rng = random.Random(5)
    skills = {"mock-small": 0.96, "mock-medium": 0.98, "mock-large": 0.99}
    for i in range(800):
        decision = policy.decide(make_ctx("classify"))  # target 0.90, same shape
        deps.stats.record_outcome(
            "classify", decision.model_id, rng.random() < skills[decision.model_id]
        )
    # posterior has learned: pure exploitation picks the cheapest eligible arm
    exploit = ThompsonPolicy(deps, eps=0.0, seed=99)
    late = [exploit.decide(make_ctx("classify")).model_id for _ in range(50)]
    assert late.count("mock-small") >= 45, f"late picks: {late}"
    st = deps.stats.get("classify", "mock-small")
    assert st.n_labeled > 100  # learned within a bounded number of samples


def test_thompson_none_eligible_falls_back_to_best_looking() -> None:
    deps = make_deps()
    # make every arm look terrible; summarize candidates are medium/large
    for _ in range(30):
        deps.stats.record_outcome("summarize", "mock-medium", False)
        deps.stats.record_outcome("summarize", "mock-large", False)
    policy = ThompsonPolicy(deps, eps=0.0, seed=3)
    decision = policy.decide(make_ctx("summarize"))
    assert decision.model_id in {"mock-medium", "mock-large"}
    assert "none eligible" in decision.reason


def test_thompson_explore_flag_with_eps() -> None:
    deps = make_deps()
    policy = ThompsonPolicy(deps, eps=1.0, seed=1)  # always explore
    decision = policy.decide(make_ctx("extract"))
    assert decision.explore is True
    policy0 = ThompsonPolicy(deps, eps=0.0, explore_boost=0.0, seed=1)
    assert policy0.decide(make_ctx("extract")).explore is False


def test_thompson_warmup_boost_ends() -> None:
    deps = make_deps()
    policy = ThompsonPolicy(deps, eps=0.0, explore_boost=1.0, warmup_rounds=10, seed=2)
    # during warm-up every decision explores
    for _ in range(10):
        assert policy.decide(make_ctx("extract")).explore is True
        deps.stats.record_outcome("extract", "mock-small", True)
    # warm-up budget exhausted (10 labeled outcomes) -> no more forced exploration
    assert policy.decide(make_ctx("extract")).explore is False


def test_static_policy_returns_baseline() -> None:
    deps = make_deps()
    decision = StaticPolicy(deps).decide(make_ctx("extract"))
    assert decision.model_id == "mock-large"
    assert decision.policy == "static"
    assert "mock-large" not in decision.fallbacks


def test_cascade_orders_candidates_cheapest_first() -> None:
    deps = make_deps()
    decision = CascadePolicy(deps).decide(make_ctx("extract"))
    assert decision.model_id == "mock-small"  # cheapest
    assert decision.fallbacks[0] == "mock-medium"
    assert decision.fallbacks[1] == "mock-large"


def test_router_header_override_unknown_policy_falls_back() -> None:
    rt = make_app().state.rt
    decision = rt.router.decide(make_ctx("extract", policy="does-not-exist"))
    assert decision.policy == "thompson"  # app.yaml default
    decision2 = rt.router.decide(make_ctx("extract", policy="static"))
    assert decision2.policy == "static"


def test_stats_posterior_and_idempotent_adjust() -> None:
    deps = make_deps()
    deps.stats.record_outcome("extract", "mock-small", True)
    deps.stats.record_outcome("extract", "mock-small", True)
    st = deps.stats.get("extract", "mock-small")
    assert st.posterior_mean() > 0.5
    st.adjust_outcome(old_success=True, new_success=True)  # no-op
    assert st.n_labeled == 2
    st.adjust_outcome(old_success=True, new_success=False)
    assert (st.successes, st.failures) == (1, 1)


async def test_cascade_escalates_on_verify_failure(tmp_path) -> None:
    """Cheapest model answers wrong -> cascade escalates until verified."""
    from httpx import ASGITransport, AsyncClient

    from dietgate.providers.mock import MockProvider
    from tests.conftest import CHAT_URL, chat_body

    app = make_app(tmp_path)
    rt = app.state.rt
    provider: MockProvider = rt.registry.resolve("mock-small")
    # find a task where mock-small is wrong but mock-medium is right
    task_id = next(
        f"cas-{i}"
        for i in range(2000)
        if not provider.is_correct("mock-small", f"cas-{i}", "extract")
        and provider.is_correct("mock-medium", f"cas-{i}", "extract")
    )
    headers = {"Authorization": f"Bearer {DEMO_API_KEY}", "X-Task-Id": task_id,
               "X-Task-Type": "extract", "X-Policy": "cascade"}
    body = chat_body(model="auto", metadata={"mock_expected": "TRUTH"})
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test", headers=headers
        ) as client:
            resp = await client.post(CHAT_URL, json=body)
            await rt.writer.drain()
            assert resp.status_code == 200
            assert resp.headers["X-Dg-Model"] == "mock-medium"  # escalated past mock-small
            row = (await rt.db.get_requests_by_task_id(task_id))[0]
    assert row["attempts"] == 2
    assert row["policy"] == "cascade"
    # header cost matches the persisted row and includes both attempts (>0)
    assert resp.headers["X-Dg-Cost-Usd"] == f"{row['cost_usd']:.8f}"
    assert row["cost_usd"] > 0
