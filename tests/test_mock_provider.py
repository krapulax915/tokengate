"""Mock provider: determinism, skill calibration, token estimates."""
from __future__ import annotations

from dietgate.providers.base import ChatRequest
from dietgate.providers.mock import MockProvider, stable_draw_value
from tests.conftest import make_app


def make_provider() -> MockProvider:
    rt = make_app().state.rt
    return MockProvider(rt.catalog, seed=42)


def make_req(task_id: str, model: str = "mock-small", **meta) -> ChatRequest:
    return ChatRequest(
        model=model,
        messages=[{"role": "user", "content": "Extract the total from: INV-1, 1250.50 EUR"}],
        task_id=task_id,
        task_type="extract",
        metadata=dict(meta),
    )


async def test_same_seed_same_result() -> None:
    provider = make_provider()
    meta = {"mock_expected": '{"total": "1250.50"}', "mock_wrong": ['{"total": "0.00"}']}
    r1 = await provider.chat(make_req("task-a", **meta))
    r2 = await provider.chat(make_req("task-a", **meta))
    assert r1.content == r2.content
    assert r1.finish_reason == "stop"


async def test_correct_draw_returns_expected() -> None:
    provider = make_provider()
    meta = {"mock_expected": "RIGHT", "mock_wrong": ["WRONG"]}
    task_id = next(t for t in (f"t{i}" for i in range(500)) if provider.is_correct("mock-small", t, "extract"))
    resp = await provider.chat(make_req(task_id, **meta))
    assert resp.content == "RIGHT"


async def test_failed_draw_returns_wrong() -> None:
    provider = make_provider()
    meta = {"mock_expected": "RIGHT", "mock_wrong": ["WRONG"]}
    task_id = next(
        t for t in (f"t{i}" for i in range(500)) if not provider.is_correct("mock-small", t, "extract")
    )
    resp = await provider.chat(make_req(task_id, **meta))
    assert resp.content == "WRONG"


def test_skill_probability_calibrated() -> None:
    provider = make_provider()
    n = 10_000
    wins = sum(provider.is_correct("mock-small", f"task-{i}", "extract") for i in range(n))
    # spec: skill probability matches over 10_000 draws (within +/-2%)
    assert abs(wins / n - 0.96) <= 0.02


def test_stable_draw_matches_python_random_seed() -> None:
    v1 = stable_draw_value(42, "task-x", "mock-small")
    v2 = stable_draw_value(42, "task-x", "mock-small")
    assert v1 == v2
    assert 0.0 <= v1 < 1.0


async def test_usage_is_len_over_4_estimate() -> None:
    provider = make_provider()
    resp = await provider.chat(make_req("task-usage"))
    expected_content_len = len(resp.content)
    assert resp.usage.completion_tokens == max(1, expected_content_len // 4)
    assert resp.usage.prompt_tokens > 0


def test_unknown_task_type_uses_default_skill() -> None:
    provider = make_provider()
    assert provider._spec("mock-small").mock.skill_for("unknown") == 0.9
