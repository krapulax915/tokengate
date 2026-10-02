"""Simulator end-to-end against the in-process ASGI app (small scenario)."""
from __future__ import annotations

import random

import httpx
from httpx import ASGITransport, AsyncClient

from sim.checkers import make_checker
from sim.scenarios import load_scenarios
from sim.tasks import generate_task
from sim.traffic import run_scenario
from tests.conftest import DEMO_API_KEY, make_app


def test_checkers_behave() -> None:
    assert make_checker("exact", "billing")("Billing ")
    assert not make_checker("exact", "billing")("shipping")
    assert make_checker("numeric", "57")("There are 57 good boxes.")
    assert make_checker("numeric", "57")("57")
    assert not make_checker("numeric", "57")("56 boxes")
    assert make_checker("contains", ["deploy", "billing"])("we deploy and handle billing")
    assert not make_checker("contains", ["deploy", "billing"])("only deploy")
    assert make_checker("regex", r"def add\(a, b\):[\s\S]*return a \+ b")(
        "def add(a, b):\n    return a + b"
    )
    assert make_checker("json_schema", {"a": 1})('{"a": 1}')
    assert not make_checker("json_schema", {"a": 1})('{"a": 2}')


def test_generators_have_valid_ground_truth() -> None:
    rng = random.Random(1)
    for task_type in ("extract", "classify", "summarize", "reason", "code"):
        for i in range(30):
            task = generate_task(task_type, rng, f"{task_type}-{i}")
            assert task.messages, task_type
            correct = task.metadata["mock_expected"]
            wrong = task.metadata["mock_wrong"][0]
            assert task.checker()(correct) is True, (task_type, correct)
            assert task.checker()(wrong) is False, (task_type, wrong)


def test_scenarios_load() -> None:
    scenarios = load_scenarios()
    assert "default" in scenarios
    assert scenarios["default"]["count"] == 2000
    assert abs(sum(scenarios["default"]["mix"].values()) - 1.0) < 1e-9


async def test_traffic_runs_against_gateway(tmp_path) -> None:
    app = make_app(tmp_path)
    rt = app.state.rt
    async with app.router.lifespan_context(app):
        scenario = {
            "count": 12,
            "rate": 50.0,
            "concurrency": 6,
            "mix": {"extract": 0.5, "classify": 0.25, "reason": 0.25},
        }
        sim_client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        try:
            result = await run_scenario(
                scenario,
                name="test",
                base_url="http://test",
                api_key=DEMO_API_KEY,
                policy="thompson",
                seed=3,
                quiet=True,
                client=sim_client,
            )
        finally:
            await sim_client.aclose()
        await rt.learner.drain()
        await rt.writer.drain()
        assert result.total == 12
        assert result.successes > 6  # mock skill is well above 0.5 for these types
        assert result.cost_sum > 0
        assert result.baseline_sum > result.cost_sum  # savings vs baseline exist
        assert await rt.db.count_outcomes() == 12  # one checker label per task
