"""Feedback -> learner -> arm stats; idempotency; restart rebuild; shadow eval."""
from __future__ import annotations

import asyncio
import time

from httpx import ASGITransport, AsyncClient

from tests.conftest import CHAT_URL, chat_body, make_app, runtime_of


async def serve_one(client: AsyncClient, task_id: str) -> None:
    resp = await client.post(
        CHAT_URL,
        json=chat_body(model="mock-small"),
        headers={"X-Task-Id": task_id, "X-Task-Type": "extract"},
    )
    assert resp.status_code == 200


async def test_unknown_task_feedback_404(client: AsyncClient) -> None:
    resp = await client.post("/v1/feedback", json={"task_id": "nope", "success": True})
    assert resp.status_code == 404


async def test_feedback_updates_arm_stats(client: AsyncClient) -> None:
    rt = runtime_of(client)
    await serve_one(client, "fb-1")
    assert rt.stats.get("extract", "mock-small").n_labeled == 0  # call counted, not labeled
    resp = await client.post("/v1/feedback", json={"task_id": "fb-1", "success": True})
    assert resp.status_code == 200
    await rt.learner.drain()
    stats = rt.stats.get("extract", "mock-small")
    assert stats.n_labeled == 1 and stats.successes == 1 and stats.failures == 0
    await rt.writer.drain()
    assert await rt.db.count_outcomes() == 1


async def test_feedback_is_idempotent(client: AsyncClient) -> None:
    rt = runtime_of(client)
    await serve_one(client, "fb-2")
    await client.post("/v1/feedback", json={"task_id": "fb-2", "success": True})
    await rt.learner.drain()
    await client.post("/v1/feedback", json={"task_id": "fb-2", "success": False})
    await rt.learner.drain()
    stats = rt.stats.get("extract", "mock-small")
    assert (stats.successes, stats.failures) == (0, 1)  # replaced, not double-counted
    await rt.writer.drain()
    assert await rt.db.count_outcomes() == 1  # one row per task (INSERT OR REPLACE)


async def test_feedback_validation(client: AsyncClient) -> None:
    resp = await client.post("/v1/feedback", json={"task_id": "x", "success": "yes"})
    assert resp.status_code == 400
    resp = await client.post("/v1/feedback", json={"success": True})
    assert resp.status_code == 400


async def test_stats_survive_restart(tmp_path) -> None:
    import dataclasses

    db_path = tmp_path / "restart.db"
    app1 = make_app(tmp_path, db_path=db_path)
    async with app1.router.lifespan_context(app1):
        async with AsyncClient(
            transport=ASGITransport(app=app1),
            base_url="http://test",
            headers={"Authorization": "Bearer dg-demo-key", "X-Task-Id": "rs-1",
                     "X-Task-Type": "extract"},
        ) as client:
            await client.post(CHAT_URL, json=chat_body(model="mock-small"))
            await client.post("/v1/feedback", json={"task_id": "rs-1", "success": True})
        await app1.state.rt.learner.drain()
        await app1.state.rt.writer.drain()

    app2 = make_app(tmp_path, db_path=db_path)
    async with app2.router.lifespan_context(app2):
        stats = app2.state.rt.stats.get("extract", "mock-small")
        assert stats is not None and stats.n_labeled == 1 and stats.successes == 1
        assert app2.state.rt.learner.has_task("rs-1")  # feedback still resolvable
    assert dataclasses  # keep import honest


async def test_shadow_eval_labels_unlabeled_task(client: AsyncClient) -> None:
    import dataclasses

    rt = runtime_of(client)
    rt.evaluator.cfg = dataclasses.replace(
        rt.evaluator.cfg, sample_rate=1.0, delay_s=0.05
    )
    await client.post(
        CHAT_URL,
        json=chat_body(model="mock-small", metadata={"mock_expected": "TRUTH"}),
        headers={"X-Task-Id": "shadow-1", "X-Task-Type": "extract"},
    )
    # the answer only matches ground truth when the mock draw succeeded; either
    # way a shadow_evals row must exist and the task gets labeled
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and rt.stats.get("extract", "mock-small").n_labeled == 0:
        await asyncio.sleep(0.02)
    stats = rt.stats.get("extract", "mock-small")
    assert stats is not None and stats.n_labeled == 1
    assert rt.evaluator.processed >= 1
