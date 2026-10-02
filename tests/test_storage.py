"""Phase 2 acceptance: rows land in SQLite, costs are correct, writer can die."""
from __future__ import annotations

import time

from httpx import AsyncClient

from tests.conftest import CHAT_URL, chat_body, runtime_of


async def test_100_requests_produce_100_rows(client: AsyncClient) -> None:
    rt = runtime_of(client)
    for i in range(100):
        resp = await client.post(CHAT_URL, json=chat_body(), headers={"X-Task-Id": f"t-{i}"})
        assert resp.status_code == 200
    await rt.writer.drain()
    assert await rt.db.count_requests() == 100


async def test_request_row_cost_math(client: AsyncClient) -> None:
    rt = runtime_of(client)
    resp = await client.post(
        CHAT_URL,
        json=chat_body(model="mock-small", metadata={"mock_expected": "hi"}),
        headers={"X-Task-Id": "cost-1", "X-Task-Type": "extract"},
    )
    await rt.writer.drain()
    rows = await rt.db.get_requests_by_task_id("cost-1")
    row = rows[0]
    pt, ct = row["prompt_tokens"], row["completion_tokens"]
    expected_cost = pt / 1e6 * 0.10 + ct / 1e6 * 0.30          # mock-small prices
    expected_baseline = pt / 1e6 * 10.0 + ct / 1e6 * 30.0      # mock-large baseline
    assert abs(row["cost_usd"] - expected_cost) < 1e-9
    assert abs(row["baseline_cost_usd"] - expected_baseline) < 1e-9
    assert row["task_type"] == "extract"
    assert row["chosen_model"] == "mock-small"
    assert row["status"] == 200
    assert row["stream"] == 0
    assert row["overhead_ms"] >= 0


async def test_stream_row_marked_as_stream(client: AsyncClient) -> None:
    rt = runtime_of(client)
    async with client.stream("POST", CHAT_URL, json=chat_body(model="mock-large", stream=True)) as resp:
        async for _ in resp.aiter_bytes():
            pass
    await rt.writer.drain()
    rows = await rt.db.get_stream_requests()
    assert rows, "expected at least one streamed request row"
    row = rows[-1]
    assert row["chosen_model"] == "mock-large"
    assert row["cost_usd"] > 0
    assert row["ttft_ms"] is not None


async def test_writer_down_does_not_break_responses(client: AsyncClient) -> None:
    import asyncio

    rt = runtime_of(client)
    await rt.writer.stop()  # kill the background writer
    rt.writer.queue = asyncio.Queue(maxsize=2)  # tiny queue so overflow happens fast
    dropped_before = rt.writer.dropped_events
    for i in range(12):
        resp = await client.post(CHAT_URL, json=chat_body(), headers={"X-Task-Id": f"dead-{i}"})
        assert resp.status_code == 200
    assert rt.writer.dropped_events > dropped_before  # events dropped, not blocking


async def test_shutdown_drains_pending_events(client: AsyncClient) -> None:
    rt = runtime_of(client)
    resp = await client.post(CHAT_URL, json=chat_body(), headers={"X-Task-Id": "drain-1"})
    assert resp.status_code == 200
    await rt.writer.stop()
    assert await rt.db.count_requests() >= 1
