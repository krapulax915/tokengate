"""Shortcuts: local answers with zero cost, no model call."""
from __future__ import annotations

from httpx import AsyncClient

from dietgate.core.shortcuts import run_shortcuts
from tests.conftest import CHAT_URL, runtime_of


def test_ping_rule() -> None:
    result = run_shortcuts([{"role": "user", "content": "ping"}])
    assert result is not None and result.name == "ping" and result.content == "pong"


def test_reformat_json_rule() -> None:
    import json

    result = run_shortcuts([{"role": "user", "content": 'reformat {"b":1,"a":[1,2]}'}])
    assert result is not None and result.name == "reformat_json"
    assert json.loads(result.content) == {"b": 1, "a": [1, 2]}  # pretty-printed, same data
    assert "\n" in result.content  # indented output


def test_reformat_invalid_json_is_noop() -> None:
    assert run_shortcuts([{"role": "user", "content": "reformat {oops"}]) is None


def test_disabled_rules() -> None:
    assert run_shortcuts([{"role": "user", "content": "ping"}], enabled=False) is None


async def test_shortcut_end_to_end_zero_cost(client: AsyncClient) -> None:
    rt = runtime_of(client)
    resp = await client.post(
        CHAT_URL,
        json={
            "model": "mock-small",
            "messages": [{"role": "user", "content": 'reformat {"z": 1, "a": 2}'}],
            "temperature": 0.9,
        },
        headers={"X-Task-Id": "sc-1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert '"a": 2' in body["choices"][0]["message"]["content"]
    assert float(resp.headers["X-Dg-Cost-Usd"]) == 0.0
    assert resp.headers["X-Dg-Model"] == "shortcut"
    await rt.writer.drain()
    row = (await rt.db.get_requests_by_task_id("sc-1"))[0]
    assert row["shortcut"] == "reformat_json"
    assert row["cost_usd"] == 0.0
    assert row["baseline_cost_usd"] > 0.0  # savings vs. baseline still visible


async def test_shortcut_streaming(client: AsyncClient) -> None:
    raw = b""
    async with client.stream(
        "POST",
        CHAT_URL,
        json={
            "model": "mock-small",
            "stream": True,
            "messages": [{"role": "user", "content": "ping"}],
        },
    ) as resp:
        assert resp.status_code == 200
        async for chunk in resp.aiter_bytes():
            raw += chunk
    assert b"pong" in raw
    assert b": dg-cost=" in raw
    assert b'"model":"shortcut"' in raw or b'"model": "shortcut"' in raw
