"""Admin API: shapes match docs/API.md, auth enforced, controls demo-gated."""
from __future__ import annotations

import dataclasses

import pytest
from httpx import ASGITransport, AsyncClient

from tests.conftest import (
    CHAT_URL,
    DEMO_ADMIN_KEY,
    DEMO_CONTROL_KEY,
    chat_body,
    make_app,
    runtime_of,
)


def admin_headers(client: AsyncClient) -> dict:
    return {"X-Admin-Key": DEMO_ADMIN_KEY}


async def feed(client: AsyncClient, n: int = 3) -> None:
    rt = runtime_of(client)
    for i in range(n):
        resp = await client.post(
            CHAT_URL,
            json=chat_body(model="auto", metadata={"mock_expected": f"ans-{i}"}),
            headers={"X-Task-Id": f"adm-{i}", "X-Task-Type": "extract"},
        )
        assert resp.status_code == 200
        await client.post("/v1/feedback", json={"task_id": f"adm-{i}", "success": True})
    await rt.learner.drain()
    await rt.writer.drain()


async def test_admin_requires_key(client: AsyncClient) -> None:
    resp = await client.get("/admin/summary")
    assert resp.status_code == 401
    resp = await client.get("/admin/summary", headers={"X-Admin-Key": "wrong"})
    assert resp.status_code == 401


async def test_summary_shape(client: AsyncClient) -> None:
    await feed(client)
    resp = await client.get("/admin/summary?window=1h", headers=admin_headers(client))
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "requests", "routed_requests", "pinned_requests", "spend_usd", "routed_spend_usd",
        "routed_baseline_spend_usd", "savings_usd", "savings_pct", "cache_hit_rate",
        "success_rate", "overhead_p50_ms", "overhead_p99_ms",
    ):
        assert key in body, key
    assert body["requests"] >= 3
    assert body["routed_requests"] == body["requests"]  # sim uses model:auto
    assert body["pinned_requests"] == 0
    assert body["success_rate"] == 1.0
    assert body["overhead_p50_ms"] is not None and body["overhead_p50_ms"] >= 0


async def test_savings_exclude_pinned_traffic(client: AsyncClient) -> None:
    """Pinned-model requests count as spend, not as routing savings."""
    rt = runtime_of(client)
    await feed(client, n=2)  # model:auto
    resp = await client.post(
        CHAT_URL,
        json=chat_body(model="mock-large"),  # pinned
        headers={"X-Task-Id": "pinned-1", "X-Task-Type": "extract"},
    )
    assert resp.status_code == 200
    await rt.writer.drain()
    body = (await client.get("/admin/summary?window=1h", headers=admin_headers(client))).json()
    assert body["pinned_requests"] == 1
    assert body["routed_requests"] == body["requests"] - 1
    # baseline on the summary card comes only from routed traffic
    assert body["routed_baseline_spend_usd"] > 0
    assert body["savings_pct"] >= 0  # exploration may pick the big model; never negative here


async def test_learning_curve_shape(client: AsyncClient) -> None:
    await feed(client)
    resp = await client.get("/admin/learning_curve?bucket=1m&window=1h", headers=admin_headers(client))
    body = resp.json()
    assert body["bucket_s"] == 60
    assert len(body["series"]) >= 1
    point = body["series"][0]
    for key in ("ts", "tasks", "successes", "cost_per_success_usd", "baseline_per_success_usd", "explore_share"):
        assert key in point, key


async def test_routing_matrix_shape(client: AsyncClient) -> None:
    await feed(client)
    resp = await client.get("/admin/routing_matrix?window=1h", headers=admin_headers(client))
    body = resp.json()
    assert body["total"] >= 3
    row = body["matrix"][0]
    for key in ("task_type", "model", "n", "share", "success_rate", "avg_cost_usd"):
        assert key in row, key


async def test_latency_shape(client: AsyncClient) -> None:
    await feed(client)
    resp = await client.get("/admin/latency?window=1h", headers=admin_headers(client))
    body = resp.json()
    assert body["overhead_ms"]["n"] >= 3  # non-stream requests still carry overhead
    assert body["ttft_ms"]["n"] == 0  # streams only; these were non-stream
    assert body["total_ms"]["n"] >= 3


async def test_requests_endpoints(client: AsyncClient) -> None:
    await feed(client)
    resp = await client.get("/admin/requests?limit=5", headers=admin_headers(client))
    rows = resp.json()["requests"]
    assert 1 <= len(rows) <= 5
    row = rows[0]
    detail = await client.get(f"/admin/requests/{row['id']}", headers=admin_headers(client))
    assert detail.status_code == 200
    assert detail.json()["id"] == row["id"]
    missing = await client.get("/admin/requests/dg-missing", headers=admin_headers(client))
    assert missing.status_code == 404


async def test_arms_after_feedback(client: AsyncClient) -> None:
    """Phase 5 acceptance: feedback changes /admin/arms."""
    rt = runtime_of(client)
    before_total = sum(a["n_labeled"] for a in rt.stats.snapshot())
    await feed(client, n=1)
    resp = await client.get("/admin/arms", headers=admin_headers(client))
    arms = resp.json()["arms"]
    after_total = sum(a["n_labeled"] for a in arms)
    assert after_total == before_total + 1  # the router's chosen arm got the label
    arm = next(a for a in arms if a["n_labeled"] > 0)
    assert "posterior_mean" in arm and "ci95" in arm


async def test_learning_cost_shape(client: AsyncClient) -> None:
    await feed(client)
    resp = await client.get("/admin/learning_cost?window=1h", headers=admin_headers(client))
    body = resp.json()
    for key in ("exploration_spend_usd", "shadow_evals", "shadow_eval_spend_usd"):
        assert key in body, key


async def test_demo_controls(client: AsyncClient) -> None:
    rt = runtime_of(client)
    resp = await client.post(
        "/admin/policy", json={"policy": "static"},
        headers={"X-Control-Key": DEMO_CONTROL_KEY},
    )
    assert resp.status_code == 200
    assert rt.router.default_policy == "static"
    await client.post(
        "/admin/policy", json={"policy": "thompson"},
        headers={"X-Control-Key": DEMO_CONTROL_KEY},
    )

    resp = await client.post(
        "/admin/policy", json={"policy": "nope"},
        headers={"X-Control-Key": DEMO_CONTROL_KEY},
    )
    assert resp.status_code == 400

    status = await client.get("/admin/sim/status", headers=admin_headers(client))
    assert status.status_code == 200


async def test_controls_reject_admin_key_and_missing_key(client: AsyncClient) -> None:
    """The control endpoints must NOT accept the read-only admin key."""
    for headers in ({}, {"X-Admin-Key": DEMO_ADMIN_KEY}, {"X-Control-Key": "wrong"}):
        resp = await client.post("/admin/policy", json={"policy": "static"}, headers=headers)
        assert resp.status_code in (401, 400), headers


async def test_controls_disabled_without_control_key(tmp_path) -> None:
    """Public deployments: empty control_key -> controls off even in DEMO_MODE."""
    app = make_app(tmp_path)
    rt = app.state.rt
    rt.app_cfg = dataclasses.replace(rt.app_cfg, control_key="")
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/admin/policy", json={"policy": "static"},
                headers={"X-Control-Key": DEMO_CONTROL_KEY},
            )
    assert resp.status_code == 400
    assert "no control_key" in resp.json()["error"]["message"]


async def test_controls_disabled_without_demo_mode(tmp_path) -> None:
    app = make_app(tmp_path)
    rt = app.state.rt
    rt.app_cfg = dataclasses.replace(rt.app_cfg, demo_mode=False)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/admin/policy", json={"policy": "static"},
                headers={"X-Control-Key": DEMO_CONTROL_KEY},
            )
    assert resp.status_code == 400
