"""Read-only /admin/* endpoints for the dashboard + DEMO_MODE controls (spec 10)."""
from __future__ import annotations

import secrets
import time
from typing import Any, AsyncIterator

import orjson
from fastapi import APIRouter, Request, Response

from dietgate.core.errors import AuthError, BadRequestError, NotFoundError

router = APIRouter(prefix="/admin")


def _require_admin(request: Request) -> None:
    rt = request.app.state.rt
    provided = request.headers.get("x-admin-key") or ""
    if not provided:
        auth = request.headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            provided = auth[7:]
    if not provided or not secrets.compare_digest(provided, rt.app_cfg.admin_key):
        raise AuthError("admin key required")


def _require_control(request: Request) -> None:
    """POST /admin/* controls need DEMO_MODE **and** a configured control key.

    The control key is separate from the read-only admin key and is disabled
    (empty) by default, so a public demo cannot be reset or stopped by anyone
    who knows the dashboard key.
    """
    rt = request.app.state.rt
    if not rt.app_cfg.demo_mode:
        raise BadRequestError("controls are disabled (DEMO_MODE is off)")
    expected = rt.app_cfg.control_key
    if not expected:
        raise BadRequestError("controls are disabled (no control_key configured)")
    provided = request.headers.get("x-control-key") or ""
    if not provided or not secrets.compare_digest(provided, expected):
        raise AuthError("control key required")


def _json(data: Any, status: int = 200) -> Response:
    return Response(orjson.dumps(data), status_code=status, media_type="application/json")


def _parse_duration(raw: str | None, default_s: float) -> float:
    """Parse '30s' / '5m' / '1h' / '7d' / plain seconds."""
    raw = (raw or "").strip().lower()
    try:
        for suffix, mult in (("s", 1), ("m", 60), ("h", 3600), ("d", 86400)):
            if raw.endswith(suffix):
                return float(raw[:-1]) * mult
        return float(raw) if raw else default_s
    except ValueError:
        raise BadRequestError(f"invalid duration: {raw}")


def _window_seconds(request: Request, default: str = "24h") -> float:
    return _parse_duration(request.query_params.get("window", default), 86400.0)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
    return ordered[idx]


@router.get("/summary")
async def summary(request: Request) -> Response:
    _require_admin(request)
    rt = request.app.state.rt
    window_s = _window_seconds(request)
    ts = time.time() - window_s
    rows = await rt.db.get_request_metrics_since(ts)
    outcomes = await rt.db.get_outcome_stats_since(ts)
    overheads = [r["overhead_ms"] for r in rows if r["overhead_ms"] is not None]
    spend = sum(r["cost_usd"] or 0.0 for r in rows)
    cache_hits = sum(1 for r in rows if r["cache_hit"])
    # Savings are only meaningful for traffic the router actually decided
    # about: pinned-model requests already fixed the model, so they count
    # towards spend but not towards savings (see docs/METHODOLOGY.md).
    routed = [r for r in rows if (r["requested_model"] or "") == "auto"]
    routed_spend = sum(r["cost_usd"] or 0.0 for r in routed)
    routed_baseline = sum(r["baseline_cost_usd"] or 0.0 for r in routed)
    return _json(
        {
            "window_s": window_s,
            "requests": len(rows),
            "routed_requests": len(routed),
            "pinned_requests": len(rows) - len(routed),
            "spend_usd": round(spend, 8),
            "routed_spend_usd": round(routed_spend, 8),
            "routed_baseline_spend_usd": round(routed_baseline, 8),
            "savings_usd": round(routed_baseline - routed_spend, 8),
            "savings_pct": round((1 - routed_spend / routed_baseline) * 100, 2)
            if routed_baseline > 0
            else None,
            "cache_hit_rate": round(cache_hits / len(rows), 4) if rows else None,
            "success_rate": (outcomes["successes"] / outcomes["n"]) if outcomes and outcomes["n"] else None,
            "labeled_tasks": outcomes["n"] if outcomes else 0,
            "overhead_p50_ms": _percentile(overheads, 50),
            "overhead_p99_ms": _percentile(overheads, 99),
        }
    )


@router.get("/learning_curve")
async def learning_curve(request: Request) -> Response:
    _require_admin(request)
    rt = request.app.state.rt
    bucket_s = _parse_duration(request.query_params.get("bucket", "1m"), 60.0)
    if bucket_s <= 0:
        raise BadRequestError("bucket must be positive")
    ts = time.time() - _window_seconds(request, default="7d")
    rows = await rt.db.get_labeled_requests_since(ts)
    buckets: dict[int, dict] = {}
    for row in rows:
        key = int(row["ts"] // bucket_s)
        b = buckets.setdefault(key, {"cost": 0.0, "baseline": 0.0, "n": 0, "successes": 0, "explore": 0})
        b["cost"] += row["cost_usd"] or 0.0
        b["baseline"] += row["baseline"] or 0.0
        b["n"] += 1
        b["successes"] += 1 if row["success"] else 0
        b["explore"] += 1 if row["explore"] else 0
    series = []
    for key in sorted(buckets):
        b = buckets[key]
        successes = max(1, b["successes"])
        series.append(
            {
                "ts": key * bucket_s,
                "bucket_s": bucket_s,
                "tasks": b["n"],
                "successes": b["successes"],
                "cost_per_success_usd": round(b["cost"] / successes, 8),
                "baseline_per_success_usd": round(b["baseline"] / successes, 8),
                "explore_share": round(b["explore"] / b["n"], 3) if b["n"] else None,
            }
        )
    return _json({"bucket_s": bucket_s, "series": series})


@router.get("/routing_matrix")
async def routing_matrix(request: Request) -> Response:
    _require_admin(request)
    rt = request.app.state.rt
    ts = time.time() - _window_seconds(request)
    rows = await rt.db.get_routing_rows_since(ts)
    total = sum(r["n"] for r in rows) or 1
    matrix = [
        {
            "task_type": r["task_type"],
            "model": r["model"],
            "n": r["n"],
            "share": round(r["n"] / total, 4),
            "success_rate": round(r["successes"] / r["n"], 4) if r["n"] else None,
            "avg_cost_usd": round(r["cost"] / r["n"], 8) if r["n"] else None,
            "avg_baseline_usd": round(r["baseline"] / r["n"], 8) if r["n"] else None,
        }
        for r in rows
    ]
    return _json({"total": total, "matrix": matrix})


@router.get("/latency")
async def latency(request: Request) -> Response:
    _require_admin(request)
    rt = request.app.state.rt
    ts = time.time() - _window_seconds(request)
    stream_raw = request.query_params.get("stream")
    stream: int | None = None
    if stream_raw is not None and stream_raw != "":
        if stream_raw not in ("0", "1"):
            raise BadRequestError("stream must be 0 or 1")
        stream = int(stream_raw)
    rows = await rt.db.get_request_metrics_since(ts, stream=stream)

    def pct(name: str) -> dict:
        values = [r[name] for r in rows if r[name] is not None]
        return {
            "p50": _percentile(values, 50),
            "p90": _percentile(values, 90),
            "p99": _percentile(values, 99),
            "n": len(values),
        }

    return _json(
        {
            "stream": stream,
            "overhead_ms": pct("overhead_ms"),
            "ttft_ms": pct("ttft_ms"),
            "total_ms": pct("total_ms"),
        }
    )


@router.get("/requests")
async def recent_requests(request: Request) -> Response:
    _require_admin(request)
    rt = request.app.state.rt
    limit = min(500, max(1, int(request.query_params.get("limit", "50"))))
    return _json({"requests": await rt.db.get_recent_requests(limit)})


@router.get("/requests/{request_id}")
async def request_detail(request_id: str, request: Request) -> Response:
    _require_admin(request)
    rt = request.app.state.rt
    row = await rt.db.get_request_by_id(request_id)
    if row is None:
        raise NotFoundError(f"unknown request id: {request_id}")
    return _json(row)


@router.get("/arms")
async def arms(request: Request) -> Response:
    _require_admin(request)
    rt = request.app.state.rt
    return _json({"arms": rt.stats.snapshot()})


@router.get("/learning_cost")
async def learning_cost(request: Request) -> Response:
    _require_admin(request)
    rt = request.app.state.rt
    ts = time.time() - _window_seconds(request)
    explore = await rt.db.get_explore_spend_since(ts)
    shadow = await rt.db.get_shadow_spend_since(ts)
    return _json(
        {
            "exploration_spend_usd": round(explore, 8),
            "shadow_evals": shadow["n"] if shadow else 0,
            "shadow_eval_spend_usd": round(float(shadow["s"] or 0.0), 8) if shadow else 0.0,
        }
    )


# ------------------------------------------------------- DEMO_MODE controls --

@router.get("/sim/status")
async def sim_status(request: Request) -> Response:
    _require_admin(request)
    rt = request.app.state.rt
    if rt.sim_runner is None:
        return _json({"running": False, "demo_mode": False})
    return _json(rt.sim_runner.status())


@router.post("/sim/start")
async def sim_start(request: Request) -> Response:
    _require_control(request)
    rt = request.app.state.rt
    if rt.sim_runner is None:
        raise BadRequestError("simulator unavailable")
    body = orjson.loads(await request.body() or b"{}")
    out = rt.sim_runner.start(str(body.get("scenario", "fast")), body.get("policy"))
    return _json(out)


@router.post("/sim/stop")
async def sim_stop(request: Request) -> Response:
    _require_control(request)
    rt = request.app.state.rt
    if rt.sim_runner is None:
        raise BadRequestError("simulator unavailable")
    return _json(rt.sim_runner.stop())


@router.post("/policy")
async def set_policy(request: Request) -> Response:
    _require_control(request)
    rt = request.app.state.rt
    body = orjson.loads(await request.body() or b"{}")
    policy = str(body.get("policy", ""))
    if not rt.router.set_default(policy):
        raise BadRequestError(f"unknown policy: {policy}")
    return _json({"status": "ok", "policy": rt.router.default_policy})


@router.post("/reset")
async def reset(request: Request) -> Response:
    _require_control(request)
    rt = request.app.state.rt
    await rt.db.reset_all()
    rt.stats.reset()
    rt.learner.reset()
    return _json({"status": "ok"})
