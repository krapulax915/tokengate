"""Direct mock handler: same provider, zero gateway logic. Used by bench/overhead.py.

The zero-latency catalog is constructed in code (no config files, no env-driven
paths) so the bare handler has no filesystem surface at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fastapi import FastAPI, Request, Response  # noqa: E402

import orjson  # noqa: E402

from dietgate.core.config import MockSpec, ModelSpec  # noqa: E402
from dietgate.providers.base import ChatRequest  # noqa: E402
from dietgate.providers.mock import MockProvider  # noqa: E402

_ZERO_SPEC = ModelSpec(
    id="mock-small",
    provider="mock",
    price_in_per_mtok=0.10,
    price_out_per_mtok=0.30,
    mock=MockSpec(ttft_ms=0, tokens_per_second=0, jitter_pct=0, skill={"extract": 1.0}),
)

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
provider = MockProvider({"mock-small": _ZERO_SPEC}, seed=42)


@app.get("/healthz")
async def healthz() -> Response:
    return Response('{"status":"ok"}', media_type="application/json")


@app.post("/direct")
async def direct(request: Request) -> Response:
    body = orjson.loads(await request.body())
    resp = await provider.chat(
        ChatRequest(
            model=body["model"],
            messages=body["messages"],
            task_id=body.get("task_id", ""),
            task_type="extract",
        )
    )
    return Response(
        orjson.dumps({"content": resp.content}), media_type="application/json"
    )
