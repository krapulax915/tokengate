"""Phase 0 acceptance: /healthz returns 200."""
from __future__ import annotations

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from dietgate.main import create_app


def app_client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_healthz_ok() -> None:
    app = create_app()
    async with app_client(app) as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_readyz_ok() -> None:
    app = create_app()
    async with app_client(app) as client:
        resp = await client.get("/readyz")
    assert resp.status_code == 200
