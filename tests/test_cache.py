"""Cache: key stability, TTL, LRU eviction, singleflight dedup."""
from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from dietgate.core.cache import ResponseCache, cache_key
from dietgate.core.config import CacheConfig
from dietgate.providers.base import ChatResponse, Usage
from tests.conftest import CHAT_URL, DEMO_API_KEY, chat_body, make_app


def cfg(**kw) -> CacheConfig:
    base = {"enabled": True, "ttl_s": 300.0, "max_entries": 1000}
    base.update(kw)
    return CacheConfig(**base)


def resp(text: str = "hello") -> ChatResponse:
    return ChatResponse(content=text, finish_reason="stop", usage=Usage(1, 1), model="mock-small")


def test_key_stability_across_key_order() -> None:
    k1 = cache_key([{"role": "user", "content": "a", "name": "n"}], None, None, 5)
    k2 = cache_key([{"name": "n", "content": "a", "role": "user"}], None, None, 5)
    assert k1 == k2


def test_key_changes_with_messages() -> None:
    assert cache_key([{"role": "user", "content": "a"}], None, None, None) != cache_key(
        [{"role": "user", "content": "b"}], None, None, None
    )


def test_ttl_expiry() -> None:
    import time

    cache = ResponseCache(cfg(ttl_s=0.05))
    cache.put("k", resp())
    assert cache.get("k") is not None
    time.sleep(0.07)
    assert cache.get("k") is None


def test_lru_eviction() -> None:
    cache = ResponseCache(cfg(max_entries=2))
    cache.put("k1", resp("1"))
    cache.put("k2", resp("2"))
    cache.get("k1")  # refresh k1 -> k2 becomes LRU
    cache.put("k3", resp("3"))
    assert cache.get("k2") is None
    assert cache.get("k1") is not None and cache.get("k3") is not None


async def test_singleflight_10_concurrent_1_call() -> None:
    cache = ResponseCache(cfg())
    calls = 0

    async def upstream() -> ChatResponse:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return resp("shared")

    results = await asyncio.gather(*(cache.singleflight("k", upstream) for _ in range(10)))
    assert calls == 1
    assert all(r.content == "shared" for r in results)


async def test_singleflight_propagates_errors() -> None:
    cache = ResponseCache(cfg())

    async def boom() -> ChatResponse:
        raise RuntimeError("upstream down")

    results = await asyncio.gather(
        *(cache.singleflight("k", boom) for _ in range(5)), return_exceptions=True
    )
    assert all(isinstance(r, RuntimeError) for r in results)
    assert "k" not in cache._inflight  # cleaned up for the next caller


def test_disabled_cache_never_stores() -> None:
    cache = ResponseCache(cfg(enabled=False))
    cache.put("k", resp())
    assert cache.get("k") is None


async def test_second_identical_request_is_cache_hit_zero_cost(tmp_path) -> None:
    """Phase 3 acceptance: identical temperature=0 request -> hit, cost 0."""
    app = make_app(tmp_path)
    headers = {"Authorization": f"Bearer {DEMO_API_KEY}", "X-Task-Id": "cache-e2e"}
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test", headers=headers
        ) as client:
            body = chat_body(model="mock-small", temperature=0.0)
            r1 = await client.post(CHAT_URL, json=body)
            r2 = await client.post(CHAT_URL, json=body)
    assert r1.headers["X-Dg-Cache"] == "miss"
    assert r2.headers["X-Dg-Cache"] == "hit"
    assert float(r2.headers["X-Dg-Cost-Usd"]) == 0.0
    assert (
        r1.json()["choices"][0]["message"]["content"]
        == r2.json()["choices"][0]["message"]["content"]
    )


@pytest.mark.parametrize("key", ["a", "b", "c"])
def test_put_get_roundtrip(key: str) -> None:
    cache = ResponseCache(cfg())
    cache.put(key, resp(key))
    assert cache.get(key).content == key
