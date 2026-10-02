"""Exact-match response cache: LRU + TTL + in-flight dedup (spec section 5.5).

The key is model-agnostic on purpose: a cached answer avoids ANY model call.
In-memory only, synchronous get/put on the hot path.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from typing import Awaitable, Callable

import orjson

from dietgate.core.config import CacheConfig
from dietgate.providers.base import ChatResponse


def cache_key(
    messages: list[dict],
    tools: list | None,
    response_format: dict | None,
    max_tokens: int | None,
) -> str:
    payload = {
        "messages": messages,
        "tools": tools,
        "response_format": response_format,
        "max_tokens": max_tokens,
    }
    normalized = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(normalized).hexdigest()


class ResponseCache:
    def __init__(self, cfg: CacheConfig) -> None:
        self.enabled = cfg.enabled
        self._ttl_s = cfg.ttl_s
        self._max_entries = max(1, cfg.max_entries)
        self._store: OrderedDict[str, tuple[float, ChatResponse]] = OrderedDict()
        self._inflight: dict[str, asyncio.Future] = {}

    def get(self, key: str) -> ChatResponse | None:
        if not self.enabled:
            return None
        entry = self._store.get(key)
        if entry is None:
            return None
        stored_mono, resp = entry
        if time.monotonic() - stored_mono > self._ttl_s:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return resp

    def put(self, key: str, resp: ChatResponse) -> None:
        if not self.enabled:
            return
        self._store[key] = (time.monotonic(), resp)
        self._store.move_to_end(key)
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)

    async def singleflight(
        self, key: str, coro_factory: Callable[[], Awaitable[ChatResponse]]
    ) -> ChatResponse:
        """Concurrent identical requests share one upstream call (leader computes)."""
        existing = self._inflight.get(key)
        if existing is not None:
            return await asyncio.shield(existing)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._inflight[key] = fut
        try:
            resp = await coro_factory()
        except BaseException as exc:
            if not fut.done():
                fut.set_exception(exc)
            raise
        else:
            if not fut.done():
                fut.set_result(resp)
            return resp
        finally:
            self._inflight.pop(key, None)

    # ------------------------------------------------------------- test info
    @property
    def size(self) -> int:
        return len(self._store)
