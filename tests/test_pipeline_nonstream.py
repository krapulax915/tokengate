"""Non-stream pipeline: OpenAI shape, Dg headers, fallback, validation."""
from __future__ import annotations

import pytest
from httpx import AsyncClient

from dietgate.providers.base import ProviderError
from tests.conftest import CHAT_URL, chat_body


async def test_nonstream_openai_shape_and_headers(client: AsyncClient) -> None:
    resp = await client.post(CHAT_URL, json=chat_body())
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "mock-small"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["prompt_tokens"] > 0
    assert resp.headers["X-Dg-Model"] == "mock-small"
    assert resp.headers["X-Dg-Request-Id"]
    assert resp.headers["X-Dg-Task-Id"]
    assert resp.headers["X-Dg-Cache"] == "miss"
    assert resp.headers["X-Dg-Policy"] == "pinned"
    overhead = float(resp.headers["X-Dg-Overhead-Ms"])
    assert overhead >= 0.0  # spec: overhead is non-negative
    assert float(resp.headers["X-Dg-Cost-Usd"]) > 0.0


async def test_auto_routes_via_thompson(client: AsyncClient) -> None:
    resp = await client.post(CHAT_URL, json=chat_body(model="auto"))
    assert resp.status_code == 200
    assert resp.headers["X-Dg-Policy"] == "thompson"
    assert resp.headers["X-Dg-Model"] in {"mock-small", "mock-medium", "mock-large"}
    assert resp.json()["choices"]  # decision reason recorded server-side


async def test_policy_header_override(client: AsyncClient) -> None:
    resp = await client.post(
        CHAT_URL, json=chat_body(model="auto"), headers={"X-Policy": "static"}
    )
    assert resp.status_code == 200
    assert resp.headers["X-Dg-Policy"] == "static"
    assert resp.headers["X-Dg-Model"] == "mock-large"  # task baseline for extract prompts


async def test_unknown_model_is_400(client: AsyncClient) -> None:
    resp = await client.post(CHAT_URL, json=chat_body(model="nope"))
    assert resp.status_code == 400


async def test_missing_messages_is_400(client: AsyncClient) -> None:
    resp = await client.post(CHAT_URL, json={"model": "mock-small"})
    assert resp.status_code == 400


async def test_invalid_json_is_400(client: AsyncClient) -> None:
    resp = await client.post(CHAT_URL, content=b"{nope", headers={"content-type": "application/json"})
    assert resp.status_code == 400


async def test_fallback_on_upstream_failure(client: AsyncClient) -> None:
    """Pin a failing model; the gateway must succeed via the next candidate."""

    class FailingProvider:
        name = "failing"

        async def chat(self, req):
            raise ProviderError("boom")

        async def chat_stream(self, req):
            raise ProviderError("boom")
            yield  # pragma: no cover

        async def aclose(self) -> None:
            return None

    client._transport.app.state.rt.registry.register(FailingProvider(), ["mock-medium"])  # noqa: SLF001
    resp = await client.post(
        CHAT_URL,
        json=chat_body(model="mock-medium"),
        headers={"X-Task-Type": "extract"},
    )
    assert resp.status_code == 200
    assert resp.headers["X-Dg-Model"] == "mock-small"  # next candidate in tasks.yaml


async def test_all_attempts_failing_is_502(client: AsyncClient) -> None:
    class FailingProvider:
        name = "failing"

        async def chat(self, req):
            raise ProviderError("boom")

        async def chat_stream(self, req):
            raise ProviderError("boom")
            yield  # pragma: no cover

        async def aclose(self) -> None:
            return None

    rt = client._transport.app.state.rt  # noqa: SLF001
    rt.registry.register(FailingProvider(), ["mock-small", "mock-medium", "mock-large"])
    resp = await client.post(CHAT_URL, json=chat_body(model="mock-small"))
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "upstream_error"


async def test_event_queue_overflow_does_not_block(client: AsyncClient) -> None:
    """Writer queue (size 4) is never drained in tests -> it fills up; responses must still succeed."""
    for i in range(12):
        resp = await client.post(
            CHAT_URL, json=chat_body(), headers={"X-Task-Id": f"ovf-{i}"}
        )
        assert resp.status_code == 200
