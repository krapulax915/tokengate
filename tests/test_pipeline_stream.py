"""Stream pipeline: SSE shape, dg-cost trailer, equivalence with non-stream."""
from __future__ import annotations

import orjson
from httpx import AsyncClient

from tests.conftest import CHAT_URL, chat_body


async def collect_sse(client: AsyncClient, body: dict) -> tuple[list[bytes], bytes]:
    raw = b""
    async with client.stream("POST", CHAT_URL, json=body) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers["X-Dg-Request-Id"]
        async for chunk in resp.aiter_bytes():
            raw += chunk
    events: list[bytes] = []
    for block in raw.split(b"\n\n"):
        if block.startswith(b"data: "):
            events.append(block[6:])
    return events, raw


def parse_stream(events: list[bytes]) -> tuple[str, bool, bool]:
    content_parts: list[str] = []
    done = False
    has_usage = False
    for ev in events:
        if ev == b"[DONE]":
            done = True
            continue
        payload = orjson.loads(ev)
        choices = payload.get("choices") or []
        if choices:
            delta = choices[0].get("delta", {})
            if delta.get("content"):
                content_parts.append(delta["content"])
            if choices[0].get("finish_reason"):
                has_usage = "usage" in payload or has_usage
    return "".join(content_parts), done, has_usage


async def test_stream_openai_format(client: AsyncClient) -> None:
    events, raw = await collect_sse(client, chat_body(stream=True))
    assert events[-1] == b"[DONE]"
    content, done, _ = parse_stream(events)
    assert done
    assert content  # non-empty streamed text
    first_payload = orjson.loads(events[0])
    assert first_payload["object"] == "chat.completion.chunk"
    assert first_payload["choices"][0]["delta"] == {"role": "assistant", "content": ""}


async def test_stream_has_dg_cost_trailer(client: AsyncClient) -> None:
    _, raw = await collect_sse(client, chat_body(stream=True))
    assert b": dg-cost=" in raw
    line = next(l for l in raw.split(b"\n") if l.startswith(b": dg-cost="))
    payload = orjson.loads(line[len(b": dg-cost="):])
    assert payload["model"] == "mock-small"
    assert payload["cost_usd"] > 0
    assert payload["overhead_ms"] >= 0


async def test_stream_and_nonstream_same_text(client: AsyncClient) -> None:
    """Same task_id + pinned model -> same correctness draw -> identical text."""
    headers = {"X-Task-Id": "shared-task-1"}
    body_ns = chat_body(model="mock-medium")
    body_ns["metadata"] = {"mock_expected": "SAME TEXT", "mock_wrong": ["OTHER"]}
    body_st = dict(body_ns, stream=True)

    resp = await client.post(CHAT_URL, json=body_ns, headers=headers)
    ns_content = resp.json()["choices"][0]["message"]["content"]
    events, _ = await collect_sse_with_headers(client, body_st, headers)
    st_content, _, _ = parse_stream(events)
    assert ns_content == st_content


async def collect_sse_with_headers(
    client: AsyncClient, body: dict, headers: dict
) -> tuple[list[bytes], bytes]:
    raw = b""
    async with client.stream("POST", CHAT_URL, json=body, headers=headers) as resp:
        assert resp.status_code == 200
        async for chunk in resp.aiter_bytes():
            raw += chunk
    events = [block[6:] for block in raw.split(b"\n\n") if block.startswith(b"data: ")]
    return events, raw


async def test_stream_fallback_on_upstream_failure(client: AsyncClient) -> None:
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
    events, raw = await collect_sse(client, chat_body(model="mock-medium", stream=True))
    content, done, _ = parse_stream(events)
    assert done
    assert content
    assert b'"model": "mock-small"' in raw or b'"model":"mock-small"' in raw
