"""Real-provider adapters against faked upstreams (httpx.MockTransport).

No network, no API keys: these tests pin down the wire format we send and the
usage accounting we derive from each upstream's responses.
"""
from __future__ import annotations

import json

import httpx
import pytest

from dietgate.providers.base import ChatRequest, ProviderError
from dietgate.providers.registry import AnthropicProvider, OpenAICompatProvider

BASE = "https://upstream.test/v1"


def _req(**kw) -> ChatRequest:
    base = dict(
        model="some-model",
        messages=[
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Say hi"},
        ],
        task_id="t1",
        metadata={"mock_expected": "SECRET-DEMO-FIELD"},
    )
    base.update(kw)
    return ChatRequest(**base)


def _sse(*events: str) -> bytes:
    return ("".join(f"data: {e}\n\n" for e in events)).encode()


def _stream_response(body: bytes) -> httpx.Response:
    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


# --------------------------------------------------------------- OpenAI-compat
async def test_openai_non_stream_payload_and_usage() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "some-model-2025",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        })

    p = OpenAICompatProvider(BASE, "k-test", 5.0, transport=httpx.MockTransport(handler))
    resp = await p.chat(_req(temperature=0.0, max_tokens=50))
    await p.aclose()
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer k-test"
    assert seen["body"]["max_tokens"] == 50 and seen["body"]["temperature"] == 0.0
    assert "metadata" not in seen["body"] and "SECRET-DEMO-FIELD" not in json.dumps(seen["body"])
    assert (resp.content, resp.usage.prompt_tokens, resp.usage.completion_tokens) == ("hi", 11, 3)


async def test_openai_max_tokens_param_override(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MAX_TOKENS_PARAM", "max_completion_tokens")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

    p = OpenAICompatProvider(BASE, "k", 5.0, transport=httpx.MockTransport(handler))
    await p.chat(_req(max_tokens=20))
    assert seen["body"]["max_completion_tokens"] == 20 and "max_tokens" not in seen["body"]


async def test_openai_stream_collects_text_and_final_usage() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return _stream_response(_sse(
            json.dumps({"model": "m", "choices": [{"delta": {"content": "Hel"}}]}),
            json.dumps({"model": "m", "choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]}),
            json.dumps({"model": "m", "choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}}),
            "[DONE]",
        ))

    p = OpenAICompatProvider(BASE, "k", 5.0, transport=httpx.MockTransport(handler))
    chunks = [c async for c in p.chat_stream(_req(stream=True))]
    assert seen["body"]["stream"] is True
    assert seen["body"]["stream_options"] == {"include_usage": True}
    assert "".join(c.delta for c in chunks) == "Hello"
    usage = [c.usage for c in chunks if c.usage][-1]
    assert (usage.prompt_tokens, usage.completion_tokens) == (9, 2)


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_openai_retryable_status_becomes_provider_error(status: int) -> None:
    p = OpenAICompatProvider(
        BASE, "k", 5.0, transport=httpx.MockTransport(lambda r: httpx.Response(status, json={}))
    )
    with pytest.raises(ProviderError):
        await p.chat(_req())


# ------------------------------------------------------------------- Anthropic
async def test_anthropic_non_stream_wire_format() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "some-model",
            "content": [{"type": "text", "text": "hi "}, {"type": "text", "text": "there"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 14, "output_tokens": 4},
        })

    p = AnthropicProvider("k-ant", 5.0, transport=httpx.MockTransport(handler), base_url=BASE)
    resp = await p.chat(_req(temperature=1.7, max_tokens=None))
    body = seen["body"]
    assert seen["path"] == "/v1/messages"
    assert seen["headers"]["x-api-key"] == "k-ant" and "authorization" not in seen["headers"]
    assert seen["headers"]["anthropic-version"]
    assert body["system"] == "Be terse." and body["messages"] == [{"role": "user", "content": "Say hi"}]
    assert body["stream"] is False and body["max_tokens"] == 1024
    assert body["temperature"] == 1.0                      # clamped to Anthropic's range
    assert "SECRET-DEMO-FIELD" not in json.dumps(body)
    assert resp.content == "hi there"
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens) == (14, 4)


async def test_anthropic_stream_sends_stream_flag_and_counts_input_tokens() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return _stream_response(_sse(
            json.dumps({"type": "message_start", "message": {"usage": {"input_tokens": 12, "output_tokens": 1}}}),
            json.dumps({"type": "content_block_start", "index": 0}),
            json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hel"}}),
            json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "lo"}}),
            json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}}),
            json.dumps({"type": "message_stop"}),
        ))

    p = AnthropicProvider("k", 5.0, transport=httpx.MockTransport(handler), base_url=BASE)
    chunks = [c async for c in p.chat_stream(_req(stream=True))]
    assert seen["body"]["stream"] is True            # was missing before: the request was not streamed
    assert "".join(c.delta for c in chunks) == "Hello"
    last = [c for c in chunks if c.usage][-1]
    assert (last.usage.prompt_tokens, last.usage.completion_tokens) == (12, 7)  # input tokens kept
    assert last.finish_reason == "end_turn"


async def test_anthropic_stream_error_event_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _stream_response(_sse(json.dumps({"type": "error", "error": {"type": "overloaded_error"}})))

    p = AnthropicProvider("k", 5.0, transport=httpx.MockTransport(handler), base_url=BASE)
    with pytest.raises(ProviderError):
        [c async for c in p.chat_stream(_req(stream=True))]


@pytest.mark.parametrize("status", [429, 500, 529])
async def test_anthropic_retryable_status(status: int) -> None:
    p = AnthropicProvider(
        "k", 5.0, transport=httpx.MockTransport(lambda r: httpx.Response(status, json={})), base_url=BASE
    )
    with pytest.raises(ProviderError):
        await p.chat(_req())


# ------------------------------------------- per-model upstream settings (e.g. Novita)
from dietgate.core.config import ModelSpec  # noqa: E402


def _spec(**kw) -> ModelSpec:
    base = dict(
        id="novita-glm", provider="openai_compat", price_in_per_mtok=0.15, price_out_per_mtok=0.5,
        upstream_model="zai-org/glm-5.3-flash", extra_body={"reasoning_effort": "low", "model": "HIJACK"},
    )
    base.update(kw)
    return ModelSpec(**base)


async def test_upstream_model_id_and_extra_body_are_applied() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "zai-org/glm-5.3-flash",
            "choices": [{"message": {"content": "pong", "reasoning_content": "thinking..."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 120},
        })

    p = OpenAICompatProvider(
        "https://api.novita.ai/openai", "k", 5.0,
        transport=httpx.MockTransport(handler), specs={"novita-glm": _spec()},
    )
    resp = await p.chat(_req(model="novita-glm"))
    assert seen["body"]["model"] == "zai-org/glm-5.3-flash"      # gateway id mapped to provider id
    assert seen["body"]["reasoning_effort"] == "low"             # extra_body merged
    assert resp.content == "pong"                                # hidden reasoning is not returned as answer
    assert resp.usage.completion_tokens == 120                   # provider-reported (incl. thinking) tokens


async def test_reasoning_only_stream_chunks_are_skipped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _stream_response(_sse(
            json.dumps({"choices": [{"delta": {"role": "assistant", "content": None, "reasoning_content": "hmm"}}]}),
            json.dumps({"choices": [{"delta": {"content": None, "reasoning_content": "more"}}]}),
            json.dumps({"choices": [{"delta": {"content": "pong"}, "finish_reason": "stop"}]}),
            json.dumps({"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 90}}),
            "[DONE]",
        ))

    p = OpenAICompatProvider(
        "https://api.novita.ai/openai", "k", 5.0,
        transport=httpx.MockTransport(handler), specs={"novita-glm": _spec(extra_body={})},
    )
    chunks = [c async for c in p.chat_stream(_req(model="novita-glm", stream=True))]
    assert "".join(c.delta for c in chunks) == "pong"
    assert [c.usage for c in chunks if c.usage][-1].completion_tokens == 90


async def test_stream_usage_option_can_be_disabled_per_model() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return _stream_response(_sse(json.dumps({"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}), "[DONE]"))

    p = OpenAICompatProvider(
        BASE, "k", 5.0, transport=httpx.MockTransport(handler),
        specs={"m": _spec(id="m", stream_include_usage=False, extra_body={})},
    )
    [c async for c in p.chat_stream(_req(model="m", stream=True))]
    assert "stream_options" not in seen["body"]


def test_registry_keeps_providers_with_different_base_urls_apart(monkeypatch, tmp_path) -> None:
    from dietgate.core.config import load_app
    from dietgate.providers.registry import build_registry
    from tests.conftest import TEST_DIR, make_settings

    monkeypatch.setenv("OPENAI_API_KEY", "k-openai")
    monkeypatch.setenv("NOVITA_API_KEY", "k-novita")
    catalog = {
        "gpt-x": ModelSpec(id="gpt-x", provider="openai_compat", price_in_per_mtok=1, price_out_per_mtok=2),
        "novita-glm": _spec(base_url="https://api.novita.ai/openai", api_key_env="NOVITA_API_KEY", extra_body={}),
    }
    reg = build_registry(catalog, make_settings(tmp_path), load_app(TEST_DIR / "config", {}))
    a, b = reg.resolve("gpt-x"), reg.resolve("novita-glm")
    assert a is not b and a.name != b.name
    assert str(b._client.base_url).startswith("https://api.novita.ai/openai")
    assert b._client.headers["authorization"] == "Bearer k-novita"
    assert a._client.headers["authorization"] == "Bearer k-openai"


def test_model_without_its_key_env_is_not_routable(monkeypatch, tmp_path) -> None:
    import shutil

    from dietgate.core.config import load_models
    from tests.conftest import TEST_DIR

    cfg = tmp_path / "config"
    shutil.copytree(TEST_DIR / "config", cfg)
    with (cfg / "models.yaml").open("a", encoding="utf-8") as f:
        f.write(
            "  - id: novita-glm\n    provider: openai_compat\n    api_key_env: NOVITA_API_KEY\n"
            "    base_url: https://api.novita.ai/openai\n    upstream_model: zai-org/glm-5.3-flash\n"
            "    price_in_per_mtok: 0.15\n    price_out_per_mtok: 0.5\n"
        )
    monkeypatch.delenv("NOVITA_API_KEY", raising=False)
    assert "novita-glm" not in load_models(cfg)
    monkeypatch.setenv("NOVITA_API_KEY", "k")
    spec = load_models(cfg)["novita-glm"]
    assert spec.upstream_model == "zai-org/glm-5.3-flash" and spec.base_url == "https://api.novita.ai/openai"
