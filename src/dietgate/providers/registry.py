"""Model id -> provider instance registry."""
from __future__ import annotations

import os
from typing import AsyncIterator

from dietgate.core.config import AppConfig, ModelSpec
from dietgate.providers.base import ChatRequest, ChatResponse, Provider, ProviderError, StreamChunk, Usage
from dietgate.providers.mock import MockProvider
from dietgate.settings import Settings


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}
        self._model_owner: dict[str, str] = {}

    def register(self, provider: Provider, model_ids: list[str]) -> None:
        self._providers[provider.name] = provider
        for mid in model_ids:
            self._model_owner[mid] = provider.name

    def has_model(self, model_id: str) -> bool:
        return model_id in self._model_owner

    def resolve(self, model_id: str) -> Provider | None:
        name = self._model_owner.get(model_id)
        return self._providers.get(name) if name else None

    def model_ids(self) -> list[str]:
        return list(self._model_owner.keys())

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()


class _RealProviderBase:
    """Shared plumbing for real HTTP providers (pooling, timeouts).

    Note: `_payload()` never forwards `req.metadata`, so demo-only fields
    (`metadata.mock_*`) can never leak to a real upstream (spec section 14).
    """

    name = "real"

    def __init__(
        self, base_url: str, api_key: str, timeout_s: float, transport=None, name: str | None = None
    ) -> None:
        import httpx

        if name:
            self.name = name  # several providers of the same kind need distinct registry names

        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_s),
            http2=True,
            transport=transport,  # tests inject httpx.MockTransport here
        )

    def _payload(self, req: ChatRequest) -> dict:
        return {
            "model": req.model,
            "messages": req.messages,
            "stream": req.stream,
            **({"temperature": req.temperature} if req.temperature is not None else {}),
            **({"max_tokens": req.max_tokens} if req.max_tokens else {}),
            **({"tools": req.tools} if req.tools else {}),
            **({"response_format": req.response_format} if req.response_format else {}),
        }

    async def aclose(self) -> None:
        await self._client.aclose()


class OpenAICompatProvider(_RealProviderBase):
    """OpenAI-compatible upstream: OpenAI, vLLM, OpenRouter, ..."""

    name = "openai_compat"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_s: float,
        transport=None,
        name: str | None = None,
        specs: dict | None = None,
    ) -> None:
        super().__init__(base_url, api_key, timeout_s, transport=transport, name=name)
        self._specs = specs or {}   # gateway model id -> ModelSpec (upstream id, extra body, ...)

    def _payload(self, req: ChatRequest) -> dict:
        payload = super()._payload(req)
        spec = self._specs.get(req.model)
        if spec is not None:
            payload["model"] = spec.upstream_model or req.model
            for key, value in spec.extra_body.items():
                payload.setdefault(key, value)   # config can add fields, never override core ones
        # Some newer OpenAI models reject "max_tokens" and want "max_completion_tokens".
        # Set OPENAI_MAX_TOKENS_PARAM=max_completion_tokens for those models.
        param = os.environ.get("OPENAI_MAX_TOKENS_PARAM", "max_tokens")
        if param != "max_tokens" and "max_tokens" in payload:
            payload[param] = payload.pop("max_tokens")
        return payload

    async def chat(self, req: ChatRequest) -> ChatResponse:
        resp = await self._client.post("/chat/completions", json=self._payload(req))
        if resp.status_code >= 500 or resp.status_code == 429:
            from dietgate.providers.base import ProviderError

            raise ProviderError(f"upstream {req.model} -> HTTP {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return ChatResponse(
            content=choice["message"].get("content") or "",
            finish_reason=choice.get("finish_reason") or "stop",
            usage=Usage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
            ),
            model=data.get("model", req.model),
        )

    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[StreamChunk]:
        import json as _json

        payload = self._payload(req)
        spec = self._specs.get(req.model)
        if spec is None or spec.stream_include_usage:
            payload["stream_options"] = {"include_usage": True}
        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code >= 500 or resp.status_code == 429:
                from dietgate.providers.base import ProviderError

                raise ProviderError(f"upstream {req.model} -> HTTP {resp.status_code}")
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if raw == "[DONE]":
                    break
                data = _json.loads(raw)
                choice = (data.get("choices") or [{}])[0]
                usage = data.get("usage")
                yield StreamChunk(
                    delta=choice.get("delta", {}).get("content") or "",
                    finish_reason=choice.get("finish_reason"),
                    usage=Usage(
                        prompt_tokens=int(usage.get("prompt_tokens", 0)),
                        completion_tokens=int(usage.get("completion_tokens", 0)),
                    )
                    if usage
                    else None,
                    model=data.get("model"),
                )


class AnthropicProvider(_RealProviderBase):
    """Anthropic Messages API adapter (OpenAI-style request -> Messages API)."""

    name = "anthropic"

    def __init__(
        self, api_key: str, timeout_s: float, version: str = "2023-06-01", transport=None,
        base_url: str = "https://api.anthropic.com/v1",
    ) -> None:
        super().__init__(base_url, api_key, timeout_s, transport=transport)
        self._client.headers.update({"x-api-key": api_key, "anthropic-version": version})
        self._client.headers.pop("Authorization", None)

    def _payload(self, req: ChatRequest) -> dict:
        system = next((m["content"] for m in req.messages if m.get("role") == "system"), None)
        msgs = [
            {"role": m["role"], "content": m["content"]}
            for m in req.messages
            if m.get("role") in ("user", "assistant")
        ]
        return {
            "model": req.model,
            "messages": msgs,
            "stream": bool(req.stream),
            "max_tokens": req.max_tokens or 1024,  # required by the Messages API
            **({"system": system} if system else {}),
            # Anthropic accepts temperature 0..1 (OpenAI-style requests allow up to 2)
            **({"temperature": min(float(req.temperature), 1.0)} if req.temperature is not None else {}),
        }

    async def chat(self, req: ChatRequest) -> ChatResponse:
        resp = await self._client.post("/messages", json=self._payload(req))
        if resp.status_code >= 500 or resp.status_code == 429:
            from dietgate.providers.base import ProviderError

            raise ProviderError(f"upstream {req.model} -> HTTP {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", []))
        usage = data.get("usage") or {}
        return ChatResponse(
            content=text,
            finish_reason=data.get("stop_reason") or "stop",
            usage=Usage(
                prompt_tokens=int(usage.get("input_tokens", 0)),
                completion_tokens=int(usage.get("output_tokens", 0)),
            ),
            model=data.get("model", req.model),
        )

    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[StreamChunk]:
        import json as _json

        payload = self._payload(req)
        payload["stream"] = True
        input_tokens = 0
        async with self._client.stream("POST", "/messages", json=payload) as resp:
            if resp.status_code >= 500 or resp.status_code == 429:
                raise ProviderError(f"upstream {req.model} -> HTTP {resp.status_code}")
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = _json.loads(line[6:])
                etype = event.get("type")
                if etype == "message_start":
                    # input token count only arrives here, not in message_delta
                    msg_usage = (event.get("message") or {}).get("usage") or {}
                    input_tokens = int(msg_usage.get("input_tokens", 0))
                elif etype == "content_block_delta":
                    yield StreamChunk(delta=event.get("delta", {}).get("text", ""), model=req.model)
                elif etype == "message_delta":
                    usage = event.get("usage") or {}
                    yield StreamChunk(
                        finish_reason=event.get("delta", {}).get("stop_reason"),
                        usage=Usage(
                            prompt_tokens=input_tokens,
                            completion_tokens=int(usage.get("output_tokens", 0)),
                        ),
                    )
                elif etype == "error":
                    raise ProviderError(f"upstream {req.model} stream error: {event.get('error')}")


def build_registry(catalog: dict[str, ModelSpec], settings: Settings, app_cfg: AppConfig) -> ProviderRegistry:
    reg = ProviderRegistry()
    mocks = {mid: spec for mid, spec in catalog.items() if spec.is_mock}
    if mocks:
        reg.register(MockProvider(mocks, seed=settings.seed), list(mocks.keys()))
    timeout = app_cfg.upstream_timeout_s
    openai_groups: dict[tuple[str, str], dict[str, ModelSpec]] = {}
    anthropic_ids: list[str] = []
    for mid, spec in catalog.items():
        if spec.is_mock:
            continue
        if spec.provider == "openai_compat":
            base = spec.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            key_env = spec.api_key_env or "OPENAI_API_KEY"
            openai_groups.setdefault((base.rstrip("/"), key_env), {})[mid] = spec
        elif spec.provider == "anthropic":
            anthropic_ids.append(mid)
    default_key = (os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"), "OPENAI_API_KEY")
    for (base, key_env), specs in openai_groups.items():
        name = "openai_compat" if (base, key_env) == default_key else f"openai_compat:{key_env}@{base}"
        reg.register(
            OpenAICompatProvider(base, os.environ[key_env], timeout, name=name, specs=specs),
            list(specs.keys()),
        )
    if anthropic_ids:
        reg.register(AnthropicProvider(os.environ["ANTHROPIC_API_KEY"], timeout), anthropic_ids)
    return reg
