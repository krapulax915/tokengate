"""Deterministic mock LLM provider (spec section 7) - the most important provider.

Free, deterministic, demonstrates learning, enables clean overhead measurement.
Behaviour comes entirely from config/models.yaml `mock:` sections.
"""
from __future__ import annotations

import asyncio
import hashlib
import random
from typing import AsyncIterator

from dietgate.core.config import ModelSpec
from dietgate.providers.base import ChatRequest, ChatResponse, ProviderError, StreamChunk, Usage
from dietgate.utils.tokens import prompt_tokens_estimate

_CHARS_PER_TOKEN = 4
_STREAM_CHARS = 16  # ~4 tokens per streamed chunk


def stable_draw_value(seed: int, task_id: str, model_id: str) -> float:
    """Deterministic uniform(0,1) draw, reproducible across processes."""
    key = f"{seed}:{task_id}:{model_id}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big")).random()


class MockProvider:
    name = "mock"

    def __init__(self, models: dict[str, ModelSpec], seed: int = 42) -> None:
        self._models = {mid: spec for mid, spec in models.items() if spec.is_mock}
        self._seed = seed

    def has_model(self, model_id: str) -> bool:
        return model_id in self._models

    def _spec(self, model_id: str) -> ModelSpec:
        spec = self._models.get(model_id)
        if spec is None:
            raise ProviderError(f"mock provider has no model '{model_id}'")
        return spec

    def is_correct(self, model_id: str, task_id: str, task_type: str) -> bool:
        """Skill draw: True means the mock produces the correct answer."""
        spec = self._spec(model_id)
        p_correct = spec.mock.skill_for(task_type)
        return stable_draw_value(self._seed, task_id, model_id) < p_correct

    # -- content synthesis -------------------------------------------------
    def _content(self, req: ChatRequest, spec: ModelSpec) -> str:
        correct = self.is_correct(req.model, req.task_id, req.task_type)
        expected = req.metadata.get("mock_expected")
        wrong = req.metadata.get("mock_wrong")
        if correct and isinstance(expected, str):
            return expected
        if isinstance(wrong, list) and wrong:
            return str(wrong[0])
        if isinstance(expected, str):
            return f"[mock degraded answer] unable to produce: {expected[:60]}"
        return self._canned(req, spec)

    @staticmethod
    def _canned(req: ChatRequest, spec: ModelSpec) -> str:
        last_user = next(
            (m["content"] for m in reversed(req.messages) if m.get("role") == "user"), ""
        )
        snippet = str(last_user)[:160]
        return (
            f"[mock:{req.model}] This is a deterministic canned reply. "
            f"You said: \"{snippet}\""
        )

    # -- Provider interface --------------------------------------------------
    async def chat(self, req: ChatRequest) -> ChatResponse:
        spec = self._spec(req.model)
        content = self._content(req, spec)
        usage = Usage(
            prompt_tokens=prompt_tokens_estimate(req.messages),
            completion_tokens=max(1, len(content) // _CHARS_PER_TOKEN),
        )
        await self._simulate_latency(spec, usage.completion_tokens)
        return ChatResponse(content=content, finish_reason="stop", usage=usage, model=req.model)

    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[StreamChunk]:
        spec = self._spec(req.model)
        content = self._content(req, spec)
        usage = Usage(
            prompt_tokens=prompt_tokens_estimate(req.messages),
            completion_tokens=max(1, len(content) // _CHARS_PER_TOKEN),
        )
        mock = spec.mock
        ttft_s = (mock.ttft_ms / 1000.0) if mock.ttft_ms > 0 else 0.0
        if ttft_s and mock.jitter_pct:
            ttft_s *= 1.0 + random.uniform(-mock.jitter_pct, mock.jitter_pct) / 100.0
        if ttft_s:
            await asyncio.sleep(ttft_s)
        for i in range(0, len(content), _STREAM_CHARS):
            piece = content[i : i + _STREAM_CHARS]
            if mock.tokens_per_second > 0:
                piece_tokens = max(1, len(piece) // _CHARS_PER_TOKEN)
                await asyncio.sleep(piece_tokens / mock.tokens_per_second)
            yield StreamChunk(delta=piece, model=req.model)
        yield StreamChunk(delta="", finish_reason="stop", usage=usage, model=req.model)

    async def aclose(self) -> None:
        return None

    @staticmethod
    async def _simulate_latency(spec: ModelSpec, completion_tokens: int) -> None:
        mock = spec.mock
        if mock is None:
            return
        ttft_s = (mock.ttft_ms / 1000.0) if mock.ttft_ms > 0 else 0.0
        if ttft_s and mock.jitter_pct:
            ttft_s *= 1.0 + random.uniform(-mock.jitter_pct, mock.jitter_pct) / 100.0
        gen_s = (completion_tokens / mock.tokens_per_second) if mock.tokens_per_second > 0 else 0.0
        total = ttft_s + gen_s
        if total > 0:
            await asyncio.sleep(total)
