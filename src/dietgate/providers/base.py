"""Provider interface (spec section 7)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ChatRequest:
    """Normalized upstream request. `metadata` may carry demo-only fields;
    the gateway strips `metadata.mock_*` before talking to real providers."""

    model: str
    messages: list[dict]
    task_id: str = ""
    task_type: str = "unknown"
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list | None = None
    response_format: dict | None = None
    stream: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class ChatResponse:
    content: str
    finish_reason: str
    usage: Usage
    model: str


@dataclass
class StreamChunk:
    delta: str = ""
    finish_reason: str | None = None
    usage: Usage | None = None
    model: str | None = None


class ProviderError(Exception):
    """Raised for upstream failures (connection, timeout, 5xx)."""


class Provider(Protocol):
    name: str

    async def chat(self, req: ChatRequest) -> ChatResponse: ...

    def chat_stream(self, req: ChatRequest) -> AsyncIterator[StreamChunk]: ...

    async def aclose(self) -> None: ...
