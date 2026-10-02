"""SSE encoding helpers and OpenAI chunk formatting."""
from __future__ import annotations

import orjson


def sse_data(payload: dict | list | str) -> bytes:
    if isinstance(payload, str):
        raw = payload.encode()
    else:
        raw = orjson.dumps(payload)
    return b"data: " + raw + b"\n\n"


def sse_comment(text: str) -> bytes:
    return b":" + text.encode() + b"\n\n"


SSE_DONE = b"data: [DONE]\n\n"


def openai_chunk(
    request_id: str,
    model: str,
    *,
    delta: dict | None = None,
    finish_reason: str | None = None,
    usage: dict | None = None,
) -> dict:
    chunk: dict = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": 0,  # replaced by caller if needed
        "model": model,
        "choices": [
            {"index": 0, "delta": delta if delta is not None else {}, "finish_reason": finish_reason}
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def usage_dict(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
