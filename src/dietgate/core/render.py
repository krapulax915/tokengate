"""HTTP-facing rendering of pipeline results: bodies, Dg headers, SSE trailer."""
from __future__ import annotations

import time

import orjson

from dietgate.core.context import RequestContext
from dietgate.providers.base import ChatResponse
from dietgate.utils import sse


def render_completion(ctx: RequestContext, resp: ChatResponse) -> bytes:
    body = {
        "id": ctx.request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": ctx.chosen_model or ctx.requested_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": resp.content},
                "finish_reason": resp.finish_reason,
            }
        ],
        "usage": sse.usage_dict(resp.usage.prompt_tokens, resp.usage.completion_tokens),
    }
    return orjson.dumps(body)


def dg_headers(ctx: RequestContext) -> dict[str, str]:
    return {
        "X-Dg-Request-Id": ctx.request_id,
        "X-Dg-Model": ctx.chosen_model or "",
        "X-Dg-Task-Id": ctx.task_id,
        "X-Dg-Cost-Usd": f"{ctx.cost_usd:.8f}",
        "X-Dg-Baseline-Cost-Usd": f"{ctx.baseline_cost_usd:.8f}",
        "X-Dg-Overhead-Ms": f"{ctx.overhead_ms_computed():.3f}",
        "X-Dg-Cache": "hit" if ctx.cache_hit else "miss",
        "X-Dg-Policy": ctx.policy or "pinned",
    }


def cost_comment(ctx: RequestContext) -> str:
    payload = {
        "request_id": ctx.request_id,
        "model": ctx.chosen_model,
        "policy": ctx.policy or "pinned",
        "cost_usd": round(ctx.cost_usd, 8),
        "baseline_cost_usd": round(ctx.baseline_cost_usd, 8),
        "overhead_ms": round(ctx.overhead_ms_computed(), 3),
        "cache_hit": ctx.cache_hit,
        "shortcut": ctx.shortcut,
        "attempts": ctx.attempts,
        "error": ctx.error,
    }
    return f" dg-cost={orjson.dumps(payload).decode()}"


def error_payload(exc) -> dict:
    return {"error": {"message": exc.message, "type": exc.err_type, "code": exc.code}}
