"""Streaming machinery for the hot path (spec section 5, step 8 + forwarding).

`forward_stream` yields SSE bytes; the pipeline object passed in provides the
finalize/cost hooks (duck-typed to avoid an import cycle).
"""
from __future__ import annotations

from dataclasses import replace
from typing import AsyncIterator

from dietgate.core import events
from dietgate.core.context import RequestContext, now_ns
from dietgate.core.cost import usage_estimates
from dietgate.core.errors import BadRequestError, DietGateError, UpstreamError
from dietgate.core.render import cost_comment, error_payload
from dietgate.providers.base import ChatRequest, ChatResponse, StreamChunk, Usage
from dietgate.utils import sse

_STREAM_CHARS = 16


async def open_stream_with_fallbacks(
    rt, ctx: RequestContext, chat_req: ChatRequest, chain: list[str]
) -> tuple[AsyncIterator[StreamChunk], StreamChunk, str]:
    """Try candidates until one produces a first chunk; records Dg timings."""
    last_err: Exception | None = None
    for attempt, model_id in enumerate(chain):
        provider = rt.registry.resolve(model_id)
        if provider is None:
            last_err = BadRequestError(f"model not routable: {model_id}")
            continue
        if attempt == 0:
            ctx.t_upstream_request_sent = now_ns()
        agen = provider.chat_stream(replace(chat_req, model=model_id))
        try:
            first = await agen.__anext__()
        except StopAsyncIteration:
            last_err = UpstreamError(f"model {model_id} produced an empty stream")
            continue
        except Exception as exc:
            last_err = exc
            continue
        ctx.t_first_chunk_in = now_ns()
        ctx.chosen_model = model_id
        ctx.attempts = attempt + 1
        return agen, first, model_id
    raise UpstreamError(f"all {len(chain)} attempts failed; last: {last_err}")


async def forward_stream(
    rt,
    ctx: RequestContext,
    pipeline,
    chat_req: ChatRequest | None,
    chain: list[str] | None,
    shortcut,
) -> AsyncIterator[bytes]:
    """Forward the stream (or a local shortcut answer) and emit the cost trailer."""
    if shortcut is not None:
        async for piece in _forward_shortcut(ctx, rt, pipeline, shortcut):
            yield piece
        return
    model = chat_req.model
    try:
        agen, first, model = await open_stream_with_fallbacks(rt, ctx, chat_req, chain)
    except DietGateError as exc:
        pipeline._fail(ctx, exc)
        yield sse.sse_data(error_payload(exc))
        yield sse.SSE_DONE
        return
    yield sse.sse_data(
        sse.openai_chunk(ctx.request_id, model, delta={"role": "assistant", "content": ""})
    )
    content_parts: list[str] = []
    usage: Usage | None = None
    finish = "stop"
    started = False
    try:
        async for chunk in _chain_first(agen, first):
            if chunk.model:
                model = chunk.model
            if chunk.delta:
                if not started:
                    started = True
                    ctx.t_first_chunk_out = now_ns()
                content_parts.append(chunk.delta)
            if chunk.finish_reason:
                finish = chunk.finish_reason
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.delta:
                yield sse.sse_data(
                    sse.openai_chunk(ctx.request_id, model, delta={"content": chunk.delta})
                )
        yield sse.sse_data(sse.openai_chunk(ctx.request_id, model, finish_reason=finish))
    except Exception as exc:
        ctx.error = f"mid-stream failure: {exc}"
    content = "".join(content_parts)
    ctx.t_done_in = now_ns()
    if usage is None:
        pt, ct = usage_estimates(ctx.messages, content)
        usage = Usage(prompt_tokens=pt, completion_tokens=ct)
    pipeline._apply_costs(ctx, ChatResponse(content, finish, usage, model))
    pipeline._record_spend(ctx)
    events.note_learning(rt, ctx, content)
    trailer = sse.sse_comment(cost_comment(ctx))
    ctx.t_done_out = now_ns()
    pipeline._finalize(ctx)
    yield trailer + sse.SSE_DONE


async def _forward_shortcut(ctx: RequestContext, rt, pipeline, shortcut) -> AsyncIterator[bytes]:
    """Shortcut answers stream locally: no upstream, no pacing."""
    ctx.shortcut = shortcut.name
    ctx.chosen_model = "shortcut"
    ctx.attempts = 1
    content = shortcut.content
    yield sse.sse_data(
        sse.openai_chunk(ctx.request_id, "shortcut", delta={"role": "assistant", "content": ""})
    )
    for i in range(0, len(content), _STREAM_CHARS):
        piece = content[i : i + _STREAM_CHARS]
        yield sse.sse_data(
            sse.openai_chunk(ctx.request_id, "shortcut", delta={"content": piece})
        )
    pt, ct = usage_estimates(ctx.messages, content)
    yield sse.sse_data(
        sse.openai_chunk(
            ctx.request_id, "shortcut", finish_reason="stop", usage=sse.usage_dict(pt, ct)
        )
    )
    ctx.t_done_in = now_ns()
    pipeline._apply_costs(ctx, ChatResponse(content, "stop", Usage(pt, ct), "shortcut"))
    pipeline._record_spend(ctx)
    events.note_learning(rt, ctx, content)
    trailer = sse.sse_comment(cost_comment(ctx))
    ctx.t_done_out = now_ns()
    pipeline._finalize(ctx)
    yield trailer + sse.SSE_DONE


async def _chain_first(
    agen: AsyncIterator[StreamChunk], first: StreamChunk
) -> AsyncIterator[StreamChunk]:
    yield first
    async for chunk in agen:
        yield chunk
