"""Learning + persistence bridges used by the hot path (called from pipeline)."""
from __future__ import annotations

import time

from dietgate.core.context import RequestContext


def note_learning(rt, ctx: RequestContext, content: str) -> None:
    """Feed the learner map and the sampled shadow evaluator."""
    learner = getattr(rt, "learner", None)
    if learner is None or ctx.cache_hit or ctx.shortcut is not None or not ctx.chosen_model:
        return
    learner.note_task(ctx.task_type, ctx.task_id, ctx.chosen_model)
    evaluator = getattr(rt, "evaluator", None)
    expected = ctx.metadata.get("mock_expected")
    if evaluator is not None and isinstance(expected, str):
        evaluator.maybe_submit(
            request_id=ctx.request_id,
            task_id=ctx.task_id,
            task_type=ctx.task_type,
            chosen_model=ctx.chosen_model,
            baseline_model=rt.tasks_cfg.get(ctx.task_type, rt.tasks_cfg["unknown"]).baseline,
            expected=expected,
            answer=content,
            ts=time.time(),
        )


def emit_request_event(rt, ctx: RequestContext) -> None:
    """Step 10: hand the request event to the batched writer (never blocks)."""
    writer = getattr(rt, "writer", None)
    if writer is None:
        return
    from dietgate.storage.writer import RequestEvent

    provider = None
    if ctx.chosen_model:
        spec = rt.catalog.get(ctx.chosen_model)
        provider = spec.provider if spec else ctx.chosen_model
    writer.enqueue(
        RequestEvent(
            id=ctx.request_id,
            ts=time.time(),
            api_key_id=ctx.api_key_id,
            task_id=ctx.task_id,
            task_type=ctx.task_type,
            requested_model=ctx.requested_model,
            chosen_model=ctx.chosen_model,
            provider=provider,
            policy=ctx.policy,
            explore=ctx.explore,
            decision_reason=ctx.decision_reason,
            cache_hit=ctx.cache_hit,
            shortcut=ctx.shortcut,
            stream=ctx.stream,
            status=ctx.status,
            prompt_tokens=ctx.prompt_tokens,
            completion_tokens=ctx.completion_tokens,
            cost_usd=ctx.cost_usd,
            baseline_cost_usd=ctx.baseline_cost_usd,
            ttft_ms=ctx.ttft_ms,
            total_ms=ctx.total_ms,
            overhead_ms=ctx.overhead_ms,
            attempts=ctx.attempts,
            error=ctx.error,
        )
    )
