"""Hot-path orchestration (spec section 5).

Rules: no disk, no DB, no blocking calls. Everything this module awaits is an
upstream provider call (or a shared singleflight future for one); every other
step is in-memory work plus timestamps. Streaming machinery lives in
core/streaming.py, learning/event emission in core/events.py.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import AsyncIterator

from dietgate.core import events
from dietgate.core.cache import cache_key
from dietgate.core.classifier import classify
from dietgate.core.context import RequestContext, now_ns
from dietgate.core.cost import baseline_cost_usd, request_cost_usd
from dietgate.core.errors import BadRequestError, DietGateError, UpstreamError
from dietgate.core.render import dg_headers, render_completion
from dietgate.core.shortcuts import run_shortcuts
from dietgate.providers.base import ChatRequest, ChatResponse
from dietgate.utils.timing import ns_to_ms


class Pipeline:
    """Executes the request lifecycle steps in spec order."""

    def __init__(self, rt) -> None:
        self.rt = rt

    # ------------------------------------------------------------------ entry
    async def handle_nonstream(self, ctx: RequestContext, body: dict) -> tuple[bytes, dict]:
        try:
            self._preprocess(ctx, body)
            cached = self._cache_lookup(ctx)
            if cached is not None:
                ctx.cache_hit = True
                ctx.chosen_model = cached.model
                return self._finish_nonstream(ctx, cached)
            shortcut = run_shortcuts(ctx.messages, self.rt.app_cfg.shortcuts_enabled)
            if shortcut is not None:
                ctx.shortcut = shortcut.name
                ctx.chosen_model = "shortcut"
                return self._finish_nonstream(ctx, self._local_response(ctx, shortcut.content))
            chain, _policy = self._resolve_chain(ctx)
            chat_req = self._chat_request(ctx, chain[0])
            cache = self._active_cache(ctx)
            key = self._cache_key(ctx)
            if cache is not None and key is not None:
                resp = await cache.singleflight(
                    key, lambda: self._call_with_fallbacks(ctx, chat_req, chain)
                )
                cache.put(key, resp)
            else:
                resp = await self._call_with_fallbacks(ctx, chat_req, chain)
            return self._finish_nonstream(ctx, resp)
        except DietGateError as exc:
            self._fail(ctx, exc)
            raise

    async def handle_stream(
        self, ctx: RequestContext, body: dict
    ) -> tuple[AsyncIterator[bytes], dict]:
        from dietgate.core.streaming import forward_stream

        try:
            self._preprocess(ctx, body)
            shortcut = run_shortcuts(ctx.messages, self.rt.app_cfg.shortcuts_enabled)
            chain: list[str] | None = None
            chat_req: ChatRequest | None = None
            if shortcut is None:  # streams bypass the cache (see DECISIONS D5)
                chain, _policy = self._resolve_chain(ctx)
                chat_req = self._chat_request(ctx, chain[0])
        except DietGateError as exc:
            self._fail(ctx, exc)
            raise
        headers = {
            "X-Dg-Request-Id": ctx.request_id,
            "X-Dg-Task-Id": ctx.task_id,
            "X-Dg-Cache": "miss",
        }
        return forward_stream(self.rt, ctx, self, chat_req, chain, shortcut), headers

    # ------------------------------------------------------------ pre-upstream
    def _preprocess(self, ctx: RequestContext, body: dict) -> None:
        ctx.t_auth = now_ns()
        if self.rt.authn is not None:  # step 2: auth + limits + spend cap
            ctx.api_key_id = self.rt.authn.check(ctx.api_key_id)
        if ctx.requested_model != "auto" and not self.rt.registry.has_model(ctx.requested_model):
            raise BadRequestError(f"unknown model: {ctx.requested_model}")
        ctx.task_type = classify(ctx.messages, ctx.header_task_type)  # step 3
        ctx.t_classified = now_ns()

    def _cache_usable(self, ctx: RequestContext) -> bool:
        cache = self.rt.cache
        return ctx.cache_allowed and cache is not None and cache.enabled

    def _cache_key(self, ctx: RequestContext) -> str | None:
        if not self._cache_usable(ctx):
            return None
        return cache_key(ctx.messages, ctx.tools, ctx.response_format, ctx.max_tokens)

    def _cache_lookup(self, ctx: RequestContext) -> ChatResponse | None:
        key = self._cache_key(ctx)  # step 5: model-agnostic exact-match cache
        if key is None:
            return None
        return self.rt.cache.get(key)

    def _active_cache(self, ctx: RequestContext):
        return self.rt.cache if self._cache_key(ctx) else None

    def _resolve_chain(self, ctx: RequestContext) -> tuple[list[str], str]:
        """Step 7: model 'auto' -> router Decision; pinned model -> no routing."""
        if ctx.requested_model == "auto":
            decision = self.rt.router.decide(ctx)
            ctx.policy = decision.policy
            ctx.explore = decision.explore
            ctx.decision_reason = decision.reason
            ctx.fallbacks = decision.fallbacks
            return [decision.model_id, *decision.fallbacks], decision.policy
        task_cfg = self._task_cfg(ctx.task_type)
        fallbacks = [m for m in task_cfg.candidates if m != ctx.requested_model]
        return [ctx.requested_model, *fallbacks], "pinned"

    def _chat_request(self, ctx: RequestContext, model_id: str) -> ChatRequest:
        return ChatRequest(
            model=model_id,
            messages=ctx.messages,
            task_id=ctx.task_id,
            task_type=ctx.task_type,
            temperature=ctx.temperature,
            max_tokens=ctx.max_tokens,
            tools=ctx.tools,
            response_format=ctx.response_format,
            stream=ctx.stream,
            metadata=ctx.metadata,
        )

    def _task_cfg(self, task_type: str):
        return self.rt.tasks_cfg.get(task_type) or self.rt.tasks_cfg["unknown"]

    def _local_response(self, ctx: RequestContext, content: str) -> ChatResponse:
        from dietgate.core.cost import usage_estimates
        from dietgate.providers.base import Usage

        pt, ct = usage_estimates(ctx.messages, content)
        return ChatResponse(
            content=content, finish_reason="stop", usage=Usage(pt, ct), model="shortcut"
        )

    # -------------------------------------------------------------- upstream
    async def _call_with_fallbacks(
        self, ctx: RequestContext, chat_req: ChatRequest, chain: list[str]
    ) -> ChatResponse:
        timeout = self.rt.app_cfg.upstream_timeout_s
        verify = self._cascade_verifier(ctx)
        ctx.extra_attempt_cost_usd = 0.0
        last_err: Exception | None = None
        for attempt, model_id in enumerate(chain):
            provider = self.rt.registry.resolve(model_id)
            if provider is None:
                last_err = BadRequestError(f"model not routable: {model_id}")
                continue
            if attempt == 0:
                ctx.t_upstream_request_sent = now_ns()
            t0 = now_ns()
            try:
                resp = await asyncio.wait_for(
                    provider.chat(replace(chat_req, model=model_id)), timeout=timeout
                )
            except DietGateError:
                raise
            except Exception as exc:  # timeout, connection, upstream 5xx -> fallback
                last_err = exc
                continue
            latency_ms = ns_to_ms(now_ns() - t0)
            cost_i = request_cost_usd(
                self.rt.catalog, model_id, resp.usage.prompt_tokens, resp.usage.completion_tokens
            )
            if self.rt.stats is not None:
                self.rt.stats.record_call(ctx.task_type, model_id, cost_i, latency_ms, time.time())
            if verify is not None and not verify(resp.content):
                ctx.extra_attempt_cost_usd += cost_i  # rejected answer still cost money
                last_err = UpstreamError(f"cascade verify failed on {model_id}")
                continue
            ctx.chosen_model = model_id
            ctx.attempts = attempt + 1
            return resp
        raise UpstreamError(f"all {len(chain)} attempts failed; last: {last_err}")

    def _cascade_verifier(self, ctx: RequestContext):
        """Cascade only: ground truth from the simulator; None means accept-first."""
        if ctx.policy != "cascade":
            return None
        expected = ctx.metadata.get("mock_expected")
        if not isinstance(expected, str):
            return None  # no ground truth -> accept first answer (DECISIONS D7)
        return lambda content: content == expected

    # ------------------------------------------------------------------ cost
    def _apply_costs(self, ctx: RequestContext, resp: ChatResponse) -> None:
        ctx.prompt_tokens = resp.usage.prompt_tokens
        ctx.completion_tokens = resp.usage.completion_tokens
        catalog = self.rt.catalog
        chosen = ctx.chosen_model or ctx.requested_model
        ctx.cost_usd = request_cost_usd(catalog, chosen, ctx.prompt_tokens, ctx.completion_tokens)
        ctx.cost_usd += ctx.extra_attempt_cost_usd  # cascade rejections
        if ctx.cache_hit:
            ctx.cost_usd = 0.0  # a cache hit bills nothing
        baseline = self._task_cfg(ctx.task_type).baseline
        ctx.baseline_cost_usd = baseline_cost_usd(
            catalog, baseline, ctx.prompt_tokens, ctx.completion_tokens
        )

    def _record_spend(self, ctx: RequestContext) -> None:
        if self.rt.authn is not None and ctx.api_key_id:
            self.rt.authn.add_spend(ctx.api_key_id, ctx.cost_usd)

    # -------------------------------------------------------------- lifecycle
    def _finish_nonstream(self, ctx: RequestContext, resp: ChatResponse) -> tuple[bytes, dict]:
        ctx.t_done_in = now_ns()
        self._apply_costs(ctx, resp)
        self._record_spend(ctx)
        events.note_learning(self.rt, ctx, resp.content)
        payload = render_completion(ctx, resp)
        ctx.t_done_out = now_ns()
        self._finalize(ctx)
        return payload, dg_headers(ctx)

    def _finalize(self, ctx: RequestContext) -> None:
        ctx.status = 200 if ctx.error is None else 502
        ctx.overhead_ms = ctx.overhead_ms_computed()
        ctx.ttft_ms = ctx.ttft_ms_computed()
        ctx.total_ms = ctx.total_ms_computed()
        events.emit_request_event(self.rt, ctx)

    def _fail(self, ctx: RequestContext, exc: DietGateError) -> None:
        ctx.status = exc.status
        ctx.error = exc.message
        if ctx.cost_usd == 0.0 and ctx.extra_attempt_cost_usd:
            ctx.cost_usd = ctx.extra_attempt_cost_usd  # cascade attempts were spent
        ctx.overhead_ms = ctx.overhead_ms_computed()
        ctx.total_ms = ctx.total_ms_computed()
        events.emit_request_event(self.rt, ctx)
