"""Batched async writer fed by an asyncio.Queue (spec section 9).

The hot path calls `enqueue()` (non-blocking put_nowait); a full queue drops
the event and increments a counter instead of ever blocking a response.
All SQL lives in storage.db methods with `?` placeholders.
"""
from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass

from dietgate.core.config import WriterConfig
from dietgate.storage.db import Database


@dataclass
class RequestEvent:
    id: str
    ts: float
    api_key_id: str | None
    task_id: str
    task_type: str
    requested_model: str | None
    chosen_model: str | None
    provider: str | None
    policy: str | None
    explore: bool
    decision_reason: str | None
    cache_hit: bool
    shortcut: str | None
    stream: bool
    status: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    baseline_cost_usd: float | None
    ttft_ms: float | None
    total_ms: float | None
    overhead_ms: float | None
    attempts: int
    error: str | None

    def row(self) -> tuple:
        return (
            self.id, self.ts, self.api_key_id, self.task_id, self.task_type, self.requested_model,
            self.chosen_model, self.provider, self.policy, int(self.explore),
            self.decision_reason, int(self.cache_hit), self.shortcut, int(self.stream),
            self.status, self.prompt_tokens, self.completion_tokens, self.cost_usd,
            self.baseline_cost_usd, self.ttft_ms, self.total_ms, self.overhead_ms,
            self.attempts, self.error,
        )


@dataclass
class OutcomeEvent:
    task_id: str
    ts: float
    success: bool
    score: float | None
    source: str

    def row(self) -> tuple:
        return (self.task_id, self.ts, int(self.success), self.score, self.source)


@dataclass
class ShadowEvent:
    id: str
    request_id: str
    ts: float
    baseline_model: str | None
    judge_model: str | None
    verdict: str | None
    eval_cost_usd: float

    def row(self) -> tuple:
        return (self.id, self.request_id, self.ts, self.baseline_model, self.judge_model,
                self.verdict, self.eval_cost_usd)


@dataclass
class SnapshotEvent:
    ts: float
    task_type: str
    model_id: str
    n_labeled: int
    successes: int
    cost_sum_usd: float

    def row(self) -> tuple:
        return (self.ts, self.task_type, self.model_id, self.n_labeled, self.successes,
                self.cost_sum_usd)


class Writer:
    def __init__(self, db: Database, cfg: WriterConfig) -> None:
        self.db = db
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=cfg.queue_size)
        self.dropped_events = 0
        self.last_error: str | None = None
        self._batch_size = max(1, cfg.batch_size)
        self._flush_interval_s = max(0.01, cfg.flush_interval_ms / 1000.0)
        self._task: asyncio.Task | None = None

    def enqueue(self, event: RequestEvent | OutcomeEvent | ShadowEvent | SnapshotEvent) -> None:
        """Step 10 of the hot path: never blocks, never raises."""
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped_events += 1

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="dietgate-writer")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self.drain()

    async def drain(self) -> None:
        """Immediately persist everything queued (used on shutdown and by tests)."""
        while True:
            batch = await self._grab_batch()
            if not batch:
                return
            await self._write(batch)

    async def _grab_batch(self) -> list:
        batch: list = []
        for _ in range(self._batch_size):
            if self.queue.empty():
                break
            try:
                batch.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:  # background task consumed it first
                break
        return batch

    async def _run(self) -> None:
        while True:
            batch = await self._grab_batch()
            if batch:
                await self._write(batch)
            else:
                with suppress(asyncio.TimeoutError):
                    first = await asyncio.wait_for(self.queue.get(), timeout=self._flush_interval_s)
                    await self._write([first])

    async def _write(self, batch: list) -> None:
        try:
            req_rows = [e.row() for e in batch if isinstance(e, RequestEvent)]
            if req_rows:
                await self.db.insert_requests(req_rows)
            outcome_rows = [e.row() for e in batch if isinstance(e, OutcomeEvent)]
            if outcome_rows:
                await self.db.insert_outcomes(outcome_rows)
            shadow_rows = [e.row() for e in batch if isinstance(e, ShadowEvent)]
            if shadow_rows:
                await self.db.insert_shadow_evals(shadow_rows)
            snap_rows = [e.row() for e in batch if isinstance(e, SnapshotEvent)]
            if snap_rows:
                await self.db.insert_arm_snapshots(snap_rows)
            await self.db.commit()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Persistence must never take the gateway down; the data is lost
            # but the failure stays visible for the admin API.
            self.dropped_events += len(batch)
            self.last_error = f"{type(exc).__name__}: {exc}"
