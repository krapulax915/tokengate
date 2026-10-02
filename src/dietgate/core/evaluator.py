"""Shadow evaluation (spec 6.5): sampled, off the hot path, honest accounting.

In mock mode the judge is a deterministic comparison against the simulator's
ground truth (`metadata.mock_expected`), so learning costs nothing upstream.
A real LLM judge and its billing belong to the real-provider mode.
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

from dietgate.core.config import EvaluatorConfig
from dietgate.core.learner import Learner
from dietgate.storage.writer import ShadowEvent, Writer
from dietgate.utils.ids import new_eval_id


@dataclass
class EvalCandidate:
    request_id: str
    task_id: str
    task_type: str
    ts: float
    chosen_model: str | None
    baseline_model: str | None
    expected: str | None
    answer: str


class Evaluator:
    def __init__(
        self,
        learner: Learner,
        writer: Writer | None,
        cfg: EvaluatorConfig,
        sample_rng_seed: int = 7,
    ) -> None:
        self.learner = learner
        self.writer = writer
        self.cfg = cfg
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._rng = random.Random(sample_rng_seed)
        self._task: asyncio.Task | None = None
        self.processed = 0
        self.skipped_labeled = 0

    def maybe_submit(
        self,
        request_id: str,
        task_id: str,
        task_type: str,
        chosen_model: str | None,
        baseline_model: str | None,
        expected: str | None,
        answer: str,
        ts: float,
    ) -> None:
        """Sampled, non-blocking; candidates without ground truth are dropped."""
        if self.cfg.sample_rate <= 0.0 or expected is None:
            return
        if self._rng.random() > self.cfg.sample_rate:
            return
        try:
            self.queue.put_nowait(
                EvalCandidate(
                    request_id=request_id,
                    task_id=task_id,
                    task_type=task_type,
                    ts=ts,
                    chosen_model=chosen_model,
                    baseline_model=baseline_model,
                    expected=expected,
                    answer=answer,
                )
            )
        except asyncio.QueueFull:
            pass

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="dietgate-evaluator")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            import contextlib

            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        while True:
            candidate = await self.queue.get()
            await asyncio.sleep(self.cfg.delay_s)  # let client feedback win the race
            self._evaluate(candidate)

    def _evaluate(self, candidate: EvalCandidate) -> None:
        info = self.learner.task_arms.get(candidate.task_id)
        if info is not None and info.labeled_success is not None:
            self.skipped_labeled += 1
            return  # a client/checker label already exists
        verdict = "equal_or_better" if candidate.answer == candidate.expected else "worse"
        success = verdict == "equal_or_better"
        self.processed += 1
        if self.writer is not None:
            # Mock mode: deterministic judge, zero upstream cost (DECISIONS D9).
            self.writer.enqueue(
                ShadowEvent(
                    id=new_eval_id(),
                    request_id=candidate.request_id,
                    ts=candidate.ts,
                    baseline_model=candidate.baseline_model,
                    judge_model="mock-ground-truth",
                    verdict=verdict,
                    eval_cost_usd=0.0,
                )
            )
        self.learner.submit(
            candidate.task_id, success=success, score=None, source="shadow_judge",
            ts=candidate.ts,
        )
