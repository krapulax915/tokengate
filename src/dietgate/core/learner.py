"""Background learner: applies outcome labels to arm statistics (spec 6.6).

Idempotent per task_id: a second label replaces the first in both the DB row
and the in-memory arm stats (adjustment by delta while the task is known).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from dietgate.core.stats import ArmStatsStore
from dietgate.storage.writer import OutcomeEvent, Writer

_MAX_TASKS = 200_000


@dataclass
class TaskArmInfo:
    task_type: str
    model_ids: list[str]
    labeled_success: bool | None = None


class Learner:
    def __init__(self, stats: ArmStatsStore, writer: Writer | None = None) -> None:
        self.stats = stats
        self.writer = writer
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)
        self.task_arms: dict[str, TaskArmInfo] = {}
        self._task: asyncio.Task | None = None
        self.dropped = 0

    # ------------------------------------------------------------- hot path
    def note_task(self, task_type: str, task_id: str, model_id: str) -> None:
        """Record that this task used this model arm (called per request)."""
        info = self.task_arms.get(task_id)
        if info is None:
            if len(self.task_arms) >= _MAX_TASKS:
                self.task_arms.pop(next(iter(self.task_arms)))
            info = self.task_arms[task_id] = TaskArmInfo(task_type=task_type, model_ids=[])
        if model_id not in info.model_ids:
            info.model_ids.append(model_id)

    def has_task(self, task_id: str) -> bool:
        return task_id in self.task_arms

    def submit(self, task_id: str, success: bool, score: float | None, source: str, ts: float) -> bool:
        try:
            self.queue.put_nowait((task_id, success, score, source, ts))
            return True
        except asyncio.QueueFull:
            self.dropped += 1
            return False

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="dietgate-learner")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            import contextlib

            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self.drain()

    async def drain(self) -> None:
        while True:
            try:
                item = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._apply(item)

    async def _run(self) -> None:
        while True:
            item = await self.queue.get()
            self._apply(item)

    # --------------------------------------------------------------- labels
    def _apply(self, item: tuple) -> None:
        task_id, success, score, source, ts = item
        self.apply_label(task_id, success, score, source, ts)

    def apply_label(
        self, task_id: str, success: bool, score: float | None, source: str, ts: float
    ) -> None:
        info = self.task_arms.get(task_id)
        if info is None:
            return  # unknown task: nothing to update (API rejects these with 404)
        if info.labeled_success is None:
            for model_id in info.model_ids:
                self.stats.record_outcome(info.task_type, model_id, success)
        else:
            for model_id in info.model_ids:
                self.stats.adjust_outcome(
                    info.task_type, model_id, info.labeled_success, success
                )
        info.labeled_success = success
        if self.writer is not None:
            self.writer.enqueue(
                OutcomeEvent(task_id=task_id, ts=ts, success=success, score=score, source=source)
            )

    def reset(self) -> None:
        self.task_arms.clear()

    # -------------------------------------------------------------- restart
    async def rebuild_from_db(self, db) -> int:
        """Rebuild task->arm map and arm stats from persisted outcomes (spec 9)."""
        rows = await db.get_labeled_requests()
        by_task: dict[str, dict] = {}
        for row in rows:
            entry = by_task.setdefault(
                row["task_id"], {"task_type": row["task_type"], "models": [], "outcomes": []}
            )
            if row["chosen_model"] and row["chosen_model"] not in entry["models"]:
                entry["models"].append(row["chosen_model"])
            entry["outcomes"].append((row["success"], row["score"], row["source"], row["ts"]))
        for task_id, entry in by_task.items():
            self.note_task(entry["task_type"], task_id, entry["models"][0])
            for model_id in entry["models"][1:]:
                self.note_task(entry["task_type"], task_id, model_id)
            success, score, source, ts = entry["outcomes"][0]
            info = self.task_arms[task_id]
            for model_id in info.model_ids:
                self.stats.record_outcome(entry["task_type"], model_id, bool(success))
            info.labeled_success = bool(success)
        return len(by_task)
