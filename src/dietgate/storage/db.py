"""SQLite (WAL) access. Off the hot path: only the writer and admin queries use this.

No SQL crosses this module's boundary: every query is a string literal inside a
named method, and all dynamic values are bound via `?` placeholders.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import aiosqlite

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


class Database:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def connected(self) -> bool:
        return self._conn is not None

    async def _rows(self, cursor: aiosqlite.Cursor) -> list[dict[str, Any]]:
        return [dict(row) for row in await cursor.fetchall()]

    def _require_conn(self) -> aiosqlite.Connection:
        assert self._conn is not None, "database not connected"
        return self._conn

    async def commit(self) -> None:
        await self._require_conn().commit()

    # ------------------------------------------------------------- inserts --
    # `rows` are pre-ordered tuples produced by storage.writer events; every
    # value travels through a `?` placeholder.

    async def insert_requests(self, rows: Sequence[tuple]) -> None:
        conn = self._require_conn()
        await conn.executemany("INSERT OR REPLACE INTO requests (id, ts, api_key_id, task_id, task_type, requested_model, chosen_model, provider, policy, explore, decision_reason, cache_hit, shortcut, stream, status, prompt_tokens, completion_tokens, cost_usd, baseline_cost_usd, ttft_ms, total_ms, overhead_ms, attempts, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

    async def insert_outcomes(self, rows: Sequence[tuple]) -> None:
        conn = self._require_conn()
        await conn.executemany("INSERT OR REPLACE INTO outcomes (task_id, ts, success, score, source) VALUES (?, ?, ?, ?, ?)", rows)

    async def insert_shadow_evals(self, rows: Sequence[tuple]) -> None:
        conn = self._require_conn()
        await conn.executemany("INSERT OR REPLACE INTO shadow_evals (id, request_id, ts, baseline_model, judge_model, verdict, eval_cost_usd) VALUES (?, ?, ?, ?, ?, ?, ?)", rows)

    async def insert_arm_snapshots(self, rows: Sequence[tuple]) -> None:
        conn = self._require_conn()
        await conn.executemany("INSERT OR REPLACE INTO arm_snapshots (ts, task_type, model_id, n_labeled, successes, cost_sum_usd) VALUES (?, ?, ?, ?, ?, ?)", rows)

    # ------------------------------------------------ fixed read queries ----

    async def count_requests(self) -> int:
        async with self._require_conn().execute("SELECT COUNT(*) AS n FROM requests") as cur:
            return int((await self._rows(cur))[0]["n"])

    async def count_outcomes(self) -> int:
        async with self._require_conn().execute("SELECT COUNT(*) AS n FROM outcomes") as cur:
            return int((await self._rows(cur))[0]["n"])

    async def get_requests_by_task_id(self, task_id: str) -> list[dict[str, Any]]:
        async with self._require_conn().execute(
            "SELECT * FROM requests WHERE task_id = ?", (task_id,)
        ) as cur:
            return await self._rows(cur)

    async def get_stream_requests(self) -> list[dict[str, Any]]:
        async with self._require_conn().execute("SELECT * FROM requests WHERE stream = 1") as cur:
            return await self._rows(cur)

    async def get_labeled_requests(self, limit: int = 100000) -> list[dict[str, Any]]:
        """Outcomes joined with their requests, for the learner's restart rebuild."""
        async with self._require_conn().execute(
            "SELECT r.task_id AS task_id, r.task_type AS task_type, r.chosen_model AS chosen_model,"
            " o.success AS success, o.score AS score, o.source AS source, o.ts AS ts"
            " FROM requests r JOIN outcomes o ON o.task_id = r.task_id LIMIT ?",
            (int(limit),),
        ) as cur:
            return await self._rows(cur)

    # ----------------------------------------------------- admin API reads --

    async def get_request_metrics_since(
        self, ts: float, stream: int | None = None
    ) -> list[dict[str, Any]]:
        if stream is None:
            sql = "SELECT requested_model, status, cache_hit, explore, shortcut, attempts, cost_usd, baseline_cost_usd, overhead_ms, ttft_ms, total_ms FROM requests WHERE ts >= ?"
            params: tuple = (float(ts),)
        elif stream == 1:
            sql = "SELECT requested_model, status, cache_hit, explore, shortcut, attempts, cost_usd, baseline_cost_usd, overhead_ms, ttft_ms, total_ms FROM requests WHERE ts >= ? AND stream = 1"
            params = (float(ts),)
        else:
            sql = "SELECT requested_model, status, cache_hit, explore, shortcut, attempts, cost_usd, baseline_cost_usd, overhead_ms, ttft_ms, total_ms FROM requests WHERE ts >= ? AND stream = 0"
            params = (float(ts),)
        async with self._require_conn().execute(sql, params) as cur:
            return await self._rows(cur)

    async def get_outcome_stats_since(self, ts: float) -> dict[str, Any] | None:
        async with self._require_conn().execute(
            "SELECT COUNT(*) AS n, SUM(success) AS successes FROM outcomes WHERE ts >= ?",
            (float(ts),),
        ) as cur:
            rows = await self._rows(cur)
        return rows[0] if rows else None

    async def get_labeled_requests_since(self, ts: float) -> list[dict[str, Any]]:
        async with self._require_conn().execute(
            "SELECT r.task_id AS task_id, r.task_type AS task_type, r.chosen_model AS chosen_model,"
            " r.explore AS explore, r.cost_usd AS cost_usd, r.baseline_cost_usd AS baseline,"
            " o.success AS success, o.ts AS ts"
            " FROM requests r JOIN outcomes o ON o.task_id = r.task_id WHERE o.ts >= ?",
            (float(ts),),
        ) as cur:
            return await self._rows(cur)

    async def get_routing_rows_since(self, ts: float) -> list[dict[str, Any]]:
        async with self._require_conn().execute(
            "SELECT r.task_type AS task_type, r.chosen_model AS model, COUNT(*) AS n,"
            " SUM(r.cost_usd) AS cost, SUM(r.baseline_cost_usd) AS baseline,"
            " SUM(o.success) AS successes"
            " FROM requests r LEFT JOIN outcomes o ON o.task_id = r.task_id"
            " WHERE r.ts >= ? AND r.chosen_model IS NOT NULL"
            " GROUP BY r.task_type, r.chosen_model",
            (float(ts),),
        ) as cur:
            return await self._rows(cur)

    async def get_recent_requests(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self._require_conn().execute(
            "SELECT * FROM requests ORDER BY ts DESC LIMIT ?", (int(limit),)
        ) as cur:
            return await self._rows(cur)

    async def get_request_by_id(self, request_id: str) -> dict[str, Any] | None:
        async with self._require_conn().execute(
            "SELECT * FROM requests WHERE id = ?", (request_id,)
        ) as cur:
            rows = await self._rows(cur)
        return rows[0] if rows else None

    async def get_explore_spend_since(self, ts: float) -> float:
        async with self._require_conn().execute(
            "SELECT SUM(cost_usd) AS s FROM requests WHERE explore = 1 AND ts >= ?",
            (float(ts),),
        ) as cur:
            rows = await self._rows(cur)
        return float(rows[0]["s"] or 0.0) if rows else 0.0

    async def get_shadow_spend_since(self, ts: float) -> dict[str, Any] | None:
        async with self._require_conn().execute(
            "SELECT COUNT(*) AS n, SUM(eval_cost_usd) AS s FROM shadow_evals WHERE ts >= ?",
            (float(ts),),
        ) as cur:
            rows = await self._rows(cur)
        return rows[0] if rows else None

    async def reset_all(self) -> None:
        conn = self._require_conn()
        await conn.execute("DELETE FROM requests")
        await conn.execute("DELETE FROM outcomes")
        await conn.execute("DELETE FROM shadow_evals")
        await conn.execute("DELETE FROM arm_snapshots")
        await conn.commit()
