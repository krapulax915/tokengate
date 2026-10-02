"""Pre-fill the database with a realistic learning curve for instant demos.

Writes directly to SQLite (requests + outcomes): ~2 hours of synthetic
history in which the router visibly converges from baseline to the cheapest
eligible model per task type. No gateway required.
"""
from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys_path = str(REPO / "src")
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from dietgate.core.config import load_models, load_tasks  # noqa: E402
from dietgate.core.cost import request_cost_usd  # noqa: E402
from dietgate.settings import load_settings  # noqa: E402
from dietgate.storage.db import Database  # noqa: E402
from dietgate.storage.writer import OutcomeEvent, RequestEvent  # noqa: E402
from sim.tasks import generate_mix  # noqa: E402

HISTORY_S = 2 * 3600
TASKS_PER_TYPE = 500
MIX = {"extract": 0.6, "classify": 0.2, "summarize": 0.1, "reason": 0.06, "code": 0.04}


def _skill(task_type: str, model_id: str, catalog) -> float:
    return catalog[model_id].mock.skill_for(task_type)


async def seed() -> None:
    settings = load_settings()
    catalog = load_models(settings.config_dir, disable_real_providers=True)
    tasks_cfg = load_tasks(settings.config_dir)
    db = Database(settings.db_path)
    await db.connect()

    rng = random.Random(11)
    now = time.time()
    start = now - HISTORY_S
    rows_req: list = []
    rows_out: list = []
    n = 0
    for task in generate_mix(MIX, rng, TASKS_PER_TYPE * len(MIX), prefix="seed"):
        cfg = tasks_cfg[task.task_type]
        progress = (n % TASKS_PER_TYPE) / TASKS_PER_TYPE  # 0..1 learning progress
        # early: mostly baseline; late: converged; always ~5% exploration
        candidates = cfg.candidates
        if rng.random() < 0.05:
            chosen = rng.choice(candidates)
            explore = True
        elif rng.random() < progress ** 1.5:
            # converged pick: cheapest that is probably good enough
            good = [m for m in candidates if _skill(task.task_type, m, catalog) >= cfg.min_success + 0.04]
            chosen = good[0] if good else candidates[-1]
            explore = False
        else:
            chosen = rng.choice(candidates[-1:])  # baseline fallback
            explore = False
        success = rng.random() < _skill(task.task_type, chosen, catalog)
        answer = task.metadata["mock_expected"] if success else task.metadata["mock_wrong"][0]
        pt, ct = 120, len(answer) // 4
        cost = request_cost_usd(catalog, chosen, pt, ct)
        baseline = request_cost_usd(catalog, cfg.baseline, pt, ct)
        ts = start + (n / (TASKS_PER_TYPE * len(MIX))) * HISTORY_S
        rows_req.append(
            RequestEvent(
                id=f"seed-{n:06d}", ts=ts, api_key_id="key-seeded", task_id=task.task_id,
                task_type=task.task_type, requested_model="auto", chosen_model=chosen,
                provider="mock", policy="thompson", explore=explore,
                decision_reason="seeded history", cache_hit=False, shortcut=None,
                stream=False, status=200, prompt_tokens=pt, completion_tokens=ct,
                cost_usd=cost, baseline_cost_usd=baseline, ttft_ms=None, total_ms=None,
                overhead_ms=0.3 + rng.random() * 0.5, attempts=1, error=None,
            ).row()
        )
        rows_out.append(
            OutcomeEvent(task_id=task.task_id, ts=ts + 0.05, success=success, score=None,
                         source="checker").row()
        )
        n += 1
        if len(rows_req) >= 500:
            await db.insert_requests(rows_req)
            await db.insert_outcomes(rows_out)
            rows_req, rows_out = [], []
    if rows_req:
        await db.insert_requests(rows_req)
        await db.insert_outcomes(rows_out)
    await db.commit()
    await db.close()
    print(f"seeded {n} tasks with outcomes into {settings.db_path}")


if __name__ == "__main__":
    asyncio.run(seed())
