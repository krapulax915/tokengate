"""In-process simulator runner for the dashboard controls (DEMO_MODE only)."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from sim.scenarios import load_scenarios
from sim.traffic import ScenarioResult, run_scenario


@dataclass
class SimState:
    running: bool = False
    scenario: str = ""
    policy: str = ""
    started_ts: float = 0.0
    done: int = 0
    total: int = 0
    result: ScenarioResult | None = None
    last_error: str | None = None
    progress_log: list = field(default_factory=list)


class SimRunner:
    """Starts/stops a scenario against a running gateway (this process)."""

    def __init__(self, base_url: str, api_key: str = "dg-demo-key") -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.state = SimState()
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None

    def status(self) -> dict:
        s = self.state
        return {
            "running": s.running,
            "scenario": s.scenario,
            "policy": s.policy,
            "done": s.done,
            "total": s.total,
            "elapsed_s": round(time.time() - s.started_ts, 1) if s.running else 0.0,
            "last_error": s.last_error,
            "result": self._result_summary(s.result) if (s.result and not s.running) else None,
            "progress_log": s.progress_log[-12:],
        }

    @staticmethod
    def _result_summary(res: ScenarioResult | None) -> dict | None:
        if res is None:
            return None
        return {
            "tasks": res.total,
            "success_rate": round(res.successes / res.total * 100, 1) if res.total else 0.0,
            "spend_usd": round(res.cost_sum, 6),
            "baseline_usd": round(res.baseline_sum, 6),
        }

    def start(self, scenario: str = "fast", policy: str | None = None) -> dict:
        if self.state.running:
            return {"status": "already-running", **self.status()}
        scenarios = load_scenarios()
        cfg = scenarios.get(scenario)
        if cfg is None:
            return {"status": "unknown-scenario", "known": list(scenarios.keys())}
        self._stop_event = asyncio.Event()
        self.state = SimState(
            running=True,
            scenario=scenario,
            policy=policy or "gateway-default",
            started_ts=time.time(),
            total=int(cfg.get("count", 0)),
        )
        self._task = asyncio.create_task(self._run(cfg, policy))
        return {"status": "started", **self.status()}

    def stop(self) -> dict:
        if self._stop_event is not None:
            self._stop_event.set()
        return {"status": "stopping", **self.status()}

    async def wait_stopped(self) -> None:
        if self._task is not None:
            import contextlib

            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self, cfg: dict, policy: str | None) -> None:
        assert self._stop_event is not None
        state = self.state
        try:
            # The auto-start fires inside the app's startup, before uvicorn accepts connections:
            # without this wait the very first task hits "connection refused" and is counted as a failure.
            await self._wait_until_ready()
            await run_scenario(
                cfg,
                name=f"dashboard-{state.scenario}",
                base_url=self.base_url,
                api_key=self.api_key,
                policy=policy,
                stop_event=self._stop_event,
                quiet=True,
                on_result=lambda tt, ok, cost: self._on_result(tt, ok),
            )
        except Exception as exc:  # keep the dashboard alive, show the error
            state.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            state.running = False

    async def _wait_until_ready(self, timeout_s: float = 20.0) -> None:
        import httpx

        deadline = time.monotonic() + timeout_s
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.monotonic() < deadline:
                if self._stop_event is not None and self._stop_event.is_set():
                    return
                try:
                    if (await client.get(f"{self.base_url}/healthz")).status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.25)

    def _on_result(self, task_type: str, ok: bool) -> None:
        self.state.done += 1
        if self.state.done % 50 == 0:
            self.state.progress_log.append(
                {"ts": round(time.time(), 1), "done": self.state.done, "type": task_type, "ok": ok}
            )
