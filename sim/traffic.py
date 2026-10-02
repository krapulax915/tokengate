"""Traffic simulator: tasks through the gateway, checker verdicts, feedback.

`python -m sim.traffic --scenario default --url http://127.0.0.1:8080 [--ab]`
"""
from __future__ import annotations

import argparse
import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import httpx
import orjson

from sim.scenarios import load_scenarios
from sim.tasks import SimTask, generate_mix


@dataclass
class TypeStat:
    n: int = 0
    successes: int = 0
    cost_sum: float = 0.0
    baseline_sum: float = 0.0
    models: dict = field(default_factory=dict)


@dataclass
class ScenarioResult:
    name: str
    policy: str
    total: int = 0
    successes: int = 0
    cost_sum: float = 0.0
    baseline_sum: float = 0.0
    per_type: dict = field(default_factory=dict)
    cost_per_success_curve: list[float] = field(default_factory=list)

    def finish(self) -> None:
        self.success_rate = self.successes / self.total if self.total else 0.0


def _print_summary(res: ScenarioResult) -> None:
    print(f"\n=== scenario '{res.name}' (policy {res.policy}) ===")
    spend = res.cost_sum
    baseline = res.baseline_sum
    savings = (1 - spend / baseline) * 100 if baseline > 0 else 0.0
    success_rate = res.successes / res.total * 100 if res.total else 0.0
    print(
        f"tasks {res.total} | success {success_rate:.1f}% | spend ${spend:.6f} "
        f"| baseline ${baseline:.6f} | savings {savings:.1f}%"
    )
    for ttype, st in sorted(res.per_type.items()):
        share = ", ".join(f"{m} {n / st.n * 100:.0f}%" for m, n in sorted(st.models.items(), key=lambda kv: -kv[1]))
        avg = st.cost_sum / st.n if st.n else 0
        base_avg = st.baseline_sum / st.n if st.n else 0
        print(
            f"  {ttype:9s} n={st.n:5d} success={st.successes / st.n * 100:5.1f}% "
            f"cost/task ${avg:.6f} (baseline ${base_avg:.6f}) | {share}"
        )
    curve = res.cost_per_success_curve
    if len(curve) >= 400:
        head = sum(curve[:200]) / 200
        tail = sum(curve[-200:]) / 200
        drop = (1 - tail / head) * 100 if head > 0 else 0.0
        print(
            f"cost per successful task: first 200 avg ${head:.6f} -> last 200 avg ${tail:.6f} "
            f"({drop:.1f}% lower)"
        )


async def _run_task(
    client: httpx.AsyncClient, task: SimTask, base_url: str, api_key: str, policy: str | None
) -> tuple[bool, float, float, str, bool]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Task-Id": task.task_id,
        "X-Task-Type": task.task_type,
    }
    if policy:
        headers["X-Policy"] = policy
    body = {
        "model": "auto",
        "messages": task.messages,
        "temperature": 1.0,
        "metadata": task.metadata,
    }
    resp = await _post_with_backoff(client, f"{base_url}/v1/chat/completions", body, headers)
    if resp is None or resp.status_code != 200:
        return False, 0.0, 0.0, "error", False
    payload = resp.json()
    answer = payload["choices"][0]["message"]["content"]
    model = payload.get("model", "?")
    cost = float(resp.headers.get("X-Dg-Cost-Usd", "0") or 0)
    baseline = float(resp.headers.get("X-Dg-Baseline-Cost-Usd", "0") or 0)
    success = task.checker()(answer)
    feedback = {"task_id": task.task_id, "success": success, "source": "checker"}
    await _post_with_backoff(client, f"{base_url}/v1/feedback", feedback, headers)
    return success, cost, baseline, model, False


async def _post_with_backoff(
    client: httpx.AsyncClient, url: str, payload: dict, headers: dict
) -> httpx.Response | None:
    """429s are retried with exponential backoff (0.25s, 0.5s, 1s)."""
    merged = {**headers, "Content-Type": "application/json"}
    delay = 0.25
    for attempt in range(4):
        try:
            resp = await client.post(
                url, content=orjson.dumps(payload), headers=merged
            )
        except httpx.HTTPError:
            return None
        if resp.status_code == 429 and attempt < 3:
            await asyncio.sleep(delay)
            delay *= 2
            continue
        return resp
    return resp


async def run_scenario(
    scenario: dict,
    *,
    name: str = "default",
    base_url: str = "http://127.0.0.1:8080",
    api_key: str = "dg-demo-key",
    policy: str | None = None,
    seed: int = 7,
    on_result: Callable[[str, bool, float], None] | None = None,
    stop_event: asyncio.Event | None = None,
    quiet: bool = False,
    client: httpx.AsyncClient | None = None,
) -> ScenarioResult:
    count = int(scenario.get("count", 100))
    rate = float(scenario.get("rate", 20.0))
    concurrency = int(scenario.get("concurrency", 10))
    mix = scenario.get("mix") or {}
    rng = random.Random(seed)
    tasks = generate_mix(mix, rng, count, prefix=f"{name}")
    result = ScenarioResult(name=name, policy=policy or "gateway-default")

    limits = httpx.Limits(max_connections=concurrency + 10, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(90.0)
    owns_client = client is None
    client = client or httpx.AsyncClient(limits=limits, timeout=timeout)
    try:
        semaphore = asyncio.Semaphore(concurrency)
        stop = stop_event or asyncio.Event()
        launched = 0
        finished = 0

        async def worker(task: SimTask) -> None:
            nonlocal finished
            async with semaphore:
                if stop.is_set():
                    return
                try:
                    ok, cost, baseline, model, _ = await _run_task(client, task, base_url, api_key, policy)
                except Exception:
                    ok, cost, baseline, model = False, 0.0, 0.0, "error"
                st = result.per_type.setdefault(task.task_type, TypeStat())
                st.n += 1
                st.successes += 1 if ok else 0
                st.cost_sum += cost
                st.baseline_sum += baseline
                st.models[model] = st.models.get(model, 0) + 1
                result.total += 1
                result.successes += 1 if ok else 0
                result.cost_sum += cost
                result.baseline_sum += baseline
                successful_so_far = max(1, result.successes)
                result.cost_per_success_curve.append(result.cost_sum / successful_so_far)
                finished += 1
                if on_result is not None:
                    on_result(task.task_type, ok, cost)

        interval = 1.0 / rate if rate > 0 else 0.0
        pending: set[asyncio.Task] = set()
        for task in tasks:
            if stop.is_set():
                break
            t = asyncio.create_task(worker(task))
            pending.add(t)
            launched += 1
            if launched % 200 == 0 and not quiet:
                print(f"  launched {launched}/{count} ...")
            if interval:
                await asyncio.sleep(interval)
            if len(pending) >= concurrency * 3:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    finally:
        if owns_client:
            await client.aclose()
    _print_summary(result)
    return result


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="DietGate traffic simulator")
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--key", default="dg-demo-key")
    parser.add_argument("--policy", default=None, choices=["static", "thompson", "cascade"])
    parser.add_argument("--ab", action="store_true", help="A/B: static vs thompson")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    scenarios = load_scenarios()
    scenario = scenarios.get(args.scenario) or scenarios.get("default")
    # health check
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{args.url}/healthz")
        resp.raise_for_status()

    if args.ab:
        print("A/B: static (baseline) vs thompson on the same task mix")
        static_res = await run_scenario(
            scenario, name=f"{args.scenario}-static", base_url=args.url, api_key=args.key,
            policy="static", seed=args.seed,
        )
        thompson_res = await run_scenario(
            scenario, name=f"{args.scenario}-thompson", base_url=args.url, api_key=args.key,
            policy="thompson", seed=args.seed,
        )
        spend_s = static_res.cost_sum
        spend_t = thompson_res.cost_sum
        base = static_res.baseline_sum
        print("\n=== A/B result ===")
        print(f"static  : spend ${spend_s:.6f} | success {static_res.successes / max(1, static_res.total) * 100:.1f}%")
        print(f"thompson: spend ${spend_t:.6f} | success {thompson_res.successes / max(1, thompson_res.total) * 100:.1f}%")
        if spend_s > 0:
            print(f"thompson saves {(1 - spend_t / spend_s) * 100:.1f}% vs static; baseline ${base:.6f}")
    else:
        await run_scenario(
            scenario, name=args.scenario, base_url=args.url, api_key=args.key,
            policy=args.policy, seed=args.seed,
        )


if __name__ == "__main__":
    asyncio.run(main_async())
