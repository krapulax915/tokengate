# DietGate

> A fast LLM gateway + self-optimizing router that measures **cost per successful task** - and learns to make it lower.

**Measured on this machine** (Windows 10, Intel 12 logical cores, Python 3.12.0): the gateway's own per-request overhead is **p50 0.14 ms / p99 0.24 ms** non-stream and **p50 0.43 ms / p99 0.68 ms** streaming (spec section 5 definition, zero-latency mocks, concurrency 1, 1200+ measured requests per mode after warmup - full table with hardware in [`bench/results/latest.md`](bench/results/latest.md)).

DietGate is an OpenAI-compatible gateway where every request is priced, every task outcome is labeled, and a cost-aware Thompson-sampling router sends each task type to the cheapest model that still meets its quality target. The dashboard shows the full loop live: `request -> execution -> cost -> latency -> outcome -> learn -> optimize`.

**Demo story (from a real 2000-task simulated run on this machine):** success 95.5%, routed-traffic savings **83.9%** vs baseline; routing converged to `mock-small` for extraction (89% share), `mock-medium` for summarization (67%), `mock-large` for reasoning/code (65-74%); cost per successful task fell **52%** from the first 200 to the last 200 tasks.

---

## Quick start (3 commands)

```bash
pip install -e ".[dev]"          # or: pip install -e .
python -m uvicorn dietgate.main:app --host 127.0.0.1 --port 8080   # start the gateway
python -m sim.traffic --scenario default --url http://127.0.0.1:8080
```

Open the dashboard at [http://127.0.0.1:8080/](http://127.0.0.1:8080/) and watch the learning curve drop. The dashboard needs the admin key for its panels (default `admin-dev-key`, from `config/app.yaml` / `DIETGATE_ADMIN_KEY`); API calls use the demo key `dg-demo-key`.

One-command variant (starts gateway + simulator + opens the browser):

```bash
python scripts/run_demo.py --open
```

Docker (dashboard starts filling immediately, auto-start simulator enabled):

```bash
# create .env next to docker-compose.yml:
#   DIETGATE_ADMIN_KEY=<a strong random key>     (dashboard read key; required)
#   DIETGATE_CONTROL_KEY=...                     (optional: enables sim/policy/reset controls)
docker compose up
```

## What the demo proves, with numbers

All numbers below were measured on this machine and come from the database or the benchmark harness - never hard-coded:

| Claim | Measurement |
|---|---|
| Cost awareness | every request row carries prompt/completion tokens, `cost_usd` and `baseline_cost_usd` |
| Learning | 2000-task sim: `extract` traffic -> `mock-small` 89%, `summarize` -> `mock-medium` 67%, `reason`/`code` -> `mock-large` 65-74%; success 95.5%; cost per successful task -52% (first 200 vs last 200) |
| Savings | routed (`model: auto`) traffic (the sim's whole 2000 tasks): spend $0.277 vs baseline $1.719 (**83.9%**); exploration spend shown separately, savings reported net of it |
| Low overhead | gateway-measured overhead p50 0.14 ms / p99 0.24 ms (non-stream), p50 0.43 ms / p99 0.68 ms (stream) - [`bench/results/latest.md`](bench/results/latest.md) |

## Known rough edges of the demo run (so you don't have to ask)

- **Low-traffic task types had not converged yet** within the 2000-task run: `reason` (127 tasks, 86.6% success) and `code` (76 tasks, 88.2% success) still sent 26-35% of traffic to `mock-medium`, and both sit marginally below their 90% quality target. With the demo's traffic mix (60% extract, 4-6% for these types) they simply see too few labeled outcomes per run; a longer run or a higher share converges them like the others.
- **Savings are meaningful only for routed traffic** (`model: "auto"`). Pinned-model requests already fixed the model, so the dashboard computes savings over routed rows only and labels the baseline card accordingly.
- The `stream` "rps" numbers in the benchmark are for whole SSE responses (all chunks), not for time-to-first-token.
- Run-to-run results vary by a few tenths of a percent: the shadow evaluator races client feedback, so a handful of labels differ between runs.

## Try it by hand

```bash
# routed request (model "auto" -> the router decides)
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer dg-demo-key" \
  -H "X-Task-Id: my-task-1" -H "X-Task-Type: extract" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Extract the total from: INV-1, 10 EUR"}]}'

# pinned model, streamed
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer dg-demo-key" \
  -d '{"model":"mock-small","stream":true,"messages":[{"role":"user","content":"hello"}]}'

# report an outcome -> feeds the learning loop
curl http://127.0.0.1:8080/v1/feedback \
  -H "Content-Type: application/json" -H "Authorization: Bearer dg-demo-key" \
  -d '{"task_id":"my-task-1","success":true,"source":"client"}'

# A/B: static baseline vs thompson on the same task mix
python -m sim.traffic --scenario default --ab --url http://127.0.0.1:8080
```

## How it works

- **Hot path** (no disk, no DB, no blocking): auth -> classify -> cache -> shortcut -> route -> provider call -> cost calc -> event enqueue -> respond. Upstream wait is excluded from the measured overhead by construction.
- **Learning loop** (off the hot path): feedback or shadow-evaluation labels update per-`(task_type, model)` arm statistics through a background learner; the router reads only in-memory stats; stats rebuild from SQLite on restart.
- **Providers**: the deterministic mock provider runs everything with **zero API keys**. OpenAI-compatible upstreams and Anthropic are supported - add entries to `config/models.yaml` with prices from the provider's pricing page; without keys those models are excluded from routing.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | [docs/API.md](docs/API.md) | [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | [docs/GO_PORT_NOTES.md](docs/GO_PORT_NOTES.md) | [docs/DECISIONS.md](docs/DECISIONS.md)

## Benchmark

`python bench/overhead.py` starts a zero-latency-mock gateway, warms it up, drives it at concurrency 1 (non-stream and stream), and reports the **gateway's own `overhead_ms` percentiles** (the spec section 5 definition: pre-processing + forwarding, upstream wait excluded) plus an external client-measured cross-check against a bare mock handler. Results with hardware go to `bench/results/latest.{json,md}`.

There are deliberately **no concurrency >= 50 rows**: subtracting quantiles of two separate single-process asyncio servers measures queueing order, not routing overhead (earlier runs showed negative "overhead" there). Load beyond saturation needs a real load generator with multiple workers (wrk/oha/locust).

## Keys and controls

- Data-plane API keys: `config/app.yaml` `api_keys`, or override via `DIETGATE_API_KEYS` (comma-separated). The shipped demo key is `dg-demo-key`.
- `admin_key` / `DIETGATE_ADMIN_KEY`: read access to `/admin/*` (dashboard panels). Use a strong random value on any shared deployment.
- `control_key` / `DIETGATE_CONTROL_KEY`: **separate** key for the mutating controls (simulator start/stop, policy switch, reset). It is **empty (disabled) by default** and docker-compose does not set it - a public demo can be watched, not stopped or reset, by people who only know the dashboard key.
- **Startup protection**: with `DIETGATE_PUBLIC=true` (or `demo_mode=false`), the server **refuses to start** while the default credentials (`admin-dev-key` as admin key or `dg-demo-key` among the data-plane keys) are still configured, with an actionable error message. Local development (demo mode, no PUBLIC flag) is unaffected.

## Tests

```bash
python -m pytest     # 94 tests: cost math, cache/singleflight, policies (Thompson convergence
                     # with a seeded RNG), pipeline stream/non-stream, feedback learning,
                     # restart rebuild, admin API shapes + key separation, simulator e2e
```

## Honest limitations

1. Savings and quality in the demo come from **synthetic traffic and simulated model skill**; they demonstrate the mechanism, not real-world savings.
2. Baseline cost assumes the chosen model's completion length.
3. Exploration and shadow evaluation cost money; the dashboard shows savings net of that cost.
4. Python was chosen for fast iteration; the hot path is isolated for a Go port (`docs/GO_PORT_NOTES.md`).
5. Caching is exact-match only; semantic caching is future work with real correctness risk.
6. This is an independent demo built for a job application; it is not TokenDiet's product or code.

## Configuration

| File | Contents |
|---|---|
| `config/models.yaml` | model catalog, prices, mock latency/skill profiles |
| `config/tasks.yaml` | task types: quality target, baseline, candidate models |
| `config/app.yaml` | API keys, rate limits, spend cap, cache, policy, evaluator, writer |
| `.env.example` | `DIETGATE_*` environment overrides (admin/control keys, demo mode, DB path, ...) |
