# DietGate

> A fast LLM gateway + self-optimizing router that measures **cost per successful task** - and learns to make it lower.

**Measured on this machine** (Windows 10, Intel 12 logical cores, Python 3.12.0): the gateway's **own per-request processing** overhead is **p50 0.14 ms / p99 0.23 ms** non-stream and **p50 0.43 ms / p99 0.75 ms** streaming (spec section 5 definition, zero-latency mocks, concurrency 1, 1200+ measured requests per mode after warmup). **End to end, a client sees more**: the external cross-check puts the gateway-vs-bare-handler difference at **~0.37 ms p50 over loopback** on this machine (+0.66 ms on a 1-core Linux sandbox) - that number includes one extra HTTP hop plus the ASGI server, so it is the honest client-visible figure; the internal ~0.1 ms is what the routing logic itself costs. Full tables with hardware: [`bench/results/latest.md`](bench/results/latest.md).

DietGate is an OpenAI-compatible gateway where every request is priced, every task outcome is labeled, and a cost-aware Thompson-sampling router sends each task type to the cheapest model that still meets its quality target. The dashboard shows the full loop live: `request -> execution -> cost -> latency -> outcome -> learn -> optimize`.

**Demo story (fresh-database 2000-task simulated runs; results vary from run to run because mock outcomes and the shadow evaluator are random):** success 94.7-95.7%, routed-traffic savings **about 80-85%** vs baseline over the whole run, net of exploration cost; routing converges to `mock-small` for extraction/classification, `mock-medium` for summarization and mostly `mock-large` for reasoning/code; cost per successful task falls **about 40-56%** from the first 200 to the last 200 tasks (three fresh runs: 39%, 51%, 56%; an earlier pre-D33 run: 55%).

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
cp .env.example .env
# edit .env and set at least:
#   DIETGATE_ADMIN_KEY=<random string, 16+ chars>   (dashboard read key; required)
# optional: DIETGATE_CONTROL_KEY=<random>  enables the simulator/policy/reset buttons
docker compose up --build
```

Then open [http://127.0.0.1:8080/](http://127.0.0.1:8080/), enter the admin key in the dashboard header, and give the auto-started simulator about 80 seconds: it runs the 2000-task `default` scenario (set `DIETGATE_AUTO_SIM_SCENARIO=fast` for a 300-task run). Data lives in the `dietgate-data` Docker volume and **survives restarts, including what the router has learned** (arm statistics are rebuilt from the database on startup). A second `docker compose up` therefore starts warm: savings are higher (about 93% in one such run) and the learning curve is flat, because the router no longer explores. For a clean learning curve (screen recordings, benchmarks) start fresh with `docker compose down -v` first. The numbers in this README are from fresh databases. (Volumes created by versions before task ids were made unique per run contain mixed-up statistics; always reset them with `down -v` after upgrading.)

The container port is bound to `127.0.0.1` by default; see "Deploying a public, mock-only demo" below for internet-facing use.

## What the demo proves, with numbers

All numbers below were measured on this machine and come from the database or the benchmark harness - never hard-coded:

| Claim | Measurement |
|---|---|
| Cost awareness | every request row carries prompt/completion tokens, `cost_usd` and `baseline_cost_usd` |
| Learning | 2000-task sim: `extract`/`classify` traffic -> `mock-small` (about 85-95%), `summarize` -> `mock-medium` (about 70%), `reason`/`code` -> mostly `mock-large`; success 94.7-95.7%; cost per successful task about -40 to -56% (first 200 vs last 200) |
| Savings | routed (`model: auto`) traffic (the sim's whole 2000 tasks): spend about $0.29-0.34 vs baseline $1.72 (**about 80-83%** over three fresh post-D33 runs: 83.2%, 81.7%, 80.2%; 84.6% before exploration pruning); exploration spend shown separately, savings reported net of it |
| Low overhead | gateway's own processing: p50 0.14 ms / p99 0.23 ms (non-stream), p50 0.43 ms / p99 0.75 ms (stream); client-visible end-to-end delta over loopback: ~0.37 ms p50 (includes one extra HTTP hop) - [`bench/results/latest.md`](bench/results/latest.md) |

## Known rough edges of the demo run (so you don't have to ask)

- **Low-traffic task types sit close to their 90% quality target.** `reason` (127 tasks per run) and `code` (76 tasks) see few labeled outcomes per run, so exploration of weaker models still costs some quality. Since exploration only targets arms the data has not yet ruled out (D33), fresh runs landed at `reason` 91.3-92.1% and `code` 86.8-93.4% (two of three above 90%); across 40 simulated seeds `reason` averaged 92.6% (90%+ in 33 of 40 runs) and `code` 91.1% (90%+ in 27 of 40). Individual runs land on either side of the target; a longer run or a higher traffic share makes them converge like the others. The price of the stricter exploration is slightly lower savings (about 2-4 points in these runs).
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

## Deploying a public, mock-only demo

For a link you can send to someone (no real provider keys on the server):

```bash
# .env on the server
DIETGATE_PUBLIC=true                         # refuse default/short keys at startup
DIETGATE_ADMIN_KEY=<random, 16+ chars>
DIETGATE_API_KEYS=<random, 16+ chars>        # comma-separated; replaces the local demo key
DIETGATE_PUBLIC_DASHBOARD=true               # anyone can WATCH the dashboard (read-only, no key)
DIETGATE_DISABLE_REAL_PROVIDERS=true
# leave DIETGATE_CONTROL_KEY empty: nobody can reset or stop the simulator
docker compose up -d --build
```

Put a TLS reverse proxy (Caddy, nginx) in front of `127.0.0.1:8080`. For streaming (SSE) make sure the proxy does not buffer responses (`flush_interval -1` in Caddy, `proxy_buffering off` in nginx). This is a long-running asyncio service: it needs a VPS or any host that can run Docker, not a classic shared-hosting plan.

Safety rails built in: with `DIETGATE_PUBLIC=true` (or `demo_mode: false`) the app **refuses to start** with the shipped default keys or keys shorter than 16 characters; `DIETGATE_PUBLIC_DASHBOARD` is refused if any real provider or stored prompts are configured; control endpoints always need their own key.

## Trying a real provider (bring your own key)

The demo numbers above come from the mock provider. To check that the adapters work against a real model:

1. Copy an entry from [`config/models.live.example.yaml`](config/models.live.example.yaml) into `config/models.yaml`; fill in the exact model id and the **prices from the provider's pricing page** (DietGate refuses to start a real model without explicit prices).
2. Add the model to the `candidates` of the task types you want it to serve in `config/tasks.yaml`.
3. `export ANTHROPIC_API_KEY=...` (or `OPENAI_API_KEY=...`, optionally `OPENAI_BASE_URL` for vLLM/OpenRouter), start the gateway, and run
   `python scripts/live_smoke.py --model <model-id> --api-key <gateway key> --admin-key <admin key>`
   It sends a few tiny requests (non-stream and stream), checks the text, token counts and cost, and prints the gateway overhead. Expect a cost of a few cents at most.

### Example: GLM-5.3-Flash on Novita

Any OpenAI-compatible provider can be added per model (`base_url`, `api_key_env`, `upstream_model`, optional `extra_body`), so several providers coexist. The worked Novita entry is at the end of [`config/models.live.example.yaml`](config/models.live.example.yaml):

```bash
# 1. copy that entry into config/models.yaml, then (Docker) rebuild: docker compose up --build
# 2. put the key in .env (never commit it):   NOVITA_API_KEY=...
# 3. smoke-test it, pinned to that model:
python scripts/live_smoke.py --model novita-glm-5.3-flash --api-key <gateway key> --admin-key <admin key>
```

Things to know about this particular model: it always reasons (the vendor documents thinking as non-disableable), thinking tokens are billed as output and consume `max_tokens` (use 1024+ or the visible answer is empty), and time to first token is seconds, not milliseconds. The smoke script fails loudly if the request was silently served by a fallback model. Keep real models out of the simulator's task `candidates` so the 2000-task demo never spends real money.

Keep a daily spend cap in `config/app.yaml` (`daily_spend_cap_usd`) while experimenting, and never run a real-provider instance with `DIETGATE_PUBLIC_DASHBOARD=true`.

## Keys and controls

- `api_keys` in `config/app.yaml` (default `dg-demo-key`) / `DIETGATE_API_KEYS` (comma-separated, replaces the YAML list): data-plane keys; the YAML default is for local development only.
- `admin_key` / `DIETGATE_ADMIN_KEY`: read access to `/admin/*` (dashboard panels). Use a strong random value on any shared deployment.
- `control_key` / `DIETGATE_CONTROL_KEY`: **separate** key for the mutating controls (simulator start/stop, policy switch, reset). It is **empty (disabled) by default** and docker-compose does not set it - a public demo can be watched, not stopped or reset, by people who only know the dashboard key.
- **Startup protection**: with `DIETGATE_PUBLIC=true` (or `demo_mode=false`), the server **refuses to start** while the default credentials (`admin-dev-key` as admin key or `dg-demo-key` among the data-plane keys) are still configured, or while any key is shorter than 16 characters, with an actionable error message. Local development (demo mode, no PUBLIC flag) is unaffected.

## Tests

```bash
python -m pytest     # 133 tests: cost math, cache/singleflight, policies (Thompson convergence
                     # with a seeded RNG), pipeline stream/non-stream, feedback learning,
                     # restart rebuild, admin API shapes + key separation, simulator e2e,
                     # real-provider adapters against faked upstreams, startup key guard
```

## Honest limitations

1. Savings and quality in the demo come from **synthetic traffic and simulated model skill**; they demonstrate the mechanism, not real-world savings.
2. Baseline cost assumes the chosen model's completion length.
3. Exploration and shadow evaluation cost money; the dashboard shows savings net of that cost.
4. Python was chosen for fast iteration; the hot path is isolated for a Go port (`docs/GO_PORT_NOTES.md`).
5. Caching is exact-match only; semantic caching is future work with real correctness risk.
6. The OpenAI-compatible and Anthropic adapters are tested against faked upstreams (wire format, streaming, token accounting). Whether they work against your account is something `scripts/live_smoke.py` checks with your own key; no live-provider numbers are part of the demo results.
7. This is an independent demo built for a job application; it is not TokenDiet's product or code.

## Configuration

| File | Contents |
|---|---|
| `config/models.yaml` | model catalog, prices, mock latency/skill profiles |
| `config/tasks.yaml` | task types: quality target, baseline, candidate models |
| `config/app.yaml` | API keys, rate limits, spend cap, cache, policy, evaluator, writer |
| `.env.example` | `DIETGATE_*` environment overrides (admin/control keys, demo mode, DB path, ...) |
