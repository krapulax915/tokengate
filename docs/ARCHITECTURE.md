# DietGate Architecture

```
 client / agent
      |  OpenAI-compatible request (+ optional X-Task-Id / X-Task-Type headers)
      v
+---------------------------- DietGate (single process, async) ----------------------------+
|                                                                                           |
|  HOT PATH (no disk, no blocking)                                                          |
|  auth -> classify -> cache lookup -> shortcut? -> route (policy) -> provider call ->      |
|  stream/forward -> cost calc -> enqueue event -> respond                                  |
|                                                                                           |
|  OFF-HOT-PATH (background tasks)                                                          |
|  writer (batched SQLite inserts)  |  learner (feedback -> stats)  |  evaluator (shadow)  |
|                                                                                           |
|  In-memory: arm stats (task_type x model), cache, rate-limit counters, config             |
+--------------------------------------------------------------------------------------------+
```

## Modules

| Module | Responsibility |
|---|---|
| `main.py` | app factory, lifespan (start/stop background components), router mounting |
| `settings.py` | `DIETGATE_*` environment settings (pydantic-settings) |
| `core/config.py` | YAML config loading/validation (`config/*.yaml`) |
| `core/runtime.py` | component wiring; one `Runtime` object per process |
| `core/context.py` | `RequestContext` dataclass: lifecycle timestamps, decisions |
| `core/pipeline.py` | hot-path orchestration in spec section 5 order |
| `core/auth.py` | API key check, per-key token bucket, daily spend cap (in-memory) |
| `core/classifier.py` | task type: `X-Task-Type` header wins, else keyword heuristic |
| `core/cache.py` | model-agnostic exact-match cache: LRU + TTL + singleflight |
| `core/shortcuts.py` | local rules that answer with no model call (cost 0) |
| `core/cost.py` | len/4 token estimates, price math, baseline pricing |
| `core/router.py` | picks the policy (config default or `X-Policy` header) |
| `core/policies/*` | `static`, `thompson` (cost-aware Beta sampling), `cascade` |
| `core/stats.py` | in-memory arm statistics per (task_type, model) |
| `core/learner.py` | background: outcome labels -> arm stats; restart rebuild |
| `core/evaluator.py` | background: sampled shadow evaluation of unlabeled answers |
| `providers/*` | `mock` (deterministic), `openai_compat`, `anthropic`, registry |
| `storage/db.py` | SQLite (WAL) access; every SQL is a fixed literal + `?` params |
| `storage/writer.py` | batched write-behind queue (200 rows / 250 ms), drops when full |
| `api/*` | data plane (`/v1/*`) and control plane (`/admin/*`) |
| `dashboard/` | static HTML + vanilla JS + Chart.js, served by the gateway |
| `sim/*` | task generators with ground truth, checkers, traffic generator |
| `bench/*` | overhead benchmark (gateway vs bare mock handler) |

## Request lifecycle (spec section 5)

1. `t_recv` - parse with orjson, validate `model` + `messages`
2. auth: bearer must match `app.yaml`; token bucket; daily spend cap -> 401/429
3. task type: header or heuristic; `task_id` from header or generated
4. model: `auto` -> router decides; pinned -> no routing (still measured)
5. cache (non-stream, `temperature == 0` or `X-Cache: allow`): sha256 of
   normalized (messages, tools, response_format, max_tokens) - **model-agnostic**;
   concurrent identical requests share one upstream call (singleflight)
6. shortcuts: first matching local rule wins, cost 0
7. route: policy returns `Decision(model, policy, explore, reason, fallbacks)`
8. provider call with fallbacks (max 2 retries on upstream error/timeout);
   streaming forwards chunks as they arrive and records first-chunk timestamps
9. usage (provider-reported or estimate) -> `cost_usd` and `baseline_cost_usd`
10. event -> writer queue (`put_nowait`, drop + counter when full, never blocks)
11. respond with `X-Dg-*` headers; streams get a `: dg-cost={...}` SSE trailer

### Overhead definition

```
overhead_ms = (t_upstream_request_sent - t_recv)       # pre-processing
            + (t_first_chunk_out - t_first_chunk_in)   # first-chunk forwarding
            + (t_done_out        - t_done_in)          # tail forwarding
```

Upstream wait is excluded by construction. Paths that never touch an upstream
(cache hit, shortcut, error) are pure overhead end to end.

## The learning loop

```
request -> execution -> cost -> latency -> outcome label -> learner -> arm stats -> router
```

- Labels come from `POST /v1/feedback` (client or simulator checker) or the
  sampled shadow evaluator; a second label for the same `task_id` replaces the first.
- Arm stats: `Beta(successes+1, failures+1)` posterior, cost EMA, latency EMA.
- Thompson policy: sample each arm's posterior; the cheapest arm whose *sample*
  clears the quality target wins; none clears -> best sample; bounded warm-up
  exploration boost protects cheap arms from unlucky early data (DECISIONS D18).
- On restart the learner rebuilds stats from `outcomes JOIN requests`.
