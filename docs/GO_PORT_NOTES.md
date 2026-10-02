# Moving the DietGate hot path to Go

Python was chosen for iteration speed on a 2-day demo. The hot path is already
isolated so the data plane can be ported to Go without touching the learning
loop.

## What moves to Go (the data plane)

| Component | Notes |
|---|---|
| auth + rate limit + spend cap | token bucket and counters are trivial in Go; `sync/atomic` |
| cache lookup + singleflight | `golang.org/x/sync/singleflight` + LRU/TTL map |
| route decision | pure function over a stats snapshot; no I/O |
| upstream proxy + SSE forwarding | `net/http` reverse proxy, chunked copying, first-byte timestamps |
| cost math | integer arithmetic on prices scaled to micro-dollars |
| event emission | non-blocking send to a buffered channel; drop + counter when full |

Target shape: a single Go binary speaking OpenAI-compatible HTTP, holding no
state of its own except an atomic pointer to the latest stats snapshot.

## What stays in Python (the control plane)

- learner (feedback -> arm statistics, restart rebuild from SQLite)
- evaluator (shadow judging, real LLM judge calls)
- simulator (task generators, checkers, A/B traffic)
- dashboard API + dashboard
- offline analysis, seeding, benchmarks

## Interface between the two

1. **Python -> Go: arm stats snapshot.** Every few seconds the learner writes a
   JSON snapshot (`task_type`, `model_id`, `successes`, `failures`, `cost_ema`,
   `candidates`, `min_success`) to a file or Redis key; Go loads it atomically
   (atomic pointer swap, no locks on the read path). A missing/stale snapshot
   falls back to the static baseline policy - fail safe, never fail open.
2. **Go -> Python: request events.** The Go data plane appends request events
   (same columns as the `requests` table, JSON lines) to a log/queue; Python
   tails it into the batched SQLite writer exactly as it consumes its in-process
   queue today. The `dropped_events` counter ships in the same stream.

## Why split it this way

The data plane needs predictable tail latency and high concurrency - Go gives
cheap goroutines, no GIL, and mature HTTP/SSE plumbing. The control plane is
about experimentation speed: bandit policy changes, judge prompts, new checkers
- Python wins there. The seam is deliberately two one-way data flows
(snapshot in, events out), so neither side blocks the other and both can be
tested independently.

## What the Python demo already proves for the port

- The hot path does no disk or DB access (`core/pipeline.py` awaits only the
  upstream call and one shared singleflight future).
- The event path is `put_nowait`-or-drop, never blocking a response.
- The route decision reads only in-memory state and is a pure function of
  (task_type, stats snapshot, catalog, policy).
- Overhead is measured as spec section 5 defines it, so the Go port can be
  benchmarked against the same definition and the same `bench/overhead.py`.
