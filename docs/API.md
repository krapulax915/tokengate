# DietGate HTTP API

Errors use OpenAI-style bodies: `{"error": {"message": "...", "type": "...", "code": "..."}}`.

## Data plane

### POST /v1/chat/completions

OpenAI-compatible. `stream: true` -> SSE (`data: {...}\n\n`, final `data: [DONE]`,
and a `: dg-cost={...}` comment just before `[DONE]` carrying cost data for streams).
`model: "auto"` enables routing.

Optional headers: `Authorization: Bearer <key>`, `X-Task-Id`, `X-Task-Type`,
`X-Policy` (`static|thompson|cascade`), `X-Cache: allow`.

Demo-only request field: `metadata.mock_expected` / `metadata.mock_wrong`
(ground truth for the mock provider and the cascade verifier; stripped before
any real upstream would be called - real upstreams never see `metadata` at all).

Response headers (non-stream): `X-Dg-Request-Id`, `X-Dg-Model`, `X-Dg-Task-Id`,
`X-Dg-Cost-Usd`, `X-Dg-Baseline-Cost-Usd`, `X-Dg-Overhead-Ms`, `X-Dg-Cache`,
`X-Dg-Policy`.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer dg-demo-key" \
  -H "X-Task-Id: my-task-1" -H "X-Task-Type: extract" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Extract the total from: INV-1, 10 EUR"}]}'
```

### GET /v1/models

Lists configured models plus `auto`. Models whose provider has no API key are
excluded at startup.

### POST /v1/feedback

```json
{"task_id": "my-task-1", "success": true, "score": 0.9, "source": "client"}
```

`source` is `client | checker | shadow_judge` (default `client`). Unknown
`task_id` -> 404. A second label for the same `task_id` replaces the first
(idempotent, spec 6.6).

## Control plane (read-only; `X-Admin-Key` header required)

| Method | Path | Returns |
|---|---|---|
| GET | `/admin/summary?window=1h` | requests (routed/pinned split), spend, routed-traffic baseline + savings, cache hit rate, success rate, p50/p99 overhead |
| GET | `/admin/learning_curve?bucket=1m&window=24h` | time series: cost per successful task (actual vs baseline) + explore share |
| GET | `/admin/routing_matrix?window=24h` | per task_type x model: traffic share, success rate, avg cost |
| GET | `/admin/latency?window=1h&stream=0` | percentiles for overhead, TTFT, total; optional `stream=0/1` filter |
| GET | `/admin/requests?limit=50` | recent requests incl. decision reasons |
| GET | `/admin/requests/{id}` | one request (incl. streamed-cost info) |
| GET | `/admin/arms` | live arm stats: posterior mean, 95% CI, cost EMA |
| GET | `/admin/learning_cost?window=24h` | exploration spend + shadow-eval spend |

`window` accepts `30m`, `12h`, `7d`, or plain seconds. `bucket` accepts `30s`,
`5m`, etc.

Savings in `/admin/summary` are computed **only over `model: "auto"` (routed)
traffic** - pinned-model requests count towards spend but not towards savings
(see `docs/METHODOLOGY.md`).

## DEMO_MODE controls (POST; separate `X-Control-Key` required)

Mutating controls are protected by a **separate control key**, not the
read-only admin key. They are enabled only when BOTH `demo_mode: true` AND a
non-empty `control_key` (or `DIETGATE_CONTROL_KEY`) are configured; the
default is **disabled**.

| Method | Path | Body | Effect |
|---|---|---|---|
| GET | `/admin/sim/status` | - (admin key) | simulator progress |
| POST | `/admin/sim/start` | `{"scenario": "fast", "policy": "thompson"}` | start in-process simulator |
| POST | `/admin/sim/stop` | - | stop the simulator |
| POST | `/admin/policy` | `{"policy": "static"}` | switch default policy |
| POST | `/admin/reset` | - | wipe demo data + in-memory stats |

## Health

| Method | Path | Returns |
|---|---|---|
| GET | `/healthz` | `{"status":"ok"}` (always) |
| GET | `/readyz` | `{"status":"ready"}` (after startup) |
