# Methodology: how savings, quality and overhead are measured (and where it bends)

Everything on the dashboard is computed from the SQLite database or the
in-memory arm statistics. Nothing is hard-coded. Still, read this before
trusting any number.

## Cost

```
cost_usd          = prompt_tokens/1e6 * price_in + completion_tokens/1e6 * price_out
baseline_cost_usd = same token counts priced at the task type's baseline model
task_cost_usd     = sum(cost_usd of all requests with the same task_id)
                    + attributed shadow/judge cost
cost_per_success  = sum(task_cost_usd of labeled tasks) / count(successful labeled tasks)
```

Token counts are provider-reported when available; the mock reports a
`len(text)/4` estimate. Cache hits bill $0. Cascade rejections still bill their
attempt (the row's `attempts` column shows how many).

**Approximation:** baseline cost prices the *chosen* model's completion length
at the baseline model's rates. A different model would write a different
length. For the demo's extraction/summary tasks the bias is small; for open
generation it would not be.

## Quality

A "successful task" is whatever the label source says:

1. client feedback (`POST /v1/feedback`),
2. deterministic checkers in the simulator (exact / regex / numeric / contains /
   JSON equality) against generated ground truth,
3. the shadow evaluator - in mock mode a deterministic comparison against the
   simulator's ground truth, not a real judge.

The mock's per-model skill probabilities are config inputs, not measurements.
The demo demonstrates the *mechanism* (router learns from labeled outcomes),
not real model quality.

## Savings

`savings % = 1 - spend/baseline_spend` over the selected window, computed
**only over `model: "auto"` (routed) traffic**. Pinned-model requests already
fixed the model by the caller, so DietGate made no decision to save anything -
they count towards total spend but not towards savings. The dashboard labels
the baseline card accordingly. Savings are reported **net of learning cost** in
a separate panel: exploration spend (`explore = 1` rows) plus shadow-eval
spend. In the 2000-task demo run exploration is roughly half the spend -
learning is not free, and the dashboard says so.

## Overhead

```
overhead_ms = (t_upstream_request_sent - t_recv)
            + (t_first_chunk_out - t_first_chunk_in)
            + (t_done_out - t_done_in)
```

Measured with `perf_counter_ns()` inside the gateway process; upstream wait is
excluded by construction.

**How the benchmark measures it** (`bench/overhead.py`, revised after the
first review): the headline numbers are the gateway's *own* `overhead_ms`
percentiles, read back from `/admin/latency` after driving a zero-latency-mock
gateway at concurrency 1 with a dedicated warmup phase (cold-start first
requests cost 1-19 ms and are discarded). An external client-measured
difference against a bare mock handler is included at concurrency 1 only as a
cross-check. Rows for concurrency >= 50 were removed: two single-process
asyncio servers saturate at different points and their quantile difference
measures queueing order, not routing overhead (earlier runs showed negative
"overhead" there). Load beyond saturation needs a real load generator with
multiple workers (wrk/oha/locust).

The overhead is still the number for *this Python implementation* on the
recorded hardware, not a Go data plane (`docs/GO_PORT_NOTES.md`).

## Honest limitations

1. All demo numbers come from synthetic traffic and simulated model skill.
2. Baseline cost assumes the chosen model's completion length.
3. Exploration and shadow evaluation cost real money in production; the
   dashboard shows savings net of them.
4. Python was chosen for iteration speed; the hot path is isolated for a Go port.
5. Caching is exact-match only; semantic caching is future work with real
   correctness risk.
6. This is an independent demo built for a job application - it is not
   TokenDiet's product or code.
