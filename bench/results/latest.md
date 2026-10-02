# DietGate gateway overhead (latest run)

- Date: 2026-10-02 21:17 Közép-európai nyári idő 
- Hardware: Intel64 Family 6 Model 44 Stepping 2, GenuineIntel | 12 logical cores | Windows-10-10.0.19045-SP0
- Python: 3.12.0 | zero-latency mock models | concurrency 1

## Headline: the gateway's own `overhead_ms` (spec section 5 definition)

Measured inside the gateway: pre-processing + first-chunk forwarding + tail
forwarding; upstream wait excluded by construction. Percentiles over all
measured requests of that mode (after a dedicated warmup phase - cold-start
first requests cost 1-19 ms and are discarded).

| mode | samples | overhead p50 | p90 | p99 | client p50 | client p99 | rps |
|---|---|---|---|---|---|---|---|
| nonstream | 1600 | 0.1382 | 0.158 | 0.2417 | 6.475 | 10.743 | 150.3 |
| stream | 1400 | 0.429 | 0.4972 | 0.6816 | 6.743 | 10.454 | 145.6 |

## Cross-check: external client-measured difference, concurrency 1 (non-stream)

- direct handler: p50 5.546 ms / p99 7.802 ms
- through gateway: p50 5.765 ms / p99 6.548 ms
- difference: **p50 0.22 ms / p99 -1.254 ms**

Cross-check interpretation: the p50 difference is consistent with the
internal measurement; the p99 difference is tail noise from two independent
request streams and is not meaningful - which is exactly why the headline
numbers come from the gateway's own per-request overhead accounting instead.

## Why there are no concurrency >= 50 rows

Two single-process asyncio servers saturate at different points on this box;
subtracting their quantiles measures queueing order, not routing overhead
(earlier runs showed negative overhead at c=50/200 for exactly this reason).
Load beyond saturation needs a real load generator with multiple workers
(wrk/oha/locust) and is out of scope for this benchmark.
