"""Request lifecycle context: timings, ids and routing decisions (hot path)."""
from __future__ import annotations

from dataclasses import dataclass, field

from dietgate.utils.timing import ns_to_ms, now_ns


@dataclass
class RequestContext:
    """Everything the hot path learns about one HTTP request.

    Timestamps are `perf_counter_ns()` values recorded at each lifecycle step
    (spec section 5). Overhead explicitly excludes time spent waiting on the
    upstream provider.
    """

    request_id: str
    task_id: str
    api_key_id: str
    stream: bool

    requested_model: str = "auto"
    header_task_type: str | None = None
    header_policy: str | None = None
    cache_allowed: bool = False

    # parsed request payload
    messages: list[dict] = field(default_factory=list)
    tools: list | None = None
    response_format: dict | None = None
    max_tokens: int | None = None
    temperature: float = 1.0
    metadata: dict = field(default_factory=dict)

    # lifecycle timestamps
    t_recv: int = 0
    t_auth: int = 0
    t_classified: int = 0
    t_cache_done: int = 0
    t_shortcut_done: int = 0
    t_routed: int = 0
    t_upstream_request_sent: int = 0
    t_first_chunk_in: int = 0
    t_first_chunk_out: int = 0
    t_done_in: int = 0
    t_done_out: int = 0

    # outcomes of the hot path
    task_type: str = "unknown"
    chosen_model: str | None = None
    policy: str | None = None
    explore: bool = False
    decision_reason: str = ""
    fallbacks: list[str] = field(default_factory=list)
    cache_hit: bool = False
    shortcut: str | None = None
    status: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    baseline_cost_usd: float = 0.0
    ttft_ms: float | None = None
    total_ms: float | None = None
    overhead_ms: float | None = None
    attempts: int = 0
    error: str | None = None
    extra_attempt_cost_usd: float = 0.0  # cascade: cost of rejected attempts

    def overhead_ms_computed(self) -> float:
        """Spec section 5 definition; non-upstream paths are pure overhead."""
        if self.t_upstream_request_sent:
            pre = self.t_upstream_request_sent - self.t_recv
            tail = (self.t_done_out - self.t_done_in) if (self.t_done_in and self.t_done_out) else 0
            first = (
                (self.t_first_chunk_out - self.t_first_chunk_in)
                if (self.t_first_chunk_in and self.t_first_chunk_out)
                else 0
            )
            return ns_to_ms(pre + first + tail)
        return ns_to_ms(self.t_done_out - self.t_recv) if self.t_done_out else 0.0

    def ttft_ms_computed(self) -> float | None:
        if self.t_first_chunk_in and self.t_upstream_request_sent:
            return ns_to_ms(self.t_first_chunk_in - self.t_upstream_request_sent)
        return None

    def total_ms_computed(self) -> float:
        end = self.t_done_out or now_ns()
        return ns_to_ms(end - self.t_recv)
