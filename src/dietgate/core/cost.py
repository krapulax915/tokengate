"""Token counting and price math (spec section 8)."""
from __future__ import annotations

from dietgate.core.config import ModelSpec
from dietgate.utils.tokens import estimate_tokens, prompt_tokens_estimate


def completion_tokens_estimate(text: str | None) -> int:
    return estimate_tokens(text)


def usage_estimates(messages: list[dict], completion_text: str | None) -> tuple[int, int]:
    """(prompt_tokens, completion_tokens) via the len/4 estimate."""
    return prompt_tokens_estimate(messages), completion_tokens_estimate(completion_text)


def _price(catalog: dict[str, ModelSpec], model_id: str) -> tuple[float, float] | None:
    spec = catalog.get(model_id)
    if spec is None:
        return None
    return spec.price_in_per_mtok, spec.price_out_per_mtok


def request_cost_usd(
    catalog: dict[str, ModelSpec], model_id: str, prompt_tokens: int, completion_tokens: int
) -> float:
    price = _price(catalog, model_id)
    if price is None:
        return 0.0
    price_in, price_out = price
    return prompt_tokens / 1e6 * price_in + completion_tokens / 1e6 * price_out


def baseline_cost_usd(
    catalog: dict[str, ModelSpec], baseline_model_id: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Baseline priced at the task type's baseline model using the *chosen*
    model's completion length (documented approximation, see METHODOLOGY.md)."""
    return request_cost_usd(catalog, baseline_model_id, prompt_tokens, completion_tokens)
