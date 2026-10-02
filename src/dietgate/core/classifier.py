"""Task-type classification: X-Task-Type header wins, then keyword heuristic."""
from __future__ import annotations

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "extract": ("extract", "json", "parse", "fields", "invoice", "pull the", "list all"),
    "classify": ("classify", "category", "categorize", "sentiment", "label"),
    "summarize": ("summarize", "summary", "tl;dr", "condense", "recap"),
    "reason": ("calculate", "how many", "how much", "solve", "why ", "compute", "arithmetic"),
    "code": ("write a function", "def ", "code", "bug", "implement", "python", "function that"),
}
_ORDER = ("extract", "classify", "summarize", "reason", "code")


def classify(messages: list[dict], header_value: str | None = None) -> str:
    """Return the task type. Header wins over the heuristic; 'unknown' fallback."""
    if header_value:
        return header_value.strip().lower()
    text = " ".join(
        str(m.get("content", "")) for m in messages if isinstance(m, dict)
    ).lower()
    for task_type in _ORDER:
        if any(kw in text for kw in _KEYWORDS[task_type]):
            return task_type
    return "unknown"
