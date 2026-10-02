"""Loading and validation of the YAML configuration files (config/*.yaml)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """Raised when a config file is invalid."""


@dataclass(frozen=True)
class MockSpec:
    ttft_ms: float
    tokens_per_second: float
    jitter_pct: float
    skill: dict[str, float]

    def skill_for(self, task_type: str) -> float:
        return self.skill.get(task_type, 0.9)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    price_in_per_mtok: float
    price_out_per_mtok: float
    mock: MockSpec | None = None

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"


@dataclass(frozen=True)
class TaskSpec:
    min_success: float
    baseline: str
    candidates: list[str]


@dataclass(frozen=True)
class RateLimitConfig:
    requests_per_min: float
    burst: int


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool
    ttl_s: float
    max_entries: int


@dataclass(frozen=True)
class EvaluatorConfig:
    sample_rate: float
    delay_s: float


@dataclass(frozen=True)
class WriterConfig:
    queue_size: int
    batch_size: int
    flush_interval_ms: float


@dataclass(frozen=True)
class AppConfig:
    api_keys: list[str]
    rate_limit: RateLimitConfig
    daily_spend_cap_usd: float
    cache: CacheConfig
    policy: str
    evaluator: EvaluatorConfig
    writer: WriterConfig
    shortcuts_enabled: bool
    admin_key: str
    control_key: str
    demo_mode: bool
    store_prompts: bool
    request_size_limit_bytes: int
    upstream_timeout_s: float


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"missing config file: {path}")
    with path.open("rb") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"invalid YAML (expected mapping) in {path}")
    return data


def load_models(config_dir: Path, disable_real_providers: bool = False) -> dict[str, ModelSpec]:
    data = _read_yaml(config_dir / "models.yaml")
    out: dict[str, ModelSpec] = {}
    for i, raw in enumerate(data.get("models", [])):
        mid = raw.get("id")
        provider = raw.get("provider", "mock")
        if not mid:
            raise ConfigError(f"models.yaml entry #{i} has no id")
        if mid in out:
            raise ConfigError(f"duplicate model id: {mid}")
        mock_raw = raw.get("mock")
        mock = MockSpec(
            ttft_ms=float(mock_raw.get("ttft_ms", 0)),
            tokens_per_second=float(mock_raw.get("tokens_per_second", 0)),
            jitter_pct=float(mock_raw.get("jitter_pct", 0)),
            skill={k: float(v) for k, v in (mock_raw.get("skill") or {}).items()},
        ) if mock_raw else None
        if provider == "mock" and mock is None:
            raise ConfigError(f"model {mid}: provider 'mock' requires a 'mock:' section")
        if provider != "mock" and disable_real_providers:
            continue  # excluded entirely; models marked unavailable are also skipped by registry
        if provider != "mock" and not _provider_env_available(provider):
            continue  # no API key -> model not routable
        out[mid] = ModelSpec(
            id=mid,
            provider=provider,
            price_in_per_mtok=float(raw.get("price_in_per_mtok", 0.0)),
            price_out_per_mtok=float(raw.get("price_out_per_mtok", 0.0)),
            mock=mock,
        )
    if not out:
        raise ConfigError("no routable models configured")
    return out


def _provider_env_available(provider: str) -> bool:
    if provider == "openai_compat":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    return False


def load_tasks(config_dir: Path) -> dict[str, TaskSpec]:
    data = _read_yaml(config_dir / "tasks.yaml")
    out: dict[str, TaskSpec] = {}
    for ttype, raw in data.get("tasks", {}).items():
        out[ttype] = TaskSpec(
            min_success=float(raw["min_success"]),
            baseline=str(raw["baseline"]),
            candidates=[str(c) for c in raw.get("candidates", [])],
        )
    if "unknown" not in out:
        raise ConfigError("tasks.yaml must define the 'unknown' task type")
    return out


def load_app(config_dir: Path, overrides: dict | None = None) -> AppConfig:
    data = _read_yaml(config_dir / "app.yaml")
    rl = data.get("rate_limit") or {}
    cache = data.get("cache") or {}
    ev = data.get("evaluator") or {}
    wr = data.get("writer") or {}
    cfg = AppConfig(
        api_keys=[str(k) for k in (data.get("api_keys") or [])],
        rate_limit=RateLimitConfig(
            requests_per_min=float(rl.get("requests_per_min", 120)),
            burst=int(rl.get("burst", 30)),
        ),
        daily_spend_cap_usd=float(data.get("daily_spend_cap_usd", 50.0)),
        cache=CacheConfig(
            enabled=bool(cache.get("enabled", True)),
            ttl_s=float(cache.get("ttl_s", 300)),
            max_entries=int(cache.get("max_entries", 1000)),
        ),
        policy=str(data.get("policy", "thompson")),
        evaluator=EvaluatorConfig(
            sample_rate=float(ev.get("sample_rate", 0.05)),
            delay_s=float(ev.get("delay_s", 2.0)),
        ),
        writer=WriterConfig(
            queue_size=int(wr.get("queue_size", 10000)),
            batch_size=int(wr.get("batch_size", 200)),
            flush_interval_ms=float(wr.get("flush_interval_ms", 250)),
        ),
        shortcuts_enabled=bool(data.get("shortcuts_enabled", True)),
        admin_key=str(data.get("admin_key", "admin-dev-key")),
        control_key=str(data.get("control_key", "")),
        demo_mode=bool(data.get("demo_mode", True)),
        store_prompts=bool(data.get("store_prompts", False)),
        request_size_limit_bytes=int(data.get("request_size_limit_bytes", 262144)),
        upstream_timeout_s=float(data.get("upstream_timeout_s", 60)),
    )
    for key, value in (overrides or {}).items():
        if value is not None:
            object.__setattr__(cfg, key, value)
    return cfg
