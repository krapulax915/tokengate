"""Loading and validation of the YAML configuration files (config/*.yaml)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

# Shipped dev-only credentials. A public or non-demo deployment must never
# boot with these (see validate_public_safety).
DEFAULT_ADMIN_KEY = "admin-dev-key"
DEFAULT_DATA_KEY = "dg-demo-key"


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
    # Optional per-model upstream settings (openai_compat only), so several OpenAI-compatible
    # providers (OpenAI, Novita, vLLM, OpenRouter, ...) can live side by side:
    base_url: str | None = None          # default: env OPENAI_BASE_URL or the OpenAI URL
    api_key_env: str | None = None       # name of the env var holding the key (default OPENAI_API_KEY)
    upstream_model: str | None = None    # the provider's model id, when it differs from `id`
    extra_body: dict = field(default_factory=dict)   # extra request fields, e.g. reasoning_effort
    stream_include_usage: bool = True    # send stream_options.include_usage (exact streamed token counts)

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
    public_dashboard: bool = False   # /admin/* reads need no key (mock-only demos; see validate_public_safety)


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
        if provider != "mock" and not (
            "price_in_per_mtok" in raw and "price_out_per_mtok" in raw
        ):
            # A forgotten price would silently report $0 cost and corrupt every savings number.
            raise ConfigError(
                f"model {mid}: real providers must set price_in_per_mtok and price_out_per_mtok "
                "explicitly (copy them from the provider's pricing page; 0 is allowed for free/local models)"
            )
        if provider != "mock" and disable_real_providers:
            continue  # excluded entirely; models marked unavailable are also skipped by registry
        api_key_env = raw.get("api_key_env")
        if provider != "mock" and not _provider_env_available(provider, api_key_env):
            continue  # no API key -> model not routable
        out[mid] = ModelSpec(
            id=mid,
            provider=provider,
            price_in_per_mtok=float(raw.get("price_in_per_mtok", 0.0)),
            price_out_per_mtok=float(raw.get("price_out_per_mtok", 0.0)),
            mock=mock,
            base_url=(str(raw["base_url"]).rstrip("/") if raw.get("base_url") else None),
            api_key_env=(str(api_key_env) if api_key_env else None),
            upstream_model=(str(raw["upstream_model"]) if raw.get("upstream_model") else None),
            extra_body=dict(raw.get("extra_body") or {}),
            stream_include_usage=bool(raw.get("stream_include_usage", True)),
        )
    if not out:
        raise ConfigError("no routable models configured")
    return out


def _provider_env_available(provider: str, api_key_env: str | None = None) -> bool:
    if api_key_env:
        return bool(os.environ.get(api_key_env))
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
        public_dashboard=bool(data.get("public_dashboard", False)),
    )
    for key, value in (overrides or {}).items():
        if value is not None:
            object.__setattr__(cfg, key, value)
    return cfg


def apply_api_keys_override(app_cfg: AppConfig, api_keys_env: str | None) -> AppConfig:
    """DIETGATE_API_KEYS (comma-separated) overrides config/app.yaml api_keys."""
    if not api_keys_env:
        return app_cfg
    keys = [k.strip() for k in api_keys_env.split(",") if k.strip()]
    if not keys:
        raise ConfigError("DIETGATE_API_KEYS is set but contains no usable keys")
    return replace(app_cfg, api_keys=keys)


MIN_SECRET_LEN = 16   # public deployments: admin/API/control keys must be at least this long


def validate_public_safety(
    app_cfg: AppConfig, *, public: bool, catalog: dict[str, ModelSpec] | None = None
) -> None:
    """Refuse to boot an internet-facing configuration that is unsafe.

    "Hardened" = DIETGATE_PUBLIC=true or demo_mode=false. Then the shipped default credentials
    (admin-dev-key / dg-demo-key) and keys shorter than MIN_SECRET_LEN characters are refused.
    A read-only public dashboard is only allowed when nothing real can leak through it.
    Local development (demo_mode=true, no PUBLIC flag) may keep the shipped demo keys.
    Error messages name the problem and the fix, never a key value other than the shipped defaults.
    """
    problems: list[str] = []
    if public or not app_cfg.demo_mode:
        if app_cfg.admin_key == DEFAULT_ADMIN_KEY:
            problems.append("admin_key is the default 'admin-dev-key' (set a strong DIETGATE_ADMIN_KEY)")
        elif len(app_cfg.admin_key) < MIN_SECRET_LEN:
            problems.append(f"admin_key is shorter than {MIN_SECRET_LEN} characters (set a strong DIETGATE_ADMIN_KEY)")
        if not app_cfg.api_keys:
            problems.append("no data-plane API keys configured (set DIETGATE_API_KEYS)")
        if DEFAULT_DATA_KEY in app_cfg.api_keys:
            problems.append("api_keys contains the default 'dg-demo-key' (set DIETGATE_API_KEYS)")
        if any(len(k) < MIN_SECRET_LEN for k in app_cfg.api_keys if k != DEFAULT_DATA_KEY):
            problems.append(f"an API key is shorter than {MIN_SECRET_LEN} characters (set DIETGATE_API_KEYS)")
        if app_cfg.control_key and len(app_cfg.control_key) < MIN_SECRET_LEN:
            problems.append(f"control_key is shorter than {MIN_SECRET_LEN} characters (DIETGATE_CONTROL_KEY)")
    if getattr(app_cfg, "public_dashboard", False):
        if catalog and any(not m.is_mock for m in catalog.values()):
            problems.append(
                "public_dashboard is mock-only but real providers are configured "
                "(set DIETGATE_DISABLE_REAL_PROVIDERS=true or unset the provider keys)"
            )
        if app_cfg.store_prompts:
            problems.append("public_dashboard cannot be combined with store_prompts")
    if problems:
        raise ConfigError(
            "refusing to start: unsafe configuration for a public or non-demo deployment"
            " (DIETGATE_PUBLIC=true or demo_mode=false). Fix: " + "; ".join(problems)
        )
