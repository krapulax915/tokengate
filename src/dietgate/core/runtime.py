"""Component wiring: one Runtime object per process, shared via app.state."""
from __future__ import annotations

from dataclasses import dataclass, field

from dietgate.core.config import AppConfig, ModelSpec, TaskSpec, load_app, load_models, load_tasks
from dietgate.providers.registry import ProviderRegistry, build_registry
from dietgate.settings import Settings


@dataclass
class Runtime:
    settings: Settings
    catalog: dict[str, ModelSpec]
    tasks_cfg: dict[str, TaskSpec]
    app_cfg: AppConfig
    registry: ProviderRegistry
    pipeline: object = None  # Pipeline; set right after construction (circular import)
    # wired in later phases:
    db: object | None = None
    authn: object | None = None
    cache: object | None = None
    stats: object | None = None
    router: object | None = None
    writer: object | None = None
    learner: object | None = None
    evaluator: object | None = None
    sim_runner: object | None = None
    extra: dict = field(default_factory=dict)


def _app_cfg_overrides(settings: Settings) -> dict:
    return {
        "admin_key": settings.admin_key,
        "control_key": settings.control_key,
        "demo_mode": settings.demo_mode,
        "store_prompts": settings.store_prompts,
    }


async def start_background(rt: Runtime) -> None:
    """Open storage and start the background tasks (called from lifespan)."""
    from sim.runner import SimRunner

    from dietgate.core.evaluator import Evaluator
    from dietgate.core.learner import Learner
    from dietgate.storage.db import Database
    from dietgate.storage.writer import Writer

    db = Database(rt.settings.db_path)
    await db.connect()
    rt.db = db
    rt.writer = Writer(db, rt.app_cfg.writer)
    await rt.writer.start()
    rt.learner = Learner(rt.stats, rt.writer)
    await rt.learner.start()
    await rt.learner.rebuild_from_db(db)  # restart recovery (spec section 9)
    rt.evaluator = Evaluator(rt.learner, rt.writer, rt.app_cfg.evaluator)
    await rt.evaluator.start()
    if rt.app_cfg.demo_mode:
        rt.sim_runner = SimRunner(
            f"http://127.0.0.1:{rt.settings.port}", api_key=rt.app_cfg.api_keys[0]
        )
        if rt.settings.auto_start_sim:
            rt.sim_runner.start("fast")


async def stop_background(rt: Runtime) -> None:
    if rt.sim_runner is not None:
        rt.sim_runner.stop()
        await rt.sim_runner.wait_stopped()
    if rt.evaluator is not None:
        await rt.evaluator.stop()
    if rt.learner is not None:
        await rt.learner.stop()
    if rt.writer is not None:
        await rt.writer.stop()
    if rt.db is not None:
        await rt.db.close()
    await rt.registry.aclose()


def build_runtime(settings: Settings) -> Runtime:
    from dietgate.core.config import apply_api_keys_override, validate_public_safety

    catalog = load_models(settings.config_dir, settings.disable_real_providers)
    tasks_cfg = load_tasks(settings.config_dir)
    app_cfg = load_app(settings.config_dir, _app_cfg_overrides(settings))
    app_cfg = apply_api_keys_override(app_cfg, settings.api_keys)
    validate_public_safety(app_cfg, public=settings.public)
    registry = build_registry(catalog, settings, app_cfg)
    rt = Runtime(
        settings=settings,
        catalog=catalog,
        tasks_cfg=tasks_cfg,
        app_cfg=app_cfg,
        registry=registry,
    )
    from dietgate.core.auth import Authenticator
    from dietgate.core.cache import ResponseCache
    from dietgate.core.pipeline import Pipeline
    from dietgate.core.policies.base import PolicyDeps
    from dietgate.core.policies.cascade import CascadePolicy
    from dietgate.core.policies.static import StaticPolicy
    from dietgate.core.policies.thompson import ThompsonPolicy
    from dietgate.core.router import Router
    from dietgate.core.stats import ArmStatsStore

    rt.pipeline = Pipeline(rt)
    rt.authn = Authenticator(
        app_cfg.api_keys,
        app_cfg.rate_limit.requests_per_min,
        app_cfg.rate_limit.burst,
        app_cfg.daily_spend_cap_usd,
    )
    rt.cache = ResponseCache(app_cfg.cache)
    rt.stats = ArmStatsStore()
    deps = PolicyDeps(catalog=catalog, tasks=tasks_cfg, stats=rt.stats)
    static_p, thompson_p, cascade_p = StaticPolicy(deps), ThompsonPolicy(deps, seed=settings.seed + 1), CascadePolicy(deps)
    rt.router = Router(
        {static_p.name: static_p, thompson_p.name: thompson_p, cascade_p.name: cascade_p},
        app_cfg.policy,
    )
    return rt
