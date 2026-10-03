"""Startup guard: weak/default credentials must never reach an internet-facing instance."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from dietgate.core.config import ConfigError, ModelSpec, validate_public_safety
from dietgate.main import create_app
from tests.conftest import DEMO_API_KEY, make_settings

STRONG_ADMIN = "a" * 24 + "-admin"
STRONG_API = "k" * 24 + "-api"
STRONG_CONTROL = "c" * 24 + "-ctl"


def _settings(tmp_path, **kw):
    s = make_settings(tmp_path)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_local_default_still_starts(tmp_path) -> None:
    create_app(_settings(tmp_path))  # demo_mode true, not public: unchanged behaviour


def test_public_refuses_default_keys(tmp_path) -> None:
    with pytest.raises(ConfigError) as exc:
        create_app(_settings(tmp_path, public=True))
    msg = str(exc.value)
    assert "admin_key" in msg and "api_keys" in msg


def test_public_refuses_default_api_key_even_with_strong_admin(tmp_path) -> None:
    with pytest.raises(ConfigError) as exc:
        create_app(_settings(tmp_path, public=True, admin_key=STRONG_ADMIN))
    assert "api_keys" in str(exc.value) and "admin_key" not in str(exc.value)


def test_public_starts_with_strong_keys(tmp_path) -> None:
    create_app(_settings(tmp_path, public=True, admin_key=STRONG_ADMIN, api_keys=STRONG_API,
                         control_key=STRONG_CONTROL))


def test_public_refuses_short_control_key(tmp_path) -> None:
    with pytest.raises(ConfigError):
        create_app(_settings(tmp_path, public=True, admin_key=STRONG_ADMIN, api_keys=STRONG_API,
                             control_key="short"))


def test_demo_mode_off_is_hardened_too(tmp_path) -> None:
    with pytest.raises(ConfigError):
        create_app(_settings(tmp_path, demo_mode=False))


def test_error_message_does_not_leak_key_values(tmp_path) -> None:
    with pytest.raises(ConfigError) as exc:
        create_app(_settings(tmp_path, public=True, admin_key="my-short-secret"))
    assert "my-short-secret" not in str(exc.value)


async def test_env_api_keys_replace_the_yaml_key(tmp_path) -> None:
    app = create_app(_settings(tmp_path, api_keys=f"{STRONG_API}, second-{STRONG_API}"))
    body = {"model": "mock-small", "messages": [{"role": "user", "content": "hi"}]}
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            old = await c.post("/v1/chat/completions", json=body,
                               headers={"Authorization": f"Bearer {DEMO_API_KEY}"})
            new = await c.post("/v1/chat/completions", json=body,
                               headers={"Authorization": f"Bearer {STRONG_API}"})
            second = await c.post("/v1/chat/completions", json=body,
                                  headers={"Authorization": f"Bearer second-{STRONG_API}"})
    assert old.status_code == 401
    assert new.status_code == 200 and second.status_code == 200


async def test_public_dashboard_reads_open_controls_closed(tmp_path) -> None:
    app = create_app(_settings(tmp_path, public_dashboard=True))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            read = await c.get("/admin/summary")                       # no key at all
            reset = await c.post("/admin/reset")                       # control: still closed
            reset_bad = await c.post("/admin/reset", headers={"X-Control-Key": "guess"})
    assert read.status_code == 200
    assert reset.status_code in (400, 401) and reset_bad.status_code in (400, 401)


async def test_admin_reads_still_need_key_by_default(tmp_path) -> None:
    app = create_app(_settings(tmp_path))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            assert (await c.get("/admin/summary")).status_code == 401


def test_public_dashboard_refuses_real_providers() -> None:
    from dietgate.core.config import load_app
    from tests.conftest import TEST_DIR

    cfg = load_app(TEST_DIR / "config", {"public_dashboard": True})
    real = ModelSpec(id="real-x", provider="anthropic", price_in_per_mtok=1.0, price_out_per_mtok=2.0)
    with pytest.raises(ConfigError, match="mock-only"):
        validate_public_safety(cfg, public=False, catalog={"real-x": real})


def test_real_model_without_prices_is_rejected(tmp_path) -> None:
    import shutil

    from dietgate.core.config import load_models
    from tests.conftest import TEST_DIR

    cfg = tmp_path / "config"
    shutil.copytree(TEST_DIR / "config", cfg)
    with (cfg / "models.yaml").open("a", encoding="utf-8") as f:
        f.write("  - id: real-no-price\n    provider: anthropic\n")
    with pytest.raises(ConfigError, match="price_in_per_mtok"):
        load_models(cfg)
