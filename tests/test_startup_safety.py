"""Startup safety: default credentials refused in public / non-demo mode."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from dietgate.core.config import ConfigError
from dietgate.settings import Settings
from tests.conftest import CHAT_URL, TEST_DIR, make_app


def make_test_settings(**overrides) -> Settings:
    base = dict(
        config_dir=TEST_DIR / "config",
        db_path=TEST_DIR / "tmp-startup.db",  # never connected in these tests
        dashboard_dir=TEST_DIR.parent / "dashboard",
        seed=42,
    )
    base.update(overrides)
    return Settings(**base)


def test_public_flag_refuses_default_keys(tmp_path) -> None:
    with pytest.raises(ConfigError) as exc:
        make_app(
            tmp_path,
            settings=make_test_settings(public=True, admin_key="admin-dev-key"),
        )
    msg = str(exc.value)
    assert "refusing to start" in msg
    assert "admin-dev-key" in msg and "dg-demo-key" in msg
    assert "DIETGATE_ADMIN_KEY" in msg and "DIETGATE_API_KEYS" in msg


def test_non_demo_mode_refuses_default_keys(tmp_path) -> None:
    with pytest.raises(ConfigError) as exc:
        make_app(tmp_path, settings=make_test_settings(demo_mode=False))
    assert "refusing to start" in str(exc.value)


def test_public_with_custom_keys_starts(tmp_path) -> None:
    app = make_app(
        tmp_path,
        settings=make_test_settings(
            public=True,
            admin_key="strong-admin-key",
            api_keys="client-a,client-b",
        ),
    )
    assert app.state.rt.app_cfg.api_keys == ["client-a", "client-b"]


async def test_api_keys_override_changes_auth(tmp_path) -> None:
    """DIETGATE_API_KEYS replaces the yaml keys: dg-demo-key stops working."""
    app = make_app(tmp_path, settings=make_test_settings(api_keys="client-a"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r_new = await client.post(
            CHAT_URL,
            json={"model": "mock-small",
                  "messages": [{"role": "user", "content": "Extract fields: 1 EUR"}]},
            headers={"Authorization": "Bearer client-a"},
        )
        r_old = await client.post(
            CHAT_URL,
            json={"model": "mock-small",
                  "messages": [{"role": "user", "content": "Extract fields: 1 EUR"}]},
            headers={"Authorization": "Bearer dg-demo-key"},
        )
    assert r_new.status_code == 200
    assert r_old.status_code == 401


def test_default_demo_mode_still_boots(tmp_path) -> None:
    """Local dev (demo_mode=true, no PUBLIC flag) keeps the shipped demo keys."""
    app = make_app(tmp_path)
    assert "dg-demo-key" in app.state.rt.app_cfg.api_keys


def test_env_flags_parsed(monkeypatch) -> None:
    monkeypatch.setenv("DIETGATE_PUBLIC", "true")
    monkeypatch.setenv("DIETGATE_API_KEYS", "k1 , k2 ,")
    settings = Settings()
    assert settings.public is True
    assert settings.api_keys == "k1 , k2 ,"
