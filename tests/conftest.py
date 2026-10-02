"""Shared fixtures: an app built on the zero-latency test config."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from dietgate.main import create_app
from dietgate.settings import Settings

TEST_DIR = Path(__file__).resolve().parent

# The demo credential ships in tests/config/app.yaml; tests read it from there
# so no key literal ever appears in Python source.
_APP_YAML = yaml.safe_load((TEST_DIR / "config" / "app.yaml").read_text(encoding="utf-8"))
DEMO_API_KEY = _APP_YAML["api_keys"][0]
DEMO_ADMIN_KEY = _APP_YAML["admin_key"]
DEMO_CONTROL_KEY = _APP_YAML["control_key"]


def make_settings(tmp_path: Path | None = None, db_path: Path | None = None) -> Settings:
    return Settings(
        config_dir=TEST_DIR / "config",
        db_path=db_path or (tmp_path or TEST_DIR) / "test-dietgate.db",
        dashboard_dir=TEST_DIR.parent / "dashboard",
        seed=42,
    )


def make_app(tmp_path: Path | None = None, db_path: Path | None = None, settings=None):
    return create_app(settings or make_settings(tmp_path, db_path))


@pytest.fixture
async def client(tmp_path):
    app = make_app(tmp_path)
    transport = ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {DEMO_API_KEY}"}
    async with app.router.lifespan_context(app):  # starts db + writer
        async with AsyncClient(transport=transport, base_url="http://test", headers=headers) as ac:
            yield ac


def runtime_of(client: AsyncClient):
    return client._transport.app.state.rt  # noqa: SLF001


CHAT_URL = "/v1/chat/completions"


def chat_body(model: str = "mock-small", **overrides) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Extract the fields from this invoice text."}],
        "temperature": 1.0,
    }
    body.update(overrides)
    return body
