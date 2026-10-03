"""Environment-based settings (DIETGATE_* env vars / .env file)."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Process-level settings. YAML files (config/*.yaml) carry the rest."""

    model_config = SettingsConfigDict(
        env_prefix="DIETGATE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    config_dir: Path = REPO_ROOT / "config"
    db_path: Path = REPO_ROOT / "data" / "dietgate.db"
    dashboard_dir: Path = REPO_ROOT / "dashboard"

    host: str = "127.0.0.1"
    port: int = 8080

    admin_key: str | None = None            # overrides app.yaml admin_key when set
    control_key: str | None = None          # overrides app.yaml control_key when set
    demo_mode: bool | None = None           # overrides app.yaml demo_mode when set
    store_prompts: bool | None = None       # overrides app.yaml store_prompts when set
    public_dashboard: bool | None = None    # read-only dashboard without a key (mock-only demos)
    disable_real_providers: bool = False
    auto_start_sim: bool = False
    auto_sim_scenario: str = "default"      # sim/scenarios.yaml entry used by the auto-start (2000 tasks)

    public: bool = False                    # PUBLIC deployment: refuse default/short credentials at startup
    api_keys: str | None = None             # comma-separated data-plane keys; overrides app.yaml

    seed: int = 42                          # deterministic seed for the mock provider


def load_settings() -> Settings:
    s = Settings()
    s.config_dir = Path(s.config_dir).resolve()
    s.db_path = Path(s.db_path).resolve()
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    return s
