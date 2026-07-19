"""Pydantic settings. Defaults mirror the locked decisions in _notes/IDEAS.md.

Notes
-----
Single source of configuration. Precedence once v0.1.0 wires in the YAML
layer: env vars (BALLAST_ prefix) > config/default.yaml > the defaults
coded below. The three must never disagree silently -- a test should
compare default.yaml against these defaults.

Every module gets settings via get_settings(); nothing reads os.environ or
parses YAML on its own. Numbers that are locked decisions (ewma_lambda,
walk-forward window/step, cost bps) live here so changing one place changes
every experiment, and the change shows up in git history.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BALLAST_")

    db_path: Path = Path("data/ballast.duckdb")
    universe: str = "sp500"
    ewma_lambda: float = 0.94
    wf_window: int = 252
    wf_step: int = 21
    cost_bps: float = 2.0
    risk_horizons_days: tuple[int, ...] = (1, 21)
    var_confidence: tuple[float, ...] = (0.95, 0.99)
    # SEC EDGAR requires a User-Agent identifying you with contact info.
    # Set BALLAST_EDGAR_USER_AGENT to e.g. "ballast/0.1 you@example.com".
    edgar_user_agent: str = "ballast (set BALLAST_EDGAR_USER_AGENT to identify yourself)"


def get_settings() -> Settings:
    return Settings()
