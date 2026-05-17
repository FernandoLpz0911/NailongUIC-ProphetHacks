"""Test isolation from .env runtime overrides.

`agent/settings.py` calls `load_dotenv()` at import time, which pulls our
production Sharpe-first knobs into `os.environ`. Tests that construct
`RiskConfig()` / `CalibrationConfig()` with no kwargs would otherwise pick
up those env values and assert against production tuning instead of the
documented dataclass fallback defaults.

This autouse fixture strips every override key BEFORE each test runs, so
tests see the conservative fallback defaults declared in settings.py.
Tests that explicitly want to exercise production tuning should pass the
relevant kwargs into `RiskConfig(...)` / `CalibrationConfig(...)` directly.
"""

from __future__ import annotations

import os

import pytest

# Every env var that any dataclass field in agent/settings.py reads via
# _env_float / _env_int / _env_bool. Kept exhaustive so a future addition
# to .env doesn't silently leak into the test harness.
_ENV_KEYS_TO_STRIP = (
    # CalibrationConfig
    "ALPHA_HIGH", "ALPHA_MEDIUM", "ALPHA_LOW",
    "MAX_EDGE_DEVIATION",
    "POLYMARKET_AGREEMENT_BAND", "POLYMARKET_MIN_VOLUME_USD",
    "LLM_ENSEMBLE_N",
    # RiskConfig
    "MIN_CONF_MODEL",
    "MIN_EDGE", "MIN_EDGE_RELAXED",
    "MIN_RAW_GAP", "MAX_RAW_GAP_HARD",
    "MAX_SPREAD_PCT", "TARGET_PORTFOLIO_VAR",
    "KELLY_FRACTION", "MAX_POSITION_PCT_OF_EQUITY",
    "MIN_INTENT_SIZE_USD", "TRADE_FLOOR_COUNT",
    "TAKE_PROFIT_THRESHOLD", "STOP_LOSS_THRESHOLD",
    "MAX_DAYS_TO_RESOLUTION", "NEAR_TERM_HORIZON_DAYS", "FAR_TERM_SIZE_FLOOR",
    "INVERT_STRATEGY", "POSITION_MIN_DWELL_TICKS",
    # TradingConstraints (env-only overrides)
    "MAX_INTENTS_PER_CATEGORY", "MAX_OPEN_PER_CATEGORY",
)


@pytest.fixture(autouse=True)
def _strip_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS_TO_STRIP:
        monkeypatch.delenv(key, raising=False)
