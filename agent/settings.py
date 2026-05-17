"""Typed runtime config loaded from constants/constants.csv + .env.

This is the single place that translates the Prophet Hacks ruleset PDFs into
Python values our pipeline can consume. The CSV is the source of truth; .env
only carries credentials and operator-tunable knobs (kill-switch, alpha bounds).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
CONSTANTS_CSV = REPO_ROOT / "constants" / "constants.csv"
DATA_DIR = REPO_ROOT / "data"


def _parse_money(value: str) -> float:
    return float(value.replace("$", "").replace(",", "").strip())


def _load_csv_constants(path: Path = CONSTANTS_CSV) -> dict[str, float | str]:
    """Parse constants.csv -> {Rule_or_Constant: numeric_or_string_value}."""
    out: dict[str, float | str] = {}
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row.get("Rule_or_Constant") or "").strip()
            val = (row.get("Value") or "").strip()
            if not key or not val:
                continue
            try:
                out[key] = _parse_money(val)
            except ValueError:
                out[key] = val
    return out


_CSV = _load_csv_constants()


def _csv_float(key: str, default: float) -> float:
    v = _CSV.get(key, default)
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


@dataclass(frozen=True)
class TradingConstraints:
    """Hard caps the Core API enforces; mirrored locally to fail fast."""

    initial_cash: float = field(default_factory=lambda: _csv_float("INITIAL_CASH", 10000.0))
    max_trades_per_tick: int = field(default_factory=lambda: int(_csv_float("MAX_TRADES_PER_TICK", 20)))
    max_trades_per_day: int = field(default_factory=lambda: int(_csv_float("MAX_TRADES_PER_DAY", 100)))
    max_open_positions: int = field(default_factory=lambda: int(_csv_float("MAX_OPEN_POSITIONS", 30)))
    max_notional_per_market: float = field(default_factory=lambda: _csv_float("MAX_NOTIONAL_PER_MARKET", 1000.0))
    max_gross_exposure: float = field(default_factory=lambda: _csv_float("MAX_GROSS_EXPOSURE", 10000.0))
    max_intents_per_tick_request: int = field(
        default_factory=lambda: int(_csv_float("MAX_INTENTS_PER_TICK_REQUEST", 50))
    )
    fee_rate: float = field(default_factory=lambda: _csv_float("FEE_RATE", 0.0))
    tick_interval_seconds: int = field(
        default_factory=lambda: int(_csv_float("TICK_INTERVAL_SECONDS", 900))
    )
    tick_submission_deadline_secs: int = field(
        default_factory=lambda: int(_csv_float("TICK_SUBMISSION_DEADLINE_SECS", 540))
    )


@dataclass(frozen=True)
class CalibrationConfig:
    """Knobs for the market-anchoring blend in agent/calibration.py."""

    alpha_high: float = field(default_factory=lambda: _env_float("ALPHA_HIGH", 0.7))
    alpha_medium: float = field(default_factory=lambda: _env_float("ALPHA_MEDIUM", 0.5))
    alpha_low: float = field(default_factory=lambda: _env_float("ALPHA_LOW", 0.25))
    max_deviation: float = field(default_factory=lambda: _env_float("MAX_EDGE_DEVIATION", 0.30))
    polymarket_agreement_band: float = field(
        default_factory=lambda: _env_float("POLYMARKET_AGREEMENT_BAND", 0.05)
    )
    polymarket_min_volume_usd: float = field(
        default_factory=lambda: _env_float("POLYMARKET_MIN_VOLUME_USD", 10000.0)
    )
    llm_ensemble_n: int = field(default_factory=lambda: _env_int("LLM_ENSEMBLE_N", 6))


@dataclass(frozen=True)
class RiskConfig:
    """Knobs for the risk-aware action stage."""

    min_edge: float = field(default_factory=lambda: _env_float("MIN_EDGE", 0.05))
    min_edge_relaxed: float = field(default_factory=lambda: _env_float("MIN_EDGE_RELAXED", 0.03))
    # Raw gap gate: minimum |p_model - p_market| before blending required to open a position.
    # Used when the ensemble provides a raw_gap; otherwise min_edge on blended edge applies.
    min_raw_gap: float = field(default_factory=lambda: _env_float("MIN_RAW_GAP", 0.04))
    # Quarter-Kelly (0.25) is optimal under CRRA utility with rho≈4 and a Sharpe penalty.
    kelly_fraction: float = field(default_factory=lambda: _env_float("KELLY_FRACTION", 0.25))
    max_position_pct_of_equity: float = field(
        default_factory=lambda: _env_float("MAX_POSITION_PCT_OF_EQUITY", 0.05)
    )
    min_intent_size_usd: float = field(default_factory=lambda: _env_float("MIN_INTENT_SIZE_USD", 5.0))
    trade_floor_count: int = field(default_factory=lambda: _env_int("TRADE_FLOOR_COUNT", 14))


@dataclass(frozen=True)
class RuntimeConfig:
    """One-stop bundle of every config the agent needs at runtime."""

    constraints: TradingConstraints = field(default_factory=TradingConstraints)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    # Operator-set knobs from .env
    pa_server_url: str = field(default_factory=lambda: os.getenv("PA_SERVER_URL", "https://api.aiprophet.dev"))
    pa_server_api_key: str | None = field(default_factory=lambda: os.getenv("PA_SERVER_API_KEY") or None)
    openrouter_api_key: str | None = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY") or None)
    brave_api_key: str | None = field(default_factory=lambda: os.getenv("BRAVE_API_KEY") or None)
    tavily_api_key: str | None = field(default_factory=lambda: os.getenv("TAVILY_API_KEY") or None)

    cost_db_path: str = field(default_factory=lambda: os.getenv("COST_DB_PATH", "./costs.sqlite"))
    kill_switch_usd: float = field(default_factory=lambda: _env_float("KILL_SWITCH_USD", 180.0))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


def load() -> RuntimeConfig:
    """Factory; tests may construct RuntimeConfig directly with overrides."""
    return RuntimeConfig()
