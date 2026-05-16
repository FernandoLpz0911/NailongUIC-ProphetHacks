"""Cost ledger + kill-switch for the 14-day eval window.

Every LLM call should append a row via `record_call(...)`. Stages call
`is_killed()` before any expensive work; if the cumulative USD spend over
the lifetime of the SQLite ledger has crossed `KILL_SWITCH_USD`, the
forecast stage falls back to `p_yes = market_mid` (zero edge, zero risk).

The ledger is a single-table SQLite db; we never delete rows so the file
also doubles as an audit log for the post-event writeup.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from agent.settings import load as load_runtime

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_DB_PATH: Path | None = None
_CUMULATIVE_CACHE: float | None = None  # invalidated on every write
_KILL_THRESHOLD: float | None = None


def _db_path() -> Path:
    global _DB_PATH
    if _DB_PATH is None:
        rt = load_runtime()
        _DB_PATH = Path(rt.cost_db_path).resolve()
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _DB_PATH


def _kill_threshold() -> float:
    global _KILL_THRESHOLD
    if _KILL_THRESHOLD is None:
        _KILL_THRESHOLD = load_runtime().kill_switch_usd
    return _KILL_THRESHOLD


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            usd_cost REAL NOT NULL,
            stage TEXT,
            market_id TEXT
        )
    """)
    conn.commit()
    return conn


def record_call(
    *,
    provider: str,
    model: str,
    usd_cost: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    stage: str | None = None,
    market_id: str | None = None,
) -> None:
    """Persist one LLM call's cost. Safe to call from any thread."""
    global _CUMULATIVE_CACHE
    if usd_cost < 0:
        return
    try:
        with _LOCK:
            conn = _connect()
            conn.execute(
                "INSERT INTO llm_calls (ts, provider, model, prompt_tokens, "
                "completion_tokens, usd_cost, stage, market_id) VALUES (?,?,?,?,?,?,?,?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    provider, model,
                    int(prompt_tokens), int(completion_tokens),
                    float(usd_cost),
                    stage, market_id,
                ),
            )
            conn.commit()
            conn.close()
            _CUMULATIVE_CACHE = None
    except Exception as e:
        # Cost logging is best-effort. Never raise into the pipeline.
        logger.warning("Spend ledger write failed: %s", e)


def cumulative_usd() -> float:
    """Sum of all USD costs recorded so far."""
    global _CUMULATIVE_CACHE
    if _CUMULATIVE_CACHE is not None:
        return _CUMULATIVE_CACHE
    try:
        with _LOCK:
            conn = _connect()
            row = conn.execute(
                "SELECT COALESCE(SUM(usd_cost), 0) FROM llm_calls"
            ).fetchone()
            conn.close()
            _CUMULATIVE_CACHE = float(row[0] or 0.0)
            return _CUMULATIVE_CACHE
    except Exception as e:
        logger.warning("Spend ledger read failed: %s", e)
        return 0.0


def is_killed() -> bool:
    """True iff cumulative spend has crossed KILL_SWITCH_USD."""
    threshold = _kill_threshold()
    if threshold <= 0:
        return False
    return cumulative_usd() >= threshold


def reset_caches() -> None:
    """Test hook: forces re-resolution of the db path and kill threshold."""
    global _DB_PATH, _KILL_THRESHOLD, _CUMULATIVE_CACHE
    _DB_PATH = None
    _KILL_THRESHOLD = None
    _CUMULATIVE_CACHE = None


# Hook so callers can override the kill threshold at runtime (for the
# Phase-4 gate test which sets it to $0.01 and verifies fallback).
def override_kill_threshold(usd: float) -> None:
    global _KILL_THRESHOLD
    _KILL_THRESHOLD = float(usd)


_ = os  # reserved for future env-driven overrides
