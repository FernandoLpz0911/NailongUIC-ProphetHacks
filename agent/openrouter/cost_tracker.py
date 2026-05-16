from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from agent.config import COST_DB_PATH


@dataclass
class CostRecord:
    event_id: str
    model: str
    latency_ms: float
    cost_usd: float


class CostTracker:
    def __init__(self, db_path: Path = COST_DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS costs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    cost_usd REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def log(self, record: CostRecord) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO costs (event_id, model, latency_ms, cost_usd) VALUES (?, ?, ?, ?)",
                (record.event_id, record.model, record.latency_ms, record.cost_usd),
            )

    def total_spend(self) -> float:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM costs").fetchone()
        return float(row[0] if row else 0.0)
