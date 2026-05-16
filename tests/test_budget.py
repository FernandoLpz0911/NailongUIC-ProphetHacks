from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agent.openrouter.cost_tracker import CostRecord, CostTracker
from eval.budget import BUDGET_WARN_USD, project_spend


def _seed_costs(db_path: Path, records: list[tuple[str, float]]) -> None:
    tracker = CostTracker(db_path)
    for event_id, cost in records:
        tracker.log(
            CostRecord(
                event_id=event_id,
                model="test-model",
                latency_ms=100.0,
                cost_usd=cost,
            )
        )


def test_project_spend_math(tmp_path: Path) -> None:
    db = tmp_path / "costs.sqlite"
    _seed_costs(
        db,
        [
            ("E1", 0.02),
            ("E1", 0.01),
            ("E2", 0.03),
        ],
    )

    result = project_spend(events_per_day=10, days=5, db_path=db)

    assert result["distinct_events_in_db"] == 2
    assert result["avg_cost_per_event_usd"] == 0.03  # (0.03+0.03)/2
    assert result["total_events"] == 50
    assert result["projected_spend_usd"] == 1.5
    assert result["over_budget_warning"] is False


def test_project_spend_flags_over_budget(tmp_path: Path) -> None:
    db = tmp_path / "costs.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                model TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                cost_usd REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "INSERT INTO costs (event_id, model, latency_ms, cost_usd) VALUES (?, ?, ?, ?)",
            ("BIG", "opus", 1000.0, 5.0),
        )

    result = project_spend(events_per_day=20, days=10, db_path=db)

    assert result["projected_spend_usd"] == 1000.0
    assert result["over_budget_warning"] is True
    assert result["budget_warn_threshold_usd"] == BUDGET_WARN_USD


def test_budget_json_shape(tmp_path: Path) -> None:
    db = tmp_path / "costs.sqlite"
    _seed_costs(db, [("A", 0.01)])

    result = project_spend(events_per_day=1, days=1, db_path=db)
    payload = json.loads(json.dumps(result))

    assert payload["projected_spend_usd"] == 0.01
    assert "measured_spend_usd" in payload
