from __future__ import annotations

import sqlite3
from pathlib import Path

from agent.config import COST_DB_PATH


def cost_summary(db_path: Path | None = None) -> dict[str, float | int]:
    path = db_path or COST_DB_PATH
    if not path.exists():
        return {"total_spend_usd": 0.0, "event_count": 0, "record_count": 0, "avg_cost_per_event": 0.0}

    with sqlite3.connect(path) as conn:
        total_row = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM costs").fetchone()
        count_row = conn.execute("SELECT COUNT(*) FROM costs").fetchone()
        events_row = conn.execute("SELECT COUNT(DISTINCT event_id) FROM costs").fetchone()

    total = float(total_row[0] if total_row else 0.0)
    record_count = int(count_row[0] if count_row else 0)
    event_count = int(events_row[0] if events_row else 0)
    avg = total / event_count if event_count else 0.0
    return {
        "total_spend_usd": total,
        "event_count": event_count,
        "record_count": record_count,
        "avg_cost_per_event": avg,
    }
