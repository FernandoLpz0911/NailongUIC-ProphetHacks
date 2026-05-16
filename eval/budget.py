#!/usr/bin/env python3
"""Project OpenRouter spend from measured cost-per-event."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.config import COST_DB_PATH, DATA_DIR
from agent.openrouter.cost_tracker import CostTracker
from eval.costs import cost_summary

BUDGET_WARN_USD = 150.0


def project_spend(
    *,
    events_per_day: int,
    days: int,
    db_path: Path | None = None,
) -> dict:
    path = db_path or COST_DB_PATH
    tracker = CostTracker(path)
    summary = cost_summary(path)
    avg_per_event = float(summary["avg_cost_per_event"])
    total_events = events_per_day * days
    projected_usd = avg_per_event * total_events
    measured_total = tracker.total_spend()

    return {
        "events_per_day": events_per_day,
        "days": days,
        "total_events": total_events,
        "avg_cost_per_event_usd": round(avg_per_event, 6),
        "projected_spend_usd": round(projected_usd, 4),
        "measured_spend_usd": round(measured_total, 4),
        "cost_db": str(path),
        "distinct_events_in_db": int(summary["event_count"]),
        "cost_records_in_db": int(summary["record_count"]),
        "over_budget_warning": projected_usd > BUDGET_WARN_USD,
        "budget_warn_threshold_usd": BUDGET_WARN_USD,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Project hackathon spend from costs.sqlite")
    parser.add_argument("--events-per-day", type=int, default=20)
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--db", type=Path, default=COST_DB_PATH)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "budget_projection.json")
    args = parser.parse_args()

    result = project_spend(
        events_per_day=args.events_per_day,
        days=args.days,
        db_path=args.db,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"events_per_day={result['events_per_day']} days={result['days']}")
    print(f"avg_cost_per_event=${result['avg_cost_per_event_usd']:.6f}")
    print(f"projected_spend=${result['projected_spend_usd']:.2f} ({result['total_events']} events)")
    print(f"measured_spend_so_far=${result['measured_spend_usd']:.4f}")
    if result["over_budget_warning"]:
        print(f"WARNING: projected spend exceeds ${BUDGET_WARN_USD:.0f} — reduce volume or cost per event")
    else:
        print(f"OK: projected spend under ${BUDGET_WARN_USD:.0f} threshold")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
