#!/usr/bin/env python3
"""CLI dashboard: Brier, avg return, cost, top wins/losses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent.config import DATA_DIR
from agent.schemas import Prediction
from eval.costs import cost_summary
from eval.harness import load_events
from eval.scoring import aggregate_scores, outcome_yes, score_one


def load_predictions(
    path: Path,
    events_by_id: dict[str, dict[str, Any]],
) -> dict[str, Prediction]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"predictions file must be a JSON list: {path}")
    out: dict[str, Prediction] = {}
    for row in rows:
        event_id = row["event_id"]
        if event_id not in events_by_id:
            continue
        out[event_id] = Prediction.model_validate(row["prediction"])
    return out


def per_event_returns(
    events_by_id: dict[str, dict[str, Any]],
    predictions_by_id: dict[str, Prediction],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event_id, pred in predictions_by_id.items():
        event = events_by_id.get(event_id)
        if event is None:
            continue
        brier, ret = score_one(pred, event)
        rows.append(
            {
                "event_id": event_id,
                "title": event.get("title", ""),
                "brier": brier,
                "return": ret,
                "outcome_yes": outcome_yes(event),
            }
        )
    return rows


def format_dashboard(
    scores: dict[str, float | int],
    cost: dict[str, float | int],
    event_rows: list[dict[str, Any]],
) -> str:
    count = int(scores["count"])
    lines = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Brier | {scores['brier']:.4f} |" if count else "| Brier | N/A |",
        f"| Avg Return | {scores['avg_return']:.4f} |" if count else "| Avg Return | N/A |",
        f"| Cost (total) | ${cost['total_spend_usd']:.4f} |",
        f"| Cost (avg/event) | ${cost['avg_cost_per_event']:.4f} |",
        f"| Events scored | {count} |",
        f"| Cost DB events | {cost['event_count']} |",
        "",
        "### Top 5 wins (by return)",
    ]
    if not event_rows:
        lines.append("_No scored events._")
    else:
        wins = sorted(event_rows, key=lambda r: r["return"], reverse=True)[:5]
        lines.extend(_format_leaderboard(wins))

    lines.append("")
    lines.append("### Top 5 losses (by return)")
    if not event_rows:
        lines.append("_No scored events._")
    else:
        losses = sorted(event_rows, key=lambda r: r["return"])[:5]
        lines.extend(_format_leaderboard(losses))

    return "\n".join(lines) + "\n"


def _format_leaderboard(rows: list[dict[str, Any]]) -> list[str]:
    out = ["| Event | Return | Brier | Title |", "| --- | --- | --- | --- |"]
    for row in rows:
        title = str(row["title"])[:60].replace("|", "/")
        out.append(
            f"| {row['event_id']} | {row['return']:.4f} | {row['brier']:.4f} | {title} |"
        )
    return out


def run_dashboard(
    events_path: Path,
    predictions_path: Path,
    *,
    cost_db: Path | None = None,
) -> str:
    events = load_events(events_path)
    events_by_id = {e["event_id"]: e for e in events}
    predictions = load_predictions(predictions_path, events_by_id)
    scores = aggregate_scores(events_by_id, predictions)
    cost = cost_summary(cost_db)
    event_rows = per_event_returns(events_by_id, predictions)
    return format_dashboard(scores, cost, event_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest dashboard (Brier, return, cost)")
    parser.add_argument("--events", type=Path, default=DATA_DIR / "events_test.json")
    parser.add_argument("--predictions", type=Path, default=DATA_DIR / "submission.json")
    parser.add_argument("--cost-db", type=Path, default=None, help="SQLite cost log path")
    args = parser.parse_args()

    if not args.events.exists():
        print(f"Missing events file: {args.events}")
        return
    if not args.predictions.exists():
        print(f"Missing predictions file: {args.predictions}")
        return

    print(run_dashboard(args.events, args.predictions, cost_db=args.cost_db))


if __name__ == "__main__":
    main()
