#!/usr/bin/env python3
"""CLI: load events, score predictions (Brier + avg return proxy)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.config import DATA_DIR
from agent.schemas import MarketStat, Prediction
from eval.harness import load_events
from eval.metrics import brier_score
from eval.simulator import simulate_return


def main() -> None:
    parser = argparse.ArgumentParser(description="Prophet Hacks local evaluation harness")
    parser.add_argument(
        "--events",
        type=Path,
        default=DATA_DIR / "events_test.json",
        help="Resolved events with outcomes for scoring",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DATA_DIR / "submission.json",
        help="Predictions file (list of {event_id, prediction})",
    )
    args = parser.parse_args()

    if not args.events.exists():
        print(f"No events file at {args.events} — run prophet forecast events first (P4).")
        return

    events = {e["event_id"]: e for e in load_events(args.events)}
    if not args.predictions.exists():
        print(f"No predictions at {args.predictions} — generate via harness after /predict runs.")
        return

    preds = json.loads(args.predictions.read_text(encoding="utf-8"))
    brier_total = 0.0
    return_total = 0.0
    count = 0

    for row in preds:
        event_id = row["event_id"]
        event = events.get(event_id)
        if not event:
            continue
        outcome_yes = bool(event.get("outcome_yes") or event.get("resolved_yes"))
        pred = Prediction.model_validate(row["prediction"])
        stats = {
            k: MarketStat.model_validate(v) for k, v in (event.get("market_stats") or {}).items()
        }
        brier_total += brier_score(pred.YES, outcome_yes)
        return_total += simulate_return(pred, stats, outcome_yes=outcome_yes)
        count += 1

    if count == 0:
        print("No overlapping event_ids between events and predictions.")
        return

    print(f"events_scored={count}")
    print(f"brier={brier_total / count:.4f}")
    print(f"avg_return_proxy={return_total / count:.4f}")


if __name__ == "__main__":
    main()
