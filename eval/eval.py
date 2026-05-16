#!/usr/bin/env python3
"""CLI: load events, score predictions (Brier + avg return proxy)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from agent.config import DATA_DIR
from agent.schemas import Prediction
from eval.harness import load_events
from eval.scoring import aggregate_scores, outcome_yes, score_one

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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

    preds_rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions_by_id: dict[str, Prediction] = {}
    for row in preds_rows:
        event_id = row["event_id"]
        event = events.get(event_id)
        if not event:
            continue
        pred = Prediction.model_validate(row["prediction"])
        predictions_by_id[event_id] = pred
        brier, ret = score_one(pred, event)
        logger.info(
            "event_id=%s brier=%.4f return=%.4f outcome_yes=%s",
            event_id,
            brier,
            ret,
            outcome_yes(event),
        )

    scores = aggregate_scores(events, predictions_by_id)
    count = int(scores["count"])
    if count == 0:
        print("No overlapping event_ids between events and predictions.")
        return

    print(f"events_scored={count}")
    print(f"brier={scores['brier']:.4f}")
    print(f"avg_return_proxy={scores['avg_return']:.4f}")


if __name__ == "__main__":
    main()
