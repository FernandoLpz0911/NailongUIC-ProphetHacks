#!/usr/bin/env python3
"""Grid-search calibration α per event category; write alpha_by_category.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent.config import CALIBRATION_ALPHA, DATA_DIR
from agent.schemas import MarketStat, Prediction
from eval.harness import load_events
from eval.scoring import aggregate_scores
from forecasting.calibration import calibrate_vs_market
from retrieval.category import CATEGORIES, Category, detect_category

DEFAULT_GRID = [round(x, 2) for x in [i / 10 for i in range(3, 9)]]  # 0.3 .. 0.8
DEFAULT_OUT = DATA_DIR / "alpha_by_category.json"


def load_raw_predictions(
    path: Path,
    events_by_id: dict[str, dict[str, Any]],
) -> dict[str, Prediction]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"predictions file must be a JSON list: {path}")
    out: dict[str, Prediction] = {}
    for row in rows:
        event_id = row["event_id"]
        event = events_by_id.get(event_id)
        if event is None:
            continue
        if "raw_prediction" in row:
            out[event_id] = Prediction.model_validate(row["raw_prediction"])
        else:
            out[event_id] = Prediction.model_validate(row["prediction"])
    return out


def events_by_category(events: list[dict[str, Any]]) -> dict[Category, list[str]]:
    grouped: dict[Category, list[str]] = {cat: [] for cat in CATEGORIES}
    for event in events:
        cat = detect_category(event.get("title", ""), event.get("rules", ""))
        grouped.setdefault(cat, []).append(event["event_id"])
    return grouped


def tune_alpha_for_category(
    event_ids: list[str],
    events_by_id: dict[str, dict[str, Any]],
    raw_by_id: dict[str, Prediction],
    *,
    grid: list[float],
) -> float:
    if not event_ids:
        return CALIBRATION_ALPHA

    best_alpha = CALIBRATION_ALPHA
    best_return = float("-inf")

    for alpha in grid:
        preds: dict[str, Prediction] = {}
        for event_id in event_ids:
            raw = raw_by_id.get(event_id)
            event = events_by_id.get(event_id)
            if raw is None or event is None:
                continue
            stats = {
                k: MarketStat.model_validate(v)
                for k, v in (event.get("market_stats") or {}).items()
            }
            preds[event_id] = calibrate_vs_market(raw, stats, alpha=alpha)

        scores = aggregate_scores(events_by_id, preds)
        avg_return = float(scores["avg_return"])
        if avg_return > best_return:
            best_return = avg_return
            best_alpha = alpha

    return best_alpha


def tune_all(
    events_path: Path,
    predictions_path: Path,
    *,
    grid: list[float] | None = None,
) -> dict[str, float]:
    events = load_events(events_path)
    events_by_id = {e["event_id"]: e for e in events}
    raw_by_id = load_raw_predictions(predictions_path, events_by_id)
    grouped = events_by_category(events)
    alphas = grid or DEFAULT_GRID

    result: dict[str, float] = {}
    for category in CATEGORIES:
        ids = [eid for eid in grouped.get(category, []) if eid in raw_by_id]
        result[category] = tune_alpha_for_category(
            ids,
            events_by_id,
            raw_by_id,
            grid=alphas,
        )
    return result


def write_alpha_config(path: Path, alphas: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(alphas, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune per-category calibration α")
    parser.add_argument("--events", type=Path, default=DATA_DIR / "events_test.json")
    parser.add_argument("--predictions", type=Path, default=DATA_DIR / "submission.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--grid",
        type=float,
        nargs="+",
        default=None,
        help="Alpha values to search (default 0.3–0.8 step 0.1)",
    )
    args = parser.parse_args()

    if not args.events.exists():
        print(f"Missing events file: {args.events}")
        return
    if not args.predictions.exists():
        print(f"Missing predictions file: {args.predictions}")
        return

    alphas = tune_all(args.events, args.predictions, grid=args.grid)
    write_alpha_config(args.out, alphas)
    print(f"Wrote {args.out}")
    for category, value in alphas.items():
        print(f"  {category}: {value:.2f}")


if __name__ == "__main__":
    main()
