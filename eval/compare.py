#!/usr/bin/env python3
"""Compare trading strategies: market_only, single model, ensemble."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx

from agent.config import CALIBRATION_ALPHA, DATA_DIR
from agent.schemas import MarketStat, Prediction
from eval.costs import cost_summary
from eval.harness import load_events
from eval.scoring import aggregate_scores
from forecasting.calibration import calibrate_vs_market, market_probability

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODES = ("market_only", "single", "ensemble")


def _load_predictions(
    path: Path | None,
    events_by_id: dict[str, dict[str, Any]],
    *,
    alpha: float | None = None,
) -> dict[str, Prediction]:
    if path is None or not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"predictions file must be a JSON list: {path}")
    out: dict[str, Prediction] = {}
    for row in rows:
        event_id = row["event_id"]
        event = events_by_id.get(event_id)
        if "raw_prediction" in row and event is not None:
            stats = {
                k: MarketStat.model_validate(v)
                for k, v in (event.get("market_stats") or {}).items()
            }
            raw = Prediction.model_validate(row["raw_prediction"])
            out[event_id] = calibrate_vs_market(raw, stats, alpha=alpha or CALIBRATION_ALPHA)
        else:
            out[event_id] = Prediction.model_validate(row["prediction"])
    return out


def market_only_predictions(events: list[dict[str, Any]]) -> dict[str, Prediction]:
    preds: dict[str, Prediction] = {}
    for event in events:
        stats = {
            k: MarketStat.model_validate(v) for k, v in (event.get("market_stats") or {}).items()
        }
        p = market_probability(stats, "Yes")
        preds[event["event_id"]] = Prediction(YES=p, NO=1.0 - p)
    return preds


async def _fetch_live(
    events: list[dict[str, Any]],
    *,
    agent_url: str,
    use_ensemble: bool,
) -> dict[str, Prediction]:
    preds: dict[str, Prediction] = {}
    headers: dict[str, str] = {}
    if not use_ensemble:
        headers["X-Use-Ensemble"] = "false"

    async with httpx.AsyncClient() as client:
        for event in events:
            payload = {
                "event_id": event["event_id"],
                "title": event["title"],
                "markets": event.get("markets") or ["Yes", "No"],
                "rules": event.get("rules") or "",
                "market_stats": event.get("market_stats") or {},
            }
            response = await client.post(
                f"{agent_url.rstrip('/')}/predict",
                json=payload,
                headers=headers or None,
                timeout=300.0,
            )
            response.raise_for_status()
            body = response.json()
            preds[event["event_id"]] = Prediction.model_validate(body["prediction"])
    return preds


def _cost_per_mode(mode: str, summary: dict[str, float | int]) -> float:
    if mode == "market_only":
        return 0.0
    return float(summary["avg_cost_per_event"])


def _format_table(rows: list[dict[str, Any]]) -> str:
    headers = ["Mode", "Brier", "Avg Return", "Est. cost/event"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {brier} | {avg_return} | {cost} |".format(
                mode=row["mode"],
                brier=row["brier"],
                avg_return=row["avg_return"],
                cost=row["cost"],
            )
        )
    return "\n".join(lines) + "\n"


def run_comparison(
    events_path: Path,
    *,
    predictions_single: Path | None = None,
    predictions_ensemble: Path | None = None,
    live: bool = False,
    agent_url: str = "http://localhost:8000",
    alpha: float | None = None,
) -> tuple[list[dict[str, Any]], str]:
    events = load_events(events_path)
    events_by_id = {e["event_id"]: e for e in events}
    summary = cost_summary()

    raw_single = _load_predictions(predictions_single, events_by_id, alpha=alpha)
    raw_ensemble = _load_predictions(predictions_ensemble, events_by_id, alpha=alpha)

    if live:
        logger.info("live mode: fetching predictions from %s", agent_url)
        raw_single = asyncio.run(_fetch_live(events, agent_url=agent_url, use_ensemble=False))
        raw_ensemble = asyncio.run(_fetch_live(events, agent_url=agent_url, use_ensemble=True))

    mode_preds: dict[str, dict[str, Prediction]] = {
        "market_only": market_only_predictions(events),
        "single": raw_single,
        "ensemble": raw_ensemble,
    }

    table_rows: list[dict[str, Any]] = []
    for mode in MODES:
        preds = mode_preds[mode]
        scores = aggregate_scores(events_by_id, preds)
        count = int(scores["count"])
        if count == 0:
            brier_s = "N/A"
            ret_s = "N/A"
        else:
            brier_s = f"{scores['brier']:.4f}"
            ret_s = f"{scores['avg_return']:.4f}"
        cost = _cost_per_mode(mode, summary)
        table_rows.append(
            {
                "mode": mode,
                "brier": brier_s,
                "avg_return": ret_s,
                "cost": f"${cost:.4f}" if mode != "market_only" else "$0.0000",
                "events_scored": count,
            }
        )

    markdown = "# Stage 3 strategy comparison\n\n"
    markdown += f"Events file: `{events_path}`\n\n"
    markdown += _format_table(table_rows)
    return table_rows, markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare market_only vs single vs ensemble")
    parser.add_argument("--events", type=Path, default=DATA_DIR / "events_test.json")
    parser.add_argument(
        "--predictions-single",
        type=Path,
        default=DATA_DIR / "predictions_single.json",
        help="Offline raw single-model predictions (calibrated in compare)",
    )
    parser.add_argument(
        "--predictions-ensemble",
        type=Path,
        default=DATA_DIR / "predictions_ensemble.json",
        help="Offline raw ensemble predictions (calibrated in compare)",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DATA_DIR / "submission.json",
        help="Fallback predictions file if mode-specific file missing",
    )
    parser.add_argument("--out", type=Path, default=DATA_DIR / "stage3_comparison.md")
    parser.add_argument("--live", action="store_true", help="Call agent /predict per event")
    parser.add_argument("--agent-url", default="http://localhost:8000")
    parser.add_argument("--alpha", type=float, default=None, help="Calibration alpha override")
    args = parser.parse_args()

    if not args.events.exists():
        print(f"Missing events file: {args.events}")
        return

    single_path = args.predictions_single if args.predictions_single.exists() else args.predictions
    ensemble_path = (
        args.predictions_ensemble if args.predictions_ensemble.exists() else args.predictions
    )

    rows, markdown = run_comparison(
        args.events,
        predictions_single=None if args.live else single_path,
        predictions_ensemble=None if args.live else ensemble_path,
        live=args.live,
        agent_url=args.agent_url,
        alpha=args.alpha,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {args.out}")
    for row in rows:
        if row["events_scored"] == 0 and row["mode"] != "market_only":
            print(
                f"  hint: no predictions for {row['mode']} — run backtest or pass --predictions / --live"
            )


if __name__ == "__main__":
    main()
