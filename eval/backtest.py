#!/usr/bin/env python3
"""Generate predictions by calling the agent for each event."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import httpx

from agent.config import DATA_DIR
from eval.harness import load_events, save_submission

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def _predict_one(
    client: httpx.AsyncClient,
    agent_url: str,
    event: dict,
) -> dict:
    payload = {
        "event_id": event["event_id"],
        "title": event["title"],
        "markets": event.get("markets") or ["Yes", "No"],
        "rules": event.get("rules") or "",
        "market_stats": event.get("market_stats") or {},
    }
    response = await client.post(f"{agent_url.rstrip('/')}/predict", json=payload, timeout=300.0)
    response.raise_for_status()
    body = response.json()
    return {
        "event_id": body["event_id"],
        "prediction": body["prediction"],
        "rationale": body.get("rationale", ""),
    }


async def run_backtest(
    events_path: Path,
    output_path: Path,
    *,
    agent_url: str = "http://localhost:8000",
    limit: int | None = None,
    concurrency: int = 3,
) -> None:
    events = load_events(events_path)
    if limit is not None:
        events = events[:limit]

    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async with httpx.AsyncClient() as client:

        async def worker(event: dict) -> None:
            async with semaphore:
                try:
                    row = await _predict_one(client, agent_url, event)
                    results.append(row)
                    logger.info("ok %s YES=%.3f", event["event_id"], row["prediction"]["YES"])
                except Exception as exc:  # noqa: BLE001
                    logger.error("fail %s: %s", event.get("event_id"), exc)

        await asyncio.gather(*[worker(e) for e in events])

        try:
            stats = await client.get(f"{agent_url.rstrip('/')}/stats")
            if stats.status_code == 200:
                logger.info("agent total_spend_usd=%.4f", stats.json().get("total_spend_usd", 0))
        except Exception:  # noqa: BLE001
            pass

    save_submission(output_path, results)
    logger.info("wrote %d predictions to %s", len(results), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest: call /predict for each event")
    parser.add_argument("--events", type=Path, default=DATA_DIR / "events.json")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "submission.json")
    parser.add_argument("--agent-url", default="http://localhost:8000")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    if not args.events.exists():
        print(f"Missing {args.events}")
        print("  Run: bash scripts/fetch_events.sh   (needs: pip install ai-prophet)")
        print("  Or use: --events data/sample_event.json --limit 1")
        return

    asyncio.run(
        run_backtest(
            args.events,
            args.out,
            agent_url=args.agent_url,
            limit=args.limit,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
