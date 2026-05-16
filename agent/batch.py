from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from agent.logging_utils import setup_logging
from agent.pipeline import run_predict_pipeline
from agent.schemas import PredictRequest, PredictResponse, normalize_market_keys

logger = logging.getLogger(__name__)


def event_to_request(raw: dict) -> PredictRequest:
    return PredictRequest(
        event_id=raw["event_id"],
        title=raw["title"],
        markets=raw["markets"],
        rules=raw["rules"],
        market_stats=normalize_market_keys(raw.get("market_stats", {})),
    )


async def predict_many(
    events: list[dict],
    *,
    concurrency: int = 5,
) -> list[PredictResponse]:
    semaphore = asyncio.Semaphore(concurrency)

    async def _predict_one(raw: dict) -> PredictResponse:
        async with semaphore:
            request = event_to_request(raw)
            return await run_predict_pipeline(request)

    return list(await asyncio.gather(*[_predict_one(event) for event in events]))


def _load_events(path: Path, limit: int | None) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    if limit is not None:
        return data[:limit]
    return data


async def _cli_main(events_path: Path, limit: int | None, concurrency: int) -> None:
    events = _load_events(events_path, limit)
    logger.info("batch_start count=%d concurrency=%d", len(events), concurrency)
    results = await predict_many(events, concurrency=concurrency)
    for response in results:
        print(
            json.dumps(
                {
                    "event_id": response.event_id,
                    "prediction": response.prediction.model_dump(),
                    "rationale": response.rationale[:120],
                }
            )
        )
    logger.info("batch_done count=%d", len(results))


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Run predict pipeline on many events.")
    parser.add_argument("--events", type=Path, required=True, help="JSON file with event list")
    parser.add_argument("--limit", type=int, default=None, help="Max events to process")
    parser.add_argument("--concurrency", type=int, default=5, help="Parallel predict limit")
    args = parser.parse_args()
    asyncio.run(_cli_main(args.events, args.limit, args.concurrency))


if __name__ == "__main__":
    main()
