from __future__ import annotations

from typing import Any

from agent.schemas import MarketStat, Prediction
from eval.metrics import brier_score
from eval.simulator import simulate_return


def outcome_yes(event: dict[str, Any]) -> bool:
    return bool(event.get("outcome_yes") or event.get("resolved_yes"))


def parse_market_stats(event: dict[str, Any]) -> dict[str, MarketStat]:
    return {
        k: MarketStat.model_validate(v) for k, v in (event.get("market_stats") or {}).items()
    }


def score_one(prediction: Prediction, event: dict[str, Any]) -> tuple[float, float]:
    """Return (brier, return_proxy) for one event."""
    stats = parse_market_stats(event)
    yes = outcome_yes(event)
    brier = brier_score(prediction.YES, yes)
    ret = simulate_return(prediction, stats, outcome_yes=yes)
    return brier, ret


def aggregate_scores(
    events_by_id: dict[str, dict[str, Any]],
    predictions_by_id: dict[str, Prediction],
) -> dict[str, float | int]:
    brier_total = 0.0
    return_total = 0.0
    count = 0
    for event_id, pred in predictions_by_id.items():
        event = events_by_id.get(event_id)
        if event is None:
            continue
        brier, ret = score_one(pred, event)
        brier_total += brier
        return_total += ret
        count += 1
    if count == 0:
        return {"count": 0, "brier": 0.0, "avg_return": 0.0}
    return {
        "count": count,
        "brier": brier_total / count,
        "avg_return": return_total / count,
    }
