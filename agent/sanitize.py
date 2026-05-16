from __future__ import annotations

import math

from agent.schemas import MarketStat, PredictResponse, Prediction
from forecasting.calibration import market_probability


def sanitize_prediction(
    prediction: Prediction,
    rationale: str,
    market_stats: dict[str, MarketStat],
    *,
    max_edge: float = 0.30,
) -> tuple[Prediction, str]:
    """Clamp probabilities, cap edge vs market, trim rationale."""
    yes = prediction.YES
    if math.isnan(yes) or math.isinf(yes):
        yes = market_probability(market_stats, "Yes")

    yes = max(0.01, min(0.99, yes))
    p_market = market_probability(market_stats, "Yes")
    if abs(yes - p_market) > max_edge:
        sign = 1.0 if yes > p_market else -1.0
        yes = p_market + sign * max_edge

    yes = max(0.01, min(0.99, yes))
    clean = Prediction(YES=yes, NO=1.0 - yes)
    text = (rationale or "").strip()[:500]
    return clean, text


def market_fallback_response(
    event_id: str,
    market_stats: dict[str, MarketStat],
    reason: str,
) -> PredictResponse:
    p = market_probability(market_stats, "Yes")
    return PredictResponse(
        event_id=event_id,
        prediction=Prediction(YES=p, NO=1.0 - p),
        rationale=reason[:500],
    )
