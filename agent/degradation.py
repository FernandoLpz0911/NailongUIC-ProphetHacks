from __future__ import annotations

import logging

from agent.config import (
    CHEAP_MODEL,
    DEFAULT_MODEL,
    EXPENSIVE_MODEL,
    FALLBACK_MODELS,
    USE_ENSEMBLE,
)
from agent.router import classify_event
from agent.schemas import PredictRequest, Prediction
from forecasting.calibration import market_probability
from forecasting.consistency import forecast_with_consistency
from forecasting.ensemble_forecast import ensemble_forecast
from forecasting.forecaster import forecast
from retrieval.search import SearchDocument

logger = logging.getLogger(__name__)


def degradation_model_chain(request: PredictRequest) -> list[str]:
    """Opus (hard) → Sonnet/default → cheap → configured fallbacks."""
    chain: list[str] = []
    if classify_event(request) == "hard":
        chain.append(EXPENSIVE_MODEL)
    for model in (DEFAULT_MODEL, CHEAP_MODEL, *FALLBACK_MODELS):
        if model and model not in chain:
            chain.append(model)
    return chain


def _market_fallback_raw(request: PredictRequest) -> tuple[Prediction, str, dict]:
    p = market_probability(request.market_stats, "Yes")
    return (
        Prediction(YES=p, NO=1.0 - p),
        "Market-anchored fallback prediction.",
        {
            "model": "market-fallback",
            "model_used": "market-fallback",
            "latency_ms": 0.0,
            "cost_usd": 0.0,
            "degraded": True,
            "low_agreement": False,
        },
    )


async def forecast_with_fallback(
    request: PredictRequest,
    context: list[SearchDocument],
) -> tuple[Prediction, str, dict]:
    """
    Graceful degradation: ensemble → routed consistency → model chain → market.
    """
    if USE_ENSEMBLE:
        try:
            return await ensemble_forecast(request, context)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ensemble forecast failed: %s", exc)

    for model in degradation_model_chain(request):
        try:
            return await forecast_with_consistency(request, context, model=model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("consistency forecast failed for %s: %s", model, exc)
        try:
            return await forecast(request, context, model=model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("forecast failed for %s: %s", model, exc)

    return _market_fallback_raw(request)
