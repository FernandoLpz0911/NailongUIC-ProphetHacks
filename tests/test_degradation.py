from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agent.degradation import forecast_with_fallback
from agent.schemas import MarketStat, PredictRequest, Prediction
from forecasting.calibration import market_probability


def _sample_request() -> PredictRequest:
    return PredictRequest(
        event_id="evt-deg",
        title="Will it rain?",
        markets=["Yes", "No"],
        rules="Resolves Yes if measurable rain in NYC.",
        market_stats={"Yes": MarketStat(last_price=0.62, yes_ask=0.64, no_ask=0.38)},
    )


@pytest.mark.asyncio
async def test_fallback_returns_primary_on_success() -> None:
    expected = (
        Prediction(YES=0.7, NO=0.3),
        "Primary ok.",
        {"model": "primary", "latency_ms": 1.0, "cost_usd": 0.01},
    )
    with (
        patch("agent.degradation.USE_ENSEMBLE", False),
        patch("agent.degradation.forecast_with_consistency", new_callable=AsyncMock, return_value=expected),
        patch("agent.degradation.ensemble_forecast", new_callable=AsyncMock),
    ):
        prediction, rationale, meta = await forecast_with_fallback(_sample_request(), [])

    assert prediction.YES == 0.7
    assert rationale == "Primary ok."
    assert meta["model"] == "primary"


@pytest.mark.asyncio
async def test_fallback_uses_cheap_model_after_primary_failure() -> None:
    cheap = (
        Prediction(YES=0.55, NO=0.45),
        "Cheap model.",
        {"model": "cheap", "latency_ms": 2.0, "cost_usd": 0.001},
    )
    with (
        patch("agent.degradation.USE_ENSEMBLE", True),
        patch(
            "agent.degradation.ensemble_forecast",
            new_callable=AsyncMock,
            side_effect=RuntimeError("ensemble down"),
        ),
        patch(
            "agent.degradation.degradation_model_chain",
            return_value=["cheap"],
        ),
        patch(
            "agent.degradation.forecast_with_consistency",
            new_callable=AsyncMock,
            side_effect=RuntimeError("consistency down"),
        ),
        patch("agent.degradation.forecast", new_callable=AsyncMock, return_value=cheap) as mock_forecast,
    ):
        prediction, rationale, meta = await forecast_with_fallback(_sample_request(), [])

    mock_forecast.assert_awaited_once()
    assert prediction.YES == 0.55
    assert meta["model"] == "cheap"


@pytest.mark.asyncio
async def test_fallback_market_when_all_models_fail() -> None:
    request = _sample_request()
    mock_consistency = AsyncMock(side_effect=RuntimeError("api down"))
    mock_forecast = AsyncMock(side_effect=RuntimeError("api down"))
    with (
        patch("agent.degradation.USE_ENSEMBLE", False),
        patch("agent.degradation.forecast_with_consistency", mock_consistency),
        patch("agent.degradation.forecast", mock_forecast),
        patch(
            "agent.degradation.degradation_model_chain",
            return_value=["model-a", "model-b"],
        ),
    ):
        prediction, rationale, meta = await forecast_with_fallback(request, [])

    assert mock_consistency.await_count == 2
    assert mock_forecast.await_count == 2
    assert meta["model"] == "market-fallback"
    p_market = market_probability(request.market_stats, "Yes")
    assert prediction.YES == pytest.approx(p_market)
    assert meta.get("degraded") is True
    assert "fallback" in rationale.lower()
