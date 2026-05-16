from unittest.mock import AsyncMock, patch

import pytest

from agent.schemas import Prediction
from forecasting.calibration import market_probability


def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_demo_mode_market_anchored(client, sample_event) -> None:
    """Without OPENROUTER_API_KEY, agent returns market-anchored probabilities."""
    response = client.post("/predict", json=sample_event)
    assert response.status_code == 200
    body = response.json()
    assert body["event_id"] == sample_event["event_id"]
    assert abs(body["prediction"]["YES"] + body["prediction"]["NO"] - 1.0) < 1e-6
    assert "rationale" in body
    # Sample event Yes market ~0.72
    assert body["prediction"]["YES"] > 0.65


@patch("agent.pipeline.forecast_with_fallback", new_callable=AsyncMock)
@patch("agent.pipeline.build_context", new_callable=AsyncMock)
@patch("agent.pipeline.OPENROUTER_API_KEY", "test-key")
def test_predict_with_mocked_llm(mock_context, mock_forecast, client, sample_event) -> None:
    from retrieval.search import SearchDocument

    mock_context.return_value = (
        [SearchDocument(title="News", url="https://reuters.com/a", snippet="Election likely.")],
        True,
    )
    mock_forecast.return_value = (
        Prediction(YES=0.8, NO=0.2),
        "Strong polling signal.",
        {"model": "test", "latency_ms": 10.0, "cost_usd": 0.001},
    )

    response = client.post("/predict", json=sample_event)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 < body["prediction"]["YES"] < 1.0
    assert body["rationale"]
