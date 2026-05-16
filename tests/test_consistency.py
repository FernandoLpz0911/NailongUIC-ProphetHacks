from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.openrouter.client import OpenRouterClient
from agent.schemas import PredictRequest
from forecasting.consistency import forecast_with_consistency


def _sample_request() -> PredictRequest:
    return PredictRequest(
        event_id="evt-1",
        title="Test event",
        markets=["Yes", "No"],
        rules="Resolves Yes if condition holds by March 31, 2026.",
        market_stats={},
    )


def _mock_chat_payload(yes: float) -> dict:
    content = json.dumps(
        {
            "prediction": {"YES": yes, "NO": 1.0 - yes},
            "rationale": "ok",
        }
    )
    return {
        "_model_used": "test-model",
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


@pytest.mark.asyncio
async def test_consistency_high_agreement() -> None:
    temps: list[float] = []

    async def fake_chat(_messages, *, model=None, temperature=0.3, **_kwargs):
        temps.append(temperature)
        return _mock_chat_payload(0.55)

    client = MagicMock()
    client.chat = fake_chat

    with (
        patch("forecasting.forecaster.OPENROUTER_API_KEY", "test-key"),
        patch("forecasting.parser.OPENROUTER_API_KEY", ""),
        patch("forecasting.forecaster.OpenRouterClient") as mock_cls,
    ):
        mock_cls.return_value = client
        mock_cls.extract_text = OpenRouterClient.extract_text
        prediction, _, meta = await forecast_with_consistency(_sample_request(), [])

    assert 0.2 in temps and 0.5 in temps
    assert abs(prediction.YES - 0.55) < 1e-6
    assert meta["low_agreement"] is False


@pytest.mark.asyncio
async def test_consistency_low_agreement() -> None:
    call = 0

    async def fake_chat(_messages, *, model=None, temperature=0.3, **_kwargs):
        nonlocal call
        call += 1
        yes = 0.30 if temperature == 0.2 else 0.75
        return _mock_chat_payload(yes)

    client = MagicMock()
    client.chat = fake_chat

    with (
        patch("forecasting.forecaster.OPENROUTER_API_KEY", "test-key"),
        patch("forecasting.parser.OPENROUTER_API_KEY", ""),
        patch("forecasting.forecaster.OpenRouterClient") as mock_cls,
    ):
        mock_cls.return_value = client
        mock_cls.extract_text = OpenRouterClient.extract_text
        prediction, _, meta = await forecast_with_consistency(_sample_request(), [])

    assert call == 2
    assert abs(prediction.YES - 0.525) < 1e-6
    assert meta["low_agreement"] is True
