from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.schemas import PredictRequest, Prediction
from forecasting.ensemble import weighted_average
from forecasting.ensemble_forecast import ensemble_forecast
def _sample_request() -> PredictRequest:
    return PredictRequest(
        event_id="evt-1",
        title="Test event",
        markets=["Yes", "No"],
        rules="Resolves Yes if condition holds.",
        market_stats={},
    )


def _mock_chat_payload(model: str, yes: float, rationale: str) -> dict:
    content = json.dumps(
        {
            "prediction": {"YES": yes, "NO": 1.0 - yes},
            "rationale": rationale,
        }
    )
    return {
        "_model_used": model,
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


@pytest.mark.asyncio
async def test_ensemble_weighted_average_three_models() -> None:
    models = ["model-a", "model-b", "model-c"]
    yes_by_model = {"model-a": 0.6, "model-b": 0.7, "model-c": 0.8}
    chat_lock = asyncio.Lock()

    async def fake_chat(_messages, *, model=None, **_kwargs):
        async with chat_lock:
            yes = yes_by_model[model]
            return _mock_chat_payload(model, yes, f"rationale-{model}")

    client = MagicMock()
    client.chat = fake_chat

    with (
        patch("forecasting.ensemble_forecast.ENSEMBLE_MODELS", models),
        patch("forecasting.ensemble_forecast.OPENROUTER_API_KEY", "test-key"),
        patch("forecasting.parser.OPENROUTER_API_KEY", ""),
        patch("forecasting.ensemble_forecast.OpenRouterClient", return_value=client),
    ):
        prediction, rationale, meta = await ensemble_forecast(_sample_request(), [])

    assert abs(prediction.YES - 0.7) < 1e-6
    assert "model-a" in rationale and "model-c" in rationale
    assert meta["ensemble_count"] == 3
    assert meta["cost_usd"] >= 0
    assert meta["model"].startswith("ensemble:")
    assert meta["low_agreement"] is True
    assert meta["ensemble_spread"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_ensemble_uses_asyncio_gather_parallel() -> None:
    models = ["m1", "m2", "m3"]
    gather_calls: list[int] = []

    real_gather = asyncio.gather

    async def tracking_gather(*coros, **kwargs):
        gather_calls.append(len(coros))
        return await real_gather(*coros, **kwargs)

    chat_lock = asyncio.Lock()
    chat_calls = 0

    async def fake_chat(_messages, *, model=None, **_kwargs):
        nonlocal chat_calls
        async with chat_lock:
            chat_calls += 1
            return _mock_chat_payload(model, 0.5, "ok")

    client = MagicMock()
    client.chat = fake_chat

    with (
        patch("forecasting.ensemble_forecast.asyncio.gather", side_effect=tracking_gather),
        patch("forecasting.ensemble_forecast.ENSEMBLE_MODELS", models),
        patch("forecasting.ensemble_forecast.OPENROUTER_API_KEY", "test-key"),
        patch("forecasting.parser.OPENROUTER_API_KEY", ""),
        patch("forecasting.ensemble_forecast.OpenRouterClient", return_value=client),
    ):
        await ensemble_forecast(_sample_request(), [])

    assert gather_calls == [3]
    assert chat_calls == 3


@pytest.mark.asyncio
async def test_ensemble_tolerates_one_model_failure() -> None:
    models = ["good", "bad", "good2"]

    chat_lock = asyncio.Lock()

    async def fake_chat(_messages, *, model=None, **_kwargs):
        async with chat_lock:
            if model == "bad":
                raise RuntimeError("upstream error")
            yes = 0.4 if model == "good" else 0.8
            return _mock_chat_payload(model, yes, model)

    client = MagicMock()
    client.chat = fake_chat

    with (
        patch("forecasting.ensemble_forecast.ENSEMBLE_MODELS", models),
        patch("forecasting.ensemble_forecast.OPENROUTER_API_KEY", "test-key"),
        patch("forecasting.parser.OPENROUTER_API_KEY", ""),
        patch("forecasting.ensemble_forecast.OpenRouterClient", return_value=client),
    ):
        prediction, _, meta = await ensemble_forecast(_sample_request(), [])

    assert abs(prediction.YES - 0.6) < 1e-6
    assert meta["ensemble_count"] == 2


@pytest.mark.asyncio
async def test_ensemble_all_fail_raises() -> None:
    client = MagicMock()
    client.chat = AsyncMock(side_effect=RuntimeError("down"))

    with (
        patch("forecasting.ensemble_forecast.ENSEMBLE_MODELS", ["a", "b"]),
        patch("forecasting.ensemble_forecast.OPENROUTER_API_KEY", "test-key"),
        patch("forecasting.parser.OPENROUTER_API_KEY", ""),
        patch("forecasting.ensemble_forecast.OpenRouterClient", return_value=client),
    ):
        with pytest.raises(RuntimeError, match="all ensemble models failed"):
            await ensemble_forecast(_sample_request(), [])


@pytest.mark.asyncio
async def test_ensemble_flags_low_agreement_on_spread() -> None:
    models = ["m1", "m2"]
    chat_lock = asyncio.Lock()

    async def fake_chat(_messages, *, model=None, **_kwargs):
        async with chat_lock:
            yes = 0.2 if model == "m1" else 0.8
            return _mock_chat_payload(model, yes, "r")

    client = MagicMock()
    client.chat = fake_chat

    with (
        patch("forecasting.ensemble_forecast.ENSEMBLE_MODELS", models),
        patch("forecasting.ensemble_forecast.OPENROUTER_API_KEY", "test-key"),
        patch("forecasting.parser.OPENROUTER_API_KEY", ""),
        patch("forecasting.ensemble_forecast.OpenRouterClient", return_value=client),
    ):
        _, _, meta = await ensemble_forecast(_sample_request(), [])

    assert meta["low_agreement"] is True
    assert meta["ensemble_spread"] > 0.10


def test_weighted_average_equal_weights() -> None:
    preds = [
        Prediction(YES=0.2, NO=0.8),
        Prediction(YES=0.5, NO=0.5),
        Prediction(YES=0.8, NO=0.2),
    ]
    result = weighted_average(preds)
    assert abs(result.YES - 0.5) < 1e-6
