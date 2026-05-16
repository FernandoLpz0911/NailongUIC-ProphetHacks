from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from agent.config import ENSEMBLE_MODELS, MODEL_CALL_TIMEOUT_SECONDS, OPENROUTER_API_KEY
from agent.openrouter.client import OpenRouterClient, extract_message_text
from agent.openrouter.pricing import estimate_cost_usd
from agent.schemas import PredictRequest, Prediction
from forecasting.ensemble import weighted_average
from forecasting.forecaster import build_forecast_messages
from forecasting.parser import parse_with_retry
from retrieval.search import SearchDocument

logger = logging.getLogger(__name__)


@dataclass
class _ModelResult:
    model: str
    prediction: Prediction
    rationale: str
    latency_ms: float
    cost_usd: float
    usage: dict


async def _forecast_one(
    client: OpenRouterClient,
    messages: list[dict[str, str]],
    model: str,
) -> _ModelResult | None:
    started = time.perf_counter()
    try:
        payload = await asyncio.wait_for(
            client.chat(
                messages,
                model=model,
                temperature=0.3,
                max_tokens=2048,
                timeout=120.0,
            ),
            timeout=MODEL_CALL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("ensemble model %s timed out", model)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("ensemble model %s failed: %s", model, exc)
        return None

    latency_ms = (time.perf_counter() - started) * 1000.0
    text = extract_message_text(payload)
    try:
        prediction, rationale = await parse_with_retry(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ensemble model %s parse failed: %s", model, exc)
        return None

    model_used = str(payload.get("_model_used") or model)
    usage = payload.get("usage") or {}
    return _ModelResult(
        model=model_used,
        prediction=prediction,
        rationale=rationale,
        latency_ms=latency_ms,
        cost_usd=estimate_cost_usd(model_used, usage),
        usage=usage,
    )


async def ensemble_forecast(
    request: PredictRequest,
    context: list[SearchDocument],
) -> tuple[Prediction, str, dict]:
    """
    Run ENSEMBLE_MODELS in parallel, average successful predictions with equal weights.

    Returns (prediction, rationale, meta) where meta has model, latency_ms, cost_usd.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not configured")
    if not ENSEMBLE_MODELS:
        raise RuntimeError("ENSEMBLE_MODELS is empty")

    messages = build_forecast_messages(request, context)
    client = OpenRouterClient()
    started = time.perf_counter()

    results = await asyncio.gather(
        *[_forecast_one(client, messages, model) for model in ENSEMBLE_MODELS]
    )
    successes = [r for r in results if r is not None]
    if not successes:
        raise RuntimeError("all ensemble models failed")

    yes_vals = [r.prediction.YES for r in successes]
    spread = max(yes_vals) - min(yes_vals) if yes_vals else 0.0
    low_agreement = spread > 0.10

    prediction = weighted_average([r.prediction for r in successes])
    rationale_parts = [f"[{r.model}] {r.rationale}" for r in successes if r.rationale]
    rationale = "\n\n".join(rationale_parts) if rationale_parts else "Ensemble forecast."

    latency_ms = (time.perf_counter() - started) * 1000.0
    models_used = [r.model for r in successes]
    meta = {
        "model": f"ensemble:{','.join(models_used)}",
        "latency_ms": latency_ms,
        "cost_usd": sum(r.cost_usd for r in successes),
        "ensemble_models": models_used,
        "ensemble_count": len(successes),
        "ensemble_spread": spread,
        "low_agreement": low_agreement,
        "usage": {},
    }
    return prediction, rationale, meta
