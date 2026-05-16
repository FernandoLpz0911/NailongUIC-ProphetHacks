from __future__ import annotations

from agent.schemas import PredictRequest, Prediction
from forecasting.forecaster import forecast
from retrieval.search import SearchDocument

AGREEMENT_THRESHOLD = 0.10
TEMP_LOW = 0.2
TEMP_HIGH = 0.5


async def forecast_with_consistency(
    request: PredictRequest,
    context: list[SearchDocument],
    *,
    model: str | None = None,
) -> tuple[Prediction, str, dict]:
    """
    Run forecaster at two temperatures; flag low_agreement when |p1 - p2| > 0.10.

    Returns averaged prediction, primary rationale, and meta (includes low_agreement).
    """
    pred_low, rationale_low, meta_low = await forecast(
        request, context, model=model, temperature=TEMP_LOW
    )
    pred_high, rationale_high, meta_high = await forecast(
        request, context, model=model, temperature=TEMP_HIGH
    )

    low_agreement = abs(pred_low.YES - pred_high.YES) > AGREEMENT_THRESHOLD
    yes = (pred_low.YES + pred_high.YES) / 2.0
    prediction = Prediction(YES=yes, NO=1.0 - yes)
    rationale = rationale_low or rationale_high

    meta = {
        "model": meta_low.get("model") or meta_high.get("model"),
        "model_used": meta_low.get("model_used") or meta_high.get("model_used"),
        "tier": meta_low.get("tier") or meta_high.get("tier"),
        "latency_ms": meta_low.get("latency_ms", 0.0) + meta_high.get("latency_ms", 0.0),
        "cost_usd": meta_low.get("cost_usd", 0.0) + meta_high.get("cost_usd", 0.0),
        "usage": {},
        "low_agreement": low_agreement,
        "consistency_yes_low": pred_low.YES,
        "consistency_yes_high": pred_high.YES,
    }
    return prediction, rationale, meta
