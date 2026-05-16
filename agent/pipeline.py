from __future__ import annotations

import asyncio
import logging
import time

from agent.config import MAX_EDGE_DEVIATION, OPENROUTER_API_KEY, PREDICT_TIMEOUT_SECONDS
from agent.degradation import forecast_with_fallback
from agent.openrouter.cost_tracker import CostRecord
from agent.sanitize import market_fallback_response, sanitize_prediction
from agent.schemas import PredictRequest, PredictResponse
from agent.services import get_cost_tracker
from eval.config_loader import alpha_for_event
from forecasting.calibration import calibrate_vs_market
from forecasting.guards import guard_flags
from retrieval.context import build_context

logger = logging.getLogger(__name__)


def _calibration_alpha(
    request: PredictRequest,
    used_live_search: bool,
    num_docs: int,
    *,
    low_agreement: bool = False,
) -> float:
    """Blend model vs market; α ∈ [0.3, 0.8], tuned per category when configured."""
    category_alpha = alpha_for_event(
        {"title": request.title, "rules": request.rules},
    )
    if not used_live_search:
        raw = min(category_alpha, 0.4)
    else:
        raw = max(category_alpha, 0.35 + 0.05 * min(num_docs, 6))
    if low_agreement:
        raw -= 0.15
    return max(0.3, min(0.8, raw))


async def run_predict_pipeline(request: PredictRequest) -> PredictResponse:
    started = time.perf_counter()

    try:
        return await asyncio.wait_for(
            _run_pipeline(request),
            timeout=PREDICT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error("predict timeout event_id=%s", request.event_id)
        return market_fallback_response(
            request.event_id,
            request.market_stats,
            "Timed out — returned market-anchored prediction.",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("predict failed event_id=%s: %s", request.event_id, exc)
        return market_fallback_response(
            request.event_id,
            request.market_stats,
            f"Error fallback: {exc}",
        )
    finally:
        elapsed = (time.perf_counter() - started) * 1000.0
        total_cost = get_cost_tracker().total_spend()
        logger.info(
            "predict_done event_id=%s latency_ms=%.0f total_spend_usd=%.4f",
            request.event_id,
            elapsed,
            total_cost,
        )


async def _run_pipeline(request: PredictRequest) -> PredictResponse:
    context, used_live = await build_context(request)
    flags = guard_flags(request, context)

    if flags.force_market:
        return market_fallback_response(
            request.event_id,
            request.market_stats,
            "Ambiguous event — market-anchored (thin rules or retrieval).",
        )

    if not OPENROUTER_API_KEY:
        return market_fallback_response(
            request.event_id,
            request.market_stats,
            "Demo mode: set OPENROUTER_API_KEY for model forecasts.",
        )

    raw, rationale, meta = await forecast_with_fallback(request, context)
    low_agreement = bool(meta.get("low_agreement"))
    alpha = _calibration_alpha(
        request,
        used_live,
        len(context),
        low_agreement=low_agreement,
    )
    calibrated = calibrate_vs_market(raw, request.market_stats, alpha=alpha)
    final, clean_rationale = sanitize_prediction(
        calibrated,
        rationale,
        request.market_stats,
        max_edge=MAX_EDGE_DEVIATION,
    )

    get_cost_tracker().log(
        CostRecord(
            event_id=request.event_id,
            model=meta["model"],
            latency_ms=meta["latency_ms"],
            cost_usd=meta["cost_usd"],
        )
    )
    logger.info(
        "predict event_id=%s model_used=%s tier=%s cost_usd=%.5f alpha=%.2f docs=%d low_agreement=%s",
        request.event_id,
        meta.get("model_used", meta["model"]),
        meta.get("tier", "?"),
        meta["cost_usd"],
        alpha,
        len(context),
        low_agreement,
    )

    return PredictResponse(
        event_id=request.event_id,
        prediction=final,
        rationale=clean_rationale or "Forecast blended with market prices for trading track calibration.",
    )
