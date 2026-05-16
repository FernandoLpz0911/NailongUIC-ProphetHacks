"""ForecastStage subclass that blends p_model with market consensus.

We override `_generate_forecast` to:
  1. Run the SDK's LLM forecast exactly as before to get p_model.
  2. Pull the Kalshi mid from `tick_ctx.candidates[market_id].yes_mark`.
  3. Pull a Polymarket second-market signal (cached per process for the
     tick lifetime so we don't re-hit the API).
  4. Determine retrieval confidence from the search summary; map to alpha.
  5. Return `{p_yes: p_final, rationale: ...}` so the existing SDK
     SchemaValidator and downstream stages don't notice a difference.

The blend math lives in `agent.calibration` and is unit-tested in isolation.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ai_prophet.trade.agent.stages.forecast import ForecastStage
from ai_prophet.trade.core import TickContext
from ai_prophet.trade.llm import LLMClient

from agent.calibration import calibrate, market_mid
from agent.settings import CalibrationConfig
from agent.spend import is_killed
from retrieval.retrieval import fetch_polymarket

logger = logging.getLogger(__name__)

# Process-level Polymarket cache. Cleared lazily once entries are older than
# `_POLY_CACHE_TTL_SEC`. Keyed on the candidate question text.
_POLY_CACHE: dict[str, tuple[float, dict | None]] = {}
_POLY_CACHE_TTL_SEC = 60 * 15  # ~1 tick


def _cached_polymarket(question: str) -> dict | None:
    """Per-process Polymarket lookup with a 1-tick TTL.

    Why: the SDK runs the forecast stage market-by-market within a tick.
    Without caching we'd hit Polymarket's API once per candidate, which
    can rate-limit the run for marginal extra signal.
    """
    if not question:
        return None
    now = time.monotonic()
    cached = _POLY_CACHE.get(question)
    if cached and now - cached[0] < _POLY_CACHE_TTL_SEC:
        return cached[1]
    value = fetch_polymarket(question)
    _POLY_CACHE[question] = (now, value)
    return value


def _confidence_from_summary(summary: dict[str, Any]) -> str:
    """Estimate retrieval confidence from the SDK search summary shape.

    The SDK's SearchStage produces `summary["key_points"]` (list of strings)
    and `summary["open_questions"]` (list). Our retrieval module also fills
    `summary["sources"]` when available. We map these to one of high/medium/low.

    Heuristic: lots of concrete key points + few open questions => high.
    """
    key_points = summary.get("key_points") or []
    open_qs = summary.get("open_questions") or []
    if len(key_points) >= 4 and len(open_qs) <= 1:
        return "high"
    if len(key_points) >= 2:
        return "medium"
    return "low"


class CalibratedForecastStage(ForecastStage):
    """ForecastStage that anchors `p_yes` to market consensus."""

    def __init__(
        self,
        llm_client: LLMClient,
        calibration: CalibrationConfig | None = None,
    ) -> None:
        super().__init__(llm_client=llm_client)
        # Late import so tests can construct with no calibration override.
        if calibration is None:
            from agent.settings import CalibrationConfig as _CC
            calibration = _CC()
        self.calibration = calibration

    def _generate_forecast(
        self,
        market_id: str,
        summary: dict[str, Any],
        tick_ctx: TickContext,
    ) -> dict[str, Any]:
        # 0. Budget kill-switch: skip LLM entirely and anchor to market.
        if is_killed():
            market_info = tick_ctx.get_candidate(market_id)
            p_market = market_mid(
                market_info.yes_bid if market_info else 0.5,
                market_info.yes_ask if market_info else 0.5,
            ) or 0.5
            logger.warning(
                "Forecast %s: KILL SWITCH active, returning p_yes=p_market=%.3f",
                market_id, p_market,
            )
            return {
                "p_yes":     p_market,
                "rationale": "[KILL_SWITCH] Budget exhausted; anchored to market mid.",
            }

        # 1. Run the SDK LLM forecast.
        raw = super()._generate_forecast(market_id, summary, tick_ctx)
        try:
            p_model = float(raw.get("p_yes", 0.5))
        except (TypeError, ValueError):
            p_model = 0.5
        rationale = raw.get("rationale") or ""

        # 2. Kalshi mid for this market.
        market_info = tick_ctx.get_candidate(market_id)
        if market_info is None:
            logger.warning("Forecast: market %s missing from tick_ctx, returning raw", market_id)
            return raw
        p_market = market_mid(market_info.yes_bid, market_info.yes_ask)
        if p_market is None:
            logger.warning("Forecast: market %s has no usable quote, returning raw", market_id)
            return raw

        # 3. Polymarket second-market signal (cached per process).
        poly = _cached_polymarket(market_info.question)

        # 4. Confidence label from search summary.
        confidence = _confidence_from_summary(summary)

        # 5. Blend, cap deviation, return SDK schema.
        result = calibrate(
            p_model=p_model,
            p_market=p_market,
            confidence=confidence,
            market_history=poly,
            config=self.calibration,
        )

        calib_note = (
            f"[anchor={'POLY+KALSHI' if result.used_polymarket else 'KALSHI'}"
            f" alpha={result.alpha:.2f}"
            f" p_model={result.p_model:.3f}"
            f" p_market={result.p_market:.3f}"
            f" -> p_final={result.p_final:.3f}"
            f"{' clipped' if result.clipped_to_cap else ''}]"
        )

        logger.info(
            "Forecast %s: %s (confidence=%s)",
            market_id, calib_note, confidence,
        )

        return {
            "p_yes":     result.p_final,
            "rationale": f"{rationale} {calib_note}".strip(),
        }
