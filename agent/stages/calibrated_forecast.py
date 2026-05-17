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
import math
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ai_prophet.trade.agent.stages.forecast import ForecastStage
from ai_prophet.trade.core import TickContext
from ai_prophet.trade.llm import LLMClient, LLMMessage
from ai_prophet.trade.llm.base import LLMRequest

from agent.calibration import calibrate, market_mid
from agent.settings import CalibrationConfig
from agent.spend import is_killed
from retrieval.retrieval import fetch_polymarket

logger = logging.getLogger(__name__)

# Process-level Polymarket cache. Cleared lazily once entries are older than
# `_POLY_CACHE_TTL_SEC`. Keyed on the candidate question text.
_POLY_CACHE: dict[str, tuple[float, dict | None]] = {}
_POLY_CACHE_TTL_SEC = 60 * 15  # ~1 tick

# Tick-scoped sidecar for forecast extras (raw_gap, conf_model, llm_var).
# The SDK's forecast JSON schema has additionalProperties:false, so we can't
# include these fields in the returned dict. Instead we stash them here and
# the action stage reads them directly. Entries are overwritten each tick so
# stale data from a previous tick never leaks into the current one.
TICK_FORECAST_EXTRAS: dict[str, dict] = {}


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
    `summary["sources"]` when available. Maps to one of high/medium/low.

    Heuristic: lots of concrete key points + few open questions => high.
    """
    key_points = summary.get("key_points") or []
    open_qs = summary.get("open_questions") or []
    if len(key_points) >= 4 and len(open_qs) <= 1:
        return "high"
    if len(key_points) >= 2:
        return "medium"
    return "low"


_SUPERFORECASTER_SYSTEM = """\
You are an expert prediction market forecaster.

In your internal reasoning (not visible in your response), work through:
  A. Reference class — 3 historical analogues → compute base_rate
  B. Scenario decomposition — 2-4 MECE scenarios with probabilities
  C. Reconcile base_rate vs. inside view; commit to a final probability

OUTPUT RULES — critical:
- Your entire response must be exactly ONE line of JSON, nothing else.
- Do NOT write "Step A", "Step B", or any prose in your response.
- Do NOT use markdown, code fences, or tool calls.
- Format: {"p_yes": 0.XX}

Forecasting principles (apply in your reasoning, not in your response):
- Reason from evidence, NOT from market price (shown for context only).
- Disagree with the market when evidence warrants.
- Extremes (p < 0.05 or p > 0.95) require strong specific evidence.
- Be calibrated: 70% should be right roughly 70% of the time.\
"""


class CalibratedForecastStage(ForecastStage):
    """ForecastStage that anchors `p_yes` to market consensus via ensemble."""

    def __init__(
        self,
        llm_client: LLMClient,
        calibration: CalibrationConfig | None = None,
    ) -> None:
        super().__init__(llm_client=llm_client)
        if calibration is None:
            from agent.settings import CalibrationConfig as _CC
            calibration = _CC()
        self.calibration = calibration

    # ------------------------------------------------------------------
    # Ensemble helpers
    # ------------------------------------------------------------------

    def _one_sample(
        self, messages: list[LLMMessage], market_id: str
    ) -> float | None:
        """Single LLM call; returns clamped p_yes or None on failure.

        Uses plain text generation (no tool call) to avoid Gemini's
        MALFORMED_FUNCTION_CALL bug where it wraps calls in print().
        """
        try:
            # Gemini 2.5 Flash counts thinking tokens against max_tokens.
            # The model uses ~500 thinking tokens, then writes Steps A-C
            # in response text, then emits the final JSON. 4096 gives
            # ~3500 output tokens after thinking — enough for all steps.
            response = self.llm_client.generate(
                LLMRequest(messages=messages, max_tokens=4096)
            )
            text = response.content or ""
            m = re.search(r'"p_yes"\s*:\s*([0-9.]+)', text)
            if not m:
                logger.warning(
                    "No p_yes in forecast response for %s: %.100s",
                    market_id, text,
                )
                return None
            p = float(m.group(1))
            if 1.0 < p <= 100.0:
                p /= 100.0
            return max(0.0, min(1.0, p))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Ensemble sample failed for %s: %s", market_id, exc
            )
            return None

    def _ensemble_samples(
        self,
        messages: list[LLMMessage],
        market_id: str,
        n: int,
    ) -> tuple[float, float, float]:
        """Run n LLM calls concurrently; return (p_model, conf_model, llm_var).

        Aggregation: trimmed mean on logit scale (drop min and max).
        Confidence: 1 - std(raw samples), clipped to [0, 1].
        """
        with ThreadPoolExecutor(max_workers=min(n, 8)) as pool:
            results = list(pool.map(
                lambda _: self._one_sample(messages, market_id), range(n)
            ))

        valid = [p for p in results if p is not None]
        if not valid:
            return 0.5, 0.0, 0.0
        if len(valid) == 1:
            return valid[0], 0.0, 0.0

        eps = 1e-6
        logits = [math.log((p + eps) / (1.0 - p + eps)) for p in valid]
        trimmed = sorted(logits)[1:-1] if len(logits) > 2 else logits
        mean_logit = sum(trimmed) / len(trimmed)
        p_agg = 1.0 / (1.0 + math.exp(-mean_logit))

        std = statistics.stdev(valid)
        conf_model = max(0.0, min(1.0, 1.0 - std))
        llm_var = statistics.variance(valid)

        return float(max(0.0, min(1.0, p_agg))), conf_model, llm_var

    # ------------------------------------------------------------------
    # Main forecast entry point
    # ------------------------------------------------------------------

    def _generate_forecast(
        self,
        market_id: str,
        summary: dict[str, Any],
        tick_ctx: TickContext,
    ) -> dict[str, Any]:
        # 0. Budget kill-switch.
        if is_killed():
            market_info = tick_ctx.get_candidate(market_id)
            p_market = market_mid(
                market_info.yes_bid if market_info else 0.5,
                market_info.yes_ask if market_info else 0.5,
            ) or 0.5
            logger.warning(
                "Forecast %s: KILL SWITCH active, p_yes=p_market=%.3f",
                market_id, p_market,
            )
            return {
                "p_yes": p_market,
                "rationale": "[KILL_SWITCH] anchored to market mid.",
            }

        # 1. Kalshi mid. Try bare ID first, then with kalshi: prefix in case
        # the review stage stripped it from the model output.
        market_info = tick_ctx.get_candidate(market_id)
        if market_info is None and not market_id.startswith("kalshi:"):
            market_info = tick_ctx.get_candidate("kalshi:" + market_id)
        if market_info is None:
            logger.warning(
                "Forecast: %s missing from tick_ctx, falling back to super",
                market_id,
            )
            return super()._generate_forecast(market_id, summary, tick_ctx)
        p_market = market_mid(market_info.yes_bid, market_info.yes_ask)
        if p_market is None:
            logger.warning(
                "Forecast: %s has no usable quote, falling back to super",
                market_id,
            )
            return super()._generate_forecast(market_id, summary, tick_ctx)

        # 1b. Pre-filter: skip ensemble for markets that can't generate edge.
        #     Returning p_market costs zero LLM tokens; action stage drops it.
        #
        #     Tail-risk filter: near-boundary prices have severe asymmetric
        #     risk and almost never move enough to justify a bet.
        if p_market < 0.07 or p_market > 0.93:
            logger.info(
                "Forecast %s: skipping ensemble (p_market=%.3f near boundary)",
                market_id, p_market,
            )
            return {
                "p_yes": p_market,
                "rationale": "[pre-filter: boundary price]",
            }

        #     Low-liquidity filter: spreads too wide for reliable fills.
        spread = float(market_info.yes_ask) - float(market_info.yes_bid)
        if spread > 0.12:
            logger.info(
                "Forecast %s: skipping ensemble (spread=%.3f too wide)",
                market_id, spread,
            )
            return {
                "p_yes": p_market,
                "rationale": "[pre-filter: wide spread]",
            }

        # 2. Build structured superforecaster prompt.
        summary_text = summary.get("summary", "No summary available")
        key_points = summary.get("key_points") or []
        open_qs = summary.get("open_questions") or []
        kp_text = (
            "\n".join(f"- {kp}" for kp in key_points)
            if key_points else "None identified"
        )
        oq_text = (
            "\n".join(f"- {q}" for q in open_qs)
            if open_qs else "None identified"
        )
        memory_by_market = getattr(tick_ctx, "memory_by_market", None) or {}
        market_memory = memory_by_market.get(market_id, "")
        memory_block = (
            f"\n\nRECENT MEMORY:\n{market_memory}" if market_memory else ""
        )

        user_prompt = (
            f"Event: {market_info.question}\n"
            f"Market price (context only): {p_market:.1%}\n\n"
            f"RESEARCH SUMMARY:\n{summary_text}\n\n"
            f"KEY POINTS:\n{kp_text}\n\n"
            f"OPEN QUESTIONS:\n{oq_text}"
            f"{memory_block}"
        )

        messages = [
            LLMMessage(role="system", content=_SUPERFORECASTER_SYSTEM),
            LLMMessage(role="user", content=user_prompt),
        ]

        # 3. Ensemble: n calls, trimmed-mean on logit scale.
        n = self.calibration.llm_ensemble_n
        p_model, conf_model, llm_var = self._ensemble_samples(
            messages, market_id, n
        )

        # Single-call fallback rationale (we don't capture per-sample rationale
        # in the ensemble path to keep logging tractable).
        rationale = f"ensemble n={n} conf={conf_model:.2f}"

        # 4. Polymarket second-market signal (cached per process).
        poly = _cached_polymarket(market_info.question)

        # 5. Confidence label from summary (used only when conf_model=0).
        confidence = _confidence_from_summary(summary)

        # 6. Low-liquidity flag for dynamic alpha.
        low_liquidity = (market_info.volume_24h or 0.0) < 2000.0

        # 7. Blend, cap deviation, return SDK schema + extras for action stage.
        result = calibrate(
            p_model=p_model,
            p_market=p_market,
            confidence=confidence,
            market_history=poly,
            config=self.calibration,
            conf_model=conf_model,
            low_liquidity=low_liquidity,
        )

        calib_note = (
            f"[anchor={'POLY+KALSHI' if result.used_polymarket else 'KALSHI'}"
            f" alpha={result.alpha:.2f}"
            f" p_model={result.p_model:.3f}"
            f" p_market={result.p_market:.3f}"
            f" raw_gap={result.raw_gap:+.3f}"
            f" -> p_final={result.p_final:.3f}"
            f"{' clipped' if result.clipped_to_cap else ''}]"
        )

        logger.info(
            "Forecast %s: %s (confidence=%s conf_model=%.2f)",
            market_id, calib_note, confidence, conf_model,
        )

        # Stash extras for the action stage via the sidecar dict.
        # Cannot include these in the returned dict — SDK schema validator
        # enforces additionalProperties:false on forecast results.
        TICK_FORECAST_EXTRAS[market_id] = {
            "raw_gap":    result.raw_gap,
            "conf_model": result.conf_model,
            "llm_var":    llm_var,
        }

        return {
            "p_yes":     result.p_final,
            "rationale": f"{rationale} {calib_note}".strip(),
        }
