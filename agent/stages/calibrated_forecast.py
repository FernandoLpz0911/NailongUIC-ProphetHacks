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
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from ai_prophet.trade.agent.stages.forecast import ForecastStage
from ai_prophet.trade.agent.tool_schemas import FORECAST_TOOL
from ai_prophet.trade.core import TickContext
from ai_prophet.trade.llm import LLMClient, LLMMessage

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


def _secs_to_expiry(market_info: object, tick_ctx: object) -> float | None:
    res = getattr(market_info, "resolution_time", None)
    ref_ts = getattr(tick_ctx, "tick_ts", None)
    if not isinstance(res, datetime) or ref_ts is None:
        return None
    if res.tzinfo is None:
        res = res.replace(tzinfo=timezone.utc)
    if ref_ts.tzinfo is None:
        ref_ts = ref_ts.replace(tzinfo=timezone.utc)
    return (res - ref_ts).total_seconds()


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


_SUPERFORECASTER_SYSTEM = """\
You are an expert prediction market forecaster trained in superforecasting.

Reason through FOUR steps before calling submit_forecast:

Step A — Reference Class: Name 3 historical events analogous to this \
question. For each give the outcome (YES/NO). Compute base_rate = #YES / 3.

Step B — Scenario Decomposition: Identify 2-4 MECE scenarios leading to \
YES or NO. Assign a probability to each (must sum to 1.0).

Step C — Inside-Outside Reconciliation: State your base_rate (Step A) and \
your inside-view estimate (Step B). Commit to a final probability and explain \
what specific evidence pulls you away from the base rate.

Step D — Call submit_forecast with your final probability.

Principles:
- Reason from evidence, NOT from the market price (shown for context only).
- Disagree with the market when evidence warrants.
- Extremes (p < 0.05 or p > 0.95) require strong, specific evidence.
- Be calibrated: a 70% estimate should be right ~70% of the time.\
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

    def _one_sample(self, messages: list[LLMMessage], market_id: str) -> float | None:
        """Single LLM call; returns clamped p_yes or None on failure."""
        try:
            raw = self.llm_client.generate_json(messages, tool=FORECAST_TOOL)
            p = raw.get("p_yes", 0.5)
            if isinstance(p, (int, float)) and 1.0 < p <= 100.0:
                p = p / 100.0
            return float(max(0.0, min(1.0, float(p))))
        except Exception as exc:
            logger.warning("Ensemble sample failed for %s: %s", market_id, exc)
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
            results = list(pool.map(lambda _: self._one_sample(messages, market_id), range(n)))

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
                "rationale": "[KILL_SWITCH] Budget exhausted; anchored to market mid.",
            }

        # 1. Kalshi mid. Try bare ID first, then with kalshi: prefix in case
        # the review stage stripped it from the model output.
        market_info = tick_ctx.get_candidate(market_id)
        if market_info is None and not market_id.startswith("kalshi:"):
            market_info = tick_ctx.get_candidate("kalshi:" + market_id)
        if market_info is None:
            logger.warning(
                "Forecast: market %s missing from tick_ctx, falling back to super",
                market_id,
            )
            return super()._generate_forecast(market_id, summary, tick_ctx)
        p_market = market_mid(market_info.yes_bid, market_info.yes_ask)
        if p_market is None:
            logger.warning(
                "Forecast: market %s has no usable quote, falling back to super",
                market_id,
            )
            return super()._generate_forecast(market_id, summary, tick_ctx)

        # 1b. Pre-filter: skip the ensemble for markets that will never generate
        #     a tradeable edge. Returning p_market here costs zero LLM tokens;
        #     the action stage computes edge≈0 and drops it automatically.
        #
        #     Near-expiry markets (≤ 1 h) get a relaxed boundary because prices
        #     are legitimately near 0.90+ when resolution is imminent, yet the
        #     remaining edge is still real and collectible quickly.
        secs_left = _secs_to_expiry(market_info, tick_ctx)
        near_expiry = secs_left is not None and secs_left <= 3_600

        boundary_lo = 0.04 if near_expiry else 0.07
        boundary_hi = 1.0 - boundary_lo
        if p_market < boundary_lo or p_market > boundary_hi:
            logger.info(
                "Forecast %s: skipping ensemble (p_market=%.3f near boundary, near_expiry=%s)",
                market_id, p_market, near_expiry,
            )
            return {"p_yes": p_market, "rationale": "[pre-filter: boundary price]"}

        #     Low-liquidity filter: spreads too wide for reliable fills.
        #     Near-expiry markets tolerate a wider spread — you hold briefly.
        spread = float(market_info.yes_ask) - float(market_info.yes_bid)
        spread_limit = 0.18 if near_expiry else 0.12
        if spread > spread_limit:
            logger.info(
                "Forecast %s: skipping ensemble (spread=%.3f too wide, near_expiry=%s)",
                market_id, spread, near_expiry,
            )
            return {"p_yes": p_market, "rationale": "[pre-filter: wide spread]"}

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
        p_model, conf_model, llm_var = self._ensemble_samples(messages, market_id, n)

        # Single-call fallback rationale (we don't capture per-sample rationale
        # in the ensemble path to keep logging tractable).
        rationale = f"ensemble n={n} conf={conf_model:.2f}"

        # 4. Polymarket second-market signal (cached per process).
        poly = _cached_polymarket(market_info.question)

        # 5. Confidence label from search summary (used only when conf_model=0).
        confidence = _confidence_from_summary(summary)

        # 6. Low-liquidity flag for dynamic alpha.
        low_liquidity = (market_info.volume_24h or 0.0) < 2000.0

        # 7. Blend, cap deviation, return SDK schema + extra fields for action stage.
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
        # Cannot include these in the returned dict — the SDK schema validator
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
