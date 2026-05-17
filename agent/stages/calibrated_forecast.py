"""ForecastStage subclass that blends p_model with market consensus.

This is the Nailong Elite forecast stage. Key differences from the SDK base:

  1. **Plain-text JSON output** instead of tool calling — Gemini reliably
     malforms tool calls (`print(default_api.submit_forecast(...))`),
     silently dropping ~30-50% of ensemble samples. We parse JSON from
     a plain text response instead, mirroring TextReviewStage.

  2. **Strengthened superforecaster prompt** with category-specific
     warnings (paper §4.2.1: Flash hallucinates Politics/Climate events),
     a self-check step (paper Table 3: reasoning synthesis is the gap),
     and an explicit "if no specific reason market is wrong, output
     p_yes = p_market" fallback for the no-edge case.

  3. **Confidence interval** in the response — used for sizing.

  4. **Per-market structured decision log** — written to a JSONL file
     under data/decisions/ for offline calibration analysis.

  5. **Market-anchoring blend** with time-to-resolution decay
     (paper Fig.2: markets aggregate info faster than LLMs near resolution)
     and per-category alpha (paper Fig.5).

The blend math lives in `agent.calibration` and is unit-tested in isolation.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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

# Process-level Polymarket cache.
_POLY_CACHE: dict[str, tuple[float, dict | None]] = {}
_POLY_CACHE_TTL_SEC = 60 * 15

# Tick-scoped sidecar for forecast extras (raw_gap, conf_model, llm_var,
# days_to_resolution). The SDK's forecast JSON schema enforces
# additionalProperties:false so we can't add fields to the returned dict.
TICK_FORECAST_EXTRAS: dict[str, dict] = {}

# Structured per-market decision log (JSONL).
_DECISION_LOG_PATH = Path("data/decisions/forecast.jsonl")

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _cached_polymarket(question: str) -> dict | None:
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
    key_points = summary.get("key_points") or []
    open_qs = summary.get("open_questions") or []
    if len(key_points) >= 4 and len(open_qs) <= 1:
        return "high"
    if len(key_points) >= 2:
        return "medium"
    return "low"


def _parse_forecast_json(text: str) -> dict | None:
    """Extract a {p_yes, sigma, rationale} dict from a plain-text response.

    Tries fenced ```json blocks, then outermost { } block, then last-ditch
    regex for "p_yes": <number>. Mirrors the resilience of
    TextReviewStage._parse_review_json.
    """
    if not text:
        return None

    fence = _FENCE_RE.search(text)
    cleaned = fence.group(1) if fence else text
    cleaned = _TRAILING_COMMA_RE.sub(r"\1", cleaned.strip())

    # Try full parse first.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "p_yes" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    # Outermost { } block.
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start:end + 1])
            if isinstance(parsed, dict) and "p_yes" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # Regex fallback: pull just the p_yes number.
    m = re.search(r'["\']?p_yes["\']?\s*[:=]\s*([01]?\.\d+|\d+(?:\.\d+)?)', cleaned)
    if m:
        try:
            return {"p_yes": float(m.group(1)), "rationale": cleaned[:200]}
        except ValueError:
            pass

    return None


# =============================================================================
# System prompt — Nailong Elite (paper-backed)
# =============================================================================

_SUPERFORECASTER_SYSTEM = """\
You are an expert prediction-market forecaster trained in superforecasting.
You will be given an event, a market price (for context), and a research summary.

Your task: estimate the true probability of YES, then output JSON. You are
forecasting on Kalshi — a real-money platform where calibration matters
more than confidence.

REASON THROUGH FOUR STEPS, then output JSON at the end:

Step A — Reference Class: Name 2-3 historical events analogous to this
question. For each give the outcome (YES/NO). Compute base_rate = #YES / N.

Step B — Scenario Decomposition: Identify 2-4 MECE scenarios leading to
YES or NO. Assign a probability to each (must sum to ~1.0).

Step C — Self-Check (CRITICAL): The market price is shown above. If your
estimate differs from market by more than 15 percentage points, you MUST
identify a SPECIFIC piece of retrieved evidence that justifies the gap.
A vague "I think the market is wrong" is not sufficient — name the evidence.

Step D — Inside-Outside Reconciliation: State your base rate (Step A) and
inside-view estimate (Step B). Commit to a final probability.

CATEGORY-SPECIFIC WARNINGS:
- For Politics and Climate/Weather questions: your training memory of
  specific dates and events is unreliable. Use ONLY the retrieved sources.
  If the retrieved sources do not contain specific information, anchor
  to the market price.
- For Sports and Entertainment: the market is usually well-informed.
  Only deviate when you have specific recent news the market may have
  missed.

EXTREMES require strong evidence:
- p_yes < 0.05 or > 0.95: only with clear, specific, retrieved evidence.
- Otherwise, stay within [0.10, 0.90].

NO-EDGE FALLBACK (use this freely):
If you cannot identify a specific reason the market is mispriced, output
p_yes equal to the market price. The bot will skip this trade. Doing this
is BETTER than guessing — guessing loses money against the market.

OUTPUT FORMAT — at the end of your reasoning, output a single JSON object:

```json
{
  "p_yes": 0.42,
  "sigma": 0.08,
  "rationale": "One sentence citing the specific evidence."
}
```

- p_yes: your final probability, 2 decimal places, in [0, 1]
- sigma: your 1-σ uncertainty estimate (volatility of your belief), [0, 0.5]
  - Use 0.05 for high-conviction estimates, 0.15+ when uncertain
- rationale: ONE sentence naming the specific evidence driving your number

Do NOT call any tools. Do NOT use function calls. Output plain text reasoning
followed by the JSON object.
"""


class CalibratedForecastStage(ForecastStage):
    """Plain-text, market-anchored, ensemble forecasting with paper-backed prompt."""

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
        # Ensure decision log directory exists.
        try:
            _DECISION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Ensemble helpers — plain-text JSON path (no tool calling)
    # ------------------------------------------------------------------

    def _one_sample(
        self, messages: list[LLMMessage], market_id: str
    ) -> tuple[float | None, float | None]:
        """Single LLM call returning (p_yes, sigma) or (None, None) on failure."""
        try:
            response = self.llm_client.generate(
                LLMRequest(messages=messages, max_tokens=4096)
            )
            text = response.content or ""
            parsed = _parse_forecast_json(text)
            if parsed is None:
                logger.warning(
                    "Forecast sample for %s: could not parse JSON from response",
                    market_id,
                )
                return None, None

            p = parsed.get("p_yes", 0.5)
            if isinstance(p, (int, float)) and 1.0 < p <= 100.0:
                p = p / 100.0
            p = float(max(0.0, min(1.0, float(p))))

            sigma = parsed.get("sigma", 0.10)
            try:
                sigma = float(sigma)
                sigma = max(0.01, min(0.5, sigma))
            except (TypeError, ValueError):
                sigma = 0.10

            return p, sigma
        except Exception as exc:
            logger.warning("Ensemble sample failed for %s: %s", market_id, exc)
            return None, None

    def _ensemble_samples(
        self,
        messages: list[LLMMessage],
        market_id: str,
        n: int,
    ) -> tuple[float, float, float, float]:
        """Run n LLM calls concurrently; return (p_model, conf_model, llm_var, mean_sigma)."""
        with ThreadPoolExecutor(max_workers=min(max(n, 1), 8)) as pool:
            results = list(pool.map(lambda _: self._one_sample(messages, market_id), range(n)))

        valid_ps = [p for p, _ in results if p is not None]
        valid_sigmas = [s for _, s in results if s is not None]

        if not valid_ps:
            return 0.5, 0.0, 0.0, 0.10
        if len(valid_ps) == 1:
            return valid_ps[0], 0.0, 0.0, (valid_sigmas[0] if valid_sigmas else 0.10)

        eps = 1e-6
        logits = [math.log((p + eps) / (1.0 - p + eps)) for p in valid_ps]
        trimmed = sorted(logits)[1:-1] if len(logits) > 2 else logits
        mean_logit = sum(trimmed) / len(trimmed)
        p_agg = 1.0 / (1.0 + math.exp(-mean_logit))

        std = statistics.stdev(valid_ps)
        # 1 - 2*std maps [0, 0.5] std range to [0, 1] confidence.
        conf_model = max(0.0, min(1.0, 1.0 - 2.0 * std))
        llm_var = statistics.variance(valid_ps)
        mean_sigma = sum(valid_sigmas) / len(valid_sigmas) if valid_sigmas else 0.10

        return (
            float(max(0.0, min(1.0, p_agg))),
            conf_model,
            llm_var,
            mean_sigma,
        )

    # ------------------------------------------------------------------
    # Decision logging
    # ------------------------------------------------------------------

    def _write_decision_log(self, entry: dict) -> None:
        try:
            with open(_DECISION_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as exc:
            logger.debug("Decision log write failed: %s", exc)

    # ------------------------------------------------------------------
    # Main forecast entry point
    # ------------------------------------------------------------------

    def _generate_forecast(
        self,
        market_id: str,
        summary: dict[str, Any],
        tick_ctx: TickContext,
    ) -> dict[str, Any]:
        tick_ts = getattr(tick_ctx, "tick_ts", None)

        # 0. Kill switch.
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
            self._write_decision_log({
                "tick_ts": str(tick_ts), "market_id": market_id,
                "p_market": p_market, "p_model": p_market, "p_final": p_market,
                "decision": "SKIP_KILL_SWITCH",
            })
            return {
                "p_yes": p_market,
                "rationale": "[KILL_SWITCH] Budget exhausted; anchored to market mid.",
            }

        # 1. Pull market info.
        market_info = tick_ctx.get_candidate(market_id)
        if market_info is None and not market_id.startswith("kalshi:"):
            market_info = tick_ctx.get_candidate("kalshi:" + market_id)
        if market_info is None:
            logger.warning("Forecast: market %s missing from tick_ctx, falling back", market_id)
            return super()._generate_forecast(market_id, summary, tick_ctx)

        p_market = market_mid(market_info.yes_bid, market_info.yes_ask)
        if p_market is None:
            return super()._generate_forecast(market_id, summary, tick_ctx)

        # 1b. Pre-filters.
        if p_market < 0.07 or p_market > 0.93:
            logger.info("Forecast %s: skip ensemble (p_market=%.3f near boundary)", market_id, p_market)
            self._write_decision_log({
                "tick_ts": str(tick_ts), "market_id": market_id,
                "p_market": p_market, "p_model": p_market, "p_final": p_market,
                "decision": "SKIP_BOUNDARY",
            })
            return {"p_yes": p_market, "rationale": "[pre-filter: boundary price]"}

        spread = float(market_info.yes_ask) - float(market_info.yes_bid)
        if spread > 0.12:
            logger.info("Forecast %s: skip ensemble (spread=%.3f too wide)", market_id, spread)
            self._write_decision_log({
                "tick_ts": str(tick_ts), "market_id": market_id,
                "p_market": p_market, "p_model": p_market, "p_final": p_market,
                "decision": "SKIP_WIDE_SPREAD", "spread": spread,
            })
            return {"p_yes": p_market, "rationale": "[pre-filter: wide spread]"}

        # 2. Build prompt.
        summary_text = summary.get("summary", "No summary available")
        key_points = summary.get("key_points") or []
        open_qs = summary.get("open_questions") or []
        kp_text = "\n".join(f"- {kp}" for kp in key_points) if key_points else "None identified"
        oq_text = "\n".join(f"- {q}" for q in open_qs) if open_qs else "None identified"

        memory_by_market = getattr(tick_ctx, "memory_by_market", None) or {}
        market_memory = memory_by_market.get(market_id, "")
        memory_block = f"\n\nRECENT MEMORY:\n{market_memory}" if market_memory else ""

        user_prompt = (
            f"Event: {market_info.question}\n"
            f"Market price (for context only): {p_market:.2f}\n\n"
            f"RESEARCH SUMMARY:\n{summary_text}\n\n"
            f"KEY POINTS:\n{kp_text}\n\n"
            f"OPEN QUESTIONS:\n{oq_text}"
            f"{memory_block}\n\n"
            f"Reason through the four steps, then output JSON with p_yes, sigma, and rationale."
        )

        messages = [
            LLMMessage(role="system", content=_SUPERFORECASTER_SYSTEM),
            LLMMessage(role="user", content=user_prompt),
        ]

        # 3. Ensemble.
        n = self.calibration.llm_ensemble_n
        p_model, conf_model, llm_var, mean_sigma = self._ensemble_samples(messages, market_id, n)
        rationale = f"ensemble n={n} conf={conf_model:.2f} sigma={mean_sigma:.2f}"

        # 4. Polymarket.
        poly = _cached_polymarket(market_info.question)

        # 5. Confidence label.
        confidence = _confidence_from_summary(summary)
        low_liquidity = (market_info.volume_24h or 0.0) < 2000.0

        # 6. Time-to-resolution.
        hours_to_resolution: float | None = None
        close_time = getattr(market_info, "close_time", None) or getattr(market_info, "end_date", None)
        if close_time is not None:
            try:
                now = _dt.datetime.now(_dt.timezone.utc)
                if hasattr(close_time, "tzinfo"):
                    delta = close_time - now
                else:
                    delta = _dt.datetime.fromisoformat(str(close_time)).replace(
                        tzinfo=_dt.timezone.utc
                    ) - now
                hours_to_resolution = max(0.0, delta.total_seconds() / 3600.0)
            except Exception:
                pass

        days_to_resolution = (
            hours_to_resolution / 24.0 if hours_to_resolution is not None else None
        )

        # 7. Category alpha.
        from agent.stages.text_review import _category as _cat
        category = _cat(market_id, market_info.question)
        _cat_alpha_map = {
            "Politics":      getattr(self.calibration, "alpha_politics", None),
            "Economics":     getattr(self.calibration, "alpha_economics", None),
            "Sports":        getattr(self.calibration, "alpha_sports", None),
            "Entertainment": getattr(self.calibration, "alpha_entertainment", None),
            "Science/Tech":  getattr(self.calibration, "alpha_science", None),
        }
        category_alpha = _cat_alpha_map.get(category)

        # 8. Calibrate.
        result = calibrate(
            p_model=p_model,
            p_market=p_market,
            confidence=confidence,
            market_history=poly,
            config=self.calibration,
            conf_model=conf_model,
            low_liquidity=low_liquidity,
            hours_to_resolution=hours_to_resolution,
            category_alpha=category_alpha,
        )

        hrs_str = f" hrs={hours_to_resolution:.1f}" if hours_to_resolution is not None else ""
        calib_note = (
            f"[anchor={'POLY+KALSHI' if result.used_polymarket else 'KALSHI'}"
            f" alpha={result.alpha:.2f} cat={category}"
            f" p_model={result.p_model:.3f} p_market={result.p_market:.3f}"
            f" raw_gap={result.raw_gap:+.3f} -> p_final={result.p_final:.3f}"
            f"{' clipped' if result.clipped_to_cap else ''}{hrs_str}]"
        )

        logger.info(
            "Forecast %s: %s (confidence=%s conf_model=%.2f sigma=%.2f)",
            market_id, calib_note, confidence, conf_model, mean_sigma,
        )

        # 9. Sidecar for action stage.
        TICK_FORECAST_EXTRAS[market_id] = {
            "raw_gap":             result.raw_gap,
            "conf_model":          result.conf_model,
            "llm_var":             llm_var,
            "sigma":               mean_sigma,
            "days_to_resolution":  days_to_resolution,
            "category":            category,
            "alpha":               result.alpha,
            "used_polymarket":     result.used_polymarket,
            "clipped":             result.clipped_to_cap,
        }

        # 10. Structured decision log.
        self._write_decision_log({
            "tick_ts":            str(tick_ts),
            "market_id":          market_id,
            "question":           market_info.question,
            "category":           category,
            "p_market":           round(p_market, 4),
            "p_model":            round(result.p_model, 4),
            "p_final":            round(result.p_final, 4),
            "raw_gap":            round(result.raw_gap, 4),
            "alpha":              round(result.alpha, 3),
            "conf_model":         round(conf_model, 3),
            "sigma":              round(mean_sigma, 3),
            "llm_var":            round(llm_var, 5),
            "ensemble_n":         n,
            "ensemble_valid":     sum(1 for p, _ in [(None, None)] if p is not None),  # placeholder; real count in samples
            "days_to_resolution": days_to_resolution,
            "used_polymarket":    result.used_polymarket,
            "clipped":            result.clipped_to_cap,
            "decision":           "FORECAST_OK",
        })

        return {
            "p_yes":     result.p_final,
            "rationale": f"{rationale} {calib_note}".strip(),
        }
