"""
Ingestion stage for the Prophet trading agent.

Pipeline position: step 1 of 3
  raw markets (256) → [Ingestion] → candidates (~25) → Hypothesis → Verify

Two-phase approach:
  1. Rule-based pre-filter  — cheap, instant, no LLM calls
  2. Gemini 2.5 Flash       — semantic filter on whatever survives phase 1
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from pydantic import BaseModel

from prophet_agent.constants import (
    EXTRACTION_MAX_ASK,
    EXTRACTION_MAX_DAYS,
    EXTRACTION_MAX_OUTPUT,
    EXTRACTION_MIN_ASK,
    EXTRACTION_MIN_SPREAD,
    EXTRACTION_MIN_VOLUME,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

class FilteredMarket(BaseModel):
    """A single market that passed extraction and is ready for hypothesis."""

    market_id: str
    question: str
    best_ask: float
    best_bid: float
    spread: float
    volume_24h: float
    resolution_time: str
    filter_reason: str  # "rule_based" | "gemini_selected"


class ExtractionResult(BaseModel):
    """Full output of the extraction stage."""

    markets: list[FilteredMarket]
    total_input: int
    pre_filter_count: int
    final_count: int


# ---------------------------------------------------------------------------
# Gemini system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a prediction market analyst. Your job is to select \
the most forecastable markets from a list of live binary prediction markets.

Select markets where:
- The outcome depends on a concrete, verifiable real-world event
- Current public information can meaningfully shift the probability
- The question is unambiguous — one clear YES/NO outcome

Skip markets that are:
- Purely based on random or chaotic events (exact weather readings, coin flips)
- So near certain (price < 0.10 or > 0.90) that research adds no edge
- Vague or subjective in their resolution criteria

Return ONLY valid JSON — no markdown, no explanation:
{"selected_ids": ["market_id_1", "market_id_2"]}"""


# ---------------------------------------------------------------------------
# Phase 1 — rule-based pre-filter (no LLM)
# ---------------------------------------------------------------------------

def pre_filter(
    markets: list[Any],
    *,
    min_spread: float = EXTRACTION_MIN_SPREAD,
    min_volume: float = EXTRACTION_MIN_VOLUME,
    min_ask: float = EXTRACTION_MIN_ASK,
    max_ask: float = EXTRACTION_MAX_ASK,
    max_days_to_resolution: int = EXTRACTION_MAX_DAYS,
) -> list[Any]:
    """
    Fast, deterministic filter before any LLM call.

    Args:
        markets:                  Raw market objects from get_candidates().
        min_spread:               Minimum bid-ask spread required (edge potential).
        min_volume:               Minimum 24-hour dollar volume (liquidity floor).
        min_ask / max_ask:        Price band — avoids near-certain markets.
        max_days_to_resolution:   Ignore markets resolving too far in the future.

    Returns:
        Subset of input markets that pass all filters.
    """
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=max_days_to_resolution)
    passed: list[Any] = []

    for m in markets:
        try:
            ask = float(m.quote.best_ask)
            bid = float(m.quote.best_bid)
            spread = ask - bid
            volume = float(m.quote.volume_24h)

            resolution = datetime.fromisoformat(str(m.resolution_time))
            if resolution.tzinfo is None:
                resolution = resolution.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError, AttributeError):
            logger.debug("Skipping market %s — unparseable fields", getattr(m, "market_id", "?"))
            continue

        if spread < min_spread:
            continue
        if volume < min_volume:
            continue
        if not (min_ask <= ask <= max_ask):
            continue
        if resolution > cutoff:
            continue

        passed.append(m)

    logger.info(
        "Pre-filter: %d → %d markets (spread≥%.2f, vol≥%.0f, ask∈[%.2f,%.2f], ≤%dd)",
        len(markets),
        len(passed),
        min_spread,
        min_volume,
        min_ask,
        max_ask,
        max_days_to_resolution,
    )
    return passed


# ---------------------------------------------------------------------------
# Phase 2 — Gemini 2.5 Flash semantic filter
# ---------------------------------------------------------------------------

async def _gemini_select(
    markets: list[Any],
    openrouter_api_key: str,
    max_output: int,
    timeout: float = 45.0,
) -> set[str]:
    """
    Ask Gemini 2.5 Flash to pick the most forecastable market IDs.

    Returns:
        Set of selected market_ids (may be smaller than max_output if Gemini
        decides fewer are worth forecasting).
    """
    market_list = [
        {
            "market_id": m.market_id,
            "question": m.question,
            "ask": float(m.quote.best_ask),
            "bid": float(m.quote.best_bid),
            "volume_24h": float(m.quote.volume_24h),
            "resolves": str(m.resolution_time),
        }
        for m in markets
    ]

    user_prompt = (
        f"From the {len(market_list)} markets below, select the best "
        f"{max_output} to research and forecast. "
        "Prioritise markets where current public information can beat the market price.\n\n"
        f"Markets:\n{json.dumps(market_list, indent=2)}\n\n"
        f'Return ONLY JSON: {{"selected_ids": ["id1", "id2", ...]}}'
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "google/gemini-2.5-flash",
                "temperature": 0.0,
                "max_tokens": 1500,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()

    raw = response.json()["choices"][0]["message"]["content"]

    try:
        parsed = json.loads(raw)
        selected: set[str] = set(parsed.get("selected_ids", []))
    except json.JSONDecodeError:
        logger.warning("Gemini returned invalid JSON — falling back to top-%d by spread", max_output)
        selected = set()

    logger.info("Gemini selected %d / %d markets", len(selected), len(markets))
    return selected


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_extraction(
    markets: list[Any],
    openrouter_api_key: str,
    *,
    max_output: int = EXTRACTION_MAX_OUTPUT,
    pre_filter_kwargs: dict[str, Any] | None = None,
) -> ExtractionResult:
    """
    Full extraction stage: rule-based pre-filter → Gemini semantic filter.

    Args:
        markets:             Raw list from api.get_candidates().markets.
        openrouter_api_key:  OpenRouter API key for Gemini calls.
        max_output:          Maximum number of markets to pass downstream.
        pre_filter_kwargs:   Optional overrides for pre_filter() thresholds.

    Returns:
        ExtractionResult with filtered markets ready for the hypothesis stage.
    """
    total_input = len(markets)

    # --- Phase 1: rule-based ---
    kwargs = pre_filter_kwargs or {}
    pre_filtered = pre_filter(markets, **kwargs)
    pre_filter_count = len(pre_filtered)

    # Fast path: already small enough, skip Gemini
    if pre_filter_count <= max_output:
        logger.info("Skipping Gemini — pre-filter output (%d) ≤ max_output (%d)", pre_filter_count, max_output)
        result_markets = [
            FilteredMarket(
                market_id=m.market_id,
                question=m.question,
                best_ask=float(m.quote.best_ask),
                best_bid=float(m.quote.best_bid),
                spread=float(m.quote.best_ask) - float(m.quote.best_bid),
                volume_24h=float(m.quote.volume_24h),
                resolution_time=str(m.resolution_time),
                filter_reason="rule_based",
            )
            for m in pre_filtered
        ]
        return ExtractionResult(
            markets=result_markets,
            total_input=total_input,
            pre_filter_count=pre_filter_count,
            final_count=len(result_markets),
        )

    # --- Phase 2: Gemini semantic filter ---
    selected_ids = await _gemini_select(pre_filtered, openrouter_api_key, max_output)

    # Fallback: if Gemini response was empty, sort by spread and take top N
    if not selected_ids:
        logger.warning("Gemini returned no IDs — falling back to top-%d by spread", max_output)
        fallback = sorted(pre_filtered, key=lambda m: float(m.quote.best_ask) - float(m.quote.best_bid), reverse=True)
        selected_ids = {m.market_id for m in fallback[:max_output]}

    market_map = {m.market_id: m for m in pre_filtered}
    result_markets = []

    for mid in selected_ids:
        if mid not in market_map:
            logger.debug("Gemini selected unknown market_id %s — skipping", mid)
            continue
        if len(result_markets) >= max_output:
            break
        m = market_map[mid]
        result_markets.append(
            FilteredMarket(
                market_id=m.market_id,
                question=m.question,
                best_ask=float(m.quote.best_ask),
                best_bid=float(m.quote.best_bid),
                spread=float(m.quote.best_ask) - float(m.quote.best_bid),
                volume_24h=float(m.quote.volume_24h),
                resolution_time=str(m.resolution_time),
                filter_reason="gemini_selected",
            )
        )

    return ExtractionResult(
        markets=result_markets,
        total_input=total_input,
        pre_filter_count=pre_filter_count,
        final_count=len(result_markets),
    )