"""
Hypothesis stage for the Prophet trading agent.

Pipeline position: step 2 of 3
  Extraction → Retrieval → [Hypothesis] → Verify

Adapts the forecaster prompt from the team's original design to the
Trading Track market format (FilteredMarket from extraction.py).

JSON schema produced by this stage:
{
  "market_id": "kalshi:KXBTCMAX100-26-DEC",
  "estimated_probability": 0.61,
  "confidence": "high" | "medium" | "low",
  "reasoning": "step-by-step reasoning...",
  "sources_used": ["reuters.com/...", "apnews.com/..."]
}
"""

from __future__ import annotations

import json
import logging
import re

import httpx
from pydantic import BaseModel

from prophet_agent.constants import HYPOTHESIS_MIN_EDGE
from prophet_agent.extraction import FilteredMarket
from prophet_agent.retrieval import RetrievalResult, build_context_string, run_retrieval

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class HypothesisResult(BaseModel):
    """
    Probability estimate for a single market.

    JSON schema (what Claude must return):
    {
      "market_id": "kalshi:KXBTCMAX100-26-DEC",
      "estimated_probability": 0.61,
      "confidence": "high",
      "reasoning": "step-by-step reasoning...",
      "sources_used": ["reuters.com: ...", "apnews.com: ..."]
    }
    """

    market_id: str
    question: str
    market_ask: float
    market_bid: float
    estimated_probability: float   # Our estimate of true YES probability (0.01–0.99)
    edge: float                    # estimated_probability - market_ask (positive = BUY YES)
    confidence: str                # "high" | "medium" | "low"
    reasoning: str
    sources_used: list[str]
    retry_count: int = 0


# ---------------------------------------------------------------------------
# System prompt (adapted from team forecaster prompt — Trading Track format)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an elite quantitative forecaster specializing in \
binary prediction markets. Markets pay $1 if YES resolves, $0 if NO.

CRITICAL SCORING RULE:
You are scored on PnL against the current market price (best_ask).
- If retrieved context gives NO clear informational edge, anchor your \
estimate to the market price to minimize risk.
- If you have strong, verifiable evidence that contradicts the market, \
deviate — but never exceed 0.30 deviation from market price unless \
retrieval confidence is very high.
- Never output 0.0 or 1.0. Cap range at 0.01 to 0.99.

REASONING PROCESS:
1. Read the market question and resolution criteria carefully.
2. Observe the current market price (best_ask) — this is the crowd consensus.
3. Analyze the retrieved news context. Weight recent, credible sources higher.
4. Reason step by step:
   Base rate → Evidence review → Market comparison → Final adjustment
5. Output ONLY a valid JSON object matching the required schema.

REQUIRED JSON SCHEMA:
{
  "market_id": "<string>",
  "estimated_probability": <float 0.01-0.99>,
  "confidence": "<high|medium|low>",
  "reasoning": "<step-by-step reasoning, 3-5 sentences>",
  "sources_used": ["<source 1 description>", "<source 2 description>"]
}

CONFIDENCE GUIDE:
  high   — strong recent evidence, clear resolution criteria, low ambiguity
  medium — mixed evidence or moderate uncertainty in resolution criteria
  low    — sparse evidence, ambiguous outcome, or fast-changing situation

OUTPUT ONLY RAW JSON — no markdown, no backticks, no preamble."""


# ---------------------------------------------------------------------------
# User prompt builder (Trading Track format)
# ---------------------------------------------------------------------------

def build_user_prompt(market: FilteredMarket, context_string: str) -> str:
    """
    Construct the hypothesis user prompt.

    Combines Trading Track market fields (from FilteredMarket) with
    retrieved news context. Mirrors the structure of the team's original
    build_user_prompt() but uses the SDK market format.
    """
    return (
        f"MARKET DETAILS:\n"
        f"Market ID:       {market.market_id}\n"
        f"Question:        {market.question}\n"
        f"Current ask:     {market.best_ask:.2f}  (cost to buy YES — crowd consensus)\n"
        f"Current bid:     {market.best_bid:.2f}  (cost to buy NO reversed)\n"
        f"Bid-ask spread:  {market.spread:.2f}\n"
        f"24h volume:      ${market.volume_24h:,.0f}\n"
        f"Resolves:        {market.resolution_time}\n\n"
        f"RETRIEVED NEWS CONTEXT:\n"
        f"{context_string}\n\n"
        f"Formulate your hypothesis using the reasoning process above.\n"
        f"Output ONLY the raw JSON object — no markdown, no backticks."
    )


# ---------------------------------------------------------------------------
# JSON extraction + validation (adapted from team's extract_and_validate_prediction)
# ---------------------------------------------------------------------------

def extract_and_validate(
    llm_response: str,
    market: FilteredMarket,
) -> dict | None:
    """
    Extract and validate the JSON block from the LLM response.

    Handles:
    - JSON embedded in conversational text (regex extraction)
    - Probabilities outside [0.01, 0.99] (clamped)
    - Missing fields (returns None → triggers retry or skip)

    Returns parsed dict on success, None on unrecoverable failure.
    """
    try:
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?", "", llm_response).strip()

        # Find the outermost JSON object
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON object found in response")

        data = json.loads(json_match.group(0))

        # Validate required fields
        required = {"estimated_probability", "confidence", "reasoning", "sources_used"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Clamp probability to valid range
        prob = float(data["estimated_probability"])
        prob = max(0.01, min(0.99, prob))
        data["estimated_probability"] = round(prob, 4)

        # Ensure market_id is set correctly
        data["market_id"] = market.market_id

        # Validate confidence value
        if data.get("confidence") not in ("high", "medium", "low"):
            data["confidence"] = "low"

        return data

    except (ValueError, KeyError, json.JSONDecodeError) as e:
        logger.warning(
            "JSON parsing failed for market %s: %s | raw: %.100s",
            market.market_id,
            e,
            llm_response,
        )
        return None


def _market_anchored_fallback(market: FilteredMarket, retry_count: int) -> HypothesisResult:
    """
    Safe fallback: anchor to market price when all LLM calls fail.
    Produces zero edge — no trade will be placed, but pipeline doesn't crash.
    """
    logger.warning(
        "Hypothesis fallback to market-anchored probability for %s", market.market_id
    )
    return HypothesisResult(
        market_id=market.market_id,
        question=market.question,
        market_ask=market.best_ask,
        market_bid=market.best_bid,
        estimated_probability=market.best_ask,   # zero edge
        edge=0.0,
        confidence="low",
        reasoning=(
            "Fallback triggered — all LLM calls failed or returned invalid output. "
            "Defaulting to market consensus price. No trade will be placed."
        ),
        sources_used=[],
        retry_count=retry_count,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_hypothesis(
    market: FilteredMarket,
    openrouter_api_key: str,
    tavily_api_key: str | None = None,
    *,
    retry_count: int = 0,
    timeout: float = 90.0,
) -> HypothesisResult:
    """
    Full hypothesis stage for a single filtered market.

    Steps:
      1. Run retrieval (web search + cache) to get news context
      2. Build user prompt with market data + context
      3. Call Claude Sonnet 4 via OpenRouter
      4. Extract + validate JSON response
      5. Compute edge vs market price
      6. Return HypothesisResult (or market-anchored fallback)

    Args:
        market:              FilteredMarket from the extraction stage.
        openrouter_api_key:  OpenRouter API key.
        tavily_api_key:      Tavily search API key (optional).
        retry_count:         How many times Verify has looped back to us.
        timeout:             HTTP timeout for the Claude call.

    Returns:
        HypothesisResult — always returns (uses fallback on failure).
    """
    # --- Step 1: Retrieval ---
    retrieval: RetrievalResult = await run_retrieval(
        market_id=market.market_id,
        question=market.question,
        openrouter_api_key=openrouter_api_key,
        tavily_api_key=tavily_api_key,
    )
    context_string = build_context_string(retrieval)

    # --- Step 2: Build prompt ---
    user_prompt = build_user_prompt(market, context_string)

    # --- Step 3: Call Claude Sonnet 4 ---
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "anthropic/claude-sonnet-4",
                    "temperature": 0.2,
                    "max_tokens": 1000,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            response.raise_for_status()
            raw_content = response.json()["choices"][0]["message"]["content"]

    except httpx.TimeoutException:
        logger.error("Hypothesis timeout for market %s", market.market_id)
        return _market_anchored_fallback(market, retry_count)
    except Exception as e:
        logger.error("Hypothesis API error for market %s: %s", market.market_id, e)
        return _market_anchored_fallback(market, retry_count)

    # --- Step 4: Extract + validate ---
    parsed = extract_and_validate(raw_content, market)
    if parsed is None:
        return _market_anchored_fallback(market, retry_count)

    # --- Step 5: Compute edge ---
    estimated_prob = parsed["estimated_probability"]
    edge = round(estimated_prob - market.best_ask, 4)

    result = HypothesisResult(
        market_id=market.market_id,
        question=market.question,
        market_ask=market.best_ask,
        market_bid=market.best_bid,
        estimated_probability=estimated_prob,
        edge=edge,
        confidence=parsed["confidence"],
        reasoning=parsed["reasoning"],
        sources_used=parsed.get("sources_used", []),
        retry_count=retry_count,
    )

    logger.info(
        "Hypothesis [%s] prob=%.3f ask=%.3f edge=%.3f conf=%s cached=%s",
        market.market_id[:30],
        estimated_prob,
        market.best_ask,
        edge,
        result.confidence,
        retrieval.cached,
    )

    # Log if edge is below threshold (won't trade, but useful for tuning)
    if abs(edge) < HYPOTHESIS_MIN_EDGE:
        logger.debug(
            "Market %s edge %.3f below threshold %.3f — will not trade",
            market.market_id,
            edge,
            HYPOTHESIS_MIN_EDGE,
        )

    return result