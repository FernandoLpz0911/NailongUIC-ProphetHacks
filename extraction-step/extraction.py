"""Extraction and hypothesis generation module for Prophet Hacks 2026."""

import csv
import json
import logging
import os
import re

current_dir = os.path.dirname(os.path.abspath(__file__))

log_dir = os.path.join(current_dir, "../logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "extraction.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CONSTANTS = {}
csv_path = os.path.join(current_dir, "../constants-step/constants.csv")

try:
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("Rule_or_Constant")
            val = row.get("Value", "").strip()
            val = val.replace("$", "")

            if key and val:
                try:
                    CONSTANTS[key] = float(val)
                except ValueError:
                    CONSTANTS[key] = val
    logger.info("Successfully loaded constants.csv")
except (OSError, csv.Error) as e:
    logger.error("Failed to load constants.csv: %s", e)

HYPOTHESIS_MIN_EDGE = CONSTANTS.get("HYPOTHESIS_MIN_EDGE", 0.10)
EXTRACTION_MIN_ASK = CONSTANTS.get("EXTRACTION_MIN_ASK", 0.10)
EXTRACTION_MAX_ASK = CONSTANTS.get("EXTRACTION_MAX_ASK", 0.90)

FORECASTER_SYSTEM_PROMPT = f"""
You are an elite quantitative forecaster in a high-stakes prediction market.
Your goal is to predict the true probability of future events resolving
as 'Yes' or 'No'.

IMPORTANT SCORING RULE:
You are scored on Average Return against the current market consensus
(last_price or mid-price).
- If the retrieved context gives you NO clear informational edge, you MUST
  anchor your prediction to the market probabilities to minimize risk.
- If you have strong evidence that contradicts the market, you
  may deviate, but avoid extreme overconfidence.
  Cap your deviation from the market at 0.30.
- MINIMUM EDGE: Your deviation from the market MUST be at least
  {HYPOTHESIS_MIN_EDGE}. If confidence does not meet this threshold,
  output the exact market consensus.

INSTRUCTIONS:
1. Read the Market Title and Resolution Rules carefully.
2. Analyze the Retrieved News Context. Weigh recent and credible sources.
3. Observe the current Market Quote (consensus).
4. Reason step-by-step to form your hypothesis.
5. Output your final answer STRICTLY as a valid JSON object.

REQUIRED JSON SCHEMA:
{{
  "market_id": "kalshi:KXNFLGAME-25NOV23DAL-DAL",
  "prediction": {{
    "Yes": 0.55,
    "No": 0.45
  }},
  "rationale": "A single, concise sentence summarizing evidence."
}}
"""


def build_user_prompt(market_data: dict, retrieved_context: str) -> str:
    """Construct user prompt combining SDK
    market JSON and retrieved context."""
    market_id = market_data.get("market_id", "UNKNOWN")
    logger.info("--- Starting Prompt Build for %s ---", market_id)

    context_str = (
        retrieved_context
        if retrieved_context
        else "No specific news found. Rely on base rates and market quotes."
    )

    return (
        f"MARKET DETAILS:\n"
        f"Title: {market_data.get('title')}\n"
        f"Market ID: {market_id}\n"
        f"Rules: {market_data.get('rules', 'Standard rules apply.')}\n\n"
        f"CURRENT MARKET QUOTE:\n"
        f"{market_data.get('quote', {})}\n\n"
        f"RETRIEVED NEWS CONTEXT:\n"
        f"{context_str}\n\n"
        f"Formulate your hypothesis. Provide your reasoning, then output "
        f"the final JSON block. Do not include markdown formatting like "
        f"```json in the final output, just the raw JSON string at the end."
    )


def extract_and_validate_prediction(
    llm_response: str, fallback_market_id: str, quote: dict
) -> dict | None:
    """Extract and validate JSON. Returns None if fails to avoid bad trades."""
    logger.info("--- Starting Extraction for %s ---", fallback_market_id)

    try:
        yes_ask = float(quote.get("best_ask", 0.5))
        yes_bid = float(quote.get("best_bid", yes_ask))
        p_market = round((yes_ask + yes_bid) / 2, 2)
    except (ValueError, TypeError, AttributeError):
        p_market = 0.5

    try:
        json_match = re.search(r"\{.*\}", llm_response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON object found in LLM response.")

        prediction_data = json.loads(json_match.group(0))

        yes_prob = prediction_data["prediction"].get("Yes", 0.5)
        no_prob = prediction_data["prediction"].get("No", 0.5)

        total = yes_prob + no_prob
        if total != 1.0:
            yes_prob = round(yes_prob / total, 2)
            no_prob = round(no_prob / total, 2)

        if p_market < EXTRACTION_MIN_ASK or p_market > EXTRACTION_MAX_ASK:
            logger.info("Constraint Hit: Extreme Market (%s)", p_market)
            yes_prob = p_market
            no_prob = round(1.0 - p_market, 2)
            rationale = prediction_data.get("rationale", "")
            prediction_data["rationale"] = f"[Extreme Market] {rationale}"
        else:
            edge = abs(yes_prob - p_market)
            if 0 < edge < HYPOTHESIS_MIN_EDGE:
                logger.info("Constraint Hit: Edge below minimum.")
                yes_prob = p_market
                no_prob = round(1.0 - p_market, 2)
                rationale = prediction_data.get("rationale", "")
                prediction_data["rationale"] = f"[Low Edge] {rationale}"

        if "market_id" not in prediction_data:
            prediction_data["market_id"] = fallback_market_id

        prediction_data["prediction"]["Yes"] = yes_prob
        prediction_data["prediction"]["No"] = no_prob

        logger.info("Final Validated Prediction generated successfully.")
        return prediction_data

    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        logger.error("FATAL EXTRACT ERROR for %s: %s", fallback_market_id, e)
        return None
