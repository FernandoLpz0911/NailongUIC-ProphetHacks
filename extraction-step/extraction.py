"""Extraction and hypothesis generation module for Prophet Hacks 2026."""

import json
import re

FORECASTER_SYSTEM_PROMPT = """
You are an elite quantitative forecaster in a high-stakes prediction market.
Your goal is to predict the true probability of future events resolving
as 'Yes' or 'No'.

IMPORTANT SCORING RULE:
You are scored on Average Return against the current market consensus
(last_price).
- If the retrieved context gives you NO clear informational edge, you MUST
  anchor your prediction to the current market probabilities to minimize risk.
- If you have strong, verifiable evidence that contradicts the market, you
  may deviate, but avoid extreme overconfidence (never predict 1.0 or 0.0).
  Cap your deviation from the market at 0.30.

INSTRUCTIONS:
1. Read the Event Title and Resolution Rules carefully.
2. Analyze the Retrieved News Context. Weigh recent and credible sources.
3. Observe the current Market Stats (consensus).
4. Reason step-by-step to form your hypothesis (Base rate -> Evidence ->
   Adjustment -> Final Probability).
5. Output your final answer STRICTLY as a valid JSON object matching the
   required schema.

REQUIRED JSON SCHEMA:
{
  "event_id": "EVT_1234",
  "prediction": {
    "Yes": 0.55,
    "No": 0.45
  },
  "rationale": "A single, concise sentence summarizing the strongest evidence."
}
"""


def build_user_prompt(event_payload: dict, retrieved_context: str) -> str:
    """Construct user prompt combining event JSON and retrieved context."""
    context_str = (
        retrieved_context
        if retrieved_context
        else "No specific news found. Rely on base rates and market stats."
    )

    return (
        f"EVENT DETAILS:\n"
        f"Title: {event_payload.get('title')}\n"
        f"Event ID: {event_payload.get('event_id')}\n"
        f"Rules: {event_payload.get('rules')}\n\n"
        f"CURRENT MARKET STATS:\n"
        f"{event_payload.get('market_stats')}\n\n"
        f"RETRIEVED NEWS CONTEXT:\n"
        f"{context_str}\n\n"
        f"Formulate your hypothesis. Provide your reasoning, then output "
        f"the final JSON block. Do not include markdown formatting like "
        f"```json in the final output, just the raw JSON string at the end."
    )


def extract_and_validate_prediction(
    llm_response: str, fallback_event_id: str, market_stats: dict
) -> dict:
    """Extract and validate JSON from the LLM response."""
    try:
        # Regex to find JSON block even if conversational text is present
        json_match = re.search(r"\{.*\}", llm_response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON object found in response.")

        prediction_data = json.loads(json_match.group(0))

        # Validation: Ensure probabilities sum to 1.0
        yes_prob = prediction_data["prediction"].get("Yes", 0.5)
        no_prob = prediction_data["prediction"].get("No", 0.5)

        total = yes_prob + no_prob
        if total != 1.0:
            prediction_data["prediction"]["Yes"] = round(yes_prob / total, 2)
            prediction_data["prediction"]["No"] = round(no_prob / total, 2)

        return prediction_data

    except (ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"[ERROR] JSON Parsing failed for {fallback_event_id}: {e}")

        # FALLSAFE: Calculate p_market = (1 - no_ask + yes_ask) / 2
        try:
            yes_ask = market_stats.get("Yes", {}).get("yes_ask", 0.5)
            no_ask = market_stats.get("No", {}).get("no_ask", 0.5)
            p_market = round((1 - no_ask + yes_ask) / 2, 2)
        except (AttributeError, TypeError):
            p_market = 0.5

        return {
            "event_id": fallback_event_id,
            "prediction": {"Yes": p_market, "No": round(1.0 - p_market, 2)},
            "rationale": (
                "Fallback triggered due to model parsing error. "
                "Defaulting to market consensus."
            ),
        }