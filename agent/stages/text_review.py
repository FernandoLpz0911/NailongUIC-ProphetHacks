"""TextReviewStage: ReviewStage that requests plain-text JSON output.

Gemini consistently generates Python-style function calls
(print(default_api.submit_review(...))) instead of proper JSON tool responses,
causing MALFORMED_FUNCTION_CALL with a truncated finishMessage — only the first
market survives the salvage path. By asking for JSON in the message body with
no tool binding, we get the full list every tick.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence

from ai_prophet.trade.agent.stages.review import ReviewStage
from ai_prophet.trade.agent.utils import format_portfolio_summary
from ai_prophet.trade.core import TickContext
from ai_prophet.trade.core.tick_context import CandidateMarket
from ai_prophet.trade.llm import LLMMessage
from ai_prophet.trade.llm.base import LLMRequest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category classification — used to label candidates and enforce diversity.
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Politics": [
        "president", "senate", "congress", "election", "vote", "democrat",
        "republican", "governor", "mayor", "primary", "kxpres", "controls-",
        "senat", "kxvote", "kxcanal", "kxlamayoradvance",
    ],
    "Economics": [
        "fed", "gdp", "inflation", "bond", "price index", "dollar", "bitcoin",
        "crypto", "oil", "interest rate", "kxbond", "kxfed", "kxbtc",
        "kxgtaprice", "kxgreenlandprice",
    ],
    "Sports": [
        "nba", "nfl", "mlb", "nhl", "super bowl", "championship", "world cup",
        "wimbledon", "playoff", "mvp", "kxnba", "kxnfl", "kxmlb", "kxnhl",
        "kxsb-", "kxmenworldcup", "kxnbawest",
    ],
    "Entertainment": [
        "oscar", "emmy", "grammy", "award", "actor", "actress", "movie",
        "celebrity", "taylor swift", "bond film", "james bond",
        "kxoscar", "kxswift", "kxbond-",
    ],
    "Science/Tech": [
        "spacex", "nasa", "launch", "satellite", "alien", "climate",
        "temperature", "greenland", "ai model", "kxspacex", "kxgreenland",
        "kxaliens", "kxustests",
    ],
}


def _category(market_id: str, question: str) -> str:
    """Infer a broad category from market ID and question text."""
    combined = (market_id + " " + question).lower()
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return cat
    return "Other"


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _clean(text: str) -> str:
    """Strip markdown fences and trailing commas."""
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1)
    return _TRAILING_COMMA_RE.sub(r"\1", text.strip())


def _extract_objects(s: str) -> list[dict]:
    """Pull every complete top-level JSON object out of a truncated string."""
    objects: list[dict] = []
    i = 0
    while i < len(s):
        if s[i] != "{":
            i += 1
            continue
        depth, in_str, escaped, closed_at = 0, False, False, -1
        j = i
        while j < len(s):
            c = s[j]
            if escaped:
                escaped = False
            elif c == "\\" and in_str:
                escaped = True
            elif c == '"':
                in_str = not in_str
            elif not in_str:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        closed_at = j
                        break
            j += 1
        if closed_at == -1:
            break
        try:
            obj = json.loads(_TRAILING_COMMA_RE.sub(r"\1", s[i:closed_at + 1]))
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            pass
        i = closed_at + 1
    return objects


def _parse_review_json(text: str) -> dict | None:
    """Extract and parse the review JSON from a free-text model response.

    Tries four strategies in order:
    1. Full parse — fast path, works when output is complete.
    2. Outermost { } block — handles leading/trailing prose.
    3. Object-by-object from top level — mild truncation / trailing commas.
    4. Extract entries from inside the review array — severe truncation where
       the outer { } is never closed (typical MAX_TOKENS mid-entry cutoff).
    """
    cleaned = _clean(text)

    # 1. Full parse — must be a dict.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Outermost { } block.
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start:end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 3. Object-by-object from top level.
    objects = _extract_objects(cleaned)
    if objects:
        if "review" in objects[0]:
            return objects[0]
        return {"review": objects}

    # 4. Scan inside the "review": [ array directly — handles the case where
    # the outer object is never closed (model cut off mid-entry).
    arr_match = re.search(r'"review"\s*:\s*\[', cleaned)
    if arr_match:
        entries = _extract_objects(cleaned[arr_match.end() - 1:])
        if entries:
            return {"review": entries}

    return None


class TextReviewStage(ReviewStage):
    """ReviewStage variant that uses plain-text JSON output.

    Sidesteps Gemini's MALFORMED_FUNCTION_CALL by asking for a JSON
    object in the reply body rather than a tool call.
    """

    @property
    def name(self) -> str:
        return "review"

    def _generate_review(
        self,
        candidates: Sequence[CandidateMarket],
        tick_ctx: TickContext,
    ) -> dict:
        # Pre-filter: drop near-resolved markets; cap to top 80 by volume.
        # Sending all 250+ candidates exhausts Gemini's thinking budget.
        active = [
            m for m in candidates
            if 0.08 < (float(m.yes_bid) + float(m.yes_ask)) / 2.0 < 0.92
        ]
        # Deprioritize long-horizon markets (year >= 2027 in ID or question).
        # These won't resolve in the 14-day competition window; sort them last
        # so the LLM sees near-term markets first in its capped list of 80.
        def _is_long_horizon(m: object) -> bool:
            text = (getattr(m, "market_id", "") + " "
                    + getattr(m, "question", "")).lower()
            return any(f"-{y}" in text or f" {y}" in text
                       for y in ("2027", "2028", "2029", "2030"))

        active.sort(
            key=lambda m: (
                _is_long_horizon(m),          # long-horizon goes last
                -float(m.volume_24h or 0),    # then highest volume first
            )
        )
        candidates = active[:80] if len(active) > 80 else active

        candidates_text = "\n".join(
            f"{m.market_id} | {m.question[:80]} | "
            f"{m.yes_bid:.2f}/{m.yes_ask:.2f} | ${m.volume_24h:.0f} "
            f"[{_category(m.market_id, m.question)}]"
            for m in candidates
        )

        positions_text = format_portfolio_summary(
            tick_ctx, include_positions=True
        )
        memory_summary = getattr(tick_ctx, "memory_summary", "") or ""
        memory_block = (
            f"\n\nRECENT MEMORY:\n{memory_summary}" if memory_summary else ""
        )
        logger.info(
            "Review prompt memory_in_prompt=%s memory_chars=%d",
            bool(memory_block),
            len(memory_summary),
        )

        system_prompt = f"""\
You are a prediction market analyst selecting markets for detailed analysis.

HOW PREDICTION MARKETS WORK:
- Price = probability (0.50 = 50% chance of YES)
- BUY YES at 0.40: profit if event happens (you think >40% likely)
- BUY NO at 0.40: profit if event does NOT happen (you think <40% likely)
- Spread between bid/ask indicates liquidity

Review ALL {len(candidates)} markets and select up to {self.max_markets} \
for deeper research.

THIS IS A 14-DAY COMPETITION. The only reliable P&L path is resolution gains
— markets that actually RESOLVE in our favor during the competition window.
A position in "2028 elections" or "2029 policy" will never resolve; it just
drifts on sentiment and bleeds spread costs every tick.

STRONGLY PREFER markets that:
- Resolve within the next 2-4 weeks (look for near-term dates in the question)
- Are about ongoing events with imminent conclusions: active playoffs, votes
  happening this week, scheduled announcements, earnings in the next few days
- Have high 24h volume (>$5,000) — volume signals near-term resolution activity

GOOD REASONS TO SELECT:
- The event resolves soon and recent news may not be priced in yet
- An ongoing sports series, trial, vote, or announcement is days away
- The probability seems clearly off vs. publicly available information

SKIP markets where:
- Price is below 0.10 or above 0.90 (near resolution, limited upside)
- The question contains a year ≥ 2027 (too far out to resolve in competition)
- The event is more than 6 weeks away (won't resolve; pure sentiment drift)
- You have no way to research or form a view
- Question is too vague or ambiguous

DIVERSITY RULE: Select at most 2 markets from any single category. \
Aim for markets from at least 3 different categories. \
Each market's category is shown in [brackets] after its line.

OUTPUT RULES — read carefully:
- Output ONLY a JSON object. No prose, no markdown fences, no function calls.
- Start your response with {{ and end with }}
- Use this exact schema:
{{
  "review": [
    {{
      "market_id": "<exact ID from the list>",
      "priority": <integer, 1 = highest priority>,
      "queries": ["<web search query 1>", "<query 2>", "<query 3>"],
      "rationale": "<one concise sentence>"
    }}
  ]
}}"""

        user_prompt = (
            f"Current tick: {tick_ctx.tick_ts}\n"
            f"Cash available: ${float(tick_ctx.cash):,.0f}\n"
            f"{positions_text}\n"
            f"All {len(candidates)} candidate markets "
            f"(ID | Question | Bid/Ask | 24h Volume):\n"
            f"{candidates_text}\n\n"
            f"Select up to {self.max_markets} markets worth researching."
            f"{memory_block}"
        )

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        # Gemini 2.5 Flash counts thinking tokens against max_tokens.
        # With ~8k thinking tokens, 8192 only leaves ~300 for output
        # (≈2 markets). 32768 gives 24k tokens of headroom for output.
        response = self.llm_client.generate(
            LLMRequest(messages=messages, max_tokens=32768)
        )
        text = response.content or ""

        logger.debug("TextReview raw response: %d chars", len(text))

        parsed = _parse_review_json(text)
        if parsed is None:
            logger.error(
                "TextReview: could not parse JSON from response: %.200s", text
            )
            return {"review": []}

        raw_list = parsed.get("review", [])
        if not isinstance(raw_list, list):
            logger.error("TextReview: 'review' value is not a list")
            return {"review": []}

        valid: list[dict] = []
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            mid = item.get("market_id")
            queries = item.get("queries")
            if not mid or not isinstance(queries, list) or not queries:
                continue
            valid.append({
                "market_id":  str(mid),
                "priority":   int(item.get("priority", len(valid) + 1)),
                "queries":    [str(q) for q in queries if q][:3],
                "rationale":  str(item.get("rationale", "")),
            })

        logger.info(
            "TextReview: recovered %d markets from JSON output", len(valid)
        )
        return {"review": valid}
