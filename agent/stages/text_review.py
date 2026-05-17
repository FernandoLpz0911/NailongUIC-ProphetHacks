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
from datetime import datetime, timezone

from ai_prophet.trade.agent.stages.review import ReviewStage
from ai_prophet.trade.agent.utils import format_portfolio_summary
from ai_prophet.trade.core import TickContext
from ai_prophet.trade.core.tick_context import CandidateMarket
from ai_prophet.trade.llm import LLMMessage
from ai_prophet.trade.llm.base import LLMRequest

logger = logging.getLogger(__name__)

# Near-expiry strategy constants.
NEAR_EXPIRY_SECS = 3_600   # ≤ 1 h — strongly preferred, relaxed price filter
MAX_HORIZON_SECS = 7_200   # 2 h outer cutoff; drop everything farther out

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


def _secs_to_expiry(market: CandidateMarket, ref_ts: datetime | None) -> float | None:
    if ref_ts is None:
        return None
    res = getattr(market, "resolution_time", None)
    if not isinstance(res, datetime):
        return None
    if res.tzinfo is None:
        res = res.replace(tzinfo=timezone.utc)
    if ref_ts.tzinfo is None:
        ref_ts = ref_ts.replace(tzinfo=timezone.utc)
    return (res - ref_ts).total_seconds()


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _clean(text: str) -> str:
    """Strip markdown fences and trailing commas."""
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1)
    return _TRAILING_COMMA_RE.sub(r"\1", text.strip())


def _extract_objects(s: str) -> list[dict]:
    """Pull every complete top-level JSON object out of a possibly-truncated string."""
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

    # 4. Scan inside the "review": [ array directly. This handles the case where
    # the outer object is never closed because the model was cut off mid-entry.
    arr_match = re.search(r'"review"\s*:\s*\[', cleaned)
    if arr_match:
        entries = _extract_objects(cleaned[arr_match.end() - 1:])
        if entries:
            return {"review": entries}

    return None


class TextReviewStage(ReviewStage):
    """ReviewStage variant that uses plain-text JSON output instead of tool calling.

    Identical selection logic and prompt as the SDK's ReviewStage, but the
    system prompt instructs the model to emit a raw JSON object in its reply
    rather than invoking the submit_review tool. This sidesteps Gemini's
    MALFORMED_FUNCTION_CALL behaviour entirely.
    """

    @property
    def name(self) -> str:
        return "review"

    def _generate_review(
        self,
        candidates: Sequence[CandidateMarket],
        tick_ctx: TickContext,
    ) -> dict:
        # --- Near-expiry filter & sort -----------------------------------
        ref_ts = getattr(tick_ctx, "tick_ts", None)
        if isinstance(ref_ts, datetime) and ref_ts.tzinfo is None:
            ref_ts = ref_ts.replace(tzinfo=timezone.utc)

        active: list[CandidateMarket] = []
        for m in candidates:
            secs = _secs_to_expiry(m, ref_ts)
            mid = (float(m.yes_bid) + float(m.yes_ask)) / 2.0

            if secs is not None and secs <= 0:
                continue  # already resolved
            if secs is not None and secs > MAX_HORIZON_SECS:
                continue  # farther than 2 h — skip entirely

            # Near-expiry markets can be near-boundary and still have edge;
            # use a relaxed price filter. Standard filter applies otherwise.
            if secs is not None and secs <= NEAR_EXPIRY_SECS:
                if not (0.03 < mid < 0.97):
                    continue
            else:
                if not (0.08 < mid < 0.92):
                    continue
            active.append(m)

        # Soonest-expiring first; tiebreak on liquidity.
        active.sort(key=lambda m: (
            _secs_to_expiry(m, ref_ts) or 9_999_999,
            -float(m.volume_24h or 0),
        ))
        candidates_list = active[:80]

        logger.info(
            "Review: %d candidates after near-expiry filter (≤%ds), %d shown to LLM",
            len(active), MAX_HORIZON_SECS, len(candidates_list),
        )

        def _expiry_label(m: CandidateMarket) -> str:
            secs = _secs_to_expiry(m, ref_ts)
            if secs is None:
                return "?min"
            return f"{int(secs / 60)}min"

        candidates_text = "\n".join(
            f"{m.market_id} | {m.question[:80]} | "
            f"{m.yes_bid:.2f}/{m.yes_ask:.2f} | ${m.volume_24h:.0f} "
            f"| exp-in {_expiry_label(m)} [{_category(m.market_id, m.question)}]"
            for m in candidates_list
        )

        positions_text = format_portfolio_summary(tick_ctx, include_positions=True)
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

STRATEGY — NEAR-EXPIRY FOCUS:
All markets listed expire within the next 2 hours (shown as "exp-in Xmin").
STRONGLY PREFER markets expiring within 60 minutes — prices move fastest near
resolution, and you collect your payoff quickly with minimal duration risk.

HOW PREDICTION MARKETS WORK:
- Price = probability (0.50 = 50% chance of YES)
- BUY YES at 0.65: profit if event resolves YES (you think >65% likely)
- BUY NO at 0.35: profit if event resolves NO (you think <35% likely)
- Near expiry the market is often MISPRICED if you have fresher information

Review ALL {len(candidates_list)} markets and select up to {self.max_markets} \
for deeper research.

GOOD REASONS TO SELECT:
- Expiring very soon (< 60 min) — fastest payoff, clearest signal
- You can verify the likely outcome with a quick search RIGHT NOW
- Recent news/events are not yet priced in (market is stale)
- High volume indicates the market is liquid and fills are reliable

SKIP markets where:
- You have absolutely no way to verify the current state of affairs
- Question is too vague or ambiguous to research quickly
- Price is below 0.05 or above 0.95 (very little upside left)

DIVERSITY RULE: Select at most 2 markets from any single category. \
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
      "queries": ["<web search query>", "<web search query>", "<web search query>"],
      "rationale": "<one concise sentence>"
    }}
  ]
}}"""

        user_prompt = (
            f"Current tick: {tick_ctx.tick_ts}\n"
            f"Cash available: ${float(tick_ctx.cash):,.0f}\n"
            f"{positions_text}\n"
            f"{len(candidates_list)} candidate markets expiring within 2 hours "
            f"(ID | Question | Bid/Ask | 24h Volume | exp-in Xmin [category]):\n"
            f"{candidates_text}\n\n"
            f"Select up to {self.max_markets} markets worth researching. "
            f"Prioritise those expiring soonest where you can verify the outcome now."
            f"{memory_block}"
        )

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        response = self.llm_client.generate(
            LLMRequest(messages=messages, max_tokens=8192)
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
                "queries":    [str(q) for q in queries if q],
                "rationale":  str(item.get("rationale", "")),
            })

        logger.info("TextReview: recovered %d markets from full JSON output", len(valid))
        return {"review": valid}
