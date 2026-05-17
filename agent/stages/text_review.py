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
        # Nailong Elite: shrink the candidate list to the top 60 by volume to
        # prevent MAX_TOKENS truncation on the review output. The original
        # behavior sent all ~200 candidates (~30k chars) which truncated
        # the response to 2-3 markets instead of 10.
        ranked = sorted(
            candidates,
            key=lambda m: (getattr(m, "volume_24h", 0) or 0),
            reverse=True,
        )
        top_candidates = ranked[:60]

        # Annotate each candidate with days_to_resolution so the LLM can
        # prefer short-duration markets (we need >=10 resolutions in 14 days
        # for win-rate / Sharpe / Brier to populate).
        import datetime as _dt
        now = _dt.datetime.now(_dt.timezone.utc)

        def _days_to_res(m: CandidateMarket) -> float | None:
            close_time = getattr(m, "close_time", None) or getattr(m, "end_date", None)
            if close_time is None:
                return None
            try:
                if hasattr(close_time, "tzinfo"):
                    delta = close_time - now
                else:
                    delta = _dt.datetime.fromisoformat(str(close_time)).replace(
                        tzinfo=_dt.timezone.utc
                    ) - now
                return max(0.0, delta.total_seconds() / 86400.0)
            except Exception:
                return None

        def _fmt(m: CandidateMarket) -> str:
            days = _days_to_res(m)
            d_str = f"{days:.0f}d" if days is not None else "?d"
            return (
                f"{m.market_id} | {m.question[:80]} | "
                f"{m.yes_bid:.2f}/{m.yes_ask:.2f} | ${m.volume_24h:.0f} | {d_str} "
                f"[{_category(m.market_id, m.question)}]"
            )

        candidates_text = "\n".join(_fmt(m) for m in top_candidates)

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
This is a 14-day evaluation window — we need markets that will RESOLVE within
that window so we accumulate enough fills for the metrics to populate.

HOW PREDICTION MARKETS WORK:
- Price = probability (0.50 = 50% chance of YES)
- BUY YES at 0.40: profit if event happens (you think >40% likely)
- BUY NO at 0.40: profit if event does NOT happen (you think <40% likely)
- Spread between bid/ask indicates liquidity

Review the {len(top_candidates)} markets below (sorted by 24h volume) and \
select up to {self.max_markets} for deeper research.

STRONG PREFERENCE — DURATION:
The last column shows days-to-resolution (e.g. "21d"). STRONGLY prefer markets
resolving within 30 days. We need actual resolutions to evaluate. Markets
resolving in 2027+ should make up at most 2 of your selections, and only if
they have an exceptional edge.

GOOD REASONS TO SELECT:
- Short-duration market with active trading (your top priority)
- You have domain knowledge about the topic
- Recent news/events may not be fully priced in
- The probability seems off based on base rates or logic
- High volume indicates active trading interest

SKIP markets where:
- Price is below 0.10 or above 0.90 (near resolution, limited upside)
- You have no way to research or form a view
- Question is too vague or ambiguous
- Days-to-resolution >365 unless you have very high conviction

DIVERSITY RULE: Select at most 2 markets from any single category and at most
2 markets from the same underlying event (e.g. don't pick multiple "2028 Dem
nominee" sub-markets — they are mutually exclusive bets).
Aim for markets from at least 3 different categories. Each market's category
is shown in [brackets] after its line.

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
            f"Top {len(top_candidates)} candidate markets by volume "
            f"(ID | Question | Bid/Ask | 24h Volume | Days-to-Resolution | Category):\n"
            f"{candidates_text}\n\n"
            f"Select up to {self.max_markets} markets, prioritizing short-duration "
            f"high-volume opportunities."
            f"{memory_block}"
        )

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        # Bumped from 8192 to 16384 — live logs showed MAX_TOKENS truncation
        # at 325 output tokens, getting only 2-3 markets instead of 10.
        response = self.llm_client.generate(
            LLMRequest(messages=messages, max_tokens=16384)
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
