from __future__ import annotations

import logging
import re

import httpx

from agent.schemas import PredictRequest
from retrieval.search import SearchDocument

logger = logging.getLogger(__name__)

_POLYMARKET_URL = "https://gamma-api.polymarket.com/markets"
_STOPWORDS = frozenset(
    {
        "will",
        "the",
        "be",
        "been",
        "have",
        "has",
        "had",
        "this",
        "that",
        "with",
        "from",
        "into",
        "over",
        "under",
        "after",
        "before",
        "when",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "than",
        "then",
        "there",
        "their",
        "they",
        "them",
        "and",
        "for",
        "not",
        "are",
        "was",
        "were",
        "can",
        "may",
        "any",
        "all",
        "out",
        "off",
        "how",
        "why",
        "yes",
        "no",
    }
)


async def fetch_market_context(request: PredictRequest) -> list[SearchDocument]:
    """Build 1–3 documents from Kalshi stats and optional Polymarket public API."""
    docs: list[SearchDocument] = []

    kalshi_snippet = _format_kalshi_stats(request)
    if kalshi_snippet:
        docs.append(
            SearchDocument(
                title="Current Kalshi market prices",
                url="https://kalshi.com/",
                snippet=kalshi_snippet,
            )
        )

    poly_docs = await _fetch_polymarket_matches(request.title, limit=2)
    docs.extend(poly_docs)

    return docs[:3]


def _format_kalshi_stats(request: PredictRequest) -> str:
    if not request.market_stats:
        return ""

    lines: list[str] = []
    for market, stat in request.market_stats.items():
        parts: list[str] = [market]
        if stat.last_price is not None:
            parts.append(f"last={_fmt_price(stat.last_price)}")
        if stat.yes_ask is not None:
            parts.append(f"yes_ask={_fmt_price(stat.yes_ask)}")
        if stat.no_ask is not None:
            parts.append(f"no_ask={_fmt_price(stat.no_ask)}")
        lines.append(" | ".join(parts))

    header = f"Event: {request.title}\n"
    return header + "\n".join(lines)


def _fmt_price(value: float) -> str:
    if 0.0 <= value <= 1.0:
        return f"{value:.1%}"
    return f"{value:.4g}"


def _title_keywords(title: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9]{4,}", title.lower())
    return [w for w in words if w not in _STOPWORDS][:6]


async def _fetch_polymarket_matches(title: str, *, limit: int) -> list[SearchDocument]:
    keywords = _title_keywords(title)
    if not keywords:
        return []

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                _POLYMARKET_URL,
                params={"limit": 50, "active": "true", "closed": "false"},
            )
            response.raise_for_status()
            markets = response.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Polymarket fetch skipped: %s", exc)
        return []

    if not isinstance(markets, list):
        return []

    matched: list[SearchDocument] = []
    for market in markets:
        question = (market.get("question") or market.get("title") or "").lower()
        if not question or not any(kw in question for kw in keywords):
            continue

        outcome_prices = market.get("outcomePrices") or market.get("outcome_prices")
        volume = market.get("volume") or market.get("volumeNum")
        slug = market.get("slug") or market.get("id") or ""
        url = f"https://polymarket.com/event/{slug}" if slug else "https://polymarket.com/"

        snippet_parts = [f"Question: {market.get('question') or market.get('title')}"]
        if outcome_prices:
            snippet_parts.append(f"Outcome prices: {outcome_prices}")
        if volume is not None:
            snippet_parts.append(f"Volume: {volume}")

        matched.append(
            SearchDocument(
                title="Polymarket — related contract",
                url=url,
                snippet="\n".join(snippet_parts),
            )
        )
        if len(matched) >= limit:
            break

    return matched
