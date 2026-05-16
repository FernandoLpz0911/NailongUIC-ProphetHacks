"""SearchClient adapter that wires retrieval/retrieval.py into the SDK pipeline.

The SDK's `SearchStage` consumes any object exposing a synchronous
`search(query, limit) -> list[dict]` method returning dicts with the keys
{url, title, snippet, text, score}. That's what this adapter produces.

We deliberately do NOT call `retrieval.get_context(market_id, ...)` here —
the SDK's Review stage already generated a refined query. We just run that
single query through Tavily/DDG via `retrieval.raw_search()`, apply our
quality/dedup filter, and reshape to the SDK schema.

The Polymarket second-market signal is fetched in `CalibratedForecastStage`
where the candidate question is available (the SearchStage interface does
not pass market context).
"""

from __future__ import annotations

import logging
from typing import Any

from retrieval.retrieval import (
    detect_category,
    filter_chunks,
    raw_search,
    sort_chunks,
)

logger = logging.getLogger(__name__)


class RetrievalSearchClient:
    """Drop-in replacement for `ai_prophet.trade.search.SearchClient`."""

    def __init__(self, *, max_results: int = 5) -> None:
        self.default_max_results = max_results

    def search(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Run one search query and return SDK-shape result dicts.

        Args:
            query: Search query produced by the SDK's Review stage.
            limit: Max number of results to return (SDK passes this positionally).

        Returns:
            List of {url, title, snippet, text, score} dicts. Empty list on any
            failure (the SDK SearchStage tolerates empty results).
        """
        n = int(limit or self.default_max_results)
        try:
            raw = raw_search(query, max_results=n)
        except Exception as e:
            logger.warning("RetrievalSearchClient.search('%s') failed: %s", query[:60], e)
            return []

        # filter_chunks expects the internal Tavily/DDG shape (`url`, no `text`).
        # raw_search already returned SDK shape with both `snippet` and `text`,
        # so we filter on URL/domain directly without round-tripping.
        seen: set[str] = set()
        filtered: list[dict[str, Any]] = []
        # Reuse the blocked-domain logic from retrieval.filter_chunks by
        # building a list it understands, then mapping back.
        internal = [{"url": r["url"], "content": r.get("text", ""), "title": r.get("title", "")} for r in raw]
        filtered_internal = filter_chunks(internal)

        category = detect_category(query)
        sorted_internal = sort_chunks(filtered_internal, category)

        for it in sorted_internal:
            url = it.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            content = it.get("content", "") or ""
            filtered.append({
                "url":     url,
                "title":   it.get("title", ""),
                "snippet": content[:280],
                "text":    content,
                "score":   1.0,
            })

        logger.debug(
            "RetrievalSearchClient.search('%s') -> %d results (filtered from %d)",
            query[:60], len(filtered), len(raw),
        )
        return filtered[:n]

    def close(self) -> None:
        """No-op. The underlying Tavily/DDG/httpx clients are stateless or
        manage their own resources."""
        return None
