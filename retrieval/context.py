from __future__ import annotations

import asyncio
import logging

from agent.schemas import PredictRequest
from retrieval.market_context import fetch_market_context
from retrieval.category import detect_category
from retrieval.quality import is_near_term_event, rank_documents
from retrieval.query_gen import generate_search_queries
from retrieval.search import SearchDocument, search

logger = logging.getLogger(__name__)

BLOCKED_DOMAINS = ("reddit.com/r/", "pinterest.", "contentfarm")


async def build_context(
    request: PredictRequest,
    *,
    max_chunks: int = 10,
    results_per_query: int = 5,
) -> tuple[list[SearchDocument], bool]:
    """
    Build deduplicated news context.

    Returns (documents, used_live_search).
    """
    market_docs = await fetch_market_context(request)
    category = detect_category(request.title, request.rules)
    near_term = is_near_term_event(request.title, request.rules)
    queries = await generate_search_queries(request)
    batches = await asyncio.gather(
        *[search(q, limit=results_per_query, event_id=request.event_id) for q in queries]
    )

    used_live = False
    seen: set[str] = set()
    ranked: list[SearchDocument] = []

    for docs in batches:
        for doc in docs:
            if "local/stub" in doc.url:
                continue
            used_live = True
            if _is_low_quality(doc.url):
                continue
            if doc.url in seen:
                continue
            seen.add(doc.url)
            ranked.append(doc)

    if not ranked:
        ranked = [d for batch in batches for d in batch]

    ranked = rank_documents(ranked, category=category, near_term=near_term)
    combined = market_docs + ranked
    return combined[:max_chunks], used_live


def _is_low_quality(url: str) -> bool:
    lower = url.lower()
    return any(domain in lower for domain in BLOCKED_DOMAINS)
