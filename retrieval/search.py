from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import httpx

from agent.config import EXA_API_KEY, TAVILY_API_KEY
from retrieval.cache import DiskCache

logger = logging.getLogger(__name__)
_search_cache = DiskCache("search")


@dataclass
class SearchDocument:
    title: str
    url: str
    snippet: str


async def search(query: str, *, limit: int = 5, event_id: str = "") -> list[SearchDocument]:
    cache_key = f"{event_id}|{query}|{limit}"
    cached = _search_cache.get(cache_key)
    if cached is not None:
        return [SearchDocument(**doc) for doc in cached]

    docs: list[SearchDocument] = []
    if TAVILY_API_KEY:
        try:
            docs = await _search_tavily(query, limit)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.warning("Tavily rate limited (429); trying fallback providers")
                docs = await _search_with_fallback(query, limit)
            else:
                raise
    elif EXA_API_KEY:
        docs = await _search_exa(query, limit)
    else:
        docs = await _search_duckduckgo(query, limit)

    if docs:
        _search_cache.set(cache_key, [asdict(d) for d in docs])
    return docs


async def _search_with_fallback(query: str, limit: int) -> list[SearchDocument]:
    if EXA_API_KEY:
        try:
            return await _search_exa(query, limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Exa fallback failed: %s", exc)
    return await _search_duckduckgo(query, limit)


async def _search_tavily(query: str, limit: int) -> list[SearchDocument]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": limit,
                "include_answer": False,
            },
        )
        response.raise_for_status()
        results = response.json().get("results") or []
    return [
        SearchDocument(
            title=r.get("title") or "",
            url=r.get("url") or "",
            snippet=r.get("content") or r.get("snippet") or "",
        )
        for r in results[:limit]
    ]


async def _search_exa(query: str, limit: int) -> list[SearchDocument]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": EXA_API_KEY, "Content-Type": "application/json"},
            json={"query": query, "numResults": limit, "type": "auto"},
        )
        response.raise_for_status()
        results = response.json().get("results") or []
    return [
        SearchDocument(
            title=r.get("title") or "",
            url=r.get("url") or "",
            snippet=r.get("text") or r.get("snippet") or "",
        )
        for r in results[:limit]
    ]


async def _search_duckduckgo(query: str, limit: int) -> list[SearchDocument]:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("duckduckgo-search not installed; returning empty search results")
        return _stub_docs(query, limit)

    docs: list[SearchDocument] = []
    try:
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=limit):
                docs.append(
                    SearchDocument(
                        title=item.get("title") or "",
                        url=item.get("href") or "",
                        snippet=item.get("body") or "",
                    )
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("DuckDuckGo search failed: %s", exc)
        return _stub_docs(query, limit)
    return docs or _stub_docs(query, limit)


def _stub_docs(query: str, limit: int) -> list[SearchDocument]:
    return [
        SearchDocument(
            title=f"No live search — configure TAVILY_API_KEY ({query[:40]}…)",
            url="https://local/stub",
            snippet="Add TAVILY_API_KEY or EXA_API_KEY for real news retrieval.",
        )
    ][:limit]
