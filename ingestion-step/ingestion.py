"""
Retrieval module for the Prophet trading agent.

Pipeline position: feeds into Stage 2 (Hypothesis)
  Extraction → [Retrieval] → Hypothesis → Verify

Andres owns this file (P2 — Retrieval Engineer).

Responsibilities:
  - Detect market category (crypto, politics, economics, sports, science)
  - Generate focused search queries per category via Gemini Flash
  - Execute web searches (Tavily primary, DuckDuckGo fallback)
  - Cache results to disk — same market never searched twice in 24 hours
  - Filter low-quality sources (Reddit threads, content farms)
  - Return clean, ranked context snippets for the hypothesis prompt
"""

from __future__ import annotations

import hashlib
import json
import logging
import shelve
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from prophet_agent.constants import TICK_LEASE_SECONDS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache config
# ---------------------------------------------------------------------------

CACHE_DIR = Path(".cache")
CACHE_FILE = CACHE_DIR / "retrieval_cache"
CACHE_TTL_HOURS = 24

# ---------------------------------------------------------------------------
# Source quality config
# ---------------------------------------------------------------------------

# Domains that add noise — skip results from these
_BLOCKED_DOMAINS = {
    "reddit.com",
    "quora.com",
    "answers.yahoo.com",
    "ask.com",
    "ehow.com",
    "wikihow.com",
    "buzzfeed.com",
    "dailymail.co.uk",
    "thesun.co.uk",
    "nypost.com",
}

# Domains that are high-quality — boost these to the top
_PREFERRED_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "ft.com",
    "wsj.com",
    "bloomberg.com",
    "economist.com",
    "politico.com",
    "axios.com",
    "federalreserve.gov",
    "sec.gov",
    "coindesk.com",
    "coingecko.com",
    "espn.com",
    "sports-reference.com",
}

# ---------------------------------------------------------------------------
# Category detection
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "crypto": [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain",
        "defi", "nft", "stablecoin", "altcoin", "binance", "coinbase",
    ],
    "politics": [
        "election", "president", "congress", "senate", "bill", "vote",
        "democrat", "republican", "parliament", "prime minister", "referendum",
        "legislation", "policy", "government", "federal",
    ],
    "economics": [
        "gdp", "inflation", "fed", "federal reserve", "interest rate",
        "cpi", "recession", "unemployment", "earnings", "stock", "s&p",
        "nasdaq", "dow", "treasury", "bond", "yield", "tariff", "trade",
    ],
    "sports": [
        "nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball",
        "baseball", "hockey", "championship", "playoff", "tournament",
        "super bowl", "world cup", "olympics", "match", "game",
    ],
    "science": [
        "fda", "clinical trial", "study", "research", "vaccine", "drug",
        "approval", "nasa", "space", "launch", "climate", "emissions",
    ],
}

# Search sources to prioritize per category
_CATEGORY_SEARCH_SOURCES: dict[str, list[str]] = {
    "crypto": ["coindesk.com", "coingecko.com", "bloomberg.com", "reuters.com"],
    "politics": ["politico.com", "reuters.com", "apnews.com", "axios.com"],
    "economics": ["ft.com", "bloomberg.com", "reuters.com", "wsj.com", "federalreserve.gov"],
    "sports": ["espn.com", "sports-reference.com", "apnews.com"],
    "science": ["reuters.com", "apnews.com", "bbc.com", "nature.com"],
    "general": ["reuters.com", "apnews.com", "bbc.com", "axios.com"],
}


def detect_category(question: str) -> str:
    """
    Detect the market category from the question text.

    Returns one of: "crypto", "politics", "economics", "sports", "science", "general"
    """
    q = question.lower()
    scores: dict[str, int] = {cat: 0 for cat in _CATEGORY_KEYWORDS}

    for category, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                scores[category] += 1

    best = max(scores, key=lambda c: scores[c])
    detected = best if scores[best] > 0 else "general"
    logger.debug("Category detected: %s for question: %.60s...", detected, question)
    return detected


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class RetrievalResult(BaseModel):
    """Context returned to the hypothesis stage."""

    market_id: str
    category: str
    queries_used: list[str]
    snippets: list[str]          # Top ranked, deduplicated text chunks
    sources: list[str]           # Source URLs or domain labels
    cached: bool
    retrieved_at: str


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(market_id: str, queries: list[str]) -> str:
    """Deterministic cache key from market_id + queries."""
    raw = f"{market_id}:{'|'.join(sorted(queries))}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _is_cache_valid(entry: dict[str, Any]) -> bool:
    """Check if a cache entry is still within TTL."""
    try:
        cached_at = datetime.fromisoformat(entry["retrieved_at"])
        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - cached_at < timedelta(hours=CACHE_TTL_HOURS)
    except (KeyError, ValueError):
        return False


def _load_from_cache(key: str) -> dict[str, Any] | None:
    """Load a retrieval result from disk cache if valid."""
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        with shelve.open(str(CACHE_FILE)) as db:
            entry = db.get(key)
            if entry and _is_cache_valid(entry):
                logger.debug("Cache HIT for key %s", key[:8])
                return entry
    except Exception as e:
        logger.warning("Cache read error: %s", e)
    return None


def _save_to_cache(key: str, data: dict[str, Any]) -> None:
    """Save a retrieval result to disk cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        with shelve.open(str(CACHE_FILE)) as db:
            db[key] = data
        logger.debug("Cache WRITE for key %s", key[:8])
    except Exception as e:
        logger.warning("Cache write error: %s", e)


# ---------------------------------------------------------------------------
# Query generation (Gemini Flash)
# ---------------------------------------------------------------------------

async def _generate_queries(
    question: str,
    category: str,
    openrouter_api_key: str,
    n_queries: int = 3,
    timeout: float = 20.0,
) -> list[str]:
    """
    Use Gemini Flash to generate focused search queries for a market question.

    Returns 2–3 specific queries tailored to the category.
    Falls back to a single keyword query if the LLM call fails.
    """
    category_hint = f"This is a {category} prediction market." if category != "general" else ""

    prompt = (
        f"{category_hint}\n"
        f"Prediction market question: {question}\n\n"
        f"Generate {n_queries} specific, focused web search queries to find "
        f"the most relevant recent information that would help predict the outcome. "
        f"Prefer queries that target authoritative sources.\n\n"
        f'Return ONLY a JSON array of strings: ["query 1", "query 2", "query 3"]'
    )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "google/gemini-2.5-flash",
                    "temperature": 0.0,
                    "max_tokens": 200,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(raw)

            # Handle both {"queries": [...]} and bare [...] responses
            if isinstance(parsed, list):
                queries = parsed
            elif isinstance(parsed, dict):
                queries = parsed.get("queries", parsed.get("0", [question]))
            else:
                queries = [question]

            return [str(q) for q in queries[:n_queries] if q]

    except Exception as e:
        logger.warning("Query generation failed (%s) — using raw question as query", e)
        return [question]


# ---------------------------------------------------------------------------
# Web search (Tavily primary, DuckDuckGo fallback)
# ---------------------------------------------------------------------------

async def _search_tavily(
    query: str,
    api_key: str,
    max_results: int = 5,
    timeout: float = 15.0,
) -> list[dict[str, str]]:
    """Search via Tavily API. Returns list of {url, title, content} dicts."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_answer": False,
                    "include_raw_content": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return [
                {
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                }
                for r in data.get("results", [])
            ]
    except Exception as e:
        logger.warning("Tavily search failed for query '%s': %s", query[:50], e)
        return []


async def _search_duckduckgo(
    query: str,
    max_results: int = 5,
) -> list[dict[str, str]]:
    """
    DuckDuckGo fallback search (no API key required).
    Uses duckduckgo-search library if available.
    """
    try:
        from duckduckgo_search import DDGS  # type: ignore[import]

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "url": r.get("href", ""),
                    "title": r.get("title", ""),
                    "content": r.get("body", ""),
                })
        return results
    except ImportError:
        logger.warning("duckduckgo-search not installed — no fallback search available")
        return []
    except Exception as e:
        logger.warning("DuckDuckGo fallback failed: %s", e)
        return []


async def _execute_search(
    query: str,
    tavily_api_key: str | None,
    max_results: int = 5,
) -> list[dict[str, str]]:
    """
    Execute a search with automatic fallback:
    Tavily (if key provided) → DuckDuckGo → empty list
    """
    if tavily_api_key:
        results = await _search_tavily(query, tavily_api_key, max_results)
        if results:
            return results
        logger.info("Tavily returned empty — falling back to DuckDuckGo")

    return await _search_duckduckgo(query, max_results)


# ---------------------------------------------------------------------------
# Source filtering and ranking
# ---------------------------------------------------------------------------

def _get_domain(url: str) -> str:
    """Extract root domain from a URL."""
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def _score_result(result: dict[str, str], category: str, days_old: int = 0) -> float:
    """
    Score a search result for ranking.

    Higher = better. Factors:
    - Preferred domain: +2.0
    - Blocked domain: -10.0 (effectively filtered)
    - Category-specific preferred domain: +1.0
    - Content length (proxy for depth): up to +0.5
    - Recency boost: newer = higher (if date info available)
    """
    domain = _get_domain(result.get("url", ""))
    score = 0.0

    if domain in _BLOCKED_DOMAINS:
        return -10.0

    if domain in _PREFERRED_DOMAINS:
        score += 2.0

    category_sources = _CATEGORY_SEARCH_SOURCES.get(category, [])
    if domain in category_sources:
        score += 1.0

    content_len = len(result.get("content", ""))
    score += min(content_len / 2000, 0.5)

    # Recency: penalize older results slightly
    score -= days_old * 0.05

    return score


def _filter_and_rank(
    results: list[dict[str, str]],
    category: str,
    max_snippets: int = 10,
) -> tuple[list[str], list[str]]:
    """
    Filter blocked domains, deduplicate, rank, and return top snippets + sources.

    Returns:
        (snippets, sources) — parallel lists of text chunks and their URLs
    """
    seen_content: set[str] = set()
    scored: list[tuple[float, dict[str, str]]] = []

    for r in results:
        domain = _get_domain(r.get("url", ""))
        if domain in _BLOCKED_DOMAINS:
            continue

        content = r.get("content", "").strip()
        if not content:
            continue

        # Deduplicate by content fingerprint (first 100 chars)
        fingerprint = content[:100].lower()
        if fingerprint in seen_content:
            continue
        seen_content.add(fingerprint)

        score = _score_result(r, category)
        if score > -1.0:
            scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    snippets = []
    sources = []
    for _, r in scored[:max_snippets]:
        title = r.get("title", "")
        content = r.get("content", "")
        url = r.get("url", "")
        domain = _get_domain(url)

        snippet = f"[{domain}] {title}: {content[:400]}" if title else f"[{domain}] {content[:400]}"
        snippets.append(snippet)
        sources.append(url or domain)

    return snippets, sources


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_retrieval(
    market_id: str,
    question: str,
    openrouter_api_key: str,
    tavily_api_key: str | None = None,
    *,
    max_snippets: int = 10,
    results_per_query: int = 5,
    n_queries: int = 3,
    timeout: float = 30.0,
) -> RetrievalResult:
    """
    Full retrieval pipeline for a single market.

    Steps:
      1. Detect category (rule-based keyword match)
      2. Check disk cache — return immediately if fresh hit
      3. Generate focused search queries via Gemini Flash
      4. Execute searches (Tavily → DuckDuckGo fallback)
      5. Filter, deduplicate, rank results
      6. Save to cache
      7. Return RetrievalResult

    Args:
        market_id:           From the extraction stage.
        question:            The market question text.
        openrouter_api_key:  For Gemini Flash query generation.
        tavily_api_key:      Tavily search API key (optional; falls back to DDG).
        max_snippets:        Max text chunks to return to hypothesis.
        results_per_query:   Max search results per query.
        n_queries:           Number of search queries to generate.
        timeout:             Per-search HTTP timeout.

    Returns:
        RetrievalResult with ranked snippets and source URLs.
    """
    start = time.monotonic()
    category = detect_category(question)

    # --- Step 1: Check cache ---
    queries_preview = [question]  # used for cache key before real queries generated
    cache_key = _cache_key(market_id, queries_preview)
    cached_entry = _load_from_cache(cache_key)

    if cached_entry:
        return RetrievalResult(**cached_entry, cached=True)

    # --- Step 2: Generate search queries ---
    queries = await _generate_queries(question, category, openrouter_api_key, n_queries)
    logger.info("Retrieval [%s] category=%s queries=%s", market_id[:20], category, queries)

    # Re-check cache with real queries
    cache_key = _cache_key(market_id, queries)
    cached_entry = _load_from_cache(cache_key)
    if cached_entry:
        return RetrievalResult(**cached_entry, cached=True)

    # --- Step 3: Execute searches ---
    all_results: list[dict[str, str]] = []
    for query in queries:
        results = await _execute_search(query, tavily_api_key, results_per_query)
        all_results.extend(results)
        logger.debug("Query '%s' → %d results", query[:50], len(results))

    # --- Step 4: Filter, deduplicate, rank ---
    snippets, sources = _filter_and_rank(all_results, category, max_snippets)

    elapsed = time.monotonic() - start
    logger.info(
        "Retrieval [%s] done: %d snippets from %d raw results in %.1fs",
        market_id[:20],
        len(snippets),
        len(all_results),
        elapsed,
    )

    # --- Step 5: Cache and return ---
    result_data = {
        "market_id": market_id,
        "category": category,
        "queries_used": queries,
        "snippets": snippets,
        "sources": sources,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_to_cache(cache_key, result_data)

    return RetrievalResult(**result_data, cached=False)


# ---------------------------------------------------------------------------
# Context string builder (called by hypothesis.py)
# ---------------------------------------------------------------------------

def build_context_string(retrieval: RetrievalResult) -> str:
    """
    Format a RetrievalResult into a single context string for the hypothesis prompt.

    Returns a clean, numbered list of snippets with source labels.
    Returns a fallback message if no snippets were found.
    """
    if not retrieval.snippets:
        return (
            "No relevant news found. "
            "Rely on base rates and current market price as the primary signal."
        )

    lines = [f"Category: {retrieval.category.upper()}\n"]
    for i, (snippet, source) in enumerate(
        zip(retrieval.snippets, retrieval.sources), start=1
    ):
        lines.append(f"{i}. {snippet}")

    return "\n".join(lines)