"""Retrieval module for the Prophet Hacks Trading-Track agent.

Surfaces:
    get_context(market_id, title, rules, resolution_date=None) -> dict
        Full retrieval: LLM-generated queries -> Tavily/DDG fan-out -> filter
        & rank -> Polymarket second-market lookup -> CSV log -> cached.

    raw_search(query, max_results=5) -> list[dict]
        Single-query passthrough used by the SDK SearchClient adapter; returns
        SDK-shaped dicts ({url, title, snippet, text, score}).

    fetch_polymarket(question) -> dict | None
        Standalone Polymarket consensus lookup the ForecastStage calls
        per-market (cached for the tick lifetime).

    detect_category(question) -> str
    category_for_market_id(market_id, question) -> str
        Convenience helpers for category-aware routing.

    prewarm_cache(events) -> None
        Bulk warm before the eval window opens.

Cache and CSV log live under `data/` so every run artifact is in one place
(`data/cache/search`, `data/retrieval_log.csv`).
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import diskcache
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = _DATA_DIR / "cache" / "search"
CACHE_TTL_SECONDS = 60 * 60 * 24
RAW_SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL_SECONDS", str(4 * 60 * 60)))  # 4 hours
CSV_LOG = _DATA_DIR / "retrieval_log.csv"

cache = diskcache.Cache(str(CACHE_DIR))

_tavily_key = os.getenv("TAVILY_API_KEY") or ""
tavily_client: TavilyClient | None = (
    TavilyClient(api_key=_tavily_key) if _tavily_key else None
)

_openrouter_key = os.getenv("OPENROUTER_API_KEY") or ""
router: OpenAI | None = (
    OpenAI(
        api_key=_openrouter_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    if _openrouter_key
    else None
)


HIGH_QUALITY_DOMAINS = {
    "reuters.com", "apnews.com", "axios.com",
    "bbc.com", "bbc.co.uk", "npr.org",
    "bloomberg.com", "ft.com", "wsj.com",
    "federalreserve.gov", "sec.gov", "treasury.gov",
    "politico.com", "thehill.com", "rollcall.com",
    "whitehouse.gov", "congress.gov",
    "coindesk.com", "cointelegraph.com", "decrypt.co",
    "espn.com", "nba.com", "nfl.com", "mlb.com", "fifa.com",
    "europa.eu", "un.org", "who.int", "nih.gov", "cdc.gov",
}

BLOCKED_DOMAINS = {
    "reddit.com", "quora.com", "pinterest.com",
    "buzzfeed.com", "dailymail.co.uk", "thesun.co.uk",
    "infowars.com", "naturalnews.com", "breitbart.com",
    "tmz.com", "answers.yahoo.com",
}


CATEGORY_PROFILES = {
    "crypto": {
        "keywords": [
            "bitcoin", "btc", "ethereum", "eth", "crypto",
            "blockchain", "defi", "token", "altcoin", "stablecoin",
        ],
        "priority_domains": [
            "coindesk.com", "cointelegraph.com", "coingecko.com", "decrypt.co",
        ],
    },
    "finance": {
        "keywords": [
            "fed", "fomc", "rate", "inflation", "gdp", "unemployment",
            "stock", "market", "economy", "treasury", "recession", "cpi",
        ],
        "priority_domains": [
            "bloomberg.com", "ft.com", "wsj.com", "federalreserve.gov",
        ],
    },
    "politics": {
        "keywords": [
            "election", "president", "congress", "senate", "vote",
            "poll", "party", "government", "minister", "parliament", "referendum",
        ],
        "priority_domains": [
            "politico.com", "reuters.com", "apnews.com", "thehill.com",
        ],
    },
    "sports": {
        "keywords": [
            "nba", "nfl", "mlb", "nhl", "soccer", "football",
            "basketball", "baseball", "championship", "playoff", "tournament", "fifa",
        ],
        "priority_domains": [
            "espn.com", "nba.com", "nfl.com", "mlb.com",
        ],
    },
    "general": {"keywords": [], "priority_domains": []},
}

QUERY_SYSTEM_PROMPT = """You are a research assistant helping fact-check prediction market events.
Given an event title and resolution rules, generate 2-3 focused web search queries
that would find the most relevant, recent news to help forecast this event.

Return ONLY a JSON array of strings. No explanation, no markdown, no extra text.
Example: ["Fed rate decision June 2026", "FOMC meeting outcome June 2026"]"""

CSV_HEADERS = [
    "market_id",
    "retrieved_at",
    "category",
    "confidence",
    "chunk_count",
    "high_quality_count",
    "polymarket_yes_price",
    "polymarket_no_price",
    "polymarket_volume",
    "top_sources",
    "top_titles",
]


def log_to_csv(result: dict) -> None:
    """Append one row per retrieval result for offline inspection."""
    path = CSV_LOG
    write_header = not path.exists()
    chunks = result.get("chunks", [])
    mh = result.get("market_history") or {}
    n_high = sum(1 for c in chunks if c.get("quality") == "high")
    seen_sources: set[str] = set()
    unique_sources: list[str] = []
    for c in chunks[:5]:
        s = c.get("source", "")
        if s and s not in seen_sources:
            seen_sources.add(s)
            unique_sources.append(s)
    top_sources = " | ".join(unique_sources)
    top_titles = " | ".join(c.get("title", "")[:60] for c in chunks[:3])

    row = {
        "market_id":            result.get("market_id", result.get("event_id", "")),
        "retrieved_at":         result.get("retrieved_at", ""),
        "category":             result.get("category", ""),
        "confidence":           result.get("confidence", ""),
        "chunk_count":          len(chunks),
        "high_quality_count":   n_high,
        "polymarket_yes_price": mh.get("yes_price", ""),
        "polymarket_no_price":  mh.get("no_price", ""),
        "polymarket_volume":    mh.get("volume", ""),
        "top_sources":          top_sources,
        "top_titles":           top_titles,
    }

    try:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except OSError as e:
        # Logging is best-effort; never break the agent over a log write.
        print(f"[csv log err] {e}")


def get_context(
    market_id: str | None = None,
    title: str = "",
    rules: str = "",
    resolution_date: str | None = None,
    *,
    event_id: str | None = None,  # backward-compatible alias
) -> dict:
    """Full retrieval pipeline for one market.

    Returns a dict with the keys: market_id, retrieved_at, category, confidence,
    market_history (Polymarket signal or None), chunks (list of news snippets).
    """
    mid = market_id or event_id or ""
    if not mid:
        raise ValueError("get_context requires market_id (or legacy event_id)")

    if mid in cache:
        result = cache[mid]
        log_to_csv(result)
        return result

    category = detect_category(title)
    queries = generate_queries(title, rules)

    raw_chunks = fetch_with_fallback(queries)
    filtered = filter_chunks(raw_chunks)
    boosted = apply_recency_boost(filtered, resolution_date)
    sorted_ = sort_chunks(boosted, category)
    clean_chunks = [format_chunk(c) for c in sorted_[:10]]

    market_history = fetch_polymarket(title)

    result = {
        "market_id":      mid,
        "event_id":       mid,  # back-compat
        "retrieved_at":   datetime.now(timezone.utc).isoformat(),
        "category":       category,
        "confidence":     compute_confidence(clean_chunks),
        "market_history": market_history,
        "chunks":         clean_chunks,
    }

    cache.set(mid, result, expire=CACHE_TTL_SECONDS)
    log_to_csv(result)
    return result


def raw_search(query: str, max_results: int = 5) -> list[dict]:
    """Single-query Tavily/DDG fan-out used by the SDK SearchClient adapter.

    Returns SDK-shaped dicts: {url, title, snippet, text, score}.
    Filtering (blocked domains, dedup) and category-aware ranking happen in
    the adapter so this function stays a thin wrapper around the providers.

    Results are cached for RAW_SEARCH_CACHE_TTL seconds (default 4 hours) to
    avoid burning search credits on identical queries across ticks.
    """
    cache_key = f"raw:{query}:{max_results}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    chunks = _search_one_query(query, max_results=max_results)
    result = [_to_sdk_shape(c) for c in chunks]
    if result:
        cache.set(cache_key, result, expire=RAW_SEARCH_CACHE_TTL)
    return result


def _search_one_query(query: str, max_results: int = 5) -> list[dict]:
    """Tavily primary, DDG fallback. Never raises; returns at worst an empty list."""
    if tavily_client is not None:
        try:
            results = tavily_client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
            )
            hits = results.get("results", []) or []
            if hits:
                return hits
        except Exception as e:
            print(f"[tavily err] '{query}' -> {e}; falling back to DuckDuckGo")

    try:
        from duckduckgo_search import DDGS

        out: list[dict] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                out.append({
                    "title":          r.get("title", ""),
                    "url":            r.get("href", ""),
                    "content":        r.get("body", ""),
                    "published_date": "",
                })
        return out
    except Exception as e:
        print(f"[ddg err] '{query}' -> {e}")
        return []


def _to_sdk_shape(chunk: dict) -> dict:
    """Convert a Tavily/DDG result dict to the SDK SearchClient shape."""
    url = chunk.get("url", "")
    content = chunk.get("content", "") or ""
    return {
        "url":     url,
        "title":   chunk.get("title", ""),
        "snippet": content[:280],
        "text":    content,
        "score":   chunk.get("score", 1.0),
    }


def detect_category(title: str) -> str:
    """Keyword-based category detection from a market title."""
    lower = (title or "").lower()
    for category, profile in CATEGORY_PROFILES.items():
        if category == "general":
            continue
        if any(kw in lower for kw in profile["keywords"]):
            return category
    return "general"


def category_for_market_id(market_id: str, question: str) -> str:
    """Stable category label for a (market_id, question) pair.

    market_id is reserved for future use; today we just delegate to
    detect_category(question). Useful for callers that want a deterministic
    label without needing to import CATEGORY_PROFILES.
    """
    return detect_category(question)


def generate_queries(title: str, rules: str) -> list[str]:
    """LLM-generated search queries via OpenRouter Gemini Flash Lite.

    Falls back to the raw title on any provider error or missing key.
    """
    if router is None or not title:
        return [title] if title else []

    try:
        response = router.chat.completions.create(
            model="google/gemini-2.5-flash-lite",
            messages=[
                {"role": "system", "content": QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Title: {title}\n\nRules: {rules}"},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        raw = (response.choices[0].message.content or "").strip()
        queries = json.loads(raw)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries[:3]
    except Exception as e:
        print(f"[query gen err] {e}; falling back to title")
    return [title]


def fetch_with_fallback(queries: list[str]) -> list[dict]:
    """Run every query through Tavily; fall back to DDG on first failure."""
    chunks: list[dict] = []
    tavily_failed = tavily_client is None

    for query in queries:
        if not tavily_failed and tavily_client is not None:
            try:
                results = tavily_client.search(
                    query=query,
                    max_results=5,
                    search_depth="basic",
                )
                chunks.extend(results.get("results", []) or [])
                continue
            except Exception as e:
                print(f"[tavily err] '{query}' -> {e}; switching to DuckDuckGo")
                tavily_failed = True

        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    chunks.append({
                        "title":          r.get("title", ""),
                        "url":            r.get("href", ""),
                        "content":        r.get("body", ""),
                        "published_date": "",
                    })
        except Exception as e:
            print(f"[ddg err] '{query}' -> {e}")

    return chunks


def get_domain(url: str) -> str:
    try:
        return url.split("/")[2].replace("www.", "")
    except Exception:
        return ""


def is_high_quality(domain: str) -> bool:
    if domain in BLOCKED_DOMAINS:
        return False
    if domain in HIGH_QUALITY_DOMAINS:
        return True
    return any(domain.endswith("." + hq) for hq in HIGH_QUALITY_DOMAINS)


def is_blocked(domain: str) -> bool:
    return domain in BLOCKED_DOMAINS


def filter_chunks(chunks: list[dict]) -> list[dict]:
    """Drop blocked domains and duplicate URLs."""
    seen: set[str] = set()
    out: list[dict] = []
    for c in chunks:
        url = c.get("url", "")
        domain = get_domain(url)
        if not url or url in seen or is_blocked(domain):
            continue
        seen.add(url)
        out.append(c)
    return out


def parse_date(date_str: str) -> datetime | None:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def apply_recency_boost(chunks: list[dict], resolution_date: str | None) -> list[dict]:
    """For events resolving within 7 days, surface last-48h articles first."""
    if not resolution_date:
        return chunks

    try:
        res_dt = datetime.fromisoformat(resolution_date).replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days_to_resolution = (res_dt - now).days
    except Exception:
        return chunks

    if days_to_resolution > 7:
        return chunks

    cutoff = now - timedelta(hours=48)
    recent: list[dict] = []
    the_rest: list[dict] = []
    for c in chunks:
        pub = parse_date(c.get("published_date", ""))
        (recent if pub and pub >= cutoff else the_rest).append(c)
    return recent + the_rest


def sort_chunks(chunks: list[dict], category: str) -> list[dict]:
    """Category-priority domains first, then high-quality, then everything else."""
    priority = set(CATEGORY_PROFILES.get(category, {}).get("priority_domains", []))

    def score(c: dict) -> int:
        domain = get_domain(c.get("url", ""))
        if domain in priority:
            return 0
        if is_high_quality(domain):
            return 1
        return 2

    return sorted(chunks, key=score)


def format_chunk(c: dict) -> dict:
    url = c.get("url", "")
    domain = get_domain(url)
    return {
        "title":          c.get("title", ""),
        "url":            url,
        "snippet":        c.get("content", ""),
        "source":         domain,
        "published_date": c.get("published_date", ""),
        "quality":        "high" if is_high_quality(domain) else "normal",
    }


def fetch_polymarket(title: str) -> dict | None:
    """Find a matching Polymarket market by question text and return its prices.

    First-class output field: the ForecastStage uses this as a second-market
    consensus signal alongside Kalshi's `market_stats`.

    Returns None on any error, no match, or a stale/low-volume market that
    shouldn't be trusted as a consensus signal.
    """
    if not title:
        return None
    try:
        query = " ".join(title.split()[:6])
        resp = httpx.get(
            "https://gamma-api.polymarket.com/markets",
            params={"q": query, "limit": 3, "active": "true"},
            timeout=10,
        )
        resp.raise_for_status()
        markets = resp.json()
        if not markets:
            return None

        top = markets[0]
        prices = top.get("outcomePrices", [None, None])
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except Exception:
                prices = [None, None]

        def _to_float(v):
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        yes_price = _to_float(prices[0] if len(prices) > 0 else None)
        no_price = _to_float(prices[1] if len(prices) > 1 else None)
        volume = _to_float(top.get("volume"))

        return {
            "source":    "polymarket",
            "question":  top.get("question", ""),
            "yes_price": yes_price,
            "no_price":  no_price,
            "volume":    volume,
            "url":       f"https://polymarket.com/event/{top.get('slug', '')}",
        }
    except Exception as e:
        print(f"[polymarket err] {e}")
        return None


def compute_confidence(chunks: list[dict]) -> str:
    """Confidence label used by the calibration step to pick alpha."""
    if not chunks:
        return "low"
    n_high = sum(1 for c in chunks if c.get("quality") == "high")
    pct_high = n_high / len(chunks)
    if len(chunks) >= 6 and pct_high >= 0.5:
        return "high"
    if len(chunks) >= 3 and pct_high >= 0.2:
        return "medium"
    return "low"


def prewarm_cache(events: list[dict]) -> None:
    """Bulk warm before the eval window opens."""
    total = len(events)
    print(f"[prewarm] starting cache prewarm for {total} events")
    for i, event in enumerate(events):
        mid = event.get("market_id") or event.get("event_id") or f"EVT_{i}"
        if mid in cache:
            print(f"[prewarm] {i+1}/{total} {mid} — cached, skipping")
            continue
        print(f"[prewarm] {i+1}/{total} {mid} — fetching")
        try:
            get_context(
                market_id=mid,
                title=event.get("title", "") or event.get("question", ""),
                rules=event.get("rules", ""),
                resolution_date=event.get("resolution_date"),
            )
        except Exception as e:
            print(f"[prewarm err] {mid} -> {e}")
    print(f"[prewarm] done; processed {total} events")
