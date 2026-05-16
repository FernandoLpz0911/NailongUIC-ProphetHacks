import os
import json
import httpx
import diskcache
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from tavily import TavilyClient
from openai import OpenAI
import csv
from pathlib import Path

load_dotenv()

# --- Clients ---
cache         = diskcache.Cache("./search_cache")
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
router        = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# ---------------------------------------------------------------------------
# Stage 3 — Source quality tiers
# ---------------------------------------------------------------------------

HIGH_QUALITY_DOMAINS = {
    # Wire / general news
    "reuters.com", "apnews.com", "axios.com",
    "bbc.com", "bbc.co.uk", "npr.org",
    # Finance
    "bloomberg.com", "ft.com", "wsj.com",
    "federalreserve.gov", "sec.gov", "treasury.gov",
    # Politics
    "politico.com", "thehill.com", "rollcall.com",
    "whitehouse.gov", "congress.gov",
    # Crypto
    "coindesk.com", "cointelegraph.com", "decrypt.co",
    # Sports
    "espn.com", "nba.com", "nfl.com", "mlb.com", "fifa.com",
    # Government / international
    "europa.eu", "un.org", "who.int", "nih.gov", "cdc.gov"
}

BLOCKED_DOMAINS = {
    "reddit.com", "quora.com", "pinterest.com",
    "buzzfeed.com", "dailymail.co.uk", "thesun.co.uk",
    "infowars.com", "naturalnews.com", "breitbart.com",
    "tmz.com", "answers.yahoo.com"
}

# ---------------------------------------------------------------------------
# Stage 4 — Per-category retrieval profiles
# ---------------------------------------------------------------------------

CATEGORY_PROFILES = {
    "crypto": {
        "keywords": [
            "bitcoin", "btc", "ethereum", "eth", "crypto",
            "blockchain", "defi", "token", "altcoin", "stablecoin"
        ],
        "priority_domains": ["coindesk.com", "cointelegraph.com", "coingecko.com", "decrypt.co"]
    },
    "finance": {
        "keywords": [
            "fed", "fomc", "rate", "inflation", "gdp", "unemployment",
            "stock", "market", "economy", "treasury", "recession", "cpi"
        ],
        "priority_domains": ["bloomberg.com", "ft.com", "wsj.com", "federalreserve.gov"]
    },
    "politics": {
        "keywords": [
            "election", "president", "congress", "senate", "vote",
            "poll", "party", "government", "minister", "parliament", "referendum"
        ],
        "priority_domains": ["politico.com", "reuters.com", "apnews.com", "thehill.com"]
    },
    "sports": {
        "keywords": [
            "nba", "nfl", "mlb", "nhl", "soccer", "football",
            "basketball", "baseball", "championship", "playoff", "tournament", "fifa"
        ],
        "priority_domains": ["espn.com", "nba.com", "nfl.com", "mlb.com"]
    },
    "general": {
        "keywords": [],
        "priority_domains": []
    }
}

QUERY_SYSTEM_PROMPT = """You are a research assistant helping fact-check prediction market events.
Given an event title and resolution rules, generate 2-3 focused web search queries
that would find the most relevant, recent news to help forecast this event.

Return ONLY a JSON array of strings. No explanation, no markdown, no extra text.
Example: ["Fed rate decision June 2026", "FOMC meeting outcome June 2026"]"""

CSV_LOG = "retrieval_log.csv"
CSV_HEADERS = [
    "event_id",
    "retrieved_at",
    "category",
    "confidence",
    "chunk_count",
    "high_quality_count",
    "polymarket_yes_price",
    "polymarket_no_price",
    "polymarket_volume",
    "top_sources",
    "top_titles"
]

def log_to_csv(result: dict) -> None:
    """
    Append one row per retrieval result to retrieval_log.csv.
    P3 can open this in Excel/Sheets to inspect context quality per event.
    """
    path       = Path(CSV_LOG)
    write_header = not path.exists()
    chunks     = result.get("chunks", [])
    mh         = result.get("market_history") or {}
    n_high     = sum(1 for c in chunks if c.get("quality") == "high")
    seen_sources = set()
    unique_sources = []
    for c in chunks[:5]:
        s = c["source"]
        if s not in seen_sources:
            seen_sources.add(s)
            unique_sources.append(s)
    top_sources = " | ".join(unique_sources)
    top_titles  = " | ".join(c["title"][:60] for c in chunks[:3])

    row = {
        "event_id":            result["event_id"],
        "retrieved_at":        result["retrieved_at"],
        "category":            result["category"],
        "confidence":          result["confidence"],
        "chunk_count":         len(chunks),
        "high_quality_count":  n_high,
        "polymarket_yes_price": mh.get("yes_price", ""),
        "polymarket_no_price":  mh.get("no_price", ""),
        "polymarket_volume":    mh.get("volume", ""),
        "top_sources":         top_sources,
        "top_titles":          top_titles
    }

    with open(CSV_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[csv log]    {result['event_id']} -> {CSV_LOG}")


# ===========================================================================
# Main entry point
# ===========================================================================

def get_context(
    event_id: str,
    title: str,
    rules: str,
    resolution_date: str | None = None   # ISO date string e.g. "2026-06-01"
) -> dict:
    """
    Main entry point for P3's forecaster.

    Return shape:
    {
        "event_id":       str,
        "retrieved_at":   str  (ISO 8601),
        "category":       str  ("crypto" | "finance" | "politics" | "sports" | "general"),
        "confidence":     str  ("high" | "medium" | "low"),
        "market_history": dict | None,   # Polymarket price data if found
        "chunks": [
            {
                "title":          str,
                "url":            str,
                "snippet":        str,
                "source":         str,
                "published_date": str,
                "quality":        "high" | "normal"
            }
        ]
    }

    P3 alpha mapping:
        confidence=high   -> alpha 0.6-0.8  (lean on model, deviate from market)
        confidence=medium -> alpha 0.4-0.5  (blend evenly)
        confidence=low    -> alpha 0.2-0.3  (stay close to market price)
    """
    if event_id in cache:
        print(f"[cache hit]  {event_id}")
        result = cache[event_id]
        log_to_csv(result)   
        return result

    category = detect_category(title)
    print(f"[category]   {event_id} -> {category}")

    queries = generate_queries(title, rules)
    print(f"[queries]    {queries}")

    raw_chunks   = fetch_with_fallback(queries)
    filtered     = filter_chunks(raw_chunks)
    boosted      = apply_recency_boost(filtered, resolution_date)
    sorted_      = sort_chunks(boosted, category)
    clean_chunks = [format_chunk(c) for c in sorted_[:10]]

    market_history = fetch_polymarket(title)

    result = {
        "event_id":       event_id,
        "retrieved_at":   datetime.now(timezone.utc).isoformat(),
        "category":       category,
        "confidence":     compute_confidence(clean_chunks),
        "market_history": market_history,
        "chunks":         clean_chunks
    }

    cache.set(event_id, result, expire=60 * 60 * 24)
    log_to_csv(result)
    print(
        f"[cache miss] {event_id} — "
        f"{len(clean_chunks)} chunks, "
        f"confidence={result['confidence']}, "
        f"polymarket={'yes' if market_history else 'no'}"
    )
    return result


# ===========================================================================
# Stage 4 — Category detection
# ===========================================================================

def detect_category(title: str) -> str:
    """Keyword-based category detection from event title."""
    lower = title.lower()
    for category, profile in CATEGORY_PROFILES.items():
        if category == "general":
            continue
        if any(kw in lower for kw in profile["keywords"]):
            return category
    return "general"


# ===========================================================================
# Stage 2 — Query generation
# ===========================================================================

def generate_queries(title: str, rules: str) -> list[str]:
    """
    LLM-generated search queries via OpenRouter (Gemini Flash Lite).
    Falls back to event title on any failure.
    """
    try:
        response = router.chat.completions.create(
            model="google/gemini-2.5-flash-lite",
            messages=[
                {"role": "system", "content": QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Title: {title}\n\nRules: {rules}"}
            ],
            temperature=0.2,
            max_tokens=200
        )
        raw     = response.choices[0].message.content.strip()
        queries = json.loads(raw)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries[:3]
    except Exception as e:
        print(f"[query gen err] {e} — falling back to title")

    return [title]


# ===========================================================================
# Stage 5 — Search with DuckDuckGo fallback
# ===========================================================================

def fetch_with_fallback(queries: list[str]) -> list[dict]:
    """
    Try Tavily for every query. If Tavily fails, fall back to DuckDuckGo.
    """
    chunks        = []
    tavily_failed = False

    for query in queries:
        if not tavily_failed:
            try:
                results = tavily_client.search(
                    query=query,
                    max_results=5,
                    search_depth="advanced"
                )
                chunks.extend(results.get("results", []))
                continue
            except Exception as e:
                print(f"[tavily err] '{query}' -> {e}. Switching to DuckDuckGo.")
                tavily_failed = True

        # DuckDuckGo fallback
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    chunks.append({
                        "title":          r.get("title", ""),
                        "url":            r.get("href", ""),
                        "content":        r.get("body", ""),
                        "published_date": ""
                    })
        except Exception as e:
            print(f"[ddg err] '{query}' -> {e}")

    return chunks


# ===========================================================================
# Stage 3 — Source filtering, recency boost, sorting
# ===========================================================================

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
    # Catch subdomains e.g. data.reuters.com
    return any(domain.endswith("." + hq) for hq in HIGH_QUALITY_DOMAINS)


def is_blocked(domain: str) -> bool:
    return domain in BLOCKED_DOMAINS


def filter_chunks(chunks: list[dict]) -> list[dict]:
    """Remove blocked domains and duplicate URLs."""
    seen = set()
    out  = []
    for c in chunks:
        url    = c.get("url", "")
        domain = get_domain(url)
        if url in seen or is_blocked(domain):
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
    """
    Stage 4: If the event resolves within 7 days, prepend articles
    from the last 48 hours so they rank first after sorting.
    """
    if not resolution_date:
        return chunks

    try:
        res_dt             = datetime.fromisoformat(resolution_date).replace(tzinfo=timezone.utc)
        now                = datetime.now(timezone.utc)
        days_to_resolution = (res_dt - now).days
    except Exception:
        return chunks

    if days_to_resolution > 7:
        return chunks

    cutoff   = now - timedelta(hours=48)
    recent   = []
    the_rest = []

    for c in chunks:
        pub = parse_date(c.get("published_date", ""))
        if pub and pub >= cutoff:
            recent.append(c)
        else:
            the_rest.append(c)

    print(f"[recency boost] {len(recent)} articles from last 48h (resolves in {days_to_resolution}d)")
    return recent + the_rest


def sort_chunks(chunks: list[dict], category: str) -> list[dict]:
    """
    Stage 3: Category-priority sources first, then high-quality, then rest.
    """
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
    url    = c.get("url", "")
    domain = get_domain(url)
    return {
        "title":          c.get("title", ""),
        "url":            url,
        "snippet":        c.get("content", ""),
        "source":         domain,
        "published_date": c.get("published_date", ""),
        "quality":        "high" if is_high_quality(domain) else "normal"
    }


# ===========================================================================
# Stage 3 — Polymarket market history
# ===========================================================================

def fetch_polymarket(title: str) -> dict | None:
    try:
        query = " ".join(title.split()[:6])
        resp  = httpx.get(
            "https://gamma-api.polymarket.com/markets",
            params={"q": query, "limit": 3, "active": "true"},
            timeout=10
        )
        resp.raise_for_status()
        markets = resp.json()

        if not markets:
            return None

        top    = markets[0]
        prices = top.get("outcomePrices", [None, None])

        # Polymarket returns prices as a JSON string e.g. '["0.62", "0.38"]'
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except Exception:
                prices = [None, None]

        return {
            "source":    "polymarket",
            "question":  top.get("question", ""),
            "yes_price": prices[0] if len(prices) > 0 else None,
            "no_price":  prices[1] if len(prices) > 1 else None,
            "volume":    top.get("volume", None),
            "url":       f"https://polymarket.com/event/{top.get('slug', '')}"
        }

    except Exception as e:
        print(f"[polymarket err] {e}")
        return None


# ===========================================================================
# Stage 3 — Confidence scoring (source-quality aware)
# ===========================================================================

def compute_confidence(chunks: list[dict]) -> str:
    """
    Confidence signal for P3's alpha (market-anchoring weight).

    Factors:
      - Number of chunks
      - Proportion of high-quality sources

    high   -> alpha 0.6-0.8  (deviate from market when evidence is strong)
    medium -> alpha 0.4-0.5  (blend model + market evenly)
    low    -> alpha 0.2-0.3  (stay close to market price, don't risk it)
    """
    if not chunks:
        return "low"

    n_high   = sum(1 for c in chunks if c.get("quality") == "high")
    pct_high = n_high / len(chunks)

    if len(chunks) >= 6 and pct_high >= 0.5:
        return "high"
    elif len(chunks) >= 3 and pct_high >= 0.2:
        return "medium"
    return "low"


# ===========================================================================
# Stage 5 — Cache prewarm
# ===========================================================================

def prewarm_cache(events: list[dict]) -> None:
    """
    Pre-fetch and cache context for a list of events.
    Run this before the 10-day eval window opens so the first prod call
    for each event is an instant cache hit.

    Usage:
        from retrieval import prewarm_cache
        import json
        events = json.load(open("data/events.json"))
        prewarm_cache(events[:50])
    """
    total = len(events)
    print(f"[prewarm] Starting cache prewarm for {total} events...")

    for i, event in enumerate(events):
        event_id = event.get("event_id", f"EVT_{i}")

        if event_id in cache:
            print(f"[prewarm] {i+1}/{total} {event_id} — already cached, skipping")
            continue

        print(f"[prewarm] {i+1}/{total} {event_id} — fetching...")
        try:
            get_context(
                event_id=event_id,
                title=event.get("title", ""),
                rules=event.get("rules", ""),
                resolution_date=event.get("resolution_date")
            )
        except Exception as e:
            print(f"[prewarm err] {event_id} — {e}")

    print(f"[prewarm] Done. {total} events processed.")