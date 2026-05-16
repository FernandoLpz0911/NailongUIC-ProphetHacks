import os
import json
import diskcache
from datetime import datetime
from dotenv import load_dotenv
from tavily import TavilyClient
from openai import OpenAI

load_dotenv()

cache = diskcache.Cache("./search_cache", timeout=60 * 60 * 24)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
router = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

QUERY_SYSTEM_PROMPT = """You are a research assistant helping fact-check prediction market events.
Given an event title and resolution rules, generate 2-3 focused web search queries
that would find the most relevant, recent news to help forecast this event.

Return ONLY a JSON array of strings. No explanation, no markdown, no extra text.
Example: ["Fed rate decision June 2026", "FOMC meeting outcome June 2026"]"""


def get_context(event_id: str, title: str, rules: str) -> dict:
    """
    Main entry point for P3's forecaster.
    Returns a structured context dict with news chunks and retrieval confidence.
    Cached by event_id for 24 hours.

    Return shape:
    {
        "event_id": str,
        "retrieved_at": str (ISO 8601),
        "confidence": "high" | "medium" | "low",
        "chunks": [
            {
                "title": str,
                "url": str,
                "snippet": str,
                "source": str,
                "published_date": str
            }
        ]
    }
    """
    if event_id in cache:
        print(f"[cache hit]  {event_id}")
        return cache[event_id]

    queries = generate_queries(title, rules)
    print(f"[queries]    {queries}")

    chunks = []
    for query in queries:
        try:
            results = tavily.search(
                query=query,
                max_results=5,
                search_depth="advanced"
            )
            chunks.extend(results.get("results", []))
        except Exception as e:
            print(f"[tavily err] query='{query}' error={e}")

    deduped = dedupe(chunks)[:10]

    clean_chunks = [
        {
            "title": c.get("title", ""),
            "url": c.get("url", ""),
            "snippet": c.get("content", ""),
            "source": c.get("url", "").split("/")[2] if c.get("url") else "",
            "published_date": c.get("published_date", "")
        }
        for c in deduped
    ]

    result = {
        "event_id": event_id,
        "retrieved_at": datetime.utcnow().isoformat() + "Z",
        "confidence": compute_confidence(clean_chunks),
        "chunks": clean_chunks
    }

    cache.set(event_id, result, expire=60 * 60 * 24)
    print(f"[cache miss] {event_id} — stored {len(clean_chunks)} chunks, confidence={result['confidence']}")
    return result


def generate_queries(title: str, rules: str) -> list[str]:
    """
    LLM-generated search queries via OpenRouter.
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
        raw = response.choices[0].message.content.strip()
        queries = json.loads(raw)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries[:3]
    except Exception as e:
        print(f"[query gen err] {e} — falling back to title")

    return [title]


def compute_confidence(chunks: list[dict]) -> str:
    """
    Simple confidence signal based on chunk count.
    P3 uses this to set alpha (market-anchoring weight).
    high   -> alpha 0.6-0.8 (lean on model, deviate from market)
    medium -> alpha 0.4-0.5 (blend evenly)
    low    -> alpha 0.2-0.3 (stay close to market, don't risk it)
    """
    if len(chunks) >= 7:
        return "high"
    elif len(chunks) >= 3:
        return "medium"
    return "low"


def dedupe(chunks: list[dict]) -> list[dict]:
    """Remove duplicate URLs, keep first occurrence."""
    seen = set()
    out = []
    for chunk in chunks:
        url = chunk.get("url", "")
        if url not in seen:
            seen.add(url)
            out.append(chunk)
    return out