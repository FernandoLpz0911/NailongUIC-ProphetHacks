from __future__ import annotations

import asyncio
import json
import logging
import re

from agent.config import CHEAP_MODEL, MODEL_CALL_TIMEOUT_SECONDS, OPENROUTER_API_KEY
from agent.openrouter.client import OpenRouterClient
from agent.schemas import PredictRequest
from retrieval.category import detect_category
from retrieval.profiles import category_search_queries

logger = logging.getLogger(__name__)


async def generate_search_queries(request: PredictRequest, *, max_queries: int = 3) -> list[str]:
    """Return 2–3 focused search queries for an event."""
    if OPENROUTER_API_KEY:
        try:
            return await _llm_queries(request, max_queries)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM query generation failed, using heuristics: %s", exc)
    return _heuristic_queries(request, max_queries)


async def _llm_queries(request: PredictRequest, max_queries: int) -> list[str]:
    client = OpenRouterClient()
    prompt = (
        f"Event: {request.title}\nRules: {request.rules[:400]}\n"
        f"Return JSON: {{\"queries\": [\"q1\", \"q2\"]}} with {max_queries} focused web search queries."
    )
    payload = await asyncio.wait_for(
        client.chat(
            [{"role": "user", "content": prompt}],
            model=CHEAP_MODEL,
            temperature=0.0,
            max_tokens=256,
            timeout=45.0,
        ),
        timeout=MODEL_CALL_TIMEOUT_SECONDS,
    )
    text = OpenRouterClient.extract_text(payload)
    data = json.loads(_extract_json(text))
    queries = [str(q).strip() for q in data.get("queries", []) if str(q).strip()]
    base = queries[:max_queries] or _heuristic_queries(request, max_queries)
    return _merge_category_queries(request, base, max_queries)


def _heuristic_queries(request: PredictRequest, max_queries: int) -> list[str]:
    title = request.title.strip()
    year_match = re.search(r"20\d{2}", request.rules + title)
    year = year_match.group(0) if year_match else "2026"
    queries = [
        f"{title} latest news {year}",
        f"{title} forecast odds",
        f"{title} {request.rules[:60].strip()}",
    ]
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            unique.append(q)
    return _merge_category_queries(request, unique[:max_queries], max_queries)


def _merge_category_queries(
    request: PredictRequest,
    queries: list[str],
    max_queries: int,
) -> list[str]:
    category = detect_category(request.title, request.rules)
    extras = category_search_queries(category, request.title, request.rules)
    if not extras:
        return queries[:max_queries]

    seen = {q.casefold() for q in queries}
    merged = list(queries)
    for q in extras:
        key = q.casefold()
        if key not in seen:
            seen.add(key)
            merged.append(q)
    cap = max_queries + len(extras) if category != "general" else max_queries
    return merged[:cap]


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON in query-gen response")
    return text[start : end + 1]
