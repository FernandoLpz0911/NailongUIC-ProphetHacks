from __future__ import annotations

from agent.schemas import PredictRequest
from retrieval.search import SearchDocument, search


async def build_context(request: PredictRequest, *, max_chunks: int = 10) -> list[SearchDocument]:
    """Build deduplicated news context for an event (query generation in Stage 2)."""
    query = f"{request.title} {request.rules[:120]}"
    docs = await search(query, limit=max_chunks)
    seen: set[str] = set()
    unique: list[SearchDocument] = []
    for doc in docs:
        if doc.url in seen:
            continue
        seen.add(doc.url)
        unique.append(doc)
    return unique[:max_chunks]
