from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchDocument:
    title: str
    url: str
    snippet: str


async def search(query: str, *, limit: int = 5) -> list[SearchDocument]:
    """
    Web search entry point (Tavily / Exa / fallback).

    Stage 1 stub returns placeholder docs so P2 can iterate without blocking P1.
    """
    _ = query
    return [
        SearchDocument(
            title=f"Stub result {index}",
            url=f"https://example.com/{index}",
            snippet="Replace with Tavily/Exa integration in Stage 2.",
        )
        for index in range(1, min(limit, 3) + 1)
    ]
