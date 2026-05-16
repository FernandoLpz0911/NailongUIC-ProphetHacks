import pytest

from retrieval.search import search


@pytest.mark.asyncio
async def test_search_returns_docs() -> None:
    docs = await search("Fed rate decision May 2026", event_id="test")
    assert len(docs) >= 1
    assert docs[0].title

    # Second call should hit cache (same result count)
    docs2 = await search("Fed rate decision May 2026", event_id="test")
    assert len(docs2) == len(docs)
