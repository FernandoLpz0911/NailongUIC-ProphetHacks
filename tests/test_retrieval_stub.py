import pytest

from retrieval.search import search


@pytest.mark.asyncio
async def test_search_stub_returns_docs() -> None:
    docs = await search("Fed rate decision May 2026")
    assert len(docs) >= 1
    assert docs[0].title
