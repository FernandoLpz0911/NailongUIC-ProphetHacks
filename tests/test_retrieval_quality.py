from retrieval.quality import rank_documents, recency_boost, score_source
from retrieval.search import SearchDocument


def test_score_source_prefers_reuters_over_reddit() -> None:
    reuters = score_source("https://www.reuters.com/world/us/example-article")
    reddit = score_source("https://www.reddit.com/r/politics/comments/abc123/title/")
    assert reuters > reddit


def test_score_source_boosts_gov() -> None:
    gov = score_source("https://www.sec.gov/news/press-release")
    generic = score_source("https://example-blog.net/random-post")
    assert gov > generic


def test_rank_documents_puts_reuters_above_reddit() -> None:
    docs = [
        SearchDocument(
            title="Discussion thread",
            url="https://www.reddit.com/r/worldnews/comments/xyz",
            snippet="Users speculate on outcome.",
        ),
        SearchDocument(
            title="Official update",
            url="https://www.reuters.com/world/europe/story-idUS123",
            snippet="Leaders met today to discuss the deal.",
        ),
        SearchDocument(
            title="Pinterest pin",
            url="https://www.pinterest.com/pin/12345/",
            snippet="Infographic summary.",
        ),
    ]
    ranked = rank_documents(docs)
    assert "reuters.com" in ranked[0].url


def test_recency_boost_recent_snippet() -> None:
    assert recency_boost("Markets moved after report published 3 hours ago", None) > 0.0
    assert recency_boost("Historical analysis from 2019", None) == 0.0
