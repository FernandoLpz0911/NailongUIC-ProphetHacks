from datetime import date
from unittest.mock import patch

from retrieval.category import detect_category
from retrieval.profiles import category_search_queries, preferred_domains_for
from retrieval.quality import (
    is_near_term_event,
    near_term_freshness_adjustment,
    preferred_domain_boost,
    rank_documents,
)
from retrieval.query_gen import _merge_category_queries
from retrieval.search import SearchDocument
from agent.schemas import PredictRequest


def _request(title: str, rules: str = "Resolves YES if the stated outcome occurs.") -> PredictRequest:
    return PredictRequest(
        event_id="EVT_CAT",
        title=title,
        markets=["Yes", "No"],
        rules=rules,
    )


def test_detect_politics() -> None:
    assert detect_category("Will the Senate pass the bill?", "election vote congress") == "politics"
    assert detect_category("2026 presidential election winner", "") == "politics"


def test_detect_crypto() -> None:
    assert detect_category("Will Bitcoin hit $100k?", "BTC price threshold") == "crypto"
    assert detect_category("Ethereum ETF approval", "crypto token listing") == "crypto"


def test_detect_sports() -> None:
    assert detect_category("Super Bowl 2027 winner", "NFL championship") == "sports"
    assert detect_category("NBA Finals MVP", "playoff series outcome") == "sports"


def test_detect_general() -> None:
    assert detect_category("Will it rain in Chicago tomorrow?", "Weather resolves YES if measurable.") == "general"


def test_category_search_queries_for_crypto() -> None:
    queries = category_search_queries("crypto", "Bitcoin ETF", "rules text")
    assert len(queries) == 2
    assert any("CoinGecko" in q for q in queries)
    assert any("Yahoo Finance" in q for q in queries)


def test_merge_category_queries_adds_extras() -> None:
    req = _request("Will Bitcoin reach $150k?", "Crypto resolves on price.")
    merged = _merge_category_queries(req, ["base query"], max_queries=3)
    assert len(merged) > 1
    assert merged[0] == "base query"


@patch("retrieval.quality._today")
def test_is_near_term_event_within_week(mock_today) -> None:
    mock_today.return_value = date(2026, 5, 16)
    assert is_near_term_event(
        "Will the bill pass?",
        "Resolves YES if passed on or before May 20, 2026.",
    )


@patch("retrieval.quality._today")
def test_is_near_term_event_far_future(mock_today) -> None:
    mock_today.return_value = date(2026, 5, 16)
    assert not is_near_term_event(
        "Long horizon event",
        "Resolves YES if the event occurs on or before December 31, 2028.",
    )


def test_is_near_term_event_in_days_phrase() -> None:
    assert is_near_term_event("Quick resolution", "Market resolves within 3 days of announcement.")


def test_preferred_domain_boost() -> None:
    assert preferred_domain_boost("https://www.espn.com/nba/story", ("espn.com",)) > 0.0
    assert preferred_domain_boost("https://example.com/post", ("espn.com",)) == 0.0


def test_near_term_penalizes_stale_snippet() -> None:
    stale = near_term_freshness_adjustment("Historical recap from 2019", "Title", near_term=True)
    fresh = near_term_freshness_adjustment("Updated 2 hours ago with breaking news", "Title", near_term=True)
    assert stale < 0
    assert fresh > stale


@patch("retrieval.quality._today")
def test_rank_documents_prefers_category_domain(mock_today) -> None:
    mock_today.return_value = date(2026, 5, 16)
    docs = [
        SearchDocument(
            title="Blog take",
            url="https://random-blog.example/post",
            snippet="Opinion piece without a date.",
        ),
        SearchDocument(
            title="ESPN update",
            url="https://www.espn.com/nba/story",
            snippet="Game recap published May 15, 2026.",
        ),
    ]
    ranked = rank_documents(docs, category="sports", near_term=False)
    assert "espn.com" in ranked[0].url


def test_preferred_domains_for_politics() -> None:
    domains = preferred_domains_for("politics")
    assert "reuters.com" in domains
    assert "politico.com" in domains
