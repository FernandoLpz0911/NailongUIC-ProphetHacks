from __future__ import annotations

from agent.schemas import PredictRequest
from forecasting.guards import count_real_docs, guard_flags
from retrieval.search import SearchDocument


def _request(rules: str = "Resolves YES if a general election is officially announced by March 31, 2026.") -> PredictRequest:
    return PredictRequest(
        event_id="evt-1",
        title="Test",
        markets=["Yes", "No"],
        rules=rules,
        market_stats={},
    )


def _doc(url: str = "https://reuters.com/a") -> SearchDocument:
    return SearchDocument(title="News", url=url, snippet="snippet")


def test_guard_short_rules() -> None:
    flags = guard_flags(_request(rules="TBD soon"), [_doc(), _doc("https://b.com")])
    assert flags.force_market is True


def test_guard_tbd_in_rules() -> None:
    rules = "Resolution date is TBD pending commission review in 2026."
    flags = guard_flags(_request(rules=rules), [_doc(), _doc("https://b.com")])
    assert flags.force_market is True


def test_guard_insufficient_real_docs() -> None:
    flags = guard_flags(_request(), [_doc("http://local/stub/1")])
    assert flags.force_market is True


def test_guard_passes_with_two_real_docs() -> None:
    flags = guard_flags(_request(), [_doc(), _doc("https://apnews.com/b")])
    assert flags.force_market is False


def test_count_real_docs_excludes_stubs() -> None:
    docs = [
        _doc("http://local/stub/x"),
        _doc(),
        SearchDocument(title="", url="", snippet=""),
    ]
    assert count_real_docs(docs) == 1
