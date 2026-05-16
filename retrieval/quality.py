from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from retrieval.search import SearchDocument

if TYPE_CHECKING:
    from retrieval.category import Category

_TRUSTED_NEWS = ("reuters.com", "apnews.com")
_OFFICIAL_DOMAINS = (
    "whitehouse.gov",
    "sec.gov",
    "treasury.gov",
    "federalreserve.gov",
    "who.int",
    "un.org",
    "europa.eu",
)
_LOW_QUALITY = ("reddit.com/r/", "pinterest.")
_CONTENT_FARMS = (
    "contentfarm",
    "beforeitsnews",
    "naturalnews",
    "thegatewaypundit",
    "worldnewsdailyreport",
)

_RELATIVE_RECENCY = re.compile(
    r"\b(\d{1,2})\s*(minute|hour|day)s?\s+ago\b",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_MONTH_DAY = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+(\d{1,2})(?:,?\s*(20\d{2}))?\b",
    re.IGNORECASE,
)
_MONTH_NUM = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_NEAR_TERM_DAYS = 7
_NEAR_TERM_STALE_PENALTY = -0.6
_PREFERRED_DOMAIN_BOOST = 0.35

_RESOLUTION_MONTH_DAY_YEAR = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\w*\s+(\d{1,2}),?\s+(20\d{2})\b",
    re.IGNORECASE,
)
_RESOLUTION_BEFORE_MONTH = re.compile(
    r"\b(?:by|before|until|on)\s+"
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\w*(?:\s+(\d{1,2}))?,?\s*(20\d{2})\b",
    re.IGNORECASE,
)
_RESOLUTION_MONTH_YEAR = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\w*\s+(20\d{2})\b",
    re.IGNORECASE,
)
_RESOLUTION_YEAR_ONLY = re.compile(r"\b(20\d{2})\b")
_IN_DAYS = re.compile(r"\b(?:in|within)\s+(\d{1,2})\s+days?\b", re.IGNORECASE)


def is_near_term_event(title: str, rules: str) -> bool:
    """True when the event appears to resolve within the next 7 days."""
    combined = f"{title} {rules}"
    today = _today()

    in_days = _IN_DAYS.search(combined)
    if in_days is not None and int(in_days.group(1)) < _NEAR_TERM_DAYS:
        return True

    dates = _parse_resolution_dates(combined)
    if not dates:
        return False

    horizon = today + timedelta(days=_NEAR_TERM_DAYS)
    return any(today <= resolved <= horizon for resolved in dates)


def preferred_domain_boost(url: str, preferred_domains: tuple[str, ...]) -> float:
    if not preferred_domains:
        return 0.0
    host = urlparse(url.lower()).netloc.removeprefix("www.")
    if any(domain in host for domain in preferred_domains):
        return _PREFERRED_DOMAIN_BOOST
    return 0.0


def near_term_freshness_adjustment(
    snippet: str,
    title: str,
    *,
    near_term: bool,
    published_hint: str | None = None,
) -> float:
    """Strongly penalize stale snippets when the event resolves soon."""
    if not near_term:
        return 0.0
    text = " ".join(part for part in (title, snippet, published_hint or "") if part)
    if recency_boost(text, None) > 0:
        return 0.15
    return _NEAR_TERM_STALE_PENALTY


def score_source(url: str, *, preferred_domains: tuple[str, ...] = ()) -> float:
    """Higher is better. Boost trusted news and official sources; penalize low-quality domains."""
    lower = url.lower()
    host = urlparse(lower).netloc.removeprefix("www.")

    score = 0.0

    if any(domain in host or domain in lower for domain in _TRUSTED_NEWS):
        score += 0.45
    if host.endswith(".gov") or ".gov/" in lower:
        score += 0.4
    if any(official in host for official in _OFFICIAL_DOMAINS):
        score += 0.3
    if any(part in lower for part in ("/press-release", "/newsroom", "investor.")):
        score += 0.15

    if any(bad in lower for bad in _LOW_QUALITY):
        score -= 0.55
    if any(farm in lower for farm in _CONTENT_FARMS):
        score -= 0.5

    score += preferred_domain_boost(url, preferred_domains)
    return score


def recency_boost(snippet: str, published_hint: str | None) -> float:
    """Boost when text suggests publication within the last 48 hours."""
    text = " ".join(part for part in (snippet, published_hint or "") if part).strip()
    if not text:
        return 0.0

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=48)

    relative = _RELATIVE_RECENCY.search(text)
    if relative:
        amount = int(relative.group(1))
        unit = relative.group(2).lower()
        if unit.startswith("minute") or unit.startswith("hour"):
            return 0.35
        if unit.startswith("day") and amount <= 2:
            return 0.25

    for match in _ISO_DATE.finditer(text):
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            published = datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            continue
        if published >= cutoff:
            return 0.3

    month_match = _MONTH_DAY.search(text)
    if month_match:
        month_key = month_match.group(1)[:3].lower()
        month = _MONTH_NUM.get(month_key)
        if month is not None:
            day = int(month_match.group(2))
            year = int(month_match.group(3)) if month_match.group(3) else now.year
            try:
                published = datetime(year, month, day, tzinfo=timezone.utc)
            except ValueError:
                published = None
            if published is not None and published >= cutoff:
                return 0.25

    return 0.0


def _document_score(
    doc: SearchDocument,
    *,
    preferred_domains: tuple[str, ...] = (),
    near_term: bool = False,
) -> float:
    return (
        score_source(doc.url, preferred_domains=preferred_domains)
        + recency_boost(doc.snippet, None)
        + recency_boost(doc.title, None)
        + near_term_freshness_adjustment(
            doc.snippet,
            doc.title,
            near_term=near_term,
        )
    )


def rank_documents(
    docs: list[SearchDocument],
    *,
    category: Category | None = None,
    near_term: bool = False,
) -> list[SearchDocument]:
    """Sort documents by composite quality score (descending)."""
    preferred: tuple[str, ...] = ()
    if category is not None:
        from retrieval.profiles import preferred_domains_for

        preferred = preferred_domains_for(category)

    return sorted(
        docs,
        key=lambda doc: _document_score(
            doc,
            preferred_domains=preferred,
            near_term=near_term,
        ),
        reverse=True,
    )


def _today() -> date:
    return date.today()


def _parse_resolution_dates(text: str) -> list[date]:
    found: list[date] = []
    for match in _RESOLUTION_MONTH_DAY_YEAR.finditer(text):
        found.append(_resolution_date(match.group(1), int(match.group(2)), int(match.group(3))))
    for match in _RESOLUTION_BEFORE_MONTH.finditer(text):
        day = int(match.group(2)) if match.group(2) else 1
        found.append(_resolution_date(match.group(1), day, int(match.group(3))))
    for match in _RESOLUTION_MONTH_YEAR.finditer(text):
        month = _month_key_to_num(match.group(1))
        if month is not None:
            found.append(date(int(match.group(2)), month, 28))
    for match in _RESOLUTION_YEAR_ONLY.finditer(text):
        found.append(date(int(match.group(1)), 12, 31))
    return found


def _resolution_date(month_token: str, day: int, year: int) -> date:
    month = _month_key_to_num(month_token)
    if month is None:
        return date(year, 12, 31)
    try:
        return date(year, month, min(day, 28))
    except ValueError:
        return date(year, month, 28)


def _month_key_to_num(token: str) -> int | None:
    return _MONTH_NUM.get(token[:3].lower())
