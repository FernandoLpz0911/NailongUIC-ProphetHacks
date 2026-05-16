from __future__ import annotations

from dataclasses import dataclass

from retrieval.category import Category

_MAX_EXTRA_QUERIES = 2


@dataclass(frozen=True)
class RetrievalProfile:
    query_templates: tuple[str, ...]
    preferred_domains: tuple[str, ...]


_PROFILES: dict[Category, RetrievalProfile] = {
    "politics": RetrievalProfile(
        query_templates=(
            "{title} Politico latest news",
            "{title} Reuters politics analysis",
        ),
        preferred_domains=(
            "politico.com",
            "reuters.com",
            "apnews.com",
            "thehill.com",
            "whitehouse.gov",
        ),
    ),
    "crypto": RetrievalProfile(
        query_templates=(
            "{title} CoinGecko price news",
            "{title} Yahoo Finance crypto",
        ),
        preferred_domains=(
            "coingecko.com",
            "finance.yahoo.com",
            "coinmarketcap.com",
            "coindesk.com",
            "cointelegraph.com",
        ),
    ),
    "sports": RetrievalProfile(
        query_templates=(
            "{title} ESPN latest",
            "{title} official league standings news",
        ),
        preferred_domains=(
            "espn.com",
            "nba.com",
            "nfl.com",
            "mlb.com",
            "nhl.com",
            "premierleague.com",
            "uefa.com",
        ),
    ),
    "general": RetrievalProfile(
        query_templates=(),
        preferred_domains=(),
    ),
}


def get_profile(category: Category) -> RetrievalProfile:
    return _PROFILES.get(category, _PROFILES["general"])


def category_search_queries(category: Category, title: str, rules: str) -> list[str]:
    """Format category-specific search query templates."""
    profile = get_profile(category)
    if not profile.query_templates:
        return []

    title = title.strip()
    rules_hint = rules[:80].strip()
    queries: list[str] = []
    for template in profile.query_templates[:_MAX_EXTRA_QUERIES]:
        q = template.format(title=title, rules=rules_hint).strip()
        if q:
            queries.append(q)
    return queries


def preferred_domains_for(category: Category) -> tuple[str, ...]:
    return get_profile(category).preferred_domains
