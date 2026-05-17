"""Prewarm the retrieval cache for the current candidate universe.

Run this once before the eval window opens so the first real tick's
forecast stage gets an instant cache hit instead of fanning out to Tavily
+ Polymarket for every market.

Usage:
    python scripts/prewarm.py                 # use Core API current snapshot
    python scripts/prewarm.py --limit 50      # cap how many to warm
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

# Allow running as a script (no installed agent package needed).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_prophet.trade.core.credentials import load_dotenv_file  # noqa: E402
from ai_prophet_core.client import ServerAPIClient  # noqa: E402

from agent.settings import load as load_runtime  # noqa: E402
from retrieval.retrieval import prewarm_cache  # noqa: E402


@click.command()
@click.option("--api-url", default=None, help="Core API URL (default PA_SERVER_URL).")
@click.option("--limit", type=int, default=100, show_default=True, help="Max markets to warm.")
def main(api_url: str | None, limit: int) -> None:
    load_dotenv_file()
    runtime = load_runtime()
    api_url = api_url or runtime.pa_server_url

    if not runtime.pa_server_api_key:
        raise click.ClickException("PA_SERVER_API_KEY missing; cannot fetch snapshot.")

    click.echo(f"Fetching live market snapshot from {api_url} ...")
    api = ServerAPIClient(base_url=api_url, api_key=runtime.pa_server_api_key)
    try:
        snapshot = api.get_market_snapshot()
    finally:
        try:
            api.close()
        except Exception:
            pass

    markets = snapshot.markets[:limit]
    click.echo(f"Snapshot returned {len(markets)} markets (capped at {limit}).")

    events: list[dict] = []
    for m in markets:
        events.append({
            "market_id": m.market_id,
            "title":     m.question,
            "rules":     "",
            # The SDK doesn't expose a resolution_date today; pass None and
            # let retrieval skip the recency boost.
            "resolution_date": None,
        })

    prewarm_cache(events)


if __name__ == "__main__":
    main()
