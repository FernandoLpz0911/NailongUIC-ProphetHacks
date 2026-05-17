"""Entrypoint: `python -m agent.run` or `bash scripts/run.sh`.

Mirrors the SDK CLI's `_run_impl` flag set so anyone who has used
`prophet trade eval run` already knows how to drive this. The only
substantive difference is that we hand ExperimentRunner our custom
`build_pipeline` callback from `agent.pipeline`.

Defaults are tuned for the 14-day eval window:
  --slug      nailong_v01
  --max-ticks 1344   (14 days * 24h * 4 ticks/h)
  --starting-cash 10000 (matches INITIAL_CASH from constants.csv)
<<<<<<< HEAD
  -m          gemini:gemini-2.5-pro
=======
  -m          gemini-3.1-pro-preview
>>>>>>> claude/elated-hypatia-03a7b3

Use --dry to wire everything up but skip `runner.run()` so you can verify
the build_pipeline closure resolves credentials without burning a tick lease.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path

import click

from ai_prophet.trade.core.config import ClientConfig
from ai_prophet.trade.core.credentials import Credentials, load_dotenv_file
from ai_prophet.trade.runner import ExperimentRunner

from agent.pipeline import make_pipeline_builder
from agent.settings import load as load_runtime


def _setup_logging(verbose: bool, log_level: str) -> None:
    level = logging.DEBUG if verbose else getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Quiet down chatty deps.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("trafilatura.main_extractor").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


@click.command()
@click.option(
    "-m", "--models", multiple=True,
    default=("gemini:gemini-2.5-flash",),
    show_default=True,
    help="Model specs (provider:model). Repeatable.",
)
@click.option(
    "-s", "--slug",
    default="eval_nailonguic", show_default=True,
    help="Experiment slug (stable across restarts).",
)
@click.option(
    "-r", "--replicates", type=int, default=1, show_default=True,
    help="Replicates per model.",
)
@click.option(
    "-t", "--max-ticks", type=int, default=1500, show_default=True,
    help="Target completed ticks. 1500 covers the 14-day eval window (1344 max) with buffer.",
)
@click.option(
    "--starting-cash", type=float, default=None,
    help="Per-participant starting cash. Default = INITIAL_CASH from constants.csv.",
)
@click.option(
    "--trace-dir", type=click.Path(), default="./data/traces",
    show_default=True, help="Local trace directory.",
)
@click.option(
    "--publish-reasoning/--no-publish-reasoning",
    default=True, show_default=True,
    help="Persist per-stage reasoning in plan_json for the post-event writeup.",
)
@click.option(
    "--api-url", default=None,
    help="Core API URL. Default reads PA_SERVER_URL from .env.",
)
@click.option(
    "--dry", is_flag=True,
    help="Build pipeline and verify creds without running the tick loop.",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging.")
def main(
    models: tuple[str, ...],
    slug: str,
    replicates: int,
    max_ticks: int,
    starting_cash: float | None,
    trace_dir: str,
    publish_reasoning: bool,
    api_url: str | None,
    dry: bool,
    verbose: bool,
) -> None:
    """Run the Nailong trading agent against the Prophet Arena Core API."""
    # 1. Load .env BEFORE we read any config.
    load_dotenv_file()
    runtime = load_runtime()
    _setup_logging(verbose, runtime.log_level)
    logger = logging.getLogger(__name__)

    # 2. Validate credentials early so we fail fast.
    if not runtime.pa_server_api_key:
        raise click.ClickException("PA_SERVER_API_KEY is not set. See .env.template.")

    creds = Credentials.from_env()
    client_config = ClientConfig.load_runtime()
    api_url = api_url or runtime.pa_server_url
    if starting_cash is None:
        starting_cash = runtime.constraints.initial_cash

    # 3. Echo plan to the operator.
    click.echo("=" * 64)
    click.echo(f"  Nailong Trading Agent  (slug={slug})")
    click.echo("=" * 64)
    click.echo(f"  API URL          : {api_url}")
    click.echo(f"  Models           : {', '.join(models)} x {replicates} rep(s)")
    click.echo(f"  Target ticks     : {max_ticks}")
    click.echo(f"  Starting cash    : ${starting_cash:,.2f}")
    click.echo(f"  Trace dir        : {trace_dir}")
    click.echo(f"  Kill-switch      : ${runtime.kill_switch_usd:.2f}")
    click.echo(f"  Cost ledger      : {runtime.cost_db_path}")
    click.echo("=" * 64)

    # 4. Build the model_configs list ExperimentRunner expects.
    model_configs: list[dict] = []
    for spec in models:
        for rep in range(replicates):
            model_configs.append({"model": spec, "rep": rep})

    # 5. Construct the closure ExperimentRunner will call per tick.
    build_pipeline = make_pipeline_builder(runtime=runtime, verbose=verbose)

    runner = ExperimentRunner(
        api_url=api_url,
        api_key=runtime.pa_server_api_key,
        experiment_slug=slug,
        models=model_configs,
        config={
            "models":         list(models),
            "replicates":     replicates,
            "starting_cash":  starting_cash,
            "strategy":       "nailong-calibrated-kelly-v1",
        },
        n_ticks=max_ticks,
        starting_cash=starting_cash,
        trace_dir=Path(trace_dir) if trace_dir else None,
        build_pipeline=build_pipeline,
        publish_reasoning=publish_reasoning,
        client_config=client_config,
        memory_dir=Path(os.environ.get("PA_MEMORY_DIR", ".pa_memory")).expanduser(),
        memory_max_rows=int(os.environ.get("PA_MEMORY_MAX_ROWS", "1000")),
    )

    if dry:
        click.echo("\n[--dry] Skipping runner.run(); pipeline construction verified.")
        try:
            build_pipeline({"model": model_configs[0]["model"], "rep": 0})
            click.echo("[--dry] Test pipeline build OK.")
        except Exception as e:
            click.echo(f"[--dry] Pipeline build FAILED: {e}", err=True)
            traceback.print_exc()
            sys.exit(1)
        return

    # 6. Run the tick loop. ExperimentRunner handles lease bumping, transient
    #    error retries, and finalize-on-failure internally.
    try:
        runner.run()
    except Exception as e:
        logger.error("FATAL: %s: %s", type(e).__name__, e)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
