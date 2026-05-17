"""build_custom_pipeline: the callback we hand to ExperimentRunner.

ExperimentRunner calls this once per participant per tick to get an
AgentPipeline instance. We construct the SDK's pipeline normally to inherit
all the orchestration/logging plumbing, then swap in:

  * RetrievalSearchClient  (Phase 1) - replaces Brave Search
  * CalibratedForecastStage (Phase 2) - market-anchored probability blend
  * RiskAwareActionStage   (Phase 3) - deterministic Kelly + ruleset caps

If the credentials for a particular provider are missing, we raise a clear
ClickException-equivalent so ExperimentRunner finalizes the participant as
FAILED rather than crashing the loop.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ai_prophet.trade.agent.pipeline import AgentPipeline
from ai_prophet.trade.core.config import ClientConfig
from ai_prophet.trade.core.credentials import Credentials, normalize_provider_name
from ai_prophet.trade.llm import create_llm_client
from ai_prophet_core.client import ServerAPIClient

from agent.settings import RuntimeConfig, load as load_runtime
from agent.stages.calibrated_forecast import CalibratedForecastStage
from agent.stages.retrieval_search import RetrievalSearchClient
from agent.stages.risk_action import RiskAwareActionStage
from agent.stages.text_review import TextReviewStage

logger = logging.getLogger(__name__)


def _split_model_spec(model_spec: str) -> tuple[str, str]:
    """Match the SDK CLI's `provider:model` parser exactly."""
    if ":" in model_spec:
        provider, model_name = model_spec.split(":", 1)
        return provider, model_name
    return "openai", model_spec


def build_custom_pipeline(
    participant_cfg: dict[str, Any],
    *,
    runtime: RuntimeConfig | None = None,
    client_config: ClientConfig | None = None,
    creds: Credentials | None = None,
    verbose: bool = False,
) -> AgentPipeline:
    """Construct an AgentPipeline for one participant with our custom stages.

    Args:
        participant_cfg: Dict from ExperimentRunner, must contain "model".
        runtime: Optional pre-loaded RuntimeConfig (tests use this).
        client_config: SDK ClientConfig; defaults to ClientConfig.get().
        creds: SDK Credentials; defaults to Credentials.from_env().
        verbose: Bubble through to llm_client for debug logging.
    """
    runtime = runtime or load_runtime()
    client_config = client_config or ClientConfig.get()
    creds = creds or Credentials.from_env()

    model_spec = participant_cfg["model"]
    provider, model_name = _split_model_spec(model_spec)
    llm_provider = normalize_provider_name(provider)

    api_key = creds.get_api_key(llm_provider)
    if not api_key:
        raise RuntimeError(
            f"No API key found for provider '{llm_provider}'. "
            f"Set {llm_provider.upper()}_API_KEY in .env."
        )

    llm_client = create_llm_client(
        provider=llm_provider,
        model=model_name,
        api_key=api_key,
        verbose=verbose,
        config=client_config.llm,
    )

    api_client = ServerAPIClient(
        base_url=runtime.pa_server_url,
        api_key=runtime.pa_server_api_key,
    )

    search_client = RetrievalSearchClient(
        max_results=client_config.search.max_results_per_query,
    )

    pipeline_config: dict[str, Any] = {
        "search_client": search_client,
        "max_queries_per_market": client_config.search.max_queries_per_market,
        "max_results_per_query": client_config.search.max_results_per_query,
        "max_markets": client_config.pipeline.max_markets,
        "min_size_usd": client_config.pipeline.min_size_usd,
    }

    pipeline = AgentPipeline(
        llm_client=llm_client,
        event_store=None,
        api_client=api_client,
        config=pipeline_config,
        client_config=client_config,
    )

    # ------------------------------------------------------------------
    # Swap in our custom stages.
    # AgentPipeline.stages is a list of 4: [Review, Search, Forecast, Action].
    # ------------------------------------------------------------------
    pipeline.stages[0] = TextReviewStage(
        llm_client=llm_client,
        max_markets=pipeline_config["max_markets"],
    )
    pipeline.stages[2] = CalibratedForecastStage(
        llm_client=llm_client,
        calibration=runtime.calibration,
    )
    pipeline.stages[3] = RiskAwareActionStage(
        llm_client=None,  # deterministic; no second LLM call
        constraints=runtime.constraints,
        risk=runtime.risk,
        min_size_usd=runtime.risk.min_intent_size_usd,
    )

    logger.info(
        "Built custom pipeline for %s (provider=%s, model=%s)",
        model_spec, llm_provider, model_name,
    )
    return pipeline


def make_pipeline_builder(
    *,
    runtime: RuntimeConfig | None = None,
    verbose: bool = False,
):
    """Closure factory for ExperimentRunner(build_pipeline=...).

    Captures runtime/verbose so the runner can call us with just
    participant_cfg per its expected signature.
    """
    runtime = runtime or load_runtime()
    client_config = ClientConfig.get()

    # Best-effort .env load so direct invocations work without bash export.
    from ai_prophet.trade.core.credentials import load_dotenv_file
    load_dotenv_file()
    creds = Credentials.from_env()

    def _builder(participant_cfg: dict[str, Any]) -> AgentPipeline:
        return build_custom_pipeline(
            participant_cfg,
            runtime=runtime,
            client_config=client_config,
            creds=creds,
            verbose=verbose,
        )

    # Annotate so logs are useful.
    _builder.__name__ = "build_custom_pipeline_closure"
    return _builder


# `os` import is intentionally retained (used by future logging hooks).
_ = os
