"""ResilientSearchStage: wraps SDK SearchStage to survive per-market failures.

Gemini occasionally emits MALFORMED_FUNCTION_CALL on the search-summary tool,
causing SearchStage.execute() to return success=False and kill the entire tick.
This wrapper converts a partial failure (some summaries gathered before the
crash) into a success so the pipeline continues with whatever markets did
complete their search.
"""

from __future__ import annotations

import logging

from ai_prophet.trade.agent.stages.base import StageResult
from ai_prophet.trade.agent.stages.search import SearchStage
from ai_prophet.trade.core import TickContext

logger = logging.getLogger(__name__)


class ResilientSearchStage(SearchStage):
    """SearchStage that treats per-market failures as skips, not tick failures."""

    def execute(
        self,
        tick_ctx: TickContext,
        previous_results: dict[str, StageResult],
    ) -> StageResult:
        result = super().execute(tick_ctx, previous_results)
        if result.success:
            return result

        summaries = result.data.get("summaries", {})
        if not summaries:
            return result

        review_markets = (
            previous_results.get("review", StageResult("review", False, {}))
            .data.get("review", [])
        )
        logger.warning(
            "Search stage partial failure; continuing with %d/%d summaries. "
            "Error: %s",
            len(summaries),
            len(review_markets),
            result.error,
        )
        return StageResult(
            stage_name=self.name,
            success=True,
            data={"summaries": summaries},
        )
