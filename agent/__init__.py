"""Nailong Trading-Track agent submission for Prophet Hacks 2026.

Wraps the vendored `ai-prophet` SDK ExperimentRunner with a custom
build_pipeline that injects our retrieval module, market-anchored
calibration, and risk-aware sizing.

Entry point: `python -m agent.run` (or `bash scripts/run.sh`).
"""

__version__ = "1.0.0"
