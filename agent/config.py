import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROMPTS_DIR = ROOT_DIR / "prompts"
CACHE_DIR = ROOT_DIR / ".cache"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "anthropic/claude-sonnet-4")
CHEAP_MODEL = os.getenv("CHEAP_MODEL", "google/gemini-2.5-flash")
EXPENSIVE_MODEL = os.getenv("EXPENSIVE_MODEL", "anthropic/claude-opus-4")
FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "FALLBACK_MODELS",
        "google/gemini-2.5-flash,deepseek/deepseek-v3.2",
    ).split(",")
    if m.strip()
]

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")

PREDICT_TIMEOUT_SECONDS = int(os.getenv("PREDICT_TIMEOUT_SECONDS", "180"))
CALIBRATION_ALPHA = float(os.getenv("CALIBRATION_ALPHA", "0.55"))
MAX_EDGE_DEVIATION = float(os.getenv("MAX_EDGE_DEVIATION", "0.30"))

USE_ENSEMBLE = os.getenv("USE_ENSEMBLE", "true").lower() in ("1", "true", "yes")
USE_PROMPT_CACHE = os.getenv("USE_PROMPT_CACHE", "false").lower() in ("1", "true", "yes")
MODEL_CALL_TIMEOUT_SECONDS = int(os.getenv("MODEL_CALL_TIMEOUT_SECONDS", "180"))
ENSEMBLE_MODELS = [
    m.strip()
    for m in os.getenv(
        "ENSEMBLE_MODELS",
        "anthropic/claude-opus-4,openai/gpt-4o,deepseek/deepseek-r1",
    ).split(",")
    if m.strip()
]

COST_DB_PATH = Path(os.getenv("COST_DB_PATH", str(ROOT_DIR / "costs.sqlite")))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", str(24 * 3600)))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
