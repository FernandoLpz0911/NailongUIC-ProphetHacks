import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROMPTS_DIR = ROOT_DIR / "prompts"
CACHE_DIR = ROOT_DIR / ".cache"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "anthropic/claude-sonnet-4")
FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "FALLBACK_MODELS",
        "google/gemini-2.5-flash,deepseek/deepseek-v3.2",
    ).split(",")
    if m.strip()
]

PREDICT_TIMEOUT_SECONDS = int(os.getenv("PREDICT_TIMEOUT_SECONDS", "180"))
COST_DB_PATH = Path(os.getenv("COST_DB_PATH", str(ROOT_DIR / "costs.sqlite")))
