import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.main import app

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_EVENT = ROOT / "data" / "sample_event.json"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_event() -> dict:
    return json.loads(SAMPLE_EVENT.read_text(encoding="utf-8"))
