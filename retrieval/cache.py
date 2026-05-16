from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agent.config import CACHE_DIR


class DiskCache:
    """Simple JSON file cache keyed by (namespace, key)."""

    def __init__(self, namespace: str, root: Path = CACHE_DIR) -> None:
        self.namespace = namespace
        self.root = root / namespace
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.root / f"{digest}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        path.write_text(json.dumps(value), encoding="utf-8")
