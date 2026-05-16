from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from agent.config import CACHE_DIR, CACHE_TTL_SECONDS


class DiskCache:
    """JSON file cache with TTL, keyed by namespace + key."""

    def __init__(
        self,
        namespace: str,
        root: Path = CACHE_DIR,
        ttl_seconds: int = CACHE_TTL_SECONDS,
    ) -> None:
        self.namespace = namespace
        self.root = root / namespace
        self.ttl_seconds = ttl_seconds
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.root / f"{digest}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - payload.get("_ts", 0) > self.ttl_seconds:
            path.unlink(missing_ok=True)
            return None
        return payload.get("value")

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        path.write_text(
            json.dumps({"_ts": time.time(), "value": value}),
            encoding="utf-8",
        )
