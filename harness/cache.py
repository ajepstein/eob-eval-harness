"""A JSON-file-backed cache of model responses, keyed by request fingerprint."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from harness.types import ModelResponse

# stop_reason / finish_reason values that indicate the response was cut off
# mid-generation and should never be cached.
_TRUNCATED_FINISH_REASONS = {"max_tokens"}


def make_cache_key(
    prompt: str,
    model_id: str,
    params: dict[str, Any],
    prompt_template_hash: str,
) -> str:
    """Fingerprint a request: sha256(prompt + model_id + sorted params + template hash), truncated to 16 hex chars."""
    payload = prompt + model_id + json.dumps(params, sort_keys=True) + prompt_template_hash
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class ResponseCache:
    def __init__(self, cache_dir: str | Path = ".cache", enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> ModelResponse | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        return ModelResponse(**json.loads(path.read_text()))

    def set(self, key: str, response: ModelResponse) -> None:
        if not self.enabled:
            return
        if response.finish_reason in _TRUNCATED_FINISH_REASONS:
            return
        self._path(key).write_text(json.dumps(asdict(response)))
