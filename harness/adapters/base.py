"""Provider-agnostic adapter interface and error taxonomy.

Nothing outside `adapters/` should import a provider SDK or reference a
provider-specific exception type — each adapter catches its provider's
native exceptions and re-raises as one of the three below.
"""

from __future__ import annotations

from typing import Protocol

from harness.types import ModelResponse


class AdapterError(Exception):
    """Base class for all adapter errors."""


class RateLimited(AdapterError):
    """The provider rate-limited this request. Retry after a delay."""


class TransientError(AdapterError):
    """A network blip or 5xx from the provider. Safe to retry."""


class FatalError(AdapterError):
    """Auth failure, malformed request, context overflow. Never retry."""


class Adapter(Protocol):
    name: str
    model_id: str

    async def complete(
        self, prompt: str, *, max_tokens: int = 2000, temperature: float = 0.0
    ) -> ModelResponse: ...
