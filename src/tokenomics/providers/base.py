"""Provider interface: anything that yields ModelCards."""

import abc
from datetime import datetime

from pydantic import BaseModel

from ..schemas.model_card import ModelCard


class ProviderStatus(BaseModel):
    """Health snapshot reported under /health."""

    ok: bool
    last_refreshed: datetime | None = None
    detail: str | None = None


class ModelProvider(abc.ABC):
    """A source of ModelCards (OpenRouter, local YAML, future adapters)."""

    name: str

    @abc.abstractmethod
    async def list_models(self, force: bool = False) -> list[ModelCard]:
        """Return the provider's models, refreshing caches when forced."""

    @property
    @abc.abstractmethod
    def status(self) -> ProviderStatus:
        """Current health/freshness of this provider."""
