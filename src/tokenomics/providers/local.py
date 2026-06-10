"""LocalAdapter: full ModelCards from config/local_models/*.yaml (§6.3).

Each YAML file is one ModelCard for a self-hosted/specialty model. Files are
re-read on every listing (they are local and small); `provider` defaults to
"local" and `last_refreshed` to read time when omitted.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

import yaml

from ..schemas.model_card import ModelCard
from .base import ModelProvider, ProviderStatus

logger = logging.getLogger(__name__)


class LocalAdapter(ModelProvider):
    name = "local"

    def __init__(self, models_dir: Path | str):
        self.models_dir = Path(models_dir)
        self._last_refreshed: datetime | None = None
        self._last_error: str | None = None

    async def list_models(self, force: bool = False) -> list[ModelCard]:
        now = datetime.now(UTC)
        cards: list[ModelCard] = []
        self._last_error = None
        if self.models_dir.is_dir():
            for path in sorted(self.models_dir.glob("*.y*ml")):
                try:
                    data = yaml.safe_load(path.read_text()) or {}
                    data.setdefault("provider", "local")
                    data.setdefault("last_refreshed", now)
                    cards.append(ModelCard.model_validate(data))
                except Exception as exc:
                    self._last_error = f"{path.name}: {exc}"
                    logger.warning("skipping local model file %s: %s", path, exc)
        self._last_refreshed = now
        return cards

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus(
            ok=self._last_error is None,
            last_refreshed=self._last_refreshed,
            detail=self._last_error,
        )
