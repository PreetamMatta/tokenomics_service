"""LocalBenchmarkProvider: the human-curated override layer (§6.3).

Reads config/overrides.yaml: per model_id, optional `capabilities` (tier
overrides — ALWAYS win over derived tiers), `tags` (merged), and
`benchmark_scores` (merged, e.g. internal eval results).
"""

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ..schemas.enums import Capability

logger = logging.getLogger(__name__)


class ModelOverride(BaseModel):
    """Curated per-model overrides from overrides.yaml."""

    capabilities: dict[Capability, int] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    benchmark_scores: dict[str, float] = Field(default_factory=dict)


class LocalBenchmarkProvider:
    """Loads overrides.yaml; missing file means no overrides."""

    name = "local_overrides"

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def load(self) -> dict[str, ModelOverride]:
        if not self.path.is_file():
            logger.info("overrides file %s not found, no overrides applied", self.path)
            return {}
        raw = yaml.safe_load(self.path.read_text()) or {}
        overrides: dict[str, ModelOverride] = {}
        for model_id, data in raw.items():
            try:
                overrides[model_id] = ModelOverride.model_validate(data or {})
            except Exception as exc:
                logger.warning("ignoring invalid override for %s: %s", model_id, exc)
        return overrides
