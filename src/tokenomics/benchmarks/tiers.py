"""Tier derivation: AA composite indices -> capability tiers (§6.2).

Per capability, the relevant AA index is converted into a 1-5 tier via
configurable percentile thresholds over the tracked-model population:
>= P90 -> 5, >= P70 -> 4, >= P40 -> 3, >= P15 -> 2, else 1 (defaults;
config/tiers.yaml overrides). A model's percentile is its rank within the
population of models that report that index, computed as
100 * (strictly_below + 0.5 * ties) / n.
"""

import logging
from pathlib import Path

import yaml

from ..schemas.enums import Capability
from .base import ModelQualityRecord

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS: dict[int, float] = {5: 90.0, 4: 70.0, 3: 40.0, 2: 15.0}
DEFAULT_INDEX_CAPABILITY_MAP: dict[str, Capability] = {
    "intelligence_index": Capability.REASONING,
    "coding_index": Capability.CODING,
    "agentic_index": Capability.AGENTIC,
    "math_index": Capability.MATH,
}


def percentile_of(values: list[float], value: float) -> float:
    """Percentile rank of `value` within `values` (0..100, ties count half)."""
    if not values:
        return 0.0
    below = sum(1 for v in values if v < value)
    ties = sum(1 for v in values if v == value)
    return 100.0 * (below + 0.5 * ties) / len(values)


class TierDeriver:
    """Converts quality-index populations into per-model capability tiers."""

    def __init__(
        self,
        thresholds: dict[int, float] | None = None,
        index_capability_map: dict[str, Capability] | None = None,
    ):
        self.thresholds = dict(thresholds or DEFAULT_THRESHOLDS)
        self.index_capability_map = dict(index_capability_map or DEFAULT_INDEX_CAPABILITY_MAP)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "TierDeriver":
        """Load thresholds/mapping from config/tiers.yaml; defaults on missing file."""
        path = Path(path)
        if not path.is_file():
            logger.info("tiers config %s not found, using built-in defaults", path)
            return cls()
        raw = yaml.safe_load(path.read_text()) or {}
        thresholds = {
            int(tier): float(cutoff)
            for tier, cutoff in (raw.get("percentile_thresholds") or {}).items()
        }
        mapping = {
            str(index): Capability(capability)
            for index, capability in (raw.get("index_capability_map") or {}).items()
        }
        return cls(thresholds or None, mapping or None)

    def tier_for_percentile(self, percentile: float) -> int:
        for tier in sorted(self.thresholds, reverse=True):
            if percentile >= self.thresholds[tier]:
                return tier
        return 1

    def derive_population(
        self, records: list[ModelQualityRecord]
    ) -> dict[str, dict[Capability, int]]:
        """Derive tiers for every record, per mapped index, against the
        population of models that report that index."""
        derived: dict[str, dict[Capability, int]] = {}
        for index_name, capability in self.index_capability_map.items():
            population = [
                r.quality_indices[index_name] for r in records if index_name in r.quality_indices
            ]
            if not population:
                continue
            for record in records:
                value = record.quality_indices.get(index_name)
                if value is None:
                    continue
                tier = self.tier_for_percentile(percentile_of(population, value))
                derived.setdefault(record.slug, {})[capability] = tier
        return derived

    def describe(self) -> dict:
        """Threshold/mapping summary for /v1/capabilities."""
        return {
            "percentile_thresholds": {str(k): v for k, v in sorted(self.thresholds.items())},
            "index_capability_map": {
                k: v.value for k, v in self.index_capability_map.items()
            },
        }
