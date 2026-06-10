"""ModelCard: the universal, provider-agnostic model shape.

Populated by: OpenRouterAdapter (base card: pricing, context, modalities,
supported parameters), LocalAdapter (full cards from config/local_models/),
then enriched by AABenchmarkProvider (quality_indices, benchmark_scores,
performance, derived capability tiers) and finally config/overrides.yaml
(curated tiers/tags — always win).
"""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .enums import Capability, Modality
from .pricing import ModelPricing


class PerformanceProfile(BaseModel):
    """Speed/latency metrics. Populated by AABenchmarkProvider (median
    measurements) or local YAML. Units: tokens/second and milliseconds."""

    output_tokens_per_second: float | None = Field(default=None, ge=0)
    time_to_first_token_ms: float | None = Field(default=None, ge=0)
    p50_latency_ms: float | None = Field(default=None, ge=0)
    p99_latency_ms: float | None = Field(default=None, ge=0)


class ModelCard(BaseModel):
    """The universal model shape served by the catalog.

    Raw fields (from the source adapter): id, canonical_slug, provider,
    display_name, description, pricing, context_window, max_output_tokens,
    modalities, supported_parameters, deprecation_date, metadata.
    Derived fields: capabilities (AA-index percentile tiers + overrides +
    the documented `tools`->AGENTIC>=2 floor), quality_indices (raw AA
    composite indices), benchmark_scores (named evals, 0..1 normalized),
    performance, tags (provider tags + overrides + capability signals).
    Units: prices per ModelPricing docstring; context/output sizes in tokens.
    """

    id: str  # e.g. "anthropic/claude-sonnet-4.6"
    canonical_slug: str | None = None  # permanent id from provider, never changes
    provider: str  # "openrouter" | "local" | future adapters
    display_name: str
    description: str | None = None
    pricing: ModelPricing
    context_window: int = Field(gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    input_modalities: list[Modality] = Field(default_factory=lambda: [Modality.TEXT])
    output_modalities: list[Modality] = Field(default_factory=lambda: [Modality.TEXT])
    supported_parameters: list[str] = Field(default_factory=list)  # raw provider list
    capabilities: dict[Capability, int] = Field(default_factory=dict)  # 1-5 tiers
    benchmark_scores: dict[str, float] | None = None  # named evals, 0..1 normalized
    quality_indices: dict[str, float] | None = None  # AA composite indices, raw
    performance: PerformanceProfile | None = None
    tags: list[str] = Field(default_factory=list)
    deprecation_date: date | None = None
    last_refreshed: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)  # provider extras, verbatim

    @field_validator("capabilities")
    @classmethod
    def _tiers_in_range(cls, v: dict[Capability, int]) -> dict[Capability, int]:
        for cap, tier in v.items():
            if not 1 <= tier <= 5:
                raise ValueError(f"capability tier for {cap} must be 1-5, got {tier}")
        return v

    def is_deprecated(self, today: date) -> bool:
        """True when the provider's deprecation/expiration date has passed."""
        return self.deprecation_date is not None and self.deprecation_date <= today
