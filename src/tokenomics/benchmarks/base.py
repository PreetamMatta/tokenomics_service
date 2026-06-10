"""Benchmark/quality provider interface."""

import abc

from pydantic import BaseModel, Field

from ..providers.base import ProviderStatus
from ..schemas.model_card import PerformanceProfile


class ModelQualityRecord(BaseModel):
    """Quality data for one model, keyed by the provider's native slug.

    Populated by: AABenchmarkProvider (or future quality sources).
    quality_indices are raw composite indices (e.g. coding_index); units are
    the provider's own scale. benchmark_scores are named evals normalized
    0..1. performance carries median speed metrics.
    """

    slug: str
    name: str | None = None
    quality_indices: dict[str, float] = Field(default_factory=dict)
    benchmark_scores: dict[str, float] = Field(default_factory=dict)
    performance: PerformanceProfile | None = None


class BenchmarkProvider(abc.ABC):
    """A source of ModelQualityRecords."""

    name: str

    @abc.abstractmethod
    async def records(self, force: bool = False) -> list[ModelQualityRecord]:
        """Return quality records, refreshing caches when forced."""

    @property
    @abc.abstractmethod
    def status(self) -> ProviderStatus:
        """Current health/freshness of this provider."""
