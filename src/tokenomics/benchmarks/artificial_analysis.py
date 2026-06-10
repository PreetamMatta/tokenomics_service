"""AABenchmarkProvider: Artificial Analysis Data API integration (§6.2).

GET https://artificialanalysis.ai/api/v2/data/llms/models with header
`x-api-key: $AA_API_KEY`. Free tier is 1,000 req/day, so responses are cached
aggressively (24h TTL by default) and never fetched client-side.

Parsing (fixture-driven, tolerant of new keys):
- evaluation keys containing "_index" -> quality_indices (with the
  "artificial_analysis_" prefix stripped) — these are the composite indices
  (Intelligence/Coding/Agentic/Math/Multilingual) tiers are derived from;
- other numeric evaluation keys -> benchmark_scores (0..1 normalized evals:
  GPQA, HLE, MMLU-Pro, AIME, LiveCodeBench, SciCode, IFBench, ...);
- median_output_tokens_per_second / median_time_to_first_token_seconds ->
  PerformanceProfile (seconds converted to ms).

AA pricing is intentionally ignored: OpenRouter wins for billing since we buy
through it — AA is for quality/speed only.

If AA_API_KEY is absent the provider degrades gracefully: a warning is logged
and records() returns [] (capabilities then come only from overrides.yaml).
"""

import logging
from datetime import datetime
from typing import Any

import httpx

from ..providers.base import ProviderStatus
from ..providers.cache import TTLCache
from ..schemas.model_card import PerformanceProfile
from .base import BenchmarkProvider, ModelQualityRecord

logger = logging.getLogger(__name__)

DEFAULT_AA_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
_INDEX_PREFIX = "artificial_analysis_"


def parse_aa_model(entry: dict[str, Any]) -> ModelQualityRecord:
    """Parse one AA model entry into a quality record."""
    indices: dict[str, float] = {}
    scores: dict[str, float] = {}
    for key, value in (entry.get("evaluations") or {}).items():
        if not isinstance(value, (int, float)):
            continue
        if "_index" in key:
            indices[key.removeprefix(_INDEX_PREFIX)] = float(value)
        else:
            scores[key] = float(value)

    performance = None
    tokens_per_second = entry.get("median_output_tokens_per_second")
    ttft_seconds = entry.get("median_time_to_first_token_seconds")
    if tokens_per_second is not None or ttft_seconds is not None:
        performance = PerformanceProfile(
            output_tokens_per_second=tokens_per_second,
            time_to_first_token_ms=ttft_seconds * 1000.0 if ttft_seconds is not None else None,
        )

    return ModelQualityRecord(
        slug=entry.get("slug") or entry["id"],
        name=entry.get("name"),
        quality_indices=indices,
        benchmark_scores=scores,
        performance=performance,
    )


class AABenchmarkProvider(BenchmarkProvider):
    name = "artificial_analysis"

    def __init__(
        self,
        api_key: str | None,
        url: str = DEFAULT_AA_URL,
        ttl_seconds: float = 86_400.0,
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.url = url
        self.timeout = timeout
        self._cache: TTLCache[list[ModelQualityRecord]] = TTLCache(ttl_seconds)
        if not api_key:
            logger.warning(
                "AA_API_KEY not set — skipping Artificial Analysis enrichment; "
                "capability tiers will come only from overrides.yaml"
            )

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def records(self, force: bool = False) -> list[ModelQualityRecord]:
        if not self.enabled:
            return []
        return await self._cache.get(self._fetch, force=force)

    @property
    def status(self) -> ProviderStatus:
        if not self.enabled:
            return ProviderStatus(ok=True, detail="disabled: AA_API_KEY not set")
        return ProviderStatus(
            ok=self._cache.ok,
            last_refreshed=self._cache.last_refreshed,
            detail=self._cache.last_error,
        )

    @property
    def last_refreshed(self) -> datetime | None:
        return self._cache.last_refreshed

    async def _fetch(self) -> list[ModelQualityRecord]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(self.url, headers={"x-api-key": self.api_key})
            response.raise_for_status()
            payload = response.json()
        records: list[ModelQualityRecord] = []
        for entry in payload.get("data", []):
            try:
                records.append(parse_aa_model(entry))
            except (KeyError, TypeError, ValueError) as exc:
                logger.info("skipping AA entry %s: %s", entry.get("slug") or entry.get("id"), exc)
        logger.info("artificial_analysis: parsed %d records", len(records))
        return records
