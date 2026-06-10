"""ModelRegistry: the deterministic merge of all data sources (§6.4).

Merge order:
1. OpenRouterAdapter -> base ModelCards (pricing, context, modalities, params)
2. LocalAdapter -> additional ModelCards
3. AABenchmarkProvider -> enrich with quality_indices, benchmark_scores,
   performance, and derived capability tiers (AA slug -> OpenRouter id via
   config/model_aliases.yaml; unmatched AA models are logged and skipped;
   unmatched catalog models simply have no derived tiers)
4. overrides.yaml -> final say on tiers/tags (always wins)

After the merge, two documented capability signals from supported_parameters
are applied (§6.1/§1.3):
- `tools` present and no curated/derived AGENTIC tier -> floor AGENTIC at 2
  (weak evidence that the model can drive tools);
- `structured_outputs` present -> add the "structured_outputs" tag.
"""

import difflib
import logging
from collections.abc import Sequence
from pathlib import Path

import yaml

from ..benchmarks.base import BenchmarkProvider
from ..benchmarks.local import LocalBenchmarkProvider
from ..benchmarks.tiers import TierDeriver
from ..providers.base import ModelProvider, ProviderStatus
from ..schemas.enums import Capability
from ..schemas.model_card import ModelCard

logger = logging.getLogger(__name__)


def load_aliases(path: Path | str) -> dict[str, str]:
    """Load config/model_aliases.yaml (AA slug -> OpenRouter id)."""
    path = Path(path)
    if not path.is_file():
        logger.info("alias file %s not found, no AA identity mapping", path)
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    return {str(k): str(v) for k, v in raw.items()}


class ModelRegistry:
    """Merges providers + quality sources into the served catalog."""

    def __init__(
        self,
        providers: Sequence[ModelProvider],
        quality_provider: BenchmarkProvider | None,
        tier_deriver: TierDeriver,
        aliases: dict[str, str],
        overrides_provider: LocalBenchmarkProvider,
    ):
        self.providers = list(providers)
        self.quality_provider = quality_provider
        self.tier_deriver = tier_deriver
        self.aliases = aliases
        self.overrides_provider = overrides_provider

    async def catalog(self, force: bool = False) -> dict[str, ModelCard]:
        """Build the merged catalog. Provider data is TTL-cached upstream, so
        this is cheap to call per request; `force` busts those caches."""
        cards: dict[str, ModelCard] = {}

        # 1+2: base cards, in provider order (later providers win on id clash).
        for provider in self.providers:
            try:
                for card in await provider.list_models(force=force):
                    cards[card.id] = card.model_copy(deep=True)
            except Exception as exc:
                logger.error("provider %s failed, continuing without it: %s", provider.name, exc)

        # 3: AA enrichment + derived tiers.
        if self.quality_provider is not None:
            try:
                records = await self.quality_provider.records(force=force)
            except Exception as exc:
                logger.error("quality provider failed, continuing unenriched: %s", exc)
                records = []
            derived = self.tier_deriver.derive_population(records)
            for record in records:
                target_id = self.aliases.get(record.slug)
                if target_id is None:
                    logger.debug("AA slug %s has no alias mapping, skipped", record.slug)
                    continue
                card = cards.get(target_id)
                if card is None:
                    logger.debug("AA alias %s -> %s not in catalog, skipped", record.slug, target_id)
                    continue
                card.quality_indices = dict(record.quality_indices) or None
                card.benchmark_scores = {
                    **(card.benchmark_scores or {}),
                    **record.benchmark_scores,
                } or None
                if record.performance is not None:
                    card.performance = record.performance
                card.capabilities.update(derived.get(record.slug, {}))

        # 4: curated overrides — final say on tiers/tags.
        for model_id, override in self.overrides_provider.load().items():
            card = cards.get(model_id)
            if card is None:
                logger.debug("override for unknown model %s ignored", model_id)
                continue
            card.capabilities.update(override.capabilities)
            for tag in override.tags:
                if tag not in card.tags:
                    card.tags.append(tag)
            if override.benchmark_scores:
                card.benchmark_scores = {
                    **(card.benchmark_scores or {}),
                    **override.benchmark_scores,
                }

        # Capability signals from supported_parameters (§6.1, §1.3).
        for card in cards.values():
            if "tools" in card.supported_parameters and Capability.AGENTIC not in card.capabilities:
                card.capabilities[Capability.AGENTIC] = 2
            if (
                "structured_outputs" in card.supported_parameters
                and "structured_outputs" not in card.tags
            ):
                card.tags.append("structured_outputs")

        return cards

    async def get(self, model_id: str) -> ModelCard | None:
        return (await self.catalog()).get(model_id)

    async def suggestions(self, model_id: str) -> list[str]:
        """Closest catalog ids for a miss, for the structured 404 body."""
        try:
            ids = list(await self.catalog())
        except Exception:
            return []
        return difflib.get_close_matches(model_id, ids, n=5, cutoff=0.4)

    async def refresh(self) -> dict[str, dict]:
        """Force re-fetch from every source; per-provider status report."""
        report: dict[str, dict] = {}
        for provider in self.providers:
            try:
                models = await provider.list_models(force=True)
                report[provider.name] = {
                    "ok": True,
                    "models": len(models),
                    "last_refreshed": provider.status.last_refreshed,
                }
            except Exception as exc:
                report[provider.name] = {"ok": False, "error": str(exc)}
        if self.quality_provider is not None:
            try:
                records = await self.quality_provider.records(force=True)
                report[self.quality_provider.name] = {
                    "ok": True,
                    "records": len(records),
                    "last_refreshed": self.quality_provider.status.last_refreshed,
                }
            except Exception as exc:
                report[self.quality_provider.name] = {"ok": False, "error": str(exc)}
        return report

    def provider_status(self) -> dict[str, ProviderStatus]:
        statuses = {provider.name: provider.status for provider in self.providers}
        if self.quality_provider is not None:
            statuses[self.quality_provider.name] = self.quality_provider.status
        return statuses
