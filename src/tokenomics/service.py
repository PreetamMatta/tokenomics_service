"""ModelEconomicsService: the in-process facade (§5.5).

All business logic lives here; HTTP routes in app.py are thin wrappers, and
library users can consume this class directly without running a server.
"""

import logging
from datetime import date
from typing import Any

from . import __version__
from .benchmarks.artificial_analysis import AABenchmarkProvider
from .benchmarks.local import LocalBenchmarkProvider
from .benchmarks.tiers import TierDeriver
from .cost import engine
from .errors import ModelNotFoundError, NoMatchingModelError, UnknownSchedulePresetError
from .providers.local import LocalAdapter
from .providers.openrouter import OpenRouterAdapter
from .registry import queries
from .registry.registry import ModelRegistry, load_aliases
from .schemas.breakdown import CostBreakdown, PipelineCostBreakdown
from .schemas.enums import CAPABILITY_DESCRIPTIONS, TIER_SEMANTICS, Capability, Modality
from .schemas.model_card import ModelCard
from .schemas.pipeline import Pipeline
from .schemas.pricing import ModelPricing
from .schemas.range import typical
from .schemas.schedule import PRESET_SCHEDULES, Schedule
from .schemas.workloads import WorkloadComponent
from .settings import Settings, load_settings

logger = logging.getLogger(__name__)


class ModelEconomicsService:
    """Facade over the registry, query helpers, and cost engine."""

    def __init__(self, registry: ModelRegistry, presets: dict[str, Schedule] | None = None):
        self.registry = registry
        self.presets = presets or PRESET_SCHEDULES

    # -- health & meta ------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "providers": {
                name: status.model_dump()
                for name, status in self.registry.provider_status().items()
            },
        }

    def capabilities_info(self) -> dict[str, Any]:
        return {
            "capabilities": CAPABILITY_DESCRIPTIONS,
            "tier_semantics": {str(k): v for k, v in TIER_SEMANTICS.items()},
            "derivation": {
                **self.registry.tier_deriver.describe(),
                "source": (
                    "Artificial Analysis composite indices via percentile thresholds; "
                    "overrides.yaml always wins; `tools` support floors agentic at tier 2"
                ),
            },
        }

    def schedule_presets(self) -> dict[str, Schedule]:
        return self.presets

    def resolve_schedule(self, schedule: Schedule | str) -> Schedule:
        if isinstance(schedule, Schedule):
            return schedule
        preset = self.presets.get(schedule)
        if preset is None:
            raise UnknownSchedulePresetError(schedule, sorted(self.presets))
        return preset

    # -- catalog ------------------------------------------------------------

    async def list_models(
        self,
        capability: Capability | None = None,
        min_tier: int | None = None,
        max_input_price: float | None = None,
        max_output_price: float | None = None,
        min_context_window: int | None = None,
        modalities: list[Modality] | None = None,
        tags: list[str] | None = None,
        supported_parameters: list[str] | None = None,
        include_deprecated: bool = False,
        today: date | None = None,
    ) -> list[ModelCard]:
        cards = list((await self.registry.catalog()).values())
        return queries.filter_models(
            cards,
            capability=capability,
            min_tier=min_tier,
            max_input_price=max_input_price,
            max_output_price=max_output_price,
            min_context_window=min_context_window,
            modalities=modalities,
            tags=tags,
            supported_parameters=supported_parameters,
            include_deprecated=include_deprecated,
            today=today,
        )

    async def get_model(self, model_id: str) -> ModelCard:
        card = await self.registry.get(model_id)
        if card is None:
            raise ModelNotFoundError(model_id, await self.registry.suggestions(model_id))
        return card

    async def refresh(self) -> dict[str, dict]:
        return await self.registry.refresh()

    # -- selection helpers --------------------------------------------------

    async def cheapest(
        self,
        capability: Capability,
        min_tier: int = 3,
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        cards = list((await self.registry.catalog()).values())
        result = queries.cheapest(cards, capability, min_tier=min_tier, weights=weights)
        if result is None:
            raise NoMatchingModelError(
                f"no model with {capability.value} tier >= {min_tier}",
                {"capability": capability.value, "min_tier": min_tier},
            )
        return result

    async def best(
        self,
        capability: Capability,
        min_tier: int = 1,
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        cards = list((await self.registry.catalog()).values())
        result = queries.best(cards, capability, min_tier=min_tier, weights=weights)
        if result is None:
            raise NoMatchingModelError(
                f"no model with {capability.value} tier >= {min_tier}",
                {"capability": capability.value, "min_tier": min_tier},
            )
        return result

    async def best_value(
        self,
        capability: Capability,
        min_tier: int = 1,
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        cards = list((await self.registry.catalog()).values())
        result = queries.best_value(cards, capability, min_tier=min_tier, weights=weights)
        if result is None:
            raise NoMatchingModelError(
                f"no model with {capability.value} tier >= {min_tier}",
                {"capability": capability.value, "min_tier": min_tier},
            )
        return result

    # -- cost estimation ----------------------------------------------------

    async def _pricing_map(self) -> dict[str, ModelPricing]:
        return {mid: card.pricing for mid, card in (await self.registry.catalog()).items()}

    async def estimate_cost(
        self,
        workload: WorkloadComponent,
        schedule: Schedule | str,
        model_id: str | None = None,
    ) -> CostBreakdown:
        resolved = self.resolve_schedule(schedule)
        pricing_map = await self._pricing_map()
        try:
            return engine.estimate_workload(
                workload, resolved, pricing_map, model_id_override=model_id
            )
        except ModelNotFoundError as exc:
            raise ModelNotFoundError(
                exc.model_id, await self.registry.suggestions(exc.model_id)
            ) from exc

    async def estimate_pipeline(
        self, pipeline: Pipeline, schedule: Schedule | str
    ) -> PipelineCostBreakdown:
        resolved = self.resolve_schedule(schedule)
        pricing_map = await self._pricing_map()
        try:
            return engine.estimate_pipeline(pipeline, resolved, pricing_map)
        except ModelNotFoundError as exc:
            raise ModelNotFoundError(
                exc.model_id, await self.registry.suggestions(exc.model_id)
            ) from exc

    async def compare(
        self,
        model_ids: list[str],
        workload: WorkloadComponent,
        schedule: Schedule | str,
    ) -> dict[str, Any]:
        resolved = self.resolve_schedule(schedule)
        pricing_map = await self._pricing_map()
        results: list[CostBreakdown] = []
        for model_id in model_ids:
            try:
                results.append(
                    engine.estimate_workload(
                        workload, resolved, pricing_map, model_id_override=model_id
                    )
                )
            except ModelNotFoundError as exc:
                raise ModelNotFoundError(
                    exc.model_id, await self.registry.suggestions(exc.model_id)
                ) from exc
        ranked = sorted(results, key=lambda b: typical(b.total_cost))
        return {"results": results, "ranked_by_total": [b.model_id for b in ranked]}


def build_default_service(settings: Settings | None = None) -> ModelEconomicsService:
    """Wire the default provider stack from settings (used by create_app)."""
    settings = settings or load_settings()
    providers = [
        OpenRouterAdapter(
            url=settings.openrouter_url, ttl_seconds=settings.openrouter_ttl_seconds
        ),
        LocalAdapter(settings.local_models_dir),
    ]
    quality = AABenchmarkProvider(
        api_key=settings.aa_api_key, url=settings.aa_url, ttl_seconds=settings.aa_ttl_seconds
    )
    registry = ModelRegistry(
        providers=providers,
        quality_provider=quality,
        tier_deriver=TierDeriver.from_yaml(settings.tiers_path),
        aliases=load_aliases(settings.aliases_path),
        overrides_provider=LocalBenchmarkProvider(settings.overrides_path),
    )
    return ModelEconomicsService(registry)
