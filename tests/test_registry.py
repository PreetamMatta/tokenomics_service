"""Registry merge order, override precedence, alias mapping (§6.4)."""

import pytest

from tests.conftest import make_card
from tokenomics.benchmarks.base import BenchmarkProvider, ModelQualityRecord
from tokenomics.benchmarks.local import LocalBenchmarkProvider
from tokenomics.benchmarks.tiers import TierDeriver
from tokenomics.providers.base import ModelProvider, ProviderStatus
from tokenomics.registry.registry import ModelRegistry, load_aliases
from tokenomics.schemas.enums import Capability
from tokenomics.schemas.model_card import ModelCard

pytestmark = pytest.mark.asyncio


class StaticProvider(ModelProvider):
    def __init__(self, name: str, cards: list[ModelCard]):
        self.name = name
        self.cards = cards

    async def list_models(self, force: bool = False) -> list[ModelCard]:
        return self.cards

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus(ok=True)


class StaticQuality(BenchmarkProvider):
    name = "static_quality"

    def __init__(self, records: list[ModelQualityRecord]):
        self._records = records

    async def records(self, force: bool = False) -> list[ModelQualityRecord]:
        return self._records

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus(ok=True)


def _quality_records() -> list[ModelQualityRecord]:
    # Five-model population, evenly spaced -> model-a tops out at tier 5.
    records = [
        ModelQualityRecord(
            slug=f"slug-{i}", quality_indices={"coding_index": float(i * 10)}
        )
        for i in range(1, 5)
    ]
    records.append(
        ModelQualityRecord(
            slug="model-a",
            quality_indices={"coding_index": 50.0},
            benchmark_scores={"swe_bench": 0.6},
        )
    )
    return records


def _registry(tmp_path, overrides_yaml: str | None = None) -> ModelRegistry:
    overrides_path = tmp_path / "overrides.yaml"
    if overrides_yaml is not None:
        overrides_path.write_text(overrides_yaml)
    cards = [
        make_card("prov/model-a", supported_parameters=["tools"]),
        make_card("prov/model-b"),
        make_card("prov/model-c", supported_parameters=["tools", "structured_outputs"]),
    ]
    return ModelRegistry(
        providers=[StaticProvider("static", cards)],
        quality_provider=StaticQuality(_quality_records()),
        tier_deriver=TierDeriver(),
        aliases={"model-a": "prov/model-a", "ghost-slug": "prov/ghost"},
        overrides_provider=LocalBenchmarkProvider(overrides_path),
    )


async def test_aa_enrichment_via_alias(tmp_path):
    catalog = await _registry(tmp_path).catalog()
    card = catalog["prov/model-a"]
    assert card.quality_indices == {"coding_index": 50.0}
    assert card.benchmark_scores["swe_bench"] == 0.6
    assert card.capabilities[Capability.CODING] == 5  # P90 of the population


async def test_unmatched_aa_slugs_skipped_without_error(tmp_path):
    catalog = await _registry(tmp_path).catalog()
    # slug-1..slug-4 have no alias; ghost-slug aliases to a missing card.
    assert set(catalog) == {"prov/model-a", "prov/model-b", "prov/model-c"}


async def test_overrides_always_win_over_derived(tmp_path):
    registry = _registry(
        tmp_path,
        "prov/model-a:\n  capabilities:\n    coding: 2\n  tags: [curated]\n"
        "  benchmark_scores:\n    internal_eval: 0.9\n",
    )
    card = (await registry.catalog())["prov/model-a"]
    assert card.capabilities[Capability.CODING] == 2  # override beats derived 5
    assert "curated" in card.tags
    assert card.benchmark_scores["internal_eval"] == 0.9
    assert card.benchmark_scores["swe_bench"] == 0.6  # merged, not replaced


async def test_tools_floors_agentic_at_2(tmp_path):
    catalog = await _registry(tmp_path).catalog()
    # model-a supports tools and AA derived no agentic tier -> floor 2:
    assert catalog["prov/model-a"].capabilities[Capability.AGENTIC] == 2
    # model-b has no tools support -> no agentic tier at all:
    assert Capability.AGENTIC not in catalog["prov/model-b"].capabilities


async def test_curated_agentic_beats_tools_floor(tmp_path):
    registry = _registry(tmp_path, "prov/model-a:\n  capabilities:\n    agentic: 4\n")
    catalog = await registry.catalog()
    assert catalog["prov/model-a"].capabilities[Capability.AGENTIC] == 4


async def test_structured_outputs_tag(tmp_path):
    catalog = await _registry(tmp_path).catalog()
    assert "structured_outputs" in catalog["prov/model-c"].tags
    assert "structured_outputs" not in catalog["prov/model-b"].tags


async def test_later_provider_wins_on_id_clash(tmp_path):
    base = make_card("dup/model", input_price=1.0)
    local = make_card("dup/model", input_price=9.0, provider="local")
    registry = ModelRegistry(
        providers=[StaticProvider("first", [base]), StaticProvider("second", [local])],
        quality_provider=None,
        tier_deriver=TierDeriver(),
        aliases={},
        overrides_provider=LocalBenchmarkProvider(tmp_path / "none.yaml"),
    )
    catalog = await registry.catalog()
    assert catalog["dup/model"].pricing.input_per_million == 9.0


async def test_suggestions_for_misses(tmp_path):
    registry = _registry(tmp_path)
    assert "prov/model-a" in await registry.suggestions("prov/model-aa")


async def test_load_aliases_missing_file(tmp_path):
    assert load_aliases(tmp_path / "absent.yaml") == {}
