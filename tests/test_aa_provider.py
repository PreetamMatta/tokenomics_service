"""Artificial Analysis parsing and tier derivation from fixture indices (§6.2)."""

import pytest
import respx
from httpx import Response

from tests.conftest import load_fixture
from tokenomics.benchmarks.artificial_analysis import DEFAULT_AA_URL, AABenchmarkProvider
from tokenomics.benchmarks.tiers import TierDeriver, percentile_of
from tokenomics.schemas.enums import Capability


def _mock_aa(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.get(DEFAULT_AA_URL).mock(
        return_value=Response(200, json=load_fixture("aa_models.json"))
    )


class TestProviderParsing:
    @respx.mock
    @pytest.mark.asyncio
    async def test_indices_benchmarks_and_performance_parsed(self):
        route = _mock_aa(respx.mock)
        provider = AABenchmarkProvider(api_key="test-key")
        records = {r.slug: r for r in await provider.records()}

        assert route.calls.last.request.headers["x-api-key"] == "test-key"
        claude = records["claude-sonnet-4-6"]
        # composite indices, prefix stripped, kept raw:
        assert claude.quality_indices["intelligence_index"] == 50.0
        assert claude.quality_indices["coding_index"] == 50.0
        # individual evals land in benchmark_scores:
        assert claude.benchmark_scores["gpqa"] == 0.66
        assert claude.benchmark_scores["swe_bench"] == 0.62
        assert "intelligence_index" not in claude.benchmark_scores
        # performance: seconds -> ms
        assert claude.performance.output_tokens_per_second == pytest.approx(72.4)
        assert claude.performance.time_to_first_token_ms == pytest.approx(1910.0)

    @respx.mock
    @pytest.mark.asyncio
    async def test_cached_after_first_fetch(self):
        route = _mock_aa(respx.mock)
        provider = AABenchmarkProvider(api_key="test-key")
        await provider.records()
        await provider.records()
        assert route.call_count == 1

    @pytest.mark.asyncio
    async def test_no_api_key_degrades_gracefully(self):
        provider = AABenchmarkProvider(api_key=None)
        assert await provider.records() == []
        assert provider.status.ok
        assert "disabled" in provider.status.detail


class TestPercentiles:
    def test_percentile_of(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert percentile_of(values, 50.0) == pytest.approx(90.0)
        assert percentile_of(values, 40.0) == pytest.approx(70.0)
        assert percentile_of(values, 30.0) == pytest.approx(50.0)
        assert percentile_of(values, 20.0) == pytest.approx(30.0)
        assert percentile_of(values, 10.0) == pytest.approx(10.0)

    def test_threshold_mapping(self):
        deriver = TierDeriver()
        assert deriver.tier_for_percentile(95) == 5
        assert deriver.tier_for_percentile(90) == 5
        assert deriver.tier_for_percentile(75) == 4
        assert deriver.tier_for_percentile(50) == 3
        assert deriver.tier_for_percentile(20) == 2
        assert deriver.tier_for_percentile(5) == 1


class TestTierDerivation:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fixture_population_tiers(self):
        _mock_aa(respx.mock)
        provider = AABenchmarkProvider(api_key="test-key")
        records = await provider.records()
        derived = TierDeriver().derive_population(records)

        # Fixture indices are evenly spaced 10..50 across 5 models, so each
        # model lands in a distinct tier for every mapped capability.
        expected = {
            "claude-sonnet-4-6": 5,
            "gpt-5-2-mini": 4,
            "deepseek-r1": 3,
            "mid-model": 2,
            "small-model": 1,
        }
        for slug, tier in expected.items():
            for capability in (
                Capability.REASONING,
                Capability.CODING,
                Capability.AGENTIC,
                Capability.MATH,
            ):
                assert derived[slug][capability] == tier, (slug, capability)

    def test_yaml_round_trip(self, tmp_path):
        config = tmp_path / "tiers.yaml"
        config.write_text(
            "percentile_thresholds:\n  5: 95\n  4: 60\n  3: 30\n  2: 10\n"
            "index_capability_map:\n  coding_index: coding\n"
        )
        deriver = TierDeriver.from_yaml(config)
        assert deriver.thresholds[5] == 95
        assert deriver.index_capability_map == {"coding_index": Capability.CODING}
        assert deriver.tier_for_percentile(94) == 4

    def test_missing_yaml_uses_defaults(self, tmp_path):
        deriver = TierDeriver.from_yaml(tmp_path / "absent.yaml")
        assert deriver.thresholds == {5: 90.0, 4: 70.0, 3: 40.0, 2: 15.0}
