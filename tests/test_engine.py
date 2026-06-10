"""Cost engine vs hand-computed numbers (§4, §8), Range propagation,
BuildAction multiplier, cache write premium, and the §9 pipeline shape."""

import pytest

from tokenomics.cost.conversions import frequency_count_per_month, period_multiplier
from tokenomics.cost.engine import estimate_pipeline, estimate_workload
from tokenomics.errors import ModelNotFoundError
from tokenomics.schemas.enums import Frequency, Period
from tokenomics.schemas.pipeline import Pipeline
from tokenomics.schemas.pricing import ModelPricing
from tokenomics.schemas.range import Range, typical
from tokenomics.schemas.schedule import PRESET_SCHEDULES
from tokenomics.schemas.workloads import (
    Batch,
    BuildAction,
    ContinuousStream,
    MixEntry,
    Multimodal,
    PerRequest,
    RecurringJob,
    TriggeredAction,
)

STANDARD = PRESET_SCHEDULES["standard_7x5"]  # H = 7 * 5 * 4.345 = 152.075


class TestConversions:
    def test_hours_per_month_anchor(self):
        assert typical(STANDARD.hours_per_month) == pytest.approx(152.075)

    def test_period_multipliers(self):
        assert period_multiplier(Period.DAY, STANDARD) == pytest.approx(5 * 4.345)
        assert period_multiplier(Period.WEEK, STANDARD) == pytest.approx(4.345)
        assert period_multiplier(Period.MONTH, STANDARD) == 1.0

    def test_frequency_counts(self):
        assert frequency_count_per_month(Frequency.HOURLY, STANDARD) == pytest.approx(152.075)
        assert frequency_count_per_month(Frequency.DAILY, STANDARD) == pytest.approx(21.725)
        assert frequency_count_per_month(Frequency.WEEKLY, STANDARD) == pytest.approx(4.345)
        assert frequency_count_per_month(Frequency.MONTHLY, STANDARD) == 1.0


class TestSpecAnchor:
    """The §8 hand-computed ContinuousStream case."""

    def test_continuous_stream_260_05(self):
        pricing = ModelPricing(
            input_per_million=3, output_per_million=15, cache_read_per_million=0.30
        )
        workload = ContinuousStream(
            type="continuous_stream",
            role="listener",
            model_id="m",
            input_tokens_per_hour=2_000_000,
            output_tokens_per_hour=20_000,
            cache_hit_ratio=0.85,
        )
        breakdown = estimate_workload(workload, STANDARD, {"m": pricing})

        assert typical(breakdown.tokens.input_fresh) == pytest.approx(45_622_500)
        assert typical(breakdown.tokens.input_cached) == pytest.approx(258_527_500)
        assert typical(breakdown.tokens.output) == pytest.approx(3_041_500)
        assert abs(typical(breakdown.total_cost) - 260.05) <= 0.01
        # cache write premium is zero (write price defaults to input price),
        # but the fallback default was applied and must be surfaced:
        assert typical(breakdown.costs.cache_write) == 0
        assert "cache_write_per_million" in breakdown.estimated_fields
        # cache read price came from the provider, so it is NOT estimated:
        assert "cache_read_per_million" not in breakdown.estimated_fields
        # continuous streams have no countable action:
        assert breakdown.cost_per_action is None


class TestCacheMath:
    def test_cache_write_premium_only(self):
        # Anthropic-style: write = 1.25x input -> premium 0.75/M over fresh.
        pricing = ModelPricing(
            input_per_million=3,
            output_per_million=15,
            cache_read_per_million=0.30,
            cache_write_per_million=3.75,
        )
        workload = ContinuousStream(
            type="continuous_stream",
            role="listener",
            model_id="m",
            input_tokens_per_hour=2_000_000,
            output_tokens_per_hour=0,
            cache_hit_ratio=0.85,
        )
        breakdown = estimate_workload(workload, STANDARD, {"m": pricing})
        t_fresh = 45_622_500
        assert typical(breakdown.costs.cache_write) == pytest.approx(t_fresh * 0.75 / 1e6)
        assert breakdown.estimated_fields == []

    def test_free_cache_writes_never_negative(self):
        # OpenAI-style: write price 0 (free) must clamp to zero premium.
        pricing = ModelPricing(
            input_per_million=3,
            output_per_million=15,
            cache_read_per_million=1.5,
            cache_write_per_million=0.0,
        )
        workload = ContinuousStream(
            type="continuous_stream",
            role="s",
            model_id="m",
            input_tokens_per_hour=1_000_000,
            cache_hit_ratio=0.5,
        )
        breakdown = estimate_workload(workload, STANDARD, {"m": pricing})
        assert typical(breakdown.costs.cache_write) == 0

    def test_cache_read_fallback_flagged(self):
        pricing = ModelPricing(input_per_million=3, output_per_million=15)
        workload = ContinuousStream(
            type="continuous_stream",
            role="s",
            model_id="m",
            input_tokens_per_hour=1_000_000,
            cache_hit_ratio=0.5,
        )
        breakdown = estimate_workload(workload, STANDARD, {"m": pricing})
        assert "cache_read_per_million" in breakdown.estimated_fields
        # 10%-of-input last resort actually priced the cached tokens:
        t_cached = typical(breakdown.tokens.input_cached)
        assert typical(breakdown.costs.cached_input) == pytest.approx(t_cached * 0.3 / 1e6)


class TestRangePropagation:
    def test_per_request_range_propagates(self):
        pricing = ModelPricing(input_per_million=1, output_per_million=2)
        workload = PerRequest(
            type="per_request",
            role="api",
            model_id="m",
            requests_per_period=Range(min=100, typical=200, max=400),
            period=Period.MONTH,
            input_tokens_per_request=1000,
            output_tokens_per_request=100,
            cache_hit_ratio=0.0,
        )
        breakdown = estimate_workload(workload, STANDARD, {"m": pricing})
        total = breakdown.total_cost
        assert isinstance(total, Range)
        # per request: 1000*1/1e6 + 100*2/1e6 = $0.0012
        assert total.min == pytest.approx(0.12)
        assert total.typical == pytest.approx(0.24)
        assert total.max == pytest.approx(0.48)
        cpa = breakdown.cost_per_action
        assert isinstance(cpa, Range)
        assert cpa.typical == pytest.approx(0.0012)


class TestBuildAction:
    def _workload(self, spawn: float) -> BuildAction:
        return BuildAction(
            type="build_action",
            role="builder",
            model_id="m",
            tasks_per_period=2,
            period=Period.MONTH,
            iterations_per_task=10,
            input_tokens_per_iteration=50_000,
            output_tokens_per_iteration=5_000,
            subagent_spawn_probability=spawn,
            subagent_overhead_multiplier=1.5,
            cache_hit_ratio=0.0,
        )

    def test_subagent_multiplier(self):
        pricing = ModelPricing(input_per_million=3, output_per_million=15)
        baseline = estimate_workload(self._workload(0.0), STANDARD, {"m": pricing})
        boosted = estimate_workload(self._workload(0.5), STANDARD, {"m": pricing})
        # multiplier = 1 + 0.5 * (1.5 - 1) = 1.25 on token-derived cost
        assert typical(baseline.total_cost) == pytest.approx(4.5)  # 1M*3 + 0.1M*15 per 1e6
        assert typical(boosted.total_cost) == pytest.approx(4.5 * 1.25)

    def test_multiplier_not_applied_to_fixed_request_fee(self):
        pricing = ModelPricing(input_per_million=3, output_per_million=15, request_fixed=0.01)
        boosted = estimate_workload(self._workload(0.5), STANDARD, {"m": pricing})
        # 20 iterations * $0.01 fixed, unscaled by the subagent multiplier
        assert typical(boosted.costs.fixed_request) == pytest.approx(0.20)
        assert typical(boosted.total_cost) == pytest.approx(4.5 * 1.25 + 0.20)

    def test_cost_per_action_is_per_task(self):
        pricing = ModelPricing(input_per_million=3, output_per_million=15)
        breakdown = estimate_workload(self._workload(0.0), STANDARD, {"m": pricing})
        assert typical(breakdown.cost_per_action) == pytest.approx(4.5 / 2)

    def test_iterations_range_yields_range_total(self):
        pricing = ModelPricing(input_per_million=3, output_per_million=15)
        workload = self._workload(0.3)
        workload = workload.model_copy(
            update={"iterations_per_task": Range(min=4, typical=12, max=40)}
        )
        breakdown = estimate_workload(workload, STANDARD, {"m": pricing})
        assert isinstance(breakdown.total_cost, Range)
        assert breakdown.total_cost.min < breakdown.total_cost.typical < breakdown.total_cost.max


class TestOtherWorkloads:
    def test_triggered_action_and_reasoning_fallback_flag(self):
        pricing = ModelPricing(input_per_million=2, output_per_million=10)
        workload = TriggeredAction(
            type="triggered_action",
            role="executor",
            model_id="m",
            triggers_per_hour=2,
            context_tokens_per_trigger=100_000,
            output_tokens_per_trigger=2_000,
            reasoning_tokens_per_trigger=1_000,
            cache_hit_ratio=0.0,
        )
        breakdown = estimate_workload(workload, STANDARD, {"m": pricing})
        triggers = 2 * 152.075
        expected = (
            triggers * 100_000 * 2 + triggers * 2_000 * 10 + triggers * 1_000 * 10
        ) / 1e6
        assert typical(breakdown.total_cost) == pytest.approx(expected)
        assert "reasoning_per_million" in breakdown.estimated_fields
        assert typical(breakdown.cost_per_action) == pytest.approx(expected / triggers)

    def test_batch(self):
        pricing = ModelPricing(input_per_million=1, output_per_million=4)
        workload = Batch(
            type="batch",
            role="etl",
            model_id="m",
            items_per_batch=100,
            batches_per_period=2,
            period=Period.WEEK,
            input_tokens_per_item=1_500,
            output_tokens_per_item=50,
        )
        breakdown = estimate_workload(workload, STANDARD, {"m": pricing})
        items = 100 * 2 * 4.345
        expected = (items * 1_500 * 1 + items * 50 * 4) / 1e6
        assert typical(breakdown.total_cost) == pytest.approx(expected)

    def test_multimodal_units_and_missing_image_price(self):
        workload = Multimodal(
            type="multimodal",
            role="vision",
            model_id="m",
            invocations_per_period=10,
            period=Period.MONTH,
            images_per_invocation=3,
            text_input_per_invocation=1_000,
            text_output_per_invocation=100,
        )
        priced = ModelPricing(input_per_million=1, output_per_million=2, image_per_unit=0.005)
        breakdown = estimate_workload(workload, STANDARD, {"m": priced})
        assert typical(breakdown.costs.multimodal_units) == pytest.approx(30 * 0.005)

        unpriced = ModelPricing(input_per_million=1, output_per_million=2)
        breakdown = estimate_workload(workload, STANDARD, {"m": unpriced})
        assert typical(breakdown.costs.multimodal_units) == 0
        assert "image_per_unit" in breakdown.estimated_fields

    def test_recurring_job_daily(self):
        pricing = ModelPricing(input_per_million=0.5, output_per_million=1.0)
        workload = RecurringJob(
            type="recurring_job",
            role="memory",
            model_id="m",
            frequency=Frequency.DAILY,
            input_tokens_per_invocation=200_000,
            output_tokens_per_invocation=5_000,
            cache_hit_ratio=0.0,
        )
        breakdown = estimate_workload(workload, STANDARD, {"m": pricing})
        invocations = 5 * 4.345
        expected = (invocations * 200_000 * 0.5 + invocations * 5_000 * 1.0) / 1e6
        assert typical(breakdown.total_cost) == pytest.approx(expected)


class TestMixAndErrors:
    def test_mix_is_weighted_sum(self):
        cheap = ModelPricing(input_per_million=1, output_per_million=2)
        pricey = ModelPricing(input_per_million=10, output_per_million=20)
        base = dict(
            type="per_request",
            role="api",
            requests_per_period=1000,
            period=Period.MONTH,
            input_tokens_per_request=1000,
            output_tokens_per_request=100,
            cache_hit_ratio=0.0,
        )
        mixed = PerRequest(
            **base, mix=[MixEntry(model_id="a", weight=0.7), MixEntry(model_id="b", weight=0.3)]
        )
        only_a = PerRequest(**base, model_id="a")
        only_b = PerRequest(**base, model_id="b")
        pricing_map = {"a": cheap, "b": pricey}
        mixed_total = typical(estimate_workload(mixed, STANDARD, pricing_map).total_cost)
        a_total = typical(estimate_workload(only_a, STANDARD, pricing_map).total_cost)
        b_total = typical(estimate_workload(only_b, STANDARD, pricing_map).total_cost)
        assert mixed_total == pytest.approx(0.7 * a_total + 0.3 * b_total)

    def test_mix_label(self):
        pricing_map = {
            "a": ModelPricing(input_per_million=1, output_per_million=2),
            "b": ModelPricing(input_per_million=3, output_per_million=4),
        }
        workload = PerRequest(
            type="per_request",
            role="api",
            mix=[MixEntry(model_id="a", weight=0.5), MixEntry(model_id="b", weight=0.5)],
            requests_per_period=10,
            period=Period.MONTH,
            input_tokens_per_request=100,
            output_tokens_per_request=10,
        )
        breakdown = estimate_workload(workload, STANDARD, pricing_map)
        assert breakdown.model_id == "mix(a@0.5,b@0.5)"

    def test_unknown_model_raises(self):
        workload = PerRequest(
            type="per_request",
            role="api",
            model_id="ghost",
            requests_per_period=10,
            period=Period.MONTH,
            input_tokens_per_request=100,
            output_tokens_per_request=10,
        )
        with pytest.raises(ModelNotFoundError):
            estimate_workload(workload, STANDARD, {})

    def test_model_id_override_wins(self):
        pricing_map = {
            "a": ModelPricing(input_per_million=1, output_per_million=2),
            "b": ModelPricing(input_per_million=100, output_per_million=200),
        }
        workload = PerRequest(
            type="per_request",
            role="api",
            model_id="a",
            requests_per_period=1000,
            period=Period.MONTH,
            input_tokens_per_request=1000,
            output_tokens_per_request=0,
        )
        overridden = estimate_workload(workload, STANDARD, pricing_map, model_id_override="b")
        assert overridden.model_id == "b"
        # 1M input tokens at $100/M
        assert typical(overridden.total_cost) == pytest.approx(100.0)


class TestScafflePipeline:
    """§9 worked example: shape assertions for the ambient pipeline."""

    def _pipeline(self) -> Pipeline:
        return Pipeline(
            name="scaffle_ambient",
            components=[
                ContinuousStream(
                    type="continuous_stream",
                    role="ambient_listener",
                    model_id="cheap-fast",
                    input_tokens_per_hour=1_500_000,
                    output_tokens_per_hour=5_000,
                    cache_hit_ratio=0.90,
                ),
                TriggeredAction(
                    type="triggered_action",
                    role="action_executor",
                    model_id="mid-tier",
                    triggers_per_hour=2,
                    context_tokens_per_trigger=100_000,
                    output_tokens_per_trigger=2_000,
                    cache_hit_ratio=0.2,
                ),
                BuildAction(
                    type="build_action",
                    role="code_builder",
                    model_id="frontier",
                    tasks_per_period=3,
                    period=Period.WEEK,
                    iterations_per_task=Range(min=4, typical=12, max=40),
                    input_tokens_per_iteration=60_000,
                    output_tokens_per_iteration=4_000,
                    reasoning_tokens_per_iteration=2_000,
                    subagent_spawn_probability=0.3,
                    subagent_overhead_multiplier=1.5,
                    cache_hit_ratio=0.7,
                ),
                RecurringJob(
                    type="recurring_job",
                    role="daily_memory",
                    model_id="cheap-fast",
                    frequency=Frequency.DAILY,
                    input_tokens_per_invocation=200_000,
                    output_tokens_per_invocation=5_000,
                    cache_hit_ratio=0.5,
                ),
            ],
        )

    def test_pipeline_shape(self):
        pricing_map = {
            "cheap-fast": ModelPricing(
                input_per_million=0.25, output_per_million=2, cache_read_per_million=0.125
            ),
            "mid-tier": ModelPricing(
                input_per_million=3, output_per_million=15, cache_read_per_million=0.3
            ),
            "frontier": ModelPricing(
                input_per_million=15,
                output_per_million=75,
                cache_read_per_million=1.5,
                cache_write_per_million=18.75,
            ),
        }
        result = estimate_pipeline(self._pipeline(), STANDARD, pricing_map)
        assert result.pipeline_name == "scaffle_ambient"
        assert len(result.per_component) == 4
        roles = [c.workload_role for c in result.per_component]
        assert roles == ["ambient_listener", "action_executor", "code_builder", "daily_memory"]

        builder = result.per_component[2]
        assert isinstance(builder.total_cost, Range)  # iterations Range propagates
        assert isinstance(result.total_cost, Range)  # ... into the pipeline total
        # pipeline total is the sum of component totals:
        summed = 0.0
        for component in result.per_component:
            summed = summed + component.total_cost
        assert typical(result.total_cost) == pytest.approx(typical(summed))
