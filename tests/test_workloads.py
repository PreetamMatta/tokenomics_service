"""Workload union round-trip, base validators, and per-type defaults (§3.7)."""

import pytest
from pydantic import TypeAdapter, ValidationError

from tokenomics.schemas.range import Range
from tokenomics.schemas.workloads import (
    Batch,
    BuildAction,
    ContinuousStream,
    Multimodal,
    PerRequest,
    RecurringJob,
    TriggeredAction,
    WorkloadComponent,
)

adapter = TypeAdapter(WorkloadComponent)

EXAMPLES: dict[str, dict] = {
    "continuous_stream": {
        "type": "continuous_stream",
        "role": "listener",
        "model_id": "m",
        "input_tokens_per_hour": 1_500_000,
        "output_tokens_per_hour": 5_000,
    },
    "triggered_action": {
        "type": "triggered_action",
        "role": "executor",
        "model_id": "m",
        "triggers_per_hour": 2,
        "context_tokens_per_trigger": 100_000,
        "output_tokens_per_trigger": 2_000,
    },
    "per_request": {
        "type": "per_request",
        "role": "api",
        "model_id": "m",
        "requests_per_period": 1000,
        "period": "day",
        "input_tokens_per_request": 500,
        "output_tokens_per_request": 200,
    },
    "batch": {
        "type": "batch",
        "role": "etl",
        "model_id": "m",
        "items_per_batch": 100,
        "batches_per_period": 2,
        "period": "week",
        "input_tokens_per_item": 1500,
        "output_tokens_per_item": 50,
    },
    "multimodal": {
        "type": "multimodal",
        "role": "vision",
        "model_id": "m",
        "invocations_per_period": 50,
        "period": "day",
        "images_per_invocation": 3,
        "text_input_per_invocation": 800,
        "text_output_per_invocation": 300,
    },
    "build_action": {
        "type": "build_action",
        "role": "builder",
        "model_id": "m",
        "tasks_per_period": 3,
        "period": "week",
        "iterations_per_task": {"min": 4, "typical": 12, "max": 40},
        "input_tokens_per_iteration": 60_000,
        "output_tokens_per_iteration": 4_000,
        "reasoning_tokens_per_iteration": 2_000,
        "subagent_spawn_probability": 0.3,
    },
    "recurring_job": {
        "type": "recurring_job",
        "role": "memory",
        "model_id": "m",
        "frequency": "daily",
        "input_tokens_per_invocation": 200_000,
        "output_tokens_per_invocation": 5_000,
    },
}

EXPECTED_TYPES = {
    "continuous_stream": ContinuousStream,
    "triggered_action": TriggeredAction,
    "per_request": PerRequest,
    "batch": Batch,
    "multimodal": Multimodal,
    "build_action": BuildAction,
    "recurring_job": RecurringJob,
}


class TestUnionRoundTrip:
    @pytest.mark.parametrize("name", sorted(EXAMPLES))
    def test_discriminated_parse_and_round_trip(self, name):
        parsed = adapter.validate_python(EXAMPLES[name])
        assert isinstance(parsed, EXPECTED_TYPES[name])
        reparsed = adapter.validate_python(parsed.model_dump(mode="json"))
        assert reparsed == parsed

    def test_range_field_parses_as_range(self):
        parsed = adapter.validate_python(EXAMPLES["build_action"])
        assert isinstance(parsed.iterations_per_task, Range)

    def test_unknown_type_rejected(self):
        with pytest.raises(ValidationError):
            adapter.validate_python({**EXAMPLES["per_request"], "type": "nope"})


class TestBaseValidators:
    def test_model_id_and_mix_both_set_rejected(self):
        body = {
            **EXAMPLES["per_request"],
            "mix": [{"model_id": "a", "weight": 1.0}],
        }
        with pytest.raises(ValidationError, match="exactly one"):
            adapter.validate_python(body)

    def test_neither_model_id_nor_mix_rejected(self):
        body = {k: v for k, v in EXAMPLES["per_request"].items() if k != "model_id"}
        with pytest.raises(ValidationError, match="exactly one"):
            adapter.validate_python(body)

    def test_mix_weights_must_sum_to_one(self):
        body = {k: v for k, v in EXAMPLES["per_request"].items() if k != "model_id"}
        body["mix"] = [
            {"model_id": "a", "weight": 0.5},
            {"model_id": "b", "weight": 0.4},
        ]
        with pytest.raises(ValidationError, match="sum to 1"):
            adapter.validate_python(body)

    def test_mix_weights_tolerance(self):
        body = {k: v for k, v in EXAMPLES["per_request"].items() if k != "model_id"}
        body["mix"] = [
            {"model_id": "a", "weight": 0.7},
            {"model_id": "b", "weight": 0.2995},
        ]
        parsed = adapter.validate_python(body)
        assert parsed.mix is not None

    def test_negative_token_field_rejected(self):
        with pytest.raises(ValidationError):
            adapter.validate_python(
                {**EXAMPLES["per_request"], "input_tokens_per_request": -1}
            )

    def test_range_bound_violation_rejected(self):
        with pytest.raises(ValidationError):
            adapter.validate_python(
                {
                    **EXAMPLES["build_action"],
                    "iterations_per_task": {"min": 0, "typical": 5, "max": 10},
                }
            )


class TestCacheDefaults:
    def test_continuous_stream_default(self):
        assert adapter.validate_python(EXAMPLES["continuous_stream"]).cache_hit_ratio == 0.85

    def test_build_action_default(self):
        assert adapter.validate_python(EXAMPLES["build_action"]).cache_hit_ratio == 0.70

    @pytest.mark.parametrize(
        "name", ["triggered_action", "per_request", "batch", "multimodal", "recurring_job"]
    )
    def test_other_defaults_zero(self, name):
        assert adapter.validate_python(EXAMPLES[name]).cache_hit_ratio == 0.0

    def test_cache_write_ratio_default(self):
        assert adapter.validate_python(EXAMPLES["continuous_stream"]).cache_write_ratio == 1.0

    def test_build_action_overhead_default(self):
        parsed = adapter.validate_python(EXAMPLES["build_action"])
        assert parsed.subagent_overhead_multiplier == 1.5
