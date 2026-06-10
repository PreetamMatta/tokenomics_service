"""Workload components: the discriminated union of usage patterns.

Populated by: API request bodies (cost/pipeline estimation). Every token/count
field accepts `Num` (float or Range) for variance modeling; constraints are
validated for scalars and Range bounds alike. Units: tokens, counts, hours.

Shared semantics (WorkloadBase):
- exactly one of `model_id` / `mix` must be set (validator);
- mix weights must sum to 1 +/- 1e-3;
- `cache_hit_ratio` is the fraction of input tokens served from cache
  (defaults differ per workload type: ContinuousStream 0.85, BuildAction 0.70,
  everything else 0.0);
- `cache_write_ratio` models the fraction of fresh tokens written to cache
  (default 1.0 — conservative; refine in v2 with real telemetry).
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import Frequency, Period
from .range import NonNegativeNum, PositiveNum


class MixEntry(BaseModel):
    """One model in a traffic mix. Weight is the fraction of traffic routed
    to this model; weights across the mix must sum to 1 +/- 1e-3."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str = Field(min_length=1)
    weight: float = Field(gt=0, le=1)


class WorkloadBase(BaseModel):
    """Shared fields for every workload type. See module docstring for
    cache semantics and the model_id/mix exclusivity rule."""

    model_config = ConfigDict(protected_namespaces=())

    role: str = Field(min_length=1)
    model_id: str | None = None
    mix: list[MixEntry] | None = None
    cache_hit_ratio: float = Field(default=0.0, ge=0, le=1)
    cache_write_ratio: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def _exactly_one_target(self) -> "WorkloadBase":
        if (self.model_id is None) == (self.mix is None):
            raise ValueError("exactly one of model_id or mix must be set")
        if self.mix is not None:
            if not self.mix:
                raise ValueError("mix must contain at least one entry")
            total = sum(entry.weight for entry in self.mix)
            if abs(total - 1.0) > 1e-3:
                raise ValueError(f"mix weights must sum to 1 +/- 1e-3, got {total}")
        return self


class ContinuousStream(WorkloadBase):
    """Always-on token stream (e.g. ambient listener). Token rates are per
    operating hour; monthly totals scale with the schedule's hours_per_month.
    cache_hit_ratio defaults to 0.85 — streams reuse most of their context."""

    type: Literal["continuous_stream"]
    input_tokens_per_hour: PositiveNum
    output_tokens_per_hour: NonNegativeNum = 0.0
    reasoning_tokens_per_hour: NonNegativeNum = 0.0
    cache_hit_ratio: float = Field(default=0.85, ge=0, le=1)


class TriggeredAction(WorkloadBase):
    """Event-driven call (e.g. action executor). Triggers fire per operating
    hour; each trigger consumes a context, produces output, and may think."""

    type: Literal["triggered_action"]
    triggers_per_hour: PositiveNum
    context_tokens_per_trigger: NonNegativeNum
    output_tokens_per_trigger: NonNegativeNum
    reasoning_tokens_per_trigger: NonNegativeNum = 0.0


class PerRequest(WorkloadBase):
    """Request/response traffic counted per calendar period (day/week/month),
    independent of operating hours."""

    type: Literal["per_request"]
    requests_per_period: PositiveNum
    period: Period
    input_tokens_per_request: NonNegativeNum
    output_tokens_per_request: NonNegativeNum
    reasoning_tokens_per_request: NonNegativeNum = 0.0


class Batch(WorkloadBase):
    """Bulk processing in batches per period; each item is one request."""

    type: Literal["batch"]
    items_per_batch: PositiveNum
    batches_per_period: PositiveNum
    period: Period
    input_tokens_per_item: NonNegativeNum
    output_tokens_per_item: NonNegativeNum


class Multimodal(WorkloadBase):
    """Invocations that mix text tokens with unit-priced media. Images are
    priced via ModelPricing.image_per_unit; audio seconds carry no price
    dimension in v1 and are surfaced via estimated_fields when present."""

    type: Literal["multimodal"]
    invocations_per_period: PositiveNum
    period: Period
    images_per_invocation: NonNegativeNum = 0.0
    audio_seconds_per_invocation: NonNegativeNum = 0.0
    text_input_per_invocation: NonNegativeNum = 0.0
    text_output_per_invocation: NonNegativeNum = 0.0


class BuildAction(WorkloadBase):
    """Agentic build/code task: tasks iterate, may spawn subagents, and burn
    reasoning tokens. iterations_per_task is the main variance source — use a
    Range. Subagent overhead applies a multiplier to the token-derived cost:
    multiplier = 1 + spawn_probability * (overhead_multiplier - 1).
    cache_hit_ratio defaults to 0.70 — agent loops re-read large contexts."""

    type: Literal["build_action"]
    tasks_per_period: PositiveNum
    period: Period
    iterations_per_task: PositiveNum  # the main variance source — use Range
    input_tokens_per_iteration: NonNegativeNum
    output_tokens_per_iteration: NonNegativeNum
    reasoning_tokens_per_iteration: NonNegativeNum = 0.0
    subagent_spawn_probability: float = Field(default=0.0, ge=0, le=1)
    subagent_overhead_multiplier: float = Field(default=1.5, ge=1)
    cache_hit_ratio: float = Field(default=0.70, ge=0, le=1)


class RecurringJob(WorkloadBase):
    """Cron-style job at a fixed frequency. `hourly` scales with operating
    hours; daily/weekly/monthly are calendar-based."""

    type: Literal["recurring_job"]
    frequency: Frequency
    input_tokens_per_invocation: NonNegativeNum
    output_tokens_per_invocation: NonNegativeNum
    reasoning_tokens_per_invocation: NonNegativeNum = 0.0


WorkloadComponent = Annotated[
    ContinuousStream | TriggeredAction | PerRequest | Batch | Multimodal | BuildAction | RecurringJob,
    Field(discriminator="type"),
]
