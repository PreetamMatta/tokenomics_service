"""The cost engine: pure functions implementing §4 of the spec. No I/O.

Per workload type, `monthly_totals` converts workload fields into monthly
token totals (T_in, T_out, T_reason), a monthly request count, unit counts
(images, web searches, audio seconds), a countable action count, and the
BuildAction subagent cost multiplier. `estimate_workload` then applies the
cache split and the authoritative cost formula:

    C = ( T_fresh   * p_in
        + T_cached  * p_cache_read
        + T_writes  * max(p_cache_write - p_in, 0)   # write PREMIUM only
        + T_out     * p_out
        + T_reason  * p_reasoning
        ) / 1e6
        + N_req * p_request_fixed
        + N_images * p_image_unit + ...

All arithmetic is Range-aware: any Range input yields Range outputs.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from ..errors import ModelNotFoundError
from ..schemas.breakdown import CostBreakdown, CostComponents, PipelineCostBreakdown, TokenTotals
from ..schemas.pipeline import Pipeline
from ..schemas.pricing import ModelPricing
from ..schemas.range import Num, typical
from ..schemas.schedule import Schedule
from ..schemas.workloads import (
    Batch,
    BuildAction,
    ContinuousStream,
    Multimodal,
    PerRequest,
    RecurringJob,
    TriggeredAction,
    WorkloadComponent,
)
from .conversions import frequency_count_per_month, period_multiplier


@dataclass
class MonthlyTotals:
    """Workload-type-agnostic monthly quantities consumed by the pricer.

    Units: tokens (t_*), counts (n_requests, images, web_searches, actions),
    seconds (audio_seconds). `actions` is None when the workload has no
    countable action (ContinuousStream). `cost_multiplier` is the BuildAction
    subagent factor applied to token-derived costs only.
    """

    t_in: Num = 0.0
    t_out: Num = 0.0
    t_reason: Num = 0.0
    n_requests: Num = 0.0
    images: Num = 0.0
    web_searches: Num = 0.0
    audio_seconds: Num = 0.0
    actions: Num | None = None
    cost_multiplier: float = field(default=1.0)


def monthly_totals(workload: WorkloadComponent, schedule: Schedule) -> MonthlyTotals:
    """Convert one workload component into monthly totals under `schedule`."""
    hours = schedule.hours_per_month

    if isinstance(workload, ContinuousStream):
        return MonthlyTotals(
            t_in=workload.input_tokens_per_hour * hours,
            t_out=workload.output_tokens_per_hour * hours,
            t_reason=workload.reasoning_tokens_per_hour * hours,
            actions=None,  # no countable action for a continuous stream
        )

    if isinstance(workload, TriggeredAction):
        triggers = workload.triggers_per_hour * hours
        return MonthlyTotals(
            t_in=workload.context_tokens_per_trigger * triggers,
            t_out=workload.output_tokens_per_trigger * triggers,
            t_reason=workload.reasoning_tokens_per_trigger * triggers,
            n_requests=triggers,
            actions=triggers,
        )

    if isinstance(workload, PerRequest):
        requests = workload.requests_per_period * period_multiplier(workload.period, schedule)
        return MonthlyTotals(
            t_in=workload.input_tokens_per_request * requests,
            t_out=workload.output_tokens_per_request * requests,
            t_reason=workload.reasoning_tokens_per_request * requests,
            n_requests=requests,
            actions=requests,
        )

    if isinstance(workload, Batch):
        batches = workload.batches_per_period * period_multiplier(workload.period, schedule)
        items = workload.items_per_batch * batches
        return MonthlyTotals(
            t_in=workload.input_tokens_per_item * items,
            t_out=workload.output_tokens_per_item * items,
            n_requests=items,
            actions=items,
        )

    if isinstance(workload, Multimodal):
        invocations = workload.invocations_per_period * period_multiplier(
            workload.period, schedule
        )
        return MonthlyTotals(
            t_in=workload.text_input_per_invocation * invocations,
            t_out=workload.text_output_per_invocation * invocations,
            images=workload.images_per_invocation * invocations,
            audio_seconds=workload.audio_seconds_per_invocation * invocations,
            n_requests=invocations,
            actions=invocations,
        )

    if isinstance(workload, BuildAction):
        tasks = workload.tasks_per_period * period_multiplier(workload.period, schedule)
        iterations = workload.iterations_per_task * tasks
        multiplier = 1.0 + workload.subagent_spawn_probability * (
            workload.subagent_overhead_multiplier - 1.0
        )
        return MonthlyTotals(
            t_in=workload.input_tokens_per_iteration * iterations,
            t_out=workload.output_tokens_per_iteration * iterations,
            t_reason=workload.reasoning_tokens_per_iteration * iterations,
            n_requests=iterations,
            actions=tasks,
            cost_multiplier=multiplier,
        )

    if isinstance(workload, RecurringJob):
        invocations = frequency_count_per_month(workload.frequency, schedule)
        return MonthlyTotals(
            t_in=workload.input_tokens_per_invocation * invocations,
            t_out=workload.output_tokens_per_invocation * invocations,
            t_reason=workload.reasoning_tokens_per_invocation * invocations,
            n_requests=invocations,
            actions=invocations,
        )

    raise TypeError(f"unsupported workload type: {type(workload).__name__}")


def _price_for_model(
    totals: MonthlyTotals,
    workload: WorkloadComponent,
    pricing: ModelPricing,
) -> tuple[TokenTotals, CostComponents, list[str]]:
    """Apply the §4 formula for one model. Returns tokens, costs, and the
    list of price fields that used fallback defaults."""
    estimated: list[str] = []
    multiplier = totals.cost_multiplier

    cache_ratio = workload.cache_hit_ratio
    t_cached = totals.t_in * cache_ratio
    t_fresh = totals.t_in * (1.0 - cache_ratio)
    t_writes = t_fresh * workload.cache_write_ratio

    fresh_cost = t_fresh * pricing.input_per_million / 1e6 * multiplier
    cached_cost = t_cached * pricing.effective_cache_read / 1e6 * multiplier
    if typical(t_cached) > 0 and pricing.cache_read_per_million is None:
        estimated.append("cache_read_per_million")

    # Only the write PREMIUM over fresh input; clamped at zero so providers
    # with free/discounted writes never produce a negative component.
    write_premium = max(pricing.effective_cache_write - pricing.input_per_million, 0.0)
    write_cost = t_writes * write_premium / 1e6 * multiplier
    if typical(t_writes) > 0 and pricing.cache_write_per_million is None:
        estimated.append("cache_write_per_million")

    output_cost = totals.t_out * pricing.output_per_million / 1e6 * multiplier
    reasoning_cost = totals.t_reason * pricing.effective_reasoning / 1e6 * multiplier
    if typical(totals.t_reason) > 0 and pricing.reasoning_per_million is None:
        estimated.append("reasoning_per_million")

    fixed_cost = totals.n_requests * pricing.request_fixed

    unit_cost: Num = 0.0
    if typical(totals.images) > 0:
        if pricing.image_per_unit is None:
            estimated.append("image_per_unit")
        else:
            unit_cost = unit_cost + totals.images * pricing.image_per_unit
    if typical(totals.web_searches) > 0:
        if pricing.web_search_per_unit is None:
            estimated.append("web_search_per_unit")
        else:
            unit_cost = unit_cost + totals.web_searches * pricing.web_search_per_unit
    if typical(totals.audio_seconds) > 0:
        # v1 has no audio price dimension; surface it rather than silently drop.
        estimated.append("audio_unpriced")
    if pricing.pricing_estimated:
        estimated.append("provider_pricing_estimated")

    tokens = TokenTotals(
        input_fresh=t_fresh,
        input_cached=t_cached,
        output=totals.t_out,
        reasoning=totals.t_reason,
    )
    costs = CostComponents(
        fresh_input=fresh_cost,
        cached_input=cached_cost,
        cache_write=write_cost,
        output=output_cost,
        reasoning=reasoning_cost,
        fixed_request=fixed_cost,
        multimodal_units=unit_cost,
    )
    return tokens, costs, estimated


def _scale_components(costs: CostComponents, weight: float) -> CostComponents:
    return CostComponents(
        fresh_input=costs.fresh_input * weight,
        cached_input=costs.cached_input * weight,
        cache_write=costs.cache_write * weight,
        output=costs.output * weight,
        reasoning=costs.reasoning * weight,
        fixed_request=costs.fixed_request * weight,
        multimodal_units=costs.multimodal_units * weight,
    )


def _add_components(a: CostComponents, b: CostComponents) -> CostComponents:
    return CostComponents(
        fresh_input=a.fresh_input + b.fresh_input,
        cached_input=a.cached_input + b.cached_input,
        cache_write=a.cache_write + b.cache_write,
        output=a.output + b.output,
        reasoning=a.reasoning + b.reasoning,
        fixed_request=a.fixed_request + b.fixed_request,
        multimodal_units=a.multimodal_units + b.multimodal_units,
    )


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def estimate_workload(
    workload: WorkloadComponent,
    schedule: Schedule,
    pricing_map: Mapping[str, ModelPricing],
    model_id_override: str | None = None,
) -> CostBreakdown:
    """Estimate one workload component. Pure: pricing comes from `pricing_map`
    (model_id -> ModelPricing); unknown ids raise ModelNotFoundError.

    `model_id_override` wins over workload.model_id/mix (used by the compare
    endpoint and the top-level model_id field of the estimate request).
    """
    totals = monthly_totals(workload, schedule)

    if model_id_override is not None:
        targets: list[tuple[str, float]] = [(model_id_override, 1.0)]
    elif workload.model_id is not None:
        targets = [(workload.model_id, 1.0)]
    else:
        targets = [(entry.model_id, entry.weight) for entry in workload.mix or []]

    tokens: TokenTotals | None = None
    combined = CostComponents()
    estimated: list[str] = []
    for model_id, weight in targets:
        pricing = pricing_map.get(model_id)
        if pricing is None:
            raise ModelNotFoundError(model_id)
        model_tokens, model_costs, model_estimated = _price_for_model(totals, workload, pricing)
        tokens = model_tokens  # token totals are workload-level, identical per target
        combined = _add_components(combined, _scale_components(model_costs, weight))
        if len(targets) == 1:
            estimated.extend(model_estimated)
        else:
            estimated.extend(f"{model_id}:{name}" for name in model_estimated)

    if len(targets) == 1:
        label = targets[0][0]
    else:
        label = "mix(" + ",".join(f"{mid}@{w:g}" for mid, w in targets) + ")"

    total = combined.total()
    cost_per_action: Num | None = None
    if totals.actions is not None and typical(totals.actions) > 0:
        cost_per_action = total / totals.actions

    return CostBreakdown(
        model_id=label,
        workload_role=workload.role,
        schedule_name=schedule.name,
        hours_per_month=schedule.hours_per_month,
        tokens=tokens or TokenTotals(),
        costs=combined,
        cost_per_action=cost_per_action,
        estimated_fields=_dedupe(estimated),
    )


def estimate_pipeline(
    pipeline: Pipeline,
    schedule: Schedule,
    pricing_map: Mapping[str, ModelPricing],
) -> PipelineCostBreakdown:
    """Estimate every component of a pipeline; the total is the sum (§4)."""
    breakdowns = [
        estimate_workload(component, schedule, pricing_map) for component in pipeline.components
    ]
    return PipelineCostBreakdown(
        pipeline_name=pipeline.name,
        schedule_name=schedule.name,
        per_component=breakdowns,
    )
