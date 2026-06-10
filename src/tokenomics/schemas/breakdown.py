"""Cost breakdown models returned by the estimation endpoints.

Populated by: the cost engine (tokenomics.cost.engine) — never by users.
Units: token totals are raw token counts per month; cost components and
totals are USD per month. Every field is Range-aware: if any workload or
schedule input was a Range, the corresponding outputs are Ranges.
"""

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .range import Num


class TokenTotals(BaseModel):
    """Monthly token totals after the cache split.

    Derived: input_fresh = T_in - T_in*cache_hit_ratio,
    input_cached = T_in*cache_hit_ratio. Units: tokens/month.
    """

    input_fresh: Num = 0.0
    input_cached: Num = 0.0
    output: Num = 0.0
    reasoning: Num = 0.0


class CostComponents(BaseModel):
    """Monthly USD cost per price dimension.

    Derived by the engine from TokenTotals and ModelPricing. cache_write is
    only the write PREMIUM over fresh input (zero when the provider charges
    no premium). For BuildAction, the subagent multiplier is already baked
    into the token-derived components (fresh_input, cached_input, cache_write,
    output, reasoning) — fixed_request and multimodal_units are not scaled.
    """

    fresh_input: Num = 0.0
    cached_input: Num = 0.0
    cache_write: Num = 0.0
    output: Num = 0.0
    reasoning: Num = 0.0
    fixed_request: Num = 0.0
    multimodal_units: Num = 0.0

    def total(self) -> Num:
        return (
            self.fresh_input
            + self.cached_input
            + self.cache_write
            + self.output
            + self.reasoning
            + self.fixed_request
            + self.multimodal_units
        )


class CostBreakdown(BaseModel):
    """Transparent monthly cost estimate for one workload component.

    model_id is the resolved model, or "mix(a@0.7,b@0.3)" for traffic mixes.
    total_cost is computed as the sum of CostComponents. cost_per_action is
    total / monthly action count when the workload has a countable action
    (trigger/request/item/invocation/task); None for continuous streams.
    estimated_fields lists every price that used a fallback default
    (e.g. cache_read_per_million via the 10%-of-input last resort).
    """

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    workload_role: str
    schedule_name: str
    hours_per_month: Num
    tokens: TokenTotals
    costs: CostComponents
    cost_per_action: Num | None = None
    estimated_fields: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def total_cost(self) -> Num:
        """Sum of all CostComponents, USD/month."""
        return self.costs.total()


class PipelineCostBreakdown(BaseModel):
    """Per-component breakdowns plus the pipeline total (Range-aware)."""

    pipeline_name: str
    schedule_name: str
    per_component: list[CostBreakdown]

    @computed_field
    @property
    def total_cost(self) -> Num:
        """Sum of component totals, USD/month."""
        total: Num = 0.0
        for component in self.per_component:
            total = total + component.total_cost
        return total
