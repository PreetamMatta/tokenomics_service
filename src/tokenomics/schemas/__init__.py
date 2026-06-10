"""Pydantic schema package — the service contract (§3 of the spec)."""

from .breakdown import CostBreakdown, CostComponents, PipelineCostBreakdown, TokenTotals
from .enums import (
    CAPABILITY_DESCRIPTIONS,
    TIER_SEMANTICS,
    Capability,
    Frequency,
    Modality,
    Period,
)
from .model_card import ModelCard, PerformanceProfile
from .pipeline import Pipeline
from .pricing import ModelPricing
from .range import NonNegativeNum, Num, PositiveNum, Range, bounds, typical
from .schedule import PRESET_SCHEDULES, Schedule
from .workloads import (
    Batch,
    BuildAction,
    ContinuousStream,
    MixEntry,
    Multimodal,
    PerRequest,
    RecurringJob,
    TriggeredAction,
    WorkloadBase,
    WorkloadComponent,
)

__all__ = [
    "CAPABILITY_DESCRIPTIONS",
    "PRESET_SCHEDULES",
    "TIER_SEMANTICS",
    "Batch",
    "BuildAction",
    "Capability",
    "ContinuousStream",
    "CostBreakdown",
    "CostComponents",
    "Frequency",
    "MixEntry",
    "Modality",
    "ModelCard",
    "ModelPricing",
    "Multimodal",
    "NonNegativeNum",
    "Num",
    "PerRequest",
    "PerformanceProfile",
    "Period",
    "Pipeline",
    "PipelineCostBreakdown",
    "PositiveNum",
    "Range",
    "RecurringJob",
    "Schedule",
    "TokenTotals",
    "TriggeredAction",
    "WorkloadBase",
    "WorkloadComponent",
    "bounds",
    "typical",
]
