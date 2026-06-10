"""Cost estimation: pure period math and the Range-aware engine (§4)."""

from .conversions import frequency_count_per_month, period_multiplier
from .engine import estimate_pipeline, estimate_workload, monthly_totals

__all__ = [
    "estimate_pipeline",
    "estimate_workload",
    "frequency_count_per_month",
    "monthly_totals",
    "period_multiplier",
]
