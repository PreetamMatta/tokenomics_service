"""Period/frequency -> monthly conversion math (§4). Pure, Range-aware.

Conversion table:
- period: day -> * days_per_week * weeks_per_month; week -> * weeks_per_month;
  month -> * 1.
- frequency: hourly -> H (monthly operating hours); daily -> days_per_week *
  weeks_per_month; weekly -> weeks_per_month; monthly -> 1.
"""

from ..schemas.enums import Frequency, Period
from ..schemas.range import Num
from ..schemas.schedule import Schedule


def period_multiplier(period: Period, schedule: Schedule) -> Num:
    """How many of `period` occur in one month under `schedule`."""
    if period is Period.DAY:
        return schedule.days_per_week * schedule.weeks_per_month
    if period is Period.WEEK:
        return schedule.weeks_per_month
    return 1.0


def frequency_count_per_month(frequency: Frequency, schedule: Schedule) -> Num:
    """Monthly invocation count for a recurring job at `frequency`."""
    if frequency is Frequency.HOURLY:
        return schedule.hours_per_month
    if frequency is Frequency.DAILY:
        return schedule.days_per_week * schedule.weeks_per_month
    if frequency is Frequency.WEEKLY:
        return schedule.weeks_per_month
    return 1.0
