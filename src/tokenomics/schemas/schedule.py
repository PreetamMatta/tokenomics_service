"""Schedule: converts human duty cycles into monthly hours.

Populated by: API request bodies (inline Schedule objects) or the preset
names shipped here. Units: hours, days, weeks. `hours_per_month` is derived.
"""

from pydantic import BaseModel, Field, computed_field, field_validator

from .range import Num, bounds


class Schedule(BaseModel):
    """Operating schedule for a workload.

    hours_per_day and days_per_week accept Range for variance modeling;
    bounds are validated against (0, 24] and (0, 7] respectively.
    weeks_per_month defaults to 4.345 (~365.25 / 7 / 12).
    Derived: hours_per_month = hours_per_day * days_per_week * weeks_per_month
    (Range-aware).
    """

    name: str = Field(min_length=1)
    hours_per_day: Num
    days_per_week: Num
    weeks_per_month: float = Field(default=4.345, gt=0)

    @field_validator("hours_per_day")
    @classmethod
    def _hours_bounds(cls, v: Num) -> Num:
        lo, hi = bounds(v)
        if lo <= 0 or hi > 24:
            raise ValueError("hours_per_day must satisfy 0 < x <= 24 (Range bounds included)")
        return v

    @field_validator("days_per_week")
    @classmethod
    def _days_bounds(cls, v: Num) -> Num:
        lo, hi = bounds(v)
        if lo <= 0 or hi > 7:
            raise ValueError("days_per_week must satisfy 0 < x <= 7 (Range bounds included)")
        return v

    @computed_field
    @property
    def hours_per_month(self) -> Num:
        """Monthly operating hours H = hours_per_day * days_per_week * weeks_per_month."""
        return self.hours_per_day * self.days_per_week * self.weeks_per_month


PRESET_SCHEDULES: dict[str, Schedule] = {
    "light_4x6": Schedule(name="light_4x6", hours_per_day=4, days_per_week=6),
    "standard_7x5": Schedule(name="standard_7x5", hours_per_day=7, days_per_week=5),
    "heavy_15x7": Schedule(name="heavy_15x7", hours_per_day=15, days_per_week=7),
    "always_on_24x7": Schedule(name="always_on_24x7", hours_per_day=24, days_per_week=7),
}
