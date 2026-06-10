"""Range: min/typical/max triple for uncertain quantities.

`Num = float | Range` is the type alias used by every workload/schedule field
that supports variance modeling. Any computation with a Range input produces a
Range output, so uncertainty propagates end-to-end through the cost engine.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated

from pydantic import AfterValidator, BaseModel, model_validator

_NUMERIC = (int, float)


class Range(BaseModel):
    """min/typical/max triple for uncertain quantities.

    Populated by: user input on workload/schedule fields, or derived by
    Range-aware arithmetic inside the cost engine.
    Units: whatever the quantity it models uses (tokens, hours, USD, ...).
    Derived vs raw: raw when supplied in a request; every arithmetic result
    is derived.

    Arithmetic: Range+Range, Range*scalar, Range*Range (elementwise), plus
    subtraction and division. Results are normalized so the invariant
    min <= typical <= max always holds (relevant for division, where an
    elementwise result can come out unordered).
    """

    min: float
    typical: float
    max: float

    @model_validator(mode="after")
    def _ordered(self) -> Range:
        if not (self.min <= self.typical <= self.max):
            raise ValueError(
                f"Range invariant violated: expected min <= typical <= max, "
                f"got min={self.min}, typical={self.typical}, max={self.max}"
            )
        return self

    # -- arithmetic ---------------------------------------------------------

    @classmethod
    def _normalized(cls, a: float, b: float, c: float) -> Range:
        lo, mid, hi = sorted((a, b, c))
        return cls(min=lo, typical=mid, max=hi)

    @staticmethod
    def _triple(x: Range | float) -> tuple[float, float, float]:
        if isinstance(x, Range):
            return (x.min, x.typical, x.max)
        return (float(x), float(x), float(x))

    def _combine(
        self,
        other: object,
        op: Callable[[float, float], float],
        reflected: bool = False,
    ) -> Range:
        if not isinstance(other, (Range, *_NUMERIC)):
            return NotImplemented
        left, right = (other, self) if reflected else (self, other)
        l1, l2, l3 = self._triple(left)
        r1, r2, r3 = self._triple(right)
        return Range._normalized(op(l1, r1), op(l2, r2), op(l3, r3))

    def __add__(self, other: object) -> Range:
        return self._combine(other, operator.add)

    def __radd__(self, other: object) -> Range:
        return self._combine(other, operator.add, reflected=True)

    def __sub__(self, other: object) -> Range:
        return self._combine(other, operator.sub)

    def __rsub__(self, other: object) -> Range:
        return self._combine(other, operator.sub, reflected=True)

    def __mul__(self, other: object) -> Range:
        return self._combine(other, operator.mul)

    def __rmul__(self, other: object) -> Range:
        return self._combine(other, operator.mul, reflected=True)

    def __truediv__(self, other: object) -> Range:
        return self._combine(other, operator.truediv)

    def __rtruediv__(self, other: object) -> Range:
        return self._combine(other, operator.truediv, reflected=True)


Num = float | Range
"""Type alias for fields that accept either a scalar or a Range."""


def typical(x: Num) -> float:
    """The point estimate of a Num: Range.typical, or the scalar itself."""
    return x.typical if isinstance(x, Range) else float(x)


def bounds(x: Num) -> tuple[float, float]:
    """(min, max) of a Num; for scalars both are the value itself."""
    if isinstance(x, Range):
        return (x.min, x.max)
    return (float(x), float(x))


def _validate_positive(v: Num) -> Num:
    lo, _ = bounds(v)
    if lo <= 0:
        raise ValueError("value must be > 0 (Range.min included)")
    return v


def _validate_non_negative(v: Num) -> Num:
    lo, _ = bounds(v)
    if lo < 0:
        raise ValueError("value must be >= 0 (Range.min included)")
    return v


PositiveNum = Annotated[Num, AfterValidator(_validate_positive)]
"""A Num constrained to be strictly positive (Range bounds validated)."""

NonNegativeNum = Annotated[Num, AfterValidator(_validate_non_negative)]
"""A Num constrained to be >= 0 (Range bounds validated)."""
