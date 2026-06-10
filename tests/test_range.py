"""Range arithmetic and invariants (§3.2). Property-style tests use seeded
random sampling so the invariant is exercised across many operand shapes."""

import random

import pytest
from pydantic import ValidationError

from tokenomics.schemas.range import Range, bounds, typical


def r(lo: float, mid: float, hi: float) -> Range:
    return Range(min=lo, typical=mid, max=hi)


class TestInvariant:
    def test_valid_triple(self):
        assert r(1, 2, 3).typical == 2

    def test_degenerate_triple_allowed(self):
        assert r(2, 2, 2).max == 2

    @pytest.mark.parametrize("triple", [(3, 2, 1), (1, 3, 2), (2, 1, 3)])
    def test_unordered_rejected(self, triple):
        lo, mid, hi = triple
        with pytest.raises(ValidationError):
            Range(min=lo, typical=mid, max=hi)


class TestArithmetic:
    def test_range_plus_range_elementwise(self):
        assert r(1, 2, 3) + r(10, 20, 30) == r(11, 22, 33)

    def test_range_plus_scalar_both_sides(self):
        assert r(1, 2, 3) + 1 == r(2, 3, 4)
        assert 1 + r(1, 2, 3) == r(2, 3, 4)

    def test_range_times_scalar_both_sides(self):
        assert r(1, 2, 3) * 2 == r(2, 4, 6)
        assert 2 * r(1, 2, 3) == r(2, 4, 6)

    def test_range_times_range_elementwise(self):
        assert r(1, 2, 3) * r(4, 5, 6) == r(4, 10, 18)

    def test_subtraction(self):
        assert r(10, 20, 30) - r(1, 2, 3) == r(9, 18, 27)

    def test_division_by_scalar(self):
        assert r(2, 4, 8) / 2 == r(1, 2, 4)

    def test_division_normalizes_order(self):
        # Elementwise division can produce an unordered triple; the result
        # must be re-normalized to satisfy the invariant.
        result = r(10, 20, 40) / r(4, 12, 40)
        assert result.min <= result.typical <= result.max

    def test_scalar_divided_by_range(self):
        result = 100 / r(1, 2, 4)
        assert result == r(25, 50, 100)

    def test_unsupported_operand(self):
        with pytest.raises(TypeError):
            r(1, 2, 3) + "nope"


class TestPropertyStyle:
    def test_invariant_preserved_across_random_ops(self):
        rng = random.Random(42)
        for _ in range(500):
            a = Range._normalized(*(rng.uniform(0, 1000) for _ in range(3)))
            b = Range._normalized(*(rng.uniform(0.001, 1000) for _ in range(3)))
            scalar = rng.uniform(0.001, 100)
            for result in (a + b, a * b, a * scalar, a - b, a / b, a / scalar, scalar * a):
                assert result.min <= result.typical <= result.max

    def test_commutativity(self):
        rng = random.Random(7)
        for _ in range(100):
            a = Range._normalized(*(rng.uniform(0, 100) for _ in range(3)))
            b = Range._normalized(*(rng.uniform(0, 100) for _ in range(3)))
            assert a + b == b + a
            assert a * b == b * a

    def test_scalar_identity(self):
        a = r(1.5, 2.5, 3.5)
        assert a * 1 == a
        assert a + 0 == a


class TestHelpers:
    def test_typical(self):
        assert typical(r(1, 2, 3)) == 2
        assert typical(5.0) == 5.0

    def test_bounds(self):
        assert bounds(r(1, 2, 3)) == (1, 3)
        assert bounds(4.0) == (4.0, 4.0)
