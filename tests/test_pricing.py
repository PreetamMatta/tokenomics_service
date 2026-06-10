"""ModelPricing defaults, computed effective fields, and estimated flags (§3.3)."""

import pytest
from pydantic import ValidationError

from tokenomics.schemas.pricing import ModelPricing


class TestEffectiveFields:
    def test_reasoning_defaults_to_output(self):
        pricing = ModelPricing(input_per_million=3, output_per_million=15)
        assert pricing.effective_reasoning == 15

    def test_explicit_reasoning_wins(self):
        pricing = ModelPricing(input_per_million=3, output_per_million=15, reasoning_per_million=2.19)
        assert pricing.effective_reasoning == 2.19

    def test_reasoning_zero_means_free_not_fallback(self):
        # OpenRouter "0" means free; only None triggers the output-price default.
        pricing = ModelPricing(input_per_million=3, output_per_million=15, reasoning_per_million=0)
        assert pricing.effective_reasoning == 0

    def test_cache_read_fallback_is_10_percent_of_input(self):
        pricing = ModelPricing(input_per_million=3, output_per_million=15)
        assert pricing.effective_cache_read == pytest.approx(0.3)

    def test_explicit_cache_read_wins(self):
        pricing = ModelPricing(
            input_per_million=3, output_per_million=15, cache_read_per_million=1.5
        )
        assert pricing.effective_cache_read == 1.5

    def test_cache_write_defaults_to_input_no_premium(self):
        pricing = ModelPricing(input_per_million=3, output_per_million=15)
        assert pricing.effective_cache_write == 3

    def test_explicit_cache_write_wins(self):
        pricing = ModelPricing(
            input_per_million=3, output_per_million=15, cache_write_per_million=3.75
        )
        assert pricing.effective_cache_write == 3.75


class TestDefaultsAndConstraints:
    def test_defaults(self):
        pricing = ModelPricing(input_per_million=1, output_per_million=2)
        assert pricing.request_fixed == 0.0
        assert pricing.image_per_unit is None
        assert pricing.web_search_per_unit is None
        assert pricing.cache_storage_per_million_hour is None
        assert pricing.pricing_estimated is False

    @pytest.mark.parametrize(
        "field",
        ["input_per_million", "output_per_million", "request_fixed", "cache_read_per_million"],
    )
    def test_negative_prices_rejected(self, field):
        kwargs = {"input_per_million": 1.0, "output_per_million": 2.0, field: -0.1}
        with pytest.raises(ValidationError):
            ModelPricing(**kwargs)

    def test_effective_fields_serialized(self):
        pricing = ModelPricing(input_per_million=3, output_per_million=15)
        dumped = pricing.model_dump()
        assert dumped["effective_reasoning"] == 15
        assert dumped["effective_cache_read"] == pytest.approx(0.3)
        assert dumped["effective_cache_write"] == 3
