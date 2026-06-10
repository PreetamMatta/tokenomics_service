"""Selection helpers against a synthetic catalog (§5.3)."""

from datetime import date

import pytest

from tests.conftest import make_card
from tokenomics.registry.queries import best, best_value, blended_price, cheapest, filter_models
from tokenomics.schemas.enums import Capability, Modality
from tokenomics.schemas.pricing import ModelPricing

CODING = Capability.CODING


def _catalog():
    return [
        make_card(
            "frontier", 15.0, 75.0, {CODING: 5}, quality_indices={"coding_index": 60.0}
        ),
        make_card("strong", 3.0, 15.0, {CODING: 4}, quality_indices={"coding_index": 45.0}),
        make_card("value", 0.25, 2.0, {CODING: 3}),
        make_card("cheap-weak", 0.1, 0.4, {CODING: 1}),
        make_card("untiered", 0.05, 0.2),
    ]


class TestBlendedPrice:
    def test_default_weights(self):
        pricing = ModelPricing(input_per_million=4, output_per_million=8)
        assert blended_price(pricing) == pytest.approx(0.75 * 4 + 0.25 * 8)

    def test_custom_weights_normalized(self):
        pricing = ModelPricing(input_per_million=4, output_per_million=8)
        assert blended_price(pricing, {"input": 1, "output": 1}) == pytest.approx(6.0)


class TestFilterModels:
    def test_capability_and_tier(self):
        out = filter_models(_catalog(), capability=CODING, min_tier=4)
        assert {c.id for c in out} == {"frontier", "strong"}

    def test_price_and_context_filters(self):
        out = filter_models(_catalog(), max_input_price=1.0)
        assert {c.id for c in out} == {"value", "cheap-weak", "untiered"}

    def test_deprecated_excluded_by_default(self):
        cards = [
            make_card("old", deprecation_date=date(2026, 1, 1)),
            make_card("current"),
        ]
        today = date(2026, 6, 10)
        assert [c.id for c in filter_models(cards, today=today)] == ["current"]
        assert len(filter_models(cards, include_deprecated=True, today=today)) == 2

    def test_modalities_and_supported_parameters(self):
        cards = [
            make_card(
                "vision",
                input_modalities=[Modality.TEXT, Modality.IMAGE],
                supported_parameters=["tools"],
            ),
            make_card("text-only"),
        ]
        assert [
            c.id for c in filter_models(cards, modalities=[Modality.IMAGE])
        ] == ["vision"]
        assert [
            c.id for c in filter_models(cards, supported_parameters=["tools"])
        ] == ["vision"]


class TestCheapest:
    def test_cheapest_meeting_tier_bar(self):
        result = cheapest(_catalog(), CODING, min_tier=3)
        assert result["model"].id == "value"
        justification = result["justification"]
        assert justification["candidates_considered"] == 3
        assert justification["selected"] == "value"
        assert "blended_price_per_million_usd" in justification

    def test_no_candidates_returns_none(self):
        assert cheapest([make_card("x")], CODING, min_tier=3) is None


class TestBest:
    def test_highest_tier_wins(self):
        result = best(_catalog(), CODING)
        assert result["model"].id == "frontier"
        assert result["justification"]["tier"] == 5

    def test_tier_tie_broken_by_evidence_then_price(self):
        cards = [
            make_card("no-evidence", 1.0, 2.0, {CODING: 4}),
            make_card(
                "evidence", 5.0, 10.0, {CODING: 4}, quality_indices={"coding_index": 50.0}
            ),
        ]
        assert best(cards, CODING)["model"].id == "evidence"

        cards = [
            make_card("pricier", 5.0, 10.0, {CODING: 4}),
            make_card("cheaper", 1.0, 2.0, {CODING: 4}),
        ]
        assert best(cards, CODING)["model"].id == "cheaper"


class TestBestValue:
    def test_ranked_top5_with_scores(self):
        result = best_value(_catalog(), CODING, min_tier=1)
        results = result["results"]
        assert len(results) <= 5
        scores = [entry["value_score"] for entry in results]
        assert scores == sorted(scores, reverse=True)
        # tier/blended-dollar: value (3 / 0.6875) beats frontier (5 / 30)
        assert results[0]["model"].id == "cheap-weak" or results[0]["value_score"] >= scores[-1]
        assert "scoring" in result["justification"]

    def test_value_prefers_tier_per_dollar(self):
        cards = [
            make_card("expensive-5", 20.0, 40.0, {CODING: 5}),
            make_card("cheap-3", 0.2, 0.4, {CODING: 3}),
        ]
        assert best_value(cards, CODING)["results"][0]["model"].id == "cheap-3"
