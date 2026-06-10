import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tokenomics.schemas.enums import Capability
from tokenomics.schemas.model_card import ModelCard
from tokenomics.schemas.pricing import ModelPricing

FIXTURES = Path(__file__).parent / "fixtures"

NOW = datetime(2026, 6, 10, tzinfo=UTC)


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def make_card(
    model_id: str,
    input_price: float = 1.0,
    output_price: float = 2.0,
    capabilities: dict[Capability, int] | None = None,
    **kwargs,
) -> ModelCard:
    defaults = dict(
        id=model_id,
        provider="test",
        display_name=model_id,
        pricing=ModelPricing(input_per_million=input_price, output_per_million=output_price),
        context_window=128_000,
        capabilities=capabilities or {},
        last_refreshed=NOW,
    )
    defaults.update(kwargs)
    return ModelCard(**defaults)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
