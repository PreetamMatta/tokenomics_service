"""OpenRouter parsing against the checked-in fixture (§6.1), respx-mocked."""

from datetime import date

import pytest
import respx
from httpx import Response

from tests.conftest import load_fixture
from tokenomics.providers.openrouter import DEFAULT_OPENROUTER_URL, OpenRouterAdapter
from tokenomics.schemas.enums import Modality

pytestmark = pytest.mark.asyncio


def _mock_models(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.get(DEFAULT_OPENROUTER_URL).mock(
        return_value=Response(200, json=load_fixture("openrouter_models.json"))
    )


@respx.mock
async def test_per_token_strings_become_per_million_floats():
    _mock_models(respx.mock)
    cards = {c.id: c for c in await OpenRouterAdapter().list_models()}
    claude = cards["anthropic/claude-sonnet-4.6"]
    assert claude.pricing.input_per_million == pytest.approx(3.0)
    assert claude.pricing.output_per_million == pytest.approx(15.0)
    assert claude.pricing.cache_read_per_million == pytest.approx(0.30)
    assert claude.pricing.cache_write_per_million == pytest.approx(3.75)
    # per-unit fees are NOT converted:
    assert claude.pricing.request_fixed == 0.0
    assert claude.pricing.image_per_unit == pytest.approx(0.0048)
    assert claude.pricing.web_search_per_unit == pytest.approx(0.01)
    # internal_reasoning absent -> None -> effective falls back to output:
    assert claude.pricing.reasoning_per_million is None
    assert claude.pricing.effective_reasoning == pytest.approx(15.0)


@respx.mock
async def test_explicit_reasoning_price_parsed():
    _mock_models(respx.mock)
    cards = {c.id: c for c in await OpenRouterAdapter().list_models()}
    r1 = cards["deepseek/deepseek-r1"]
    assert r1.pricing.reasoning_per_million == pytest.approx(2.19)


@respx.mock
async def test_card_fields_mapped():
    _mock_models(respx.mock)
    cards = {c.id: c for c in await OpenRouterAdapter().list_models()}
    claude = cards["anthropic/claude-sonnet-4.6"]
    assert claude.provider == "openrouter"
    assert claude.canonical_slug == "anthropic/claude-sonnet-4.6"
    assert claude.display_name == "Anthropic: Claude Sonnet 4.6"
    assert claude.context_window == 1_000_000
    assert claude.max_output_tokens == 64_000
    assert claude.input_modalities == [Modality.TEXT, Modality.IMAGE]
    assert claude.output_modalities == [Modality.TEXT]
    assert "tools" in claude.supported_parameters
    # unconsumed top-level keys preserved verbatim in metadata:
    assert claude.metadata["created"] == 1767139200
    assert "per_request_limits" in claude.metadata


@respx.mock
async def test_expiration_date_becomes_deprecation_date():
    _mock_models(respx.mock)
    cards = {c.id: c for c in await OpenRouterAdapter().list_models()}
    llama = cards["meta-llama/llama-3.3-70b-instruct"]
    assert llama.deprecation_date == date(2026, 1, 15)


@respx.mock
async def test_dynamic_priced_models_skipped():
    _mock_models(respx.mock)
    cards = {c.id: c for c in await OpenRouterAdapter().list_models()}
    assert "openrouter/auto" not in cards  # "-1" pricing is unusable
    assert len(cards) == 4


@respx.mock
async def test_ttl_cache_and_forced_refresh():
    route = _mock_models(respx.mock)
    adapter = OpenRouterAdapter(ttl_seconds=3600)
    await adapter.list_models()
    await adapter.list_models()
    assert route.call_count == 1  # served from cache
    await adapter.list_models(force=True)
    assert route.call_count == 2
    assert adapter.status.ok
    assert adapter.status.last_refreshed is not None


@respx.mock
async def test_fetch_failure_serves_stale():
    route = _mock_models(respx.mock)
    adapter = OpenRouterAdapter()
    first = await adapter.list_models()
    route.mock(return_value=Response(500))
    stale = await adapter.list_models(force=True)
    assert stale == first
    assert adapter.status.ok is False
