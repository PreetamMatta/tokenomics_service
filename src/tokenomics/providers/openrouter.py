"""OpenRouterAdapter: parses openrouter.ai/api/v1/models into ModelCards (§6.1).

Parsing rules:
- pricing values are strings, USD per TOKEN -> float(x) * 1_000_000 for the
  per-million fields. "0" means free. `request`, `image`, `web_search` are
  per-unit fees and are NOT converted.
- field map: prompt->input, completion->output, internal_reasoning->reasoning,
  input_cache_read->cache_read, input_cache_write->cache_write,
  request->request_fixed, image->image_per_unit, web_search->web_search_per_unit.
- context_length->context_window; top_provider.max_completion_tokens->
  max_output_tokens; architecture.input_modalities/output_modalities->
  modalities; supported_parameters passthrough; expiration_date->
  deprecation_date; canonical_slug passthrough; every other top-level key
  lands verbatim in ModelCard.metadata.
- entries with negative/dynamic pricing (e.g. "-1") or no context length are
  skipped and logged.

The `tools` in supported_parameters -> AGENTIC tier floor of 2 is applied in
the registry (after AA derivation and overrides), not here — see §6.1/§6.4.
"""

import logging
from datetime import UTC, date, datetime
from typing import Any

import httpx

from ..schemas.enums import Modality
from ..schemas.model_card import ModelCard
from ..schemas.pricing import ModelPricing
from .base import ModelProvider, ProviderStatus
from .cache import TTLCache

logger = logging.getLogger(__name__)

DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/models"

_TOKEN_PRICE_FIELDS = {
    "prompt": "input_per_million",
    "completion": "output_per_million",
    "internal_reasoning": "reasoning_per_million",
    "input_cache_read": "cache_read_per_million",
    "input_cache_write": "cache_write_per_million",
}
_UNIT_PRICE_FIELDS = {
    "request": "request_fixed",
    "image": "image_per_unit",
    "web_search": "web_search_per_unit",
}
_CONSUMED_KEYS = {
    "id",
    "canonical_slug",
    "name",
    "description",
    "pricing",
    "context_length",
    "architecture",
    "top_provider",
    "supported_parameters",
    "expiration_date",
}


def _parse_pricing(raw: dict[str, Any]) -> ModelPricing:
    kwargs: dict[str, Any] = {}
    for key, target in _TOKEN_PRICE_FIELDS.items():
        value = raw.get(key)
        if value is None or value == "":
            continue
        per_token = float(value)
        if per_token < 0:
            raise ValueError(f"dynamic/negative price for {key}: {value}")
        kwargs[target] = per_token * 1_000_000
    for key, target in _UNIT_PRICE_FIELDS.items():
        value = raw.get(key)
        if value is None or value == "":
            continue
        per_unit = float(value)
        if per_unit < 0:
            raise ValueError(f"dynamic/negative price for {key}: {value}")
        kwargs[target] = per_unit
    # prompt/completion are required dimensions; flag the card if absent.
    estimated = False
    for required in ("input_per_million", "output_per_million"):
        if required not in kwargs:
            kwargs[required] = 0.0
            estimated = True
    kwargs["pricing_estimated"] = estimated
    return ModelPricing(**kwargs)


def _parse_modalities(values: list[str] | None) -> list[Modality]:
    out = [Modality(v) for v in values or [] if v in Modality._value2member_map_]
    return out or [Modality.TEXT]


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(str(value)[:10])


def parse_openrouter_model(entry: dict[str, Any], refreshed_at: datetime) -> ModelCard:
    """Parse one /api/v1/models entry. Raises ValueError on unusable entries."""
    context_length = entry.get("context_length")
    if not context_length or context_length <= 0:
        raise ValueError(f"missing context_length for {entry.get('id')}")
    architecture = entry.get("architecture") or {}
    top_provider = entry.get("top_provider") or {}
    metadata = {k: v for k, v in entry.items() if k not in _CONSUMED_KEYS}
    return ModelCard(
        id=entry["id"],
        canonical_slug=entry.get("canonical_slug"),
        provider="openrouter",
        display_name=entry.get("name") or entry["id"],
        description=entry.get("description"),
        pricing=_parse_pricing(entry.get("pricing") or {}),
        context_window=int(context_length),
        max_output_tokens=top_provider.get("max_completion_tokens"),
        input_modalities=_parse_modalities(architecture.get("input_modalities")),
        output_modalities=_parse_modalities(architecture.get("output_modalities")),
        supported_parameters=list(entry.get("supported_parameters") or []),
        deprecation_date=_parse_date(entry.get("expiration_date")),
        last_refreshed=refreshed_at,
        metadata=metadata,
    )


class OpenRouterAdapter(ModelProvider):
    """Fetches and caches the OpenRouter catalog (no auth needed)."""

    name = "openrouter"

    def __init__(
        self,
        url: str = DEFAULT_OPENROUTER_URL,
        ttl_seconds: float = 86_400.0,
        timeout: float = 30.0,
    ):
        self.url = url
        self.timeout = timeout
        self._cache: TTLCache[list[ModelCard]] = TTLCache(ttl_seconds)

    async def list_models(self, force: bool = False) -> list[ModelCard]:
        return await self._cache.get(self._fetch, force=force)

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus(
            ok=self._cache.ok,
            last_refreshed=self._cache.last_refreshed,
            detail=self._cache.last_error,
        )

    async def _fetch(self) -> list[ModelCard]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(self.url)
            response.raise_for_status()
            payload = response.json()
        refreshed_at = datetime.now(UTC)
        cards: list[ModelCard] = []
        for entry in payload.get("data", []):
            try:
                cards.append(parse_openrouter_model(entry, refreshed_at))
            except (ValueError, KeyError, TypeError) as exc:
                logger.info("skipping OpenRouter entry %s: %s", entry.get("id"), exc)
        logger.info("openrouter: parsed %d models", len(cards))
        return cards
