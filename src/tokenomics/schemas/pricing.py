"""ModelPricing: all price dimensions for a model.

Populated by: OpenRouterAdapter (parsed from the /api/v1/models pricing
object, per-token strings converted to per-million floats) or local model
YAML files. Units: token prices are USD per MILLION tokens; unit prices
(request, image, web search) are USD per unit; cache storage is USD per
million tokens per hour.
"""

from pydantic import BaseModel, Field, computed_field


class ModelPricing(BaseModel):
    """All token prices in USD per MILLION tokens. Unit prices in USD per unit.

    Raw fields come straight from the provider (None when the provider does
    not publish that dimension). The `effective_*` computed fields apply the
    documented fallback defaults and are what the cost engine consumes:

    - effective_reasoning: reasoning price, defaulting to the output price
      (reasoning tokens are billed at the output rate by every major provider).
    - effective_cache_read: cache-read price, falling back to 10% of input as
      a LAST RESORT (Anthropic/DeepSeek-style). Cost breakdowns flag this
      fallback in `estimated_fields`.
    - effective_cache_write: cache-write price, defaulting to the input price
      (i.e. no write premium).
    """

    input_per_million: float = Field(ge=0)
    output_per_million: float = Field(ge=0)
    reasoning_per_million: float | None = Field(default=None, ge=0)  # default: output price
    cache_read_per_million: float | None = Field(default=None, ge=0)  # fallback 0.1*input
    cache_write_per_million: float | None = Field(default=None, ge=0)  # default: input price
    cache_storage_per_million_hour: float | None = Field(default=None, ge=0)  # Google-style fee
    request_fixed: float = Field(default=0.0, ge=0)  # fixed USD per request
    image_per_unit: float | None = Field(default=None, ge=0)  # USD per image input
    web_search_per_unit: float | None = Field(default=None, ge=0)
    pricing_estimated: bool = False  # True when any fallback default was applied upstream

    @computed_field
    @property
    def effective_reasoning(self) -> float:
        """Reasoning price actually used: provider value or output price."""
        if self.reasoning_per_million is not None:
            return self.reasoning_per_million
        return self.output_per_million

    @computed_field
    @property
    def effective_cache_read(self) -> float:
        """Cache-read price actually used: provider value or 0.1 * input (estimated)."""
        if self.cache_read_per_million is not None:
            return self.cache_read_per_million
        return 0.1 * self.input_per_million

    @computed_field
    @property
    def effective_cache_write(self) -> float:
        """Cache-write price actually used: provider value or input price (no premium)."""
        if self.cache_write_per_million is not None:
            return self.cache_write_per_million
        return self.input_per_million
