"""Catalog registry: deterministic merge (§6.4) and selection queries (§5.3)."""

from .queries import best, best_value, blended_price, cheapest, filter_models
from .registry import ModelRegistry, load_aliases

__all__ = [
    "ModelRegistry",
    "best",
    "best_value",
    "blended_price",
    "cheapest",
    "filter_models",
    "load_aliases",
]
