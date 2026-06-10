"""Selection helpers (§5.3): information, not routing.

Every helper returns a `justification` block describing which fields drove
the answer (tier, blended price, benchmark evidence cited) so the consuming
agent can make its own decision.

Blended price: weighted input/output per-million price. Default weights are
input 0.75 / output 0.25 (~3:1 input-heavy traffic); weights are normalized
before use.
"""

from datetime import date
from statistics import mean
from typing import Any

from ..schemas.enums import Capability, Modality
from ..schemas.model_card import ModelCard
from ..schemas.pricing import ModelPricing

DEFAULT_BLEND_WEIGHTS = {"input": 0.75, "output": 0.25}


def blended_price(pricing: ModelPricing, weights: dict[str, float] | None = None) -> float:
    """Weighted input/output price in USD per million tokens."""
    weights = weights or DEFAULT_BLEND_WEIGHTS
    w_in = weights.get("input", DEFAULT_BLEND_WEIGHTS["input"])
    w_out = weights.get("output", DEFAULT_BLEND_WEIGHTS["output"])
    total = w_in + w_out
    if total <= 0:
        w_in, w_out = DEFAULT_BLEND_WEIGHTS["input"], DEFAULT_BLEND_WEIGHTS["output"]
        total = 1.0
    return (w_in * pricing.input_per_million + w_out * pricing.output_per_million) / total


def filter_models(
    cards: list[ModelCard],
    capability: Capability | None = None,
    min_tier: int | None = None,
    max_input_price: float | None = None,
    max_output_price: float | None = None,
    min_context_window: int | None = None,
    modalities: list[Modality] | None = None,
    tags: list[str] | None = None,
    supported_parameters: list[str] | None = None,
    include_deprecated: bool = False,
    today: date | None = None,
) -> list[ModelCard]:
    """Catalog filter behind GET /v1/models. All criteria are AND-ed."""
    today = today or date.today()
    out: list[ModelCard] = []
    for card in cards:
        if not include_deprecated and card.is_deprecated(today):
            continue
        if capability is not None:
            tier = card.capabilities.get(capability, 0)
            if tier < (min_tier if min_tier is not None else 1):
                continue
        if max_input_price is not None and card.pricing.input_per_million > max_input_price:
            continue
        if max_output_price is not None and card.pricing.output_per_million > max_output_price:
            continue
        if min_context_window is not None and card.context_window < min_context_window:
            continue
        if modalities and not all(m in card.input_modalities for m in modalities):
            continue
        if tags and not all(t in card.tags for t in tags):
            continue
        if supported_parameters and not all(
            p in card.supported_parameters for p in supported_parameters
        ):
            continue
        out.append(card)
    return out


def _evidence_score(card: ModelCard) -> float:
    """Benchmark evidence used for tie-breaks: mean of quality indices when
    present, else mean of benchmark scores, else 0 (no evidence)."""
    if card.quality_indices:
        return mean(card.quality_indices.values())
    if card.benchmark_scores:
        return mean(card.benchmark_scores.values())
    return 0.0


def _evidence_cited(card: ModelCard, capability: Capability) -> dict[str, Any]:
    cited: dict[str, Any] = {"tier": card.capabilities.get(capability)}
    if card.quality_indices:
        cited["quality_indices"] = card.quality_indices
    if card.benchmark_scores:
        cited["benchmark_scores"] = card.benchmark_scores
    return cited


def _candidates(
    cards: list[ModelCard], capability: Capability, min_tier: int
) -> list[ModelCard]:
    return filter_models(cards, capability=capability, min_tier=min_tier)


def cheapest(
    cards: list[ModelCard],
    capability: Capability,
    min_tier: int = 3,
    weights: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """Cheapest (by blended price) model meeting the capability/tier bar."""
    candidates = _candidates(cards, capability, min_tier)
    if not candidates:
        return None
    winner = min(candidates, key=lambda c: (blended_price(c.pricing, weights), c.id))
    price = blended_price(winner.pricing, weights)
    return {
        "model": winner,
        "justification": {
            "query": "cheapest",
            "capability": capability.value,
            "min_tier": min_tier,
            "weights": weights or DEFAULT_BLEND_WEIGHTS,
            "candidates_considered": len(candidates),
            "selected": winner.id,
            "blended_price_per_million_usd": price,
            "evidence": _evidence_cited(winner, capability),
            "reason": (
                f"lowest blended price (${price:.4f}/M) among {len(candidates)} models with "
                f"{capability.value} tier >= {min_tier}"
            ),
        },
    }


def best(
    cards: list[ModelCard],
    capability: Capability,
    min_tier: int = 1,
    weights: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """Highest tier; ties broken by benchmark evidence, then by lower price."""
    candidates = _candidates(cards, capability, min_tier)
    if not candidates:
        return None
    winner = min(
        candidates,
        key=lambda c: (
            -c.capabilities.get(capability, 0),
            -_evidence_score(c),
            blended_price(c.pricing, weights),
            c.id,
        ),
    )
    return {
        "model": winner,
        "justification": {
            "query": "best",
            "capability": capability.value,
            "min_tier": min_tier,
            "candidates_considered": len(candidates),
            "selected": winner.id,
            "tier": winner.capabilities.get(capability),
            "tie_breakers": ["benchmark_evidence", "blended_price"],
            "evidence": _evidence_cited(winner, capability),
            "reason": (
                f"highest {capability.value} tier "
                f"({winner.capabilities.get(capability)}); ties broken by benchmark "
                f"evidence then price"
            ),
        },
    }


def best_value(
    cards: list[ModelCard],
    capability: Capability,
    min_tier: int = 1,
    weights: dict[str, float] | None = None,
    top_n: int = 5,
) -> dict[str, Any] | None:
    """Maximize tier per blended dollar; ranked top-N with scores."""
    candidates = _candidates(cards, capability, min_tier)
    if not candidates:
        return None
    scored = []
    for card in candidates:
        price = blended_price(card.pricing, weights)
        tier = card.capabilities.get(capability, 0)
        # Free models would divide by zero; floor the price at $0.001/M.
        score = tier / max(price, 0.001)
        scored.append(
            {
                "model": card,
                "value_score": score,
                "tier": tier,
                "blended_price_per_million_usd": price,
            }
        )
    scored.sort(key=lambda s: (-s["value_score"], s["model"].id))
    top = scored[:top_n]
    return {
        "results": top,
        "justification": {
            "query": "best-value",
            "capability": capability.value,
            "min_tier": min_tier,
            "weights": weights or DEFAULT_BLEND_WEIGHTS,
            "candidates_considered": len(candidates),
            "scoring": "value_score = tier / max(blended_price_per_million, 0.001)",
            "reason": f"top {len(top)} models by {capability.value} tier per blended dollar",
        },
    }
