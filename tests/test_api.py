"""Route contracts, 404/422 shapes, refresh (§5). Uses a stubbed provider
stack — no live HTTP ever."""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_card
from tokenomics.app import create_app
from tokenomics.benchmarks.local import LocalBenchmarkProvider
from tokenomics.benchmarks.tiers import TierDeriver
from tokenomics.providers.base import ModelProvider, ProviderStatus
from tokenomics.registry.registry import ModelRegistry
from tokenomics.schemas.enums import Capability
from tokenomics.service import ModelEconomicsService


class StaticProvider(ModelProvider):
    name = "static"

    def __init__(self, cards):
        self.cards = cards

    async def list_models(self, force: bool = False):
        return self.cards

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus(ok=True)


@pytest.fixture
def client(tmp_path) -> TestClient:
    cards = [
        make_card(
            "anthropic/claude-sonnet-4.6",
            3.0,
            15.0,
            {Capability.CODING: 5, Capability.AGENTIC: 4},
        ),
        make_card("openai/gpt-5.2-mini", 0.25, 2.0, {Capability.CODING: 3}),
        make_card("deepseek/deepseek-r1", 0.55, 2.19, {Capability.REASONING: 4}),
    ]
    registry = ModelRegistry(
        providers=[StaticProvider(cards)],
        quality_provider=None,
        tier_deriver=TierDeriver(),
        aliases={},
        overrides_provider=LocalBenchmarkProvider(tmp_path / "overrides.yaml"),
    )
    app = create_app(service=ModelEconomicsService(registry))
    return TestClient(app)


WORKLOAD = {
    "type": "continuous_stream",
    "role": "listener",
    "input_tokens_per_hour": 2_000_000,
    "output_tokens_per_hour": 20_000,
    "cache_hit_ratio": 0.85,
}


class TestHealthAndMeta:
    def test_health(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["version"]
        assert body["providers"]["static"]["ok"] is True

    def test_capabilities(self, client):
        body = client.get("/v1/capabilities").json()
        assert "coding" in body["capabilities"]
        assert body["tier_semantics"]["5"] == "top-of-class"
        assert "percentile_thresholds" in body["derivation"]

    def test_schedule_presets(self, client):
        body = client.get("/v1/schedules/presets").json()
        assert set(body) == {"light_4x6", "standard_7x5", "heavy_15x7", "always_on_24x7"}
        assert body["standard_7x5"]["hours_per_month"] == pytest.approx(152.075)

    def test_openapi_renders(self, client):
        spec = client.get("/openapi.json").json()
        schemas = spec["components"]["schemas"]
        for name in ("ModelCard", "ModelPricing", "CostBreakdown", "Range", "Schedule"):
            # models with computed fields render as separate -Input/-Output schemas
            assert any(key == name or key.startswith(f"{name}-") for key in schemas), name


class TestCatalogRoutes:
    def test_list_models(self, client):
        body = client.get("/v1/models").json()
        assert body["total"] == 3
        assert len(body["models"]) == 3

    def test_filters(self, client):
        body = client.get("/v1/models", params={"capability": "coding", "min_tier": 4}).json()
        assert [m["id"] for m in body["models"]] == ["anthropic/claude-sonnet-4.6"]

    def test_pagination(self, client):
        body = client.get("/v1/models", params={"limit": 1, "offset": 1}).json()
        assert body["total"] == 3
        assert len(body["models"]) == 1

    def test_get_model_with_slash_id(self, client):
        body = client.get("/v1/models/anthropic/claude-sonnet-4.6").json()
        assert body["id"] == "anthropic/claude-sonnet-4.6"
        assert body["pricing"]["input_per_million"] == 3.0

    def test_unknown_model_404_shape(self, client):
        response = client.get("/v1/models/anthropic/claude-sonnet-9.9")
        assert response.status_code == 404
        body = response.json()
        assert body["error"] == "model_not_found"
        assert body["model_id"] == "anthropic/claude-sonnet-9.9"
        assert "anthropic/claude-sonnet-4.6" in body["suggestions"]

    def test_refresh(self, client):
        body = client.post("/v1/models/refresh").json()
        assert body["providers"]["static"]["ok"] is True
        assert body["providers"]["static"]["models"] == 3


class TestQueryRoutes:
    def test_cheapest(self, client):
        body = client.get(
            "/v1/models/queries/cheapest", params={"capability": "coding"}
        ).json()
        assert body["model"]["id"] == "openai/gpt-5.2-mini"
        assert body["justification"]["query"] == "cheapest"

    def test_best(self, client):
        body = client.get("/v1/models/queries/best", params={"capability": "coding"}).json()
        assert body["model"]["id"] == "anthropic/claude-sonnet-4.6"
        assert body["justification"]["tier"] == 5

    def test_best_value(self, client):
        body = client.get(
            "/v1/models/queries/best-value", params={"capability": "coding"}
        ).json()
        assert body["results"]
        assert "justification" in body

    def test_no_match_404(self, client):
        response = client.get(
            "/v1/models/queries/cheapest", params={"capability": "ocr", "min_tier": 5}
        )
        assert response.status_code == 404
        assert response.json()["error"] == "no_matching_model"

    def test_capability_required(self, client):
        assert client.get("/v1/models/queries/cheapest").status_code == 422


class TestEstimateRoutes:
    def test_cost_with_preset_schedule(self, client):
        # §8 anchor, but with the 10%-of-input cache-read fallback (0.30):
        response = client.post(
            "/v1/estimates/cost",
            json={
                "workload": {**WORKLOAD, "model_id": "anthropic/claude-sonnet-4.6"},
                "schedule": "standard_7x5",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert abs(body["total_cost"] - 260.05) <= 0.01
        assert body["schedule_name"] == "standard_7x5"
        assert "cache_read_per_million" in body["estimated_fields"]

    def test_top_level_model_id_overrides(self, client):
        response = client.post(
            "/v1/estimates/cost",
            json={
                "workload": WORKLOAD,  # no model target at all
                "model_id": "openai/gpt-5.2-mini",
                "schedule": "standard_7x5",
            },
        )
        assert response.status_code == 200
        assert response.json()["model_id"] == "openai/gpt-5.2-mini"

    def test_inline_schedule_and_range_response(self, client):
        response = client.post(
            "/v1/estimates/cost",
            json={
                "workload": {
                    "type": "build_action",
                    "role": "builder",
                    "model_id": "anthropic/claude-sonnet-4.6",
                    "tasks_per_period": 3,
                    "period": "week",
                    "iterations_per_task": {"min": 4, "typical": 12, "max": 40},
                    "input_tokens_per_iteration": 60_000,
                    "output_tokens_per_iteration": 4_000,
                    "reasoning_tokens_per_iteration": 2_000,
                    "subagent_spawn_probability": 0.3,
                },
                "schedule": {"name": "custom", "hours_per_day": 8, "days_per_week": 5},
            },
        )
        assert response.status_code == 200
        total = response.json()["total_cost"]
        assert set(total) == {"min", "typical", "max"}
        assert total["min"] < total["typical"] < total["max"]

    def test_unknown_preset_422(self, client):
        response = client.post(
            "/v1/estimates/cost",
            json={
                "workload": {**WORKLOAD, "model_id": "anthropic/claude-sonnet-4.6"},
                "schedule": "nonexistent",
            },
        )
        assert response.status_code == 422
        assert response.json()["error"] == "unknown_schedule_preset"

    def test_invalid_workload_422_pydantic_detail(self, client):
        response = client.post(
            "/v1/estimates/cost",
            json={
                "workload": {
                    **WORKLOAD,
                    "model_id": "anthropic/claude-sonnet-4.6",
                    "input_tokens_per_hour": -5,
                },
                "schedule": "standard_7x5",
            },
        )
        assert response.status_code == 422
        assert isinstance(response.json()["detail"], list)

    def test_unknown_model_404_with_suggestions(self, client):
        response = client.post(
            "/v1/estimates/cost",
            json={
                "workload": {**WORKLOAD, "model_id": "anthropic/claude-sonnet-9.9"},
                "schedule": "standard_7x5",
            },
        )
        assert response.status_code == 404
        body = response.json()
        assert body["error"] == "model_not_found"
        assert body["suggestions"]

    def test_pipeline(self, client):
        response = client.post(
            "/v1/estimates/pipeline",
            json={
                "pipeline": {
                    "name": "demo",
                    "components": [
                        {**WORKLOAD, "model_id": "openai/gpt-5.2-mini"},
                        {
                            "type": "recurring_job",
                            "role": "memory",
                            "model_id": "openai/gpt-5.2-mini",
                            "frequency": "daily",
                            "input_tokens_per_invocation": 200_000,
                            "output_tokens_per_invocation": 5_000,
                        },
                    ],
                },
                "schedule": "standard_7x5",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["pipeline_name"] == "demo"
        assert len(body["per_component"]) == 2
        assert body["total_cost"] == pytest.approx(
            sum(c["total_cost"] for c in body["per_component"])
        )

    def test_duplicate_pipeline_roles_422(self, client):
        response = client.post(
            "/v1/estimates/pipeline",
            json={
                "pipeline": {
                    "name": "demo",
                    "components": [
                        {**WORKLOAD, "model_id": "openai/gpt-5.2-mini"},
                        {**WORKLOAD, "model_id": "deepseek/deepseek-r1"},
                    ],
                },
                "schedule": "standard_7x5",
            },
        )
        assert response.status_code == 422

    def test_compare(self, client):
        response = client.post(
            "/v1/estimates/compare",
            json={
                "model_ids": ["anthropic/claude-sonnet-4.6", "openai/gpt-5.2-mini"],
                "workload": WORKLOAD,  # no model target — compare supplies them
                "schedule": "standard_7x5",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["results"]) == 2
        assert body["ranked_by_total"][0] == "openai/gpt-5.2-mini"  # cheaper first
