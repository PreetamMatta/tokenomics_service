"""TokenomicsClient: typed HTTP client over the REST API (§5.5).

Wraps httpx; the same operations are available in-process via
tokenomics.service.ModelEconomicsService for library use without HTTP.
"""

from typing import Any

import httpx

from .errors import ModelNotFoundError, TokenomicsError
from .schemas.breakdown import CostBreakdown, PipelineCostBreakdown
from .schemas.enums import Capability
from .schemas.model_card import ModelCard
from .schemas.pipeline import Pipeline
from .schemas.schedule import Schedule
from .schemas.workloads import WorkloadComponent


class TokenomicsAPIError(TokenomicsError):
    """Non-2xx response from the service."""

    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self.body = body
        super().__init__(f"tokenomics API error {status_code}: {body}")


def _dump_schedule(schedule: Schedule | str) -> Any:
    return schedule if isinstance(schedule, str) else schedule.model_dump(mode="json")


class TokenomicsClient:
    """Synchronous typed client. Usable as a context manager."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def __enter__(self) -> "TokenomicsClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._client.request(method, path, **kwargs)
        if response.status_code >= 400:
            try:
                body = response.json()
            except ValueError:
                body = response.text
            if isinstance(body, dict) and body.get("error") == "model_not_found":
                raise ModelNotFoundError(body.get("model_id", ""), body.get("suggestions"))
            raise TokenomicsAPIError(response.status_code, body)
        return response.json()

    # -- health & meta ------------------------------------------------------

    def health(self) -> dict:
        return self._request("GET", "/health")

    def capabilities(self) -> dict:
        return self._request("GET", "/v1/capabilities")

    def schedule_presets(self) -> dict[str, Schedule]:
        data = self._request("GET", "/v1/schedules/presets")
        return {name: Schedule.model_validate(s) for name, s in data.items()}

    # -- catalog ------------------------------------------------------------

    def list_models(self, **filters: Any) -> list[ModelCard]:
        params = {k: v for k, v in filters.items() if v is not None}
        data = self._request("GET", "/v1/models", params=params)
        return [ModelCard.model_validate(m) for m in data["models"]]

    def get_model(self, model_id: str) -> ModelCard:
        return ModelCard.model_validate(self._request("GET", f"/v1/models/{model_id}"))

    def refresh(self) -> dict:
        return self._request("POST", "/v1/models/refresh")

    # -- selection helpers --------------------------------------------------

    def _query(
        self,
        endpoint: str,
        capability: Capability | str,
        min_tier: int | None,
        weights: str | None,
    ) -> dict:
        capability = capability.value if isinstance(capability, Capability) else capability
        params: dict[str, Any] = {"capability": capability}
        if min_tier is not None:
            params["min_tier"] = min_tier
        if weights is not None:
            params["weights"] = weights
        data = self._request("GET", f"/v1/models/queries/{endpoint}", params=params)
        if "model" in data:
            data["model"] = ModelCard.model_validate(data["model"])
        for entry in data.get("results", []):
            entry["model"] = ModelCard.model_validate(entry["model"])
        return data

    def cheapest(
        self, capability: Capability | str, min_tier: int = 3, weights: str | None = None
    ) -> dict:
        return self._query("cheapest", capability, min_tier, weights)

    def best(
        self, capability: Capability | str, min_tier: int = 1, weights: str | None = None
    ) -> dict:
        return self._query("best", capability, min_tier, weights)

    def best_value(
        self, capability: Capability | str, min_tier: int = 1, weights: str | None = None
    ) -> dict:
        return self._query("best-value", capability, min_tier, weights)

    # -- cost estimation ----------------------------------------------------

    def estimate_cost(
        self,
        workload: WorkloadComponent | dict,
        schedule: Schedule | str,
        model_id: str | None = None,
    ) -> CostBreakdown:
        body: dict[str, Any] = {
            "workload": workload if isinstance(workload, dict) else workload.model_dump(mode="json"),
            "schedule": _dump_schedule(schedule),
        }
        if model_id is not None:
            body["model_id"] = model_id
        return CostBreakdown.model_validate(self._request("POST", "/v1/estimates/cost", json=body))

    def estimate_pipeline(
        self, pipeline: Pipeline | dict, schedule: Schedule | str
    ) -> PipelineCostBreakdown:
        body = {
            "pipeline": pipeline if isinstance(pipeline, dict) else pipeline.model_dump(mode="json"),
            "schedule": _dump_schedule(schedule),
        }
        return PipelineCostBreakdown.model_validate(
            self._request("POST", "/v1/estimates/pipeline", json=body)
        )

    def compare(
        self,
        model_ids: list[str],
        workload: WorkloadComponent | dict,
        schedule: Schedule | str,
    ) -> dict:
        body = {
            "model_ids": model_ids,
            "workload": workload if isinstance(workload, dict) else workload.model_dump(mode="json"),
            "schedule": _dump_schedule(schedule),
        }
        data = self._request("POST", "/v1/estimates/compare", json=body)
        data["results"] = [CostBreakdown.model_validate(r) for r in data["results"]]
        return data
