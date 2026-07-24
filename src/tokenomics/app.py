"""FastAPI factory and routes (§5). Routes are thin wrappers over the
ModelEconomicsService facade — all logic lives in the service layer.

Boot with: uvicorn --factory tokenomics.app:create_app
"""

from typing import Annotated, Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import __version__
from .errors import ModelNotFoundError, NoMatchingModelError, UnknownSchedulePresetError
from .schemas.breakdown import CostBreakdown, PipelineCostBreakdown
from .schemas.enums import Capability, Modality
from .schemas.model_card import ModelCard
from .schemas.pipeline import Pipeline
from .schemas.schedule import Schedule
from .schemas.workloads import WorkloadComponent
from .service import ModelEconomicsService, build_default_service
from .settings import Settings

# -- request bodies ---------------------------------------------------------


def _inject_model_id(data: Any, fallback_id: str | None) -> Any:
    """Copy a top-level model id into a workload dict that names no target,
    so the §3.7 'exactly one of model_id/mix' validator passes. The engine
    applies the override regardless."""
    if isinstance(data, dict):
        workload = data.get("workload")
        if (
            isinstance(workload, dict)
            and fallback_id
            and not workload.get("model_id")
            and not workload.get("mix")
        ):
            data = {**data, "workload": {**workload, "model_id": fallback_id}}
    return data


class CostEstimateRequest(BaseModel):
    """Body of POST /v1/estimates/cost. `model_id` overrides
    workload.model_id when both are given; `schedule` is an inline Schedule
    or a preset name."""

    model_config = ConfigDict(protected_namespaces=())

    workload: WorkloadComponent
    model_id: str | None = None
    schedule: Schedule | str

    @model_validator(mode="before")
    @classmethod
    def _allow_top_level_model_id(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return _inject_model_id(data, data.get("model_id"))
        return data


class PipelineEstimateRequest(BaseModel):
    pipeline: Pipeline
    schedule: Schedule | str


class CompareRequest(BaseModel):
    """Body of POST /v1/estimates/compare. The workload's own model target is
    ignored — every id in model_ids is estimated in its place."""

    model_config = ConfigDict(protected_namespaces=())

    model_ids: list[str] = Field(min_length=1)
    workload: WorkloadComponent
    schedule: Schedule | str

    @model_validator(mode="before")
    @classmethod
    def _allow_missing_workload_target(cls, data: Any) -> Any:
        if isinstance(data, dict):
            ids = data.get("model_ids")
            fallback = ids[0] if isinstance(ids, list) and ids else None
            return _inject_model_id(data, fallback)
        return data


# -- helpers ----------------------------------------------------------------


def _get_service(request: Request) -> ModelEconomicsService:
    return request.app.state.service


ServiceDep = Annotated[ModelEconomicsService, Depends(_get_service)]


def _parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_weights(value: str | None) -> dict[str, float] | None:
    """Parse `weights=input=0.6,output=0.4` (also accepts ':' separators)."""
    if not value:
        return None
    weights: dict[str, float] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        for sep in ("=", ":"):
            if sep in part:
                key, _, raw = part.partition(sep)
                weights[key.strip()] = float(raw)
                break
    return weights or None


# -- app factory ------------------------------------------------------------


def create_app(
    service: ModelEconomicsService | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    app = FastAPI(
        title="tokenomics",
        version=__version__,
        description="Model Economics Microservice: catalog, selection helpers, cost estimation.",
    )
    app.state.service = service or build_default_service(settings)

    @app.exception_handler(ModelNotFoundError)
    async def _model_not_found(request: Request, exc: ModelNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "error": "model_not_found",
                "model_id": exc.model_id,
                "suggestions": exc.suggestions,
            },
        )

    @app.exception_handler(NoMatchingModelError)
    async def _no_match(request: Request, exc: NoMatchingModelError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"error": "no_matching_model", "message": str(exc), "criteria": exc.criteria},
        )

    @app.exception_handler(UnknownSchedulePresetError)
    async def _bad_preset(request: Request, exc: UnknownSchedulePresetError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "unknown_schedule_preset",
                "name": exc.name,
                "available": exc.available,
            },
        )

    # -- health & meta ------------------------------------------------------

    @app.get("/")
    async def root() -> dict[str, str]:
        return {
            "name": "tokenomics",
            "version": __version__,
            "docs_url": "/docs",
            "health_url": "/health",
        }

    @app.get("/health")
    async def health(service: ServiceDep) -> dict:
        return service.health()

    @app.get("/v1/capabilities")
    async def capabilities(service: ServiceDep) -> dict:
        return service.capabilities_info()

    @app.get("/v1/schedules/presets")
    async def schedule_presets(service: ServiceDep) -> dict[str, Schedule]:
        return service.schedule_presets()

    # -- selection helpers (declared before the catch-all model_id route) ---

    @app.get("/v1/models/queries/cheapest")
    async def query_cheapest(
        service: ServiceDep,
        capability: Capability,
        min_tier: Annotated[int, Query(ge=1, le=5)] = 3,
        weights: str | None = None,
    ) -> dict:
        return await service.cheapest(capability, min_tier, _parse_weights(weights))

    @app.get("/v1/models/queries/best")
    async def query_best(
        service: ServiceDep,
        capability: Capability,
        min_tier: Annotated[int, Query(ge=1, le=5)] = 1,
        weights: str | None = None,
    ) -> dict:
        return await service.best(capability, min_tier, _parse_weights(weights))

    @app.get("/v1/models/queries/best-value")
    async def query_best_value(
        service: ServiceDep,
        capability: Capability,
        min_tier: Annotated[int, Query(ge=1, le=5)] = 1,
        weights: str | None = None,
    ) -> dict:
        return await service.best_value(capability, min_tier, _parse_weights(weights))

    # -- model catalog ------------------------------------------------------

    @app.post("/v1/models/refresh")
    async def refresh_models(service: ServiceDep) -> dict:
        return {"providers": await service.refresh()}

    @app.get("/v1/models")
    async def list_models(
        service: ServiceDep,
        capability: Capability | None = None,
        min_tier: Annotated[int | None, Query(ge=1, le=5)] = None,
        max_input_price: Annotated[float | None, Query(ge=0)] = None,
        max_output_price: Annotated[float | None, Query(ge=0)] = None,
        min_context_window: Annotated[int | None, Query(gt=0)] = None,
        modalities: str | None = None,
        tags: str | None = None,
        supported_parameters: str | None = None,
        include_deprecated: bool = False,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict:
        modality_list = (
            [Modality(m) for m in _parse_csv(modalities) or []] if modalities else None
        )
        models = await service.list_models(
            capability=capability,
            min_tier=min_tier,
            max_input_price=max_input_price,
            max_output_price=max_output_price,
            min_context_window=min_context_window,
            modalities=modality_list,
            tags=_parse_csv(tags),
            supported_parameters=_parse_csv(supported_parameters),
            include_deprecated=include_deprecated,
        )
        models.sort(key=lambda c: c.id)
        return {
            "total": len(models),
            "limit": limit,
            "offset": offset,
            "models": models[offset : offset + limit],
        }

    @app.get("/v1/models/{model_id:path}")
    async def get_model(service: ServiceDep, model_id: str) -> ModelCard:
        return await service.get_model(model_id)

    # -- cost estimation ----------------------------------------------------

    @app.post("/v1/estimates/cost")
    async def estimate_cost(service: ServiceDep, body: CostEstimateRequest) -> CostBreakdown:
        return await service.estimate_cost(body.workload, body.schedule, model_id=body.model_id)

    @app.post("/v1/estimates/pipeline")
    async def estimate_pipeline(
        service: ServiceDep, body: PipelineEstimateRequest
    ) -> PipelineCostBreakdown:
        return await service.estimate_pipeline(body.pipeline, body.schedule)

    @app.post("/v1/estimates/compare")
    async def estimate_compare(service: ServiceDep, body: CompareRequest) -> dict:
        return await service.compare(body.model_ids, body.workload, body.schedule)

    return app
