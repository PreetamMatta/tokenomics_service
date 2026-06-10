"""Pipeline: a named set of workload components estimated together."""

from pydantic import BaseModel, Field, model_validator

from .workloads import WorkloadComponent


class Pipeline(BaseModel):
    """A multi-component system (e.g. listener + executor + builder + memory).

    Populated by: API request bodies. Component roles must be unique so the
    per-component breakdown is unambiguous.
    """

    name: str = Field(min_length=1)
    components: list[WorkloadComponent] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_roles(self) -> "Pipeline":
        roles = [c.role for c in self.components]
        duplicates = {r for r in roles if roles.count(r) > 1}
        if duplicates:
            raise ValueError(f"pipeline component roles must be unique; duplicated: {duplicates}")
        return self
