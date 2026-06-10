"""Domain exceptions shared by the engine, service, and API layers."""


class TokenomicsError(Exception):
    """Base class for domain errors."""


class ModelNotFoundError(TokenomicsError):
    """Raised when a model id is not present in the catalog."""

    def __init__(self, model_id: str, suggestions: list[str] | None = None):
        self.model_id = model_id
        self.suggestions = suggestions or []
        super().__init__(f"model not found: {model_id}")


class NoMatchingModelError(TokenomicsError):
    """Raised when a selection query has no candidates."""

    def __init__(self, message: str, criteria: dict | None = None):
        self.criteria = criteria or {}
        super().__init__(message)


class UnknownSchedulePresetError(TokenomicsError):
    """Raised when a schedule preset name is not recognized."""

    def __init__(self, name: str, available: list[str]):
        self.name = name
        self.available = available
        super().__init__(f"unknown schedule preset: {name}")
