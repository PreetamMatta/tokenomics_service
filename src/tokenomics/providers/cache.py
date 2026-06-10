"""In-memory TTL cache used in front of every provider HTTP call.

No database in v1: this is the only state in the service. A failed refresh
serves the previous (stale) value when one exists, so transient provider
outages degrade gracefully instead of failing requests.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TTLCache(Generic[T]):
    """Single-value async TTL cache with stampede protection."""

    def __init__(self, ttl_seconds: float):
        self.ttl_seconds = ttl_seconds
        self._value: T | None = None
        self._fetched_at: datetime | None = None
        self._last_error: str | None = None
        self._lock = asyncio.Lock()

    @property
    def last_refreshed(self) -> datetime | None:
        return self._fetched_at

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def ok(self) -> bool:
        """True when the most recent fetch attempt (if any) succeeded."""
        return self._last_error is None

    async def get(self, fetcher: Callable[[], Awaitable[T]], force: bool = False) -> T:
        """Return the cached value, refreshing when stale, missing, or forced."""
        async with self._lock:
            if not force and self._value is not None and self._fetched_at is not None:
                age = (datetime.now(UTC) - self._fetched_at).total_seconds()
                if age < self.ttl_seconds:
                    return self._value
            try:
                value = await fetcher()
            except Exception as exc:
                self._last_error = str(exc)
                if self._value is not None:
                    logger.warning("refresh failed, serving stale value: %s", exc)
                    return self._value
                raise
            self._value = value
            self._fetched_at = datetime.now(UTC)
            self._last_error = None
            return value
