"""Cold start waiting / warmup helpers.

This module provides helpers for handling model cold starts by automatically
triggering warmup and polling until the model is ready before making requests.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from kafeido.types.models import ModelStatus, WarmupResponse


# Default configuration
DEFAULT_POLL_INTERVAL = 2.0  # seconds between status checks
DEFAULT_MAX_WAIT_TIME = 300.0  # 5 minutes max wait
HEALTHY_STATUS = "healthy"


class WarmupTimeoutError(Exception):
    """Raised when model warmup times out.

    Attributes:
        model: The model ID that timed out.
        waited_seconds: How long we waited before timing out.
    """

    def __init__(self, model: str, waited_seconds: float) -> None:
        super().__init__(
            f"Model '{model}' did not become ready within {waited_seconds:.1f}s"
        )
        self.model = model
        self.waited_seconds = waited_seconds


class WarmupHelper:
    """Synchronous warmup helper for cold start waiting.

    This helper triggers model warmup and polls until the model becomes healthy.
    """

    def __init__(
        self,
        status_fn: Callable[[str], "ModelStatus"],
        warmup_fn: Callable[[str], "WarmupResponse"],
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_wait_time: float = DEFAULT_MAX_WAIT_TIME,
    ) -> None:
        """Initialize warmup helper.

        Args:
            status_fn: Function to get model status (typically models.status).
            warmup_fn: Function to trigger warmup (typically models.warmup).
            poll_interval: Seconds between status checks.
            max_wait_time: Maximum seconds to wait before timeout.
        """
        self._status_fn = status_fn
        self._warmup_fn = warmup_fn
        self._poll_interval = poll_interval
        self._max_wait_time = max_wait_time

    def wait_for_ready(
        self, model: str, timeout: Optional[float] = None
    ) -> None:
        """Wait for model to be ready, triggering warmup if needed.

        This method will:
        1. Trigger a warmup request to start loading the model
        2. If model is already warm, return immediately
        3. Otherwise, poll the status endpoint until the model is healthy
        4. Raise WarmupTimeoutError if the model doesn't become ready in time

        Args:
            model: The model ID to wait for.
            timeout: Optional timeout override in seconds. If None, uses
                the default max_wait_time from initialization.

        Raises:
            WarmupTimeoutError: If model doesn't become ready within timeout.
        """
        max_wait = timeout if timeout is not None else self._max_wait_time

        # First, trigger warmup
        warmup_response = self._warmup_fn(model)

        if warmup_response.already_warm:
            return  # Model is already ready

        # Poll until ready or timeout
        start_time = time.monotonic()

        while True:
            elapsed = time.monotonic() - start_time

            if elapsed >= max_wait:
                raise WarmupTimeoutError(model, elapsed)

            # Check status
            status = self._status_fn(model)

            if status.status and status.status.status == HEALTHY_STATUS:
                return  # Model is ready

            # Wait before next poll
            time.sleep(self._poll_interval)


class AsyncWarmupHelper:
    """Asynchronous warmup helper for cold start waiting.

    This helper triggers model warmup and polls until the model becomes healthy,
    using async/await for non-blocking operation.
    """

    def __init__(
        self,
        status_fn: Callable[[str], Awaitable["ModelStatus"]],
        warmup_fn: Callable[[str], Awaitable["WarmupResponse"]],
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_wait_time: float = DEFAULT_MAX_WAIT_TIME,
    ) -> None:
        """Initialize async warmup helper.

        Args:
            status_fn: Async function to get model status.
            warmup_fn: Async function to trigger warmup.
            poll_interval: Seconds between status checks.
            max_wait_time: Maximum seconds to wait before timeout.
        """
        self._status_fn = status_fn
        self._warmup_fn = warmup_fn
        self._poll_interval = poll_interval
        self._max_wait_time = max_wait_time

    async def wait_for_ready(
        self, model: str, timeout: Optional[float] = None
    ) -> None:
        """Wait for model to be ready asynchronously.

        This method will:
        1. Trigger a warmup request to start loading the model
        2. If model is already warm, return immediately
        3. Otherwise, poll the status endpoint until the model is healthy
        4. Raise WarmupTimeoutError if the model doesn't become ready in time

        Args:
            model: The model ID to wait for.
            timeout: Optional timeout override in seconds. If None, uses
                the default max_wait_time from initialization.

        Raises:
            WarmupTimeoutError: If model doesn't become ready within timeout.
        """
        max_wait = timeout if timeout is not None else self._max_wait_time

        # First, trigger warmup
        warmup_response = await self._warmup_fn(model)

        if warmup_response.already_warm:
            return  # Model is already ready

        # Poll until ready or timeout
        start_time = time.monotonic()

        while True:
            elapsed = time.monotonic() - start_time

            if elapsed >= max_wait:
                raise WarmupTimeoutError(model, elapsed)

            # Check status
            status = await self._status_fn(model)

            if status.status and status.status.status == HEALTHY_STATUS:
                return  # Model is ready

            # Wait before next poll
            await asyncio.sleep(self._poll_interval)
