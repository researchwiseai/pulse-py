"""Async retry utilities for HTTP requests with enhanced exponential backoff."""

from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, Iterable, Optional, Union
from dataclasses import dataclass

import httpx
from pulse.core.async_error_handling import AsyncTimeoutError, AsyncCancellationError

DEFAULT_RETRY_STATUSES = {429, 500, 502, 503, 504}
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF = 0.5
DEFAULT_MAX_BACKOFF = 60.0
DEFAULT_JITTER = True


@dataclass
class AsyncRetryConfig:
    """Configuration for async retry behavior with exponential backoff."""

    retry_statuses: Iterable[int] = None
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    base_backoff: float = DEFAULT_BACKOFF
    max_backoff: float = DEFAULT_MAX_BACKOFF
    exponential_base: float = 2.0
    jitter: bool = DEFAULT_JITTER
    jitter_range: float = 0.1

    def __post_init__(self):
        if self.retry_statuses is None:
            self.retry_statuses = DEFAULT_RETRY_STATUSES


async def async_retry_request(
    func: Callable[[], Awaitable[httpx.Response]],
    *,
    retry_statuses: Iterable[int] = DEFAULT_RETRY_STATUSES,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF,
) -> httpx.Response:
    """Execute async ``func`` retrying on network or transient server errors.

    This is the legacy interface maintained for backward compatibility.
    For enhanced retry behavior, use async_retry_request_enhanced.
    """
    attempt = 0
    delay = backoff
    last_exc: Optional[BaseException] = None

    while True:
        try:
            response = await func()
            if response.status_code not in retry_statuses:
                return response
            # Successful request (even if it's a retryable status); clear last exception
            last_exc = None
        except httpx.TransportError as exc:
            response = None
            last_exc = exc

        attempt += 1
        if attempt >= max_attempts:
            if response is not None:
                return response
            # Re-raise the last transport error if available; otherwise raise a
            # descriptive RuntimeError so callers get a meaningful failure.
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(
                "Request failed after retries with no response or exception captured"
            )

        await asyncio.sleep(delay)
        delay *= 2


async def async_retry_request_enhanced(
    func: Callable[[], Awaitable[httpx.Response]],
    config: Optional[AsyncRetryConfig] = None,
) -> httpx.Response:
    """Execute async func with enhanced exponential backoff and jitter.

    This enhanced version provides:
    - Configurable exponential backoff with maximum delay cap
    - Optional jitter to prevent thundering herd problems
    - Better error handling and logging
    - Rate limit detection and adaptive delays

    Args:
        func: Async function that returns an httpx.Response
        config: Retry configuration. Uses defaults if None.

    Returns:
        httpx.Response from successful request

    Raises:
        httpx.TransportError: If all retries fail due to transport errors
        RuntimeError: If retries are exhausted without clear error
    """
    if config is None:
        config = AsyncRetryConfig()

    attempt = 0
    last_exc: Optional[BaseException] = None
    last_response: Optional[httpx.Response] = None

    while True:
        try:
            response = await func()

            # Check if response indicates success
            if response.status_code not in config.retry_statuses:
                return response

            # Store response for potential return if retries exhausted
            last_response = response
            last_exc = None

        except (httpx.TransportError, asyncio.TimeoutError) as exc:
            last_response = None
            last_exc = exc
        except asyncio.CancelledError as exc:
            # Handle cancellation separately - don't retry on cancellation
            raise AsyncCancellationError(
                f"Request was cancelled during attempt {attempt + 1}",
                operation="async_retry_request",
                context={"attempt": attempt + 1, "max_attempts": config.max_attempts},
            ) from exc

        attempt += 1
        if attempt >= config.max_attempts:
            # Return last response if available, otherwise raise enhanced exception
            if last_response is not None:
                return last_response
            if last_exc is not None:
                if isinstance(last_exc, asyncio.TimeoutError):
                    raise AsyncTimeoutError(
                        f"Request timed out after {config.max_attempts} attempts",
                        operation="async_retry_request",
                        timeout=0.0,  # Individual timeout not tracked here
                        context={"max_attempts": config.max_attempts},
                    ) from last_exc
                else:
                    raise last_exc
            raise RuntimeError(
                f"Request failed after {config.max_attempts} attempts "
                "with no response or exception captured"
            )

        # Calculate delay with exponential backoff
        delay = min(
            config.base_backoff * (config.exponential_base ** (attempt - 1)),
            config.max_backoff,
        )

        # Add jitter to prevent thundering herd
        if config.jitter:
            jitter_amount = delay * config.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.1, delay)  # Ensure minimum delay

        # Special handling for rate limiting (429)
        if last_response and last_response.status_code == 429:
            # Check for Retry-After header
            retry_after = last_response.headers.get("Retry-After")
            if retry_after:
                try:
                    retry_delay = float(retry_after)
                    delay = min(retry_delay, config.max_backoff)
                except ValueError:
                    pass  # Use calculated delay if header is invalid

        await asyncio.sleep(delay)


class AsyncRetryManager:
    """Manager for async retry operations with batch processing support."""

    def __init__(self, config: Optional[AsyncRetryConfig] = None):
        """Initialize retry manager with configuration.

        Args:
            config: Retry configuration. Uses defaults if None.
        """
        self.config = config or AsyncRetryConfig()
        self._retry_stats = {
            "total_requests": 0,
            "total_retries": 0,
            "failed_requests": 0,
            "rate_limited_requests": 0,
        }

    async def retry_request(
        self, func: Callable[[], Awaitable[httpx.Response]]
    ) -> httpx.Response:
        """Retry a single request with configured behavior.

        Args:
            func: Async function that returns an httpx.Response

        Returns:
            httpx.Response from successful request
        """
        self._retry_stats["total_requests"] += 1

        try:
            return await async_retry_request_enhanced(func, self.config)
        except Exception:
            self._retry_stats["failed_requests"] += 1
            raise

    async def retry_batch_requests(
        self,
        funcs: list[Callable[[], Awaitable[httpx.Response]]],
        max_concurrent: int = 5,
        fail_fast: bool = False,
    ) -> list[Union[httpx.Response, Exception]]:
        """Retry multiple requests concurrently with controlled concurrency.

        Args:
            funcs: List of async functions that return httpx.Response
            max_concurrent: Maximum concurrent requests
            fail_fast: If True, stop on first failure

        Returns:
            List of responses or exceptions in same order as input
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def retry_with_semaphore(func):
            async with semaphore:
                return await self.retry_request(func)

        if fail_fast:
            # Use gather without return_exceptions to fail fast
            return await asyncio.gather(*[retry_with_semaphore(func) for func in funcs])
        else:
            # Use gather with return_exceptions to collect all results
            return await asyncio.gather(
                *[retry_with_semaphore(func) for func in funcs], return_exceptions=True
            )

    def get_retry_stats(self) -> dict:
        """Get retry statistics.

        Returns:
            Dictionary with retry statistics
        """
        return self._retry_stats.copy()

    def reset_stats(self) -> None:
        """Reset retry statistics."""
        self._retry_stats = {
            "total_requests": 0,
            "total_retries": 0,
            "failed_requests": 0,
            "rate_limited_requests": 0,
        }
