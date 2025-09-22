"""Enhanced async error handling and timeout patterns for Pulse SDK."""

import asyncio
import functools
import logging
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union, Dict, List
from dataclasses import dataclass
import httpx

from pulse.core.exceptions import PulseAPIError, NetworkError

T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass
class AsyncTimeoutConfig:
    """Configuration for async timeout behavior."""

    default_timeout: float = 180.0
    """Default timeout for operations in seconds."""

    job_timeout: float = 300.0
    """Timeout for job operations in seconds."""

    request_timeout: float = 60.0
    """Timeout for individual HTTP requests in seconds."""

    batch_timeout_multiplier: float = 1.5
    """Multiplier for batch operation timeouts."""

    enable_timeout_warnings: bool = True
    """Whether to log timeout warnings."""


class AsyncTimeoutError(asyncio.TimeoutError):
    """Enhanced timeout error with context information."""

    def __init__(
        self,
        message: str,
        operation: str,
        timeout: float,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.operation = operation
        self.timeout = timeout
        self.context = context or {}

    def __str__(self) -> str:
        context_str = ""
        if self.context:
            context_items = [f"{k}={v}" for k, v in self.context.items()]
            context_str = f" ({', '.join(context_items)})"

        return (
            f"Operation '{self.operation}' timed out after {self.timeout}s"
            f"{context_str}: {super().__str__()}"
        )


class AsyncCancellationError(asyncio.CancelledError):
    """Enhanced cancellation error with context information."""

    def __init__(
        self, message: str, operation: str, context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.operation = operation
        self.context = context or {}

    def __str__(self) -> str:
        context_str = ""
        if self.context:
            context_items = [f"{k}={v}" for k, v in self.context.items()]
            context_str = f" ({', '.join(context_items)})"

        return f"Operation '{self.operation}' was cancelled{context_str}: {super().__str__()}"


@asynccontextmanager
async def async_timeout_context(
    timeout: float,
    operation: str = "async_operation",
    context: Optional[Dict[str, Any]] = None,
    raise_on_timeout: bool = True,
):
    """
    Async context manager for timeout handling with enhanced error information.

    Compatible with Python 3.10+ using a simple yield pattern.
    Note: This doesn't provide actual timeout functionality but provides
    the interface for enhanced error handling. Actual timeout should be
    implemented at the call site using asyncio.wait_for.

    Args:
        timeout: Timeout in seconds (for error reporting)
        operation: Name of the operation for error reporting
        context: Additional context information for error reporting
        raise_on_timeout: Whether to raise AsyncTimeoutError on timeout

    Yields:
        None

    Example:
        async with async_timeout_context(30.0, "job_wait", {"job_id": "123"}):
            result = await some_long_operation()
    """
    try:
        yield
    except asyncio.TimeoutError as e:
        if raise_on_timeout:
            raise AsyncTimeoutError(
                f"Operation timed out after {timeout} seconds",
                operation=operation,
                timeout=timeout,
                context=context,
            ) from e
        else:
            logger.warning(
                f"Operation '{operation}' timed out after {timeout}s "
                f"(context: {context})"
            )
            raise


def async_timeout_decorator(
    timeout: Optional[float] = None,
    operation: Optional[str] = None,
    context_factory: Optional[Callable[..., Dict[str, Any]]] = None,
):
    """
    Decorator for adding timeout handling to async functions.

    Args:
        timeout: Timeout in seconds (uses function default if None)
        operation: Operation name (uses function name if None)
        context_factory: Function to generate context from function args

    Example:
        @async_timeout_decorator(timeout=30.0, operation="job_wait")
        async def wait_for_job(job_id: str):
            # Function implementation
            pass
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Determine timeout
            actual_timeout = timeout
            if actual_timeout is None:
                # Try to get timeout from kwargs
                actual_timeout = kwargs.get("timeout", 180.0)

            # Determine operation name
            op_name = operation or func.__name__

            # Generate context
            context = {}
            if context_factory:
                try:
                    context = context_factory(*args, **kwargs)
                except Exception as e:
                    logger.debug(f"Failed to generate context: {e}")

            async with async_timeout_context(actual_timeout, op_name, context):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


class AsyncErrorRecovery:
    """Async-specific error recovery patterns."""

    def __init__(self, config: Optional[AsyncTimeoutConfig] = None):
        self.config = config or AsyncTimeoutConfig()

    async def with_timeout_and_cancellation(
        self,
        coro: Awaitable[T],
        timeout: Optional[float] = None,
        operation: str = "async_operation",
        context: Optional[Dict[str, Any]] = None,
        handle_cancellation: bool = True,
    ) -> T:
        """
        Execute coroutine with timeout and cancellation handling.

        Args:
            coro: Coroutine to execute
            timeout: Timeout in seconds
            operation: Operation name for error reporting
            context: Additional context for error reporting
            handle_cancellation: Whether to handle CancelledError

        Returns:
            Result of the coroutine

        Raises:
            AsyncTimeoutError: If operation times out
            AsyncCancellationError: If operation is cancelled and handle_cancellation is True
        """
        actual_timeout = timeout or self.config.default_timeout

        try:
            # Use asyncio.wait_for for actual timeout functionality
            async with async_timeout_context(actual_timeout, operation, context):
                return await asyncio.wait_for(coro, timeout=actual_timeout)
        except asyncio.CancelledError as e:
            if handle_cancellation:
                raise AsyncCancellationError(
                    "Operation was cancelled", operation=operation, context=context
                ) from e
            else:
                raise

    async def with_retry_and_timeout(
        self,
        coro_factory: Callable[[], Awaitable[T]],
        max_retries: int = 3,
        timeout_per_attempt: Optional[float] = None,
        total_timeout: Optional[float] = None,
        operation: str = "async_operation",
        context: Optional[Dict[str, Any]] = None,
        retry_on_timeout: bool = True,
        retry_on_cancellation: bool = False,
    ) -> T:
        """
        Execute coroutine with retry logic and timeout handling.

        Args:
            coro_factory: Factory function that creates the coroutine
            max_retries: Maximum number of retry attempts
            timeout_per_attempt: Timeout for each attempt
            total_timeout: Total timeout for all attempts
            operation: Operation name for error reporting
            context: Additional context for error reporting
            retry_on_timeout: Whether to retry on timeout
            retry_on_cancellation: Whether to retry on cancellation

        Returns:
            Result of the coroutine

        Raises:
            AsyncTimeoutError: If operation times out
            AsyncCancellationError: If operation is cancelled
        """
        attempt_timeout = timeout_per_attempt or self.config.request_timeout
        total_timeout_val = total_timeout or self.config.default_timeout

        start_time = asyncio.get_event_loop().time()
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                # Check if we have time left for this attempt
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= total_timeout_val:
                    raise AsyncTimeoutError(
                        f"Total timeout exceeded before attempt {attempt + 1}",
                        operation=operation,
                        timeout=total_timeout_val,
                        context={**(context or {}), "attempt": attempt + 1},
                    )

                # Calculate remaining time for this attempt
                remaining_time = min(attempt_timeout, total_timeout_val - elapsed)

                async with async_timeout_context(
                    remaining_time,
                    f"{operation}_attempt_{attempt + 1}",
                    {**(context or {}), "attempt": attempt + 1},
                ):
                    return await asyncio.wait_for(
                        coro_factory(), timeout=remaining_time
                    )

            except asyncio.TimeoutError as e:
                last_exception = e
                if not retry_on_timeout or attempt == max_retries:
                    raise AsyncTimeoutError(
                        f"Operation failed after {attempt + 1} attempts",
                        operation=operation,
                        timeout=total_timeout_val,
                        context={**(context or {}), "final_attempt": attempt + 1},
                    ) from e

                # Exponential backoff for retry
                if attempt < max_retries:
                    backoff_delay = min(2.0**attempt, 10.0)
                    await asyncio.sleep(backoff_delay)

            except asyncio.CancelledError as e:
                last_exception = e
                if not retry_on_cancellation or attempt == max_retries:
                    raise AsyncCancellationError(
                        f"Operation cancelled after {attempt + 1} attempts",
                        operation=operation,
                        context={**(context or {}), "final_attempt": attempt + 1},
                    ) from e

                # Brief delay before retry on cancellation
                if attempt < max_retries:
                    await asyncio.sleep(0.1)

        # Should not reach here, but handle gracefully
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Unexpected error in retry logic for {operation}")

    async def gather_with_error_handling(
        self,
        *coroutines: Awaitable[Any],
        timeout: Optional[float] = None,
        return_exceptions: bool = True,
        operation: str = "gather_operation",
        context: Optional[Dict[str, Any]] = None,
        cancel_on_first_error: bool = False,
    ) -> List[Union[Any, Exception]]:
        """
        Gather coroutines with enhanced error handling.

        Args:
            *coroutines: Coroutines to gather
            timeout: Timeout for the entire gather operation
            return_exceptions: Whether to return exceptions instead of raising
            operation: Operation name for error reporting
            context: Additional context for error reporting
            cancel_on_first_error: Whether to cancel remaining tasks on first error

        Returns:
            List of results or exceptions
        """
        actual_timeout = timeout or self.config.default_timeout

        try:
            async with async_timeout_context(
                actual_timeout,
                operation,
                {**(context or {}), "coroutine_count": len(coroutines)},
            ):
                if cancel_on_first_error:
                    # Use gather without return_exceptions to fail fast
                    return await asyncio.wait_for(
                        asyncio.gather(*coroutines), timeout=actual_timeout
                    )
                else:
                    # Use gather with return_exceptions
                    return await asyncio.wait_for(
                        asyncio.gather(
                            *coroutines, return_exceptions=return_exceptions
                        ),
                        timeout=actual_timeout,
                    )

        except asyncio.CancelledError as e:
            # Cancel all remaining tasks
            for coro in coroutines:
                if hasattr(coro, "cancel"):
                    coro.cancel()

            raise AsyncCancellationError(
                "Gather operation cancelled",
                operation=operation,
                context={**(context or {}), "coroutine_count": len(coroutines)},
            ) from e


def handle_async_http_errors(
    func: Callable[..., Awaitable[T]],
) -> Callable[..., Awaitable[T]]:
    """
    Decorator for handling async HTTP errors with proper exception chaining.

    This decorator catches httpx exceptions and converts them to appropriate
    Pulse SDK exceptions with proper async context.
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        try:
            return await func(*args, **kwargs)
        except httpx.TimeoutException as e:
            # Convert httpx timeout to our timeout error
            raise AsyncTimeoutError(
                "HTTP request timed out",
                operation=func.__name__,
                timeout=getattr(e, "timeout", 0.0),
                context={"url": getattr(e, "url", "unknown")},
            ) from e
        except httpx.ConnectError as e:
            # Convert connection errors to network errors
            raise NetworkError(url=str(getattr(e, "url", "unknown")), cause=e) from e
        except httpx.HTTPStatusError as e:
            # Convert HTTP status errors to PulseAPIError
            raise PulseAPIError(e.response) from e
        except httpx.RequestError as e:
            # Convert other request errors to network errors
            raise NetworkError(url=str(getattr(e, "url", "unknown")), cause=e) from e
        except asyncio.CancelledError:
            # Re-raise cancellation errors without modification
            raise
        except asyncio.TimeoutError as e:
            # Convert generic asyncio timeout to our enhanced timeout
            raise AsyncTimeoutError(
                "Operation timed out",
                operation=func.__name__,
                timeout=0.0,  # Unknown timeout value
                context={},
            ) from e

    return wrapper


class AsyncJobTimeoutManager:
    """Specialized timeout manager for AsyncJob operations."""

    def __init__(self, config: Optional[AsyncTimeoutConfig] = None):
        self.config = config or AsyncTimeoutConfig()
        self.error_recovery = AsyncErrorRecovery(config)

    async def wait_with_timeout(
        self,
        job,  # AsyncJob type hint would create circular import
        timeout: Optional[float] = None,
        operation: str = "job_wait",
    ) -> Any:
        """
        Wait for job completion with enhanced timeout handling.

        Args:
            job: AsyncJob instance
            timeout: Timeout in seconds
            operation: Operation name for error reporting

        Returns:
            Job result

        Raises:
            AsyncTimeoutError: If job doesn't complete within timeout
            AsyncCancellationError: If operation is cancelled
        """
        actual_timeout = timeout or self.config.job_timeout
        context = {"job_id": getattr(job, "id", "unknown")}

        return await self.error_recovery.with_timeout_and_cancellation(
            job.wait(timeout=actual_timeout),
            timeout=actual_timeout,
            operation=operation,
            context=context,
        )

    async def wait_for_status_with_timeout(
        self,
        job,  # AsyncJob type hint would create circular import
        target_status: str,
        timeout: Optional[float] = None,
        operation: str = "job_wait_for_status",
    ):
        """
        Wait for specific job status with enhanced timeout handling.

        Args:
            job: AsyncJob instance
            target_status: Status to wait for
            timeout: Timeout in seconds
            operation: Operation name for error reporting

        Returns:
            AsyncJob instance when target status is reached

        Raises:
            AsyncTimeoutError: If status not reached within timeout
            AsyncCancellationError: If operation is cancelled
        """
        actual_timeout = timeout or self.config.job_timeout
        context = {
            "job_id": getattr(job, "id", "unknown"),
            "target_status": target_status,
        }

        return await self.error_recovery.with_timeout_and_cancellation(
            job.wait_for_status(target_status, timeout=actual_timeout),
            timeout=actual_timeout,
            operation=operation,
            context=context,
        )

    async def cancel_with_timeout(
        self,
        job,  # AsyncJob type hint would create circular import
        timeout: Optional[float] = None,
        operation: str = "job_cancel",
    ) -> bool:
        """
        Cancel job with timeout handling.

        Args:
            job: AsyncJob instance
            timeout: Timeout in seconds
            operation: Operation name for error reporting

        Returns:
            True if cancellation was successful

        Raises:
            AsyncTimeoutError: If cancellation times out
        """
        actual_timeout = timeout or self.config.request_timeout
        context = {"job_id": getattr(job, "id", "unknown")}

        return await self.error_recovery.with_timeout_and_cancellation(
            job.cancel(),
            timeout=actual_timeout,
            operation=operation,
            context=context,
            handle_cancellation=False,  # Don't wrap CancelledError for cancel operations
        )


# Global instances for convenience
default_timeout_config = AsyncTimeoutConfig()
default_error_recovery = AsyncErrorRecovery(default_timeout_config)
default_job_timeout_manager = AsyncJobTimeoutManager(default_timeout_config)


# Convenience functions
async def with_async_timeout(
    coro: Awaitable[T],
    timeout: float,
    operation: str = "async_operation",
    context: Optional[Dict[str, Any]] = None,
) -> T:
    """
    Convenience function for executing coroutine with timeout.

    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        operation: Operation name for error reporting
        context: Additional context for error reporting

    Returns:
        Result of the coroutine

    Raises:
        AsyncTimeoutError: If operation times out
    """
    return await default_error_recovery.with_timeout_and_cancellation(
        coro, timeout, operation, context
    )


async def gather_with_timeout(
    *coroutines: Awaitable[Any],
    timeout: float,
    return_exceptions: bool = True,
    operation: str = "gather_operation",
    context: Optional[Dict[str, Any]] = None,
) -> List[Union[Any, Exception]]:
    """
    Convenience function for gathering coroutines with timeout.

    Args:
        *coroutines: Coroutines to gather
        timeout: Timeout in seconds
        return_exceptions: Whether to return exceptions instead of raising
        operation: Operation name for error reporting
        context: Additional context for error reporting

    Returns:
        List of results or exceptions
    """
    return await default_error_recovery.gather_with_error_handling(
        *coroutines,
        timeout=timeout,
        return_exceptions=return_exceptions,
        operation=operation,
        context=context,
    )
