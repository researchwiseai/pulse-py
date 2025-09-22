"""Tests for async error handling and timeout patterns."""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock
import httpx

from pulse.core.async_error_handling import (
    AsyncTimeoutError,
    AsyncCancellationError,
    AsyncTimeoutConfig,
    AsyncErrorRecovery,
    AsyncJobTimeoutManager,
    async_timeout_context,
    async_timeout_decorator,
    handle_async_http_errors,
    with_async_timeout,
    gather_with_timeout,
)
from pulse.core.exceptions import PulseAPIError, NetworkError


class TestAsyncTimeoutError:
    """Test AsyncTimeoutError functionality."""

    def test_timeout_error_creation(self):
        """Test creating AsyncTimeoutError with context."""
        error = AsyncTimeoutError(
            "Operation timed out",
            operation="test_operation",
            timeout=30.0,
            context={"key": "value"},
        )

        assert error.operation == "test_operation"
        assert error.timeout == 30.0
        assert error.context == {"key": "value"}
        assert "test_operation" in str(error)
        assert "30.0s" in str(error)

    def test_timeout_error_without_context(self):
        """Test creating AsyncTimeoutError without context."""
        error = AsyncTimeoutError(
            "Operation timed out", operation="test_operation", timeout=30.0
        )

        assert error.context == {}
        assert "test_operation" in str(error)


class TestAsyncCancellationError:
    """Test AsyncCancellationError functionality."""

    def test_cancellation_error_creation(self):
        """Test creating AsyncCancellationError with context."""
        error = AsyncCancellationError(
            "Operation cancelled", operation="test_operation", context={"key": "value"}
        )

        assert error.operation == "test_operation"
        assert error.context == {"key": "value"}
        assert "test_operation" in str(error)
        assert "cancelled" in str(error)


class TestAsyncTimeoutContext:
    """Test async timeout context manager."""

    @pytest.mark.asyncio
    async def test_timeout_context_success(self):
        """Test timeout context with successful operation."""
        async with async_timeout_context(1.0, "test_op"):
            await asyncio.sleep(0.1)  # Should complete within timeout

    @pytest.mark.asyncio
    async def test_timeout_context_timeout(self):
        """Test timeout context with timeout."""
        with pytest.raises(AsyncTimeoutError) as exc_info:
            async with async_timeout_context(0.1, "test_op", {"key": "value"}):
                await asyncio.sleep(1.0)  # Should timeout

        error = exc_info.value
        assert error.operation == "test_op"
        assert error.timeout == 0.1
        assert error.context == {"key": "value"}

    @pytest.mark.asyncio
    async def test_timeout_context_no_raise(self):
        """Test timeout context with raise_on_timeout=False."""
        with pytest.raises(asyncio.TimeoutError):  # Original timeout error
            async with async_timeout_context(0.1, "test_op", raise_on_timeout=False):
                await asyncio.sleep(1.0)

    @pytest.mark.asyncio
    async def test_timeout_context_cancellation(self):
        """Test timeout context with cancellation."""

        async def cancelled_operation():
            async with async_timeout_context(1.0, "test_op"):
                await asyncio.sleep(10.0)

        task = asyncio.create_task(cancelled_operation())
        await asyncio.sleep(0.1)  # Let task start
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


class TestAsyncTimeoutDecorator:
    """Test async timeout decorator."""

    @pytest.mark.asyncio
    async def test_timeout_decorator_success(self):
        """Test timeout decorator with successful operation."""

        @async_timeout_decorator(timeout=1.0, operation="test_func")
        async def test_func():
            await asyncio.sleep(0.1)
            return "success"

        result = await test_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_timeout_decorator_timeout(self):
        """Test timeout decorator with timeout."""

        @async_timeout_decorator(timeout=0.1, operation="test_func")
        async def test_func():
            await asyncio.sleep(1.0)
            return "success"

        with pytest.raises(AsyncTimeoutError) as exc_info:
            await test_func()

        error = exc_info.value
        assert error.operation == "test_func"
        assert error.timeout == 0.1

    @pytest.mark.asyncio
    async def test_timeout_decorator_with_context_factory(self):
        """Test timeout decorator with context factory."""

        def context_factory(arg1, arg2=None):
            return {"arg1": arg1, "arg2": arg2}

        @async_timeout_decorator(
            timeout=0.1, operation="test_func", context_factory=context_factory
        )
        async def test_func(arg1, arg2=None):
            await asyncio.sleep(1.0)
            return "success"

        with pytest.raises(AsyncTimeoutError) as exc_info:
            await test_func("value1", arg2="value2")

        error = exc_info.value
        assert error.context == {"arg1": "value1", "arg2": "value2"}

    @pytest.mark.asyncio
    async def test_timeout_decorator_from_kwargs(self):
        """Test timeout decorator using timeout from kwargs."""

        @async_timeout_decorator(operation="test_func")
        async def test_func(timeout=1.0):
            await asyncio.sleep(0.1)
            return "success"

        result = await test_func(timeout=1.0)
        assert result == "success"


class TestAsyncErrorRecovery:
    """Test AsyncErrorRecovery functionality."""

    @pytest.fixture
    def error_recovery(self):
        """Create AsyncErrorRecovery instance."""
        config = AsyncTimeoutConfig(default_timeout=1.0, request_timeout=0.5)
        return AsyncErrorRecovery(config)

    @pytest.mark.asyncio
    async def test_with_timeout_and_cancellation_success(self, error_recovery):
        """Test successful operation with timeout and cancellation handling."""

        async def test_coro():
            await asyncio.sleep(0.1)
            return "success"

        result = await error_recovery.with_timeout_and_cancellation(
            test_coro(), timeout=1.0, operation="test_op"
        )
        assert result == "success"

    @pytest.mark.asyncio
    async def test_with_timeout_and_cancellation_timeout(self, error_recovery):
        """Test timeout with timeout and cancellation handling."""

        async def test_coro():
            await asyncio.sleep(2.0)
            return "success"

        with pytest.raises(AsyncTimeoutError) as exc_info:
            await error_recovery.with_timeout_and_cancellation(
                test_coro(), timeout=0.1, operation="test_op", context={"key": "value"}
            )

        error = exc_info.value
        assert error.operation == "test_op"
        assert error.context == {"key": "value"}

    @pytest.mark.asyncio
    async def test_with_timeout_and_cancellation_cancelled(self, error_recovery):
        """Test cancellation with timeout and cancellation handling."""

        async def test_operation():
            async def test_coro():
                await asyncio.sleep(10.0)
                return "success"

            return await error_recovery.with_timeout_and_cancellation(
                test_coro(), timeout=1.0, operation="test_op", handle_cancellation=True
            )

        task = asyncio.create_task(test_operation())
        await asyncio.sleep(0.1)  # Let task start
        task.cancel()

        with pytest.raises(AsyncCancellationError):
            await task

    @pytest.mark.asyncio
    async def test_with_retry_and_timeout_success(self, error_recovery):
        """Test successful retry with timeout."""
        call_count = 0

        async def test_coro():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"

        result = await error_recovery.with_retry_and_timeout(
            test_coro, max_retries=3, timeout_per_attempt=0.5, operation="test_op"
        )
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_with_retry_and_timeout_max_retries(self, error_recovery):
        """Test retry exhaustion."""

        async def test_coro():
            raise ValueError("Persistent error")

        with pytest.raises(ValueError):
            await error_recovery.with_retry_and_timeout(
                test_coro, max_retries=2, timeout_per_attempt=0.1, operation="test_op"
            )

    @pytest.mark.asyncio
    async def test_gather_with_error_handling_success(self, error_recovery):
        """Test successful gather with error handling."""

        async def test_coro(value):
            await asyncio.sleep(0.1)
            return value

        results = await error_recovery.gather_with_error_handling(
            test_coro("a"),
            test_coro("b"),
            test_coro("c"),
            timeout=1.0,
            return_exceptions=False,
        )
        assert results == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_gather_with_error_handling_with_exceptions(self, error_recovery):
        """Test gather with exceptions."""

        async def success_coro():
            return "success"

        async def error_coro():
            raise ValueError("Test error")

        results = await error_recovery.gather_with_error_handling(
            success_coro(), error_coro(), timeout=1.0, return_exceptions=True
        )

        assert results[0] == "success"
        assert isinstance(results[1], ValueError)

    @pytest.mark.asyncio
    async def test_gather_with_error_handling_timeout(self, error_recovery):
        """Test gather with timeout."""

        async def slow_coro():
            await asyncio.sleep(2.0)
            return "slow"

        with pytest.raises(AsyncTimeoutError):
            await error_recovery.gather_with_error_handling(
                slow_coro(), timeout=0.1, operation="test_gather"
            )


class TestHandleAsyncHttpErrors:
    """Test handle_async_http_errors decorator."""

    @pytest.mark.asyncio
    async def test_handle_http_timeout(self):
        """Test handling httpx timeout errors."""

        @handle_async_http_errors
        async def test_func():
            raise httpx.TimeoutException("Request timed out")

        with pytest.raises(AsyncTimeoutError) as exc_info:
            await test_func()

        error = exc_info.value
        assert error.operation == "test_func"

    @pytest.mark.asyncio
    async def test_handle_http_connect_error(self):
        """Test handling httpx connection errors."""

        @handle_async_http_errors
        async def test_func():
            raise httpx.ConnectError("Connection failed")

        with pytest.raises(NetworkError):
            await test_func()

    @pytest.mark.asyncio
    async def test_handle_http_status_error(self):
        """Test handling httpx status errors."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.headers = {}

        @handle_async_http_errors
        async def test_func():
            raise httpx.HTTPStatusError(
                "Bad request", request=Mock(), response=mock_response
            )

        with pytest.raises(PulseAPIError):
            await test_func()

    @pytest.mark.asyncio
    async def test_handle_cancellation_passthrough(self):
        """Test that cancellation errors pass through unchanged."""

        @handle_async_http_errors
        async def test_func():
            raise asyncio.CancelledError("Cancelled")

        with pytest.raises(asyncio.CancelledError):
            await test_func()

    @pytest.mark.asyncio
    async def test_handle_generic_asyncio_timeout(self):
        """Test handling generic asyncio timeout errors."""

        @handle_async_http_errors
        async def test_func():
            raise asyncio.TimeoutError("Generic timeout")

        with pytest.raises(AsyncTimeoutError) as exc_info:
            await test_func()

        error = exc_info.value
        assert error.operation == "test_func"


class TestAsyncJobTimeoutManager:
    """Test AsyncJobTimeoutManager functionality."""

    @pytest.fixture
    def timeout_manager(self):
        """Create AsyncJobTimeoutManager instance."""
        config = AsyncTimeoutConfig(job_timeout=1.0)
        return AsyncJobTimeoutManager(config)

    @pytest.fixture
    def mock_job(self):
        """Create mock AsyncJob."""
        job = Mock()
        job.id = "test-job-123"
        job.wait = AsyncMock(return_value="job_result")
        job.wait_for_status = AsyncMock(return_value=job)
        job.cancel = AsyncMock(return_value=True)
        return job

    @pytest.mark.asyncio
    async def test_wait_with_timeout_success(self, timeout_manager, mock_job):
        """Test successful job wait with timeout."""
        result = await timeout_manager.wait_with_timeout(mock_job)
        assert result == "job_result"
        mock_job.wait.assert_called_once_with(timeout=1.0)

    @pytest.mark.asyncio
    async def test_wait_with_timeout_job_timeout(self, timeout_manager, mock_job):
        """Test job wait timeout."""
        mock_job.wait.side_effect = asyncio.TimeoutError("Job timeout")

        with pytest.raises(AsyncTimeoutError) as exc_info:
            await timeout_manager.wait_with_timeout(mock_job)

        error = exc_info.value
        assert error.operation == "job_wait"
        assert "test-job-123" in str(error.context)

    @pytest.mark.asyncio
    async def test_wait_for_status_with_timeout_success(
        self, timeout_manager, mock_job
    ):
        """Test successful wait for status with timeout."""
        result = await timeout_manager.wait_for_status_with_timeout(
            mock_job, "completed"
        )
        assert result == mock_job
        mock_job.wait_for_status.assert_called_once_with("completed", timeout=1.0)

    @pytest.mark.asyncio
    async def test_cancel_with_timeout_success(self, timeout_manager, mock_job):
        """Test successful job cancellation with timeout."""
        result = await timeout_manager.cancel_with_timeout(mock_job)
        assert result is True
        mock_job.cancel.assert_called_once()


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_with_async_timeout_success(self):
        """Test with_async_timeout success."""

        async def test_coro():
            await asyncio.sleep(0.1)
            return "success"

        result = await with_async_timeout(test_coro(), 1.0, "test_op")
        assert result == "success"

    @pytest.mark.asyncio
    async def test_with_async_timeout_timeout(self):
        """Test with_async_timeout timeout."""

        async def test_coro():
            await asyncio.sleep(2.0)
            return "success"

        with pytest.raises(AsyncTimeoutError):
            await with_async_timeout(test_coro(), 0.1, "test_op")

    @pytest.mark.asyncio
    async def test_gather_with_timeout_success(self):
        """Test gather_with_timeout success."""

        async def test_coro(value):
            await asyncio.sleep(0.1)
            return value

        results = await gather_with_timeout(test_coro("a"), test_coro("b"), timeout=1.0)
        assert results == ["a", "b"]

    @pytest.mark.asyncio
    async def test_gather_with_timeout_timeout(self):
        """Test gather_with_timeout timeout."""

        async def slow_coro():
            await asyncio.sleep(2.0)
            return "slow"

        with pytest.raises(AsyncTimeoutError):
            await gather_with_timeout(slow_coro(), timeout=0.1)


class TestAsyncTimeoutConfig:
    """Test AsyncTimeoutConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AsyncTimeoutConfig()
        assert config.default_timeout == 180.0
        assert config.job_timeout == 300.0
        assert config.request_timeout == 60.0
        assert config.batch_timeout_multiplier == 1.5
        assert config.enable_timeout_warnings is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AsyncTimeoutConfig(
            default_timeout=30.0,
            job_timeout=60.0,
            request_timeout=10.0,
            batch_timeout_multiplier=2.0,
            enable_timeout_warnings=False,
        )
        assert config.default_timeout == 30.0
        assert config.job_timeout == 60.0
        assert config.request_timeout == 10.0
        assert config.batch_timeout_multiplier == 2.0
        assert config.enable_timeout_warnings is False
