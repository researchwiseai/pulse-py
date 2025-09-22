"""Async job handling for Pulse API."""

import asyncio
import time
from typing import Any, Optional, Literal, Callable, Awaitable, Union
import httpx
from pydantic import BaseModel, PrivateAttr, Field, ConfigDict
from pulse.core.exceptions import PulseAPIError
from pulse.core.async_error_handling import (
    AsyncCancellationError,
    async_timeout_context,
    handle_async_http_errors,
)


class AsyncJob(BaseModel):
    """Represents an asynchronous job in Pulse API with async capabilities."""

    id: str = Field(alias="jobId")
    status: Literal["pending", "queued", "completed", "error", "failed"] = Field(
        alias="jobStatus"
    )
    message: Optional[str] = Field(default=None, alias="message")
    result_url: Optional[str] = Field(default=None, alias="resultUrl")

    _client: httpx.AsyncClient = PrivateAttr()
    model_config = ConfigDict(populate_by_name=True)

    @handle_async_http_errors
    async def refresh(
        self, max_retries: int = 10, retry_delay: float = 10.0
    ) -> "AsyncJob":
        """
        Refresh job status via GET /jobs?jobId={id}, retrying on 500 or 404
        up to max_retries times before giving up.

        Enhanced with proper async error handling and cancellation support.
        """
        for attempt in range(max_retries):
            try:
                response = await self._client.get(f"/jobs?jobId={self.id}")

                # retry on server‐error or not‐found
                if response.status_code in (500, 404):
                    print(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} - "
                        f"Job {self.id} not found, retrying ({attempt + 1}/{max_retries})"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    raise PulseAPIError(response)

                # any other non‐200 is fatal
                if response.status_code != 200:
                    raise PulseAPIError(response)

                # success!
                data = response.json()
                if "jobId" not in data:
                    data["jobId"] = self.id

                job = AsyncJob.model_validate(data)
                job._client = self._client
                return job

            except asyncio.CancelledError as e:
                raise AsyncCancellationError(
                    f"Job refresh was cancelled during attempt {attempt + 1}",
                    operation="job_refresh",
                    context={"job_id": self.id, "attempt": attempt + 1},
                ) from e

        # should never get here
        raise PulseAPIError(response)

    async def wait(self, timeout: float = 180.0) -> Any:
        """
        Wait for job completion with enhanced timeout and cancellation support.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Job result when completed

        Raises:
            AsyncTimeoutError: If job doesn't complete within timeout
            AsyncCancellationError: If operation is cancelled
            RuntimeError: If job fails or encounters an error
            PulseAPIError: If API request fails
        """
        try:
            async with async_timeout_context(timeout, "job_wait", {"job_id": self.id}):
                return await asyncio.wait_for(self._wait_loop(), timeout=timeout)
        except asyncio.CancelledError as e:
            raise AsyncCancellationError(
                "Job wait was cancelled",
                operation="job_wait",
                context={"job_id": self.id},
            ) from e

    async def _wait_loop(self) -> Any:
        """Internal wait loop without timeout handling."""
        while True:
            job = await self.refresh()
            if job.status in ("pending", "queued"):
                pass
            elif job.status == "completed":
                if job.result_url:
                    response = await self._client.get(job.result_url)
                    if response.status_code != 200:
                        raise PulseAPIError(response)
                    return response.json()
                return job
            else:
                error_msg = job.message or ""
                raise RuntimeError(f"Job {self.id} {job.status}: {error_msg}")

            await asyncio.sleep(2.0)

    async def result(self, timeout: float = 180.0) -> Any:
        """Alias for :meth:`wait` for API parity with ``concurrent.futures``."""
        return await self.wait(timeout)

    async def cancel(self) -> bool:
        """
        Attempt to cancel the job with enhanced error handling.

        This method attempts to cancel a running job by sending a DELETE request
        to the jobs endpoint. Note that cancellation may not be possible for all
        job states (e.g., already completed jobs).

        Returns:
            True if cancellation was successful, False otherwise.

        Note:
            This method does not raise exceptions on failure, returning False instead
            to maintain compatibility with concurrent.futures.Future.cancel().
            However, it now properly handles asyncio.CancelledError.
        """
        try:
            # Use a short timeout for cancellation requests
            async with async_timeout_context(
                10.0,
                "job_cancel",
                {"job_id": self.id},
                raise_on_timeout=False,  # Don't raise on timeout, return False
            ):

                async def cancel_operation():
                    response = await self._client.delete(f"/jobs/{self.id}")
                    if response.status_code == 200:
                        # Try to update our status to reflect cancellation
                        try:
                            await self.refresh()
                        except Exception:
                            # Ignore refresh errors - cancellation was still successful
                            pass
                        return True
                    return False

                return await asyncio.wait_for(cancel_operation(), timeout=10.0)
        except asyncio.TimeoutError:
            # Timeout during cancellation - return False
            return False
        except asyncio.CancelledError:
            # If the cancellation request itself is cancelled,
            # we can't determine the job state reliably
            return False
        except Exception:
            return False

    async def wait_with_callback(
        self,
        callback: Union[Callable[[str], None], Callable[[str], Awaitable[None]]],
        timeout: float = 180.0,
    ) -> Any:
        """
        Wait for completion with periodic status callbacks and enhanced error handling.

        The callback function will be called whenever the job status changes.
        Both sync and async callback functions are supported.

        Args:
            callback: Function called with status updates. Can be sync or async.
            timeout: Maximum time to wait in seconds

        Returns:
            Job result when completed

        Raises:
            AsyncTimeoutError: If job doesn't complete within timeout
            AsyncCancellationError: If operation is cancelled
            RuntimeError: If job fails or encounters an error
            PulseAPIError: If API request fails
        """

        async def callback_loop():
            last_status = None

            while True:
                try:
                    job = await self.refresh()

                    # Call callback if status changed
                    if job.status != last_status:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(job.status)
                            else:
                                callback(job.status)
                        except asyncio.CancelledError:
                            # Re-raise cancellation from callback
                            raise
                        except Exception as e:
                            # Log callback errors but don't fail the wait
                            import logging

                            logging.warning(f"Callback error for job {self.id}: {e}")

                        last_status = job.status

                    if job.status in ("pending", "queued"):
                        pass
                    elif job.status == "completed":
                        if job.result_url:
                            response = await self._client.get(job.result_url)
                            if response.status_code != 200:
                                raise PulseAPIError(response)
                            return response.json()
                        return job
                    else:
                        error_msg = job.message or ""
                        raise RuntimeError(f"Job {self.id} {job.status}: {error_msg}")

                    await asyncio.sleep(2.0)

                except asyncio.CancelledError:
                    # Re-raise cancellation errors
                    raise

        try:
            async with async_timeout_context(
                timeout, "job_wait_with_callback", {"job_id": self.id}
            ):
                return await asyncio.wait_for(callback_loop(), timeout=timeout)
        except asyncio.CancelledError as e:
            raise AsyncCancellationError(
                "Job wait with callback was cancelled",
                operation="job_wait_with_callback",
                context={"job_id": self.id},
            ) from e

    async def wait_for_status(
        self, target_status: str, timeout: float = 180.0
    ) -> "AsyncJob":
        """
        Wait until job reaches specific status with enhanced error handling.

        This method is useful when you want to wait for a specific intermediate
        status rather than completion (e.g., waiting for "queued" status).

        Args:
            target_status: Status to wait for (e.g., "pending", "queued", "completed")
            timeout: Maximum time to wait in seconds

        Returns:
            AsyncJob instance when target status is reached

        Raises:
            AsyncTimeoutError: If target status not reached within timeout
            AsyncCancellationError: If operation is cancelled
            RuntimeError: If job fails before reaching target status
        """

        async def status_loop():
            while True:
                try:
                    job = await self.refresh()

                    if job.status == target_status:
                        return job

                    if job.status in ("error", "failed"):
                        error_msg = job.message or ""
                        raise RuntimeError(f"Job {self.id} {job.status}: {error_msg}")

                    await asyncio.sleep(2.0)

                except asyncio.CancelledError:
                    # Re-raise cancellation errors
                    raise

        try:
            async with async_timeout_context(
                timeout,
                "job_wait_for_status",
                {"job_id": self.id, "target_status": target_status},
            ):
                return await asyncio.wait_for(status_loop(), timeout=timeout)
        except asyncio.CancelledError as e:
            raise AsyncCancellationError(
                "Job wait for status was cancelled",
                operation="job_wait_for_status",
                context={"job_id": self.id, "target_status": target_status},
            ) from e
