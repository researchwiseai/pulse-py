"""Async job handling for Pulse API."""

import asyncio
import time
from typing import Any, Optional, Literal, Callable, Awaitable, Union
import httpx
from pydantic import BaseModel, PrivateAttr, Field, ConfigDict
from pulse.core.exceptions import PulseAPIError


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

    async def refresh(
        self, max_retries: int = 10, retry_delay: float = 10.0
    ) -> "AsyncJob":
        """
        Refresh job status via GET /jobs?jobId={id}, retrying on 500 or 404
        up to max_retries times before giving up.
        """
        for attempt in range(max_retries):
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

        # should never get here
        raise PulseAPIError(response)

    async def wait(self, timeout: float = 180.0) -> Any:
        """
        Wait for job completion with timeout support.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Job result when completed

        Raises:
            asyncio.TimeoutError: If job doesn't complete within timeout
            RuntimeError: If job fails or encounters an error
            PulseAPIError: If API request fails
        """
        try:
            return await asyncio.wait_for(self._wait_loop(), timeout=timeout)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"Job {self.id} did not finish in {timeout} seconds"
            )

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
        Attempt to cancel the job.

        This method attempts to cancel a running job by sending a DELETE request
        to the jobs endpoint. Note that cancellation may not be possible for all
        job states (e.g., already completed jobs).

        Returns:
            True if cancellation was successful, False otherwise.

        Note:
            This method does not raise exceptions on failure, returning False instead
            to maintain compatibility with concurrent.futures.Future.cancel()
        """
        try:
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
        except Exception:
            return False

    async def wait_with_callback(
        self,
        callback: Union[Callable[[str], None], Callable[[str], Awaitable[None]]],
        timeout: float = 180.0,
    ) -> Any:
        """
        Wait for completion with periodic status callbacks.

        The callback function will be called whenever the job status changes.
        Both sync and async callback functions are supported.

        Args:
            callback: Function called with status updates. Can be sync or async.
            timeout: Maximum time to wait in seconds

        Returns:
            Job result when completed

        Raises:
            asyncio.TimeoutError: If job doesn't complete within timeout
            RuntimeError: If job fails or encounters an error
            PulseAPIError: If API request fails
        """

        async def callback_loop():
            last_status = None

            while True:
                job = await self.refresh()

                # Call callback if status changed
                if job.status != last_status:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(job.status)
                    else:
                        callback(job.status)
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

        try:
            return await asyncio.wait_for(callback_loop(), timeout=timeout)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"Job {self.id} did not finish in {timeout} seconds"
            )

    async def wait_for_status(
        self, target_status: str, timeout: float = 180.0
    ) -> "AsyncJob":
        """
        Wait until job reaches specific status.

        This method is useful when you want to wait for a specific intermediate
        status rather than completion (e.g., waiting for "queued" status).

        Args:
            target_status: Status to wait for (e.g., "pending", "queued", "completed")
            timeout: Maximum time to wait in seconds

        Returns:
            AsyncJob instance when target status is reached

        Raises:
            asyncio.TimeoutError: If target status not reached within timeout
            RuntimeError: If job fails before reaching target status
        """

        async def status_loop():
            while True:
                job = await self.refresh()

                if job.status == target_status:
                    return job

                if job.status in ("error", "failed"):
                    error_msg = job.message or ""
                    raise RuntimeError(f"Job {self.id} {job.status}: {error_msg}")

                await asyncio.sleep(2.0)

        try:
            return await asyncio.wait_for(status_loop(), timeout=timeout)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"Job {self.id} did not reach status '{target_status}' "
                f"in {timeout} seconds"
            )
