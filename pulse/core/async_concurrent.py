"""Async concurrent processing utilities for managing multiple AsyncJob instances."""

import asyncio
import time
from typing import Any, List, Optional, Union, Callable, Awaitable, Dict, TypeVar
from dataclasses import dataclass, field
import httpx
from pulse.core.async_jobs import AsyncJob

T = TypeVar("T")


@dataclass
class AsyncConcurrentConfig:
    """Configuration for async concurrent job processing."""

    max_concurrent_jobs: int = 5
    """Maximum number of jobs to process concurrently."""

    timeout_per_job: float = 180.0
    """Timeout for individual job completion in seconds."""

    rate_limit_delay: float = 0.1
    """Minimum delay between job submissions in seconds."""

    max_retries: int = 3
    """Maximum number of retries for failed operations."""

    retry_delay: float = 1.0
    """Base delay between retries in seconds (exponential backoff)."""

    enable_progress: bool = True
    """Whether to show progress information."""

    connection_pool_limits: Dict[str, int] = field(
        default_factory=lambda: {
            "max_keepalive_connections": 20,
            "max_connections": 100,
            "keepalive_expiry": 30.0,
        }
    )
    """HTTP connection pool configuration for high concurrency."""


class AsyncJobManager:
    """Manager for concurrent AsyncJob processing with rate limiting."""

    def __init__(self, config: Optional[AsyncConcurrentConfig] = None):
        """Initialize the async job manager.

        Args:
            config: Configuration for concurrent processing.
                Defaults to AsyncConcurrentConfig().
        """
        self.config = config or AsyncConcurrentConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_jobs)
        self._rate_limiter = asyncio.Semaphore(1)  # For rate limiting job submissions
        self._last_submission_time = 0.0

    async def gather_jobs(
        self,
        jobs: List[AsyncJob],
        timeout: Optional[float] = None,
        return_exceptions: bool = False,
    ) -> List[Union[Any, Exception]]:
        """Gather results from multiple AsyncJob instances concurrently.

        This is the main utility function for awaiting multiple AsyncJob instances
        with proper concurrency control and error handling.

        Args:
            jobs: List of AsyncJob instances to await.
            timeout: Overall timeout for all jobs. Defaults to config timeout.
            return_exceptions: If True, exceptions are returned instead of raised.

        Returns:
            List of job results in the same order as input jobs.

        Raises:
            asyncio.TimeoutError: If jobs don't complete within timeout.
            RuntimeError: If jobs fail and return_exceptions is False.
        """
        if not jobs:
            return []

        timeout = timeout or self.config.timeout_per_job

        async def wait_with_semaphore(job: AsyncJob) -> Any:
            """Wait for a job with semaphore-based concurrency control."""
            async with self._semaphore:
                return await job.wait(timeout=timeout)

        # Create tasks for all jobs
        tasks = [wait_with_semaphore(job) for job in jobs]

        try:
            # Use asyncio.gather with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=return_exceptions),
                timeout=timeout * len(jobs),  # Scale timeout with job count
            )
            return results
        except asyncio.TimeoutError:
            # Cancel remaining tasks - need to create tasks first
            created_tasks = [
                asyncio.create_task(task) if asyncio.iscoroutine(task) else task
                for task in tasks
            ]
            for task in created_tasks:
                if hasattr(task, "done") and not task.done():
                    task.cancel()
            raise asyncio.TimeoutError(
                f"Failed to complete {len(jobs)} jobs within "
                f"{timeout * len(jobs)} seconds"
            )

    async def submit_jobs_with_rate_limit(
        self,
        job_submitters: List[Callable[[], Awaitable[AsyncJob]]],
        rate_limit_delay: Optional[float] = None,
    ) -> List[AsyncJob]:
        """Submit multiple jobs with rate limiting to prevent API overload.

        Args:
            job_submitters: List of async callables that return AsyncJob instances.
            rate_limit_delay: Override default rate limit delay.

        Returns:
            List of submitted AsyncJob instances.

        Raises:
            PulseAPIError: If job submission fails.
        """
        if not job_submitters:
            return []

        rate_delay = rate_limit_delay or self.config.rate_limit_delay
        jobs = []
        failed_submissions = []

        for i, submitter in enumerate(job_submitters):
            try:
                # Rate limiting
                async with self._rate_limiter:
                    current_time = time.time()
                    time_since_last = current_time - self._last_submission_time

                    if time_since_last < rate_delay:
                        await asyncio.sleep(rate_delay - time_since_last)

                    # Submit job
                    job = await submitter()
                    jobs.append(job)
                    self._last_submission_time = time.time()

            except Exception as e:
                failed_submissions.append({"index": i, "error": str(e), "exception": e})

        # Handle failed submissions
        if failed_submissions:
            if len(failed_submissions) == len(job_submitters):
                # All submissions failed
                first_error = failed_submissions[0]["exception"]
                raise RuntimeError(
                    f"All {len(failed_submissions)} job submissions failed. "
                    f"First error: {failed_submissions[0]['error']}"
                ) from first_error
            else:
                # Some submissions failed - log warning
                import warnings

                warnings.warn(
                    f"{len(failed_submissions)} of {len(job_submitters)} "
                    f"job submissions failed. Continuing with "
                    f"{len(jobs)} successful jobs.",
                    UserWarning,
                )

        return jobs

    async def submit_and_gather(
        self,
        job_submitters: List[Callable[[], Awaitable[AsyncJob]]],
        timeout: Optional[float] = None,
        return_exceptions: bool = False,
    ) -> List[Union[Any, Exception]]:
        """Submit jobs with rate limiting and gather their results.

        Combines job submission and result gathering in a single operation.

        Args:
            job_submitters: List of async callables that return AsyncJob instances.
            timeout: Timeout for job completion.
            return_exceptions: If True, exceptions are returned instead of raised.

        Returns:
            List of job results in the same order as input submitters.
        """
        jobs = await self.submit_jobs_with_rate_limit(job_submitters)
        return await self.gather_jobs(
            jobs, timeout=timeout, return_exceptions=return_exceptions
        )

    async def wait_for_any(
        self, jobs: List[AsyncJob], timeout: Optional[float] = None
    ) -> tuple[AsyncJob, Any]:
        """Wait for any job to complete and return the first result.

        Args:
            jobs: List of AsyncJob instances to monitor.
            timeout: Timeout for waiting.

        Returns:
            Tuple of (completed_job, result) for the first job to complete.

        Raises:
            asyncio.TimeoutError: If no jobs complete within timeout.
            RuntimeError: If all jobs fail.
        """
        if not jobs:
            raise ValueError("No jobs provided")

        timeout = timeout or self.config.timeout_per_job

        async def wait_with_job_info(job: AsyncJob) -> tuple[AsyncJob, Any]:
            result = await job.wait(timeout=timeout)
            return job, result

        # Create tasks properly
        created_tasks = [asyncio.create_task(wait_with_job_info(job)) for job in jobs]

        try:
            done, pending = await asyncio.wait(
                created_tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()

            if not done:
                raise asyncio.TimeoutError(
                    f"No jobs completed within {timeout} seconds"
                )

            # Get the first completed result
            completed_task = next(iter(done))
            return await completed_task

        except Exception:
            # Cancel all tasks on error
            for task in created_tasks:
                if not task.done():
                    task.cancel()
            raise

    async def batch_process_with_retry(
        self,
        job_submitters: List[Callable[[], Awaitable[AsyncJob]]],
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ) -> List[Any]:
        """Process jobs in batches with retry logic for failed operations.

        Args:
            job_submitters: List of async callables that return AsyncJob instances.
            max_retries: Maximum retry attempts. Defaults to config value.
            retry_delay: Base delay between retries. Defaults to config value.

        Returns:
            List of job results.

        Raises:
            RuntimeError: If jobs fail after all retry attempts.
        """
        max_retries = max_retries or self.config.max_retries
        retry_delay = retry_delay or self.config.retry_delay

        for attempt in range(max_retries + 1):
            try:
                return await self.submit_and_gather(
                    job_submitters, return_exceptions=False
                )
            except Exception as e:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Batch processing failed after {max_retries} retries: {e}"
                    ) from e

                # Exponential backoff
                delay = retry_delay * (2**attempt)
                await asyncio.sleep(delay)

                if self.config.enable_progress:
                    print(
                        f"Retry attempt {attempt + 1}/{max_retries} after "
                        f"{delay:.1f}s delay"
                    )


class AsyncConnectionPoolManager:
    """Manager for optimizing HTTP connection pools for high concurrency operations."""

    @staticmethod
    def create_optimized_client(
        base_url: str,
        config: Optional[AsyncConcurrentConfig] = None,
        auth: Optional[httpx.Auth] = None,
    ) -> httpx.AsyncClient:
        """Create an optimized AsyncClient for high concurrency operations.

        Args:
            base_url: Base URL for the client.
            config: Configuration with connection pool settings.
            auth: Optional authentication.

        Returns:
            Configured httpx.AsyncClient optimized for concurrent operations.
        """
        config = config or AsyncConcurrentConfig()
        pool_limits = config.connection_pool_limits

        # Create connection limits optimized for concurrency
        limits = httpx.Limits(
            max_keepalive_connections=pool_limits.get("max_keepalive_connections", 20),
            max_connections=pool_limits.get("max_connections", 100),
            keepalive_expiry=pool_limits.get("keepalive_expiry", 30.0),
        )

        # Create timeout configuration
        timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0)

        return httpx.AsyncClient(
            base_url=base_url,
            limits=limits,
            timeout=timeout,
            auth=auth,
            headers={"Accept-Encoding": "gzip, deflate", "Connection": "keep-alive"},
        )

    @staticmethod
    def get_recommended_config(concurrent_jobs: int) -> AsyncConcurrentConfig:
        """Get recommended configuration based on expected concurrent job count.

        Args:
            concurrent_jobs: Expected number of concurrent jobs.

        Returns:
            Optimized AsyncConcurrentConfig for the given concurrency level.
        """
        if concurrent_jobs <= 5:
            # Low concurrency
            return AsyncConcurrentConfig(
                max_concurrent_jobs=concurrent_jobs,
                connection_pool_limits={
                    "max_keepalive_connections": 10,
                    "max_connections": 20,
                    "keepalive_expiry": 30.0,
                },
                rate_limit_delay=0.1,
            )
        elif concurrent_jobs <= 20:
            # Medium concurrency
            return AsyncConcurrentConfig(
                max_concurrent_jobs=min(concurrent_jobs, 15),
                connection_pool_limits={
                    "max_keepalive_connections": 20,
                    "max_connections": 50,
                    "keepalive_expiry": 60.0,
                },
                rate_limit_delay=0.05,
            )
        else:
            # High concurrency
            return AsyncConcurrentConfig(
                max_concurrent_jobs=min(concurrent_jobs, 25),
                connection_pool_limits={
                    "max_keepalive_connections": 30,
                    "max_connections": 100,
                    "keepalive_expiry": 120.0,
                },
                rate_limit_delay=0.02,
            )


# Convenience functions for common use cases


async def gather_jobs(
    jobs: List[AsyncJob],
    max_concurrent: int = 5,
    timeout: Optional[float] = None,
    return_exceptions: bool = False,
) -> List[Union[Any, Exception]]:
    """Convenience function to gather results from multiple AsyncJob instances.

    This is the main utility function that users will call to await multiple
    AsyncJob instances concurrently.

    Args:
        jobs: List of AsyncJob instances to await.
        max_concurrent: Maximum number of jobs to process concurrently.
        timeout: Timeout for individual jobs.
        return_exceptions: If True, exceptions are returned instead of raised.

    Returns:
        List of job results in the same order as input jobs.

    Example:
        ```python
        # Submit multiple jobs
        job1 = await client.create_embeddings(request1, await_job_result=False)
        job2 = await client.create_embeddings(request2, await_job_result=False)
        job3 = await client.create_embeddings(request3, await_job_result=False)

        # Gather results concurrently
        results = await gather_jobs([job1, job2, job3], max_concurrent=3)
        ```
    """
    config = AsyncConcurrentConfig(max_concurrent_jobs=max_concurrent)
    if timeout:
        config.timeout_per_job = timeout

    manager = AsyncJobManager(config)
    return await manager.gather_jobs(
        jobs, timeout=timeout, return_exceptions=return_exceptions
    )


async def submit_and_gather_jobs(
    job_submitters: List[Callable[[], Awaitable[AsyncJob]]],
    max_concurrent: int = 5,
    rate_limit_delay: float = 0.1,
    timeout: Optional[float] = None,
    return_exceptions: bool = False,
) -> List[Union[Any, Exception]]:
    """Submit multiple jobs with rate limiting and gather their results.

    Args:
        job_submitters: List of async callables that return AsyncJob instances.
        max_concurrent: Maximum number of jobs to process concurrently.
        rate_limit_delay: Delay between job submissions.
        timeout: Timeout for job completion.
        return_exceptions: If True, exceptions are returned instead of raised.

    Returns:
        List of job results in the same order as input submitters.

    Example:
        ```python
        # Create job submitters
        submitters = [
            lambda: client.create_embeddings(req1, await_job_result=False),
            lambda: client.create_embeddings(req2, await_job_result=False),
            lambda: client.create_embeddings(req3, await_job_result=False),
        ]

        # Submit and gather results
        results = await submit_and_gather_jobs(submitters, max_concurrent=3)
        ```
    """
    config = AsyncConcurrentConfig(
        max_concurrent_jobs=max_concurrent, rate_limit_delay=rate_limit_delay
    )
    if timeout:
        config.timeout_per_job = timeout

    manager = AsyncJobManager(config)
    return await manager.submit_and_gather(
        job_submitters, timeout=timeout, return_exceptions=return_exceptions
    )


async def wait_for_any_job(
    jobs: List[AsyncJob], timeout: Optional[float] = None
) -> tuple[AsyncJob, Any]:
    """Wait for any job to complete and return the first result.

    Args:
        jobs: List of AsyncJob instances to monitor.
        timeout: Timeout for waiting.

    Returns:
        Tuple of (completed_job, result) for the first job to complete.

    Example:
        ```python
        # Submit multiple jobs
        jobs = [
            await client.create_embeddings(req1, await_job_result=False),
            await client.create_embeddings(req2, await_job_result=False),
            await client.create_embeddings(req3, await_job_result=False),
        ]

        # Wait for first completion
        completed_job, result = await wait_for_any_job(jobs)
        print(f"Job {completed_job.id} completed first")
        ```
    """
    manager = AsyncJobManager()
    return await manager.wait_for_any(jobs, timeout=timeout)
