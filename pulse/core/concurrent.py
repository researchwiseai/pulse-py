"""Concurrent job management for enhanced batching operations."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Optional, TypeVar, Generic
from dataclasses import dataclass

from pulse.core.jobs import Job

T = TypeVar("T")


@dataclass
class ConcurrentJobConfig:
    """Configuration for concurrent job processing."""

    max_workers: int = 5
    timeout: float = 600.0
    enable_progress: bool = True
    use_async: bool = False  # Whether to use asyncio.Semaphore or threading.Semaphore


class ConcurrentJobManager(Generic[T]):
    """Manager for controlled parallel processing of async operations in slow mode.

    Supports both async (asyncio.Semaphore) and sync (threading.Semaphore) operations
    with configurable concurrency limits for slow mode (fast=False) only.
    """

    def __init__(self, config: Optional[ConcurrentJobConfig] = None):
        """Initialize the concurrent job manager.

        Args:
            config: Configuration for concurrent processing.
                Defaults to ConcurrentJobConfig().
        """
        self.config = config or ConcurrentJobConfig()

        # Initialize appropriate semaphore based on configuration
        if self.config.use_async:
            self._async_semaphore = asyncio.Semaphore(self.config.max_workers)
            self._sync_semaphore = None
        else:
            self._sync_semaphore = threading.Semaphore(self.config.max_workers)
            self._async_semaphore = None

        self._executor: Optional[ThreadPoolExecutor] = None

    def __enter__(self):
        """Context manager entry."""
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def submit_jobs(self, job_submitters: List[Callable[[], Job]]) -> List[Job]:
        """Submit multiple jobs concurrently with semaphore-based control.

        Provides graceful error handling for individual job failures.

        Args:
            job_submitters: List of callables that return Job instances when
                called.

        Returns:
            List of submitted Job instances. Failed submissions are handled gracefully
            without affecting other jobs.

        Raises:
            RuntimeError: If called outside of context manager or if async mode
                is enabled.
        """
        if self.config.use_async:
            raise RuntimeError("Use submit_jobs_async() for async operations")

        if not self._executor:
            raise RuntimeError("ConcurrentJobManager must be used as a context manager")

        jobs = []
        futures = []
        failed_submissions = []

        try:
            # Submit all jobs
            for i, submitter in enumerate(job_submitters):
                future = self._executor.submit(
                    self._submit_with_sync_semaphore, submitter, i + 1
                )
                futures.append((future, i + 1))

            # Collect results with individual error handling
            for future, job_num in futures:
                try:
                    job = future.result()
                    jobs.append(job)
                except Exception as e:
                    failed_submissions.append(
                        {"job_number": job_num, "error": str(e), "exception": e}
                    )
                    # Continue processing other jobs instead of failing completely

            # Log failed submissions but don't raise unless all failed
            if failed_submissions:
                import warnings

                if len(failed_submissions) == len(job_submitters):
                    # All submissions failed - this is a critical error
                    raise RuntimeError(
                        f"All {len(failed_submissions)} job submissions failed. "
                        f"First error: {failed_submissions[0]['error']}"
                    )
                else:
                    # Some submissions failed - log warning and continue
                    warnings.warn(
                        f"{len(failed_submissions)} of {len(job_submitters)} "
                        f"job submissions failed. "
                        f"Continuing with {len(jobs)} successful submissions.",
                        UserWarning,
                    )

            return jobs

        except Exception:
            # Cleanup: cancel any pending futures
            for future, _ in futures:
                if not future.done():
                    future.cancel()
            raise

    def wait_for_jobs(self, jobs: List[Job]) -> List[T]:
        """Wait for multiple jobs to complete concurrently with graceful error handling.

        Args:
            jobs: List of Job instances to wait for.

        Returns:
            List of job results in the same order as input jobs. Failed jobs
            are handled gracefully without affecting other jobs.

        Raises:
            RuntimeError: If called outside of context manager or if async mode
                is enabled.
        """
        if self.config.use_async:
            raise RuntimeError("Use wait_for_jobs_async() for async operations")

        if not self._executor:
            raise RuntimeError("ConcurrentJobManager must be used as a context manager")

        # Submit wait operations with semaphore control
        futures = []
        failed_completions = []

        try:
            for i, job in enumerate(jobs):
                future = self._executor.submit(
                    self._wait_with_sync_semaphore, job, i + 1
                )
                futures.append((future, i + 1))

            # Collect results with individual error handling
            results = []
            for future, job_num in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    failed_completions.append(
                        {"job_number": job_num, "error": str(e), "exception": e}
                    )
                    results.append(None)  # Placeholder for failed job

            # Handle failed completions gracefully
            if failed_completions:
                import warnings

                if len(failed_completions) == len(jobs):
                    # All jobs failed - this is a critical error
                    raise RuntimeError(
                        f"All {len(failed_completions)} jobs failed to complete. "
                        f"First error: {failed_completions[0]['error']}"
                    )
                else:
                    # Some jobs failed - log warning and continue with successful results
                    successful_count = len(jobs) - len(failed_completions)
                    warnings.warn(
                        f"{len(failed_completions)} of {len(jobs)} "
                        f"jobs failed to complete. "
                        f"Continuing with {successful_count} successful results.",
                        UserWarning,
                    )

            return results

        except Exception:
            # Cleanup: cancel any pending futures
            for future, _ in futures:
                if not future.done():
                    future.cancel()
            raise

    def submit_and_wait(self, job_submitters: List[Callable[[], Job]]) -> List[T]:
        """Submit jobs and wait for results in a single operation.

        Args:
            job_submitters: List of callables that return Job instances when
                called.

        Returns:
            List of job results in the same order as input submitters.
        """
        jobs = self.submit_jobs(job_submitters)
        return self.wait_for_jobs(jobs)

    async def submit_jobs_async(
        self, job_submitters: List[Callable[[], Job]]
    ) -> List[Job]:
        """Submit multiple jobs concurrently with async semaphore-based control.

        Args:
            job_submitters: List of callables that return Job instances when called.

        Returns:
            List of submitted Job instances.

        Raises:
            RuntimeError: If sync mode is enabled.
        """
        if not self.config.use_async:
            raise RuntimeError("Use submit_jobs() for sync operations")

        if not self._async_semaphore:
            raise RuntimeError("Async semaphore not initialized")

        jobs = []
        failed_jobs = []

        async def submit_with_semaphore(submitter):
            async with self._async_semaphore:
                # Run the submitter in a thread pool since it's likely sync
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, submitter)

        try:
            # Submit all jobs concurrently
            tasks = [submit_with_semaphore(submitter) for submitter in job_submitters]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Separate successful jobs from exceptions
            for result in results:
                if isinstance(result, Exception):
                    failed_jobs.append(result)
                else:
                    jobs.append(result)

            # If any jobs failed to submit, raise an error with details
            if failed_jobs:
                raise RuntimeError(
                    f"Failed to submit {len(failed_jobs)} jobs: {failed_jobs}"
                )

            return jobs
        except Exception:
            # Cleanup handled by asyncio.gather
            raise

    async def wait_for_jobs_async(self, jobs: List[Job]) -> List[T]:
        """Wait for multiple jobs to complete concurrently with async semaphore control.

        Args:
            jobs: List of Job instances to wait for.

        Returns:
            List of job results in the same order as input jobs.

        Raises:
            RuntimeError: If sync mode is enabled.
        """
        if not self.config.use_async:
            raise RuntimeError("Use wait_for_jobs() for sync operations")

        if not self._async_semaphore:
            raise RuntimeError("Async semaphore not initialized")

        failed_results = []

        async def wait_with_semaphore(job):
            async with self._async_semaphore:
                # Run job.wait in a thread pool since it's likely sync
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, job.wait, self.config.timeout)

        try:
            # Wait for all jobs concurrently
            tasks = [wait_with_semaphore(job) for job in jobs]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Separate successful results from exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_results.append((i, result))
                else:
                    final_results.append(result)

            # If any jobs failed, raise an error with details
            if failed_results:
                raise RuntimeError(
                    f"Failed to complete {len(failed_results)} jobs: {failed_results}"
                )

            return final_results
        except Exception:
            # Cleanup handled by asyncio.gather
            raise

    async def submit_and_wait_async(
        self, job_submitters: List[Callable[[], Job]]
    ) -> List[T]:
        """Submit jobs and wait for results in a single async operation.

        Args:
            job_submitters: List of callables that return Job instances when called.

        Returns:
            List of job results in the same order as input submitters.
        """
        jobs = await self.submit_jobs_async(job_submitters)
        return await self.wait_for_jobs_async(jobs)

    def _submit_with_sync_semaphore(
        self, submitter: Callable[[], Job], job_num: int = 0
    ) -> Job:
        """Submit a job with sync semaphore-based concurrency control and error context."""
        with self._sync_semaphore:
            try:
                return submitter()
            except Exception as e:
                # Add job context to the error
                raise RuntimeError(f"Job {job_num} submission failed: {e}") from e

    def _wait_with_sync_semaphore(self, job: Job, job_num: int = 0) -> T:
        """Wait for a job with sync semaphore-based concurrency control and error context."""
        with self._sync_semaphore:
            try:
                return job.wait(timeout=self.config.timeout)
            except Exception as e:
                # Add job context to the error
                raise RuntimeError(f"Job {job_num} completion failed: {e}") from e


class AsyncJobProcessor:
    """Utility class for processing jobs with async-like patterns using threads."""

    @staticmethod
    def create_job_submitter(client_method: Callable, **kwargs) -> Callable[[], Job]:
        """Create a job submitter function for a client method.

        Args:
            client_method: The client method to call
                (e.g., client._submit_batch_similarity_job).
            **kwargs: Arguments to pass to the client method.

        Returns:
            A callable that returns a Job when invoked.
        """

        def submitter() -> Job:
            try:
                return client_method(**kwargs)
            except Exception as e:
                # Wrap exceptions to provide better error context
                raise RuntimeError(
                    f"Failed to submit job for {client_method.__name__}: {e}"
                ) from e

        return submitter

    @staticmethod
    def batch_process_with_concurrency(
        job_submitters: List[Callable[[], Job]],
        config: Optional[ConcurrentJobConfig] = None,
        feature: str = "unknown",
    ) -> List[Any]:
        """Process a batch of jobs with controlled concurrency and detailed error handling.

        Args:
            job_submitters: List of job submitter functions.
            config: Optional configuration for concurrent processing.
            feature: Name of the feature being processed (for error reporting).

        Returns:
            List of job results.

        Raises:
            BatchingError: If job processing fails with detailed error information.
        """
        from pulse.core.validation import BatchingErrorHelper

        total_jobs = len(job_submitters)

        try:
            with ConcurrentJobManager(config) as manager:
                # Add progress tracking if enabled
                if config and config.enable_progress and total_jobs > 1:
                    try:
                        from tqdm import tqdm

                        progress_bar = tqdm(
                            total=total_jobs,
                            desc=f"Processing {feature} batches",
                            unit="batch",
                        )
                    except ImportError:
                        progress_bar = None
                else:
                    progress_bar = None

                try:
                    # Submit jobs with progress tracking
                    jobs = manager.submit_jobs(job_submitters)
                    if progress_bar:
                        progress_bar.set_description(
                            f"Submitted {len(jobs)} {feature} jobs"
                        )

                    # Wait for results with progress updates
                    results = []
                    failed_jobs = []

                    for i, job in enumerate(jobs):
                        try:
                            result = job.wait(
                                timeout=config.timeout if config else 600.0
                            )
                            results.append(result)
                            if progress_bar:
                                progress_bar.update(1)
                                progress_bar.set_postfix(
                                    {
                                        "completed": len(results),
                                        "failed": len(failed_jobs),
                                    }
                                )
                        except Exception as e:
                            failed_jobs.append((i + 1, str(e)))
                            if progress_bar:
                                progress_bar.update(1)
                                progress_bar.set_postfix(
                                    {
                                        "completed": len(results),
                                        "failed": len(failed_jobs),
                                    }
                                )

                    if progress_bar:
                        progress_bar.close()

                    # Handle partial failures gracefully
                    if failed_jobs and len(results) == 0:
                        # All jobs failed - raise error for first failure
                        first_failure = failed_jobs[0]
                        error = BatchingErrorHelper.create_batch_failure_error(
                            feature=feature,
                            batch_num=first_failure[0],
                            total_batches=total_jobs,
                            error_message=first_failure[1],
                            input_count=total_jobs,
                        )
                        error.add_context("total_failures", len(failed_jobs))
                        error.add_context("processing_mode", "concurrent")
                        raise error
                    elif failed_jobs:
                        # Some jobs failed - log warnings but continue with successful results
                        import warnings

                        warnings.warn(
                            f"{len(failed_jobs)} of {total_jobs} {feature} "
                            f"batches failed. "
                            f"Continuing with {len(results)} successful results.",
                            UserWarning,
                        )
                        # Fill in None for failed jobs to maintain order
                        final_results = []
                        result_idx = 0
                        failed_indices = {
                            f[0] - 1 for f in failed_jobs
                        }  # Convert to 0-based

                        for i in range(total_jobs):
                            if i in failed_indices:
                                final_results.append(None)
                            else:
                                final_results.append(results[result_idx])
                                result_idx += 1

                        return final_results

                    return results

                finally:
                    if progress_bar:
                        progress_bar.close()

        except Exception as e:
            # Provide enhanced error context for batch processing failures
            if hasattr(e, "error_code"):  # Already a BatchingError
                raise

            error = BatchingErrorHelper.create_batch_failure_error(
                feature=feature,
                batch_num=0,  # Unknown which batch failed
                total_batches=total_jobs,
                error_message=str(e),
                input_count=total_jobs,
            )
            error.add_context("processing_mode", "concurrent")
            error.add_context("max_workers", config.max_workers if config else 5)
            raise error

    @staticmethod
    async def batch_process_with_concurrency_async(
        job_submitters: List[Callable[[], Job]],
        config: Optional[ConcurrentJobConfig] = None,
    ) -> List[Any]:
        """Process a batch of jobs with controlled async concurrency.

        Args:
            job_submitters: List of job submitter functions.
            config: Optional configuration for concurrent processing.

        Returns:
            List of job results.

        Raises:
            RuntimeError: If job processing fails with detailed error information.
        """
        if config is None:
            config = ConcurrentJobConfig(use_async=True)
        else:
            config.use_async = True

        try:
            # Note: ConcurrentJobManager doesn't support async context manager yet
            # For now, we'll use the sync version but this could be enhanced
            manager = ConcurrentJobManager(config)
            return await manager.submit_and_wait_async(job_submitters)
        except Exception as e:
            # Provide enhanced error context for batch processing failures
            raise RuntimeError(
                f"Async batch processing failed with {len(job_submitters)} jobs: {e}"
            ) from e
