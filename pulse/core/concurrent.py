"""Concurrent job management for enhanced batching operations."""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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


class ConcurrentJobManager(Generic[T]):
    """Manager for controlled parallel processing of async operations in slow mode."""

    def __init__(self, config: Optional[ConcurrentJobConfig] = None):
        """Initialize the concurrent job manager.

        Args:
            config: Configuration for concurrent processing.
                Defaults to ConcurrentJobConfig().
        """
        self.config = config or ConcurrentJobConfig()
        self._semaphore = threading.Semaphore(self.config.max_workers)
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

        Args:
            job_submitters: List of callables that return Job instances when
                called.

        Returns:
            List of submitted Job instances.

        Raises:
            RuntimeError: If called outside of context manager.
        """
        if not self._executor:
            raise RuntimeError("ConcurrentJobManager must be used as a context manager")

        jobs = []
        futures = []

        for submitter in job_submitters:
            future = self._executor.submit(self._submit_with_semaphore, submitter)
            futures.append(future)

        # Collect all submitted jobs
        for future in as_completed(futures):
            job = future.result()
            jobs.append(job)

        return jobs

    def wait_for_jobs(self, jobs: List[Job]) -> List[T]:
        """Wait for multiple jobs to complete concurrently with controlled
        parallelism.

        Args:
            jobs: List of Job instances to wait for.

        Returns:
            List of job results in the same order as input jobs.

        Raises:
            RuntimeError: If called outside of context manager.
        """
        if not self._executor:
            raise RuntimeError("ConcurrentJobManager must be used as a context manager")

        # Submit wait operations with semaphore control
        futures = []
        for job in jobs:
            future = self._executor.submit(self._wait_with_semaphore, job)
            futures.append(future)

        # Collect results in order
        results = []
        for future in futures:
            result = future.result()
            results.append(result)

        return results

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

    def _submit_with_semaphore(self, submitter: Callable[[], Job]) -> Job:
        """Submit a job with semaphore-based concurrency control."""
        with self._semaphore:
            return submitter()

    def _wait_with_semaphore(self, job: Job) -> T:
        """Wait for a job with semaphore-based concurrency control."""
        with self._semaphore:
            return job.wait(timeout=self.config.timeout)


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
            return client_method(**kwargs)

        return submitter

    @staticmethod
    def batch_process_with_concurrency(
        job_submitters: List[Callable[[], Job]],
        config: Optional[ConcurrentJobConfig] = None,
    ) -> List[Any]:
        """Process a batch of jobs with controlled concurrency.

        Args:
            job_submitters: List of job submitter functions.
            config: Optional configuration for concurrent processing.

        Returns:
            List of job results.
        """
        with ConcurrentJobManager(config) as manager:
            return manager.submit_and_wait(job_submitters)
