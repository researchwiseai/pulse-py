"""Tests for async concurrent processing utilities."""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock
from pulse.core.async_concurrent import (
    AsyncConcurrentConfig,
    AsyncJobManager,
    AsyncConnectionPoolManager,
    gather_jobs,
    submit_and_gather_jobs,
    wait_for_any_job,
)
from pulse.core.async_jobs import AsyncJob
import httpx


def create_mock_job(job_id: str, wait_result=None, wait_side_effect=None):
    """Helper to create a properly mocked AsyncJob."""
    job = AsyncMock(spec=AsyncJob)
    job.id = job_id
    job.status = "pending"

    if wait_side_effect:
        job.wait = AsyncMock(side_effect=wait_side_effect)
    else:
        job.wait = AsyncMock(return_value=wait_result or {"result": f"result-{job_id}"})

    return job


class TestAsyncConcurrentConfig:
    """Test AsyncConcurrentConfig configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AsyncConcurrentConfig()

        assert config.max_concurrent_jobs == 5
        assert config.timeout_per_job == 180.0
        assert config.rate_limit_delay == 0.1
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.enable_progress is True
        assert "max_keepalive_connections" in config.connection_pool_limits
        assert config.connection_pool_limits["max_keepalive_connections"] == 20

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AsyncConcurrentConfig(
            max_concurrent_jobs=10,
            timeout_per_job=300.0,
            rate_limit_delay=0.05,
            connection_pool_limits={"max_connections": 50},
        )

        assert config.max_concurrent_jobs == 10
        assert config.timeout_per_job == 300.0
        assert config.rate_limit_delay == 0.05
        assert config.connection_pool_limits["max_connections"] == 50


class TestAsyncJobManager:
    """Test AsyncJobManager for concurrent job processing."""

    @pytest.fixture
    def manager(self):
        """Create AsyncJobManager instance for testing."""
        config = AsyncConcurrentConfig(max_concurrent_jobs=3, timeout_per_job=5.0)
        return AsyncJobManager(config)

    @pytest.fixture
    def mock_jobs(self):
        """Create mock AsyncJob instances."""
        jobs = []
        for i in range(3):
            job = create_mock_job(f"job-{i}", wait_result={"result": f"result-{i}"})
            jobs.append(job)
        return jobs

    @pytest.mark.asyncio
    async def test_gather_jobs_success(self, manager, mock_jobs):
        """Test successful gathering of multiple jobs."""
        results = await manager.gather_jobs(mock_jobs)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["result"] == f"result-{i}"

    @pytest.mark.asyncio
    async def test_gather_jobs_empty_list(self, manager):
        """Test gathering empty job list."""
        results = await manager.gather_jobs([])
        assert results == []

    @pytest.mark.asyncio
    async def test_gather_jobs_with_timeout(self, manager):
        """Test gathering jobs with timeout."""

        # Create a job that takes longer than timeout
        async def slow_wait(timeout=None):
            await asyncio.sleep(10)  # Longer than test timeout
            return {"result": "slow"}

        slow_job = create_mock_job("slow-job")
        slow_job.wait = slow_wait

        with pytest.raises(asyncio.TimeoutError):
            await manager.gather_jobs([slow_job], timeout=0.1)

    @pytest.mark.asyncio
    async def test_gather_jobs_with_exceptions(self, manager):
        """Test gathering jobs with return_exceptions=True."""
        # Create jobs with mixed success/failure
        jobs = []

        # Successful job
        success_job = create_mock_job("success", wait_result={"result": "success"})
        jobs.append(success_job)

        # Failing job
        fail_job = create_mock_job("fail", wait_side_effect=RuntimeError("Job failed"))
        jobs.append(fail_job)

        results = await manager.gather_jobs(jobs, return_exceptions=True)

        assert len(results) == 2
        assert results[0]["result"] == "success"
        assert isinstance(results[1], RuntimeError)
        assert str(results[1]) == "Job failed"

    @pytest.mark.asyncio
    async def test_submit_jobs_with_rate_limit(self, manager):
        """Test job submission with rate limiting."""
        submission_times = []

        async def mock_submitter():
            submission_times.append(time.time())
            job = create_mock_job(f"job-{len(submission_times)}")
            return job

        submitters = [mock_submitter for _ in range(3)]

        time.time()
        jobs = await manager.submit_jobs_with_rate_limit(submitters)

        assert len(jobs) == 3
        assert len(submission_times) == 3

        # Check that rate limiting was applied (submissions should be spaced)
        for i in range(1, len(submission_times)):
            time_diff = submission_times[i] - submission_times[i - 1]
            assert (
                time_diff >= manager.config.rate_limit_delay * 0.9
            )  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_submit_and_gather(self, manager):
        """Test combined submit and gather operation."""

        async def mock_submitter(result_value):
            job = create_mock_job(
                f"job-{result_value}", wait_result={"result": result_value}
            )
            return job

        submitters = [
            lambda: mock_submitter("a"),
            lambda: mock_submitter("b"),
            lambda: mock_submitter("c"),
        ]

        results = await manager.submit_and_gather(submitters)

        assert len(results) == 3
        assert results[0]["result"] == "a"
        assert results[1]["result"] == "b"
        assert results[2]["result"] == "c"

    @pytest.mark.asyncio
    async def test_wait_for_any(self, manager):
        """Test waiting for any job to complete."""
        jobs = []

        # Create jobs with different completion times
        for i, delay in enumerate([0.3, 0.1, 0.2]):  # Second job completes first
            job = create_mock_job(f"job-{i}")

            # Create async wait function with proper closure
            async def make_wait_func(d, result):
                async def wait_func(timeout=None):
                    await asyncio.sleep(d)
                    return {"result": result, "delay": d}

                return wait_func

            # Set the wait function directly
            job.wait = await make_wait_func(delay, f"result-{i}")
            jobs.append(job)

        completed_job, result = await manager.wait_for_any(jobs)

        # The job with 0.1 delay should complete first
        assert result["delay"] == 0.1
        assert result["result"] == "result-1"
        assert completed_job.id == "job-1"

    @pytest.mark.asyncio
    async def test_batch_process_with_retry_success(self, manager):
        """Test batch processing with retry on first attempt success."""

        async def mock_submitter(result_value):
            job = create_mock_job(
                f"job-{result_value}", wait_result={"result": result_value}
            )
            return job

        submitters = [lambda: mock_submitter("test")]

        results = await manager.batch_process_with_retry(submitters)

        assert len(results) == 1
        assert results[0]["result"] == "test"

    @pytest.mark.asyncio
    async def test_batch_process_with_retry_failure(self, manager):
        """Test batch processing with retry after failures."""
        attempt_count = 0

        async def failing_submitter():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:  # Fail first 2 attempts
                raise RuntimeError(f"Attempt {attempt_count} failed")

            # Succeed on 3rd attempt
            job = create_mock_job("success-job", wait_result={"result": "success"})
            return job

        submitters = [failing_submitter]

        # Should succeed after retries
        results = await manager.batch_process_with_retry(submitters, max_retries=3)

        assert len(results) == 1
        assert results[0]["result"] == "success"
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_batch_process_with_retry_max_retries_exceeded(self, manager):
        """Test batch processing failure after max retries."""

        async def always_failing_submitter():
            raise RuntimeError("Always fails")

        submitters = [always_failing_submitter]

        with pytest.raises(
            RuntimeError, match="Batch processing failed after 2 retries"
        ):
            await manager.batch_process_with_retry(submitters, max_retries=2)


class TestAsyncConnectionPoolManager:
    """Test AsyncConnectionPoolManager for connection optimization."""

    def test_create_optimized_client(self):
        """Test creation of optimized async client."""
        client = AsyncConnectionPoolManager.create_optimized_client(
            "https://api.example.com"
        )

        assert isinstance(client, httpx.AsyncClient)
        assert str(client.base_url) == "https://api.example.com"
        # Check that limits were set (httpx stores them internally)
        assert hasattr(client, "_limits")

    def test_create_optimized_client_with_config(self):
        """Test creation of optimized client with custom config."""
        config = AsyncConcurrentConfig(
            connection_pool_limits={
                "max_connections": 50,
                "max_keepalive_connections": 15,
            }
        )

        client = AsyncConnectionPoolManager.create_optimized_client(
            "https://api.example.com", config=config
        )

        # Check that client was created successfully with config
        assert isinstance(client, httpx.AsyncClient)
        assert hasattr(client, "_limits")

    def test_get_recommended_config_low_concurrency(self):
        """Test recommended config for low concurrency."""
        config = AsyncConnectionPoolManager.get_recommended_config(3)

        assert config.max_concurrent_jobs == 3
        assert config.connection_pool_limits["max_connections"] == 20
        assert config.rate_limit_delay == 0.1

    def test_get_recommended_config_medium_concurrency(self):
        """Test recommended config for medium concurrency."""
        config = AsyncConnectionPoolManager.get_recommended_config(15)

        assert config.max_concurrent_jobs == 15
        assert config.connection_pool_limits["max_connections"] == 50
        assert config.rate_limit_delay == 0.05

    def test_get_recommended_config_high_concurrency(self):
        """Test recommended config for high concurrency."""
        config = AsyncConnectionPoolManager.get_recommended_config(30)

        assert config.max_concurrent_jobs == 25  # Capped at 25
        assert config.connection_pool_limits["max_connections"] == 100
        assert config.rate_limit_delay == 0.02


class TestConvenienceFunctions:
    """Test convenience functions for common use cases."""

    @pytest.fixture
    def mock_jobs_convenience(self):
        """Create mock AsyncJob instances for convenience function tests."""
        jobs = []
        for i in range(3):
            job = create_mock_job(f"job-{i}", wait_result={"result": f"result-{i}"})
            jobs.append(job)
        return jobs

    @pytest.mark.asyncio
    async def test_gather_jobs_convenience(self, mock_jobs_convenience):
        """Test the convenience gather_jobs function."""
        results = await gather_jobs(mock_jobs_convenience, max_concurrent=2)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["result"] == f"result-{i}"

    @pytest.mark.asyncio
    async def test_submit_and_gather_jobs_convenience(self):
        """Test the convenience submit_and_gather_jobs function."""

        async def mock_submitter(result_value):
            job = create_mock_job(
                f"job-{result_value}", wait_result={"result": result_value}
            )
            return job

        submitters = [lambda: mock_submitter("x"), lambda: mock_submitter("y")]

        results = await submit_and_gather_jobs(submitters, max_concurrent=2)

        assert len(results) == 2
        assert results[0]["result"] == "x"
        assert results[1]["result"] == "y"

    @pytest.mark.asyncio
    async def test_wait_for_any_job_convenience(self):
        """Test the convenience wait_for_any_job function."""
        jobs = []

        # Create jobs with different completion times
        for i, delay in enumerate([0.2, 0.05, 0.1]):  # Second job completes first
            job = create_mock_job(f"job-{i}")

            async def make_wait_func(d, result):
                async def wait_func(timeout=None):
                    await asyncio.sleep(d)
                    return {"result": result, "delay": d}

                return wait_func

            job.wait = await make_wait_func(delay, f"result-{i}")
            jobs.append(job)

        completed_job, result = await wait_for_any_job(jobs)

        # The job with 0.05 delay should complete first
        assert result["delay"] == 0.05
        assert result["result"] == "result-1"
        assert completed_job.id == "job-1"


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_large_batch_processing(self):
        """Test processing a large batch of jobs."""
        job_count = 20

        async def create_job_submitter(job_id):
            async def submitter():
                # Simulate variable processing time
                delay = 0.01 + (job_id % 3) * 0.01
                job = create_mock_job(
                    f"batch-job-{job_id}",
                    wait_result={
                        "result": f"batch-result-{job_id}",
                        "processing_time": delay,
                    },
                )
                return job

            return submitter

        submitters = [await create_job_submitter(i) for i in range(job_count)]

        # Process with controlled concurrency
        results = await submit_and_gather_jobs(
            submitters, max_concurrent=5, rate_limit_delay=0.01
        )

        assert len(results) == job_count
        for i, result in enumerate(results):
            assert result["result"] == f"batch-result-{i}"

    @pytest.mark.asyncio
    async def test_mixed_success_failure_scenario(self):
        """Test scenario with mixed successful and failing jobs."""

        async def create_mixed_submitter(job_id, should_fail=False):
            async def submitter():
                if should_fail:
                    job = create_mock_job(
                        f"mixed-job-{job_id}",
                        wait_side_effect=RuntimeError(f"Job {job_id} failed"),
                    )
                else:
                    job = create_mock_job(
                        f"mixed-job-{job_id}",
                        wait_result={"result": f"success-{job_id}"},
                    )
                return job

            return submitter

        # Create mix of successful and failing jobs
        submitters = []
        for i in range(10):
            should_fail = i % 3 == 0  # Every 3rd job fails
            submitters.append(await create_mixed_submitter(i, should_fail))

        # Process with exception handling
        results = await submit_and_gather_jobs(
            submitters, max_concurrent=3, return_exceptions=True
        )

        assert len(results) == 10

        # Check that we have both successes and failures
        successes = [r for r in results if isinstance(r, dict)]
        failures = [r for r in results if isinstance(r, Exception)]

        assert len(successes) == 7  # Jobs 1,2,4,5,7,8 succeed
        assert len(failures) == 3  # Jobs 0,3,6,9 fail

    @pytest.mark.asyncio
    async def test_timeout_handling_scenario(self):
        """Test timeout handling in realistic scenario."""

        async def create_timed_submitter(job_id, processing_time):
            async def submitter():
                job = create_mock_job(f"timed-job-{job_id}")

                async def timed_wait(timeout=None):
                    await asyncio.sleep(processing_time)
                    return {"result": f"completed-{job_id}", "time": processing_time}

                job.wait = timed_wait
                return job

            return submitter

        # Create jobs with different processing times
        submitters = [
            await create_timed_submitter(0, 0.05),  # Fast
            await create_timed_submitter(1, 0.1),  # Medium
            await create_timed_submitter(2, 0.5),  # Slow (will timeout)
        ]

        # Process with short timeout
        results = await submit_and_gather_jobs(
            submitters,
            max_concurrent=3,
            timeout=0.2,  # Short timeout
            return_exceptions=True,
        )

        assert len(results) == 3

        # First two should succeed, third should timeout
        assert results[0]["result"] == "completed-0"
        assert results[1]["result"] == "completed-1"
        assert isinstance(results[2], asyncio.TimeoutError)
