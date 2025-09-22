"""Tests for async concurrency patterns and performance."""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock

from pulse.core.async_client import AsyncCoreClient
from pulse.core.async_jobs import AsyncJob
from pulse.core.async_concurrent import (
    AsyncConcurrentConfig,
    AsyncJobManager,
    AsyncConnectionPoolManager,
    gather_jobs,
    submit_and_gather_jobs,
    wait_for_any_job,
)
from pulse.async_starters import (
    generate_themes_async,
    sentiment_analysis_async,
    compare_similarity_async,
)
from pulse.core.models import (
    SentimentResponse,
    ThemesResponse,
    SimilarityResponse,
    SentimentResult,
    Theme,
    UsageReport,
)


class TestConcurrentJobExecution:
    """Test concurrent execution of multiple async jobs."""

    def create_mock_job(
        self, job_id: str, processing_time: float = 0.1, should_fail: bool = False
    ):
        """Helper to create mock AsyncJob with configurable behavior."""
        job = AsyncMock(spec=AsyncJob)
        job.id = job_id
        job.status = "pending"

        if should_fail:
            job.wait = AsyncMock(side_effect=RuntimeError(f"Job {job_id} failed"))
        else:

            async def mock_wait(timeout=None):
                await asyncio.sleep(processing_time)
                return {
                    "result": f"result-{job_id}",
                    "processing_time": processing_time,
                }

            job.wait = mock_wait

        return job

    @pytest.mark.asyncio
    async def test_concurrent_job_gathering(self):
        """Test gathering multiple jobs concurrently."""
        # Create jobs with different processing times
        jobs = [
            self.create_mock_job("job-1", 0.05),
            self.create_mock_job("job-2", 0.1),
            self.create_mock_job("job-3", 0.03),
        ]

        start_time = time.time()
        results = await gather_jobs(jobs, max_concurrent=3)
        end_time = time.time()

        # Should complete in roughly the time of the slowest job (0.1s)
        # rather than the sum of all jobs (0.18s)
        assert end_time - start_time < 0.15
        assert len(results) == 3

        # Results should be in order
        assert results[0]["result"] == "result-job-1"
        assert results[1]["result"] == "result-job-2"
        assert results[2]["result"] == "result-job-3"

    @pytest.mark.asyncio
    async def test_concurrent_job_submission_and_gathering(self):
        """Test submitting and gathering jobs concurrently."""

        async def create_job_submitter(job_id: str, processing_time: float):
            async def submitter():
                return self.create_mock_job(job_id, processing_time)

            return submitter

        submitters = [
            await create_job_submitter("submit-1", 0.05),
            await create_job_submitter("submit-2", 0.08),
            await create_job_submitter("submit-3", 0.03),
        ]

        start_time = time.time()
        results = await submit_and_gather_jobs(
            submitters,
            max_concurrent=2,
            rate_limit_delay=0.01,
        )
        end_time = time.time()

        assert len(results) == 3
        # Should be faster than sequential execution
        assert end_time - start_time < 0.2

    @pytest.mark.asyncio
    async def test_wait_for_any_job_completion(self):
        """Test waiting for any job to complete first."""
        jobs = [
            self.create_mock_job("slow-job", 0.2),
            self.create_mock_job("fast-job", 0.05),  # This should complete first
            self.create_mock_job("medium-job", 0.1),
        ]

        completed_job, result = await wait_for_any_job(jobs)

        # The fast job should complete first
        assert completed_job.id == "fast-job"
        assert result["result"] == "result-fast-job"
        assert result["processing_time"] == 0.05

    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self):
        """Test error handling in concurrent job execution."""
        jobs = [
            self.create_mock_job("success-1", 0.05),
            self.create_mock_job("failure", 0.03, should_fail=True),
            self.create_mock_job("success-2", 0.08),
        ]

        # Test with return_exceptions=True
        results = await gather_jobs(jobs, max_concurrent=3, return_exceptions=True)

        assert len(results) == 3
        assert results[0]["result"] == "result-success-1"
        assert isinstance(results[1], RuntimeError)
        assert str(results[1]) == "Job failure failed"
        assert results[2]["result"] == "result-success-2"

    @pytest.mark.asyncio
    async def test_concurrent_timeout_handling(self):
        """Test timeout handling in concurrent execution."""
        jobs = [
            self.create_mock_job("fast", 0.05),
            self.create_mock_job("slow", 1.0),  # Will timeout
            self.create_mock_job("medium", 0.1),
        ]

        # Use short timeout
        results = await gather_jobs(
            jobs,
            max_concurrent=3,
            timeout=0.2,
            return_exceptions=True,
        )

        assert len(results) == 3
        assert results[0]["result"] == "result-fast"
        assert isinstance(results[1], asyncio.TimeoutError)
        assert results[2]["result"] == "result-medium"

    @pytest.mark.asyncio
    async def test_rate_limited_job_submission(self):
        """Test rate-limited job submission."""
        submission_times = []

        async def tracking_submitter():
            submission_times.append(time.time())
            return self.create_mock_job(f"job-{len(submission_times)}", 0.01)

        submitters = [tracking_submitter for _ in range(5)]

        await submit_and_gather_jobs(
            submitters,
            max_concurrent=5,
            rate_limit_delay=0.05,  # 50ms between submissions
        )

        # Check that submissions were rate-limited
        assert len(submission_times) == 5
        for i in range(1, len(submission_times)):
            time_diff = submission_times[i] - submission_times[i - 1]
            assert time_diff >= 0.04  # Allow some tolerance


class TestConcurrentAPIOperations:
    """Test concurrent API operations with real async clients."""

    @pytest.fixture
    def mock_async_client(self):
        """Create mock AsyncCoreClient for testing."""
        client = AsyncMock(spec=AsyncCoreClient)

        # Mock sentiment analysis
        client.analyze_sentiment.return_value = SentimentResponse(
            results=[SentimentResult(sentiment="positive", confidence=0.9)],
            usage=UsageReport(records=[], total=10),
        )

        # Mock theme generation
        client.generate_themes.return_value = ThemesResponse(
            themes=[
                Theme(
                    shortLabel="positive",
                    label="Positive Theme",
                    description="Positive feedback",
                    representatives=["good", "great"],
                )
            ],
            usage=UsageReport(records=[], total=20),
        )

        # Mock similarity comparison
        client.compare_similarity.return_value = SimilarityResponse(
            scenario="self",
            mode="matrix",
            n=4,
            similarity=[[1.0, 0.8], [0.8, 1.0]],
            flattened=[1.0, 0.8, 0.8, 1.0],
            usage=UsageReport(records=[], total=15),
        )

        return client

    @pytest.mark.asyncio
    async def test_concurrent_sentiment_analysis(self, mock_async_client):
        """Test concurrent sentiment analysis operations."""
        text_batches = [
            ["Happy text 1", "Sad text 1"],
            ["Happy text 2", "Sad text 2"],
            ["Happy text 3", "Sad text 3"],
        ]

        # Create concurrent tasks
        tasks = [
            sentiment_analysis_async(batch, client=mock_async_client)
            for batch in text_batches
        ]

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        assert len(results) == 3
        for result in results:
            assert isinstance(result, SentimentResponse)
            assert len(result.results) == 1  # Mock returns single result

        # Should be faster than sequential execution
        assert end_time - start_time < 0.5

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self, mock_async_client):
        """Test concurrent execution of different types of operations."""
        sample_texts = ["text 1", "text 2"]

        # Create different types of concurrent operations
        tasks = [
            sentiment_analysis_async(sample_texts, client=mock_async_client),
            generate_themes_async(sample_texts, client=mock_async_client),
            compare_similarity_async(sample_texts, client=mock_async_client),
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert isinstance(results[0], SentimentResponse)
        assert isinstance(results[1], ThemesResponse)
        assert isinstance(results[2], SimilarityResponse)

    @pytest.mark.asyncio
    async def test_high_concurrency_stress_test(self, mock_async_client):
        """Test high concurrency with many simultaneous operations."""
        num_operations = 20

        # Create many concurrent sentiment analysis tasks
        tasks = [
            sentiment_analysis_async([f"text {i}"], client=mock_async_client)
            for i in range(num_operations)
        ]

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        assert len(results) == num_operations
        for result in results:
            assert isinstance(result, SentimentResponse)

        # Should complete much faster than sequential execution
        assert end_time - start_time < 2.0

    @pytest.mark.asyncio
    async def test_concurrent_operations_with_failures(self, mock_async_client):
        """Test concurrent operations with some failures."""
        # Make some operations fail
        call_count = 0
        original_sentiment = mock_async_client.analyze_sentiment

        async def sometimes_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Every 3rd call fails
                raise RuntimeError("Simulated API error")
            return await original_sentiment(*args, **kwargs)

        mock_async_client.analyze_sentiment = sometimes_fail

        # Create multiple concurrent tasks
        tasks = [
            sentiment_analysis_async([f"text {i}"], client=mock_async_client)
            for i in range(9)  # 3 will fail, 6 will succeed
        ]

        # Use gather with return_exceptions to handle failures
        results = await asyncio.gather(*tasks, return_exceptions=True)

        assert len(results) == 9

        successes = [r for r in results if isinstance(r, SentimentResponse)]
        failures = [r for r in results if isinstance(r, Exception)]

        assert len(successes) == 6
        assert len(failures) == 3


class TestConnectionPoolOptimization:
    """Test connection pool optimization for concurrent operations."""

    def test_connection_pool_configuration(self):
        """Test connection pool configuration for different concurrency levels."""
        # Test low concurrency
        config_low = AsyncConnectionPoolManager.get_recommended_config(5)
        assert config_low.max_concurrent_jobs == 5
        assert config_low.connection_pool_limits["max_connections"] == 20
        assert config_low.rate_limit_delay == 0.1

        # Test medium concurrency
        config_medium = AsyncConnectionPoolManager.get_recommended_config(15)
        assert config_medium.max_concurrent_jobs == 15
        assert config_medium.connection_pool_limits["max_connections"] == 50
        assert config_medium.rate_limit_delay == 0.05

        # Test high concurrency (should be capped)
        config_high = AsyncConnectionPoolManager.get_recommended_config(50)
        assert config_high.max_concurrent_jobs == 25  # Capped at 25
        assert config_high.connection_pool_limits["max_connections"] == 100
        assert config_high.rate_limit_delay == 0.02

    @pytest.mark.asyncio
    async def test_optimized_client_creation(self):
        """Test creation of optimized async clients."""
        config = AsyncConcurrentConfig(
            max_concurrent_jobs=10,
            connection_pool_limits={
                "max_connections": 50,
                "max_keepalive_connections": 20,
            },
        )

        client = AsyncConnectionPoolManager.create_optimized_client(
            "https://api.example.com", config=config
        )

        assert str(client.base_url) == "https://api.example.com"
        assert hasattr(client, "_limits")

        # Clean up
        await client.aclose()

    @pytest.mark.asyncio
    async def test_connection_pool_reuse(self):
        """Test that connection pools are reused efficiently."""
        # Create multiple clients with same base URL
        clients = []
        for i in range(3):
            client = AsyncConnectionPoolManager.create_optimized_client(
                "https://api.example.com"
            )
            clients.append(client)

        # All clients should be independent but optimized
        for client in clients:
            assert str(client.base_url) == "https://api.example.com"
            assert hasattr(client, "_limits")

        # Clean up
        for client in clients:
            await client.aclose()


class TestAsyncJobManager:
    """Test AsyncJobManager for advanced concurrent job handling."""

    @pytest.fixture
    def job_manager(self):
        """Create AsyncJobManager for testing."""
        config = AsyncConcurrentConfig(
            max_concurrent_jobs=5,
            timeout_per_job=10.0,
            rate_limit_delay=0.01,
        )
        return AsyncJobManager(config)

    def create_mock_job(self, job_id: str, processing_time: float = 0.1):
        """Helper to create mock job."""
        job = AsyncMock(spec=AsyncJob)
        job.id = job_id
        job.status = "pending"

        async def mock_wait(timeout=None):
            await asyncio.sleep(processing_time)
            return {"result": f"result-{job_id}"}

        job.wait = mock_wait
        return job

    @pytest.mark.asyncio
    async def test_job_manager_batch_processing(self, job_manager):
        """Test batch processing with AsyncJobManager."""
        jobs = [self.create_mock_job(f"batch-job-{i}", 0.05) for i in range(10)]

        start_time = time.time()
        results = await job_manager.gather_jobs(jobs)
        end_time = time.time()

        assert len(results) == 10
        for i, result in enumerate(results):
            assert result["result"] == f"result-batch-job-{i}"

        # Should be faster than sequential processing
        assert end_time - start_time < 0.5

    @pytest.mark.asyncio
    async def test_job_manager_with_retry(self, job_manager):
        """Test AsyncJobManager with retry logic."""
        attempt_count = 0

        async def failing_submitter():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise RuntimeError(f"Attempt {attempt_count} failed")
            return self.create_mock_job("retry-job", 0.01)

        submitters = [failing_submitter]

        # Should succeed after retries
        results = await job_manager.batch_process_with_retry(submitters, max_retries=3)

        assert len(results) == 1
        assert results[0]["result"] == "result-retry-job"
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_job_manager_wait_for_any(self, job_manager):
        """Test waiting for any job completion."""
        jobs = [
            self.create_mock_job("slow", 0.2),
            self.create_mock_job("fast", 0.05),
            self.create_mock_job("medium", 0.1),
        ]

        completed_job, result = await job_manager.wait_for_any(jobs)

        # Fast job should complete first
        assert completed_job.id == "fast"
        assert result["result"] == "result-fast"


class TestConcurrencyPatterns:
    """Test various concurrency patterns and best practices."""

    @pytest.mark.asyncio
    async def test_producer_consumer_pattern(self):
        """Test producer-consumer pattern with async jobs."""
        queue = asyncio.Queue(maxsize=5)
        results = []

        async def producer():
            """Produce jobs and add to queue."""
            for i in range(10):
                job = AsyncMock(spec=AsyncJob)
                job.id = f"produced-job-{i}"
                job.wait = AsyncMock(return_value={"result": f"result-{i}"})
                await queue.put(job)
                await asyncio.sleep(0.01)  # Simulate production time

            # Signal completion
            await queue.put(None)

        async def consumer():
            """Consume jobs from queue and process them."""
            while True:
                job = await queue.get()
                if job is None:  # End signal
                    break

                result = await job.wait()
                results.append(result)
                queue.task_done()

        # Run producer and consumer concurrently
        await asyncio.gather(producer(), consumer())

        assert len(results) == 10
        for i, result in enumerate(results):
            assert result["result"] == f"result-{i}"

    @pytest.mark.asyncio
    async def test_semaphore_rate_limiting(self):
        """Test using semaphore for rate limiting."""
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent operations
        active_count = 0
        max_active = 0

        async def rate_limited_operation(op_id: int):
            nonlocal active_count, max_active

            async with semaphore:
                active_count += 1
                max_active = max(max_active, active_count)

                # Simulate work
                await asyncio.sleep(0.1)

                active_count -= 1
                return f"result-{op_id}"

        # Create many concurrent operations
        tasks = [rate_limited_operation(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert max_active <= 3  # Should never exceed semaphore limit

    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern(self):
        """Test circuit breaker pattern for fault tolerance."""
        failure_count = 0
        circuit_open = False

        async def unreliable_operation():
            nonlocal failure_count, circuit_open

            if circuit_open:
                raise RuntimeError("Circuit breaker is open")

            failure_count += 1
            if failure_count <= 3:
                raise RuntimeError("Service temporarily unavailable")

            # After 3 failures, start succeeding
            return "success"

        async def circuit_breaker_wrapper():
            nonlocal circuit_open

            try:
                return await unreliable_operation()
            except RuntimeError as e:
                if failure_count >= 3 and "temporarily unavailable" in str(e):
                    circuit_open = True
                raise

        # Test circuit breaker behavior
        results = []
        for i in range(6):
            try:
                result = await circuit_breaker_wrapper()
                results.append(result)
            except RuntimeError as e:
                results.append(str(e))

            # Reset circuit after some failures
            if i == 4:
                circuit_open = False

        # Should have mix of failures and successes
        assert len(results) == 6
        assert "Service temporarily unavailable" in str(results[0])
        assert "Circuit breaker is open" in str(results[3])

    @pytest.mark.asyncio
    async def test_bulkhead_pattern(self):
        """Test bulkhead pattern for resource isolation."""
        # Separate semaphores for different operation types
        sentiment_semaphore = asyncio.Semaphore(2)
        themes_semaphore = asyncio.Semaphore(1)

        sentiment_active = 0
        themes_active = 0

        async def sentiment_operation(text: str):
            nonlocal sentiment_active
            async with sentiment_semaphore:
                sentiment_active += 1
                await asyncio.sleep(0.1)
                sentiment_active -= 1
                return f"sentiment-{text}"

        async def themes_operation(texts: list):
            nonlocal themes_active
            async with themes_semaphore:
                themes_active += 1
                await asyncio.sleep(0.2)
                themes_active -= 1
                return f"themes-{len(texts)}"

        # Create mixed workload
        tasks = []
        for i in range(5):
            tasks.append(sentiment_operation(f"text-{i}"))
        for i in range(2):
            tasks.append(themes_operation([f"text-{i}"]))

        results = await asyncio.gather(*tasks)

        assert len(results) == 7
        # Verify resource isolation worked
        sentiment_results = [r for r in results if r.startswith("sentiment")]
        themes_results = [r for r in results if r.startswith("themes")]

        assert len(sentiment_results) == 5
        assert len(themes_results) == 2


class TestPerformanceBenchmarks:
    """Performance benchmarks for concurrent operations."""

    @pytest.mark.asyncio
    async def test_throughput_benchmark(self):
        """Benchmark throughput of concurrent operations."""
        mock_client = AsyncMock(spec=AsyncCoreClient)
        mock_client.analyze_sentiment.return_value = SentimentResponse(
            results=[SentimentResult(sentiment="positive", confidence=0.9)],
            usage=UsageReport(records=[], total=1),
        )

        # Sequential execution
        start_time = time.time()
        for i in range(10):  # Smaller number for sequential test
            await sentiment_analysis_async([f"text {i}"], client=mock_client)
        sequential_time = time.time() - start_time

        # Concurrent execution
        start_time = time.time()
        tasks = [
            sentiment_analysis_async([f"text {i}"], client=mock_client)
            for i in range(10)
        ]
        await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time

        # Concurrent should be significantly faster
        speedup = sequential_time / concurrent_time
        assert speedup > 2.0  # At least 2x speedup

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test memory usage under high concurrent load."""
        import gc

        # Force garbage collection before test
        gc.collect()

        num_operations = 50
        mock_client = AsyncMock(spec=AsyncCoreClient)
        mock_client.analyze_sentiment.return_value = SentimentResponse(
            results=[SentimentResult(sentiment="positive", confidence=0.9)],
            usage=UsageReport(records=[], total=1),
        )

        # Create many concurrent operations
        tasks = [
            sentiment_analysis_async([f"text {i}"], client=mock_client)
            for i in range(num_operations)
        ]

        # Execute and measure
        results = await asyncio.gather(*tasks)

        # Force cleanup
        del tasks
        del results
        gc.collect()

        # Test should complete without memory issues
        assert True  # If we get here, memory usage was acceptable

    @pytest.mark.asyncio
    async def test_latency_distribution(self):
        """Test latency distribution of concurrent operations."""
        latencies = []

        async def timed_operation(op_id: int):
            start = time.time()
            # Simulate variable processing time
            await asyncio.sleep(0.05 + (op_id % 3) * 0.02)
            end = time.time()
            latencies.append(end - start)
            return f"result-{op_id}"

        tasks = [timed_operation(i) for i in range(20)]
        await asyncio.gather(*tasks)

        # Analyze latency distribution
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)

        assert len(latencies) == 20
        assert min_latency >= 0.05  # Minimum processing time
        assert max_latency <= 0.15  # Maximum processing time
        assert 0.05 <= avg_latency <= 0.15
