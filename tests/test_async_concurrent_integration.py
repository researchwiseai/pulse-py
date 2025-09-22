"""Integration tests for async concurrent processing utilities with AsyncCoreClient."""

import pytest
from unittest.mock import AsyncMock, patch
from pulse.core.async_client import AsyncCoreClient
from pulse.core.async_concurrent import gather_jobs, submit_and_gather_jobs
from pulse.core.models import EmbeddingsRequest


class TestAsyncConcurrentIntegration:
    """Test integration between concurrent utilities and AsyncCoreClient."""

    @pytest.mark.asyncio
    async def test_gather_jobs_with_async_client(self):
        """Test gather_jobs with real AsyncCoreClient job submission."""
        # Mock the HTTP client to avoid real API calls
        mock_client = AsyncMock()

        # Mock successful job submission response
        mock_response = AsyncMock()
        mock_response.status_code = 202
        mock_response.json.return_value = {
            "jobId": "test-job-123",
            "jobStatus": "pending",
        }
        mock_client.request.return_value = mock_response

        # Create AsyncCoreClient with mocked HTTP client
        async_client = AsyncCoreClient(client=mock_client)

        # Create embedding requests
        requests = [
            EmbeddingsRequest(inputs=[f"Test text {i}"], fast=False) for i in range(3)
        ]

        # Submit jobs without waiting for results
        jobs = []
        for request in requests:
            with patch.object(async_client, "_request", return_value=mock_response):
                job = await async_client.create_embeddings(
                    request, await_job_result=False
                )
                # Mock the job's wait method
                job.wait = AsyncMock(return_value={"embeddings": [[0.1, 0.2, 0.3]]})
                jobs.append(job)

        # Use gather_jobs to collect results
        results = await gather_jobs(jobs, max_concurrent=2)

        assert len(results) == 3
        for result in results:
            assert "embeddings" in result
            assert len(result["embeddings"]) == 1

    @pytest.mark.asyncio
    async def test_submit_and_gather_with_async_client(self):
        """Test submit_and_gather_jobs with AsyncCoreClient job submitters."""
        # Mock the HTTP client
        mock_client = AsyncMock()

        # Mock successful job submission response
        mock_response = AsyncMock()
        mock_response.status_code = 202
        mock_response.json.return_value = {
            "jobId": "test-job-456",
            "jobStatus": "pending",
        }
        mock_client.request.return_value = mock_response

        # Create AsyncCoreClient with mocked HTTP client
        async_client = AsyncCoreClient(client=mock_client)

        # Create job submitters
        async def create_embedding_submitter(text_id):
            async def submitter():
                request = EmbeddingsRequest(
                    inputs=[f"Concurrent text {text_id}"], fast=False
                )
                with patch.object(async_client, "_request", return_value=mock_response):
                    job = await async_client.create_embeddings(
                        request, await_job_result=False
                    )
                    # Mock the job's wait method
                    job.wait = AsyncMock(
                        return_value={
                            "embeddings": [
                                [0.1 * text_id, 0.2 * text_id, 0.3 * text_id]
                            ]
                        }
                    )
                    return job

            return submitter

        # Create multiple submitters
        submitters = [await create_embedding_submitter(i) for i in range(1, 4)]

        # Use submit_and_gather_jobs
        results = await submit_and_gather_jobs(
            submitters,
            max_concurrent=2,
            rate_limit_delay=0.01,  # Small delay for testing
        )

        assert len(results) == 3
        for i, result in enumerate(results):
            assert "embeddings" in result
            assert len(result["embeddings"]) == 1
            # Check that each result has the expected scaled values
            expected_value = 0.1 * (i + 1)
            assert result["embeddings"][0][0] == expected_value

    @pytest.mark.asyncio
    async def test_concurrent_utilities_error_handling(self):
        """Test error handling in concurrent utilities."""
        # Create mock jobs with mixed success/failure
        jobs = []

        # Successful job
        success_job = AsyncMock()
        success_job.id = "success-job"
        success_job.wait = AsyncMock(return_value={"embeddings": [[1.0, 2.0, 3.0]]})
        jobs.append(success_job)

        # Failing job
        fail_job = AsyncMock()
        fail_job.id = "fail-job"
        fail_job.wait = AsyncMock(side_effect=RuntimeError("Job processing failed"))
        jobs.append(fail_job)

        # Test with return_exceptions=True
        results = await gather_jobs(jobs, max_concurrent=2, return_exceptions=True)

        assert len(results) == 2
        assert results[0]["embeddings"] == [[1.0, 2.0, 3.0]]
        assert isinstance(results[1], RuntimeError)
        assert str(results[1]) == "Job processing failed"

    @pytest.mark.asyncio
    async def test_concurrent_utilities_timeout_handling(self):
        """Test timeout handling in concurrent utilities."""
        import asyncio

        # Create jobs with different processing times
        jobs = []

        # Fast job
        fast_job = AsyncMock()
        fast_job.id = "fast-job"
        fast_job.wait = AsyncMock(return_value={"embeddings": [[1.0]]})
        jobs.append(fast_job)

        # Slow job that will timeout
        slow_job = AsyncMock()
        slow_job.id = "slow-job"

        async def slow_wait(timeout=None):
            await asyncio.sleep(1.0)  # Longer than test timeout
            return {"embeddings": [[2.0]]}

        slow_job.wait = slow_wait
        jobs.append(slow_job)

        # Test with short timeout and exception handling
        results = await gather_jobs(
            jobs, max_concurrent=2, timeout=0.1, return_exceptions=True  # Short timeout
        )

        assert len(results) == 2
        # First job should succeed
        assert results[0]["embeddings"] == [[1.0]]
        # Second job should timeout
        assert isinstance(results[1], asyncio.TimeoutError)

    @pytest.mark.asyncio
    async def test_connection_pool_optimization(self):
        """Test that connection pool optimization works."""
        from pulse.core.async_concurrent import (
            AsyncConnectionPoolManager,
        )

        # Test recommended config generation
        config = AsyncConnectionPoolManager.get_recommended_config(concurrent_jobs=10)

        assert config.max_concurrent_jobs == 10
        assert config.connection_pool_limits["max_connections"] == 50
        assert config.rate_limit_delay == 0.05

        # Test optimized client creation
        client = AsyncConnectionPoolManager.create_optimized_client(
            "https://api.example.com", config=config
        )

        assert str(client.base_url) == "https://api.example.com"
        # Verify client was created with optimized settings
        assert hasattr(client, "_limits")

        # Clean up
        await client.aclose()

    @pytest.mark.asyncio
    async def test_async_client_uses_concurrent_utilities(self):
        """Test that AsyncCoreClient uses the new concurrent utilities."""
        # This test verifies that the _process_batches_concurrently method
        # now uses the new concurrent utilities

        mock_client = AsyncMock()
        async_client = AsyncCoreClient(client=mock_client)

        # Create mock job submitters
        async def mock_submitter():
            job = AsyncMock()
            job.id = "test-job"
            job.wait = AsyncMock(return_value={"embeddings": [[1.0, 2.0]]})
            return job

        submitters = [mock_submitter for _ in range(3)]

        # Test the _process_batches_concurrently method
        results = await async_client._process_batches_concurrently(
            submitters, max_workers=2, timeout=30.0
        )

        assert len(results) == 3
        for result in results:
            assert result["embeddings"] == [[1.0, 2.0]]
