"""Comprehensive unit tests for enhanced batching logic and error scenarios.

This test suite validates all batching logic, error handling, and concurrent processing
for the enhanced automatic batching feature across all supported operations.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Any, Dict

import numpy as np

from pulse.core import batching
from pulse.core.concurrent import (
    ConcurrentJobManager,
    ConcurrentJobConfig,
    AsyncJobProcessor,
)
from pulse.core.validation import (
    BatchingError,
    BatchingErrorHelper,
    ValidationLimits,
    ERROR_DOCUMENTATION,
)


class MockJob:
    """Mock job for testing concurrent processing."""

    def __init__(
        self, result: Any = None, should_fail: bool = False, delay: float = 0.0
    ):
        self.result = result
        self.should_fail = should_fail
        self.delay = delay
        self._completed = False

    def wait(self, timeout: float = 600.0) -> Any:
        """Mock wait method that simulates job completion."""
        import time

        if self.delay > 0:
            time.sleep(min(self.delay, 0.1))  # Cap delay for tests

        if self.should_fail:
            raise RuntimeError(f"Mock job failed: {self.result}")

        self._completed = True
        return self.result

    @property
    def completed(self) -> bool:
        return self._completed


class MockSentimentResult:
    """Mock sentiment analysis result."""

    def __init__(self, sentiments: List[Dict[str, Any]], usage: Dict[str, int] = None):
        self.sentiments = sentiments
        self.usage = usage or {"input_tokens": 100, "output_tokens": 50}


class MockEmbeddingsResult:
    """Mock embeddings result."""

    def __init__(self, embeddings: List[List[float]], usage: Dict[str, int] = None):
        self.embeddings = embeddings
        self.usage = usage or {"input_tokens": 200, "output_tokens": 0}


class MockClusteringResult:
    """Mock clustering result."""

    def __init__(self, clusters: List[Dict[str, Any]], usage: Dict[str, int] = None):
        self.clusters = clusters
        self.usage = usage or {"input_tokens": 150, "output_tokens": 25}


class MockExtractionsResult:
    """Mock extractions result."""

    def __init__(self, extractions: List[Dict[str, Any]], usage: Dict[str, int] = None):
        self.extractions = extractions
        self.usage = usage or {"input_tokens": 120, "output_tokens": 30}


class TestBatchingUtilities:
    """Test core batching utility functions."""

    def test_make_self_chunks_no_split(self):
        """Test chunking when input is within limits."""
        items = list(range(100))
        chunks = batching._make_self_chunks(items)
        assert chunks == [items]

    def test_make_self_chunks_with_split(self, monkeypatch):
        """Test chunking when input exceeds limits."""
        monkeypatch.setattr(batching, "MAX_ITEMS", 1000)
        monkeypatch.setattr(batching, "HALF_CHUNK", 500)

        items = list(range(1200))
        chunks = batching._make_self_chunks(items)

        assert len(chunks) == 3  # ceil(1200/500) = 3
        assert chunks[0] == items[0:500]
        assert chunks[1] == items[500:1000]
        assert chunks[2] == items[1000:1200]

    def test_make_clustering_chunks_no_split(self):
        """Test clustering chunks when input is within threshold."""
        items = list(range(400))
        chunks = batching._make_clustering_chunks(items)
        assert chunks == [items]

    def test_make_clustering_chunks_with_split(self):
        """Test clustering chunks when input exceeds threshold."""
        items = list(range(1200))
        chunks = batching._make_clustering_chunks(items)

        expected_chunks = 3  # ceil(1200/500) = 3
        assert len(chunks) == expected_chunks
        assert chunks[0] == items[0:500]
        assert chunks[1] == items[500:1000]
        assert chunks[2] == items[1000:1200]

    def test_make_cross_bodies_combined(self):
        """Test cross-body creation when combined size fits."""
        set_a = list(range(100))
        set_b = list(range(100))

        bodies = batching._make_cross_bodies(
            set_a, set_b, flatten=True, version="v1", split="newline"
        )

        assert len(bodies) == 1
        assert bodies[0] == {
            "set_a": set_a,
            "set_b": set_b,
            "flatten": True,
            "version": "v1",
            "split": "newline",
        }

    def test_make_cross_bodies_chunk_b_only(self, monkeypatch):
        """Test cross-body creation when only set_b needs chunking."""
        monkeypatch.setattr(batching, "MAX_ITEMS", 500)

        set_a = list(range(100))  # Small set
        set_b = list(range(300))  # Medium set that fits the chunking condition

        bodies = batching._make_cross_bodies(set_a, set_b, flatten=False, version="v1")

        # Should chunk set_b while keeping set_a intact
        # With A=100, B=300, MAX_ITEMS=500: A+B=400 <= 500, so should not chunk
        # Let's test the actual chunking case
        set_b_large = list(range(450))  # This makes A+B = 550 > 500, and B < 500

        bodies = batching._make_cross_bodies(
            set_a, set_b_large, flatten=False, version="v1"
        )

        # Should chunk set_b while keeping set_a intact
        assert len(bodies) > 1
        for body in bodies:
            assert body["set_a"] == set_a
            assert body["flatten"] is False
            assert body["version"] == "v1"

    def test_stitch_results_self_similarity(self, monkeypatch):
        """Test stitching results for self-similarity matrix."""
        monkeypatch.setattr(batching, "MAX_ITEMS", 6)
        monkeypatch.setattr(batching, "HALF_CHUNK", 3)

        full = list(range(6))
        chunks = batching._make_self_chunks(full)

        # Create mock results for each chunk combination
        results = []
        for i in range(len(chunks)):
            for j in range(i, len(chunks)):
                shape = (len(chunks[i]), len(chunks[j]))
                val = (i + 1) * 10 + (j + 1)
                mat = np.full(shape, val, dtype=float)
                results.append(Mock(matrix=mat))

        matrix = batching._stitch_results(results, [], full, full)

        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (len(full), len(full))
        # Verify symmetry
        assert np.allclose(matrix, matrix.T)


class TestConcurrentJobManager:
    """Test concurrent job management functionality."""

    def test_concurrent_job_manager_initialization(self):
        """Test ConcurrentJobManager initialization with different configs."""
        # Default config
        manager = ConcurrentJobManager()
        assert manager.config.max_workers == 5
        assert manager.config.timeout == 600.0
        assert not manager.config.use_async

        # Custom config
        config = ConcurrentJobConfig(max_workers=3, timeout=300.0, use_async=True)
        manager = ConcurrentJobManager(config)
        assert manager.config.max_workers == 3
        assert manager.config.timeout == 300.0
        assert manager.config.use_async

    def test_submit_jobs_success(self):
        """Test successful job submission with concurrent processing."""
        config = ConcurrentJobConfig(max_workers=2, timeout=10.0)

        # Create job submitters that return mock jobs
        job_submitters = [lambda i=i: MockJob(result=f"result_{i}") for i in range(3)]

        with ConcurrentJobManager(config) as manager:
            jobs = manager.submit_jobs(job_submitters)

            assert len(jobs) == 3
            for i, job in enumerate(jobs):
                assert isinstance(job, MockJob)
                assert job.result == f"result_{i}"

    def test_submit_jobs_partial_failure(self):
        """Test job submission with some failures."""
        config = ConcurrentJobConfig(max_workers=2, timeout=10.0)

        # Create job submitters with one that fails
        job_submitters = [
            lambda: MockJob(result="success_1"),
            lambda: MockJob(result="failure", should_fail=True),
            lambda: MockJob(result="success_2"),
        ]

        with ConcurrentJobManager(config) as manager:
            # Should handle partial failures gracefully
            with pytest.warns(UserWarning, match="job submissions failed"):
                jobs = manager.submit_jobs(job_submitters)

            # Should get successful jobs only
            assert len(jobs) == 2

    def test_submit_jobs_all_failures(self):
        """Test job submission when all jobs fail."""
        config = ConcurrentJobConfig(max_workers=2, timeout=10.0)

        # Create job submitters that all fail
        job_submitters = [
            lambda: MockJob(result="failure_1", should_fail=True),
            lambda: MockJob(result="failure_2", should_fail=True),
        ]

        with ConcurrentJobManager(config) as manager:
            with pytest.raises(RuntimeError, match="All .* job submissions failed"):
                manager.submit_jobs(job_submitters)

    def test_wait_for_jobs_success(self):
        """Test successful job completion waiting."""
        config = ConcurrentJobConfig(max_workers=2, timeout=10.0)

        jobs = [MockJob(result=f"result_{i}") for i in range(3)]

        with ConcurrentJobManager(config) as manager:
            results = manager.wait_for_jobs(jobs)

            assert len(results) == 3
            for i, result in enumerate(results):
                assert result == f"result_{i}"

    def test_wait_for_jobs_partial_failure(self):
        """Test job waiting with some failures."""
        config = ConcurrentJobConfig(max_workers=2, timeout=10.0)

        jobs = [
            MockJob(result="success_1"),
            MockJob(result="failure", should_fail=True),
            MockJob(result="success_2"),
        ]

        with ConcurrentJobManager(config) as manager:
            with pytest.warns(UserWarning, match="jobs failed to complete"):
                results = manager.wait_for_jobs(jobs)

            # Should get results with None for failed jobs
            assert len(results) == 3
            assert results[0] == "success_1"
            assert results[1] is None  # Failed job
            assert results[2] == "success_2"

    def test_submit_and_wait_integration(self):
        """Test integrated submit and wait functionality."""
        config = ConcurrentJobConfig(max_workers=2, timeout=10.0)

        job_submitters = [lambda i=i: MockJob(result=f"result_{i}") for i in range(3)]

        with ConcurrentJobManager(config) as manager:
            results = manager.submit_and_wait(job_submitters)

            assert len(results) == 3
            for i, result in enumerate(results):
                assert result == f"result_{i}"

    def test_context_manager_cleanup(self):
        """Test that context manager properly cleans up resources."""
        config = ConcurrentJobConfig(max_workers=2)
        manager = ConcurrentJobManager(config)

        with manager:
            assert manager._executor is not None

        # After context exit, executor should be cleaned up
        assert manager._executor is None

    def test_async_mode_error_handling(self):
        """Test error handling when async mode is requested but sync methods called."""
        config = ConcurrentJobConfig(use_async=True)
        manager = ConcurrentJobManager(config)

        job_submitters = [lambda: MockJob(result="test")]

        with pytest.raises(RuntimeError, match="Use submit_jobs_async"):
            with manager:
                manager.submit_jobs(job_submitters)


class TestAsyncJobProcessor:
    """Test AsyncJobProcessor functionality."""

    def test_create_job_submitter(self):
        """Test job submitter creation."""
        mock_client_method = Mock(return_value=MockJob(result="test_result"))

        submitter = AsyncJobProcessor.create_job_submitter(
            mock_client_method, param1="value1", param2="value2"
        )

        job = submitter()
        assert isinstance(job, MockJob)
        assert job.result == "test_result"
        mock_client_method.assert_called_once_with(param1="value1", param2="value2")

    def test_create_job_submitter_with_error(self):
        """Test job submitter creation when client method fails."""
        mock_client_method = Mock(side_effect=RuntimeError("Client error"))

        submitter = AsyncJobProcessor.create_job_submitter(
            mock_client_method, param1="value1"
        )

        with pytest.raises(RuntimeError, match="Failed to submit job"):
            submitter()

    def test_batch_process_with_concurrency_success(self):
        """Test successful batch processing with concurrency."""
        job_submitters = [lambda i=i: MockJob(result=f"result_{i}") for i in range(3)]

        config = ConcurrentJobConfig(max_workers=2, enable_progress=False)

        results = AsyncJobProcessor.batch_process_with_concurrency(
            job_submitters, config, feature="test"
        )

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result == f"result_{i}"

    def test_batch_process_with_concurrency_partial_failure(self):
        """Test batch processing with some job failures."""
        job_submitters = [
            lambda: MockJob(result="success_1"),
            lambda: MockJob(result="failure", should_fail=True),
            lambda: MockJob(result="success_2"),
        ]

        config = ConcurrentJobConfig(max_workers=2, enable_progress=False)

        with pytest.warns(UserWarning, match="batches failed"):
            results = AsyncJobProcessor.batch_process_with_concurrency(
                job_submitters, config, feature="test"
            )

        # Should get results with None for failed jobs
        assert len(results) == 3
        assert results[0] == "success_1"
        assert results[1] is None  # Failed job
        assert results[2] == "success_2"

    def test_batch_process_with_concurrency_all_failures(self):
        """Test batch processing when all jobs fail."""
        job_submitters = [
            lambda: MockJob(result="failure_1", should_fail=True),
            lambda: MockJob(result="failure_2", should_fail=True),
        ]

        config = ConcurrentJobConfig(max_workers=2, enable_progress=False)

        with pytest.raises(BatchingError, match="Batch .* failed"):
            AsyncJobProcessor.batch_process_with_concurrency(
                job_submitters, config, feature="test"
            )

    @patch("pulse.core.concurrent.tqdm")
    def test_batch_process_with_progress_bar(self, mock_tqdm):
        """Test batch processing with progress bar enabled."""
        mock_progress = Mock()
        mock_tqdm.return_value = mock_progress

        job_submitters = [lambda i=i: MockJob(result=f"result_{i}") for i in range(3)]

        config = ConcurrentJobConfig(max_workers=2, enable_progress=True)

        results = AsyncJobProcessor.batch_process_with_concurrency(
            job_submitters, config, feature="test"
        )

        assert len(results) == 3
        # Verify progress bar was used
        mock_tqdm.assert_called_once()
        mock_progress.update.assert_called()
        mock_progress.close.assert_called()


class TestBatchingErrorHandling:
    """Test comprehensive error handling for batching operations."""

    def test_batching_error_creation(self):
        """Test BatchingError creation with all parameters."""
        error = BatchingError(
            message="Test error message",
            feature="sentiment",
            input_count=1000,
            limit=200,
            suggested_action="Use slow mode",
            error_code="BATCH_001",
            batch_number=2,
            total_batches=5,
            resolution_steps=["Step 1", "Step 2"],
            context={"extra": "info"},
            alternatives=["Alternative 1"],
            severity="user_error",
            category="limit_exceeded",
        )

        assert error.feature == "sentiment"
        assert error.input_count == 1000
        assert error.limit == 200
        assert error.suggested_action == "Use slow mode"
        assert error.error_code == "BATCH_001"
        assert error.batch_number == 2
        assert error.total_batches == 5
        assert error.resolution_steps == ["Step 1", "Step 2"]
        assert error.context["extra"] == "info"
        assert error.alternatives == ["Alternative 1"]
        assert error.severity == "user_error"
        assert error.category == "limit_exceeded"

    def test_batching_error_structured_info(self):
        """Test BatchingError structured information generation."""
        error = BatchingError(
            message="Test error",
            feature="embeddings",
            input_count=500,
            limit=200,
            suggested_action="Reduce input size",
            error_code="BATCH_001",
        )

        info = error.get_structured_info()

        assert info["error_code"] == "BATCH_001"
        assert info["feature"] == "embeddings"
        assert info["input_count"] == 500
        assert info["limit"] == 200
        assert info["suggested_action"] == "Reduce input size"
        assert "timestamp" in info
        assert "is_retryable" in info
        assert "expected_fix_time" in info

    def test_batching_error_retry_strategy(self):
        """Test retry strategy generation for different error types."""
        # Test retryable error
        error = BatchingError(
            message="Batch failed",
            feature="sentiment",
            input_count=100,
            limit=200,
            suggested_action="Retry",
            error_code="BATCH_003",  # Batch job failed
        )

        strategy = error.get_retry_strategy()
        assert strategy is not None
        assert strategy["strategy"] == "exponential_backoff"
        assert strategy["max_retries"] == 3

        # Test non-retryable error
        error_non_retryable = BatchingError(
            message="Limit exceeded",
            feature="sentiment",
            input_count=1000,
            limit=200,
            suggested_action="Reduce input",
            error_code="BATCH_001",  # Fast mode limit exceeded
        )

        strategy = error_non_retryable.get_retry_strategy()
        assert strategy is None

    def test_batching_error_context_management(self):
        """Test context management in BatchingError."""
        error = BatchingError(
            message="Test error",
            feature="clustering",
            input_count=100,
            limit=50,
            suggested_action="Test action",
        )

        # Test adding context
        error.add_context("custom_field", "custom_value")
        assert error.get_error_context_for_field("custom_field") == "custom_value"

        # Test getting non-existent context
        assert error.get_error_context_for_field("non_existent") is None

    def test_batching_error_logging_format(self):
        """Test error formatting for logging."""
        error = BatchingError(
            message="Test error",
            feature="extractions",
            input_count=300,
            limit=200,
            suggested_action="Test action",
            error_code="BATCH_002",
        )

        log_format = error.format_for_logging()

        assert log_format["error_type"] == "BatchingError"
        assert log_format["error_code"] == "BATCH_002"
        assert log_format["feature"] == "extractions"
        assert "timestamp" in log_format
        assert "is_retryable" in log_format

    def test_batching_error_serialization(self):
        """Test error serialization to dictionary."""
        error = BatchingError(
            message="Test error",
            feature="sentiment",
            input_count=100,
            limit=200,
            suggested_action="Test action",
            error_code="BATCH_001",
        )

        error_dict = error.to_dict()

        assert error_dict["type"] == "BatchingError"
        assert "message" in error_dict
        assert "structured_info" in error_dict
        assert "retry_strategy" in error_dict


class TestBatchingErrorHelper:
    """Test BatchingErrorHelper utility functions."""

    def test_create_fast_mode_error(self):
        """Test fast mode error creation."""
        error = BatchingErrorHelper.create_fast_mode_error(
            feature="sentiment", input_count=500, limit=200
        )

        assert isinstance(error, BatchingError)
        assert error.feature == "sentiment"
        assert error.input_count == 500
        assert error.limit == 200
        assert error.error_code == "BATCH_001"
        assert error.mode == "fast"
        assert "Fast mode supports up to 200 texts" in str(error)

    def test_create_slow_mode_error(self):
        """Test slow mode error creation."""
        error = BatchingErrorHelper.create_slow_mode_error(
            feature="embeddings", input_count=2_000_000, limit=1_000_000
        )

        assert isinstance(error, BatchingError)
        assert error.feature == "embeddings"
        assert error.input_count == 2_000_000
        assert error.limit == 1_000_000
        assert error.error_code == "BATCH_002"
        assert error.mode == "slow"
        assert "Maximum input limit is 1,000,000 texts" in str(error)

    def test_create_batch_failure_error(self):
        """Test batch failure error creation."""
        error = BatchingErrorHelper.create_batch_failure_error(
            feature="clustering",
            batch_num=3,
            total_batches=10,
            error_message="Network timeout",
            input_count=5000,
        )

        assert isinstance(error, BatchingError)
        assert error.feature == "clustering"
        assert error.batch_number == 3
        assert error.total_batches == 10
        assert error.error_code == "BATCH_003"
        assert "Batch 3 of 10 failed" in str(error)
        assert "Network timeout" in str(error)

    def test_create_concurrency_error(self):
        """Test concurrency error creation."""
        error = BatchingErrorHelper.create_concurrency_error(current_jobs=8, max_jobs=5)

        assert isinstance(error, BatchingError)
        assert error.input_count == 8
        assert error.limit == 5
        assert error.error_code == "BATCH_004"
        assert "Too many concurrent jobs: 8" in str(error)

    def test_create_timeout_error(self):
        """Test timeout error creation."""
        error = BatchingErrorHelper.create_timeout_error(
            feature="extractions", timeout_seconds=300.0, batch_num=2
        )

        assert isinstance(error, BatchingError)
        assert error.feature == "extractions"
        assert error.batch_number == 2
        assert error.error_code == "BATCH_006"
        assert "timed out after 300" in str(error)

    def test_create_resource_error(self):
        """Test resource error creation."""
        error = BatchingErrorHelper.create_resource_error(
            feature="sentiment", resource_type="memory", current_usage="85%"
        )

        assert isinstance(error, BatchingError)
        assert error.feature == "sentiment"
        assert error.error_code == "BATCH_007"
        assert "Resource exhaustion: memory usage too high (85%)" in str(error)

    def test_create_order_corruption_error(self):
        """Test order corruption error creation."""
        error = BatchingErrorHelper.create_order_corruption_error(
            feature="embeddings", expected_count=1000, actual_count=950, batch_num=5
        )

        assert isinstance(error, BatchingError)
        assert error.feature == "embeddings"
        assert error.batch_number == 5
        assert error.error_code == "BATCH_008"
        assert "order corruption detected" in str(error)
        assert "Expected 1000 results, got 950" in str(error)


class TestValidationLimits:
    """Test validation limits and error documentation."""

    def test_validation_limits_constants(self):
        """Test that validation limits are properly defined."""
        # Embeddings limits
        assert ValidationLimits.EMBEDDINGS_SYNC_MAX == 200
        assert ValidationLimits.EMBEDDINGS_ASYNC_MAX == 1_000_000
        assert ValidationLimits.EMBEDDINGS_BATCH_SIZE == 5_000

        # Sentiment limits
        assert ValidationLimits.SENTIMENT_SYNC_MAX == 200
        assert ValidationLimits.SENTIMENT_ASYNC_MAX == 1_000_000
        assert ValidationLimits.SENTIMENT_BATCH_SIZE == 2_000

        # Extractions limits
        assert ValidationLimits.EXTRACTIONS_SYNC_MAX == 200
        assert ValidationLimits.EXTRACTIONS_ASYNC_MAX == 1_000_000
        assert ValidationLimits.EXTRACTIONS_BATCH_SIZE == 2_000

        # Clustering limits
        assert ValidationLimits.CLUSTERING_SYNC_MAX == 500
        assert ValidationLimits.CLUSTERING_ASYNC_MAX == 44721
        assert ValidationLimits.CLUSTERING_AUTO_BATCH_THRESHOLD == 500

    def test_error_documentation_structure(self):
        """Test that error documentation is properly structured."""
        for error_key, error_info in ERROR_DOCUMENTATION.items():
            # Check required fields
            assert "code" in error_info
            assert "category" in error_info
            assert "severity" in error_info
            assert "message_template" in error_info
            assert "resolution_steps" in error_info
            assert "context_fields" in error_info

            # Check that resolution steps are non-empty
            assert len(error_info["resolution_steps"]) > 0

            # Check that context fields are defined
            assert len(error_info["context_fields"]) > 0

    def test_error_documentation_completeness(self):
        """Test that all expected error codes are documented."""
        expected_codes = [
            "BATCH_001",  # Fast mode limit exceeded
            "BATCH_002",  # Slow mode limit exceeded
            "BATCH_003",  # Batch job failed
            "BATCH_004",  # Concurrency limit exceeded
            "BATCH_005",  # Clustering auto-batch triggered
            "BATCH_006",  # Batch timeout
            "BATCH_007",  # Resource exhaustion
            "BATCH_008",  # Batch order corruption
            "BATCH_009",  # Invalid batch configuration
            "BATCH_010",  # Batch result aggregation failed
        ]

        documented_codes = [info["code"] for info in ERROR_DOCUMENTATION.values()]

        for code in expected_codes:
            assert code in documented_codes, f"Error code {code} not documented"


class TestBatchProcessingConcurrently:
    """Test the process_batches_concurrently function."""

    def test_process_batches_fast_mode_sequential(self):
        """Test that fast mode processes batches sequentially."""
        job_submitters = [lambda i=i: MockJob(result=f"result_{i}") for i in range(3)]

        results = batching.process_batches_concurrently(
            job_submitters,
            max_workers=5,
            timeout=10.0,
            fast=True,  # Should process sequentially
            feature="test",
            enable_progress=False,
        )

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result == f"result_{i}"

    def test_process_batches_slow_mode_concurrent(self):
        """Test that slow mode processes batches concurrently."""
        job_submitters = [lambda i=i: MockJob(result=f"result_{i}") for i in range(3)]

        results = batching.process_batches_concurrently(
            job_submitters,
            max_workers=2,
            timeout=10.0,
            fast=False,  # Should process concurrently
            feature="test",
            enable_progress=False,
        )

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result == f"result_{i}"

    def test_process_batches_fast_mode_submission_failure(self):
        """Test handling of job submission failures in fast mode."""
        job_submitters = [
            lambda: MockJob(result="success"),
            lambda: (_ for _ in ()).throw(RuntimeError("Submission failed")),
            lambda: MockJob(result="success_2"),
        ]

        with pytest.raises(BatchingError, match="Batch .* failed"):
            batching.process_batches_concurrently(
                job_submitters,
                fast=True,
                feature="test",
                enable_progress=False,
            )

    def test_process_batches_fast_mode_completion_failure(self):
        """Test handling of job completion failures in fast mode."""
        job_submitters = [
            lambda: MockJob(result="success"),
            lambda: MockJob(result="failure", should_fail=True),
            lambda: MockJob(result="success_2"),
        ]

        with pytest.raises(BatchingError, match="Batch .* failed"):
            batching.process_batches_concurrently(
                job_submitters,
                fast=True,
                feature="test",
                enable_progress=False,
            )

    def test_process_batches_slow_mode_with_failures(self):
        """Test slow mode processing with some failures."""
        job_submitters = [
            lambda: MockJob(result="success_1"),
            lambda: MockJob(result="failure", should_fail=True),
            lambda: MockJob(result="success_2"),
        ]

        with pytest.warns(UserWarning, match="batches failed"):
            results = batching.process_batches_concurrently(
                job_submitters,
                fast=False,
                feature="test",
                enable_progress=False,
            )

        # Should get results with None for failed jobs
        assert len(results) == 3
        assert results[0] == "success_1"
        assert results[1] is None  # Failed job
        assert results[2] == "success_2"


class TestCreateBatchJobSubmitters:
    """Test batch job submitter creation."""

    def test_create_batch_job_submitters_basic(self):
        """Test basic job submitter creation."""
        mock_client_method = Mock(return_value=MockJob(result="test"))
        request_bodies = [
            {"param1": "value1"},
            {"param2": "value2"},
        ]

        submitters = batching.create_batch_job_submitters(
            mock_client_method, request_bodies
        )

        assert len(submitters) == 2

        # Test first submitter
        job1 = submitters[0]()
        assert isinstance(job1, MockJob)
        assert job1.result == "test"

        # Test second submitter
        job2 = submitters[1]()
        assert isinstance(job2, MockJob)
        assert job2.result == "test"

    def test_create_batch_job_submitters_with_concurrent_module(self):
        """Test job submitter creation when concurrent module is available."""
        with patch("pulse.core.batching._get_concurrent_classes") as mock_get_classes:
            # Mock the concurrent classes as available
            mock_get_classes.return_value = (Mock, Mock, Mock, True)

            mock_client_method = Mock(return_value=MockJob(result="test"))
            request_bodies = [{"param": "value"}]

            submitters = batching.create_batch_job_submitters(
                mock_client_method, request_bodies
            )

            assert len(submitters) == 1
            assert callable(submitters[0])


class TestNetworkResourceManager:
    """Test network resource management functionality."""

    def test_network_resource_manager_initialization(self):
        """Test NetworkResourceManager initialization."""
        manager = batching.NetworkResourceManager(
            base_timeout=200.0,
            max_timeout=600.0,
            max_retries=5,
            backoff_factor=1.5,
            connection_pool_size=20,
        )

        assert manager.base_timeout == 200.0
        assert manager.max_timeout == 600.0
        assert manager.max_retries == 5
        assert manager.backoff_factor == 1.5
        assert manager.connection_pool_size == 20

    def test_adaptive_timeout_calculation(self):
        """Test adaptive timeout calculation."""
        manager = batching.NetworkResourceManager(base_timeout=100.0, max_timeout=500.0)

        # Small batch
        timeout = manager.get_adaptive_timeout(batch_size=100, attempt=0)
        assert timeout == 100.0  # Base timeout

        # Large batch
        timeout = manager.get_adaptive_timeout(batch_size=5000, attempt=0)
        assert timeout == 500.0  # Should scale up and hit max

        # With retry attempt
        timeout = manager.get_adaptive_timeout(batch_size=100, attempt=1)
        assert timeout == 200.0  # Base * backoff_factor

    def test_should_retry_request(self):
        """Test retry decision logic."""
        manager = batching.NetworkResourceManager(max_retries=3)

        # Test retryable errors
        retryable_errors = [
            RuntimeError("Connection timeout"),
            RuntimeError("Network error"),
            RuntimeError("502 Bad Gateway"),
            RuntimeError("Rate limit exceeded"),
        ]

        for error in retryable_errors:
            assert manager.should_retry_request(error, attempt=1)
            assert manager.should_retry_request(error, attempt=2)
            assert not manager.should_retry_request(error, attempt=3)  # Max retries

        # Test non-retryable error
        non_retryable_error = RuntimeError("Invalid input")
        assert not manager.should_retry_request(non_retryable_error, attempt=1)

    def test_retry_delay_calculation(self):
        """Test retry delay calculation."""
        manager = batching.NetworkResourceManager(backoff_factor=2.0)

        assert manager.get_retry_delay(attempt=1) == 1.0
        assert manager.get_retry_delay(attempt=2) == 2.0
        assert manager.get_retry_delay(attempt=3) == 4.0

    def test_request_tracking(self):
        """Test request tracking functionality."""
        manager = batching.NetworkResourceManager()

        # Track successful requests
        manager.track_request_start()
        manager.track_request_start()
        manager.track_request_end(success=True)
        manager.track_request_end(success=False)

        stats = manager.get_network_stats()
        assert stats["active_connections"] == 0
        assert stats["total_requests"] == 2
        assert stats["failed_requests"] == 1
        assert stats["success_rate"] == 0.5

    def test_throttling_decision(self):
        """Test request throttling decision."""
        manager = batching.NetworkResourceManager(connection_pool_size=10)

        # Simulate high failure rate
        for _ in range(10):
            manager.track_request_start()
            manager.track_request_end(success=False)

        assert manager.should_throttle_requests()  # High failure rate

        # Reset and test high connection usage
        manager = batching.NetworkResourceManager(connection_pool_size=10)
        for _ in range(9):
            manager.track_request_start()

        assert manager.should_throttle_requests()  # High connection usage


class TestClusteringResultReconstruction:
    """Test clustering result reconstruction functionality."""

    def test_reconstruct_clustering_results_empty(self):
        """Test reconstruction with empty results."""
        result = batching._reconstruct_clustering_results(
            results=[], original_inputs=[], k=3, algorithm="kmeans"
        )

        assert result["clusters"] == []
        assert result["usage"] is None

    def test_reconstruct_clustering_results_single_batch(self):
        """Test reconstruction with single batch result."""
        mock_result = MockClusteringResult(
            clusters=[
                {"cluster_id": 0, "text": "text1"},
                {"cluster_id": 1, "text": "text2"},
            ],
            usage={"input_tokens": 100, "output_tokens": 20},
        )

        result = batching._reconstruct_clustering_results(
            results=[mock_result],
            original_inputs=["text1", "text2"],
            k=2,
            algorithm="kmeans",
        )

        assert len(result["clusters"]) == 2
        assert result["clusters"][0]["cluster_id"] == 0
        assert result["clusters"][1]["cluster_id"] == 1
        assert result["usage"]["input_tokens"] == 100

    def test_reconstruct_clustering_results_multiple_batches(self):
        """Test reconstruction with multiple batch results."""
        mock_result1 = MockClusteringResult(
            clusters=[
                {"cluster_id": 0, "text": "text1"},
                {"cluster_id": 1, "text": "text2"},
            ],
            usage={"input_tokens": 50, "output_tokens": 10},
        )

        mock_result2 = MockClusteringResult(
            clusters=[
                {"cluster_id": 0, "text": "text3"},
                {"cluster_id": 1, "text": "text4"},
            ],
            usage={"input_tokens": 60, "output_tokens": 15},
        )

        result = batching._reconstruct_clustering_results(
            results=[mock_result1, mock_result2],
            original_inputs=["text1", "text2", "text3", "text4"],
            k=2,
            algorithm="kmeans",
        )

        assert len(result["clusters"]) == 4
        # Check cluster ID adjustment
        assert result["clusters"][0]["cluster_id"] == 0  # First batch
        assert result["clusters"][1]["cluster_id"] == 1  # First batch
        assert result["clusters"][2]["cluster_id"] == 2  # Second batch, adjusted
        assert result["clusters"][3]["cluster_id"] == 3  # Second batch, adjusted

        # Check usage aggregation
        assert result["usage"]["input_tokens"] == 110  # 50 + 60
        assert result["usage"]["output_tokens"] == 25  # 10 + 15

    def test_reconstruct_clustering_results_numeric_clusters(self):
        """Test reconstruction with numeric cluster assignments."""
        mock_result1 = MockClusteringResult(clusters=[0, 1, 0])
        mock_result2 = MockClusteringResult(clusters=[0, 1])

        result = batching._reconstruct_clustering_results(
            results=[mock_result1, mock_result2],
            original_inputs=["text1", "text2", "text3", "text4", "text5"],
            k=2,
            algorithm="kmeans",
        )

        assert len(result["clusters"]) == 5
        # Check cluster ID adjustment for numeric assignments
        assert result["clusters"][0] == 0  # First batch
        assert result["clusters"][1] == 1  # First batch
        assert result["clusters"][2] == 0  # First batch
        assert result["clusters"][3] == 2  # Second batch, adjusted
        assert result["clusters"][4] == 3  # Second batch, adjusted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
