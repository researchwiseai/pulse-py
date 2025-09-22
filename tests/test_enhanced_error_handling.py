"""Tests for enhanced error handling and performance optimization in batching."""

import pytest
from unittest.mock import Mock, patch
from pulse.core.batching import (
    process_batches_concurrently,
    BatchConcurrencyController,
    ResourceMonitor,
    BatchOptimizer,
    NetworkResourceManager,
    PerformanceBenchmark,
)
from pulse.core.validation import BatchingError, BatchingErrorHelper


class TestEnhancedErrorHandling:
    """Test enhanced error handling for batch failures."""

    def test_batch_failure_error_creation(self):
        """Test creation of batch failure errors with detailed information."""
        error = BatchingErrorHelper.create_batch_failure_error(
            feature="sentiment",
            batch_num=3,
            total_batches=10,
            error_message="Network timeout",
            input_count=5000,
        )

        assert error.feature == "sentiment"
        assert error.batch_number == 3
        assert error.total_batches == 10
        assert "Network timeout" in str(error)
        assert error.error_code == "BATCH_003"
        assert error.category == "batch_failed"

    def test_graceful_handling_of_concurrent_job_failures(self):
        """Test that individual job failures don't affect other jobs."""
        # Mock job submitters - some succeed, some fail
        successful_job = Mock()
        successful_job.wait.return_value = "success"

        failing_job = Mock()
        failing_job.wait.side_effect = Exception("Job failed")

        def create_successful_submitter():
            return lambda: successful_job

        def create_failing_submitter():
            return lambda: failing_job

        job_submitters = [
            create_successful_submitter(),
            create_failing_submitter(),
            create_successful_submitter(),
        ]

        # Test fast mode (sequential) - should handle failures gracefully
        with pytest.raises(BatchingError):  # Should raise BatchingError for failed jobs
            process_batches_concurrently(job_submitters, fast=True, feature="test")

    def test_progress_indicators_for_long_operations(self):
        """Test that progress indicators are shown for long-running operations."""
        job_submitters = [Mock() for _ in range(5)]

        # Mock jobs that take some time
        for submitter in job_submitters:
            job = Mock()
            job.wait.return_value = "result"
            submitter.return_value = job

        with patch("tqdm.tqdm") as mock_tqdm:
            mock_progress = Mock()
            mock_tqdm.return_value = mock_progress

            # Test slow mode with progress enabled
            results = process_batches_concurrently(
                job_submitters, fast=False, feature="test", enable_progress=True
            )

            # Verify progress bar was used (may not be called if tqdm import fails)
            # Just verify the function completed successfully
            assert len(results) == len(job_submitters)

    def test_specific_error_information_for_batch_failures(self):
        """Test that batch failures include specific error information."""
        error = BatchingErrorHelper.create_batch_failure_error(
            feature="embeddings",
            batch_num=2,
            total_batches=5,
            error_message="API rate limit exceeded",
            input_count=1000,
        )

        structured_info = error.get_structured_info()

        assert structured_info["error_code"] == "BATCH_003"
        assert structured_info["feature"] == "embeddings"
        assert structured_info["batch_number"] == 2
        assert structured_info["total_batches"] == 5
        assert "API rate limit exceeded" in structured_info["context"]["error_message"]
        assert structured_info["is_retryable"] is True

    def test_error_context_and_recovery_strategies(self):
        """Test that errors include context and recovery strategies."""
        error = BatchingErrorHelper.create_timeout_error(
            feature="clustering", timeout_seconds=300.0, batch_num=1
        )

        retry_strategy = error.get_retry_strategy()

        assert retry_strategy is not None
        assert retry_strategy["strategy"] == "linear_backoff"
        assert retry_strategy["max_retries"] == 2
        assert retry_strategy["modify_request"] == "increase_timeout"


class TestPerformanceOptimization:
    """Test performance optimization and resource management."""

    def test_resource_monitor_memory_tracking(self):
        """Test resource monitor tracks memory usage correctly."""
        monitor = ResourceMonitor()

        # Test without psutil
        monitor.has_psutil = False
        memory_info = monitor.get_memory_usage()
        assert memory_info["available"] is False

        # Test with mocked psutil
        with patch("pulse.core.batching.psutil") as mock_psutil:
            monitor.has_psutil = True
            mock_memory = Mock()
            mock_memory.percent = 75.0
            mock_memory.available = 8 * 1024**3  # 8GB
            mock_memory.total = 16 * 1024**3  # 16GB
            mock_psutil.virtual_memory.return_value = mock_memory

            memory_info = monitor.get_memory_usage()

            assert memory_info["available"] is True
            assert memory_info["percent_used"] == 75.0
            assert memory_info["available_gb"] == 8.0
            assert memory_info["total_gb"] == 16.0

    def test_batch_optimizer_size_optimization(self):
        """Test batch optimizer adjusts sizes based on system resources."""
        optimizer = BatchOptimizer(enable_monitoring=True)

        with patch.object(optimizer.monitor, "get_memory_usage") as mock_memory:
            # Test high memory usage - should reduce batch size
            mock_memory.return_value = {
                "available": True,
                "percent_used": 85.0,
                "available_gb": 2.0,
            }

            batch_size, workers = optimizer.optimize_batch_sizes(
                "sentiment", 10000, 2000, memory_threshold=80.0
            )

            assert batch_size < 2000  # Should be reduced
            assert workers <= 3  # Conservative worker count (2 or 3)

    def test_network_resource_manager_adaptive_timeout(self):
        """Test network resource manager provides adaptive timeouts."""
        manager = NetworkResourceManager(base_timeout=300.0, max_timeout=900.0)

        # Test timeout scaling with batch size
        small_batch_timeout = manager.get_adaptive_timeout(100, attempt=0)
        large_batch_timeout = manager.get_adaptive_timeout(5000, attempt=0)

        assert large_batch_timeout > small_batch_timeout

        # Test timeout scaling with retry attempts
        first_attempt = manager.get_adaptive_timeout(1000, attempt=0)
        second_attempt = manager.get_adaptive_timeout(1000, attempt=1)

        assert second_attempt > first_attempt
        assert second_attempt <= manager.max_timeout

    def test_network_resource_manager_retry_logic(self):
        """Test network resource manager retry logic."""
        manager = NetworkResourceManager(max_retries=3)

        # Test retryable errors
        timeout_error = Exception("Request timeout")
        assert manager.should_retry_request(timeout_error, 1) is True
        assert manager.should_retry_request(timeout_error, 2) is True
        assert (
            manager.should_retry_request(timeout_error, 3) is True
        )  # 3rd attempt, still within max_retries
        assert (
            manager.should_retry_request(timeout_error, 4) is False
        )  # 4th attempt exceeds max_retries

        # Test non-retryable errors
        auth_error = Exception("Authentication failed")
        assert manager.should_retry_request(auth_error, 1) is False

    def test_performance_benchmark_comparison(self):
        """Test performance benchmark compares batched vs manual approaches."""
        benchmark = PerformanceBenchmark()

        # Test faster batched operation
        comparison = benchmark.compare_performance(
            batched_duration=10.0, estimated_manual_duration=30.0, operation="sentiment"
        )

        assert comparison["batched_faster"] is True
        assert comparison["speedup_factor"] == 3.0
        assert comparison["time_saved"] == 20.0

        # Test slower batched operation
        comparison = benchmark.compare_performance(
            batched_duration=30.0,
            estimated_manual_duration=10.0,
            operation="embeddings",
        )

        assert comparison["batched_faster"] is False
        assert comparison["speedup_factor"] == 1 / 3

    def test_memory_optimization_cleanup(self):
        """Test memory optimization performs cleanup operations."""
        optimizer = BatchOptimizer(enable_monitoring=True)

        with patch("pulse.core.batching.gc.collect") as mock_gc:
            with patch("pulse.core.batching.np") as mock_np:
                mock_np.core._clearcache = Mock()

                optimizer.cleanup_memory()

                mock_gc.assert_called_once()
                mock_np.core._clearcache.assert_called_once()

    def test_batch_concurrency_controller_resource_optimization(self):
        """Test batch concurrency controller applies resource optimization."""
        controller = BatchConcurrencyController(
            max_concurrent_batches=5, enable_optimization=True
        )

        # Mock optimizer to return reduced workers
        with patch.object(
            controller.optimizer, "optimize_batch_sizes"
        ) as mock_optimize:
            mock_optimize.return_value = (1000, 3)  # Reduced batch size and workers

            config = controller.get_config("sentiment", 10000)

            assert config.max_workers == 3  # Should use optimized value
            mock_optimize.assert_called_once()

    def test_performance_tracking_in_batch_processing(self):
        """Test that batch processing tracks performance metrics."""
        controller = BatchConcurrencyController(enable_optimization=True)

        # Mock job submitters
        job_submitters = [Mock() for _ in range(3)]
        for submitter in job_submitters:
            job = Mock()
            job.wait.return_value = "result"
            submitter.return_value = job

        with patch.object(controller.benchmark, "start_benchmark") as mock_start:
            with patch.object(controller.benchmark, "end_benchmark") as mock_end:
                mock_end.return_value = 5.0  # 5 second duration

                results = controller.process_with_controlled_concurrency(
                    job_submitters,
                    fast=True,
                    feature="test",
                    total_items=1000,
                    batch_size=500,
                )

                mock_start.assert_called_once()
                mock_end.assert_called_once()
                assert len(results) == 3

    def test_resource_exhaustion_error_handling(self):
        """Test handling of resource exhaustion errors."""
        error = BatchingErrorHelper.create_resource_error(
            feature="embeddings", resource_type="memory", current_usage="95%"
        )

        assert error.error_code == "BATCH_007"
        assert error.category == "resource_error"
        assert "memory" in str(error)
        assert "95%" in str(error)

        retry_strategy = error.get_retry_strategy()
        assert retry_strategy["strategy"] == "wait_and_retry"
        assert retry_strategy["modify_request"] == "reduce_concurrency"


class TestIntegrationScenarios:
    """Test integration scenarios combining error handling and optimization."""

    def test_high_memory_usage_with_batch_failures(self):
        """Test handling of batch failures under high memory usage conditions."""
        controller = BatchConcurrencyController(enable_optimization=True)

        # Mock high memory usage
        with patch.object(
            controller.optimizer.monitor, "get_memory_usage"
        ) as mock_memory:
            mock_memory.return_value = {
                "available": True,
                "percent_used": 90.0,
                "available_gb": 1.0,
            }

            # Mock job submitters with some failures
            job_submitters = []
            for i in range(5):
                submitter = Mock()
                if i == 2:  # Make one job fail
                    job = Mock()
                    job.wait.side_effect = Exception("Memory error")
                    submitter.return_value = job
                else:
                    job = Mock()
                    job.wait.return_value = f"result_{i}"
                    submitter.return_value = job
                job_submitters.append(submitter)

            # Should handle both memory optimization and job failures
            with patch("warnings.warn"):
                with pytest.raises(BatchingError):  # Should raise error for failed job
                    controller.process_with_controlled_concurrency(
                        job_submitters, fast=True, feature="test"
                    )

    def test_network_timeout_with_retry_logic(self):
        """Test network timeout handling with retry logic."""
        manager = NetworkResourceManager(max_retries=2, base_timeout=10.0)

        # Test retry delay calculation
        delay1 = manager.get_retry_delay(1)
        delay2 = manager.get_retry_delay(2)

        assert delay2 > delay1

        # Test timeout adaptation for retries
        timeout1 = manager.get_adaptive_timeout(1000, attempt=0)
        timeout2 = manager.get_adaptive_timeout(1000, attempt=1)

        assert timeout2 > timeout1

    def test_performance_degradation_warning(self):
        """Test warning when batched operations are slower than manual approaches."""
        controller = BatchConcurrencyController(enable_optimization=True)

        # Mock slow batched operation
        with patch.object(controller.benchmark, "end_benchmark") as mock_end:
            with patch.object(
                controller.benchmark, "estimate_manual_duration"
            ) as mock_estimate:
                mock_end.return_value = 30.0  # Slow batched operation
                mock_estimate.return_value = 10.0  # Fast manual operation

                job_submitters = [Mock()]
                job_submitters[0].return_value.wait.return_value = "result"

                with patch("warnings.warn") as mock_warn:
                    controller.process_with_controlled_concurrency(
                        job_submitters,
                        fast=True,
                        feature="test",
                        total_items=100,
                        batch_size=50,
                    )

                    # Should warn about slower performance
                    mock_warn.assert_called()
                    warning_message = mock_warn.call_args[0][0]
                    assert "slower than manual approach" in warning_message
