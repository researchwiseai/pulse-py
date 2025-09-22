"""End-to-end integration tests for async components."""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
import tempfile
import os

from pulse.core.async_client import AsyncCoreClient
from pulse.analysis.async_analyzer import AsyncAnalyzer
from pulse.async_starters import (
    generate_themes_async,
    sentiment_analysis_async,
    theme_allocation_async,
    compare_similarity_async,
)
from pulse.async_dsl import AsyncWorkflow
from pulse.core.models import (
    ThemesResponse,
    SentimentResponse,
    SimilarityResponse,
    Theme,
    SentimentResult,
    UsageReport,
)
from pulse.analysis.processes import ThemeGeneration, SentimentProcess, ThemeAllocation


class TestAsyncEndToEndWorkflows:
    """Test complete async workflows from start to finish."""

    @pytest.fixture
    def sample_texts(self):
        """Sample text data for testing."""
        return [
            "I love this product, it's amazing!",
            "The service was terrible and slow.",
            "Great quality and fast delivery.",
            "Poor customer support experience.",
            "Excellent value for money.",
        ]

    @pytest.fixture
    def mock_async_client(self):
        """Create a comprehensive mock async client."""
        client = AsyncMock(spec=AsyncCoreClient)

        # Mock theme generation
        client.generate_themes.return_value = ThemesResponse(
            themes=[
                Theme(
                    shortLabel="positive",
                    label="Positive Experience",
                    description="Positive customer feedback",
                    representatives=["love", "amazing", "great", "excellent"],
                ),
                Theme(
                    shortLabel="negative",
                    label="Negative Experience",
                    description="Negative customer feedback",
                    representatives=["terrible", "poor", "slow"],
                ),
            ],
            usage=UsageReport(records=[], total=100),
        )

        # Mock sentiment analysis
        client.analyze_sentiment.return_value = SentimentResponse(
            results=[
                SentimentResult(sentiment="positive", confidence=0.95),
                SentimentResult(sentiment="negative", confidence=0.90),
                SentimentResult(sentiment="positive", confidence=0.85),
                SentimentResult(sentiment="negative", confidence=0.80),
                SentimentResult(sentiment="positive", confidence=0.88),
            ],
            usage=UsageReport(records=[], total=50),
        )

        # Mock similarity comparison
        client.compare_similarity.return_value = SimilarityResponse(
            scenario="cross",
            mode="matrix",
            n=10,
            similarity=[[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]],
            flattened=[0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5],
            usage=UsageReport(records=[], total=25),
        )

        return client

    @pytest.mark.asyncio
    async def test_async_analyzer_complete_workflow(
        self, sample_texts, mock_async_client
    ):
        """Test complete AsyncAnalyzer workflow with multiple processes."""
        processes = [
            ThemeGeneration(min_themes=2, max_themes=3),
            SentimentProcess(),
            ThemeAllocation(),
        ]

        async with AsyncAnalyzer(
            sample_texts,
            processes=processes,
            client=mock_async_client,
            use_cache=False,
        ) as analyzer:
            result = await analyzer.run()

            # Verify all processes completed
            assert hasattr(result, "theme_generation")
            assert hasattr(result, "sentiment")
            assert hasattr(result, "theme_allocation")

            # Verify client methods were called
            mock_async_client.generate_themes.assert_called_once()
            mock_async_client.analyze_sentiment.assert_called_once()
            mock_async_client.compare_similarity.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_dsl_complete_workflow(self, sample_texts, mock_async_client):
        """Test complete AsyncWorkflow DSL execution."""
        async with AsyncWorkflow() as workflow:
            # Build workflow
            workflow.source("reviews", sample_texts)
            workflow.theme_generation(source="reviews", min_themes=2, max_themes=3)
            workflow.sentiment(source="reviews")
            workflow.theme_allocation(inputs="reviews")

            # Execute workflow
            result = await workflow.run(client=mock_async_client)

            # Verify results
            assert hasattr(result, "theme_generation")
            assert hasattr(result, "sentiment")
            assert hasattr(result, "theme_allocation")

    @pytest.mark.asyncio
    async def test_async_starters_integration(self, sample_texts, mock_async_client):
        """Test integration between async starter functions."""
        # Generate themes
        themes_result = await generate_themes_async(
            sample_texts,
            client=mock_async_client,
            min_themes=2,
            max_themes=3,
        )
        assert isinstance(themes_result, ThemesResponse)
        assert len(themes_result.themes) == 2

        # Analyze sentiment
        sentiment_result = await sentiment_analysis_async(
            sample_texts,
            client=mock_async_client,
        )
        assert isinstance(sentiment_result, SentimentResponse)
        assert len(sentiment_result.results) == 5

        # Perform theme allocation using generated themes
        theme_labels = [theme.label for theme in themes_result.themes]
        allocation_result = await theme_allocation_async(
            sample_texts,
            themes=theme_labels,
            client=mock_async_client,
        )
        assert allocation_result is not None

    @pytest.mark.asyncio
    async def test_concurrent_async_operations(self, sample_texts, mock_async_client):
        """Test running multiple async operations concurrently."""
        # Create multiple async tasks
        tasks = [
            generate_themes_async(sample_texts, client=mock_async_client),
            sentiment_analysis_async(sample_texts, client=mock_async_client),
            compare_similarity_async(sample_texts, client=mock_async_client),
        ]

        # Run concurrently
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert isinstance(results[0], ThemesResponse)
        assert isinstance(results[1], SentimentResponse)
        assert isinstance(results[2], SimilarityResponse)

    @pytest.mark.asyncio
    async def test_async_caching_integration(self, sample_texts, mock_async_client):
        """Test async caching across multiple operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, "async_cache")

            # First run - should call client
            async with AsyncAnalyzer(
                sample_texts,
                processes=[SentimentProcess()],
                client=mock_async_client,
                cache_dir=cache_dir,
                use_cache=True,
            ) as analyzer1:
                result1 = await analyzer1.run()
                assert hasattr(result1, "sentiment")

            # Verify client was called
            assert mock_async_client.analyze_sentiment.call_count == 1

            # Second run - should use cache
            async with AsyncAnalyzer(
                sample_texts,
                processes=[SentimentProcess()],
                client=mock_async_client,
                cache_dir=cache_dir,
                use_cache=True,
            ) as analyzer2:
                result2 = await analyzer2.run()
                assert hasattr(result2, "sentiment")

            # Client should not be called again due to caching
            assert mock_async_client.analyze_sentiment.call_count == 1

    @pytest.mark.asyncio
    async def test_async_error_propagation(self, sample_texts):
        """Test error propagation through async components."""
        # Create client that raises errors
        error_client = AsyncMock(spec=AsyncCoreClient)
        error_client.analyze_sentiment.side_effect = RuntimeError("API Error")

        # Test error propagation through AsyncAnalyzer
        async with AsyncAnalyzer(
            sample_texts,
            processes=[SentimentProcess()],
            client=error_client,
        ) as analyzer:
            with pytest.raises(RuntimeError, match="API Error"):
                await analyzer.run()

        # Test error propagation through async starters
        with pytest.raises(RuntimeError, match="API Error"):
            await sentiment_analysis_async(sample_texts, client=error_client)

    @pytest.mark.asyncio
    async def test_async_timeout_handling(self, sample_texts):
        """Test timeout handling in async operations."""
        # Create client with slow responses
        slow_client = AsyncMock(spec=AsyncCoreClient)

        async def slow_sentiment(*args, **kwargs):
            await asyncio.sleep(2.0)  # Longer than timeout
            return SentimentResponse(results=[], usage=UsageReport(records=[], total=0))

        slow_client.analyze_sentiment = slow_sentiment

        # Test timeout with asyncio.wait_for
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                sentiment_analysis_async(sample_texts, client=slow_client), timeout=0.1
            )

    @pytest.mark.asyncio
    async def test_async_resource_cleanup(self, sample_texts, mock_async_client):
        """Test proper resource cleanup in async operations."""
        # Test AsyncAnalyzer cleanup
        analyzer = AsyncAnalyzer(
            sample_texts,
            processes=[SentimentProcess()],
            client=mock_async_client,
        )

        async with analyzer:
            await analyzer.run()
        # Context manager should handle cleanup

        # Test AsyncWorkflow cleanup
        workflow = AsyncWorkflow()
        workflow.source("texts", sample_texts)
        workflow.sentiment(source="texts")

        async with workflow:
            await workflow.run(client=mock_async_client)
        # Context manager should handle cleanup

    @pytest.mark.asyncio
    async def test_mixed_sync_async_compatibility(self, sample_texts):
        """Test compatibility between sync and async components."""
        # This test ensures that async components can work alongside
        # sync components without conflicts

        # Mock both sync and async clients
        mock_async_client = AsyncMock(spec=AsyncCoreClient)
        mock_async_client.analyze_sentiment.return_value = SentimentResponse(
            results=[SentimentResult(sentiment="positive", confidence=0.9)],
            usage=UsageReport(records=[], total=10),
        )

        # Use async components
        result = await sentiment_analysis_async(
            sample_texts[:1],  # Single text for simplicity
            client=mock_async_client,
        )

        assert isinstance(result, SentimentResponse)
        assert len(result.results) == 1

    @pytest.mark.asyncio
    async def test_async_batch_processing_integration(self, mock_async_client):
        """Test async batch processing with large datasets."""
        # Create large dataset that will trigger batching
        large_dataset = [f"Sample text {i}" for i in range(1000)]

        # Mock batch processing
        mock_async_client.create_embeddings.return_value = Mock()
        mock_async_client.create_embeddings.return_value.embeddings = [
            Mock(text=text, vector=[0.1, 0.2, 0.3]) for text in large_dataset
        ]

        # Test with AsyncAnalyzer
        async with AsyncAnalyzer(
            large_dataset,
            processes=[],  # No processes, just test data handling
            client=mock_async_client,
        ) as analyzer:
            # Should handle large dataset without issues
            assert len(analyzer.dataset) == 1000

    @pytest.mark.asyncio
    async def test_async_workflow_dependency_resolution(
        self, sample_texts, mock_async_client
    ):
        """Test automatic dependency resolution in async workflows."""
        async with AsyncWorkflow() as workflow:
            workflow.source("texts", sample_texts)
            # Theme allocation should automatically add theme generation
            workflow.theme_allocation(inputs="texts")

            result = await workflow.run(client=mock_async_client)

            # Both theme generation and allocation should be present
            assert hasattr(result, "theme_generation")
            assert hasattr(result, "theme_allocation")

            # Both client methods should have been called
            mock_async_client.generate_themes.assert_called_once()
            mock_async_client.compare_similarity.assert_called_once()


class TestAsyncPerformanceScenarios:
    """Test performance-related async scenarios."""

    @pytest.mark.asyncio
    async def test_high_concurrency_scenario(self):
        """Test high concurrency async operations."""
        # Create many concurrent tasks
        num_tasks = 50
        mock_client = AsyncMock(spec=AsyncCoreClient)
        mock_client.analyze_sentiment.return_value = SentimentResponse(
            results=[SentimentResult(sentiment="positive", confidence=0.9)],
            usage=UsageReport(records=[], total=1),
        )

        # Create concurrent sentiment analysis tasks
        tasks = [
            sentiment_analysis_async([f"text {i}"], client=mock_client)
            for i in range(num_tasks)
        ]

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)

        assert len(results) == num_tasks
        for result in results:
            assert isinstance(result, SentimentResponse)

    @pytest.mark.asyncio
    async def test_memory_efficient_streaming(self, mock_async_client):
        """Test memory-efficient processing of large datasets."""
        # Simulate processing large dataset in chunks
        chunk_size = 100
        total_items = 1000

        mock_async_client.analyze_sentiment.return_value = SentimentResponse(
            results=[SentimentResult(sentiment="positive", confidence=0.9)]
            * chunk_size,
            usage=UsageReport(records=[], total=chunk_size),
        )

        # Process in chunks
        all_results = []
        for i in range(0, total_items, chunk_size):
            chunk = [f"text {j}" for j in range(i, min(i + chunk_size, total_items))]
            result = await sentiment_analysis_async(chunk, client=mock_async_client)
            all_results.extend(result.results)

        assert len(all_results) == total_items

    @pytest.mark.asyncio
    async def test_async_rate_limiting_compliance(self):
        """Test that async operations respect rate limiting."""
        import time

        mock_client = AsyncMock(spec=AsyncCoreClient)
        call_times = []

        async def track_call_time(*args, **kwargs):
            call_times.append(time.time())
            return SentimentResponse(
                results=[SentimentResult(sentiment="positive", confidence=0.9)],
                usage=UsageReport(records=[], total=1),
            )

        mock_client.analyze_sentiment = track_call_time

        # Make multiple calls with rate limiting
        from pulse.core.async_concurrent import submit_and_gather_jobs

        async def create_submitter(text):
            async def submitter():
                # Simulate job creation
                job = AsyncMock()
                job.wait = AsyncMock(return_value={"sentiment": "positive"})
                return job

            return submitter

        submitters = [await create_submitter(f"text {i}") for i in range(5)]

        # Use rate-limited submission
        results = await submit_and_gather_jobs(
            submitters,
            max_concurrent=2,
            rate_limit_delay=0.1,
        )

        assert len(results) == 5


class TestAsyncErrorRecoveryScenarios:
    """Test error recovery in async scenarios."""

    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self, sample_texts):
        """Test recovery from partial failures in batch operations."""
        # Create client that fails on some requests
        unreliable_client = AsyncMock(spec=AsyncCoreClient)
        call_count = 0

        async def sometimes_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise RuntimeError("Simulated failure")
            return SentimentResponse(
                results=[SentimentResult(sentiment="positive", confidence=0.9)],
                usage=UsageReport(records=[], total=1),
            )

        unreliable_client.analyze_sentiment = sometimes_fail

        # Test with multiple concurrent operations and exception handling
        tasks = [
            sentiment_analysis_async([text], client=unreliable_client)
            for text in sample_texts
        ]

        # Use asyncio.gather with return_exceptions=True
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should have mix of successful results and exceptions
        successes = [r for r in results if isinstance(r, SentimentResponse)]
        failures = [r for r in results if isinstance(r, Exception)]

        assert len(successes) > 0
        assert len(failures) > 0
        assert len(successes) + len(failures) == len(sample_texts)

    @pytest.mark.asyncio
    async def test_retry_mechanism_integration(self):
        """Test retry mechanisms in async operations."""
        # Create client that succeeds after retries
        retry_client = AsyncMock(spec=AsyncCoreClient)
        attempt_count = 0

        async def succeed_after_retries(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise RuntimeError("Temporary failure")
            return SentimentResponse(
                results=[SentimentResult(sentiment="positive", confidence=0.9)],
                usage=UsageReport(records=[], total=1),
            )

        retry_client.analyze_sentiment = succeed_after_retries

        # Test with retry logic (would need to be implemented in the actual client)
        # For now, just test that the mechanism can handle retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await sentiment_analysis_async(["test"], client=retry_client)
                break
            except RuntimeError:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1)  # Brief delay before retry

        assert isinstance(result, SentimentResponse)
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, sample_texts):
        """Test graceful degradation when some async services are unavailable."""
        # Create client where some methods work and others don't
        partial_client = AsyncMock(spec=AsyncCoreClient)

        # Sentiment works
        partial_client.analyze_sentiment.return_value = SentimentResponse(
            results=[SentimentResult(sentiment="positive", confidence=0.9)]
            * len(sample_texts),
            usage=UsageReport(records=[], total=len(sample_texts)),
        )

        # Themes fail
        partial_client.generate_themes.side_effect = RuntimeError(
            "Themes service unavailable"
        )

        # Test that we can still get sentiment results even if themes fail
        sentiment_result = await sentiment_analysis_async(
            sample_texts, client=partial_client
        )
        assert isinstance(sentiment_result, SentimentResponse)

        # Test that theme generation fails as expected
        with pytest.raises(RuntimeError, match="Themes service unavailable"):
            await generate_themes_async(sample_texts, client=partial_client)
