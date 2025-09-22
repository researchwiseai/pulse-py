"""Tests for AsyncAnalyzer functionality."""

import pytest
import tempfile
import os
from unittest.mock import AsyncMock
import pandas as pd

from pulse.analysis.async_analyzer import AsyncAnalyzer, AsyncAnalysisResult
from pulse.analysis.processes import (
    ThemeGeneration,
    SentimentProcess,
    ThemeAllocation,
    SimilarityProcess,
    ThemeExtraction,
)
from pulse.core.async_client import AsyncCoreClient
from pulse.core.models import (
    ThemesResponse,
    SentimentResponse,
    SimilarityResponse,
    ExtractionsResponse,
    ClusteringResponse,
    Theme,
    SentimentResult,
    UsageReport,
    Cluster,
)


@pytest.fixture
def sample_texts():
    """Sample text data for testing."""
    return ["This is great!", "I love this product", "Not bad", "Could be better"]


@pytest.fixture
def mock_async_client():
    """Mock AsyncCoreClient for testing."""
    client = AsyncMock(spec=AsyncCoreClient)

    # Mock theme generation response
    client.generate_themes.return_value = ThemesResponse(
        themes=[
            Theme(
                shortLabel="positive",
                label="Positive Sentiment",
                description="Themes expressing positive sentiment",
                representatives=["great", "love"],
            ),
            Theme(
                shortLabel="negative",
                label="Negative Sentiment",
                description="Themes expressing negative sentiment",
                representatives=["bad", "worse"],
            ),
        ],
        usage=UsageReport(records=[], total=100),
    )

    # Mock sentiment analysis response
    client.analyze_sentiment.return_value = SentimentResponse(
        results=[
            SentimentResult(sentiment="positive", confidence=0.9),
            SentimentResult(sentiment="positive", confidence=0.8),
            SentimentResult(sentiment="neutral", confidence=0.6),
            SentimentResult(sentiment="negative", confidence=0.7),
        ],
        usage=UsageReport(records=[], total=50),
    )

    # Mock similarity response
    client.compare_similarity.return_value = SimilarityResponse(
        scenario="cross",
        mode="matrix",
        n=4,
        flattened=[0.9, 0.1, 0.8, 0.2, 0.3, 0.7, 0.2, 0.8],
        matrix=[[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]],
        usage=UsageReport(records=[], total=25),
    )

    # Mock extraction response
    client.extract_elements.return_value = ExtractionsResponse(
        columns=[
            ExtractionsResponse.ExtractionColumn(category="positive", term="great"),
            ExtractionsResponse.ExtractionColumn(category="positive", term="love"),
        ],
        matrix=[["great"], ["love"], [], []],
        usage=UsageReport(records=[], total=30),
    )

    # Mock clustering response
    client.cluster_texts.return_value = ClusteringResponse(
        algorithm="kmeans",
        clusters=[
            Cluster(clusterId=0, items=["This is great!", "I love this product"]),
            Cluster(clusterId=1, items=["Not bad", "Could be better"]),
        ],
        usage=UsageReport(records=[], total=40),
    )

    return client


@pytest.mark.asyncio
async def test_async_analyzer_initialization(sample_texts):
    """Test AsyncAnalyzer initialization."""
    analyzer = AsyncAnalyzer(sample_texts)

    assert len(analyzer.dataset) == 4
    assert analyzer.fast is False
    assert analyzer.use_cache is True
    assert analyzer.processes == []
    assert isinstance(analyzer.client, AsyncCoreClient)

    await analyzer.close()


@pytest.mark.asyncio
async def test_async_analyzer_with_pandas_series(sample_texts):
    """Test AsyncAnalyzer with pandas Series input."""
    series = pd.Series(sample_texts)
    analyzer = AsyncAnalyzer(series)

    assert len(analyzer.dataset) == 4
    assert analyzer.dataset.equals(series)

    await analyzer.close()


@pytest.mark.asyncio
async def test_async_analyzer_with_processes(sample_texts, mock_async_client):
    """Test AsyncAnalyzer with processes."""
    processes = [ThemeGeneration(), SentimentProcess()]
    analyzer = AsyncAnalyzer(
        sample_texts,
        processes=processes,
        client=mock_async_client,
    )

    assert len(analyzer.processes) == 2
    assert analyzer.processes[0].id == "theme_generation"
    assert analyzer.processes[1].id == "sentiment"

    await analyzer.close()


@pytest.mark.asyncio
async def test_async_analyzer_dependency_resolution(sample_texts, mock_async_client):
    """Test automatic dependency resolution."""
    # ThemeAllocation depends on ThemeGeneration
    processes = [ThemeAllocation()]
    analyzer = AsyncAnalyzer(
        sample_texts,
        processes=processes,
        client=mock_async_client,
    )

    # Should automatically include ThemeGeneration
    assert len(analyzer.processes) == 2
    assert analyzer.processes[0].id == "theme_generation"
    assert analyzer.processes[1].id == "theme_allocation"

    await analyzer.close()


@pytest.mark.asyncio
async def test_async_analyzer_run_theme_generation(sample_texts, mock_async_client):
    """Test running theme generation process."""
    processes = [ThemeGeneration(min_themes=2, max_themes=5)]
    analyzer = AsyncAnalyzer(
        sample_texts,
        processes=processes,
        client=mock_async_client,
    )

    result = await analyzer.run()

    assert isinstance(result, AsyncAnalysisResult)
    assert hasattr(result, "theme_generation")

    # Verify client was called correctly
    mock_async_client.generate_themes.assert_called_once()
    call_args = mock_async_client.generate_themes.call_args
    assert call_args[1]["min_themes"] == 2
    assert call_args[1]["max_themes"] == 5

    await analyzer.close()


@pytest.mark.asyncio
async def test_async_analyzer_run_sentiment(sample_texts, mock_async_client):
    """Test running sentiment analysis process."""
    processes = [SentimentProcess()]
    analyzer = AsyncAnalyzer(
        sample_texts,
        processes=processes,
        client=mock_async_client,
    )

    result = await analyzer.run()

    assert isinstance(result, AsyncAnalysisResult)
    assert hasattr(result, "sentiment")

    # Verify client was called correctly
    mock_async_client.analyze_sentiment.assert_called_once()

    await analyzer.close()


@pytest.mark.asyncio
async def test_async_analyzer_run_theme_allocation(sample_texts, mock_async_client):
    """Test running theme allocation process with dependency."""
    processes = [ThemeAllocation()]
    analyzer = AsyncAnalyzer(
        sample_texts,
        processes=processes,
        client=mock_async_client,
    )

    result = await analyzer.run()

    assert isinstance(result, AsyncAnalysisResult)
    assert hasattr(result, "theme_generation")
    assert hasattr(result, "theme_allocation")

    # Verify both clients were called
    mock_async_client.generate_themes.assert_called_once()
    mock_async_client.compare_similarity.assert_called_once()

    await analyzer.close()


@pytest.mark.asyncio
async def test_async_analyzer_run_similarity(sample_texts, mock_async_client):
    """Test running similarity process."""
    processes = [SimilarityProcess()]
    analyzer = AsyncAnalyzer(
        sample_texts,
        processes=processes,
        client=mock_async_client,
    )

    result = await analyzer.run()

    assert isinstance(result, AsyncAnalysisResult)
    assert hasattr(result, "similarity")

    # Verify client was called correctly
    mock_async_client.compare_similarity.assert_called_once()

    await analyzer.close()


@pytest.mark.asyncio
async def test_async_analyzer_run_extraction(sample_texts, mock_async_client):
    """Test running theme extraction process."""
    processes = [ThemeExtraction()]
    analyzer = AsyncAnalyzer(
        sample_texts,
        processes=processes,
        client=mock_async_client,
    )

    result = await analyzer.run()

    assert isinstance(result, AsyncAnalysisResult)
    assert hasattr(result, "theme_generation")
    assert hasattr(result, "theme_extraction")

    # Verify both clients were called
    mock_async_client.generate_themes.assert_called_once()
    mock_async_client.extract_elements.assert_called_once()

    await analyzer.close()


@pytest.mark.asyncio
async def test_async_analyzer_run_clustering(sample_texts, mock_async_client):
    """Test running clustering process."""
    processes = [Cluster(k=2, algorithm="kmeans")]
    analyzer = AsyncAnalyzer(
        sample_texts,
        processes=processes,
        client=mock_async_client,
    )

    result = await analyzer.run()

    assert isinstance(result, AsyncAnalysisResult)
    assert hasattr(result, "cluster")

    # Verify client was called correctly
    mock_async_client.cluster_texts.assert_called_once()
    call_args = mock_async_client.cluster_texts.call_args
    assert call_args[1]["k"] == 2
    assert call_args[1]["algorithm"] == "kmeans"

    await analyzer.close()


@pytest.mark.asyncio
async def test_async_analyzer_caching():
    """Test caching functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, "test_cache")

        # Mock client
        mock_client = AsyncMock(spec=AsyncCoreClient)
        mock_client.analyze_sentiment.return_value = SentimentResponse(
            results=[SentimentResult(sentiment="positive", confidence=0.9)],
            usage=UsageReport(records=[], total=10),
        )

        processes = [SentimentProcess()]

        # First run - should call client
        analyzer1 = AsyncAnalyzer(
            ["test text"],
            processes=processes,
            client=mock_client,
            cache_dir=cache_dir,
            use_cache=True,
        )

        await analyzer1.run()
        assert mock_client.analyze_sentiment.call_count == 1
        await analyzer1.close()

        # Second run - should use cache
        analyzer2 = AsyncAnalyzer(
            ["test text"],
            processes=processes,
            client=mock_client,
            cache_dir=cache_dir,
            use_cache=True,
        )

        await analyzer2.run()
        # Should not call client again due to caching
        assert mock_client.analyze_sentiment.call_count == 1
        await analyzer2.close()


@pytest.mark.asyncio
async def test_async_analyzer_context_manager(sample_texts, mock_async_client):
    """Test AsyncAnalyzer as async context manager."""
    processes = [SentimentProcess()]

    async with AsyncAnalyzer(
        sample_texts,
        processes=processes,
        client=mock_async_client,
    ) as analyzer:
        result = await analyzer.run()
        assert isinstance(result, AsyncAnalysisResult)
        assert hasattr(result, "sentiment")

    # Context manager should have called close
    # We can't easily verify this with the mock, but the test passing
    # means no exceptions


@pytest.mark.asyncio
async def test_async_analyzer_clear_cache():
    """Test cache clearing functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, "test_cache")

        analyzer = AsyncAnalyzer(
            ["test"],
            cache_dir=cache_dir,
            use_cache=True,
        )

        # Should not raise any exceptions
        analyzer.clear_cache()

        await analyzer.close()


@pytest.mark.asyncio
async def test_async_analysis_result_attribute_access():
    """Test AsyncAnalysisResult attribute access."""
    results = {
        "sentiment": "mock_sentiment_result",
        "themes": "mock_themes_result",
    }

    result = AsyncAnalysisResult(results)

    assert result.sentiment == "mock_sentiment_result"
    assert result.themes == "mock_themes_result"

    with pytest.raises(AttributeError):
        _ = result.nonexistent


@pytest.mark.asyncio
async def test_async_analyzer_fast_mode(sample_texts, mock_async_client):
    """Test AsyncAnalyzer with fast mode enabled."""
    processes = [ThemeGeneration(), SentimentProcess()]
    analyzer = AsyncAnalyzer(
        sample_texts,
        processes=processes,
        client=mock_async_client,
        fast=True,
    )

    await analyzer.run()

    # Verify fast=True was passed to client methods
    theme_call = mock_async_client.generate_themes.call_args
    sentiment_call = mock_async_client.analyze_sentiment.call_args

    assert theme_call[1]["fast"] is True
    assert sentiment_call[1]["fast"] is True

    await analyzer.close()


@pytest.mark.asyncio
async def test_async_analyzer_error_handling(sample_texts):
    """Test error handling in AsyncAnalyzer."""
    # Mock client that raises an exception
    mock_client = AsyncMock(spec=AsyncCoreClient)
    mock_client.analyze_sentiment.side_effect = Exception("API Error")

    processes = [SentimentProcess()]
    analyzer = AsyncAnalyzer(
        sample_texts,
        processes=processes,
        client=mock_client,
    )

    with pytest.raises(Exception, match="API Error"):
        await analyzer.run()

    await analyzer.close()


@pytest.mark.asyncio
async def test_async_analyzer_missing_dependency_error(sample_texts, mock_async_client):
    """Test error when dependency is missing."""

    # Create a mock process with unknown dependency
    class MockProcess:
        id = "mock_process"
        depends_on = ("unknown_dependency",)

        def run(self, ctx):
            return "mock_result"

    with pytest.raises(
        RuntimeError, match="Missing dependency process 'unknown_dependency'"
    ):
        AsyncAnalyzer(
            sample_texts,
            processes=[MockProcess()],
            client=mock_async_client,
        )
