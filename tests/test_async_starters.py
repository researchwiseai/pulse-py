"""Tests for async starter functions."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from pulse.async_starters import (
    generate_themes_async,
    sentiment_analysis_async,
    theme_allocation_async,
    compare_similarity_async,
    cluster_analysis_async,
    extract_elements_async,
    summarize_async,
    estimate_usage_async,
    get_strings,
)
from pulse.core.async_client import AsyncCoreClient
from pulse.core.async_jobs import AsyncJob
from pulse.core.models import (
    ThemesResponse,
    SentimentResponse,
    SimilarityResponse,
    ClusteringResponse,
    ExtractionsResponse,
    SummariesResponse,
    UsageEstimateResponse,
)
from pulse.analysis.results import ThemeAllocationResult


class TestAsyncStarters:
    """Test async starter functions."""

    @pytest.mark.asyncio
    async def test_generate_themes_async_with_client(self):
        """Test generate_themes_async with provided client."""
        mock_client = AsyncMock(spec=AsyncCoreClient)
        mock_response = MagicMock(spec=ThemesResponse)
        mock_client.generate_themes.return_value = mock_response

        result = await generate_themes_async(
            ["test text 1", "test text 2"],
            client=mock_client,
            min_themes=2,
            max_themes=5,
        )

        assert result == mock_response
        mock_client.generate_themes.assert_called_once_with(
            ["test text 1", "test text 2"],
            min_themes=2,
            max_themes=5,
            context=None,
            version=None,
            prune=None,
            interactive=None,
            initial_sets=None,
            fast=True,  # len(texts) <= 200
            await_job_result=True,
        )

    @pytest.mark.asyncio
    async def test_generate_themes_async_without_client(self):
        """Test generate_themes_async without provided client."""
        mock_response = MagicMock(spec=ThemesResponse)

        with patch("pulse.async_starters.AsyncCoreClient") as mock_client_class:
            mock_client = AsyncMock(spec=AsyncCoreClient)
            mock_client.generate_themes.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await generate_themes_async(
                ["test text 1", "test text 2"],
                min_themes=2,
                max_themes=5,
            )

            assert result == mock_response
            mock_client_class.assert_called_once_with(auth=None)
            mock_client.generate_themes.assert_called_once()

    @pytest.mark.asyncio
    async def test_sentiment_analysis_async_with_client(self):
        """Test sentiment_analysis_async with provided client."""
        mock_client = AsyncMock(spec=AsyncCoreClient)
        mock_response = MagicMock(spec=SentimentResponse)
        mock_client.analyze_sentiment.return_value = mock_response

        result = await sentiment_analysis_async(
            ["positive text", "negative text"],
            client=mock_client,
            version="v1",
        )

        assert result == mock_response
        mock_client.analyze_sentiment.assert_called_once_with(
            ["positive text", "negative text"],
            version="v1",
            fast=True,  # len(texts) <= 200
            await_job_result=True,
        )

    @pytest.mark.asyncio
    async def test_sentiment_analysis_async_without_client(self):
        """Test sentiment_analysis_async without provided client."""
        mock_response = MagicMock(spec=SentimentResponse)

        with patch("pulse.async_starters.AsyncCoreClient") as mock_client_class:
            mock_client = AsyncMock(spec=AsyncCoreClient)
            mock_client.analyze_sentiment.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await sentiment_analysis_async(
                ["positive text", "negative text"],
                version="v1",
            )

            assert result == mock_response
            mock_client_class.assert_called_once_with(auth=None)
            mock_client.analyze_sentiment.assert_called_once()

    @pytest.mark.asyncio
    async def test_theme_allocation_async_with_client(self):
        """Test theme_allocation_async with provided client."""
        mock_client = AsyncMock(spec=AsyncCoreClient)
        mock_result = MagicMock(spec=ThemeAllocationResult)

        with patch("pulse.async_starters.AsyncAnalyzer") as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            mock_analyzer.__aenter__.return_value = mock_analyzer
            mock_analyzer.__aexit__.return_value = None
            mock_response = MagicMock()
            mock_response.theme_allocation = mock_result
            mock_analyzer.run.return_value = mock_response
            mock_analyzer_class.return_value = mock_analyzer

            result = await theme_allocation_async(
                ["text 1", "text 2"],
                client=mock_client,
                themes=["theme1", "theme2"],
            )

            assert result == mock_result
            mock_analyzer_class.assert_called_once()
            mock_analyzer.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_compare_similarity_async_cross_similarity(self):
        """Test compare_similarity_async with cross-similarity."""
        mock_client = AsyncMock(spec=AsyncCoreClient)
        mock_response = MagicMock(spec=SimilarityResponse)
        mock_client.compare_similarity.return_value = mock_response

        result = await compare_similarity_async(
            [],  # input_data not used for cross-similarity
            set_a=["text 1", "text 2"],
            set_b=["text 3", "text 4"],
            client=mock_client,
            flatten=True,
        )

        assert result == mock_response
        mock_client.compare_similarity.assert_called_once_with(
            set_a=["text 1", "text 2"],
            set_b=["text 3", "text 4"],
            split=None,
            flatten=True,
            version=None,
            fast=True,  # 2 * 2 <= 20_000
            await_job_result=True,
        )

    @pytest.mark.asyncio
    async def test_compare_similarity_async_self_similarity(self):
        """Test compare_similarity_async with self-similarity."""
        mock_client = AsyncMock(spec=AsyncCoreClient)
        mock_response = MagicMock(spec=SimilarityResponse)
        mock_client.compare_similarity.return_value = mock_response

        result = await compare_similarity_async(
            ["text 1", "text 2", "text 3"],
            client=mock_client,
            flatten=False,
        )

        assert result == mock_response
        mock_client.compare_similarity.assert_called_once_with(
            set=["text 1", "text 2", "text 3"],
            split=None,
            flatten=False,
            version=None,
            fast=True,  # len(texts) <= 500
            await_job_result=True,
        )

    @pytest.mark.asyncio
    async def test_cluster_analysis_async_with_client(self):
        """Test cluster_analysis_async with provided client."""
        mock_client = AsyncMock(spec=AsyncCoreClient)
        mock_response = MagicMock(spec=ClusteringResponse)
        mock_client.cluster_texts.return_value = mock_response

        result = await cluster_analysis_async(
            ["text 1", "text 2", "text 3"],
            k=2,
            algorithm="kmeans",
            client=mock_client,
        )

        assert result == mock_response
        mock_client.cluster_texts.assert_called_once_with(
            inputs=["text 1", "text 2", "text 3"],
            k=2,
            algorithm="kmeans",
            fast=True,  # len(texts) <= 500
            await_job_result=True,
        )

    @pytest.mark.asyncio
    async def test_extract_elements_async_with_client(self):
        """Test extract_elements_async with provided client."""
        mock_client = AsyncMock(spec=AsyncCoreClient)
        mock_response = MagicMock(spec=ExtractionsResponse)
        mock_client.extract_elements.return_value = mock_response

        result = await extract_elements_async(
            ["text with entities", "another text"],
            dictionary=["entity1", "entity2"],
            type="named-entities",
            client=mock_client,
        )

        assert result == mock_response
        mock_client.extract_elements.assert_called_once_with(
            inputs=["text with entities", "another text"],
            dictionary=["entity1", "entity2"],
            type="named-entities",
            expand_dictionary=False,
            expand_dictionary_limit=None,
            version=None,
            fast=True,  # len(texts) <= 200
            await_job_result=True,
        )

    @pytest.mark.asyncio
    async def test_summarize_async_with_client(self):
        """Test summarize_async with provided client."""
        mock_client = AsyncMock(spec=AsyncCoreClient)
        mock_response = MagicMock(spec=SummariesResponse)
        mock_client.generate_summary.return_value = mock_response

        result = await summarize_async(
            ["long text to summarize", "another long text"],
            question="What is the main point?",
            length="short",
            client=mock_client,
        )

        assert result == mock_response
        mock_client.generate_summary.assert_called_once_with(
            ["long text to summarize", "another long text"],
            "What is the main point?",
            length="short",
            preset=None,
            fast=True,  # len(texts) <= 200
            await_job_result=True,
        )

    @pytest.mark.asyncio
    async def test_estimate_usage_async_with_client(self):
        """Test estimate_usage_async with provided client."""
        mock_client = AsyncMock(spec=AsyncCoreClient)
        mock_response = MagicMock(spec=UsageEstimateResponse)
        mock_client.estimate_usage.return_value = mock_response

        result = await estimate_usage_async(
            "sentiment",
            ["text 1", "text 2"],
            client=mock_client,
        )

        assert result == mock_response
        mock_client.estimate_usage.assert_called_once_with(
            feature="sentiment",
            inputs=["text 1", "text 2"],
        )

    @pytest.mark.asyncio
    async def test_estimate_usage_async_without_client(self):
        """Test estimate_usage_async without provided client."""
        mock_response = MagicMock(spec=UsageEstimateResponse)

        with patch("pulse.async_starters.AsyncCoreClient") as mock_client_class:
            mock_client = AsyncMock(spec=AsyncCoreClient)
            mock_client.estimate_usage.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await estimate_usage_async(
                "themes",
                ["text 1", "text 2"],
            )

            assert result == mock_response
            mock_client_class.assert_called_once_with()
            mock_client.estimate_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_job_return(self):
        """Test that async starters can return AsyncJob when await_job_result=False."""
        mock_client = AsyncMock(spec=AsyncCoreClient)
        mock_job = MagicMock(spec=AsyncJob)
        mock_client.generate_themes.return_value = mock_job

        result = await generate_themes_async(
            ["test text"],
            client=mock_client,
            await_job_result=False,
        )

        assert result == mock_job
        mock_client.generate_themes.assert_called_once()
        args, kwargs = mock_client.generate_themes.call_args
        assert kwargs["await_job_result"] is False

    def test_get_strings_with_list(self):
        """Test get_strings with list input."""
        texts = ["text 1", "text 2", "text 3"]
        result = get_strings(texts)
        assert result == texts

    def test_get_strings_with_invalid_input(self):
        """Test get_strings with invalid input."""
        with pytest.raises(
            ValueError, match="Provide a list of strings or a valid file path"
        ):
            get_strings("nonexistent_file.txt")

        with pytest.raises(
            ValueError, match="Provide a list of strings or a valid file path"
        ):
            get_strings(123)  # Invalid type

    @pytest.mark.asyncio
    async def test_fast_flag_calculation(self):
        """Test that fast flag is calculated correctly based on input size."""
        mock_client = AsyncMock(spec=AsyncCoreClient)
        mock_response = MagicMock()
        mock_client.generate_themes.return_value = mock_response

        # Small input should use fast=True
        small_texts = ["text"] * 100
        await generate_themes_async(small_texts, client=mock_client)
        args, kwargs = mock_client.generate_themes.call_args
        assert kwargs["fast"] is True

        # Large input should use fast=False
        large_texts = ["text"] * 300
        await generate_themes_async(large_texts, client=mock_client)
        args, kwargs = mock_client.generate_themes.call_args
        assert kwargs["fast"] is False

    @pytest.mark.asyncio
    async def test_context_manager_usage(self):
        """Test async context managers are used correctly when client not provided."""
        mock_response = MagicMock(spec=SentimentResponse)

        with patch("pulse.async_starters.AsyncCoreClient") as mock_client_class:
            mock_client = AsyncMock(spec=AsyncCoreClient)
            mock_client.analyze_sentiment.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await sentiment_analysis_async(["test text"])

            # Verify context manager methods were called
            mock_client.__aenter__.assert_called_once()
            mock_client.__aexit__.assert_called_once()
            assert result == mock_response
