"""Tests for AsyncWorkflow DSL builder."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from pulse.async_dsl import AsyncWorkflow
from pulse.core.async_client import AsyncCoreClient
from pulse.core.models import (
    ThemesResponse,
    SentimentResponse,
    SimilarityResponse,
    ClusteringResponse,
    ExtractionsResponse,
)


@pytest.fixture
def mock_async_client():
    """Create a mock async client for testing."""
    client = AsyncMock(spec=AsyncCoreClient)

    # Mock theme generation response
    client.generate_themes.return_value = ThemesResponse(
        themes=[
            {
                "shortLabel": "Food Quality",
                "label": "Food Quality and Taste",
                "description": "Comments about food quality and taste",
                "representatives": ["tasty", "delicious"],
            },
            {
                "shortLabel": "Service",
                "label": "Customer Service",
                "description": "Comments about customer service experience",
                "representatives": ["friendly", "helpful"],
            },
        ]
    )

    # Mock sentiment response
    client.analyze_sentiment.return_value = SentimentResponse(
        sentiments=[
            {"sentiment": "positive", "confidence": 0.9},
            {"sentiment": "negative", "confidence": 0.8},
        ]
    )

    # Mock similarity response
    client.compare_similarity.return_value = SimilarityResponse(
        similarity=[[0.8, 0.2], [0.3, 0.9]]
    )

    # Mock clustering response
    client.cluster_texts.return_value = ClusteringResponse(
        clusters=[
            {"cluster": 0, "texts": ["text1"]},
            {"cluster": 1, "texts": ["text2"]},
        ]
    )

    # Mock extraction response
    client.extract_elements.return_value = ExtractionsResponse(
        extractions=[["element1"], ["element2"]]
    )

    return client


@pytest.mark.asyncio
async def test_async_workflow_basic_creation():
    """Test basic AsyncWorkflow creation and method chaining."""
    wf = AsyncWorkflow()

    # Test method chaining returns AsyncWorkflow
    result = wf.theme_generation(min_themes=2, max_themes=5)
    assert isinstance(result, AsyncWorkflow)
    assert result is wf  # Should return self for chaining

    # Test adding sentiment step
    result = wf.sentiment()
    assert isinstance(result, AsyncWorkflow)

    # Verify processes were added
    assert len(wf._processes) == 2
    assert wf._processes[0].id == "theme_generation"
    assert wf._processes[1].id == "sentiment"


@pytest.mark.asyncio
async def test_async_workflow_source_registration():
    """Test source registration and validation."""
    wf = AsyncWorkflow()

    # Test source registration
    texts = ["text1", "text2"]
    result = wf.source("comments", texts)
    assert isinstance(result, AsyncWorkflow)
    assert "comments" in wf._sources
    assert wf._sources["comments"] == texts

    # Test duplicate source registration raises error
    with pytest.raises(ValueError, match="Source 'comments' already registered"):
        wf.source("comments", ["other texts"])


@pytest.mark.asyncio
async def test_async_workflow_process_naming():
    """Test process naming and aliasing."""
    wf = AsyncWorkflow()

    # Test custom naming
    wf.theme_generation(name="custom_themes")
    assert wf._processes[0].id == "custom_themes"

    # Test auto-aliasing for duplicate process types
    wf.theme_generation()  # Should get alias theme_generation_2
    assert wf._processes[1].id == "theme_generation_2"

    # Test duplicate name raises error
    with pytest.raises(
        ValueError, match="Process name 'custom_themes' already registered"
    ):
        wf.sentiment(name="custom_themes")


@pytest.mark.asyncio
async def test_async_workflow_input_validation():
    """Test input source validation."""
    wf = AsyncWorkflow()

    # Test invalid source reference
    with pytest.raises(
        ValueError, match="Unknown source for theme_generation: 'nonexistent'"
    ):
        wf.theme_generation(source="nonexistent")

    # Test valid source reference after registration
    wf.source("texts", ["text1", "text2"])
    wf.theme_generation(source="texts")  # Should not raise


@pytest.mark.asyncio
async def test_async_workflow_theme_allocation_dependencies():
    """Test theme allocation automatic dependency injection."""
    wf = AsyncWorkflow()

    # Theme allocation without explicit themes should inject theme_generation
    wf.theme_allocation()

    # Should have auto-injected theme_generation
    assert len(wf._processes) == 2
    assert wf._processes[0].id == "theme_generation"
    assert wf._processes[1].id == "theme_allocation"


@pytest.mark.asyncio
async def test_async_workflow_context_manager():
    """Test async context manager support."""
    wf = AsyncWorkflow()

    # Test context manager protocol
    async with wf as workflow:
        assert workflow is wf

    # Verify close was called (no exception should be raised)


@pytest.mark.asyncio
async def test_async_workflow_monitor_callbacks():
    """Test async monitor callbacks."""
    wf = AsyncWorkflow()

    # Create async callback mocks
    on_run_start = AsyncMock()
    on_process_start = AsyncMock()
    on_process_end = AsyncMock()
    on_run_end = AsyncMock()

    # Register callbacks
    wf.monitor(
        on_run_start=on_run_start,
        on_process_start=on_process_start,
        on_process_end=on_process_end,
        on_run_end=on_run_end,
    )

    # Verify callbacks are stored
    assert hasattr(wf, "_monitors")
    assert wf._monitors["on_run_start"] is on_run_start


@pytest.mark.asyncio
async def test_async_workflow_linear_mode(mock_async_client):
    """Test linear mode execution using AsyncAnalyzer."""
    wf = AsyncWorkflow()
    wf.sentiment()

    # Mock AsyncAnalyzer
    from unittest.mock import patch

    with patch("pulse.async_dsl.AsyncAnalyzer") as mock_analyzer_class:
        mock_analyzer = AsyncMock()
        mock_analyzer.run.return_value = MagicMock()
        mock_analyzer_class.return_value.__aenter__.return_value = mock_analyzer

        texts = ["text1", "text2"]
        await wf.run(texts, client=mock_async_client)

        # Verify AsyncAnalyzer was used
        mock_analyzer_class.assert_called_once()
        mock_analyzer.run.assert_called_once()


@pytest.mark.asyncio
async def test_async_workflow_dsl_mode_execution(mock_async_client):
    """Test DSL mode execution with sources."""
    wf = AsyncWorkflow()
    wf.source("texts", ["positive text", "negative text"])
    wf.sentiment(source="texts")

    # Execute workflow
    result = await wf.run(client=mock_async_client)

    # Verify client methods were called
    mock_async_client.analyze_sentiment.assert_called_once()

    # Verify result structure
    assert hasattr(result, "sentiment")


@pytest.mark.asyncio
async def test_async_workflow_theme_generation_process(mock_async_client):
    """Test theme generation process execution."""
    wf = AsyncWorkflow()
    wf.source("texts", ["text about food", "text about service"])
    wf.theme_generation(source="texts", min_themes=2, max_themes=5)

    await wf.run(client=mock_async_client)

    # Verify theme generation was called with correct parameters
    mock_async_client.generate_themes.assert_called_once()
    call_args = mock_async_client.generate_themes.call_args
    assert call_args[1]["min_themes"] == 2
    assert call_args[1]["max_themes"] == 5


@pytest.mark.asyncio
async def test_async_workflow_theme_allocation_process(mock_async_client):
    """Test theme allocation process execution."""
    wf = AsyncWorkflow()
    wf.source("texts", ["text1", "text2"])
    wf.source("themes", ["Theme1", "Theme2"])
    wf.theme_allocation(inputs="texts", themes_from="themes")

    await wf.run(client=mock_async_client)

    # Verify similarity comparison was called for allocation
    mock_async_client.compare_similarity.assert_called_once()


@pytest.mark.asyncio
async def test_async_workflow_similarity_process(mock_async_client):
    """Test similarity process execution."""
    wf = AsyncWorkflow()
    wf.source("texts", ["text1", "text2"])
    wf.similarity(source="texts", flatten=True)

    await wf.run(client=mock_async_client)

    # Verify similarity was called
    mock_async_client.compare_similarity.assert_called_once()
    call_args = mock_async_client.compare_similarity.call_args
    assert call_args[1]["flatten"] is True


@pytest.mark.asyncio
async def test_async_workflow_cluster_process(mock_async_client):
    """Test clustering process execution."""
    wf = AsyncWorkflow()
    wf.source("texts", ["text1", "text2"])
    wf.cluster(source="texts", k=3, algorithm="kmeans")

    await wf.run(client=mock_async_client)

    # Verify clustering was called
    mock_async_client.cluster_texts.assert_called_once()
    call_args = mock_async_client.cluster_texts.call_args
    assert call_args[1]["k"] == 3
    assert call_args[1]["algorithm"] == "kmeans"


@pytest.mark.asyncio
async def test_async_workflow_theme_extraction_process(mock_async_client):
    """Test theme extraction process execution."""
    wf = AsyncWorkflow()
    wf.source("texts", ["text1", "text2"])
    wf.theme_extraction(
        source="texts", dictionary=["word1", "word2"], type="named-entities"
    )

    await wf.run(client=mock_async_client)

    # Verify extraction was called
    mock_async_client.extract_elements.assert_called_once()
    call_args = mock_async_client.extract_elements.call_args
    assert call_args[1]["dictionary"] == ["word1", "word2"]
    assert call_args[1]["type"] == "named-entities"


@pytest.mark.asyncio
async def test_async_workflow_graph_generation():
    """Test workflow graph generation."""
    wf = AsyncWorkflow()
    wf.theme_generation()
    wf.theme_allocation()  # Should depend on theme_generation
    wf.sentiment()

    graph = wf.graph()

    # Verify graph structure
    assert isinstance(graph, dict)
    assert "theme_generation" in graph
    assert "theme_allocation" in graph
    assert "sentiment" in graph

    # Verify dependencies
    assert "theme_generation" in graph["theme_allocation"]


@pytest.mark.asyncio
async def test_async_workflow_from_file():
    """Test loading workflow from file."""
    import tempfile
    import json

    # Create temporary config file
    config = {
        "pipeline": [
            {"theme_generation": {"min_themes": 2, "max_themes": 5}},
            {"sentiment": {"fast": True}},
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        temp_path = f.name

    try:
        wf = AsyncWorkflow.from_file(temp_path)

        # Verify processes were loaded
        assert len(wf._processes) == 2
        assert wf._processes[0].id == "theme_generation"
        assert wf._processes[1].id == "sentiment"
        assert wf._processes[0].min_themes == 2
        assert wf._processes[0].max_themes == 5
        assert wf._processes[1].fast is True
    finally:
        import os

        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_async_workflow_error_handling():
    """Test error handling in async workflow execution."""
    wf = AsyncWorkflow()

    # Test missing source error - this should raise during workflow building,
    # not execution
    wf.source("texts", ["text1", "text2"])

    with pytest.raises(
        ValueError,
        match="Unknown inputs source for theme_allocation: 'nonexistent'",
    ):
        wf.theme_allocation(inputs="nonexistent")


@pytest.mark.asyncio
async def test_async_workflow_client_cleanup():
    """Test proper client cleanup."""
    wf = AsyncWorkflow()
    wf.source("texts", ["text1", "text2"])
    wf.sentiment(source="texts")

    # Mock client creation
    from unittest.mock import patch

    with patch("pulse.async_dsl.AsyncCoreClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.analyze_sentiment.return_value = SentimentResponse(sentiments=[])

        await wf.run()

        # Verify client was closed
        mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_async_workflow_fast_flag_propagation(mock_async_client):
    """Test fast flag propagation to processes."""
    wf = AsyncWorkflow()
    wf.source("texts", ["text1", "text2"])
    wf.sentiment(source="texts", fast=True)

    await wf.run(client=mock_async_client, fast=False)

    # Verify process-level fast flag takes precedence
    call_args = mock_async_client.analyze_sentiment.call_args
    assert call_args[1]["fast"] is True


@pytest.mark.asyncio
async def test_async_workflow_nested_sentiment_processing(mock_async_client):
    """Test nested sentiment processing with flatten/reconstruct."""
    wf = AsyncWorkflow()

    # Nested input structure
    nested_texts = [["positive text", "negative text"], ["neutral text"]]
    wf.source("nested", nested_texts)
    wf.sentiment(source="nested")

    # Mock flattened sentiment response
    mock_async_client.analyze_sentiment.return_value = SentimentResponse(
        sentiments=[
            {"sentiment": "positive", "confidence": 0.9},
            {"sentiment": "negative", "confidence": 0.8},
            {"sentiment": "neutral", "confidence": 0.7},
        ]
    )

    result = await wf.run(client=mock_async_client)

    # Verify sentiment was called with flattened texts
    mock_async_client.analyze_sentiment.assert_called_once()

    # Result should have nested structure preserved
    assert hasattr(result, "sentiment")
