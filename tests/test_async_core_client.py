"""Tests for AsyncCoreClient functionality."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
import httpx

from pulse.config import BASE_URL
from pulse.async_auth import AsyncClientCredentialsAuth
from pulse.core.async_client import AsyncCoreClient
from pulse.core.async_jobs import AsyncJob
from pulse.core.exceptions import PulseAPIError
from pulse.core.models import (
    EmbeddingsRequest,
    EmbeddingsResponse,
    SimilarityRequest,
    SimilarityResponse,
    ThemesResponse,
    SentimentResponse,
    ClusteringResponse,
    ExtractionsResponse,
    SummariesResponse,
    UsageEstimateResponse,
)


class TestAsyncCoreClientInitialization:
    """Test AsyncCoreClient initialization and configuration."""

    def test_default_initialization(self):
        """Test default AsyncCoreClient initialization."""
        client = AsyncCoreClient()

        assert client.base_url == BASE_URL
        assert client.timeout == 30.0
        assert isinstance(client.client, httpx.AsyncClient)
        assert str(client.client.base_url).rstrip("/") == BASE_URL.rstrip("/")

    def test_custom_initialization(self):
        """Test AsyncCoreClient with custom parameters."""
        custom_base_url = "https://custom.api.com"
        custom_timeout = 60.0

        client = AsyncCoreClient(base_url=custom_base_url, timeout=custom_timeout)

        assert client.base_url == custom_base_url
        assert client.timeout == custom_timeout
        assert client.client.base_url == custom_base_url

    def test_initialization_with_custom_client(self):
        """Test AsyncCoreClient with custom httpx client."""
        custom_client = httpx.AsyncClient(base_url="https://custom.com")

        client = AsyncCoreClient(client=custom_client)

        assert client.client is custom_client
        assert client.base_url == BASE_URL  # base_url is set from constructor parameter

    def test_initialization_with_auth(self):
        """Test AsyncCoreClient with authentication."""
        auth = AsyncClientCredentialsAuth(
            client_id="test_id", client_secret="test_secret"
        )

        client = AsyncCoreClient(auth=auth)

        assert client.client.auth is auth

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test AsyncCoreClient as async context manager."""
        async with AsyncCoreClient() as client:
            assert isinstance(client, AsyncCoreClient)
            assert isinstance(client.client, httpx.AsyncClient)

        # Client should be closed after context exit
        # We can't easily test this without implementation details

    def test_factory_method_with_client_credentials_async(self):
        """Test with_client_credentials_async factory method."""
        client = AsyncCoreClient.with_client_credentials_async(
            client_id="test_id", client_secret="test_secret", audience="test_audience"
        )

        assert isinstance(client.client.auth, AsyncClientCredentialsAuth)
        assert client.client.auth.client_id == "test_id"
        assert client.client.auth.client_secret == "test_secret"
        assert client.client.auth.audience == "test_audience"

    def test_factory_method_with_pkce_async(self):
        """Test with_pkce_async factory method."""
        from pulse.async_auth import AsyncAuthorizationCodePKCEAuth

        client = AsyncCoreClient.with_pkce_async(
            code="test_code",
            code_verifier="test_verifier",
            client_id="test_id",
            redirect_uri="http://localhost:8888/callback",
        )

        assert isinstance(client.client.auth, AsyncAuthorizationCodePKCEAuth)
        assert client.client.auth.code == "test_code"
        assert client.client.auth.code_verifier == "test_verifier"


class TestAsyncCoreClientEmbeddings:
    """Test AsyncCoreClient embeddings functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create mock async client."""
        return AsyncMock(spec=httpx.AsyncClient)

    @pytest.fixture
    def async_core_client(self, mock_client):
        """Create AsyncCoreClient with mock client."""
        return AsyncCoreClient(client=mock_client)

    @pytest.mark.asyncio
    async def test_create_embeddings_fast_response(
        self, async_core_client, mock_client
    ):
        """Test create_embeddings with fast=True returning direct response."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [
                {"text": "test", "vector": [0.1, 0.2, 0.3]},
                {"text": "text", "vector": [0.4, 0.5, 0.6]},
            ],
            "requestId": "req-123",
            "usage": {"records": [], "total": 100},
        }
        mock_client.request.return_value = mock_response

        request = EmbeddingsRequest(inputs=["test", "text"], fast=True)
        result = await async_core_client.create_embeddings(request)

        assert isinstance(result, EmbeddingsResponse)
        assert len(result.embeddings) == 2
        assert result.embeddings[0].text == "test"
        assert result.embeddings[0].vector == [0.1, 0.2, 0.3]

        # Verify request was made correctly
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0].upper() == "POST"
        assert "/embeddings" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_create_embeddings_job_response(self, async_core_client, mock_client):
        """Test create_embeddings with fast=False returning job."""
        # Mock job response
        mock_response = Mock()
        mock_response.status_code = 202
        mock_response.json.return_value = {"jobId": "job-123", "jobStatus": "pending"}
        mock_client.request.return_value = mock_response

        request = EmbeddingsRequest(inputs=["test"], fast=False)
        result = await async_core_client.create_embeddings(
            request, await_job_result=False
        )

        assert isinstance(result, AsyncJob)
        assert result.id == "job-123"
        assert result.status == "pending"

    @pytest.mark.asyncio
    async def test_create_embeddings_await_job_result(
        self, async_core_client, mock_client
    ):
        """Test create_embeddings with await_job_result=True."""
        # Mock job submission response
        job_response = Mock()
        job_response.status_code = 202
        job_response.json.return_value = {"jobId": "job-123", "jobStatus": "pending"}

        # Mock job completion response
        completion_response = Mock()
        completion_response.status_code = 200
        completion_response.json.return_value = {
            "jobId": "job-123",
            "jobStatus": "completed",
            "resultUrl": "https://api.example.com/results/123",
        }

        # Mock result response
        result_response = Mock()
        result_response.status_code = 200
        result_response.json.return_value = {
            "embeddings": [{"text": "test", "vector": [0.1, 0.2]}],
            "requestId": "req-123",
        }

        mock_client.request.side_effect = [job_response]
        mock_client.get.side_effect = [completion_response, result_response]

        request = EmbeddingsRequest(inputs=["test"], fast=False)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await async_core_client.create_embeddings(request)

        assert isinstance(result, EmbeddingsResponse)
        assert len(result.embeddings) == 1
        assert result.embeddings[0].text == "test"


class TestAsyncCoreClientSimilarity:
    """Test AsyncCoreClient similarity functionality."""

    @pytest.fixture
    def mock_client(self):
        return AsyncMock(spec=httpx.AsyncClient)

    @pytest.fixture
    def async_core_client(self, mock_client):
        return AsyncCoreClient(client=mock_client)

    @pytest.mark.asyncio
    async def test_compare_similarity_self(self, async_core_client, mock_client):
        """Test compare_similarity with self-similarity."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "scenario": "self",
            "mode": "matrix",
            "n": 2,
            "matrix": [[1.0, 0.8], [0.8, 1.0]],
            "flattened": [1.0, 0.8, 0.8, 1.0],
            "requestId": "req-123",
        }
        mock_client.request.return_value = mock_response

        request = SimilarityRequest(set=["text1", "text2"], fast=True)
        result = await async_core_client.compare_similarity(request)

        assert isinstance(result, SimilarityResponse)
        assert result.scenario == "self"
        assert result.n == 2
        assert len(result.matrix) == 2

    @pytest.mark.asyncio
    async def test_compare_similarity_cross(self, async_core_client, mock_client):
        """Test compare_similarity with cross-similarity."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "scenario": "cross",
            "mode": "matrix",
            "n": 4,
            "matrix": [[0.9, 0.1], [0.8, 0.2]],
            "flattened": [0.9, 0.1, 0.8, 0.2],
            "requestId": "req-123",
        }
        mock_client.request.return_value = mock_response

        request = SimilarityRequest(
            set_a=["text1", "text2"], set_b=["text3", "text4"], fast=True
        )
        result = await async_core_client.compare_similarity(request)

        assert isinstance(result, SimilarityResponse)
        assert result.scenario == "cross"
        assert result.n == 4


class TestAsyncCoreClientThemes:
    """Test AsyncCoreClient themes functionality."""

    @pytest.fixture
    def mock_client(self):
        return AsyncMock(spec=httpx.AsyncClient)

    @pytest.fixture
    def async_core_client(self, mock_client):
        return AsyncCoreClient(client=mock_client)

    @pytest.mark.asyncio
    async def test_generate_themes(self, async_core_client, mock_client):
        """Test generate_themes functionality."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "themes": [
                {
                    "shortLabel": "positive",
                    "label": "Positive Sentiment",
                    "description": "Positive feedback",
                    "representatives": ["great", "excellent"],
                },
                {
                    "shortLabel": "negative",
                    "label": "Negative Sentiment",
                    "description": "Negative feedback",
                    "representatives": ["bad", "poor"],
                },
            ],
            "requestId": "req-123",
            "usage": {"records": [], "total": 50},
        }
        mock_client.request.return_value = mock_response

        result = await async_core_client.generate_themes(
            texts=["great product", "bad service"],
            min_themes=2,
            max_themes=5,
            fast=True,
        )

        assert isinstance(result, ThemesResponse)
        assert len(result.themes) == 2
        assert result.themes[0].shortLabel == "positive"
        assert result.themes[1].shortLabel == "negative"


class TestAsyncCoreClientSentiment:
    """Test AsyncCoreClient sentiment functionality."""

    @pytest.fixture
    def mock_client(self):
        return AsyncMock(spec=httpx.AsyncClient)

    @pytest.fixture
    def async_core_client(self, mock_client):
        return AsyncCoreClient(client=mock_client)

    @pytest.mark.asyncio
    async def test_analyze_sentiment(self, async_core_client, mock_client):
        """Test analyze_sentiment functionality."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"sentiment": "positive", "confidence": 0.9},
                {"sentiment": "negative", "confidence": 0.8},
            ],
            "requestId": "req-123",
            "usage": {"records": [], "total": 25},
        }
        mock_client.request.return_value = mock_response

        result = await async_core_client.analyze_sentiment(
            texts=["I love this!", "I hate this!"], fast=True
        )

        assert isinstance(result, SentimentResponse)
        assert len(result.results) == 2
        assert result.results[0].sentiment == "positive"
        assert result.results[0].confidence == 0.9
        assert result.results[1].sentiment == "negative"
        assert result.results[1].confidence == 0.8


class TestAsyncCoreClientClustering:
    """Test AsyncCoreClient clustering functionality."""

    @pytest.fixture
    def mock_client(self):
        return AsyncMock(spec=httpx.AsyncClient)

    @pytest.fixture
    def async_core_client(self, mock_client):
        return AsyncCoreClient(client=mock_client)

    @pytest.mark.asyncio
    async def test_cluster_texts(self, async_core_client, mock_client):
        """Test cluster_texts functionality."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "algorithm": "kmeans",
            "clusters": [
                {"clusterId": 0, "items": ["text1", "text2"]},
                {"clusterId": 1, "items": ["text3", "text4"]},
            ],
            "requestId": "req-123",
            "usage": {"records": [], "total": 30},
        }
        mock_client.request.return_value = mock_response

        result = await async_core_client.cluster_texts(
            inputs=["text1", "text2", "text3", "text4"],
            k=2,
            algorithm="kmeans",
            fast=True,
        )

        assert isinstance(result, ClusteringResponse)
        assert result.algorithm == "kmeans"
        assert len(result.clusters) == 2
        assert result.clusters[0].clusterId == 0
        assert len(result.clusters[0].items) == 2


class TestAsyncCoreClientExtractions:
    """Test AsyncCoreClient extractions functionality."""

    @pytest.fixture
    def mock_client(self):
        return AsyncMock(spec=httpx.AsyncClient)

    @pytest.fixture
    def async_core_client(self, mock_client):
        return AsyncCoreClient(client=mock_client)

    @pytest.mark.asyncio
    async def test_extract_elements(self, async_core_client, mock_client):
        """Test extract_elements functionality."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "columns": [
                {"category": "person", "term": "John"},
                {"category": "location", "term": "NYC"},
            ],
            "matrix": [["John"], ["NYC"]],
            "requestId": "req-123",
            "usage": {"records": [], "total": 20},
        }
        mock_client.request.return_value = mock_response

        result = await async_core_client.extract_elements(
            inputs=["John lives in NYC", "Mary works in LA"],
            dictionary=["John", "Mary", "NYC", "LA"],
            type="named-entities",
            fast=True,
        )

        assert isinstance(result, ExtractionsResponse)
        assert len(result.columns) == 2
        assert result.columns[0].category == "person"
        assert result.columns[0].term == "John"


class TestAsyncCoreClientSummaries:
    """Test AsyncCoreClient summaries functionality."""

    @pytest.fixture
    def mock_client(self):
        return AsyncMock(spec=httpx.AsyncClient)

    @pytest.fixture
    def async_core_client(self, mock_client):
        return AsyncCoreClient(client=mock_client)

    @pytest.mark.asyncio
    async def test_generate_summary(self, async_core_client, mock_client):
        """Test generate_summary functionality."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "summary": "This is a test summary of the provided texts.",
            "requestId": "req-123",
            "usage": {"records": [], "total": 15},
        }
        mock_client.request.return_value = mock_response

        result = await async_core_client.generate_summary(
            inputs=["Long text to summarize", "Another long text"],
            question="What is the main point?",
            length="short",
            fast=True,
        )

        assert isinstance(result, SummariesResponse)
        assert result.summary == "This is a test summary of the provided texts."


class TestAsyncCoreClientUsageEstimate:
    """Test AsyncCoreClient usage estimation functionality."""

    @pytest.fixture
    def mock_client(self):
        return AsyncMock(spec=httpx.AsyncClient)

    @pytest.fixture
    def async_core_client(self, mock_client):
        return AsyncCoreClient(client=mock_client)

    @pytest.mark.asyncio
    async def test_estimate_usage(self, async_core_client, mock_client):
        """Test estimate_usage functionality."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "usage": {"records": [{"feature": "sentiment", "quantity": 2}], "total": 2}
        }
        mock_client.request.return_value = mock_response

        result = await async_core_client.estimate_usage(
            feature="sentiment", inputs=["text1", "text2"]
        )

        assert isinstance(result, UsageEstimateResponse)
        assert result.usage["total"] == 2
        assert result.usage["records"][0]["feature"] == "sentiment"


class TestAsyncCoreClientErrorHandling:
    """Test AsyncCoreClient error handling."""

    @pytest.fixture
    def mock_client(self):
        return AsyncMock(spec=httpx.AsyncClient)

    @pytest.fixture
    def async_core_client(self, mock_client):
        return AsyncCoreClient(client=mock_client)

    @pytest.mark.asyncio
    async def test_api_error_handling(self, async_core_client, mock_client):
        """Test API error handling."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "error": "Bad Request",
            "message": "Invalid input data",
        }
        mock_client.request.return_value = mock_response

        request = EmbeddingsRequest(inputs=["test"], fast=True)

        with pytest.raises(PulseAPIError) as exc_info:
            await async_core_client.create_embeddings(request)

        error = exc_info.value
        assert error.status == 400
        assert "Invalid input data" in error.message

    @pytest.mark.asyncio
    async def test_network_error_handling(self, async_core_client, mock_client):
        """Test network error handling."""
        mock_client.request.side_effect = httpx.ConnectError("Connection failed")

        request = EmbeddingsRequest(inputs=["test"], fast=True)

        with pytest.raises(httpx.ConnectError):
            await async_core_client.create_embeddings(request)

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, async_core_client, mock_client):
        """Test timeout error handling."""
        mock_client.request.side_effect = httpx.TimeoutException("Request timed out")

        request = EmbeddingsRequest(inputs=["test"], fast=True)

        with pytest.raises(httpx.TimeoutException):
            await async_core_client.create_embeddings(request)


class TestAsyncCoreClientBatching:
    """Test AsyncCoreClient batching functionality."""

    @pytest.fixture
    def mock_client(self):
        return AsyncMock(spec=httpx.AsyncClient)

    @pytest.fixture
    def async_core_client(self, mock_client):
        return AsyncCoreClient(client=mock_client)

    @pytest.mark.asyncio
    async def test_batch_processing(self, async_core_client, mock_client):
        """Test batch processing with concurrent job handling."""
        # Mock job submission responses
        job_responses = []
        for i in range(3):
            response = Mock()
            response.status_code = 202
            response.json.return_value = {"jobId": f"job-{i}", "jobStatus": "pending"}
            job_responses.append(response)

        mock_client.request.side_effect = job_responses

        # Create large input that will trigger batching
        large_input = ["text"] * 1000  # Assuming batch size is smaller
        request = EmbeddingsRequest(inputs=large_input, fast=False)

        # Mock the _process_batches_concurrently method to return mock results
        mock_results = [
            {"embeddings": [{"text": "text", "vector": [0.1, 0.2]}] * 334},
            {"embeddings": [{"text": "text", "vector": [0.1, 0.2]}] * 333},
            {"embeddings": [{"text": "text", "vector": [0.1, 0.2]}] * 333},
        ]

        with patch.object(
            async_core_client,
            "_process_batches_concurrently",
            return_value=mock_results,
        ):
            result = await async_core_client.create_embeddings(request)

        assert isinstance(result, EmbeddingsResponse)
        assert len(result.embeddings) == 1000

    @pytest.mark.asyncio
    async def test_concurrent_job_processing(self, async_core_client):
        """Test concurrent job processing utility method."""

        # Create mock job submitters
        async def mock_submitter(job_id):
            job = AsyncMock(spec=AsyncJob)
            job.id = f"job-{job_id}"
            job.wait = AsyncMock(return_value={"result": f"result-{job_id}"})
            return job

        submitters = [lambda i=i: mock_submitter(i) for i in range(3)]

        results = await async_core_client._process_batches_concurrently(
            submitters, max_workers=2, timeout=30.0
        )

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["result"] == f"result-{i}"


class TestAsyncCoreClientResourceManagement:
    """Test AsyncCoreClient resource management."""

    @pytest.mark.asyncio
    async def test_client_cleanup(self):
        """Test proper client cleanup."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)

        async_core_client = AsyncCoreClient(client=mock_client)
        await async_core_client.close()

        mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self):
        """Test cleanup via context manager."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)

        async with AsyncCoreClient(client=mock_client) as client:
            assert isinstance(client, AsyncCoreClient)

        # Client should have context manager methods called
        mock_client.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_close_calls(self):
        """Test that multiple close calls don't cause issues."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)

        async_core_client = AsyncCoreClient(client=mock_client)
        await async_core_client.close()
        await async_core_client.close()  # Should not raise

        # Close should be called twice
        assert mock_client.aclose.call_count == 2
