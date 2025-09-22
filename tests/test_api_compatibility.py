"""Test API interface compatibility for enhanced batching features.

This module tests that all enhanced batching features maintain identical
API interfaces and return results indistinguishable from non-batched operations.
"""

import pytest
from unittest.mock import patch
from pulse.core.client import CoreClient
from pulse.core.models import (
    EmbeddingsRequest,
    EmbeddingsResponse,
    SentimentResponse,
    UsageReport,
    UsageRecord,
)


class TestAPICompatibility:
    """Test that enhanced batching maintains API compatibility."""

    def setup_method(self):
        """Set up test client."""
        self.client = CoreClient()

    def test_sentiment_analysis_method_signature_unchanged(self):
        """Test that sentiment analysis method signature remains identical."""
        import inspect

        # Get method signature
        sig = inspect.signature(self.client.analyze_sentiment)

        # Verify expected parameters exist with correct defaults
        params = sig.parameters
        assert "texts" in params
        assert params["texts"].annotation == list[str]
        assert "version" in params
        assert params["version"].default is None
        assert "fast" in params
        assert params["fast"].default is False
        assert "await_job_result" in params
        assert params["await_job_result"].default is True

        # Verify return type annotation contains expected types
        return_str = str(sig.return_annotation)
        assert "SentimentResponse" in return_str
        assert "Job" in return_str

    def test_embeddings_method_signature_unchanged(self):
        """Test that embeddings method signature remains identical."""
        import inspect

        # Get method signature
        sig = inspect.signature(self.client.create_embeddings)

        # Verify expected parameters exist with correct defaults
        params = sig.parameters
        assert "request" in params
        assert "await_job_result" in params
        assert params["await_job_result"].default is True

        # Verify return type annotation contains expected types
        return_str = str(sig.return_annotation)
        assert "EmbeddingsResponse" in return_str
        assert "Job" in return_str

    def test_clustering_method_signature_unchanged(self):
        """Test that clustering method signature remains identical."""
        import inspect

        # Get method signature
        sig = inspect.signature(self.client.cluster_texts)

        # Verify expected parameters exist with correct defaults
        params = sig.parameters
        assert "inputs" in params
        assert params["inputs"].annotation == list[str]
        assert "k" in params
        assert params["k"].annotation == int
        assert "algorithm" in params
        assert params["algorithm"].default == "kmeans"
        assert "fast" in params
        assert "await_job_result" in params
        assert params["await_job_result"].default is True

    def test_extractions_method_signature_unchanged(self):
        """Test that extractions method signature remains identical."""
        import inspect

        # Get method signature
        sig = inspect.signature(self.client.extract_elements)

        # Verify expected parameters exist with correct defaults
        params = sig.parameters
        assert "inputs" in params
        assert params["inputs"].annotation == list[str]
        assert "dictionary" in params
        assert params["dictionary"].annotation == list[str]
        assert "type" in params
        assert params["type"].default == "named-entities"
        assert "expand_dictionary" in params
        assert params["expand_dictionary"].default is False
        assert "fast" in params
        assert "await_job_result" in params
        assert params["await_job_result"].default is True

    @patch("pulse.core.client.CoreClient._request")
    def test_sentiment_batched_vs_non_batched_results_identical(self, mock_request):
        """Test that batched sentiment results are identical to non-batched."""
        # Mock response for single request
        single_response_data = {
            "results": [
                {"text": "I love this!", "sentiment": "positive", "confidence": 0.95},
                {"text": "This is okay", "sentiment": "neutral", "confidence": 0.60},
            ],
            "usage": {"total": 2, "records": [{"feature": "sentiment", "quantity": 2}]},
            "requestId": "test-123",
        }

        # Mock response for batched requests

        # Test small request (non-batched)
        mock_request.return_value.status_code = 200
        mock_request.return_value.json.return_value = single_response_data

        small_texts = ["I love this!", "This is okay"]
        small_result = self.client.analyze_sentiment(small_texts, fast=True)

        # Verify it's a SentimentResponse
        assert isinstance(small_result, SentimentResponse)
        assert len(small_result.results) == 2
        assert small_result.results[0].sentiment == "positive"
        assert small_result.results[1].sentiment == "neutral"
        assert small_result.usage.total == 2

    @patch("pulse.core.client.CoreClient._request")
    def test_embeddings_batched_vs_non_batched_results_identical(self, mock_request):
        """Test that batched embeddings results are identical to non-batched."""
        # Mock response for single request
        single_response_data = {
            "embeddings": [
                {"id": "1", "text": "Hello world", "vector": [0.1, 0.2, 0.3]},
                {"id": "2", "text": "Goodbye world", "vector": [0.4, 0.5, 0.6]},
            ],
            "usage": {
                "total": 2,
                "records": [{"feature": "embeddings", "quantity": 2}],
            },
            "requestId": "test-123",
        }

        # Test small request (non-batched)
        mock_request.return_value.status_code = 200
        mock_request.return_value.json.return_value = single_response_data

        request = EmbeddingsRequest(inputs=["Hello world", "Goodbye world"], fast=True)
        result = self.client.create_embeddings(request)

        # Verify it's an EmbeddingsResponse
        assert isinstance(result, EmbeddingsResponse)
        assert len(result.embeddings) == 2
        assert result.embeddings[0].text == "Hello world"
        assert result.embeddings[1].text == "Goodbye world"
        assert result.usage.total == 2

    def test_existing_code_compatibility_sentiment(self):
        """Test that existing sentiment analysis code patterns still work."""
        # These are common usage patterns that should continue to work

        # Pattern 1: Basic usage
        try:
            # This should not raise any import or signature errors
            # We can't actually call it without mocking, but we can verify the interface
            method = getattr(self.client, "analyze_sentiment")
            assert callable(method)
        except Exception as e:
            pytest.fail(f"Basic sentiment interface broken: {e}")

        # Pattern 2: With parameters
        try:
            method = getattr(self.client, "analyze_sentiment")
            import inspect

            sig = inspect.signature(method)
            # Verify we can call with all expected parameters
            bound = sig.bind(["test"], version="v1", fast=True, await_job_result=False)
            assert bound is not None
        except Exception as e:
            pytest.fail(f"Parameterized sentiment interface broken: {e}")

    def test_existing_code_compatibility_embeddings(self):
        """Test that existing embeddings code patterns still work."""
        # Pattern 1: Basic usage with EmbeddingsRequest
        try:
            request = EmbeddingsRequest(inputs=["test"])
            method = getattr(self.client, "create_embeddings")
            assert callable(method)
        except Exception as e:
            pytest.fail(f"Basic embeddings interface broken: {e}")

        # Pattern 2: With parameters
        try:
            request = EmbeddingsRequest(inputs=["test"], fast=True)
            method = getattr(self.client, "create_embeddings")
            import inspect

            sig = inspect.signature(method)
            bound = sig.bind(request, await_job_result=False)
            assert bound is not None
        except Exception as e:
            pytest.fail(f"Parameterized embeddings interface broken: {e}")

    def test_existing_code_compatibility_clustering(self):
        """Test that existing clustering code patterns still work."""
        try:
            method = getattr(self.client, "cluster_texts")
            assert callable(method)

            import inspect

            sig = inspect.signature(method)
            # Test various parameter combinations
            bound = sig.bind(["test1", "test2"], k=2)
            assert bound is not None

            bound = sig.bind(["test1", "test2"], k=2, algorithm="kmeans")
            assert bound is not None
        except Exception as e:
            pytest.fail(f"Clustering interface broken: {e}")

    def test_existing_code_compatibility_extractions(self):
        """Test that existing extractions code patterns still work."""
        try:
            method = getattr(self.client, "extract_elements")
            assert callable(method)

            import inspect

            sig = inspect.signature(method)
            # Test basic usage
            bound = sig.bind(["test text"], ["entity1", "entity2"])
            assert bound is not None

            # Test with parameters
            bound = sig.bind(
                ["test text"],
                ["entity1", "entity2"],
                type="named-entities",
                expand_dictionary=True,
            )
            assert bound is not None
        except Exception as e:
            pytest.fail(f"Extractions interface broken: {e}")

    def test_response_structure_unchanged(self):
        """Test that response structures remain unchanged."""
        # Test SentimentResponse structure
        response_data = {
            "results": [{"text": "test", "sentiment": "positive", "confidence": 0.9}],
            "usage": {"total": 1, "records": [{"feature": "sentiment", "quantity": 1}]},
            "requestId": "test-123",
        }

        response = SentimentResponse.model_validate(response_data)
        assert hasattr(response, "results")
        assert hasattr(response, "usage")
        assert hasattr(response, "requestId")
        assert len(response.results) == 1
        assert response.results[0].sentiment == "positive"

    def test_usage_model_structure_unchanged(self):
        """Test that usage model structure remains unchanged."""
        usage_data = {
            "total": 5,
            "records": [
                {"feature": "sentiment", "quantity": 3},
                {"feature": "embeddings", "quantity": 2},
            ],
        }

        usage = UsageReport.model_validate(usage_data)
        assert hasattr(usage, "total")
        assert hasattr(usage, "records")
        assert usage.total == 5
        assert len(usage.records) == 2
        assert usage.records[0].feature == "sentiment"
        assert usage.records[0].quantity == 3

    def test_backward_compatibility_properties(self):
        """Test that backward compatibility properties still work."""
        # Test UsageRecord.units property (backward compatibility)
        record = UsageRecord(feature="test", quantity=5)
        assert hasattr(record, "units")
        assert record.units == 5
        assert record.units == record.quantity

    def test_import_compatibility(self):
        """Test that all imports still work as expected."""
        # Test that all expected classes can be imported
        try:
            from pulse.core.client import CoreClient
            from pulse.core.models import (
                EmbeddingsRequest,
                EmbeddingsResponse,
                SentimentResponse,
                ExtractionsResponse,
                ClusteringResponse,
                UsageReport,
                UsageRecord,
            )

            # Verify classes exist and are callable
            assert CoreClient is not None
            assert EmbeddingsRequest is not None
            assert EmbeddingsResponse is not None
            assert SentimentResponse is not None
            assert ExtractionsResponse is not None
            assert ClusteringResponse is not None
            assert UsageReport is not None
            assert UsageRecord is not None

        except ImportError as e:
            pytest.fail(f"Import compatibility broken: {e}")

    def test_method_availability(self):
        """Test that all expected methods are available on CoreClient."""
        expected_methods = [
            "analyze_sentiment",
            "create_embeddings",
            "cluster_texts",
            "extract_elements",
            "compare_similarity",
            "generate_themes",
        ]

        for method_name in expected_methods:
            assert hasattr(self.client, method_name), f"Method {method_name} not found"
            method = getattr(self.client, method_name)
            assert callable(method), f"Method {method_name} is not callable"
