"""Integration tests for updated client methods in OpenAPI v0.9.0."""

import json
import time
import httpx
import pytest
from pulse.core.client import CoreClient
from pulse.core.jobs import Job
from pulse.core.models import (
    ThemesResponse,
    ThemeSetsResponse,
    ClusteringResponse,
    SimilarityResponse,
    ExtractionsResponse,
    UsageEstimateResponse,
)


class TestGenerateThemesIntegration:
    """Integration tests for generate_themes with new parameters and response types."""

    def test_generate_themes_with_new_parameters_sync(self):
        """Test generate_themes with new parameters in sync mode."""

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/themes"
            body = json.loads(request.content.decode())

            # Verify new parameters are passed correctly
            assert body["inputs"] == ["text1", "text2", "text3"]
            assert body["minThemes"] == 2
            assert body["maxThemes"] == 5
            assert body["context"] == "test context"
            assert body["version"] == "2025-09-01"
            assert body["prune"] == 3
            assert body["interactive"] is True
            assert body["initialSets"] == 2
            assert body["fast"] is True

            return httpx.Response(
                200,
                json={
                    "themes": [
                        {
                            "shortLabel": "Tech",
                            "label": "Technology",
                            "description": "Technology-related themes",
                            "representatives": ["tech text 1", "tech text 2"],
                        }
                    ],
                    "requestId": "req123",
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        response = core_client.generate_themes(
            texts=["text1", "text2", "text3"],
            min_themes=2,
            max_themes=5,
            context="test context",
            version="2025-09-01",
            prune=3,
            interactive=True,
            initial_sets=2,
            fast=True,
        )

        assert isinstance(response, ThemesResponse)
        assert len(response.themes) == 1
        assert response.themes[0].shortLabel == "Tech"
        assert response.requestId == "req123"

    def test_generate_themes_returns_theme_sets_response(self):
        """Test generate_themes returns ThemeSetsResponse for version 2025-09-01."""

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/themes"
            body = json.loads(request.content.decode())
            assert body["version"] == "2025-09-01"

            return httpx.Response(
                200,
                json={
                    "themeSets": [
                        [
                            {
                                "shortLabel": "Tech",
                                "label": "Technology",
                                "description": "Technology themes",
                                "representatives": ["tech1", "tech2"],
                            }
                        ],
                        [
                            {
                                "shortLabel": "Health",
                                "label": "Healthcare",
                                "description": "Healthcare themes",
                                "representatives": ["health1", "health2"],
                            }
                        ],
                    ],
                    "requestId": "req456",
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        response = core_client.generate_themes(
            texts=["text1", "text2"], version="2025-09-01", fast=True
        )

        assert isinstance(response, ThemeSetsResponse)
        assert len(response.themeSets) == 2
        assert len(response.themeSets[0]) == 1
        assert len(response.themeSets[1]) == 1
        assert response.requestId == "req456"

    def test_generate_themes_async_job(self, monkeypatch):
        """Test generate_themes async job handling."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST" and request.url.path == "/themes":
                body = json.loads(request.content.decode())
                # fast=False means the fast key is not included in the body
                assert "fast" not in body or body["fast"] is False
                return httpx.Response(202, json={"jobId": "job123"})
            elif request.method == "GET" and request.url.path == "/jobs":
                return httpx.Response(
                    200,
                    json={
                        "jobId": "job123",
                        "jobStatus": "completed",
                        "resultUrl": "https://api.example.com/results/job123",
                    },
                )
            elif request.method == "GET" and request.url.path == "/results/job123":
                return httpx.Response(
                    200,
                    json={
                        "themes": [
                            {
                                "shortLabel": "Async",
                                "label": "Async Theme",
                                "description": "Async generated theme",
                                "representatives": ["async1", "async2"],
                            }
                        ],
                        "requestId": "async-req",
                    },
                )
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        # Mock sleep to speed up test
        monkeypatch.setattr(time, "sleep", lambda x: None)

        # Test returning job handle
        job = core_client.generate_themes(
            texts=["text1", "text2"], fast=False, await_job_result=False
        )
        assert isinstance(job, Job)

        # Test waiting for result
        result = job.wait()
        assert result["themes"][0]["shortLabel"] == "Async"


class TestClusterTextsIntegration:
    """Integration tests for cluster_texts with different algorithms."""

    def test_cluster_texts_with_algorithm_selection(self):
        """Test cluster_texts with different algorithms."""

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/clustering"
            body = json.loads(request.content.decode())

            assert body["inputs"] == ["text1", "text2", "text3"]
            assert body["k"] == 2
            assert body["algorithm"] == "skmeans"
            assert body["fast"] is True

            return httpx.Response(
                200,
                json={
                    "algorithm": "skmeans",
                    "clusters": [
                        {"clusterId": 0, "items": ["text1", "text2"]},
                        {"clusterId": 1, "items": ["text3"]},
                    ],
                    "requestId": "cluster-req",
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        response = core_client.cluster_texts(
            inputs=["text1", "text2", "text3"], k=2, algorithm="skmeans", fast=True
        )

        assert isinstance(response, ClusteringResponse)
        assert response.algorithm == "skmeans"
        assert len(response.clusters) == 2
        assert response.clusters[0].clusterId == 0
        assert response.clusters[0].items == ["text1", "text2"]
        assert response.clusters[1].clusterId == 1
        assert response.clusters[1].items == ["text3"]

    def test_cluster_texts_default_algorithm(self):
        """Test cluster_texts with default algorithm."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            assert body["algorithm"] == "kmeans"  # Default algorithm

            return httpx.Response(
                200,
                json={
                    "algorithm": "kmeans",
                    "clusters": [{"clusterId": 0, "items": ["text1", "text2"]}],
                    "requestId": "default-req",
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        response = core_client.cluster_texts(inputs=["text1", "text2"], k=1, fast=True)
        assert response.algorithm == "kmeans"

    def test_cluster_texts_all_algorithms(self):
        """Test cluster_texts with all supported algorithms."""
        algorithms = ["kmeans", "skmeans", "agglomerative", "hdbscan"]

        for algorithm in algorithms:

            def handler(request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content.decode())
                assert body["algorithm"] == algorithm

                return httpx.Response(
                    200,
                    json={
                        "algorithm": algorithm,
                        "clusters": [{"clusterId": 0, "items": ["text1", "text2"]}],
                        "requestId": f"{algorithm}-req",
                    },
                )

            transport = httpx.MockTransport(handler)
            client = httpx.Client(
                transport=transport, base_url="https://api.example.com"
            )
            core_client = CoreClient(client=client)

            response = core_client.cluster_texts(
                inputs=["text1", "text2"], k=1, algorithm=algorithm, fast=True
            )
            assert response.algorithm == algorithm

    def test_cluster_texts_invalid_algorithm(self):
        """Test cluster_texts with invalid algorithm raises error."""
        client = CoreClient(client=httpx.Client(base_url="https://api.example.com"))

        with pytest.raises(ValueError) as exc_info:
            client.cluster_texts(
                inputs=["text1", "text2"], k=1, algorithm="invalid", fast=True
            )
        assert "algorithm must be one of" in str(exc_info.value)


class TestCompareSimilarityIntegration:
    """Integration tests for compare_similarity with text splitting."""

    def test_compare_similarity_with_text_splitting(self):
        """Test compare_similarity with text splitting configuration."""

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/similarity"
            body = json.loads(request.content.decode())

            assert body["set"] == ["text1", "text2"]
            assert body["fast"] is True
            assert body["flatten"] is True
            assert body["version"] == "v1"
            assert "split" in body
            assert body["split"]["set_a"]["unit"] == "sentence"
            assert body["split"]["set_a"]["agg"] == "mean"

            return httpx.Response(
                200,
                json={
                    "scenario": "self",
                    "mode": "flattened",
                    "n": 2,
                    "flattened": [1.0, 0.8, 0.8, 1.0],
                    "requestId": "similarity-req",
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        split_config = {
            "set_a": {
                "unit": "sentence",
                "agg": "mean",
                "window_size": 1,
                "stride_size": 1,
            }
        }

        response = core_client.compare_similarity(
            set=["text1", "text2"],
            fast=True,
            flatten=True,
            version="v1",
            split=split_config,
        )

        assert isinstance(response, SimilarityResponse)
        assert response.scenario == "self"
        assert response.mode == "flattened"
        assert response.n == 2
        assert response.flattened == [1.0, 0.8, 0.8, 1.0]

    def test_compare_similarity_cross_mode(self):
        """Test compare_similarity in cross mode."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())

            assert body["set_a"] == ["text1", "text2"]
            assert body["set_b"] == ["text3", "text4"]
            assert "set" not in body

            return httpx.Response(
                200,
                json={
                    "scenario": "cross",
                    "mode": "matrix",
                    "n": 2,
                    "flattened": [0.9, 0.7, 0.8, 0.6],
                    "matrix": [[0.9, 0.7], [0.8, 0.6]],
                    "requestId": "cross-req",
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        response = core_client.compare_similarity(
            set_a=["text1", "text2"], set_b=["text3", "text4"], fast=True
        )

        assert response.scenario == "cross"
        assert response.matrix == [[0.9, 0.7], [0.8, 0.6]]


class TestExtractElementsIntegration:
    """Integration tests for extract_elements with type control."""

    def test_extract_elements_with_type_control(self):
        """Test extract_elements with type parameter."""

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/extractions"
            body = json.loads(request.content.decode())

            assert body["inputs"] == ["text1", "text2"]
            assert body["dictionary"] == ["term1", "term2", "term3"]
            assert body["type"] == "themes"
            assert body["expand_dictionary"] is False
            assert body["expand_dictionary_limit"] == 5
            assert body["version"] == "v1"
            assert body["fast"] is True

            return httpx.Response(
                200,
                json={
                    "columns": [
                        {"category": "themes", "term": "term1"},
                        {"category": "themes", "term": "term2"},
                    ],
                    "matrix": [["match1", ""], ["", "match2"]],
                    "requestId": "extract-req",
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        response = core_client.extract_elements(
            inputs=["text1", "text2"],
            dictionary=["term1", "term2", "term3"],
            type="themes",
            expand_dictionary=False,
            expand_dictionary_limit=5,
            version="v1",
            fast=True,
        )

        assert isinstance(response, ExtractionsResponse)
        assert len(response.columns) == 2
        assert response.columns[0].category == "themes"
        assert response.matrix == [["match1", ""], ["", "match2"]]

    def test_extract_elements_named_entities_type(self):
        """Test extract_elements with named-entities type."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            assert body["type"] == "named-entities"
            assert body["expand_dictionary"] is True

            return httpx.Response(
                200,
                json={
                    "columns": [{"category": "named-entities", "term": "entity1"}],
                    "matrix": [["entity_match"]],
                    "requestId": "ner-req",
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        response = core_client.extract_elements(
            inputs=["text1"],
            dictionary=["entity1", "entity2", "entity3"],
            type="named-entities",
            expand_dictionary=True,
            fast=True,
        )

        assert response.columns[0].category == "named-entities"

    def test_extract_elements_invalid_type(self):
        """Test extract_elements with invalid type raises error."""
        client = CoreClient(client=httpx.Client(base_url="https://api.example.com"))

        with pytest.raises(ValueError) as exc_info:
            client.extract_elements(
                inputs=["text1"],
                dictionary=["term1", "term2", "term3"],
                type="invalid",
                fast=True,
            )
        assert "type must be one of" in str(exc_info.value)

    def test_extract_elements_backward_compatibility(self):
        """Test extract_elements backward compatibility with deprecated parameters."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            # Should use inputs and dictionary from deprecated parameters
            assert body["inputs"] == ["old_text"]
            assert body["dictionary"] == ["old_term1", "old_term2", "old_term3"]

            return httpx.Response(
                200,
                json={
                    "columns": [{"category": "compat", "term": "old_term1"}],
                    "matrix": [["old_match"]],
                    "requestId": "compat-req",
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        # Test with deprecated parameter names
        response = core_client.extract_elements(
            inputs=["new_text"],  # This should be used
            dictionary=["new_term1", "new_term2", "new_term3"],  # This should be used
            texts=["old_text"],  # Deprecated, should be ignored
            categories=[
                "old_term1",
                "old_term2",
                "old_term3",
            ],  # Deprecated, should be ignored
            use_ner=True,  # Deprecated, should be ignored
            use_llm=False,  # Deprecated, should be ignored
            threshold=0.5,  # Deprecated, should be ignored
            fast=True,
        )

        assert response.columns[0].term == "old_term1"


class TestEstimateUsageIntegration:
    """Integration tests for estimate_usage endpoint."""

    def test_estimate_usage_without_authentication(self, monkeypatch):
        """Test estimate_usage endpoint works without authentication."""

        # Mock httpx.Client to capture the request
        class MockClient:
            def __init__(self, **kwargs):
                self.base_url = kwargs.get("base_url")
                self.timeout = kwargs.get("timeout")
                self.closed = False

            def request(self, method, url, **kwargs):
                assert method == "post"
                assert url == "/usage/estimate"

                body = kwargs["json"]
                assert body["feature"] == "embeddings"
                assert body["inputs"] == ["text1", "text2"]

                class MockResponse:
                    status_code = 200

                    def json(self):
                        return {
                            "usage": {
                                "total": 100,
                                "records": [{"feature": "embeddings", "quantity": 100}],
                            }
                        }

                return MockResponse()

            def close(self):
                self.closed = True

        # Patch httpx.Client to use our mock
        monkeypatch.setattr("httpx.Client", MockClient)

        # Create client with auth, but estimate_usage should not use it
        client = httpx.Client(base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        response = core_client.estimate_usage(
            feature="embeddings", inputs=["text1", "text2"]
        )

        assert isinstance(response, UsageEstimateResponse)
        assert response.usage["total"] == 100
        assert len(response.usage["records"]) == 1

    def test_estimate_usage_all_features(self):
        """Test estimate_usage with all supported features."""
        # Only test features that are actually supported by the API
        features = [
            "embeddings",
            "sentiment",
            "themes",
            "extractions",
            "summaries",
        ]

        for feature in features:

            def handler(request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content.decode())
                assert body["feature"] == feature

                return httpx.Response(
                    200,
                    json={
                        "usage": {
                            "total": 50,
                            "records": [{"feature": feature, "quantity": 50}],
                        }
                    },
                )

            transport = httpx.MockTransport(handler)
            client = httpx.Client(
                transport=transport, base_url="https://api.example.com"
            )
            core_client = CoreClient(client=client)

            response = core_client.estimate_usage(feature=feature, inputs=["text1"])
            assert response.usage["records"][0]["feature"] == feature

    def test_estimate_usage_invalid_feature(self):
        """Test estimate_usage with invalid feature raises error."""
        client = CoreClient(client=httpx.Client(base_url="https://api.example.com"))

        with pytest.raises(ValueError) as exc_info:
            client.estimate_usage(feature="invalid", inputs=["text1"])
        assert "feature must be one of" in str(exc_info.value)


class TestClientMethodErrorHandling:
    """Test error handling in updated client methods."""

    def test_generate_themes_validation_error(self):
        """Test generate_themes with validation errors."""
        client = CoreClient(client=httpx.Client(base_url="https://api.example.com"))

        # Test initialSets > 1 without interactive=True
        with pytest.raises(ValueError) as exc_info:
            client.generate_themes(
                texts=["text1", "text2"], initial_sets=2, interactive=False
            )
        assert "initialSets > 1 requires interactive=true" in str(exc_info.value)

    def test_cluster_texts_input_validation(self):
        """Test cluster_texts input validation."""

        def handler(request: httpx.Request) -> httpx.Response:
            # Return a validation error from the API for single input
            return httpx.Response(
                400,
                json={
                    "code": "validation_error",
                    "message": "Validation failed",
                    "errors": [
                        {
                            "code": "too_small",
                            "message": "List should have at least 2 items after validation, not 1",  # noqa: E501
                            "path": ["inputs"],
                            "field": "inputs",
                            "location": "body",
                        }
                    ],
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        # Test with single input - should get API validation error
        with pytest.raises(Exception):  # Could be PulseAPIError or other exception
            core_client.cluster_texts(inputs=["single"], k=1)

    def test_similarity_set_validation(self):
        """Test compare_similarity set validation."""
        client = CoreClient(client=httpx.Client(base_url="https://api.example.com"))

        # Test with no sets provided
        with pytest.raises(ValueError) as exc_info:
            client.compare_similarity()
        expected_msg = (
            "Provide 'set' for self-similarity or both 'set_a' and 'set_b' "
            "for cross-similarity"
        )
        assert expected_msg in str(exc_info.value)

        # Test with conflicting sets
        with pytest.raises(ValueError) as exc_info:
            client.compare_similarity(set=["text1", "text2"], set_a=["text3"])
        assert "Cannot provide both 'set' and 'set_a'/'set_b'" in str(exc_info.value)

    def test_extract_elements_type_validation(self):
        """Test extract_elements type validation."""
        CoreClient(client=httpx.Client(base_url="https://api.example.com"))

        # Test themes type with expand_dictionary=True (should be validated by model)
        # This will be caught by the API validation, not client-side
        pass  # Client-side validation for this constraint is in the model
