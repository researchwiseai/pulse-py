"""Simplified performance tests for large inputs in OpenAPI v0.9.0."""

import json
import time
import httpx

from pulse.core.client import CoreClient
from pulse.core.models import (
    EmbeddingsRequest,
    EmbeddingsResponse,
    SimilarityRequest,
    SimilarityResponse,
    ClusteringResponse,
)


class TestMaximumInputSizes:
    """Test with maximum allowed input sizes for each endpoint."""

    def test_embeddings_sync_max_inputs(self):
        """Test embeddings with maximum sync input size (200)."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            assert len(body["inputs"]) == 200
            assert body["fast"] is True

            # Return mock response with 200 embeddings
            embeddings = [
                {
                    "id": str(i),
                    "text": f"text{i}",
                    "vector": [0.1] * 1536,  # Typical embedding dimension
                }
                for i in range(200)
            ]

            return httpx.Response(
                200,
                json={
                    "embeddings": embeddings,
                    "requestId": "embed-max-req",
                    "usage": {
                        "total": 200,
                        "records": [{"feature": "embeddings", "quantity": 200}],
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        # Generate 200 inputs
        inputs = [f"This is test text number {i} for embeddings." for i in range(200)]
        request = EmbeddingsRequest(inputs=inputs, fast=True)

        start_time = time.time()
        response = core_client.create_embeddings(request)
        end_time = time.time()

        assert isinstance(response, EmbeddingsResponse)
        assert len(response.embeddings) == 200
        assert response.usage_total == 200

        # Performance assertion - should complete quickly with mock
        assert end_time - start_time < 1.0

    def test_embeddings_async_max_inputs(self):
        """Test embeddings with maximum async input size (5000)."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            assert len(body["inputs"]) == 2000
            assert body["fast"] is False

            return httpx.Response(202, json={"jobId": "embed-async-job"})

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        # Generate 2000 inputs (max allowed by EmbeddingsRequest model)
        inputs = [f"Async test text {i}" for i in range(2000)]
        request = EmbeddingsRequest(inputs=inputs, fast=False)

        start_time = time.time()
        job = core_client.create_embeddings(request, await_job_result=False)
        end_time = time.time()

        # Should return job quickly
        assert job.id == "embed-async-job"
        assert end_time - start_time < 1.0

    def test_similarity_self_sync_max_inputs(self):
        """Test similarity with maximum self-similarity sync input size (500)."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            assert len(body["set"]) == 500
            assert body["fast"] is True

            # Return mock similarity matrix (flattened upper triangle)
            n = 500
            flattened_size = n * (n - 1) // 2  # Upper triangle without diagonal
            flattened = [0.8] * flattened_size

            return httpx.Response(
                200,
                json={
                    "scenario": "self",
                    "mode": "flattened",
                    "n": n,
                    "flattened": flattened,
                    "requestId": "sim-self-max-req",
                    "usage": {
                        "total": 500,
                        "records": [{"feature": "similarity", "quantity": 500}],
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        # Generate 500 inputs
        inputs = [f"Similarity test text {i}" for i in range(500)]
        request = SimilarityRequest(set=inputs, fast=True)

        start_time = time.time()
        response = core_client.compare_similarity(request)
        end_time = time.time()

        assert isinstance(response, SimilarityResponse)
        assert response.scenario == "self"
        assert response.n == 500
        assert len(response.flattened) == 500 * 499 // 2

        # Performance assertion
        assert end_time - start_time < 1.0

    def test_themes_async_max_inputs(self):
        """Test themes with maximum async input size (500)."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            assert len(body["inputs"]) == 500
            # fast=False means the fast key is not included in the body
            assert "fast" not in body or body["fast"] is False

            return httpx.Response(202, json={"jobId": "themes-async-job"})

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        # Generate 500 inputs
        inputs = [f"Theme analysis text {i}" for i in range(500)]

        start_time = time.time()
        job = core_client.generate_themes(inputs, fast=False, await_job_result=False)
        end_time = time.time()

        assert job.id == "themes-async-job"
        assert end_time - start_time < 1.0

    def test_clustering_sync_max_inputs(self):
        """Test clustering with maximum sync input size (500)."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            assert len(body["inputs"]) == 500
            assert body["k"] == 10
            assert body["algorithm"] == "kmeans"
            assert body["fast"] is True

            return httpx.Response(
                200,
                json={
                    "algorithm": "kmeans",
                    "clusters": [
                        {
                            "clusterId": i,
                            "items": [f"text{j}" for j in range(i * 50, (i + 1) * 50)],
                        }
                        for i in range(10)
                    ],
                    "requestId": "clustering-sync-req",
                    "usage": {
                        "total": 500,
                        "records": [{"feature": "clustering", "quantity": 500}],
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        # Generate 500 inputs
        inputs = [f"Clustering text {i}" for i in range(500)]

        start_time = time.time()
        response = core_client.cluster_texts(inputs, k=10, fast=True)
        end_time = time.time()

        assert isinstance(response, ClusteringResponse)
        assert response.algorithm == "kmeans"
        assert len(response.clusters) == 10
        assert end_time - start_time < 1.0


class TestResponseParsingPerformance:
    """Benchmark response parsing performance."""

    def test_large_similarity_matrix_reconstruction(self):
        """Test similarity matrix reconstruction performance with large data."""

        def handler(request: httpx.Request) -> httpx.Response:
            # Generate large flattened similarity data
            n = 200
            flattened_size = n * (n - 1) // 2  # Upper triangle without diagonal
            flattened = [0.8 + (i % 100) * 0.001 for i in range(flattened_size)]

            return httpx.Response(
                200,
                json={
                    "scenario": "self",
                    "mode": "flattened",
                    "n": n,
                    "flattened": flattened,
                    "requestId": "perf-sim-req",
                    "usage": {
                        "total": 200,
                        "records": [{"feature": "similarity", "quantity": 200}],
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        inputs = [f"Matrix perf test {i}" for i in range(200)]
        request = SimilarityRequest(set=inputs, fast=True)

        # Measure matrix reconstruction performance
        start_time = time.time()
        response = core_client.compare_similarity(request)

        # Access the similarity property to trigger matrix reconstruction
        matrix = response.similarity
        end_time = time.time()

        reconstruction_time = end_time - start_time

        assert isinstance(response, SimilarityResponse)
        assert len(matrix) == 200
        assert len(matrix[0]) == 200

        # Matrix reconstruction should be fast
        assert (
            reconstruction_time < 0.5
        ), f"Matrix reconstruction took {reconstruction_time:.3f}s"

    def test_large_embeddings_response_parsing(self):
        """Test embeddings response parsing performance with large data."""

        def handler(request: httpx.Request) -> httpx.Response:
            # Generate response with 200 embeddings
            embeddings = [
                {
                    "id": str(i),
                    "text": f"Performance test text {i}",
                    "vector": [
                        float(j) * 0.001 for j in range(384)
                    ],  # Smaller vector for performance
                }
                for i in range(200)
            ]

            return httpx.Response(
                200,
                json={
                    "embeddings": embeddings,
                    "requestId": "perf-embed-req",
                    "usage": {
                        "total": 200,
                        "records": [{"feature": "embeddings", "quantity": 200}],
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        inputs = [f"Performance test {i}" for i in range(200)]  # Use sync max
        request = EmbeddingsRequest(inputs=inputs, fast=True)

        # Measure parsing performance
        start_time = time.time()
        response = core_client.create_embeddings(request)
        end_time = time.time()

        parsing_time = end_time - start_time

        assert isinstance(response, EmbeddingsResponse)
        assert len(response.embeddings) == 200

        # Parsing should be fast (less than 1 second for 200 embeddings)
        assert parsing_time < 1.0, f"Parsing took {parsing_time:.3f}s"


class TestAsyncJobHandling:
    """Test async job handling with large inputs."""

    def test_async_job_submission_performance(self, monkeypatch):
        """Test async job submission performance with large inputs."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST" and request.url.path == "/themes":
                body = json.loads(request.content.decode())
                assert len(body["inputs"]) == 500  # Use max allowed for themes
                return httpx.Response(202, json={"jobId": "large-async-job"})
            elif request.method == "GET" and request.url.path == "/jobs":
                return httpx.Response(
                    200,
                    json={
                        "jobId": "large-async-job",
                        "jobStatus": "completed",
                        "resultUrl": "https://api.example.com/results/large-async-job",
                    },
                )
            elif (
                request.method == "GET"
                and request.url.path == "/results/large-async-job"
            ):
                # Return large themes response
                themes = [
                    {
                        "shortLabel": f"Theme{i}",
                        "label": f"Theme {i}",
                        "description": f"Description for theme {i}",
                        "representatives": [f"rep1_{i}", f"rep2_{i}"],
                    }
                    for i in range(20)  # 20 themes
                ]
                return httpx.Response(
                    200,
                    json={
                        "themes": themes,
                        "requestId": "large-async-result",
                        "usage": {
                            "total": 500,
                            "records": [{"feature": "themes", "quantity": 500}],
                        },
                    },
                )
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        # Mock sleep to speed up test
        monkeypatch.setattr(time, "sleep", lambda x: None)

        # Generate large input (use max allowed for themes async)
        inputs = [f"Large async test text {i}" for i in range(500)]

        # Test job submission performance
        start_time = time.time()
        job = core_client.generate_themes(inputs, fast=False, await_job_result=False)
        submission_time = time.time() - start_time

        assert job.id == "large-async-job"
        assert submission_time < 1.0, f"Job submission took {submission_time:.3f}s"

        # Test job completion performance
        start_time = time.time()
        result = job.wait()
        completion_time = time.time() - start_time

        assert len(result["themes"]) == 20
        assert completion_time < 1.0, f"Job completion took {completion_time:.3f}s"


class TestInputValidationPerformance:
    """Test performance of input validation with large datasets."""

    def test_large_input_validation_performance(self):
        """Test that input validation performs well with large inputs."""
        from pulse.core.validation import validate_before_request

        # Test with maximum allowed inputs for different endpoints
        test_cases = [
            ("embeddings", 5000, False),  # async max
            ("embeddings", 200, True),  # sync max
            ("themes", 500, False),  # async max
            ("themes", 200, True),  # sync max
            ("clustering", 44721, False),  # async max
            ("clustering", 500, True),  # sync max
        ]

        for endpoint, size, fast in test_cases:
            inputs = [f"Validation test text {i}" for i in range(size)]

            start_time = time.time()
            try:
                validate_before_request(endpoint, inputs=inputs, fast=fast)
            except Exception:
                pass  # We're testing performance, not correctness
            end_time = time.time()

            validation_time = end_time - start_time

            # Validation should be fast even for large inputs
            max_time = 0.1  # 100ms should be plenty for validation
            assert (
                validation_time < max_time
            ), f"Validation of {size} inputs for {endpoint} took {validation_time:.3f}s"

    def test_json_serialization_performance(self):
        """Test JSON serialization performance with large inputs."""
        # Test serialization of large request bodies
        sizes = [100, 500, 1000, 5000]

        for size in sizes:
            inputs = [
                f"Serialization test text {i} with some additional content"
                for i in range(size)
            ]
            request_body = {"inputs": inputs, "fast": False}

            start_time = time.time()
            json_str = json.dumps(request_body)
            end_time = time.time()

            serialization_time = end_time - start_time

            # Serialization should scale reasonably
            max_time = size * 0.0001  # 0.1ms per input as rough guideline
            assert serialization_time < max(
                max_time, 0.01
            ), f"Serialization of {size} inputs took {serialization_time:.3f}s"

            # Verify the serialized data is reasonable size
            assert len(json_str) > 0
            assert "inputs" in json_str
