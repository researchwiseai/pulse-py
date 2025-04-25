"""Tests for CoreClient and Job behavior."""

import time
import pytest

from pulse_client.core.client import CoreClient
from pulse_client.core.models import (
    EmbeddingsResponse,
    SimilarityResponse,
    ThemesResponse,
    SentimentResponse,
)
from pulse_client.core.jobs import Job
from pulse_client.core.exceptions import PulseAPIError


class DummyResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("No JSON")
        return self._json


class DummyClient:
    """Stub HTTPX client that simulates responses."""

    def __init__(self):
        self.requests = []
        self._job_counter = 0

    def post(self, path, json=None, params=None):
        self.requests.append(("POST", path, json, params))
        # Simulate for each endpoint
        if path == "/embeddings":
            if params.get("fast") == "true":
                return DummyResponse(200, {"embeddings": [[1.0, 2.0], [3.0, 4.0]]})
            return DummyResponse(
                202, {"id": "job1", "status": "queued", "result_url": "/jobs/job1"}
            )
        if path == "/similarity":
            if params.get("fast") == "true":
                return DummyResponse(200, {"similarity": [[0.1, 0.2], [0.3, 0.4]]})
            return DummyResponse(
                202, {"id": "job2", "status": "queued", "result_url": "/jobs/job2"}
            )
        if path == "/themes":
            if params.get("fast") == "true":
                return DummyResponse(200, {"themes": ["A", "B"], "assignments": [0, 1]})
            return DummyResponse(
                202, {"id": "job3", "status": "queued", "result_url": "/jobs/job3"}
            )
        if path == "/sentiment":
            if params.get("fast") == "true":
                return DummyResponse(200, {"sentiments": ["pos", "neg"]})
            return DummyResponse(
                202, {"id": "job4", "status": "queued", "result_url": "/jobs/job4"}
            )
        # Default error
        return DummyResponse(400, None, "Bad Request")

    def get(self, path):
        self.requests.append(("GET", path))
        # Fetch results first
        if path.endswith("/results"):
            # Return a dummy payload
            if "job1" in path:
                return DummyResponse(200, {"embeddings": [[9.0]]})
            if "job2" in path:
                return DummyResponse(200, {"similarity": [[0.9]]})
            if "job3" in path:
                return DummyResponse(200, {"themes": ["X"], "assignments": [1]})
            if "job4" in path:
                return DummyResponse(200, {"sentiments": ["neu"]})
        # Polling jobs
        if path.startswith("/jobs/"):
            # increment counter per job
            self._job_counter += 1
            if self._job_counter < 2:
                return DummyResponse(
                    200,
                    {
                        "id": path.split("/")[-1],
                        "status": "running",
                        "result_url": path,
                    },
                )
            # succeeded
            return DummyResponse(
                200,
                {
                    "id": path.split("/")[-1],
                    "status": "succeeded",
                    "result_url": f"{path}/results",
                },
            )
        return DummyResponse(404, None, "Not Found")

    def close(self):
        pass


@pytest.fixture(autouse=True)
def disable_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda x: None)


def test_create_embeddings_fast():
    dummy = DummyClient()
    client = CoreClient(client=dummy)
    resp = client.create_embeddings(["a", "b"], fast=True)
    assert isinstance(resp, EmbeddingsResponse)
    assert resp.embeddings == [[1.0, 2.0], [3.0, 4.0]]
    # JSON body should use 'inputs' as per OpenAPI spec
    assert dummy.requests == [
        ("POST", "/embeddings", {"inputs": ["a", "b"]}, {"fast": "true"})
    ]


def test_create_embeddings_job_wait():
    dummy = DummyClient()
    client = CoreClient(client=dummy)
    job = client.create_embeddings(["x"], fast=False)
    assert isinstance(job, Job)
    result = job.wait(timeout=5)
    assert result == {"embeddings": [[9.0]]}


@pytest.mark.parametrize(
    "method, path, response_key, response_class",
    [
        ("compare_similarity", "/similarity", "similarity", SimilarityResponse),
        ("generate_themes", "/themes", "themes", ThemesResponse),
        ("analyze_sentiment", "/sentiment", "sentiments", SentimentResponse),
    ],
)
def test_methods_fast(method, path, response_key, response_class):
    dummy = DummyClient()
    client = CoreClient(client=dummy)
    func = getattr(client, method)
    # call with fast=True
    resp = (
        func(["u", "v"], fast=True)
        if method != "compare_similarity"
        else func(["u", "v"], fast=True, flatten=False)
    )
    assert isinstance(resp, response_class)
    # ensure last request path matches
    assert dummy.requests[-1][1] == path


def test_error_raises():
    class BadClient(DummyClient):
        def post(self, path, json=None, params=None):
            return DummyResponse(500, {"error": "server"})

    client = CoreClient(client=BadClient())
    with pytest.raises(PulseAPIError):
        client.create_embeddings([], fast=True)
