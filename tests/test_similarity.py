import json
import time
import httpx
from pulse.core.client import CoreClient
from pulse.core.jobs import Job
from pulse.core.models import SimilarityResponse, SimilarityRequest, Split


def make_sync_client():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/similarity"
        body = json.loads(request.content.decode())
        assert body["set"] == ["a", "b"]
        assert body["version"] == "v1"
        assert body["split"] == {"unit": "newline"}
        assert body["fast"] is True
        assert body["flatten"] is False
        data = {
            "scenario": "self",
            "mode": "flattened",
            "n": 2,
            "flattened": [1.0],
            "requestId": "r1",
        }
        return httpx.Response(200, json=data)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://api.example.com")
    return CoreClient(client=client)


def make_async_client():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/similarity":
            return httpx.Response(202, json={"jobId": "job123"})
        if request.method == "GET" and request.url.path == "/jobs":
            return httpx.Response(
                200,
                json={
                    "jobId": "job123",
                    "jobStatus": "completed",
                    "resultUrl": "https://api.example.com/results/job123",
                },
            )
        if request.method == "GET" and request.url.path == "/results/job123":
            return httpx.Response(
                200,
                json={
                    "scenario": "self",
                    "mode": "flattened",
                    "n": 2,
                    "flattened": [0.5],
                    "requestId": "r2",
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://api.example.com")
    return CoreClient(client=client)


def test_compare_similarity_sync():
    client = make_sync_client()
    resp = client.compare_similarity(
        SimilarityRequest(
            set=["a", "b"], fast=True, version="v1", split=Split(unit="newline")
        )
    )
    assert isinstance(resp, SimilarityResponse)
    assert resp.n == 2


def test_compare_similarity_async_job(monkeypatch):
    client = make_async_client()
    monkeypatch.setattr(time, "sleep", lambda x: None)
    job = client.compare_similarity(
        SimilarityRequest(set=["a", "b"], fast=False), await_job_result=False
    )
    assert isinstance(job, Job)
    monkeypatch.setattr(time, "sleep", lambda x: None)
    result = job.wait()
    assert result["flattened"] == [0.5]


def test_compare_similarity_async_wait(monkeypatch):
    client = make_async_client()
    monkeypatch.setattr(time, "sleep", lambda x: None)
    resp = client.compare_similarity(
        SimilarityRequest(set=["a", "b"], fast=False), await_job_result=True
    )
    assert isinstance(resp, SimilarityResponse)
    assert resp.flattened == [0.5]
