import json
import time
import httpx

from pulse.core.client import CoreClient
from pulse.core.jobs import Job
from pulse.core.models import EmbeddingsResponse, EmbeddingsRequest


def make_sync_client():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embeddings"
        body = json.loads(request.content.decode())
        assert body.get("fast") is True
        data = {
            "embeddings": [{"text": "a", "vector": [1.0]}],
            "requestId": "r1",
        }
        return httpx.Response(200, json=data)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://api.example.com")
    return CoreClient(client=client)


def make_async_client():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/embeddings":
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
                    "embeddings": [{"text": "a", "vector": [1.0]}],
                    "requestId": "r2",
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://api.example.com")
    return CoreClient(client=client)


def test_create_embeddings_sync():
    client = make_sync_client()
    resp = client.create_embeddings(EmbeddingsRequest(inputs=["a"], fast=True))
    assert isinstance(resp, EmbeddingsResponse)
    assert resp.embeddings[0].text == "a"


def test_create_embeddings_async_job(monkeypatch):
    client = make_async_client()
    monkeypatch.setattr(time, "sleep", lambda x: None)
    job = client.create_embeddings(
        EmbeddingsRequest(inputs=["a"], fast=False), await_job_result=False
    )
    assert isinstance(job, Job)
    monkeypatch.setattr(time, "sleep", lambda x: None)
    result = job.wait()
    assert result["embeddings"][0]["text"] == "a"


def test_create_embeddings_async_wait(monkeypatch):
    client = make_async_client()
    monkeypatch.setattr(time, "sleep", lambda x: None)
    resp = client.create_embeddings(
        EmbeddingsRequest(inputs=["a"], fast=False), await_job_result=True
    )
    assert isinstance(resp, EmbeddingsResponse)
    assert resp.embeddings[0].text == "a"
