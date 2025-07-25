import time
import httpx
from pulse.core.client import CoreClient
from pulse.core.models import ClusteringResponse
from pulse.core.jobs import Job


def make_sync_client():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/clustering"
        return httpx.Response(
            200,
            json={
                "algorithm": "kmeans",
                "clusters": [
                    {"clusterId": 0, "items": ["a"]},
                    {"clusterId": 1, "items": ["b"]},
                ],
                "requestId": "r1",
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://api.example.com")
    return CoreClient(client=client)


def make_async_client():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/clustering":
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
                    "algorithm": "kmeans",
                    "clusters": [{"clusterId": 0, "items": ["a", "b"]}],
                    "requestId": "r2",
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://api.example.com")
    return CoreClient(client=client)


def test_cluster_texts_sync():
    client = make_sync_client()
    resp = client.cluster_texts(["a", "b"], k=2, fast=True)
    assert isinstance(resp, ClusteringResponse)
    assert len(resp.clusters) == 2
    assert resp.clusters[0].items == ["a"]


def test_cluster_texts_async_job(monkeypatch):
    client = make_async_client()
    monkeypatch.setattr(time, "sleep", lambda x: None)
    job = client.cluster_texts(["a", "b"], k=1, await_job_result=False)
    assert isinstance(job, Job)
    monkeypatch.setattr(time, "sleep", lambda x: None)
    result = job.wait()
    assert result["algorithm"] == "kmeans"


def test_cluster_texts_async_wait(monkeypatch):
    client = make_async_client()
    monkeypatch.setattr(time, "sleep", lambda x: None)
    resp = client.cluster_texts(["a", "b"], k=1, await_job_result=True)
    assert isinstance(resp, ClusteringResponse)
    assert resp.algorithm == "kmeans"
