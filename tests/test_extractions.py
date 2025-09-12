import time
import httpx
import pytest

from pulse.core.client import CoreClient
from pulse.core.models import ExtractionsResponse
from pulse.core.jobs import Job


def make_sync_client() -> CoreClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/extractions"
        return httpx.Response(
            200,
            json={
                "columns": [{"category": "b", "term": "b"}],
                "matrix": [["foo"]],
                "requestId": "r1",
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://api.example.com")
    return CoreClient(client=client)


def make_async_client() -> CoreClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/extractions":
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
                    "columns": [{"category": "b", "term": "b"}],
                    "matrix": [["bar"]],
                    "requestId": "r2",
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://api.example.com")
    return CoreClient(client=client)


def test_extract_elements_sync():
    client = make_sync_client()
    resp = client.extract_elements(texts=["a"], categories=["b"], fast=True)
    assert isinstance(resp, ExtractionsResponse)
    assert resp.columns[0].category == "b"
    assert resp.matrix[0][0] == "foo"


def test_extract_elements_async_job(monkeypatch):
    client = make_async_client()
    monkeypatch.setattr(time, "sleep", lambda s: None)
    job = client.extract_elements(
        texts=["a"], categories=["b"], fast=False, await_job_result=False
    )
    assert isinstance(job, Job)
    result = job.wait()
    assert result["matrix"][0][0] == "bar"


def test_extract_elements_async_wait(monkeypatch):
    client = make_async_client()
    monkeypatch.setattr(time, "sleep", lambda s: None)
    resp = client.extract_elements(
        texts=["a"], categories=["b"], fast=False, await_job_result=True
    )
    assert isinstance(resp, ExtractionsResponse)
    assert resp.matrix[0][0] == "bar"


def test_extract_elements_limits():
    client = make_sync_client()
    with pytest.raises(ValueError):
        client.extract_elements(texts=["a"] * 201, categories=["b"])
    with pytest.raises(ValueError):
        client.extract_elements(texts=["a"], categories=["b"] * 51)
