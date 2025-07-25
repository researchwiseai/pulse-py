import httpx
import time
import warnings
from pulse.core.client import CoreClient
from pulse.core.models import ExtractionsResponse
from pulse.core.jobs import Job


def make_sync_client():
    def handler(request: httpx.Request) -> httpx.Response:
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


def make_async_client():
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
    assert resp.columns[0].term == "b"
    assert resp.matrix[0][0] == "foo"


def test_extract_elements_async_job(monkeypatch):
    client = make_async_client()
    monkeypatch.setattr(time, "sleep", lambda x: None)
    job = client.extract_elements(texts=["a"], categories=["b"], await_job_result=False)
    assert isinstance(job, Job)
    monkeypatch.setattr(time, "sleep", lambda x: None)
    result = job.wait()
    assert result["columns"][0]["category"] == "b"
    assert result["matrix"][0][0] == "bar"


def test_extract_elements_async_wait(monkeypatch):
    client = make_async_client()
    monkeypatch.setattr(time, "sleep", lambda x: None)
    resp = client.extract_elements(texts=["a"], categories=["b"], await_job_result=True)
    assert isinstance(resp, ExtractionsResponse)
    assert resp.columns[0].category == "b"
    assert resp.matrix[0][0] == "bar"


def test_extract_elements_themes_deprecation():
    client = make_sync_client()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        resp = client.extract_elements(texts=["a"], themes=["b"], fast=True)
    assert isinstance(resp, ExtractionsResponse)
    assert any(issubclass(item.category, DeprecationWarning) for item in w)


def test_extract_elements_inputs_compat():
    client = make_sync_client()
    resp = client.extract_elements(inputs=["a"], categories=["b"], fast=True)
    assert isinstance(resp, ExtractionsResponse)
