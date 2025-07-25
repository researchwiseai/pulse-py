import time
import httpx
from pulse.core.client import CoreClient
from pulse.core.models import SummariesResponse
from pulse.core.jobs import Job


def make_sync_client():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/summaries"
        data = {"summary": "foo", "requestId": "r1"}
        return httpx.Response(200, json=data)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://api.example.com")
    return CoreClient(client=client)


def make_async_client():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/summaries":
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
                json={"summary": "bar", "requestId": "r2"},
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://api.example.com")
    return CoreClient(client=client)


def test_generate_summary_sync():
    client = make_sync_client()
    resp = client.generate_summary(["a"], "question", fast=True)
    assert isinstance(resp, SummariesResponse)
    assert resp.summary == "foo"


def test_generate_summary_async_job(monkeypatch):
    client = make_async_client()
    monkeypatch.setattr(time, "sleep", lambda x: None)
    job = client.generate_summary(["a"], "q", await_job_result=False)
    assert isinstance(job, Job)
    monkeypatch.setattr(time, "sleep", lambda x: None)
    result = job.wait()
    assert result["summary"] == "bar"


def test_generate_summary_async_wait(monkeypatch):
    client = make_async_client()
    monkeypatch.setattr(time, "sleep", lambda x: None)
    resp = client.generate_summary(["a"], "q", await_job_result=True)
    assert isinstance(resp, SummariesResponse)
    assert resp.summary == "bar"
