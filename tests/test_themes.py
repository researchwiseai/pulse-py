import json
import httpx
import time
from pulse.core.client import CoreClient
from pulse.core.jobs import Job
from pulse.core.models import ThemesResponse


def make_sync_client():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/themes"
        body = json.loads(request.content.decode())
        assert body["minThemes"] == 1
        assert body["maxThemes"] == 3
        assert body["context"] == "ctx"
        assert body["version"] == "v1"
        assert body["prune"] == 2
        assert body["fast"] is True
        return httpx.Response(200, json={"themes": [], "requestId": "r1"})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://api.example.com")
    return CoreClient(client=client)


def make_async_client():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/themes":
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
                    "themes": [],
                    "requestId": "r2",
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://api.example.com")
    return CoreClient(client=client)


def test_generate_themes_sync():
    client = make_sync_client()
    resp = client.generate_themes(
        ["a", "b"],
        min_themes=1,
        max_themes=3,
        fast=True,
        context="ctx",
        version="v1",
        prune=2,
    )
    assert isinstance(resp, ThemesResponse)


def test_generate_themes_async_job(monkeypatch):
    client = make_async_client()
    monkeypatch.setattr(time, "sleep", lambda x: None)
    job = client.generate_themes(["a", "b"], fast=False, await_job_result=False)
    assert isinstance(job, Job)
    monkeypatch.setattr(time, "sleep", lambda x: None)
    result = job.wait()
    assert result["themes"] == []


def test_generate_themes_async_wait(monkeypatch):
    client = make_async_client()
    monkeypatch.setattr(time, "sleep", lambda x: None)
    resp = client.generate_themes(["a", "b"], fast=False, await_job_result=True)
    assert isinstance(resp, ThemesResponse)
