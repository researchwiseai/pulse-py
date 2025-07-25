import httpx
import time
from pulse.core.client import CoreClient
from pulse.core.models import SentimentResponse
from pulse.core.jobs import Job


def make_sync_client():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/sentiment"
        import json

        payload = json.loads(request.content.decode())
        assert payload.get("version") == "v1"
        data = {
            "results": [{"sentiment": "positive", "confidence": 0.9}],
            "requestId": "r1",
        }
        return httpx.Response(200, json=data)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://api.example.com")
    return CoreClient(client=client)


def make_async_client():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/sentiment":
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
                    "results": [{"sentiment": "negative", "confidence": 0.8}],
                    "requestId": "r2",
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://api.example.com")
    return CoreClient(client=client)


def test_analyze_sentiment_sync():
    client = make_sync_client()
    resp = client.analyze_sentiment(["a"], version="v1", fast=True)
    assert isinstance(resp, SentimentResponse)
    assert resp.results[0].sentiment == "positive"


def test_analyze_sentiment_async_job(monkeypatch):
    client = make_async_client()
    monkeypatch.setattr(time, "sleep", lambda x: None)
    job = client.analyze_sentiment(["a"], fast=False, await_job_result=False)
    assert isinstance(job, Job)
    monkeypatch.setattr(time, "sleep", lambda x: None)
    result = job.wait()
    assert result["results"][0]["sentiment"] == "negative"


def test_analyze_sentiment_async_wait(monkeypatch):
    client = make_async_client()
    monkeypatch.setattr(time, "sleep", lambda x: None)
    resp = client.analyze_sentiment(["a"], fast=False, await_job_result=True)
    assert isinstance(resp, SentimentResponse)
    assert resp.results[0].sentiment == "negative"
