import httpx
import time

from pulse.core.client import CoreClient
from pulse.core.jobs import Job


def make_polling_client():
    state = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/sentiment":
            return httpx.Response(202, json={"jobId": "job123"})
        if request.method == "GET" and request.url.path == "/jobs":
            state["count"] += 1
            if state["count"] == 1:
                return httpx.Response(
                    200, json={"jobId": "job123", "jobStatus": "pending"}
                )
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
                json={"results": [{"sentiment": "negative", "confidence": 0.5}]},
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://api.example.com")
    return CoreClient(client=client)


def test_get_job_status():
    client = make_polling_client()
    job = client.analyze_sentiment(["x"], fast=False, await_job_result=False)
    status = client.get_job_status(job.id)
    assert isinstance(status, Job)
    assert status.status == "pending"
    status = client.get_job_status(job.id)
    assert status.status == "completed"
    assert status.result_url is not None


def test_job_result_wrapper(monkeypatch):
    job = Job(id="job1", status="pending")
    job._client = httpx.Client()

    called = {}

    def fake_wait(self, timeout=180.0):
        called["called"] = True
        return "done"

    monkeypatch.setattr(Job, "wait", fake_wait)
    assert job.result(1) == "done"
    assert called.get("called")


def test_manual_polling(monkeypatch):
    client = make_polling_client()
    job = client.analyze_sentiment(["x"], fast=False, await_job_result=False)
    monkeypatch.setattr(time, "sleep", lambda x: None)
    while True:
        status = client.get_job_status(job.id)
        if status.status == "completed":
            resp = client.client.get(status.result_url)
            break
    assert resp.json()["results"][0]["sentiment"] == "negative"
