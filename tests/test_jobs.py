"""Unit tests for asynchronous Job model and polling logic."""
import pytest

from pulse_client.core.jobs import Job
from pulse_client.core.exceptions import PulseAPIError


class DummyResponse:
    """Simulates HTTPX response with JSON payload."""

    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def json(self):
        return self._json


class DummyClient:
    """Simulates httpx.Client by returning predefined responses in sequence."""

    def __init__(self, responses):
        # Copy list to avoid mutation by tests
        self._responses = list(responses)
        self.calls = []

    def get(self, url):
        if not self._responses:
            raise RuntimeError("No more dummy responses")
        resp = self._responses.pop(0)
        self.calls.append(url)
        return resp


@pytest.fixture(autouse=True)
def disable_sleep(monkeypatch):
    """Prevent actual sleeping during tests."""
    import time

    monkeypatch.setattr(time, "sleep", lambda x: None)


def test_job_initial_state():
    # Job can be constructed with id and pending status
    job = Job(id="job-123", status="pending")
    assert job.id == "job-123"
    assert job.status == "pending"
    assert job.result_url is None
    assert job.message is None


def test_wait_until_completed_and_fetch_result():
    # Simulate polling: pending -> completed -> fetch result
    # First two .get calls return job status, then third returns result JSON
    responses = [
        DummyResponse({"jobId": "job-1", "jobStatus": "pending"}),
        DummyResponse(
            {
                "jobId": "job-1",
                "jobStatus": "completed",
                "resultUrl": "http://test/result",
            }
        ),
        DummyResponse({"value": 42}),
    ]
    client = DummyClient(responses)
    job = Job(id="job-1", status="pending")
    job._client = client
    result = job.wait(timeout=1)
    # Should return the JSON from the resultUrl
    assert result == {"value": 42}
    # Verify the sequence of URLs called: two polls then result fetch
    assert client.calls[:2] == [
        "/jobs?jobId=job-1",
        "/jobs?jobId=job-1",
    ]
    assert client.calls[2] == "http://test/result"


@pytest.mark.parametrize(
    "status, message",
    [
        ("error", "Something went wrong"),
        ("failed", "Process failed"),
    ],
)
def test_wait_raises_on_error_status(status, message):
    # Simulate polling: status transitions to error/failed
    responses = [
        DummyResponse({"jobId": "job-2", "jobStatus": status, "message": message}),
    ]
    client = DummyClient(responses)
    job = Job(id="job-2", status="pending")
    job._client = client
    with pytest.raises(RuntimeError) as excinfo:
        job.wait(timeout=1)
    # Error message should include job id and status
    assert f"Job {job.id} {status}" in str(excinfo.value)
    assert message in str(excinfo.value)


def test_refresh_raises_on_http_error():
    # Simulate non-200 status code on refresh
    bad_response = DummyResponse({"error": "bad"}, status_code=500)
    client = DummyClient([bad_response])
    job = Job(id="job-3", status="pending")
    job._client = client
    with pytest.raises(PulseAPIError):
        job.refresh()
