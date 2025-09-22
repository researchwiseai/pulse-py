"""Tests for AsyncJob functionality."""

import asyncio
import pytest
import httpx
from unittest.mock import AsyncMock, Mock, patch
from pulse.core.async_jobs import AsyncJob
from pulse.core.exceptions import PulseAPIError


@pytest.fixture
def mock_async_client():
    """Create a mock async HTTP client."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


@pytest.fixture
def sample_job_data():
    """Sample job data for testing."""
    return {
        "jobId": "test-job-123",
        "jobStatus": "pending",
        "message": None,
        "resultUrl": None,
    }


@pytest.fixture
def async_job(mock_async_client, sample_job_data):
    """Create an AsyncJob instance for testing."""
    job = AsyncJob.model_validate(sample_job_data)
    job._client = mock_async_client
    return job


class TestAsyncJobCreation:
    """Test AsyncJob creation and basic properties."""

    def test_job_creation_from_dict(self, sample_job_data):
        """Test creating AsyncJob from dictionary."""
        job = AsyncJob.model_validate(sample_job_data)
        assert job.id == "test-job-123"
        assert job.status == "pending"
        assert job.message is None
        assert job.result_url is None

    def test_job_creation_with_aliases(self):
        """Test creating AsyncJob with field aliases."""
        data = {
            "jobId": "test-job-456",
            "jobStatus": "completed",
            "message": "Success",
            "resultUrl": "https://api.example.com/results/456",
        }
        job = AsyncJob.model_validate(data)
        assert job.id == "test-job-456"
        assert job.status == "completed"
        assert job.message == "Success"
        assert job.result_url == "https://api.example.com/results/456"


class TestAsyncJobRefresh:
    """Test AsyncJob refresh functionality."""

    @pytest.mark.asyncio
    async def test_refresh_success(self, async_job, mock_async_client):
        """Test successful job refresh."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "jobId": "test-job-123",
            "jobStatus": "completed",
            "message": "Done",
            "resultUrl": "https://api.example.com/results/123",
        }
        mock_async_client.get.return_value = mock_response

        refreshed_job = await async_job.refresh()

        assert refreshed_job.id == "test-job-123"
        assert refreshed_job.status == "completed"
        assert refreshed_job.message == "Done"
        mock_async_client.get.assert_called_once_with("/jobs?jobId=test-job-123")

    @pytest.mark.asyncio
    async def test_refresh_with_retries(self, async_job, mock_async_client):
        """Test refresh with retries on 404/500 errors."""
        # Mock 404 then success
        mock_404_response = Mock()
        mock_404_response.status_code = 404

        mock_success_response = Mock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {
            "jobId": "test-job-123",
            "jobStatus": "completed",
        }

        mock_async_client.get.side_effect = [mock_404_response, mock_success_response]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            refreshed_job = await async_job.refresh(max_retries=2, retry_delay=0.1)

        assert refreshed_job.status == "completed"
        assert mock_async_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_refresh_max_retries_exceeded(self, async_job, mock_async_client):
        """Test refresh when max retries are exceeded."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.headers = {}
        mock_async_client.get.return_value = mock_response

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(PulseAPIError):
                await async_job.refresh(max_retries=2, retry_delay=0.1)

    @pytest.mark.asyncio
    async def test_refresh_fatal_error(self, async_job, mock_async_client):
        """Test refresh with fatal error (non-404/500)."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.headers = {}
        mock_async_client.get.return_value = mock_response

        with pytest.raises(PulseAPIError):
            await async_job.refresh()


class TestAsyncJobWait:
    """Test AsyncJob wait functionality."""

    @pytest.mark.asyncio
    async def test_wait_for_completion_with_result_url(
        self, async_job, mock_async_client
    ):
        """Test waiting for job completion with result URL."""
        # Mock job status progression
        pending_response = Mock()
        pending_response.status_code = 200
        pending_response.json.return_value = {
            "jobId": "test-job-123",
            "jobStatus": "pending",
        }

        completed_response = Mock()
        completed_response.status_code = 200
        completed_response.json.return_value = {
            "jobId": "test-job-123",
            "jobStatus": "completed",
            "resultUrl": "https://api.example.com/results/123",
        }

        result_response = Mock()
        result_response.status_code = 200
        result_response.json.return_value = {"data": "test result"}

        mock_async_client.get.side_effect = [
            pending_response,
            completed_response,
            result_response,
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await async_job.wait()

        assert result == {"data": "test result"}

    @pytest.mark.asyncio
    async def test_wait_for_completion_without_result_url(
        self, async_job, mock_async_client
    ):
        """Test waiting for job completion without result URL."""
        completed_response = Mock()
        completed_response.status_code = 200
        completed_response.json.return_value = {
            "jobId": "test-job-123",
            "jobStatus": "completed",
            "resultUrl": None,
        }

        mock_async_client.get.return_value = completed_response

        result = await async_job.wait()

        assert isinstance(result, AsyncJob)
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_wait_timeout(self, async_job, mock_async_client):
        """Test wait timeout."""
        pending_response = Mock()
        pending_response.status_code = 200
        pending_response.json.return_value = {
            "jobId": "test-job-123",
            "jobStatus": "pending",
        }
        mock_async_client.get.return_value = pending_response

        # Use a very short timeout to trigger the timeout quickly
        with pytest.raises(asyncio.TimeoutError):
            await async_job.wait(timeout=0.001)

    @pytest.mark.asyncio
    async def test_wait_job_error(self, async_job, mock_async_client):
        """Test wait when job fails."""
        error_response = Mock()
        error_response.status_code = 200
        error_response.json.return_value = {
            "jobId": "test-job-123",
            "jobStatus": "error",
            "message": "Processing failed",
        }
        mock_async_client.get.return_value = error_response

        with pytest.raises(
            RuntimeError, match="Job test-job-123 error: Processing failed"
        ):
            await async_job.wait()


class TestAsyncJobCancel:
    """Test AsyncJob cancellation functionality."""

    @pytest.mark.asyncio
    async def test_cancel_success(self, async_job, mock_async_client):
        """Test successful job cancellation."""
        # Mock successful cancellation
        cancel_response = Mock()
        cancel_response.status_code = 200

        # Mock refresh after cancellation
        refresh_response = Mock()
        refresh_response.status_code = 200
        refresh_response.json.return_value = {
            "jobId": "test-job-123",
            "jobStatus": "cancelled",
        }

        mock_async_client.delete.return_value = cancel_response
        mock_async_client.get.return_value = refresh_response

        result = await async_job.cancel()

        assert result is True
        mock_async_client.delete.assert_called_once_with("/jobs/test-job-123")
        # Verify refresh was called after successful cancellation
        mock_async_client.get.assert_called_with("/jobs?jobId=test-job-123")

    @pytest.mark.asyncio
    async def test_cancel_failure(self, async_job, mock_async_client):
        """Test failed job cancellation."""
        cancel_response = Mock()
        cancel_response.status_code = 404
        mock_async_client.delete.return_value = cancel_response

        result = await async_job.cancel()

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_exception(self, async_job, mock_async_client):
        """Test cancellation with exception."""
        mock_async_client.delete.side_effect = Exception("Network error")

        result = await async_job.cancel()

        assert result is False


class TestAsyncJobWaitWithCallback:
    """Test AsyncJob wait with callback functionality."""

    @pytest.mark.asyncio
    async def test_wait_with_sync_callback(self, async_job, mock_async_client):
        """Test wait with synchronous callback."""
        callback_calls = []

        def sync_callback(status):
            callback_calls.append(status)

        # Mock status progression
        pending_response = Mock()
        pending_response.status_code = 200
        pending_response.json.return_value = {
            "jobId": "test-job-123",
            "jobStatus": "pending",
        }

        queued_response = Mock()
        queued_response.status_code = 200
        queued_response.json.return_value = {
            "jobId": "test-job-123",
            "jobStatus": "queued",
        }

        completed_response = Mock()
        completed_response.status_code = 200
        completed_response.json.return_value = {
            "jobId": "test-job-123",
            "jobStatus": "completed",
        }

        mock_async_client.get.side_effect = [
            pending_response,
            queued_response,
            completed_response,
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await async_job.wait_with_callback(sync_callback)

        assert isinstance(result, AsyncJob)
        assert callback_calls == ["pending", "queued", "completed"]

    @pytest.mark.asyncio
    async def test_wait_with_async_callback(self, async_job, mock_async_client):
        """Test wait with asynchronous callback."""
        callback_calls = []

        async def async_callback(status):
            callback_calls.append(status)

        completed_response = Mock()
        completed_response.status_code = 200
        completed_response.json.return_value = {
            "jobId": "test-job-123",
            "jobStatus": "completed",
        }

        mock_async_client.get.return_value = completed_response

        result = await async_job.wait_with_callback(async_callback)

        assert isinstance(result, AsyncJob)
        assert callback_calls == ["completed"]


class TestAsyncJobWaitForStatus:
    """Test AsyncJob wait for specific status functionality."""

    @pytest.mark.asyncio
    async def test_wait_for_specific_status(self, async_job, mock_async_client):
        """Test waiting for specific status."""
        # Mock status progression
        pending_response = Mock()
        pending_response.status_code = 200
        pending_response.json.return_value = {
            "jobId": "test-job-123",
            "jobStatus": "pending",
        }

        queued_response = Mock()
        queued_response.status_code = 200
        queued_response.json.return_value = {
            "jobId": "test-job-123",
            "jobStatus": "queued",
        }

        mock_async_client.get.side_effect = [pending_response, queued_response]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await async_job.wait_for_status("queued")

        assert result.status == "queued"

    @pytest.mark.asyncio
    async def test_wait_for_status_timeout(self, async_job, mock_async_client):
        """Test wait for status timeout."""
        pending_response = Mock()
        pending_response.status_code = 200
        pending_response.json.return_value = {
            "jobId": "test-job-123",
            "jobStatus": "pending",
        }
        mock_async_client.get.return_value = pending_response

        # Use a very short timeout to trigger the timeout quickly
        with pytest.raises(asyncio.TimeoutError):
            await async_job.wait_for_status("completed", timeout=0.001)

    @pytest.mark.asyncio
    async def test_wait_for_status_job_fails(self, async_job, mock_async_client):
        """Test wait for status when job fails."""
        error_response = Mock()
        error_response.status_code = 200
        error_response.json.return_value = {
            "jobId": "test-job-123",
            "jobStatus": "error",
            "message": "Processing failed",
        }
        mock_async_client.get.return_value = error_response

        with pytest.raises(
            RuntimeError, match="Job test-job-123 error: Processing failed"
        ):
            await async_job.wait_for_status("completed")


class TestAsyncJobResult:
    """Test AsyncJob result method (alias for wait)."""

    @pytest.mark.asyncio
    async def test_result_alias(self, async_job, mock_async_client):
        """Test that result() is an alias for wait()."""
        completed_response = Mock()
        completed_response.status_code = 200
        completed_response.json.return_value = {
            "jobId": "test-job-123",
            "jobStatus": "completed",
        }

        mock_async_client.get.return_value = completed_response

        result = await async_job.result()

        assert isinstance(result, AsyncJob)
        assert result.status == "completed"
