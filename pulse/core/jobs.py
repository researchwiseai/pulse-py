import time
from typing import Any, Optional, Literal
import httpx
from pydantic import BaseModel, PrivateAttr, Field, ConfigDict
from pulse.core.exceptions import PulseAPIError


class Job(BaseModel):
    """Represents an asynchronous job in Pulse API."""

    id: str = Field(alias="jobId")
    status: Literal["pending", "completed", "error", "failed"] = Field(
        alias="jobStatus"
    )
    message: Optional[str] = Field(default=None, alias="message")
    result_url: Optional[str] = Field(default=None, alias="resultUrl")

    _client: httpx.Client = PrivateAttr()
    model_config = ConfigDict(populate_by_name=True)

    def refresh(self, max_retries: int = 10, retry_delay: float = 10.0) -> "Job":
        """
        Refresh job status via GET /jobs?jobId={id}, retrying on 500 or 404
        up to max_retries times before giving up.
        """
        for attempt in range(max_retries):
            response = self._client.get(f"/jobs?jobId={self.id}")

            # retry on server‐error or not‐found
            if response.status_code in (500, 404):
                print(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} - "
                    f"Job {self.id} not found, retrying ({attempt + 1}/{max_retries})"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise PulseAPIError(response)

            # any other non‐200 is fatal
            if response.status_code != 200:
                raise PulseAPIError(response)

            # success!
            data = response.json()
            if "jobId" not in data:
                data["jobId"] = self.id

            job = Job.model_validate(data)
            job._client = self._client
            return job

        # should never get here
        raise PulseAPIError(response)

    def wait(self, timeout: float = 180.0) -> Any:
        start = time.time()
        while True:
            job = self.refresh()
            if job.status == "pending":
                pass
            elif job.status == "completed":
                if job.result_url:
                    response = self._client.get(job.result_url)
                    if response.status_code != 200:
                        raise PulseAPIError(response)
                    return response.json()
                return job
            else:
                error_msg = job.message or ""
                raise RuntimeError(f"Job {self.id} {job.status}: {error_msg}")

            if time.time() - start > timeout:
                raise TimeoutError(f"Job {self.id} did not finish in {timeout} seconds")
            time.sleep(2.0)
