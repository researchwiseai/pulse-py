"""Job model and polling helpers."""

import time
from typing import Any, Optional, Literal

import httpx
from pydantic import BaseModel, PrivateAttr

from pulse_client.core.exceptions import PulseAPIError


class Job(BaseModel):
    """Represents an asynchronous job in Pulse API."""

    id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    result_url: Optional[str] = None

    _client: httpx.Client = PrivateAttr()

    def refresh(self) -> "Job":
        """Refresh job status via GET /jobs/{id}."""
        response = self._client.get(f"/jobs/{self.id}")
        if response.status_code != 200:
            raise PulseAPIError(response)
        data = response.json()
        # Pydantic v2: use model_validate instead of deprecated parse_obj
        job = Job.model_validate(data)
        job._client = self._client
        return job

    def wait(self, timeout: float = 60.0) -> Any:
        """Block until the job finishes or the timeout is hit."""
        start = time.time()
        while True:
            job = self.refresh()
            if job.status in ("succeeded", "failed"):
                if job.status == "failed":
                    raise RuntimeError(f"Job {self.id} failed")
                if job.result_url:
                    response = self._client.get(job.result_url)
                    if response.status_code != 200:
                        raise PulseAPIError(response)
                    return response.json()
                return job
            if time.time() - start > timeout:
                raise TimeoutError(f"Job {self.id} did not finish in {timeout} seconds")
            time.sleep(2.0)
