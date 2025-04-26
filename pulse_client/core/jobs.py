"""Job model and polling helpers."""

import time
from typing import Any, Optional, Literal

import httpx
from pydantic import BaseModel, PrivateAttr

from pulse_client.core.exceptions import PulseAPIError


from pydantic import Field, ConfigDict


class Job(BaseModel):
    """Represents an asynchronous job in Pulse API."""

    id: str = Field(alias="jobId")
    status: Literal["pending", "completed", "error", "failed"] = Field(
        alias="jobStatus"
    )
    message: Optional[str] = Field(default=None, alias="message")
    result_url: Optional[str] = Field(default=None, alias="resultUrl")

    _client: httpx.Client = PrivateAttr()
    # Allow population by field name or alias
    model_config = ConfigDict(populate_by_name=True)

    def refresh(self) -> "Job":
        """Refresh job status via GET /jobs?jobId={id}"""
        response = self._client.get(f"/jobs?jobId={self.id}")
        if response.status_code != 200:
            raise PulseAPIError(response)
        data = response.json()
        # Ensure jobId is preserved if not returned in status payload
        if "jobId" not in data:
            data["jobId"] = self.id
        # Pydantic v2: use model_validate instead of deprecated parse_obj
        job = Job.model_validate(data)
        job._client = self._client
        return job

    def wait(self, timeout: float = 180.0) -> Any:
        """Block until the job finishes or the timeout is hit."""
        start = time.time()
        while True:
            job = self.refresh()
            if job.status == "pending":
                # still processing
                pass
            elif job.status == "completed":
                # completed successfully
                if job.result_url:
                    response = self._client.get(job.result_url)
                    if response.status_code != 200:
                        raise PulseAPIError(response)
                    return response.json()
                return job
            else:
                # error or failed
                error_msg = job.message or ""
                raise RuntimeError(f"Job {self.id} {job.status}: {error_msg}")
            if time.time() - start > timeout:
                raise TimeoutError(f"Job {self.id} did not finish in {timeout} seconds")
            time.sleep(2.0)
