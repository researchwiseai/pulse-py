"""CoreClient for interacting with the Pulse API synchronously."""

import os
from typing import Any, Dict, Union, Optional
import httpx
import gzip

from pulse.config import DEV_BASE_URL, DEFAULT_TIMEOUT
from pulse.core.jobs import Job
from pulse.core.models import (
    EmbeddingsResponse,
    SimilarityResponse,
    ThemesResponse,
    SentimentResponse,
    ExtractionsResponse,
    JobSubmissionResponse,
)
from pulse.core.exceptions import PulseAPIError


def _gzip_request_hook(request: httpx.Request) -> None:
    if os.getenv("DISABLE_GZIP"):
        return
    # Only gzip if there is a body
    if request.content:
        # compress the existing body
        compressed = gzip.compress(request.content)
        request.headers["Content-Encoding"] = "gzip"
        request.headers["Content-Length"] = str(len(compressed))
        # assign compressed bytes to the private _content attribute
        request._content = compressed


class CoreClient:
    """Synchronous CoreClient for Pulse API."""

    def __init__(
        self,
        base_url: str = DEV_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        client: Optional[httpx.Client] = None,
    ) -> None:
        """Initialize CoreClient with optional HTTPX client (for testing)."""
        self.base_url = base_url
        self.timeout = timeout
        self.client = (
            client
            if client is not None
            else httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                event_hooks={"request": [_gzip_request_hook]},
            )
        )

    @classmethod
    def with_client_credentials(
        cls, domain: str, client_id: str, client_secret: str
    ) -> "CoreClient":
        # TODO: implement OAuth2 credentials
        return cls()

    def create_embeddings(
        self, texts: list[str], fast: bool = True
    ) -> Union[EmbeddingsResponse, Job]:
        """Generate dense vector embeddings."""

        # Request body according to OpenAPI spec: inputs
        body: Dict[str, Any] = {"inputs": texts}
        if fast:
            # API expects a JSON boolean for fast
            body["fast"] = True

        response = self.client.post("/embeddings", json=body)

        if response.status_code not in (200, 202):
            raise PulseAPIError(response)

        data = response.json()

        # If service enqueues an async job during fast sync, treat as error
        if response.status_code == 202 and fast:
            raise PulseAPIError(response)

        # Async/job path: wrap and wait for completion (slow sync)
        if response.status_code == 202:
            # Async/job path: initial submission returned only jobId
            submission = JobSubmissionResponse.model_validate(data)
            job = Job(id=submission.jobId, status="pending")
            job._client = self.client
            result = job.wait()
            return EmbeddingsResponse.model_validate(result)
        # Synchronous response
        return EmbeddingsResponse.model_validate(data)

    def compare_similarity(
        self,
        *,
        set: list[str] | None = None,
        set_a: list[str] | None = None,
        set_b: list[str] | None = None,
        fast: bool = True,
        flatten: bool = True,
    ) -> Union[SimilarityResponse, Job]:
        """
        Compute cosine similarity.

        Must provide exactly one of:
          - set: list[str]         (self-similarity)
          - set_a: list[str] and set_b: list[str]   (cross-similarity)
        """
        # validate arguments
        if set is None and (set_a is None or set_b is None):
            raise ValueError(
                "You must provide either `set` or both `set_a` and `set_b`."
            )
        if set is not None and (set_a is not None or set_b is not None):
            raise ValueError("Cannot provide both `set` and `set_a`/`set_b`.")
        # client-side validation: limit number of similarity calculations
        # compute number of comparisons: square for self-similarity, product
        # for cross-similarity
        if set is not None:
            n = len(set)
            num_calcs = n * n
        else:
            # set_a and set_b are both defined (validated above)
            n_a = len(set_a)  # type: ignore[assignment]
            n_b = len(set_b)  # type: ignore[assignment]
            num_calcs = n_a * n_b
        # maximum allowed comparisons: smaller for fast mode
        max_calcs = 1_000_000 if fast else 2_000_000_000
        if num_calcs > max_calcs:
            raise ValueError(
                f"Number of similarity calculations ({num_calcs}) "
                + "exceeds the maximum allowed "
                f"({max_calcs}) for fast={fast}. Reduce input size "
                + "or adjust parameters."
            )

        body: Dict[str, Any] = {}
        if set is not None:
            body["set"] = set
        else:
            body["set_a"] = set_a
            body["set_b"] = set_b

        if fast:
            body["fast"] = True
        # API expects JSON boolean for flatten
        body["flatten"] = flatten

        response = self.client.post("/similarity", json=body)

        # handle error / single-item self-similarity fallback
        if response.status_code not in (200, 202):
            raise PulseAPIError(response)

        data = response.json()

        # async enqueued during fast sync is error
        if response.status_code == 202 and fast:
            raise PulseAPIError(response)

        # async/job path
        if response.status_code == 202:
            # Async/job path: initial submission returned only jobId
            submission = JobSubmissionResponse.model_validate(data)
            job = Job(id=submission.jobId, status="pending")
            job._client = self.client
            result = job.wait()
            return SimilarityResponse.model_validate(result)

        # sync path
        return SimilarityResponse.model_validate(data)

    def generate_themes(
        self,
        texts: list[str],
        min_themes: int = 2,
        max_themes: int = 50,
        fast: bool = True,
    ) -> Union[ThemesResponse, Job]:
        """Cluster texts into latent themes."""
        # Build request body according to OpenAPI spec: inputs and theme options
        # For single-text input, return empty themes and assignments without API call
        if len(texts) < 2:
            # No-op placeholder for single input
            return ThemesResponse(themes=[], requestId=None)
        body: Dict[str, Any] = {"inputs": texts}
        # Optionally include theme count bounds
        if min_themes is not None:
            body["minThemes"] = min_themes
        if max_themes is not None:
            body["maxThemes"] = max_themes
        # Fast flag for sync vs async
        if fast:
            # API expects a JSON boolean for fast
            body["fast"] = True
        response = self.client.post("/themes", json=body)
        if response.status_code not in (200, 202):
            raise PulseAPIError(response)
        data = response.json()
        # Async job enqueued during fast sync: error
        if response.status_code == 202 and fast:
            raise PulseAPIError(response)
        if response.status_code == 202:
            # Async/job path: initial submission returned only jobId
            submission = JobSubmissionResponse.model_validate(data)
            job = Job(id=submission.jobId, status="pending")
            job._client = self.client
            result = job.wait()
            return ThemesResponse.model_validate(result)
        # Synchronous response
        return ThemesResponse.model_validate(data)

    def analyze_sentiment(
        self, texts: list[str], fast: bool = True
    ) -> Union[SentimentResponse, Job]:
        """Classify sentiment."""
        # For single-text input, return empty sentiments without API call
        if len(texts) < 2:
            # No-op placeholder for single input
            return SentimentResponse(results=[], requestId=None)
        # Build request body according to OpenAPI spec: input array
        body: Dict[str, Any] = {"inputs": texts}
        if fast:
            # API expects a JSON boolean for fast
            body["fast"] = True
        response = self.client.post("/sentiment", json=body)
        # Raise on any error response
        if response.status_code not in (200, 202):
            raise PulseAPIError(response)
        # Parse payload
        data = response.json()
        # Async job enqueued during fast sync: error
        if response.status_code == 202 and fast:
            raise PulseAPIError(response)
        # Async job path: wait and parse
        if response.status_code == 202:
            # Async/job path: initial submission returned only jobId
            submission = JobSubmissionResponse.model_validate(data)
            job = Job(id=submission.jobId, status="pending")
            job._client = self.client
            result = job.wait()
            return SentimentResponse.model_validate(result)
        # Sync path
        return SentimentResponse.model_validate(data)

    def close(self) -> None:
        """Close underlying HTTP connection."""
        self.client.close()

    def extract_elements(
        self,
        inputs: list[str],
        themes: list[str],
        version: Optional[str] = None,
        fast: bool = True,
    ) -> Union[ExtractionsResponse, Job]:
        """Extract elements matching themes from input strings."""
        # Skip extraction when no themes provided (e.g., single-text low-level example)
        if not themes:
            # No-op placeholder when no themes provided
            return ExtractionsResponse(extractions=[], requestId=None)
        # Build request body according to OpenAPI spec: inputs, themes, optional version
        body: Dict[str, Any] = {"inputs": inputs, "themes": themes}
        if version is not None:
            body["version"] = version
        if fast:
            # API expects a JSON boolean for fast
            body["fast"] = True
        response = self.client.post("/extractions", json=body)
        # Raise on any error response
        if response.status_code not in (200, 202):
            raise PulseAPIError(response)
        data = response.json()
        # Async job enqueued during fast sync: error
        if response.status_code == 202 and fast:
            raise PulseAPIError(response)
        # Async job path: wait and parse
        if response.status_code == 202:
            # Async/job path: initial submission returned only jobId
            submission = JobSubmissionResponse.model_validate(data)
            job = Job(id=submission.jobId, status="pending")
            job._client = self.client
            result = job.wait()
            return ExtractionsResponse.model_validate(result)
        # Sync path
        return ExtractionsResponse.model_validate(data)
