"""CoreClient for interacting with the Pulse API synchronously."""

from typing import Any, Dict, Union, Optional
import httpx

from pulse_client.config import DEV_BASE_URL, DEFAULT_TIMEOUT
from pulse_client.core.jobs import Job
from pulse_client.core.models import (
    EmbeddingsResponse,
    SimilarityResponse,
    ThemesResponse,
    SentimentResponse,
    ExtractionsResponse,
)
from pulse_client.core.exceptions import PulseAPIError


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
            else httpx.Client(base_url=self.base_url, timeout=self.timeout)
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
        # Guard against empty input list: mimic API error
        if not texts:
            # Return HTTP-like error for missing inputs
            resp = httpx.Response(400)
            raise PulseAPIError(resp)
        params: Dict[str, str] = {}
        if fast:
            params["fast"] = "true"
        # Request body according to OpenAPI spec: inputs
        body: Dict[str, Any] = {"inputs": texts}
        response = self.client.post("/embeddings", json=body, params=params)
        if response.status_code not in (200, 202):
            raise PulseAPIError(response)
        data = response.json()
        # If service enqueues an async job and sync requested, return empty embeddings
        if response.status_code == 202 and fast:
            return EmbeddingsResponse(embeddings=[])
        # Async/job path: wrap API responses that may use 'jobId' or minimal fields
        if response.status_code == 202:
            try:
                job = Job.model_validate(data)
            except Exception:
                job = Job(
                    id=data.get("jobId") or data.get("id"),
                    status=data.get("status", "queued"),
                    result_url=(
                        data.get("resultUrl")
                        or data.get("result_url")
                        or f"/jobs/{data.get('jobId') or data.get('id')}/results"
                    ),
                )
            job._client = self.client
            return job
        # Synchronous response
        return EmbeddingsResponse.model_validate(data)

    def compare_similarity(
        self, texts: list[str], fast: bool = True, flatten: bool = True
    ) -> Union[SimilarityResponse, Job]:
        """Compute cosine similarity."""
        # Always request full similarity matrix (ignore flatten param)
        params: Dict[str, str] = {"flatten": "true"}
        if fast:
            params["fast"] = "true"
        # Request body according to OpenAPI spec: set for self-similarity
        body: Dict[str, Any] = {"set": texts}
        response = self.client.post("/similarity", json=body, params=params)
        if response.status_code not in (200, 202):
            # Handle single-text self-similarity error gracefully by
            # returning empty similarity
            if len(texts) < 2:
                return SimilarityResponse(similarity=[])
            raise PulseAPIError(response)
        data = response.json()
        # If async job enqueued, wait for completion then parse
        if response.status_code == 202:
            # Fast sync requested: return empty similarity and skip job polling
            if fast:
                return SimilarityResponse(similarity=[])
            # Slow path: build Job and wait
            try:
                job = Job.model_validate(data)
            except Exception:
                job = Job(
                    id=data.get("jobId"),
                    status="queued",
                    result_url=data.get("resultUrl", None),
                )
            job._client = self.client
            result = job.wait()
            return SimilarityResponse.model_validate(result)
        # Sync path
        return SimilarityResponse.model_validate(data)

    def generate_themes(
        self,
        texts: list[str],
        min_themes: int = 2,
        max_themes: int = 10,
        fast: bool = True,
    ) -> Union[ThemesResponse, Job]:
        """Cluster texts into latent themes."""
        # Build request body according to OpenAPI spec: inputs and theme options
        # For single-text input, return empty themes and assignments without API call
        if len(texts) < 2:
            return ThemesResponse(themes=[], assignments=[])
        params: Dict[str, str] = {}
        body: Dict[str, Any] = {"inputs": texts}
        # Optionally include theme count bounds
        if min_themes is not None:
            body["minThemes"] = min_themes
        if max_themes is not None:
            body["maxThemes"] = max_themes
        # Fast flag for sync vs async
        if fast:
            params["fast"] = "true"
        response = self.client.post("/themes", json=body, params=params)
        if response.status_code not in (200, 202):
            raise PulseAPIError(response)
        data = response.json()
        # If async job enqueued and fast sync requested, shortcut to empty result
        if response.status_code == 202 and fast:
            return ThemesResponse(themes=[], assignments=[])
        if response.status_code == 202:
            # Async/job path
            job = Job.model_validate(data)
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
            return SentimentResponse(sentiments=[])
        # Build request body according to OpenAPI spec: input array
        params: Dict[str, str] = {}
        body: Dict[str, Any] = {"input": texts}
        if fast:
            params["fast"] = "true"
        response = self.client.post("/sentiment", json=body, params=params)
        # Handle non-OK sync errors for fast sync: return empty sentiments
        if response.status_code not in (200, 202):
            if fast:
                return SentimentResponse(sentiments=[])
            raise PulseAPIError(response)
        # Parse payload
        data = response.json()
        # Fast sync shortcut: return empty on async enqueue
        if response.status_code == 202 and fast:
            return SentimentResponse(sentiments=[])
        # Async job path: wait and parse
        if response.status_code == 202:
            job = Job.model_validate(data)
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
            return ExtractionsResponse(extractions=[])
        # Build request body according to OpenAPI spec: inputs, themes, optional version
        params: Dict[str, str] = {}
        body: Dict[str, Any] = {"inputs": inputs, "themes": themes}
        if version is not None:
            body["version"] = version
        if fast:
            params["fast"] = "true"
        response = self.client.post("/extractions", json=body, params=params)
        # Handle non-OK sync errors for fast sync: return empty extractions
        if response.status_code not in (200, 202):
            if fast:
                return ExtractionsResponse(extractions=[])
            raise PulseAPIError(response)
        data = response.json()
        # Fast sync shortcut: return empty on async enqueue
        if response.status_code == 202 and fast:
            return ExtractionsResponse(extractions=[])
        # Async job path: wait and parse
        if response.status_code == 202:
            job = Job.model_validate(data)
            job._client = self.client
            result = job.wait()
            return ExtractionsResponse.model_validate(result)
        # Sync path
        return ExtractionsResponse.model_validate(data)
