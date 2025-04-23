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
        self.client = client if client is not None else httpx.Client(
            base_url=self.base_url, timeout=self.timeout
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
         params: Dict[str, str] = {}
         if fast:
             params["fast"] = "true"
         response = self.client.post("/embeddings", json={"texts": texts}, params=params)
         if response.status_code not in (200, 202):
             raise PulseAPIError(response)
         data = response.json()
         if fast:
             return EmbeddingsResponse.parse_obj(data)
         job = Job.parse_obj(data)
         job._client = self.client
         return job

    def compare_similarity(
         self, texts: list[str], fast: bool = True, flatten: bool = True
     ) -> Union[SimilarityResponse, Job]:
         """Compute cosine similarity."""
         params: Dict[str, str] = {"flatten": str(flatten).lower()}
         if fast:
             params["fast"] = "true"
         response = self.client.post(
             "/similarity", json={"texts": texts}, params=params
         )
         if response.status_code not in (200, 202):
             raise PulseAPIError(response)
         data = response.json()
         if fast:
             return SimilarityResponse.parse_obj(data)
         job = Job.parse_obj(data)
         job._client = self.client
         return job

    def generate_themes(
         self, texts: list[str], min_themes: int = 2, max_themes: int = 10, fast: bool = True
     ) -> Union[ThemesResponse, Job]:
         """Cluster texts into latent themes."""
         params: Dict[str, str] = {"min_themes": str(min_themes), "max_themes": str(max_themes)}
         if fast:
             params["fast"] = "true"
         response = self.client.post(
             "/themes", json={"texts": texts}, params=params
         )
         if response.status_code not in (200, 202):
             raise PulseAPIError(response)
         data = response.json()
         if fast:
             return ThemesResponse.parse_obj(data)
         job = Job.parse_obj(data)
         job._client = self.client
         return job

    def analyze_sentiment(
         self, texts: list[str], fast: bool = True
     ) -> Union[SentimentResponse, Job]:
         """Classify sentiment."""
         params: Dict[str, str] = {}
         if fast:
             params["fast"] = "true"
         response = self.client.post("/sentiment", json={"texts": texts}, params=params)
         if response.status_code not in (200, 202):
             raise PulseAPIError(response)
         data = response.json()
         if fast:
             return SentimentResponse.parse_obj(data)
         job = Job.parse_obj(data)
         job._client = self.client
         return job

    def close(self) -> None:
         """Close underlying HTTP connection."""
         self.client.close()