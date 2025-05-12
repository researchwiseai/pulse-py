"""CoreClient for interacting with the Pulse API synchronously."""

from typing import Any, Dict, List, Union, Optional
import httpx
from pulse.core.gzip_client import GzipClient
from pulse.core.batching import _make_self_chunks, _make_cross_bodies, _stitch_results
from pulse.auth import ClientCredentialsAuth, AuthorizationCodePKCEAuth

from pulse.config import PROD_BASE_URL, DEFAULT_TIMEOUT
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


class CoreClient:
    """Synchronous CoreClient for Pulse API."""

    def __init__(
        self,
        base_url: str = PROD_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        client: Optional[httpx.Client] = None,
        auth: Optional[httpx.Auth] = None,
    ) -> None:
        """Initialize CoreClient with optional HTTPX client
        (for testing) and optional auth."""
        self.base_url = base_url
        self.timeout = timeout
        if client is not None:
            # Use provided HTTP client (user is responsible for auth)
            self.client = client
        else:
            # Create a GzipClient, apply auth for core API calls if provided
            self.client = GzipClient(
                base_url=self.base_url,
                timeout=self.timeout,
                auth=auth,
            )

    @classmethod
    def with_client_credentials(
        cls,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        audience: Optional[str] = None,
        token_url: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> "CoreClient":
        """
        Construct a CoreClient using OAuth2 Client Credentials flow.

        Credentials and configurations are resolved in the following
        order of preference:
        1. Direct function arguments.
        2. Environment variables:
           - PULSE_CLIENT_ID
           - PULSE_CLIENT_SECRET
           - PULSE_AUDIENCE
           - PULSE_TOKEN_URL
           - PULSE_BASE_URL
           - PULSE_ORGANIZATION_ID
        3. Default values:
           - token_url defaults to
             "https://research-wise-ai-eu.eu.auth0.com/oauth/token"
           - base_url defaults to PROD_BASE_URL (from pulse.config)
           - audience defaults to None if not otherwise specified.

        Args:
            client_id: OAuth2 client ID.
            client_secret: OAuth2 client secret.
            audience: The audience for the token request. Defaults to None.
            token_url: The token endpoint URL.
            base_url: The base URL for the Pulse API.
            scope: OAuth2 scope(s), space-separated.
            organization: The organization ID for the client.

        Returns:
            An instance of CoreClient configured with Client Credentials authentication.

        Raises:
            ValueError: If client_id or client_secret is not provided via arguments
                        or environment variables.
        """
        # Ensure 'import os' is at the top of the file for os.getenv()
        import os  # This import is placed here for snippet completeness,

        # ideally it's at the module level.

        # Resolve client_id: argument > environment variable
        final_client_id = client_id or os.getenv("PULSE_CLIENT_ID")
        if not final_client_id:
            raise ValueError(
                "Client ID must be provided either as an argument "
                "or via the PULSE_CLIENT_ID environment variable."
            )

        # Resolve client_secret: argument > environment variable
        final_client_secret = client_secret or os.getenv("PULSE_CLIENT_SECRET")
        if not final_client_secret:
            raise ValueError(
                "Client secret must be provided either as an argument "
                "or via the PULSE_CLIENT_SECRET environment variable."
            )

        # Resolve token_url: argument > environment variable > default
        default_token_url = "https://research-wise-ai-eu.eu.auth0.com/oauth/token"
        final_token_url = token_url or os.getenv("PULSE_TOKEN_URL") or default_token_url

        # Resolve audience: argument > environment variable (default is None if not set)
        final_audience = audience or os.getenv("PULSE_AUDIENCE")

        # Resolve base_url: argument > environment variable > default (PROD_BASE_URL)
        # PROD_BASE_URL should be imported from pulse.config at the module level.
        final_base_url = base_url or os.getenv("PULSE_BASE_URL") or PROD_BASE_URL

        auth = ClientCredentialsAuth(
            token_url=final_token_url,
            client_id=final_client_id,
            organization=organization,
            client_secret=final_client_secret,
            audience=final_audience,
        )

        return cls(base_url=final_base_url, auth=auth)

    @classmethod
    def with_pkce(
        cls,
        code: str,
        code_verifier: str,
        client_id: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        base_url: Optional[str] = None,
        token_url: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> "CoreClient":
        """
        Construct a CoreClient using OAuth2 Authorization Code flow with PKCE.

        Parameters like client_id, redirect_uri, base_url, and token_url are resolved
        in the following order of preference:
        1. Direct function arguments.
        2. Environment variables:
           - PULSE_CLIENT_ID
           - PULSE_REDIRECT_URI
           - PULSE_BASE_URL
           - PULSE_TOKEN_URL
           - PULSE_SCOPE
        3. Default values:
           - base_url defaults to PROD_BASE_URL (from pulse.config).
           - token_url defaults to
             "https://research-wise-ai-eu.eu.auth0.com/oauth/token".
           - scope defaults to None if not otherwise specified.

        `code` and `code_verifier` must always be provided as direct arguments.

        Args:
            code: The authorization code received from the authorization server.
            code_verifier: The PKCE code verifier.
            client_id: OAuth2 client ID.
            redirect_uri: The redirect URI used in the authorization request.
            base_url: The base URL for the Pulse API.
            token_url: The token endpoint URL.
            scope: OAuth2 scope(s), space-separated.

        Returns:
            An instance of CoreClient configured with PKCE authentication.

        Raises:
            ValueError: If `client_id` or `redirect_uri` is not provided via
                        arguments or environment variables.
        """
        # Ensure 'import os' is at the top of the file for os.getenv()
        import os  # This import is placed here for snippet completeness,

        # ideally it's at the module level.

        # Resolve client_id: argument > environment variable
        final_client_id = client_id or os.getenv("PULSE_CLIENT_ID")
        if not final_client_id:
            raise ValueError(
                "Client ID must be provided either as an argument "
                "or via the PULSE_CLIENT_ID environment variable."
            )

        # Resolve redirect_uri: argument > environment variable
        final_redirect_uri = redirect_uri or os.getenv("PULSE_REDIRECT_URI")
        if not final_redirect_uri:
            raise ValueError(
                "Redirect URI must be provided either as an argument "
                "or via the PULSE_REDIRECT_URI environment variable."
            )

        # Resolve base_url: argument > environment variable > default (PROD_BASE_URL)
        # PROD_BASE_URL should be imported from pulse.config at the module level.
        final_base_url = base_url or os.getenv("PULSE_BASE_URL") or PROD_BASE_URL

        # Resolve token_url: argument > environment variable > default
        default_token_url = "https://research-wise-ai-eu.eu.auth0.com/oauth/token"
        final_token_url = token_url or os.getenv("PULSE_TOKEN_URL") or default_token_url

        # Resolve scope: argument > environment variable (default is None if not set)
        final_scope = scope or os.getenv("PULSE_SCOPE")

        auth = AuthorizationCodePKCEAuth(
            token_url=final_token_url,
            client_id=final_client_id,
            code=code,  # Direct argument
            redirect_uri=final_redirect_uri,
            code_verifier=code_verifier,  # Direct argument
            scope=final_scope,
        )

        return cls(base_url=final_base_url, auth=auth)

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

        body: Dict[str, Any] = {}
        oversized = False
        if set is not None:
            body["set"] = set
            if len(set) > 200:
                oversized = True
        else:
            body["set_a"] = set_a
            body["set_b"] = set_b
            if len(set_a) * len(set_b) > 10_000:
                oversized = True

        if fast:
            body["fast"] = True

        if oversized and not fast:
            # If not fast and total size exceeds 10k, use batch similarity
            return self.batch_similarity(
                set=set,
                set_a=set_a,
                set_b=set_b,
                flatten=flatten,
            )

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
            result = job.wait(600)
            return SimilarityResponse.model_validate(result)

        # sync path
        return SimilarityResponse.model_validate(data)

    def _submit_batch_similarity_job(self, **kwargs) -> Any:
        body: Dict[str, Any] = {}
        body["flatten"] = kwargs["flatten"]
        if "set" in kwargs:
            body["set"] = kwargs["set"]
        elif "set_a" in kwargs and "set_b" in kwargs:
            body["set_a"] = kwargs["set_a"]
            body["set_b"] = kwargs["set_b"]
        else:
            raise ValueError("Must provide either `set` or both `set_a` and `set_b`.")

        response = self.client.post("/similarity", json=body)

        if response.status_code != 202:
            raise PulseAPIError(response)
        data = response.json()
        # Async/job path: initial submission returned only jobId
        submission = JobSubmissionResponse.model_validate(data)
        job = Job(id=submission.jobId, status="pending")
        job._client = self.client

        return job

    def batch_similarity(
        self,
        *,
        set: Optional[List[str]] = None,
        set_a: Optional[List[str]] = None,
        set_b: Optional[List[str]] = None,
        flatten: bool = True,
    ) -> Any:
        """
        Batch large similarity requests intelligently under the 10k-item limit.
        """
        if set is not None:
            chunks = _make_self_chunks(set)
            bodies: List[Dict[str, Any]] = []
            k = len(chunks)
            for i in range(k):
                for j in range(i, k):
                    if i == j:
                        bodies.append({"set": chunks[i], "flatten": flatten})
                    else:
                        bodies.append(
                            {"set_a": chunks[i], "set_b": chunks[j], "flatten": flatten}
                        )
        else:
            bodies = _make_cross_bodies(set_a or [], set_b or [], flatten)

        # submit all jobs
        jobs = [self._submit_batch_similarity_job(**body) for body in bodies]

        # wait for all jobs sequentially to preserve thread safety (e.g., under VCR)
        results = [job.wait(600) for job in jobs]

        full_a = set or set_a or []
        full_b = set or set_b or []
        return _stitch_results(results, bodies, full_a, full_b)

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
