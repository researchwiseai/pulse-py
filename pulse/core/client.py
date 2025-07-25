"""CoreClient for interacting with the Pulse API synchronously."""

from typing import Any, Dict, List, Union, Optional
import warnings
import httpx
from pulse.core.gzip_client import GzipClient
from pulse.core.batching import _make_self_chunks, _make_cross_bodies, _stitch_results
from pulse.auth import ClientCredentialsAuth, AuthorizationCodePKCEAuth, auto_auth

from pulse.config import PROD_BASE_URL, DEFAULT_TIMEOUT
from pulse.core.jobs import Job
from pulse.core.models import (
    EmbeddingsResponse,
    SimilarityResponse,
    Theme,
    ThemesResponse,
    SentimentResponse,
    ExtractionsResponse,
    JobSubmissionResponse,
    ClusteringResponse,
    SummariesResponse,
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
                auth=auth or auto_auth(),
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
        self, inputs: list[str], fast: bool = True, *, await_job_result: bool = True
    ) -> Union[EmbeddingsResponse, Job]:
        """Generate dense vector embeddings.

        Args:
            inputs: Texts to embed.
            fast: Use synchronous (True) or asynchronous (False) mode.
            await_job_result: When False, return a :class:`Job` handle instead of
                waiting for the result when the server responds with HTTP 202.
        """

        # Request body according to OpenAPI spec: inputs
        body: Dict[str, Any] = {"inputs": inputs}
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
            if not await_job_result:
                return job
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
        fast: Optional[bool] = None,
        flatten: bool = False,
        version: str | None = None,
        split: Any | None = None,
        await_job_result: bool = True,
    ) -> Union[SimilarityResponse, Job]:
        """Compute cosine similarity between strings.

        Exactly one of ``set`` (self-similarity) or the pair ``set_a``/``set_b``
        (cross-similarity) must be provided. Optional ``version`` and ``split``
        values are forwarded to the API. When ``await_job_result`` is ``False``
        the asynchronous job handle is returned instead of waiting for
        completion.
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

        if oversized and not fast:
            # If not fast and total size exceeds 10k, use batch similarity
            return self.batch_similarity(
                set=set,
                set_a=set_a,
                set_b=set_b,
                flatten=flatten,
                version=version,
                split=split,
            )

        # API expects JSON boolean for flatten
        body["flatten"] = flatten
        if version is not None:
            body["version"] = version
        if split is not None:
            body["split"] = split

        if fast:
            # API expects a JSON boolean for fast
            body["fast"] = True
        else:
            # If not fast, set to False
            body["fast"] = False

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
            submission = JobSubmissionResponse.model_validate(data)
            job = Job(id=submission.jobId, status="pending")
            job._client = self.client
            if not await_job_result:
                return job
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

        if "version" in kwargs and kwargs["version"] is not None:
            body["version"] = kwargs["version"]
        if "split" in kwargs and kwargs["split"] is not None:
            body["split"] = kwargs["split"]

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
        flatten: bool = False,
        version: str | None = None,
        split: Any | None = None,
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
                        body = {"set": chunks[i], "flatten": flatten}
                        if version is not None:
                            body["version"] = version
                        if split is not None:
                            body["split"] = split
                        bodies.append(body)
                    else:
                        body = {
                            "set_a": chunks[i],
                            "set_b": chunks[j],
                            "flatten": flatten,
                        }
                        if version is not None:
                            body["version"] = version
                        if split is not None:
                            body["split"] = split
                        bodies.append(body)
        else:
            bodies = _make_cross_bodies(
                set_a or [], set_b or [], flatten, version, split
            )

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
        *,
        context: Any | None = None,
        version: str | None = None,
        prune: int | None = None,
        await_job_result: bool = True,
    ) -> Union[ThemesResponse, Job]:
        """Cluster texts into latent themes.

        Args:
            texts: Input strings to cluster.
            min_themes: Minimum number of themes.
            max_themes: Maximum number of themes.
            fast: Use synchronous (True) or asynchronous (False) mode.
            context: Optional context string guiding theme generation.
            version: Optional model version for reproducible output.
            prune: Optionally prune the specified number of
                lowest-frequency themes.
            await_job_result: When False, return a :class:`Job` handle
                instead of waiting.
        """
        # Build request body according to OpenAPI spec: inputs and theme options
        # For single-text input, return empty themes and assignments without API call
        if len(texts) < 2:
            # No-op placeholder for single input
            return ThemesResponse(themes=[], requestId=None)
        body: Dict[str, Any] = {}
        # Optionally include theme count bounds
        if min_themes is not None:
            body["minThemes"] = min_themes
        if max_themes is not None:
            body["maxThemes"] = max_themes
        if context is not None:
            body["context"] = context
        if version is not None:
            body["version"] = version
        if prune is not None:
            body["prune"] = prune
        # Fast flag for sync vs async
        if fast:
            # API expects a JSON boolean for fast
            body["fast"] = True

        max_inputs = 200 if fast is True else 500
        # Shuffle copy using Fisher-Yates algorithm
        # to avoid bias in theme generation
        if len(texts) > max_inputs:
            import random

            shuffled_texts = texts[:]
            random.shuffle(shuffled_texts)
            texts = shuffled_texts[:max_inputs]

        body["inputs"] = texts

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
            if not await_job_result:
                return job
            result = job.wait()
            return ThemesResponse.model_validate(result)
        # Synchronous response
        return ThemesResponse.model_validate(data)

    def analyze_sentiment(
        self,
        texts: list[str],
        *,
        version: str | None = None,
        fast: bool = True,
        await_job_result: bool = True,
    ) -> Union[SentimentResponse, Job]:
        """Classify sentiment for the given texts.

        Args:
            texts: List of input strings.
            version: Optional model version to use for reproducible output.
            fast: Use synchronous (True) or asynchronous (False) mode.
            await_job_result: When False, return a :class:`Job` handle instead of
                waiting for the result.
        """

        body: Dict[str, Any] = {"inputs": texts}
        if version is not None:
            body["version"] = version
        if fast:
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
        if response.status_code == 202:
            submission = JobSubmissionResponse.model_validate(data)
            job = Job(id=submission.jobId, status="pending")
            job._client = self.client
            if not await_job_result:
                return job
            result = job.wait()
            return SentimentResponse.model_validate(result)
        # Sync path
        return SentimentResponse.model_validate(data)

    def close(self) -> None:
        """Close underlying HTTP connection."""
        self.client.close()

    def get_job_status(self, job_id: str) -> Job:
        """Retrieve the status of a previously submitted job."""

        response = self.client.get("/jobs", params={"jobId": job_id})
        if response.status_code != 200:
            raise PulseAPIError(response)

        data = response.json()
        if "jobId" not in data:
            data["jobId"] = job_id

        job = Job.model_validate(data)
        job._client = self.client
        return job

    def extract_elements(
        self,
        texts: list[str] | None = None,
        categories: list[Union[str, Theme]] | None = None,
        *,
        version: str | None = None,
        dictionary: bool | None = None,
        fast: bool | None = None,
        await_job_result: bool = True,
        **legacy_kwargs: Any,
    ) -> Union[ExtractionsResponse, Job]:
        """Extract elements matching categories from input texts.

        Args:
            texts: Input strings to analyze.
            categories: List of categories or Theme objects.
            version: Optional model version for reproducible output.
            dictionary: When True, also include dictionary matches.
            fast: Use synchronous (True) or asynchronous (False) mode.
            await_job_result: When False, return a :class:`Job` handle
                instead of waiting.
        """

        if "inputs" in legacy_kwargs and texts is None:
            texts = legacy_kwargs.pop("inputs")
        if "themes" in legacy_kwargs and categories is None:
            warnings.warn(
                "'themes' argument is deprecated; use 'categories' instead",
                DeprecationWarning,
            )
            categories = legacy_kwargs.pop("themes")

        if legacy_kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(legacy_kwargs)}")

        if texts is None:
            raise TypeError("'texts' parameter is required")
        if categories is None:
            raise TypeError("'categories' parameter is required")

        serialized_categories = [
            c if isinstance(c, str) else c.label for c in categories
        ]

        body: Dict[str, Any] = {"texts": texts, "categories": serialized_categories}
        if version is not None:
            body["version"] = version
        if dictionary is not None:
            body["dictionary"] = dictionary
        if fast is not None:
            body["fast"] = fast

        response = self.client.post("/extractions", json=body)

        if response.status_code not in (200, 202):
            raise PulseAPIError(response)

        data = response.json()

        if response.status_code == 202:
            if fast:
                raise PulseAPIError(response)
            submission = JobSubmissionResponse.model_validate(data)
            job = Job(id=submission.jobId, status="pending")
            job._client = self.client
            if not await_job_result:
                return job
            result = job.wait()
            return ExtractionsResponse.model_validate(result)

        return ExtractionsResponse.model_validate(data)

    def cluster_texts(
        self,
        inputs: list[str],
        *,
        k: int,
        algorithm: str | None = None,
        fast: bool | None = None,
        await_job_result: bool = True,
    ) -> Union[ClusteringResponse, Job]:
        """Cluster texts into groups using embeddings.

        Args:
            inputs: Input strings to cluster.
            k: Desired number of clusters.
            algorithm: Optional clustering algorithm (``kmeans``, ``skmeans``,
                ``agglomerative``, ``hdbscan``).
            fast: Use synchronous (True) or asynchronous (False) mode.
            await_job_result: When False, return a :class:`Job` handle
                instead of waiting.
        """

        body: Dict[str, Any] = {"inputs": inputs, "k": k}
        if algorithm is not None:
            body["algorithm"] = algorithm
        if fast is not None:
            body["fast"] = fast

        response = self.client.post("/clustering", json=body)

        if response.status_code not in (200, 202):
            raise PulseAPIError(response)

        data = response.json()

        if response.status_code == 202:
            if fast:
                raise PulseAPIError(response)
            submission = JobSubmissionResponse.model_validate(data)
            job = Job(id=submission.jobId, status="pending")
            job._client = self.client
            if not await_job_result:
                return job
            result = job.wait()
            return ClusteringResponse.model_validate(result)

        return ClusteringResponse.model_validate(data)

    def generate_summary(
        self,
        inputs: list[str],
        question: str,
        *,
        length: str | None = None,
        preset: str | None = None,
        fast: bool | None = None,
        await_job_result: bool = True,
    ) -> Union[SummariesResponse, Job]:
        """Summarize text according to a question.

        Args:
            inputs: Input strings to summarize.
            question: Prompt describing the desired summary focus.
            length: Optional length specifier
                (``bullet-points``, ``short``, ``medium``, ``long``).
            preset: Optional preset controlling style.
            fast: Use synchronous (True) or asynchronous (False) mode.
            await_job_result: When False, return a :class:`Job` handle
                instead of waiting.
        """

        body: Dict[str, Any] = {"inputs": inputs, "question": question}
        if length is not None:
            body["length"] = length
        if preset is not None:
            body["preset"] = preset
        if fast is not None:
            body["fast"] = fast

        response = self.client.post("/summaries", json=body)

        if response.status_code not in (200, 202):
            raise PulseAPIError(response)

        data = response.json()

        if response.status_code == 202:
            if fast:
                raise PulseAPIError(response)
            submission = JobSubmissionResponse.model_validate(data)
            job = Job(id=submission.jobId, status="pending")
            job._client = self.client
            if not await_job_result:
                return job
            result = job.wait()
            return SummariesResponse.model_validate(result)

        return SummariesResponse.model_validate(data)
