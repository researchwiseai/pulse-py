"""AsyncCoreClient for interacting with the Pulse API asynchronously."""

import asyncio
from typing import Any, Dict, List, Union, Optional
import httpx
from pulse.core.async_retry import async_retry_request
from pulse.core.utils import chunk_texts
from pulse.core.async_gzip_client import AsyncGzipClient

try:
    from pulse.core.batching import (
        _make_self_chunks,
        _make_cross_bodies,
        _stitch_results,
    )

    HAS_BATCHING = True
except ImportError:
    HAS_BATCHING = False
    _make_self_chunks = None
    _make_cross_bodies = None
    _stitch_results = None

from pulse.auth import ClientCredentialsAuth, AuthorizationCodePKCEAuth, auto_auth
from pulse.debug import debug_request, _log_response, log_request_failure

from pulse.config import BASE_URL, DEFAULT_TIMEOUT
from pulse.core.async_jobs import AsyncJob
from pulse.core.models import (
    EmbeddingsRequest,
    EmbeddingsResponse,
    SimilarityRequest,
    SimilarityResponse,
    ThemesResponse,
    ThemeSetsResponse,
    SentimentResponse,
    JobSubmissionResponse,
    ExtractionsResponse,
    ClusteringResponse,
    SummariesResponse,
    UsageEstimateResponse,
)
from pulse.core.exceptions import PulseAPIError
from pulse.core.validation import (
    validate_before_request,
    PulseValidationError,
)

MAX_SENTIMENT = 10000
MAX_THEMES = 500
MAX_SUMMARIES = 5000


class AsyncCoreClient:
    """Asynchronous CoreClient for Pulse API."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        client: Optional[httpx.AsyncClient] = None,
        auth: Optional[httpx.Auth] = None,
    ) -> None:
        """Initialize AsyncCoreClient with optional HTTPX async client
        (for testing) and optional auth."""
        self.base_url = base_url
        self.timeout = timeout
        if client is not None:
            # Use provided HTTP client (user is responsible for auth)
            self.client = client
        else:
            # Create an AsyncGzipClient, apply auth for core API calls if provided
            self.client = AsyncGzipClient(
                base_url=self.base_url,
                timeout=self.timeout,
                auth=auth or auto_auth(),
            )

    async def __aenter__(self) -> "AsyncCoreClient":
        """Async context manager entry."""
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with proper resource cleanup."""
        await self.client.__aexit__(exc_type, exc_val, exc_tb)

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make an async HTTP request with retry logic and debug support."""
        with debug_request(method, url, **kwargs) as timing:
            try:
                response = await async_retry_request(
                    lambda: self.client.request(method, url, **kwargs)
                )
                timing.status_code = response.status_code
                _log_response(response)
                return response
            except Exception as e:
                log_request_failure(e)
                raise

    @classmethod
    async def with_client_credentials_async(
        cls,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        audience: Optional[str] = None,
        token_url: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> "AsyncCoreClient":
        """
        Construct an AsyncCoreClient using OAuth2 Client Credentials flow.

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
           - base_url defaults to BASE_URL (from pulse.config)
           - audience defaults to None if not otherwise specified.

        Args:
            client_id: OAuth2 client ID.
            client_secret: OAuth2 client secret.
            audience: OAuth2 audience.
            token_url: OAuth2 token endpoint URL.
            base_url: Pulse API base URL.
            organization: Organization ID for multi-tenant setups.

        Returns:
            AsyncCoreClient instance configured with Client Credentials auth.
        """
        auth = ClientCredentialsAuth(
            client_id=client_id,
            client_secret=client_secret,
            audience=audience,
            token_url=token_url,
            organization=organization,
        )

        final_base_url = base_url or BASE_URL
        return cls(base_url=final_base_url, auth=auth)

    @classmethod
    async def with_pkce_async(
        cls,
        code: str,
        code_verifier: str,
        redirect_uri: str,
        client_id: Optional[str] = None,
        audience: Optional[str] = None,
        token_url: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> "AsyncCoreClient":
        """
        Construct an AsyncCoreClient using OAuth2 Authorization Code with PKCE flow.

        Args:
            code: Authorization code from the OAuth2 callback.
            code_verifier: PKCE code verifier.
            redirect_uri: Redirect URI used in the authorization request.
            client_id: OAuth2 client ID.
            audience: OAuth2 audience.
            token_url: OAuth2 token endpoint URL.
            base_url: Pulse API base URL.
            organization: Organization ID for multi-tenant setups.

        Returns:
            AsyncCoreClient instance configured with PKCE auth.
        """
        auth = AuthorizationCodePKCEAuth(
            code=code,
            code_verifier=code_verifier,
            redirect_uri=redirect_uri,
            client_id=client_id,
            audience=audience,
            token_url=token_url,
            organization=organization,
        )

        final_base_url = base_url or BASE_URL
        return cls(base_url=final_base_url, auth=auth)

    async def close(self) -> None:
        """Close underlying HTTP connection."""
        await self.client.aclose()

    async def create_embeddings(
        self, request: EmbeddingsRequest, *, await_job_result: bool = True
    ) -> Union[EmbeddingsResponse, AsyncJob]:
        """Create embeddings for input texts asynchronously.

        Args:
            request: Embeddings request containing input texts and options.
            await_job_result: When False, return an AsyncJob handle
                instead of waiting for completion.

        Returns:
            EmbeddingsResponse with embeddings data, or AsyncJob if
            await_job_result=False.
        """
        validate_before_request(request)

        # Use parallel batching if available and beneficial
        if HAS_BATCHING and len(request.inputs) > 1 and not request.fast:
            return await self._create_embeddings_with_parallel_batching(
                request, await_job_result=await_job_result
            )
        else:
            return await self._create_single_embeddings_request(
                request, await_job_result=await_job_result
            )

    async def _create_single_embeddings_request(
        self, request: EmbeddingsRequest, *, await_job_result: bool = True
    ) -> Union[EmbeddingsResponse, AsyncJob]:
        """Create a single embeddings request."""
        body = request.model_dump(exclude_none=True, by_alias=True)
        response = await self._request("post", "/embeddings", json=body)

        if response.status_code not in (200, 202):
            raise PulseAPIError(response)

        data = response.json()

        if response.status_code == 202:
            submission = JobSubmissionResponse.model_validate(data)
            job = AsyncJob(jobId=submission.jobId, jobStatus="pending")
            job._client = self.client
            if not await_job_result:
                return job
            result = await job.wait()
            return EmbeddingsResponse.model_validate(result)

        return EmbeddingsResponse.model_validate(data)

    async def _create_embeddings_with_parallel_batching(
        self, request: EmbeddingsRequest, *, await_job_result: bool = True
    ) -> Union[EmbeddingsResponse, AsyncJob]:
        """Create embeddings using parallel batching for improved performance."""
        if not HAS_BATCHING:
            return await self._create_single_embeddings_request(
                request, await_job_result=await_job_result
            )

        # Create batches
        batches = _make_self_chunks(request.inputs)
        if len(batches) == 1:
            return await self._create_single_embeddings_request(
                request, await_job_result=await_job_result
            )

        # Create job submitters for each batch
        async def create_batch_submitter(batch_inputs):
            batch_request = EmbeddingsRequest(
                inputs=batch_inputs,
                model=request.model,
                dimensions=request.dimensions,
                fast=request.fast,
            )
            return await self._create_single_embeddings_request(
                batch_request, await_job_result=True
            )

        # Submit all batches concurrently
        tasks = [create_batch_submitter(batch) for batch in batches]
        results = await asyncio.gather(*tasks)

        # Stitch results together
        return _stitch_results(results, batches, request.inputs, request.inputs)

    async def compare_similarity(
        self,
        request: Optional[SimilarityRequest] = None,
        *,
        a: Optional[List[str]] = None,
        b: Optional[List[str]] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        fast: Optional[bool] = None,
        flatten: bool = False,
        await_job_result: bool = True,
    ) -> Union[SimilarityResponse, AsyncJob]:
        """Compare similarity between two sets of texts asynchronously.

        Args:
            request: Optional SimilarityRequest object.
            a: First set of texts to compare.
            b: Second set of texts to compare.
            model: Model to use for embeddings.
            dimensions: Number of dimensions for embeddings.
            fast: Use fast mode.
            flatten: Flatten the similarity matrix.
            await_job_result: When False, return AsyncJob instead of waiting.

        Returns:
            SimilarityResponse with similarity scores, or AsyncJob if
            await_job_result=False.
        """
        if request is None:
            if a is None or b is None:
                raise ValueError(
                    "Either 'request' or both 'a' and 'b' must be provided"
                )
            request = SimilarityRequest(
                a=a, b=b, model=model, dimensions=dimensions, fast=fast
            )

        validate_before_request(request)

        body = request.model_dump(exclude_none=True, by_alias=True)
        body["flatten"] = flatten

        response = await self._request("post", "/similarity", json=body)

        if response.status_code not in (200, 202):
            raise PulseAPIError(response)

        data = response.json()

        if response.status_code == 202:
            submission = JobSubmissionResponse.model_validate(data)
            job = AsyncJob(jobId=submission.jobId, jobStatus="pending")
            job._client = self.client
            if not await_job_result:
                return job
            result = await job.wait()
            return SimilarityResponse.model_validate(result)

        return SimilarityResponse.model_validate(data)

    async def generate_themes(
        self,
        texts: list[str],
        *,
        num_themes: Optional[int] = None,
        theme_names: Optional[list[str]] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        fast: Optional[bool] = None,
        await_job_result: bool = True,
    ) -> Union[ThemesResponse, ThemeSetsResponse, AsyncJob]:
        """Generate themes from input texts asynchronously.

        Args:
            texts: Input texts to analyze for themes.
            num_themes: Number of themes to generate.
            theme_names: Predefined theme names.
            model: Model to use for analysis.
            dimensions: Number of dimensions for embeddings.
            fast: Use fast mode.
            await_job_result: When False, return AsyncJob instead of waiting.

        Returns:
            ThemesResponse or ThemeSetsResponse, or AsyncJob if await_job_result=False.
        """
        if len(texts) > MAX_THEMES:
            raise PulseValidationError(
                f"Too many texts for theme generation: {len(texts)} > {MAX_THEMES}"
            )

        body: Dict[str, Any] = {"inputs": texts}
        if num_themes is not None:
            body["numThemes"] = num_themes
        if theme_names is not None:
            body["themeNames"] = theme_names
        if model is not None:
            body["model"] = model
        if dimensions is not None:
            body["dimensions"] = dimensions
        if fast is not None:
            body["fast"] = fast

        response = await self._request("post", "/themes", json=body)

        if response.status_code not in (200, 202):
            raise PulseAPIError(response)

        data = response.json()

        if response.status_code == 202:
            submission = JobSubmissionResponse.model_validate(data)
            job = AsyncJob(jobId=submission.jobId, jobStatus="pending")
            job._client = self.client
            if not await_job_result:
                return job
            result = await job.wait()
            return self._parse_themes_response(result)

        return self._parse_themes_response(data)

    def _parse_themes_response(
        self, response_data: Dict[str, Any]
    ) -> Union["ThemesResponse", "ThemeSetsResponse"]:
        """Parse themes response data into appropriate response type."""
        if "themeSets" in response_data:
            return ThemeSetsResponse.model_validate(response_data)
        else:
            return ThemesResponse.model_validate(response_data)

    async def analyze_sentiment(
        self,
        texts: list[str],
        *,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        fast: Optional[bool] = None,
        await_job_result: bool = True,
    ) -> Union[SentimentResponse, AsyncJob]:
        """Analyze sentiment of input texts asynchronously.

        Args:
            texts: Input texts to analyze.
            model: Model to use for analysis.
            dimensions: Number of dimensions for embeddings.
            fast: Use fast mode.
            await_job_result: When False, return AsyncJob instead of waiting.

        Returns:
            SentimentResponse with sentiment analysis, or AsyncJob if
            await_job_result=False.
        """
        if len(texts) > MAX_SENTIMENT:
            # Use batching for large requests
            return await self._analyze_sentiment_with_batching(
                texts,
                model=model,
                dimensions=dimensions,
                fast=fast,
                await_job_result=await_job_result,
            )

        body: Dict[str, Any] = {"inputs": texts}
        if model is not None:
            body["model"] = model
        if dimensions is not None:
            body["dimensions"] = dimensions
        if fast is not None:
            body["fast"] = fast

        response = await self._request("post", "/sentiment", json=body)

        if response.status_code not in (200, 202):
            raise PulseAPIError(response)

        data = response.json()

        if response.status_code == 202:
            submission = JobSubmissionResponse.model_validate(data)
            job = AsyncJob(jobId=submission.jobId, jobStatus="pending")
            job._client = self.client
            if not await_job_result:
                return job
            result = await job.wait()
            return SentimentResponse.model_validate(result)

        return SentimentResponse.model_validate(data)

    async def _analyze_sentiment_with_batching(
        self,
        texts: list[str],
        *,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        fast: Optional[bool] = None,
        await_job_result: bool = True,
    ) -> Union[SentimentResponse, AsyncJob]:
        """Analyze sentiment using batching for large text collections."""
        chunks = chunk_texts(texts, MAX_SENTIMENT)

        async def process_chunk(chunk):
            return await self.analyze_sentiment(
                chunk,
                model=model,
                dimensions=dimensions,
                fast=fast,
                await_job_result=True,
            )

        # Process all chunks concurrently
        tasks = [process_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)

        # Combine results
        all_results = []
        total_usage = None

        for result in results:
            all_results.extend(result.results)
            if total_usage is None:
                total_usage = result.usage
            elif result.usage:
                # Add usage metrics if available
                if hasattr(total_usage, "total_tokens") and hasattr(
                    result.usage, "total_tokens"
                ):
                    total_usage.total_tokens += result.usage.total_tokens

        return SentimentResponse(results=all_results, usage=total_usage, requestId=None)

    async def cluster_texts(
        self,
        inputs: list[str],
        *,
        algorithm: Optional[str] = None,
        num_clusters: Optional[int] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        fast: Optional[bool] = None,
        await_job_result: bool = True,
    ) -> Union[ClusteringResponse, AsyncJob]:
        """Cluster input texts asynchronously.

        Args:
            inputs: Input texts to cluster.
            algorithm: Clustering algorithm to use.
            num_clusters: Number of clusters to create.
            model: Model to use for embeddings.
            dimensions: Number of dimensions for embeddings.
            fast: Use fast mode.
            await_job_result: When False, return AsyncJob instead of waiting.

        Returns:
            ClusteringResponse with clustering results, or AsyncJob if
            await_job_result=False.
        """
        body: Dict[str, Any] = {"inputs": inputs}
        if algorithm is not None:
            body["algorithm"] = algorithm
        if num_clusters is not None:
            body["numClusters"] = num_clusters
        if model is not None:
            body["model"] = model
        if dimensions is not None:
            body["dimensions"] = dimensions
        if fast is not None:
            body["fast"] = fast

        response = await self._request("post", "/clustering", json=body)

        if response.status_code not in (200, 202):
            raise PulseAPIError(response)

        data = response.json()

        if response.status_code == 202:
            submission = JobSubmissionResponse.model_validate(data)
            job = AsyncJob(jobId=submission.jobId, jobStatus="pending")
            job._client = self.client
            if not await_job_result:
                return job
            result = await job.wait()
            return ClusteringResponse.model_validate(result)

        return ClusteringResponse.model_validate(data)

    async def extract_elements(
        self,
        inputs: list[str],
        elements: list[str],
        *,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        fast: Optional[bool] = None,
        await_job_result: bool = True,
    ) -> Union[ExtractionsResponse, AsyncJob]:
        """Extract specific elements from input texts asynchronously.

        Args:
            inputs: Input texts to analyze.
            elements: Elements to extract from the texts.
            model: Model to use for analysis.
            dimensions: Number of dimensions for embeddings.
            fast: Use fast mode.
            await_job_result: When False, return AsyncJob instead of waiting.

        Returns:
            ExtractionsResponse with extraction results, or AsyncJob if
            await_job_result=False.
        """
        body: Dict[str, Any] = {"inputs": inputs, "elements": elements}
        if model is not None:
            body["model"] = model
        if dimensions is not None:
            body["dimensions"] = dimensions
        if fast is not None:
            body["fast"] = fast

        response = await self._request("post", "/extractions", json=body)

        if response.status_code not in (200, 202):
            raise PulseAPIError(response)

        data = response.json()

        if response.status_code == 202:
            submission = JobSubmissionResponse.model_validate(data)
            job = AsyncJob(jobId=submission.jobId, jobStatus="pending")
            job._client = self.client
            if not await_job_result:
                return job
            result = await job.wait()
            return ExtractionsResponse.model_validate(result)

        return ExtractionsResponse.model_validate(data)

    async def generate_summary(
        self,
        inputs: list[str],
        question: str,
        *,
        length: str | None = None,
        preset: str | None = None,
        fast: bool | None = None,
        await_job_result: bool = True,
    ) -> Union[SummariesResponse, AsyncJob]:
        """Summarize text according to a question asynchronously.

        Args:
            inputs: Input strings to summarize.
            question: Prompt describing the desired summary focus.
            length: Optional length specifier
                (``bullet-points``, ``short``, ``medium``, ``long``).
            preset: Optional preset controlling style.
            fast: Use synchronous (True) or asynchronous (False) mode.
            await_job_result: When False, return an AsyncJob handle
                instead of waiting.

        Returns:
            SummariesResponse with summary, or AsyncJob if await_job_result=False.
        """
        body: Dict[str, Any] = {"inputs": inputs, "question": question}
        if length is not None:
            body["length"] = length
        if preset is not None:
            body["preset"] = preset
        if fast is not None:
            body["fast"] = fast

        response = await self._request("post", "/summaries", json=body)

        if response.status_code not in (200, 202):
            raise PulseAPIError(response)

        data = response.json()

        if response.status_code == 202:
            submission = JobSubmissionResponse.model_validate(data)
            job = AsyncJob(jobId=submission.jobId, jobStatus="pending")
            job._client = self.client
            if not await_job_result:
                return job
            result = await job.wait()
            return SummariesResponse.model_validate(result)

        return SummariesResponse.model_validate(data)

    async def estimate_usage(
        self,
        feature: str,
        *,
        inputs: Optional[list[str]] = None,
        num_inputs: Optional[int] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> UsageEstimateResponse:
        """Estimate usage for a given feature and inputs asynchronously.

        Args:
            feature: Feature to estimate usage for.
            inputs: Input texts (optional if num_inputs provided).
            num_inputs: Number of inputs (optional if inputs provided).
            model: Model to use for estimation.
            dimensions: Number of dimensions for embeddings.

        Returns:
            UsageEstimateResponse with usage estimates.
        """
        body: Dict[str, Any] = {"feature": feature}
        if inputs is not None:
            body["inputs"] = inputs
        if num_inputs is not None:
            body["numInputs"] = num_inputs
        if model is not None:
            body["model"] = model
        if dimensions is not None:
            body["dimensions"] = dimensions

        response = await self._request("post", "/usage/estimate", json=body)

        if response.status_code != 200:
            raise PulseAPIError(response)

        data = response.json()
        return UsageEstimateResponse.model_validate(data)

    async def get_job_status(self, job_id: str) -> AsyncJob:
        """Retrieve the status of a previously submitted job asynchronously.

        Args:
            job_id: ID of the job to check.

        Returns:
            AsyncJob instance with current status.
        """
        response = await self._request("get", f"/jobs?jobId={job_id}")

        if response.status_code != 200:
            raise PulseAPIError(response)

        data = response.json()
        if "jobId" not in data:
            data["jobId"] = job_id

        job = AsyncJob.model_validate(data)
        job._client = self.client
        return job

    def debug_auth_status(self):
        """Inspect and return authentication status for debugging."""
        from pulse.debug import inspect_token

        if hasattr(self.client, "auth") and self.client.auth:
            return inspect_token(self.client.auth)
        else:
            from pulse.debug import TokenInfo

            return TokenInfo(has_token=False)
