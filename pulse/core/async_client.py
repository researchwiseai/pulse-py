"""AsyncCoreClient for interacting with the Pulse API asynchronously."""

import asyncio
from typing import Any, Dict, List, Union, Optional, cast
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

from pulse.auth import ClientCredentialsAuth, AuthorizationCodePKCEAuth
from pulse.async_auth import (
    AsyncClientCredentialsAuth,
    AsyncAuthorizationCodePKCEAuth,
    async_auto_auth,
)
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
    SentimentResult,
    JobSubmissionResponse,
    ExtractionsResponse,
    ClusteringResponse,
    SummariesResponse,
    UsageEstimateResponse,
)
from pulse.core.exceptions import PulseAPIError
from pulse.core.async_error_handling import (
    AsyncCancellationError,
    handle_async_http_errors,
)
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
        enable_batch_optimizations: bool = True,
        max_concurrent_jobs: int = 5,
    ) -> None:
        """Initialize AsyncCoreClient with enhanced connection pooling and batch optimizations.

        Args:
            base_url: Base URL for the Pulse API
            timeout: Default timeout for requests
            client: Optional pre-configured HTTPX async client (for testing)
            auth: Optional authentication
            enable_batch_optimizations: Whether to enable connection pool optimizations for batching
            max_concurrent_jobs: Expected maximum concurrent jobs for connection pool sizing
        """
        self.base_url = base_url
        self.timeout = timeout
        self.enable_batch_optimizations = enable_batch_optimizations
        self.max_concurrent_jobs = max_concurrent_jobs

        if client is not None:
            # Use provided HTTP client (user is responsible for auth)
            self.client = client
        else:
            # Create an optimized AsyncGzipClient for batch operations
            if enable_batch_optimizations:
                self.client = AsyncGzipClient.create_for_batch_operations(
                    base_url=self.base_url,
                    concurrent_jobs=max_concurrent_jobs,
                    auth=auth or async_auto_auth(),
                    timeout=self.timeout,
                )
            else:
                # Use standard AsyncGzipClient for backward compatibility
                self.client = AsyncGzipClient(
                    base_url=self.base_url,
                    timeout=self.timeout,
                    auth=auth or async_auto_auth(),
                )

    async def __aenter__(self) -> "AsyncCoreClient":
        """Async context manager entry."""
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with proper resource cleanup."""
        await self.client.__aexit__(exc_type, exc_val, exc_tb)

    @handle_async_http_errors
    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make an async HTTP request with enhanced retry logic, debug support, and error handling."""
        from pulse.core.async_retry import (
            async_retry_request_enhanced,
            AsyncRetryConfig,
        )

        with debug_request(method, url, **kwargs) as timing:
            try:
                # Use enhanced retry logic for better batch processing support
                if self.enable_batch_optimizations:
                    retry_config = AsyncRetryConfig(
                        max_attempts=4,  # More attempts for batch operations
                        base_backoff=0.5,
                        max_backoff=30.0,
                        exponential_base=2.0,
                        jitter=True,
                        jitter_range=0.1,
                    )
                    response = await async_retry_request_enhanced(
                        lambda: self.client.request(method, url, **kwargs),
                        config=retry_config,
                    )
                else:
                    # Use legacy retry for backward compatibility
                    response = await async_retry_request(
                        lambda: self.client.request(method, url, **kwargs)
                    )

                timing.status_code = response.status_code
                _log_response(response)
                return response
            except asyncio.CancelledError as e:
                log_request_failure(e)
                raise AsyncCancellationError(
                    "HTTP request was cancelled",
                    operation=f"{method.upper()}_{url}",
                    context={"method": method, "url": url},
                ) from e
            except Exception as e:
                log_request_failure(e)
                raise

    @classmethod
    def with_client_credentials(
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
            audience: The audience for the token request. Defaults to None.
            token_url: The token endpoint URL.
            base_url: The base URL for the Pulse API.
            organization: The organization ID for the client.

        Returns:
            An instance of AsyncCoreClient configured with Client Credentials auth.

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
        # nosec B105 - Auth0 URL, not a password
        default_token_url = "https://research-wise-ai-eu.eu.auth0.com/oauth/token"
        final_token_url = token_url or os.getenv("PULSE_TOKEN_URL") or default_token_url

        # Resolve audience: argument > environment variable (default is None if not set)
        final_audience = audience or os.getenv("PULSE_AUDIENCE")

        # Resolve base_url: argument > environment variable > default (BASE_URL)
        # BASE_URL should be imported from pulse.config at the module level.
        final_base_url = base_url or os.getenv("PULSE_BASE_URL") or BASE_URL

        auth = ClientCredentialsAuth(
            token_url=final_token_url,
            client_id=final_client_id,
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
    ) -> "AsyncCoreClient":
        """
        Construct an AsyncCoreClient using OAuth2 Authorization Code flow with PKCE.

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
           - base_url defaults to BASE_URL (from pulse.config).
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
            An instance of AsyncCoreClient configured with PKCE authentication.

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

        # Resolve base_url: argument > environment variable > default (BASE_URL)
        # BASE_URL should be imported from pulse.config at the module level.
        final_base_url = base_url or os.getenv("PULSE_BASE_URL") or BASE_URL

        # Resolve token_url: argument > environment variable > default
        # nosec B105 - Auth0 URL, not a password
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

    @classmethod
    def with_client_credentials_async(
        cls,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        audience: Optional[str] = None,
        token_url: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> "AsyncCoreClient":
        """
        Construct an AsyncCoreClient using async OAuth2 Client Credentials flow.

        This method creates an AsyncCoreClient with async-compatible authentication
        that properly handles token refresh in async contexts.

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
            audience: The audience for the token request. Defaults to None.
            token_url: The token endpoint URL.
            base_url: The base URL for the Pulse API.
            organization: The organization ID for the client.

        Returns:
            AsyncCoreClient configured with async Client Credentials auth.

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
        # nosec B105 - Auth0 URL, not a password
        default_token_url = "https://research-wise-ai-eu.eu.auth0.com/oauth/token"
        final_token_url = token_url or os.getenv("PULSE_TOKEN_URL") or default_token_url

        # Resolve audience: argument > environment variable (default is None if not set)
        final_audience = audience or os.getenv("PULSE_AUDIENCE")

        # Resolve base_url: argument > environment variable > default (BASE_URL)
        # BASE_URL should be imported from pulse.config at the module level.
        final_base_url = base_url or os.getenv("PULSE_BASE_URL") or BASE_URL

        auth = AsyncClientCredentialsAuth(
            token_url=final_token_url,
            client_id=final_client_id,
            client_secret=final_client_secret,
            audience=final_audience,
        )

        return cls(base_url=final_base_url, auth=auth)

    @classmethod
    def with_pkce_async(
        cls,
        code: str,
        code_verifier: str,
        client_id: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        base_url: Optional[str] = None,
        token_url: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> "AsyncCoreClient":
        """
        Construct AsyncCoreClient using async OAuth2 Authorization Code flow with PKCE.

        This method creates an AsyncCoreClient with async-compatible PKCE authentication
        that properly handles token refresh in async contexts.

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
           - base_url defaults to BASE_URL (from pulse.config).
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
            An instance of AsyncCoreClient configured with async PKCE authentication.

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

        # Resolve base_url: argument > environment variable > default (BASE_URL)
        # BASE_URL should be imported from pulse.config at the module level.
        final_base_url = base_url or os.getenv("PULSE_BASE_URL") or BASE_URL

        # Resolve token_url: argument > environment variable > default
        # nosec B105 - Auth0 URL, not a password
        default_token_url = "https://research-wise-ai-eu.eu.auth0.com/oauth/token"
        final_token_url = token_url or os.getenv("PULSE_TOKEN_URL") or default_token_url

        # Resolve scope: argument > environment variable (default is None if not set)
        final_scope = scope or os.getenv("PULSE_SCOPE")

        auth = AsyncAuthorizationCodePKCEAuth(
            token_url=final_token_url,
            client_id=final_client_id,
            code=code,  # Direct argument
            redirect_uri=final_redirect_uri,
            code_verifier=code_verifier,  # Direct argument
            scope=final_scope,
        )

        return cls(base_url=final_base_url, auth=auth)

    async def close(self) -> None:
        """Close underlying HTTP connection."""
        await self.client.aclose()

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection and performance statistics for batch operations.

        Returns:
            Dictionary containing connection statistics, compression ratios,
            and performance metrics useful for optimizing batch operations.
        """
        stats = {
            "batch_optimizations_enabled": self.enable_batch_optimizations,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "base_url": self.base_url,
            "timeout": self.timeout,
        }

        # Get connection stats from AsyncGzipClient if available
        if hasattr(self.client, "get_connection_stats"):
            client_stats = self.client.get_connection_stats()
            stats.update(client_stats)

        return stats

    def reset_connection_stats(self) -> None:
        """Reset connection statistics."""
        if hasattr(self.client, "reset_stats"):
            self.client.reset_stats()

    @classmethod
    def create_for_high_throughput(
        cls,
        base_url: str = BASE_URL,
        auth: Optional[httpx.Auth] = None,
        max_concurrent_jobs: int = 15,
    ) -> "AsyncCoreClient":
        """Create AsyncCoreClient optimized for high throughput batch operations.

        This factory method creates a client with aggressive connection pooling
        and optimizations for processing large numbers of concurrent requests.

        Args:
            base_url: Base URL for the Pulse API
            auth: Optional authentication
            max_concurrent_jobs: Maximum expected concurrent jobs

        Returns:
            AsyncCoreClient optimized for high throughput operations
        """
        return cls(
            base_url=base_url,
            auth=auth,
            enable_batch_optimizations=True,
            max_concurrent_jobs=max_concurrent_jobs,
            timeout=60.0,  # Longer timeout for high throughput
        )

    async def create_embeddings(
        self, request: EmbeddingsRequest, *, await_job_result: bool = True
    ) -> Union[EmbeddingsResponse, AsyncJob]:
        """Generate dense vector embeddings.

        Args:
            request: EmbeddingsRequest payload.
            await_job_result: When False, return an :class:`AsyncJob` handle instead of
                waiting for the result when the server responds with HTTP 202.
        """

        # Validate request before sending to API
        try:
            validate_before_request(
                "embeddings", inputs=request.inputs, fast=request.fast
            )
        except PulseValidationError as e:
            raise ValueError(str(e)) from e

        # Check if parallel batching is needed for slow mode
        input_count = len(request.inputs)
        fast = bool(request.fast)

        # Parallel batching for slow mode (fast=False) only when input > 200
        if not fast and input_count > 200:
            return await self._create_embeddings_with_parallel_batching(
                request, await_job_result=await_job_result
            )

        # Original single request logic for fast mode or small inputs
        # Request body according to OpenAPI spec: inputs
        body = request.model_dump(exclude_none=True)

        response = await self._request("post", "/embeddings", json=body)

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
            job = AsyncJob(jobId=submission.jobId, jobStatus="pending")
            job._client = self.client
            if not await_job_result:
                return job
            result = await job.wait()
            return EmbeddingsResponse.model_validate(result)
        # Synchronous response
        return EmbeddingsResponse.model_validate(data)

    async def _create_embeddings_with_parallel_batching(
        self, request: EmbeddingsRequest, *, await_job_result: bool = True
    ) -> Union[EmbeddingsResponse, AsyncJob]:
        """Create embeddings with enhanced parallel batching for slow mode (fast=False) only.

        This enhanced method implements intelligent batching for large embedding requests:
        - Adaptive batch sizing based on input count and system resources
        - Dynamic concurrency control with rate limiting
        - Enhanced retry logic with exponential backoff
        - Connection pool optimization for batch operations
        - Preserves input order in results
        - Aggregates usage metrics across all batches

        Args:
            request: EmbeddingsRequest payload with >200 inputs and fast=False
            await_job_result: When False, return an AsyncJob handle for the first batch

        Returns:
            EmbeddingsResponse with aggregated results or AsyncJob handle
        """
        from pulse.core.validation import ValidationLimits
        from pulse.core.models import UsageReport
        from pulse.core.async_concurrent import AsyncConcurrentConfig, AsyncJobManager

        inputs = request.inputs
        input_count = len(inputs)

        # Adaptive batch sizing based on input count
        if input_count <= 1000:
            batch_size = 500  # Smaller batches for smaller datasets
            max_concurrent = 3
        elif input_count <= 5000:
            batch_size = 1000  # Medium batches
            max_concurrent = 5
        elif input_count <= 20000:
            batch_size = 2500  # Larger batches for efficiency
            max_concurrent = 8
        else:
            batch_size = ValidationLimits.EMBEDDINGS_BATCH_SIZE  # 5,000
            max_concurrent = 10

        # Create batches with adaptive sizing
        batches = [
            inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)
        ]

        # Configure concurrent processing with enhanced settings
        config = AsyncConcurrentConfig(
            max_concurrent_jobs=min(max_concurrent, len(batches)),
            timeout_per_job=max(
                300.0, len(inputs) / 100.0
            ),  # Scale timeout with input size
            rate_limit_delay=0.05,  # Reduced delay for better throughput
            max_retries=3,
            retry_delay=1.0,
            connection_pool_limits={
                "max_keepalive_connections": max_concurrent * 2,
                "max_connections": max_concurrent * 4,
                "keepalive_expiry": 120.0,
            },
        )

        # Create enhanced job submitters with retry logic
        async def create_enhanced_batch_submitter(batch_inputs, batch_index):
            """Create a batch submitter with enhanced error handling and retry logic."""
            batch_request = EmbeddingsRequest(
                inputs=batch_inputs,
                fast=False,  # Always use slow mode for batching
                version=request.version,
            )

            # Enhanced retry logic for batch submission
            max_submission_retries = 3
            for attempt in range(max_submission_retries):
                try:
                    # Submit batch job without waiting for result, bypassing batching logic
                    return await self._create_single_embeddings_request(
                        batch_request, await_job_result=False
                    )
                except Exception as e:
                    if attempt == max_submission_retries - 1:
                        raise RuntimeError(
                            f"Failed to submit batch {batch_index + 1} after "
                            f"{max_submission_retries} attempts: {e}"
                        ) from e

                    # Exponential backoff for retry
                    delay = 2.0**attempt
                    await asyncio.sleep(delay)

        job_submitters = [
            lambda batch=batch, idx=i: create_enhanced_batch_submitter(batch, idx)
            for i, batch in enumerate(batches)
        ]

        # If not waiting for results, return the first job
        if not await_job_result:
            # Submit first job and return it
            first_job = await job_submitters[0]()
            return first_job

        # Use enhanced concurrent processing
        job_manager = AsyncJobManager(config)

        try:
            batch_results = await job_manager.submit_and_gather(
                job_submitters,
                timeout=config.timeout_per_job,
                return_exceptions=False,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to process {len(batches)} embedding batches: {e}"
            ) from e

        # Aggregate results while preserving order with enhanced error handling
        all_embeddings = []
        total_usage_records = []
        total_quantity = 0
        successful_batches = 0

        for i, batch_result in enumerate(batch_results):
            try:
                # Parse batch result if it's a dict (from job.wait())
                if isinstance(batch_result, dict):
                    batch_response = EmbeddingsResponse.model_validate(batch_result)
                else:
                    batch_response = batch_result

                # Add embeddings in order
                all_embeddings.extend(batch_response.embeddings)
                successful_batches += 1

                # Aggregate usage metrics
                if batch_response.usage:
                    total_usage_records.extend(batch_response.usage.records)
                    total_quantity += batch_response.usage.total

            except Exception as e:
                raise RuntimeError(
                    f"Failed to process results from batch {i + 1}: {e}"
                ) from e

        # Validate that we got results for all inputs
        if len(all_embeddings) != input_count:
            raise RuntimeError(
                f"Embedding count mismatch: expected {input_count}, "
                f"got {len(all_embeddings)} from {successful_batches} batches"
            )

        # Create aggregated usage info
        aggregated_usage = UsageReport(
            records=total_usage_records, total=total_quantity
        )

        # Create final response with aggregated results
        return EmbeddingsResponse(
            embeddings=all_embeddings,
            usage=aggregated_usage,
            request_id=(
                batch_results[0].get("request_id")
                if isinstance(batch_results[0], dict)
                else getattr(batch_results[0], "request_id", None)
            ),
        )

    async def _create_single_embeddings_request(
        self, request: EmbeddingsRequest, *, await_job_result: bool = True
    ) -> Union[EmbeddingsResponse, AsyncJob]:
        """Create embeddings for a single request without batching logic.

        This method contains the original embeddings logic and is used by the
        parallel batching system to avoid infinite recursion.

        Args:
            request: EmbeddingsRequest payload
            await_job_result: When False, return an AsyncJob handle instead of waiting

        Returns:
            EmbeddingsResponse or AsyncJob handle
        """
        # Request body according to OpenAPI spec: inputs
        body = request.model_dump(exclude_none=True)
        fast = bool(request.fast)

        response = await self._request("post", "/embeddings", json=body)

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
            job = AsyncJob(jobId=submission.jobId, jobStatus="pending")
            job._client = self.client
            if not await_job_result:
                return job
            result = await job.wait()
            return EmbeddingsResponse.model_validate(result)
        # Synchronous response
        return EmbeddingsResponse.model_validate(data)

    async def _process_batches_concurrently(
        self, job_submitters, max_workers: int = 5, timeout: float = 600.0
    ):
        """Process batches concurrently with enhanced concurrency control and retry logic."""
        from pulse.core.async_concurrent import (
            AsyncConcurrentConfig,
            AsyncJobManager,
        )

        # Create enhanced configuration for batch processing
        config = AsyncConcurrentConfig(
            max_concurrent_jobs=max_workers,
            timeout_per_job=timeout,
            rate_limit_delay=0.03,  # Optimized delay for better throughput
            max_retries=3,
            retry_delay=1.0,
            connection_pool_limits={
                "max_keepalive_connections": max_workers * 2,
                "max_connections": max_workers * 4,
                "keepalive_expiry": 120.0,
            },
        )

        # Use enhanced job manager for better error handling and resource management
        job_manager = AsyncJobManager(config)

        try:
            return await job_manager.submit_and_gather(
                job_submitters,
                timeout=timeout,
                return_exceptions=False,
            )
        except Exception as e:
            # Enhanced error reporting with context
            raise RuntimeError(
                f"Failed to process {len(job_submitters)} batches concurrently "
                f"with {max_workers} workers: {e}"
            ) from e

    async def compare_similarity(
        self,
        request: Optional[SimilarityRequest] = None,
        *,
        set: Optional[List[str]] = None,
        set_a: Optional[List[str]] = None,
        set_b: Optional[List[str]] = None,
        fast: Optional[bool] = None,
        flatten: bool = False,
        version: Optional[str] = None,
        split: Optional[Dict[str, Any]] = None,
        await_job_result: bool = True,
    ) -> Union[SimilarityResponse, AsyncJob]:
        """Compute cosine similarity between strings.

        Args:
            request: SimilarityRequest object (deprecated, use individual parameters).
            set: Input texts for self-similarity computation.
            set_a: First set of texts for cross-similarity computation.
            set_b: Second set of texts for cross-similarity computation.
            fast: Use synchronous (True) or asynchronous (False) mode.
            flatten: Return flattened results instead of matrix format.
            version: Optional model version for reproducible output.
            split: Text splitting configuration for fine-grained analysis.
            await_job_result: When False, return an AsyncJob handle instead of waiting.

        Note:
            Exactly one of `set` (self-similarity) or the pair `set_a`/`set_b`
            (cross-similarity) must be provided. Optional `version` and `split`
            values are forwarded to the API.
        """
        # Handle backward compatibility with SimilarityRequest
        if request is not None:
            set = request.set
            set_a = request.set_a
            set_b = request.set_b
            fast = request.fast
            flatten = request.flatten
            version = request.version
            split = (
                request.split.model_dump(exclude_none=True) if request.split else None
            )

        # Validate request before sending to API
        try:
            validate_before_request(
                "similarity", set=set, set_a=set_a, set_b=set_b, fast=fast
            )
        except PulseValidationError as e:
            raise ValueError(str(e)) from e

        body: Dict[str, Any] = {}
        oversized = False

        if set is not None:
            body["set"] = set
            # Use new limits for oversized detection
            if len(set) > 500:
                oversized = True
        else:
            body["set_a"] = set_a
            body["set_b"] = set_b
            if len(cast(List[str], set_a)) * len(cast(List[str], set_b)) > 20_000:
                oversized = True

        if oversized and fast is not False:
            # If not explicitly async and oversized, use batch similarity
            return await self.batch_similarity(
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

        if fast is not None:
            body["fast"] = fast

        response = await self._request("post", "/similarity", json=body)

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
            job = AsyncJob(jobId=submission.jobId, jobStatus="pending")
            job._client = self.client
            if not await_job_result:
                return job
            result = await job.wait(600)
            return SimilarityResponse.model_validate(result)

        # sync path
        return SimilarityResponse.model_validate(data)

    async def _submit_batch_similarity_job(self, **kwargs) -> Any:
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

        response = await self._request("post", "/similarity", json=body)

        if response.status_code != 202:
            raise PulseAPIError(response)
        data = response.json()
        # Async/job path: initial submission returned only jobId
        submission = JobSubmissionResponse.model_validate(data)
        job = AsyncJob(jobId=submission.jobId, jobStatus="pending")
        job._client = self.client

        return job

    async def batch_similarity(
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
        Batch large similarity requests with enhanced concurrent job processing.

        This enhanced method provides:
        - Intelligent batching under the 10k-item limit
        - Concurrent job processing with rate limiting
        - Enhanced retry logic with exponential backoff
        - Connection pool optimization for batch operations
        - Adaptive timeout scaling based on batch complexity

        Note: This method requires numpy for matrix operations. Install with:
        pip install pulse-sdk[analysis]
        """
        if not HAS_BATCHING:
            raise ImportError(
                "Batch similarity requires numpy for matrix operations. "
                "Install with: pip install pulse-sdk[analysis]"
            )

        from pulse.core.async_concurrent import AsyncConcurrentConfig, AsyncJobManager

        # Determine batch complexity for adaptive configuration
        if set is not None:
            total_items = len(set)
            batch_complexity = total_items * total_items  # Self-similarity complexity
            full_a = full_b = set
        else:
            set_a = set_a or []
            set_b = set_b or []
            total_items = len(set_a) + len(set_b)
            batch_complexity = len(set_a) * len(set_b)  # Cross-similarity complexity
            full_a = set_a
            full_b = set_b

        # Create request bodies using existing batching logic
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

        # Configure concurrent processing based on batch complexity
        if batch_complexity <= 10000:
            # Simple batches - use basic concurrency
            max_concurrent = min(3, len(bodies))
            timeout_per_job = 300.0
            rate_limit_delay = 0.1
        elif batch_complexity <= 100000:
            # Medium complexity - moderate concurrency
            max_concurrent = min(5, len(bodies))
            timeout_per_job = 450.0
            rate_limit_delay = 0.05
        else:
            # High complexity - aggressive concurrency with careful rate limiting
            max_concurrent = min(8, len(bodies))
            timeout_per_job = 600.0
            rate_limit_delay = 0.02

        config = AsyncConcurrentConfig(
            max_concurrent_jobs=max_concurrent,
            timeout_per_job=timeout_per_job,
            rate_limit_delay=rate_limit_delay,
            max_retries=3,
            retry_delay=1.0,
            connection_pool_limits={
                "max_keepalive_connections": max_concurrent * 2,
                "max_connections": max_concurrent * 4,
                "keepalive_expiry": 180.0,
            },
        )

        # Create enhanced job submitters with retry logic
        async def create_enhanced_similarity_submitter(body, job_index):
            """Create a similarity job submitter with enhanced error handling."""
            max_submission_retries = 3
            for attempt in range(max_submission_retries):
                try:
                    return await self._submit_batch_similarity_job(**body)
                except Exception as e:
                    if attempt == max_submission_retries - 1:
                        raise RuntimeError(
                            f"Failed to submit similarity job {job_index + 1} after "
                            f"{max_submission_retries} attempts: {e}"
                        ) from e

                    # Exponential backoff for retry
                    delay = 1.0 * (2.0**attempt)
                    await asyncio.sleep(delay)

        job_submitters = [
            lambda body=body, idx=i: create_enhanced_similarity_submitter(body, idx)
            for i, body in enumerate(bodies)
        ]

        # Use enhanced concurrent processing
        job_manager = AsyncJobManager(config)

        try:
            # Submit jobs with rate limiting and gather results
            jobs = await job_manager.submit_jobs_with_rate_limit(
                job_submitters, rate_limit_delay=config.rate_limit_delay
            )

            # Wait for all jobs with enhanced timeout and error handling
            results = await job_manager.gather_jobs(
                jobs, timeout=config.timeout_per_job, return_exceptions=False
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to process {len(bodies)} similarity batches "
                f"(complexity: {batch_complexity:,}): {e}"
            ) from e

        # Validate results before stitching
        if len(results) != len(bodies):
            raise RuntimeError(
                f"Result count mismatch: expected {len(bodies)} results, "
                f"got {len(results)}"
            )

        # Stitch results using existing logic with enhanced error handling
        try:
            return _stitch_results(results, bodies, full_a, full_b)
        except Exception as e:
            raise RuntimeError(
                f"Failed to stitch similarity results from {len(results)} batches: {e}"
            ) from e

    async def generate_themes(
        self,
        texts: list[str],
        min_themes: Optional[int] = None,
        max_themes: Optional[int] = None,
        fast: bool = False,
        *,
        context: Optional[str] = None,
        version: Optional[str] = None,
        prune: Optional[int] = None,
        interactive: Optional[bool] = None,
        initial_sets: Optional[int] = None,
        await_job_result: bool = True,
    ) -> Union["ThemesResponse", "ThemeSetsResponse", AsyncJob]:
        """Cluster texts into latent themes.

        Args:
            texts: Input strings to cluster.
            min_themes: Minimum number of themes.
            max_themes: Maximum number of themes.
            fast: Use synchronous (True) or asynchronous (False) mode.
            context: Optional context string guiding theme generation.
            version: Optional model version for reproducible output.
                When version="2025-09-01", returns ThemeSetsResponse.
            prune: Optionally prune the specified number of
                lowest-frequency themes.
            interactive: Enable interactive theme generation mode.
            initial_sets: Number of initial theme sets
                (requires interactive=True if > 1).
            await_job_result: When False, return an :class:`AsyncJob` handle
                instead of waiting.
        """
        # Import here to avoid circular imports
        from pulse.core.models import ThemesResponse

        # Validate request before sending to API
        try:
            validate_before_request(
                "themes",
                inputs=texts,
                fast=fast,
                initialSets=initial_sets,
                interactive=interactive,
            )
        except PulseValidationError as e:
            raise ValueError(str(e)) from e

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
        if interactive is not None:
            body["interactive"] = interactive
        if initial_sets is not None:
            body["initialSets"] = initial_sets
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

        response = await self._request("post", "/themes", json=body)
        if response.status_code not in (200, 202):
            raise PulseAPIError(response)
        data = response.json()
        # If server enqueues an async job (202), handle via job flow regardless
        # of the fast flag; callers control waiting via await_job_result.
        if response.status_code == 202:
            # Async/job path: initial submission returned only jobId
            submission = JobSubmissionResponse.model_validate(data)
            job = AsyncJob(jobId=submission.jobId, jobStatus="pending")
            job._client = self.client
            if not await_job_result:
                return job
            result = await job.wait()
            return self._parse_themes_response(result)
        # Synchronous response
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
        version: str | None = None,
        fast: bool = False,
        await_job_result: bool = True,
    ) -> Union[SentimentResponse, AsyncJob]:
        """Classify sentiment for the given texts.

        Args:
            texts: List of input strings to analyze.
            version: Optional model version for reproducible output.
            fast: Use synchronous (True) or asynchronous (False) mode.
            await_job_result: When False, return an :class:`AsyncJob` handle
                instead of waiting.
        """
        try:
            validate_before_request("sentiment", inputs=texts, fast=fast)
        except PulseValidationError as e:
            raise ValueError(str(e)) from e

        # For small requests or fast mode, use direct API call
        fast_limit = 200
        slow_batch_size = 2000

        if fast and len(texts) > fast_limit:
            # This should be caught by validation, but double-check
            from pulse.core.validation import BatchingErrorHelper

            raise ValueError(
                str(
                    BatchingErrorHelper.create_fast_mode_error(
                        "sentiment", len(texts), fast_limit
                    )
                )
            )

        if len(texts) <= fast_limit or fast:
            # Direct API call for small requests or fast mode
            body: Dict[str, Any] = {"inputs": texts}
            if version is not None:
                body["version"] = version
            if fast:
                body["fast"] = True

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

        # Large request in slow mode - use parallel batching
        # Create batches for parallel processing
        chunks = chunk_texts(texts, slow_batch_size)

        # Create request bodies for each batch
        request_bodies = []
        for chunk in chunks:
            body = {"inputs": chunk}
            if version is not None:
                body["version"] = version
            body["fast"] = False  # Always use slow mode for batching
            request_bodies.append(body)

        # Create job submitters for batch processing
        async def create_batch_job_submitter(body):
            return await self._submit_sentiment_batch_job(**body)

        job_submitters = [create_batch_job_submitter(body) for body in request_bodies]

        # Process batches with controlled concurrency (5 concurrent jobs max)
        # Only use parallel processing for slow mode (fast=False)
        try:
            batch_results = await self._process_batches_concurrently(
                job_submitters,
                max_workers=5,
                timeout=600.0,
            )
        except Exception as e:
            from pulse.core.validation import BatchingErrorHelper

            raise ValueError(
                str(
                    BatchingErrorHelper.create_batch_failure_error(
                        "sentiment", 0, len(chunks), str(e), len(texts)
                    )
                )
            ) from e

        # Aggregate results from all batches while preserving order
        all_results: List[SentimentResult] = []
        total_usage = None

        for i, batch_result in enumerate(batch_results):
            if not isinstance(batch_result, dict):
                continue

            batch_response = SentimentResponse.model_validate(batch_result)
            all_results.extend(batch_response.results)

            # Aggregate usage metrics
            if batch_response.usage and total_usage is None:
                total_usage = batch_response.usage.model_copy()
            elif batch_response.usage and total_usage:
                # Sum up the usage metrics
                total_usage.total += batch_response.usage.total
                total_usage.records.extend(batch_response.usage.records)

        return SentimentResponse(results=all_results, usage=total_usage, requestId=None)

    async def _submit_sentiment_batch_job(self, **kwargs) -> AsyncJob:
        """Submit a sentiment analysis batch job for parallel processing.

        Args:
            **kwargs: Request body parameters including inputs, version, fast

        Returns:
            AsyncJob instance for the submitted batch
        """
        response = await self._request("post", "/sentiment", json=kwargs)
        if response.status_code != 202:
            raise PulseAPIError(response)

        data = response.json()
        submission = JobSubmissionResponse.model_validate(data)
        job = AsyncJob(jobId=submission.jobId, jobStatus="pending")
        job._client = self.client
        return job

    async def cluster_texts(
        self,
        inputs: list[str],
        *,
        k: int,
        algorithm: str = "kmeans",
        fast: Optional[bool] = None,
        await_job_result: bool = True,
    ) -> Union[ClusteringResponse, AsyncJob]:
        """Cluster texts into groups using embeddings with intelligent batching.

        Automatically triggers batching when input > 500 texts, similar to
        self-similarity matrix processing. Uses parallel processing for
        async operations (fast=False) only.

        Args:
            inputs: Input strings to cluster.
            k: Desired number of clusters.
            algorithm: Clustering algorithm. Options: "kmeans", "skmeans",
                "agglomerative", "hdbscan". Defaults to "kmeans".
            fast: Use synchronous (True) or asynchronous (False) mode.
            await_job_result: When False, return an :class:`AsyncJob` handle
                instead of waiting.
        """
        # Validate algorithm
        valid_algorithms = ["kmeans", "skmeans", "agglomerative", "hdbscan"]
        if algorithm not in valid_algorithms:
            raise ValueError(
                f"algorithm must be one of {valid_algorithms}, got: {algorithm}"
            )

        # Validate request before sending to API
        try:
            validate_before_request("clustering", inputs=inputs, fast=fast)
        except PulseValidationError as e:
            raise ValueError(str(e)) from e

        # Check if batching is needed (>500 texts triggers auto-batching)
        from pulse.core.validation import ValidationLimits

        if len(inputs) > ValidationLimits.CLUSTERING_AUTO_BATCH_THRESHOLD:
            # Use intelligent batching similar to similarity analysis
            if HAS_BATCHING:
                return await self._cluster_with_batching(
                    inputs,
                    k=k,
                    algorithm=algorithm,
                    fast=fast,
                    await_job_result=await_job_result,
                )
            else:
                # Fallback to chunking if batching module not available
                return await self._cluster_with_chunking(
                    inputs,
                    k=k,
                    algorithm=algorithm,
                    fast=fast,
                    await_job_result=await_job_result,
                )

        # Direct clustering for smaller inputs
        body: Dict[str, Any] = {"inputs": inputs, "k": k, "algorithm": algorithm}
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

    async def _cluster_with_batching(
        self,
        inputs: list[str],
        *,
        k: int,
        algorithm: str = "kmeans",
        fast: Optional[bool] = None,
        await_job_result: bool = True,
    ) -> Union[ClusteringResponse, AsyncJob]:
        """Cluster texts using intelligent batching with parallel processing.

        This method implements parallel clustering with result reconstruction
        for async operations only (fast=False).
        """
        # Create clustering batch logic with clustering-specific chunking
        from pulse.core.batching import _make_clustering_chunks

        chunks = _make_clustering_chunks(inputs)

        # Create request bodies for each chunk
        request_bodies = []
        for chunk in chunks:
            body = {"inputs": chunk, "k": k, "algorithm": algorithm}
            if fast is not None:
                body["fast"] = fast
            request_bodies.append(body)

        # Create job submitters for batch processing
        async def create_clustering_job_submitter(body):
            return await self._submit_clustering_job(**body)

        job_submitters = [
            create_clustering_job_submitter(body) for body in request_bodies
        ]

        # Process batches with controlled concurrency (5 concurrent jobs max)
        # Only use parallel processing for slow mode (fast=False)
        results = await self._process_batches_concurrently(
            job_submitters,
            max_workers=5,
            timeout=600.0,
        )

        # Reconstruct clustering results from parallel batches
        return await self._reconstruct_clustering_results(
            results, inputs, k, algorithm, await_job_result
        )

    async def _cluster_with_chunking(
        self,
        inputs: list[str],
        *,
        k: int,
        algorithm: str = "kmeans",
        fast: Optional[bool] = None,
        await_job_result: bool = True,
    ) -> Union[ClusteringResponse, AsyncJob]:
        """Fallback clustering using simple chunking when batching unavailable."""
        # Simple chunking approach - process sequentially
        chunk_size = 500  # Use threshold as chunk size
        chunks = [inputs[i : i + chunk_size] for i in range(0, len(inputs), chunk_size)]

        results = []
        for chunk in chunks:
            body: Dict[str, Any] = {"inputs": chunk, "k": k, "algorithm": algorithm}
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
                result = await job.wait()
                results.append(ClusteringResponse.model_validate(result))
            else:
                results.append(ClusteringResponse.model_validate(data))

        # Simple result combination - return first result for now
        # This is a fallback implementation
        if results:
            return results[0]

        # Should not reach here, but return empty response as fallback
        return ClusteringResponse(clusters=[], usage=None)

    async def _submit_clustering_job(self, **kwargs) -> AsyncJob:
        """Submit a clustering job and return AsyncJob handle."""
        response = await self._request("post", "/clustering", json=kwargs)

        if response.status_code not in (200, 202):
            raise PulseAPIError(response)

        data = response.json()

        if response.status_code == 202:
            submission = JobSubmissionResponse.model_validate(data)
            job = AsyncJob(jobId=submission.jobId, jobStatus="pending")
            job._client = self.client
            return job
        else:
            # Immediate result - wrap in completed job
            result = ClusteringResponse.model_validate(data)
            job = AsyncJob(jobId="immediate", jobStatus="completed")
            job._result = result
            job._client = self.client
            return job

    async def _reconstruct_clustering_results(
        self,
        results: List[Any],
        original_inputs: List[str],
        k: int,
        algorithm: str,
        await_job_result: bool = True,
    ) -> Union[ClusteringResponse, AsyncJob]:
        """Reconstruct clustering results from parallel batches.

        This method combines clustering assignments from parallel batches
        to ensure results are identical to non-batched clustering for the same input.
        Uses sophisticated result reconstruction logic for async processing.
        """
        # Wait for all job results if needed
        clustering_responses = []
        for result in results:
            if isinstance(result, AsyncJob):
                if await_job_result:
                    job_result = await result.wait()
                    clustering_responses.append(
                        ClusteringResponse.model_validate(job_result)
                    )
                else:
                    # Return the first job handle for now - this is a simplification
                    return result
            else:
                clustering_responses.append(result)

        # Use the batching module's reconstruction logic
        from pulse.core.batching import _reconstruct_clustering_results

        reconstructed_result = _reconstruct_clustering_results(
            clustering_responses, original_inputs, k, algorithm
        )

        # Convert back to ClusteringResponse
        if isinstance(reconstructed_result, dict):
            return ClusteringResponse(
                clusters=reconstructed_result.get("clusters", []),
                usage=reconstructed_result.get("usage"),
            )
        else:
            return reconstructed_result

    async def extract_elements(
        self,
        inputs: list[str],
        dictionary: list[str],
        *,
        type: str = "named-entities",
        expand_dictionary: bool = False,
        expand_dictionary_limit: Optional[int] = None,
        version: Optional[str] = None,
        fast: Optional[bool] = None,
        await_job_result: bool = True,
        # Deprecated parameters for backward compatibility
        texts: Optional[list[str]] = None,
        categories: Optional[list[str]] = None,
        use_ner: Optional[bool] = None,
        use_llm: Optional[bool] = None,
        threshold: Optional[float] = None,
    ) -> Union[ExtractionsResponse, AsyncJob]:
        """Extract elements matching dictionary terms from input texts.

        Args:
            inputs: Input strings to analyze.
            dictionary: List of terms to extract from texts.
            type: Extraction type. Options: "named-entities", "themes".
                Defaults to "named-entities".
            expand_dictionary: Expand dictionary entries with synonyms.
            expand_dictionary_limit: Limit for dictionary expansions.
            version: Optional model version for reproducible output.
            fast: Use synchronous (True) or asynchronous (False) mode.
            await_job_result: When False, return an AsyncJob handle instead of waiting.

            # Deprecated parameters (for backward compatibility):
            texts: Deprecated, use 'inputs' instead.
            categories: Deprecated, use 'dictionary' instead.
            use_ner: Deprecated parameter (no longer used).
            use_llm: Deprecated parameter (no longer used).
            threshold: Deprecated parameter (no longer used).
        """
        # Handle backward compatibility
        if texts is not None:
            inputs = texts
        if categories is not None:
            dictionary = categories

        # Validate type parameter
        valid_types = ["named-entities", "themes"]
        if type not in valid_types:
            raise ValueError(f"type must be one of {valid_types}, got: {type}")

        # Validate request before sending to API
        try:
            validate_before_request(
                "extractions",
                inputs=inputs,
                fast=fast,
                type=type,
                expand_dictionary=expand_dictionary,
            )
        except PulseValidationError as e:
            raise ValueError(str(e)) from e

        if len(dictionary) < 3:
            raise ValueError("dictionary must contain at least 3 terms")
        if len(dictionary) > 200:
            raise ValueError("dictionary cannot exceed 200 terms")

        # Check if parallel batching is needed for slow mode
        input_count = len(inputs)
        fast_mode = bool(fast)

        # Parallel batching for slow mode (fast=False) only when input > 200
        if not fast_mode and input_count > 200:
            return await self._extract_elements_with_parallel_batching(
                inputs=inputs,
                dictionary=dictionary,
                type=type,
                expand_dictionary=expand_dictionary,
                expand_dictionary_limit=expand_dictionary_limit,
                version=version,
                await_job_result=await_job_result,
            )

        # Original single request logic for fast mode or small inputs
        body: Dict[str, Any] = {
            "inputs": inputs,
            "dictionary": dictionary,
            "type": type,
            "expand_dictionary": expand_dictionary,
        }

        if expand_dictionary_limit is not None:
            body["expand_dictionary_limit"] = expand_dictionary_limit
        if version is not None:
            body["version"] = version
        if fast is not None:
            body["fast"] = fast

        response = await self._request("post", "/extractions", json=body)
        if response.status_code not in (200, 202):
            raise PulseAPIError(response)

        data = response.json()

        # If service enqueues an async job during fast sync, treat as error
        if response.status_code == 202 and fast_mode:
            raise PulseAPIError(response)

        if response.status_code == 202:
            submission = JobSubmissionResponse.model_validate(data)
            job = AsyncJob(jobId=submission.jobId, jobStatus="pending")
            job._client = self.client
            if not await_job_result:
                return job
            result = await job.wait()
            return ExtractionsResponse.model_validate(result)

        return ExtractionsResponse.model_validate(data)

    async def _extract_elements_with_parallel_batching(
        self,
        inputs: list[str],
        dictionary: list[str],
        *,
        type: str = "named-entities",
        expand_dictionary: bool = False,
        expand_dictionary_limit: Optional[int] = None,
        version: Optional[str] = None,
        await_job_result: bool = True,
    ) -> Union[ExtractionsResponse, AsyncJob]:
        """Extract elements with parallel batching for slow mode (fast=False) only.

        This method implements automatic batching for large extraction requests:
        - Splits inputs into 2,000-text batches
        - Processes up to 5 batches concurrently
        - Preserves input order in results
        - Aggregates usage metrics across all batches

        Args:
            inputs: Input strings to analyze with >200 inputs and fast=False
            dictionary: List of terms to extract from texts
            type: Extraction type ("named-entities" or "themes")
            expand_dictionary: Expand dictionary entries with synonyms
            expand_dictionary_limit: Limit for dictionary expansions
            version: Optional model version for reproducible output
            await_job_result: When False, return an AsyncJob handle for the first batch

        Returns:
            ExtractionsResponse with aggregated results or AsyncJob handle
        """
        from pulse.core.validation import ValidationLimits
        from pulse.core.models import UsageReport

        batch_size = ValidationLimits.EXTRACTIONS_BATCH_SIZE  # 2,000

        # Create batches
        batches = [
            inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)
        ]

        # Create job submitters for each batch
        async def create_batch_submitter(batch_inputs):
            # Submit batch job without waiting for result, bypassing batching logic
            return await self._extract_single_elements_request(
                inputs=batch_inputs,
                dictionary=dictionary,
                type=type,
                expand_dictionary=expand_dictionary,
                expand_dictionary_limit=expand_dictionary_limit,
                version=version,
                fast=False,  # Always use slow mode for batching
                await_job_result=False,
            )

        job_submitters = [create_batch_submitter(batch) for batch in batches]

        # If not waiting for results, return the first job
        if not await_job_result:
            # Submit first job and return it
            first_job = await job_submitters[0]
            return first_job

        # Process all batches concurrently (slow mode only)
        batch_results = await self._process_batches_concurrently(
            job_submitters,
            max_workers=5,  # Maximum 5 concurrent jobs
            timeout=600.0,
        )

        # Aggregate results while preserving order
        all_columns = []
        all_matrix_rows = []
        total_usage_records = []
        total_quantity = 0

        for batch_result in batch_results:
            # Parse batch result if it's a dict (from job.wait())
            if isinstance(batch_result, dict):
                batch_response = ExtractionsResponse.model_validate(batch_result)
            else:
                batch_response = batch_result

            # For the first batch, capture the column structure
            if not all_columns:
                all_columns = batch_response.columns

            # Add matrix rows in order
            all_matrix_rows.extend(batch_response.matrix)

            # Aggregate usage metrics
            if batch_response.usage:
                total_usage_records.extend(batch_response.usage.records)
                total_quantity += batch_response.usage.total

        # Create aggregated usage info
        aggregated_usage = UsageReport(
            records=total_usage_records, total=total_quantity
        )

        # Create final response with aggregated results
        return ExtractionsResponse(
            columns=all_columns,
            matrix=all_matrix_rows,
            usage=aggregated_usage,
            requestId=(
                batch_results[0].get("request_id")
                if isinstance(batch_results[0], dict)
                else getattr(batch_results[0], "requestId", None)
            ),
        )

    async def _extract_single_elements_request(
        self,
        inputs: list[str],
        dictionary: list[str],
        *,
        type: str = "named-entities",
        expand_dictionary: bool = False,
        expand_dictionary_limit: Optional[int] = None,
        version: Optional[str] = None,
        fast: Optional[bool] = None,
        await_job_result: bool = True,
    ) -> Union[ExtractionsResponse, AsyncJob]:
        """Extract elements for a single request without batching logic.

        This method contains the original extraction logic and is used by the
        parallel batching system to avoid infinite recursion.

        Args:
            inputs: Input strings to analyze
            dictionary: List of terms to extract from texts
            type: Extraction type ("named-entities" or "themes")
            expand_dictionary: Expand dictionary entries with synonyms
            expand_dictionary_limit: Limit for dictionary expansions
            version: Optional model version for reproducible output
            fast: Use synchronous (True) or asynchronous (False) mode
            await_job_result: When False, return an AsyncJob handle instead of waiting

        Returns:
            ExtractionsResponse or AsyncJob handle
        """
        body: Dict[str, Any] = {
            "inputs": inputs,
            "dictionary": dictionary,
            "type": type,
            "expand_dictionary": expand_dictionary,
        }

        if expand_dictionary_limit is not None:
            body["expand_dictionary_limit"] = expand_dictionary_limit
        if version is not None:
            body["version"] = version
        if fast is not None:
            body["fast"] = fast

        fast_mode = bool(fast)

        response = await self._request("post", "/extractions", json=body)
        if response.status_code not in (200, 202):
            raise PulseAPIError(response)

        data = response.json()

        # If service enqueues an async job during fast sync, treat as error
        if response.status_code == 202 and fast_mode:
            raise PulseAPIError(response)

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
        inputs: list[str],
    ) -> "UsageEstimateResponse":
        """Estimate credit usage for a feature without authentication.

        Args:
            feature: Feature to estimate usage for. Options: "embeddings",
                "sentiment", "themes", "extractions", "summaries",
                "clustering", "similarity".
            inputs: Input texts for estimation.

        Returns:
            UsageEstimateResponse with estimated usage information.
        """
        # Import here to avoid circular imports
        from pulse.core.models import UsageEstimateResponse

        # Validate feature parameter
        valid_features = [
            "embeddings",
            "sentiment",
            "themes",
            "extractions",
            "summaries",
            "clustering",
            "similarity",
        ]
        if feature not in valid_features:
            raise ValueError(f"feature must be one of {valid_features}, got: {feature}")

        body: Dict[str, Any] = {"feature": feature, "inputs": inputs}

        # Create a temporary client without authentication for this endpoint
        temp_client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

        try:
            response = await temp_client.request("post", "/usage/estimate", json=body)

            if response.status_code != 200:
                raise PulseAPIError(response)

            data = response.json()
            return UsageEstimateResponse.model_validate(data)
        finally:
            await temp_client.aclose()

    async def get_job_status(self, job_id: str) -> AsyncJob:
        """Retrieve the status of a previously submitted job."""

        response = await self._request("get", "/jobs", params={"jobId": job_id})
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
