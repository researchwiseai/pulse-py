"""CoreClient for interacting with the Pulse API synchronously."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Union, Optional
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

MAX_ITEMS = 10_000
HALF_CHUNK = MAX_ITEMS // 2


def _make_self_chunks(items: List[str]) -> List[List[str]]:
    """Split a single list into chunks sized for self-similarity."""
    N = len(items)
    if N <= MAX_ITEMS:
        return [items]
    # chunk size for self-similarity, so that chunk+chunk <= MAX_ITEMS
    C = HALF_CHUNK
    return [items[i : i + C] for i in range(0, N, C)]


def _make_cross_bodies(
    set_a: List[str], set_b: List[str], flatten: bool
) -> List[Dict[str, Any]]:
    """Determine request bodies for cross-similarity with batching."""
    A, B = len(set_a), len(set_b)
    # if combined fits
    if A + B <= MAX_ITEMS:
        return [{"set_a": set_a, "set_b": set_b, "flatten": flatten}]

    # keep smallest intact if possible
    if A <= B < MAX_ITEMS:
        chunk_size = MAX_ITEMS - A
        chunks_b = [set_b[i : i + chunk_size] for i in range(0, B, chunk_size)]
        return [{"set_a": set_a, "set_b": b, "flatten": flatten} for b in chunks_b]
    if B <= A < MAX_ITEMS:
        chunk_size = MAX_ITEMS - B
        chunks_a = [set_a[i : i + chunk_size] for i in range(0, A, chunk_size)]
        return [{"set_a": a, "set_b": set_b, "flatten": flatten} for a in chunks_a]

    # else chunk both halves
    chunks_a = [set_a[i : i + HALF_CHUNK] for i in range(0, A, HALF_CHUNK)]
    chunks_b = [set_b[j : j + HALF_CHUNK] for j in range(0, B, HALF_CHUNK)]
    bodies: List[Dict[str, Any]] = []
    for i, a in enumerate(chunks_a):
        for j, b in enumerate(chunks_b):
            bodies.append({"set_a": a, "set_b": b, "flatten": flatten})
    return bodies


def _stitch_results(
    results: List[Any],
    bodies: List[Dict[str, Any]],
    full_a: List[str],
    full_b: List[str],
) -> Any:
    """
    Stitch block results back into a full matrix.
    Expects each result to be a SimilarityResponse-like object
    with .matrix or .flattened + dims.
    """
    # Determine dimensions
    A, B = len(full_a), len(full_b)
    # allocate
    import numpy as np

    matrix = np.zeros((A, B), dtype=float)

    # track offsets
    offsets_a: List[int] = []
    offsets_b: List[int] = []
    idx = 0
    if full_a is full_b:
        # self-sim
        chunks = _make_self_chunks(full_a)
        k = len(chunks)
        offsets = [0]
        for c in chunks:
            offsets.append(offsets[-1] + len(c))
        coords: List[Tuple[int, int]] = []
        for i in range(k):
            for j in range(i, k):
                coords.append((i, j))
        for res, (i, j) in zip(results, coords):
            block = res.matrix  # shape (len(chunk_i), len(chunk_j))
            r0, r1 = offsets[i], offsets[i + 1]
            c0, c1 = offsets[j], offsets[j + 1]
            matrix[r0:r1, c0:c1] = block
            if i != j:
                matrix[c0:c1, r0:r1] = block.T
    else:
        # cross-sim
        # build offsets for each body in order
        # naive: assume bodies list order corresponds to row by row
        # first compute chunk lists
        chunks_a = _make_self_chunks(full_a) if len(full_a) > MAX_ITEMS else [full_a]
        chunks_b = _make_self_chunks(full_b) if len(full_b) > MAX_ITEMS else [full_b]
        offsets_a = [0]
        for c in chunks_a:
            offsets_a.append(offsets_a[-1] + len(c))
        offsets_b = [0]
        for c in chunks_b:
            offsets_b.append(offsets_b[-1] + len(c))
        idx = 0
        for i, a in enumerate(chunks_a):
            for j, b in enumerate(chunks_b):
                res = results[idx]
                block = res.matrix
                r0, r1 = offsets_a[i], offsets_a[i + 1]
                c0, c1 = offsets_b[j], offsets_b[j + 1]
                matrix[r0:r1, c0:c1] = block
                idx += 1
    return matrix


class GzipClient(httpx.Client):
    def build_request(self, method: str, url: str, **kwargs) -> httpx.Request:
        # Only compress when the user explicitly passed `content=…`
        if "content" in kwargs and kwargs["content"]:
            original = kwargs["content"]
            # ensure bytes
            if isinstance(original, str):
                original = original.encode("utf-8")
            compressed = gzip.compress(original)

            # update the kwargs used to build the Request
            kwargs["content"] = compressed
            headers = kwargs.setdefault("headers", {})
            headers["Content-Encoding"] = "gzip"
            headers["Content-Length"] = str(len(compressed))

        return super().build_request(method, url, **kwargs)


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
            else GzipClient(
                base_url=self.base_url,
                timeout=self.timeout,
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

    def _submit_batch_similarity_job(self, **kwargs) -> Any:
        # wrapper around compare_similarity to return Job-like
        # stub here for real implementation
        return self.compare_similarity(fast=False, **kwargs)

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

        # wait for all
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(job.wait): job for job in jobs}
            results = [f.result() for f in as_completed(futures)]

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
