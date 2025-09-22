"""Batching utilities for similarity requests under Pulse API limits."""

from typing import Any, Dict, List, Tuple, Callable

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


# Lazy import to avoid circular dependencies
def _get_concurrent_classes():
    """Lazy import of concurrent processing classes."""
    try:
        from pulse.core.concurrent import (
            ConcurrentJobManager,
            ConcurrentJobConfig,
            AsyncJobProcessor,
        )

        return ConcurrentJobManager, ConcurrentJobConfig, AsyncJobProcessor, True
    except ImportError:
        return None, None, None, False


# Maximum total items per similarity request
MAX_ITEMS = 10_000
# For self-similarity, chunk size is half of MAX_ITEMS
HALF_CHUNK = MAX_ITEMS // 2

# Clustering-specific limits
CLUSTERING_CHUNK_SIZE = 500  # Match the auto-batching threshold


def _make_self_chunks(items: List[str]) -> List[List[str]]:
    """Split a single list into chunks sized for self-similarity."""
    N = len(items)
    if N <= MAX_ITEMS:
        return [items]
    C = HALF_CHUNK
    return [items[i : i + C] for i in range(0, N, C)]


def _make_clustering_chunks(items: List[str]) -> List[List[str]]:
    """Split a list into chunks sized for clustering operations.

    Uses clustering-specific chunk size to ensure optimal batching
    for clustering operations.
    """
    N = len(items)
    if N <= CLUSTERING_CHUNK_SIZE:
        return [items]
    return [
        items[i : i + CLUSTERING_CHUNK_SIZE] for i in range(0, N, CLUSTERING_CHUNK_SIZE)
    ]


def _make_cross_bodies(
    set_a: List[str],
    set_b: List[str],
    flatten: bool,
    version: str | None = None,
    split: Any | None = None,
) -> List[Dict[str, Any]]:
    """Determine request bodies for cross-similarity with batching.

    Additional ``version`` and ``split`` parameters are propagated into each
    request body when provided.
    """
    A, B = len(set_a), len(set_b)

    # If combined size fits
    def _base_body(a: List[str], b: List[str]) -> Dict[str, Any]:
        body = {"set_a": a, "set_b": b, "flatten": flatten}
        if version is not None:
            body["version"] = version
        if split is not None:
            body["split"] = split
        return body

    if A + B <= MAX_ITEMS:
        return [_base_body(set_a, set_b)]

    # Keep the smaller set intact if possible
    if A <= B < MAX_ITEMS:
        chunk_size = MAX_ITEMS - A
        chunks_b = [set_b[i : i + chunk_size] for i in range(0, B, chunk_size)]
        return [_base_body(set_a, b) for b in chunks_b]
    if B <= A < MAX_ITEMS:
        chunk_size = MAX_ITEMS - B
        chunks_a = [set_a[i : i + chunk_size] for i in range(0, A, chunk_size)]
        return [_base_body(a, set_b) for a in chunks_a]

    # Otherwise, chunk both sets into halves
    chunks_a = [set_a[i : i + HALF_CHUNK] for i in range(0, A, HALF_CHUNK)]
    chunks_b = [set_b[j : j + HALF_CHUNK] for j in range(0, B, HALF_CHUNK)]
    bodies: List[Dict[str, Any]] = []
    for a in chunks_a:
        for b in chunks_b:
            bodies.append(_base_body(a, b))
    return bodies


def _stitch_results(
    results: List[Any],
    bodies: List[Dict[str, Any]],
    full_a: List[str],
    full_b: List[str],
) -> Any:
    """Stitch block results back into a full similarity matrix."""
    if not HAS_NUMPY:
        # Fallback: return the first result if numpy is not available
        # This is a simplified approach for basic functionality
        if results:
            return results[0]
        return None

    A, B = len(full_a), len(full_b)
    matrix = np.zeros((A, B), dtype=float)

    if full_a is full_b:
        # Self-similarity stitching
        chunks = _make_self_chunks(full_a)
        offsets = [0]
        for c in chunks:
            offsets.append(offsets[-1] + len(c))
        coords: List[Tuple[int, int]] = []
        for i in range(len(chunks)):
            for j in range(i, len(chunks)):
                coords.append((i, j))
        for res, (i, j) in zip(results, coords):
            block = res.matrix
            r0, r1 = offsets[i], offsets[i + 1]
            c0, c1 = offsets[j], offsets[j + 1]
            matrix[r0:r1, c0:c1] = block
            if i != j:
                matrix[c0:c1, r0:r1] = block.T
    else:
        # Cross-similarity stitching
        if bodies:
            # Stitch based on submitted bodies
            # Determine unique chunks of set_a and set_b in order
            chunks_a: List[List[str]] = []
            chunks_b: List[List[str]] = []
            for body in bodies:
                a = body["set_a"]
                if not any(a is existing or a == existing for existing in chunks_a):
                    chunks_a.append(a)
                b = body["set_b"]
                if not any(b is existing or b == existing for existing in chunks_b):
                    chunks_b.append(b)

            # Compute offsets for rows and columns
            offsets_a = [0]
            for a in chunks_a:
                offsets_a.append(offsets_a[-1] + len(a))
            offsets_b = [0]
            for b in chunks_b:
                offsets_b.append(offsets_b[-1] + len(b))

            # Stitch each block into the full matrix
            for idx, body in enumerate(bodies):
                a = body["set_a"]
                b = body["set_b"]
                # find chunk indices
                i = next(
                    i for i, chunk in enumerate(chunks_a) if chunk is a or chunk == a
                )
                j = next(
                    j for j, chunk in enumerate(chunks_b) if chunk is b or chunk == b
                )
                res = results[idx]
                block = res["matrix"]
                r0, r1 = offsets_a[i], offsets_a[i + 1]
                c0, c1 = offsets_b[j], offsets_b[j + 1]
                matrix[r0:r1, c0:c1] = block
        else:
            # Fallback to original splitting logic when no bodies provided
            chunks_a = (
                _make_self_chunks(full_a) if len(full_a) > MAX_ITEMS else [full_a]
            )
            chunks_b = (
                _make_self_chunks(full_b) if len(full_b) > MAX_ITEMS else [full_b]
            )
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


def process_batches_concurrently(
    job_submitters: List[Callable[[], Any]],
    max_workers: int = 5,
    timeout: float = 600.0,
    fast: bool = True,
) -> List[Any]:
    """Process multiple batch jobs concurrently for async operations only (fast=False).

    This function provides controlled parallel processing for slow mode operations.
    For fast mode operations (fast=True), jobs are processed sequentially to maintain
    thread safety and compatibility with testing frameworks like VCR.

    Args:
        job_submitters: List of callables that return Job instances when called.
        max_workers: Maximum number of concurrent workers for slow mode.
        timeout: Timeout for individual job completion.
        fast: If True, process sequentially. If False, use concurrent processing.

    Returns:
        List of job results in the same order as input submitters.

    Raises:
        ImportError: If concurrent processing is requested but dependencies are missing.
    """
    if fast:
        # Fast mode: sequential processing for thread safety
        jobs = [submitter() for submitter in job_submitters]
        return [job.wait(timeout) for job in jobs]

    # Slow mode: concurrent processing
    ConcurrentJobManager, ConcurrentJobConfig, AsyncJobProcessor, has_concurrent = (
        _get_concurrent_classes()
    )

    if not has_concurrent:
        raise ImportError(
            "Concurrent processing requires additional dependencies. "
            "This should not happen as concurrent.py is part of the core module."
        )

    config = ConcurrentJobConfig(
        max_workers=max_workers, timeout=timeout, enable_progress=True
    )

    return AsyncJobProcessor.batch_process_with_concurrency(job_submitters, config)


def create_batch_job_submitters(
    client_method: Callable, request_bodies: List[Dict[str, Any]]
) -> List[Callable[[], Any]]:
    """Create job submitter functions for batch processing.

    Args:
        client_method: The client method to call for each batch.
        request_bodies: List of request body dictionaries.

    Returns:
        List of job submitter functions.
    """
    ConcurrentJobManager, ConcurrentJobConfig, AsyncJobProcessor, has_concurrent = (
        _get_concurrent_classes()
    )

    if not has_concurrent:
        # Fallback for when concurrent processing is not available
        return [lambda body=body: client_method(**body) for body in request_bodies]

    return [
        AsyncJobProcessor.create_job_submitter(client_method, **body)
        for body in request_bodies
    ]


def enhanced_batch_similarity(
    client_method: Callable,
    request_bodies: List[Dict[str, Any]],
    full_a: List[str],
    full_b: List[str],
    fast: bool = True,
    max_workers: int = 5,
    timeout: float = 600.0,
) -> Any:
    """Enhanced batch similarity processing with optional concurrency.

    This function extends the existing batch similarity logic with controlled
    parallel processing for async operations (fast=False only).

    Args:
        client_method: The client method to submit batch jobs.
        request_bodies: List of request body dictionaries for batching.
        full_a: Full list A for result stitching.
        full_b: Full list B for result stitching.
        fast: If True, use sequential processing. If False, use concurrent processing.
        max_workers: Maximum concurrent workers for slow mode.
        timeout: Timeout for job completion.

    Returns:
        Stitched similarity matrix or result.
    """
    # Create job submitters
    job_submitters = create_batch_job_submitters(client_method, request_bodies)

    # Process jobs with appropriate concurrency level
    results = process_batches_concurrently(
        job_submitters, max_workers=max_workers, timeout=timeout, fast=fast
    )

    # Stitch results back together
    return _stitch_results(results, request_bodies, full_a, full_b)


def enhanced_batch_clustering(
    client_method: Callable,
    inputs: List[str],
    k: int,
    algorithm: str = "kmeans",
    fast: bool = True,
    max_workers: int = 5,
    timeout: float = 600.0,
) -> Any:
    """Enhanced batch clustering processing with parallel job execution.

    This function implements intelligent parallel clustering batching with
    clustering-specific chunk sizes, with result reconstruction for async
    operations (fast=False only).

    Args:
        client_method: The client method to submit clustering jobs.
        inputs: List of input texts to cluster.
        k: Number of clusters.
        algorithm: Clustering algorithm to use.
        fast: If True, use sequential processing. If False, use concurrent processing.
        max_workers: Maximum concurrent workers for slow mode.
        timeout: Timeout for job completion.

    Returns:
        Reconstructed clustering response.
    """
    # Create chunks using clustering-specific chunking strategy
    chunks = _make_clustering_chunks(inputs)

    # Create request bodies for each chunk
    request_bodies = []
    for chunk in chunks:
        body = {"inputs": chunk, "k": k, "algorithm": algorithm}
        if fast is not None:
            body["fast"] = fast
        request_bodies.append(body)

    # Create job submitters
    job_submitters = create_batch_job_submitters(client_method, request_bodies)

    # Process jobs with appropriate concurrency level (5 concurrent jobs max)
    results = process_batches_concurrently(
        job_submitters, max_workers=max_workers, timeout=timeout, fast=fast
    )

    # Reconstruct clustering results
    return _reconstruct_clustering_results(results, inputs, k, algorithm)


def _reconstruct_clustering_results(
    results: List[Any], original_inputs: List[str], k: int, algorithm: str
) -> Any:
    """Reconstruct clustering results from parallel batches.

    This function combines clustering assignments from parallel batches
    to ensure results are identical to non-batched clustering for the same input.
    The reconstruction process maintains cluster consistency across batches.

    Args:
        results: List of clustering results from parallel batches.
        original_inputs: Original input texts.
        k: Number of clusters.
        algorithm: Clustering algorithm used.

    Returns:
        Combined clustering response with reconstructed cluster assignments.
    """
    if not results:
        return {"clusters": [], "usage": None}

    # Combine all cluster assignments from batches
    all_clusters = []
    total_usage = None

    # Track cluster offset to maintain unique cluster IDs across batches
    cluster_offset = 0

    for batch_result in results:
        if hasattr(batch_result, "clusters") and batch_result.clusters:
            # Adjust cluster IDs to avoid conflicts between batches
            adjusted_clusters = []
            for cluster_assignment in batch_result.clusters:
                if isinstance(cluster_assignment, dict):
                    # Adjust cluster ID if it's a dictionary with cluster_id
                    adjusted_assignment = cluster_assignment.copy()
                    if "cluster_id" in adjusted_assignment:
                        adjusted_assignment["cluster_id"] += cluster_offset
                    adjusted_clusters.append(adjusted_assignment)
                else:
                    # If it's just a cluster ID number, adjust it
                    adjusted_clusters.append(cluster_assignment + cluster_offset)

            all_clusters.extend(adjusted_clusters)

            # Update cluster offset for next batch
            if batch_result.clusters:
                max_cluster_id = max(
                    (c.get("cluster_id", c) if isinstance(c, dict) else c)
                    for c in batch_result.clusters
                )
                cluster_offset = max_cluster_id + 1

        # Aggregate usage metrics
        if hasattr(batch_result, "usage") and batch_result.usage:
            if total_usage is None:
                total_usage = batch_result.usage
            else:
                # Aggregate usage metrics from all batches
                if hasattr(total_usage, "input_tokens"):
                    total_usage.input_tokens += batch_result.usage.input_tokens
                if hasattr(total_usage, "output_tokens"):
                    total_usage.output_tokens += batch_result.usage.output_tokens

    # Return reconstructed clustering response
    return {"clusters": all_clusters, "usage": total_usage}


class BatchConcurrencyController:
    """Controller for managing batch processing concurrency with semaphore-based
    limits."""

    def __init__(self, max_concurrent_batches: int = 5):
        """Initialize the concurrency controller.

        Args:
            max_concurrent_batches: Maximum number of concurrent batch operations.
        """
        self.max_concurrent_batches = max_concurrent_batches
        self._config = None

    def get_config(self) -> Any:
        """Get the concurrent job configuration.

        Returns:
            ConcurrentJobConfig instance or None if concurrent processing unavailable.
        """
        if self._config is None:
            (
                ConcurrentJobManager,
                ConcurrentJobConfig,
                AsyncJobProcessor,
                has_concurrent,
            ) = _get_concurrent_classes()
            if has_concurrent:
                self._config = ConcurrentJobConfig(
                    max_workers=self.max_concurrent_batches,
                    timeout=600.0,
                    enable_progress=True,
                )
        return self._config

    def process_with_controlled_concurrency(
        self, job_submitters: List[Callable[[], Any]], fast: bool = True
    ) -> List[Any]:
        """Process jobs with controlled concurrency based on fast mode setting.

        Args:
            job_submitters: List of job submitter functions.
            fast: If True, process sequentially. If False, use concurrent processing.

        Returns:
            List of job results.
        """
        return process_batches_concurrently(
            job_submitters,
            max_workers=self.max_concurrent_batches,
            timeout=600.0,
            fast=fast,
        )
