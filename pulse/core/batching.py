"""Batching utilities for similarity requests under Pulse API limits."""

import gc
import time
from typing import Any, Dict, List, Tuple, Callable

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None


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
    feature: str = "unknown",
    enable_progress: bool = True,
) -> List[Any]:
    """Process multiple batch jobs concurrently for async operations only (fast=False).

    This function provides controlled parallel processing for slow mode operations
    with comprehensive error handling and progress indicators.
    For fast mode operations (fast=True), jobs are processed sequentially to maintain
    thread safety and compatibility with testing frameworks like VCR.

    Args:
        job_submitters: List of callables that return Job instances when called.
        max_workers: Maximum number of concurrent workers for slow mode.
        timeout: Timeout for individual job completion.
        fast: If True, process sequentially. If False, use concurrent processing.
        feature: Name of the feature being processed (for error reporting).
        enable_progress: Whether to show progress indicators for long operations.

    Returns:
        List of job results in the same order as input submitters.

    Raises:
        ImportError: If concurrent processing is requested but dependencies are missing.
        BatchingError: If batch processing fails with detailed error information.
    """
    from pulse.core.validation import BatchingErrorHelper

    if fast:
        # Fast mode: sequential processing with basic error handling
        jobs = []
        failed_jobs = []

        for i, submitter in enumerate(job_submitters):
            try:
                job = submitter()
                jobs.append(job)
            except Exception as e:
                failed_jobs.append((i + 1, str(e)))

        if failed_jobs:
            # Create detailed error for failed job submissions
            first_failure = failed_jobs[0]
            error = BatchingErrorHelper.create_batch_failure_error(
                feature=feature,
                batch_num=first_failure[0],
                total_batches=len(job_submitters),
                error_message=first_failure[1],
                input_count=len(job_submitters),
            )
            error.add_context("failed_submissions", len(failed_jobs))
            error.add_context("successful_submissions", len(jobs))
            raise error

        # Wait for jobs with individual error handling
        results = []
        failed_results = []

        for i, job in enumerate(jobs):
            try:
                result = job.wait(timeout)
                results.append(result)
            except Exception as e:
                failed_results.append((i + 1, str(e)))

        if failed_results:
            # Create detailed error for failed job completions
            first_failure = failed_results[0]
            error = BatchingErrorHelper.create_batch_failure_error(
                feature=feature,
                batch_num=first_failure[0],
                total_batches=len(jobs),
                error_message=first_failure[1],
                input_count=len(jobs),
            )
            error.add_context("failed_completions", len(failed_results))
            error.add_context("successful_completions", len(results))
            raise error

        return results

    # Slow mode: concurrent processing with enhanced error handling
    ConcurrentJobManager, ConcurrentJobConfig, AsyncJobProcessor, has_concurrent = (
        _get_concurrent_classes()
    )

    if not has_concurrent:
        raise ImportError(
            "Concurrent processing requires additional dependencies. "
            "This should not happen as concurrent.py is part of the core module."
        )

    config = ConcurrentJobConfig(
        max_workers=max_workers, timeout=timeout, enable_progress=enable_progress
    )

    try:
        return AsyncJobProcessor.batch_process_with_concurrency(
            job_submitters, config, feature=feature
        )
    except Exception as e:
        # Wrap any processing errors with detailed batch context
        if not isinstance(e, Exception.__subclasses__()):
            error = BatchingErrorHelper.create_batch_failure_error(
                feature=feature,
                batch_num=0,  # Unknown which batch failed
                total_batches=len(job_submitters),
                error_message=str(e),
                input_count=len(job_submitters),
            )
            error.add_context("processing_mode", "concurrent")
            error.add_context("max_workers", max_workers)
            raise error
        raise


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


class NetworkResourceManager:
    """Manager for network resources with timeout and retry logic."""

    def __init__(
        self,
        base_timeout: float = 300.0,
        max_timeout: float = 900.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        connection_pool_size: int = 10,
    ):
        """Initialize the network resource manager.

        Args:
            base_timeout: Base timeout for network requests.
            max_timeout: Maximum timeout for network requests.
            max_retries: Maximum number of retries for failed requests.
            backoff_factor: Exponential backoff factor for retries.
            connection_pool_size: Size of the connection pool.
        """
        self.base_timeout = base_timeout
        self.max_timeout = max_timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.connection_pool_size = connection_pool_size
        self._active_connections = 0
        self._failed_requests = 0
        self._total_requests = 0

    def get_adaptive_timeout(self, batch_size: int, attempt: int = 0) -> float:
        """Get adaptive timeout based on batch size and retry attempt.

        Args:
            batch_size: Size of the batch being processed.
            attempt: Current retry attempt (0-based).

        Returns:
            Adaptive timeout in seconds.
        """
        # Base timeout scales with batch size
        size_factor = max(1.0, batch_size / 1000.0)  # Scale for larger batches
        timeout = self.base_timeout * size_factor

        # Increase timeout for retries
        if attempt > 0:
            timeout *= self.backoff_factor**attempt

        return min(timeout, self.max_timeout)

    def should_retry_request(self, error: Exception, attempt: int) -> bool:
        """Determine if a request should be retried based on error type and attempt.

        Args:
            error: The exception that occurred.
            attempt: Current attempt number (1-based).

        Returns:
            True if the request should be retried.
        """
        if attempt >= self.max_retries:
            return False

        # Retry on network errors, timeouts, and server errors
        error_str = str(error).lower()
        retryable_errors = [
            "timeout",
            "connection",
            "network",
            "502",
            "503",
            "504",
            "rate limit",
            "too many requests",
            "server error",
        ]

        return any(retryable_error in error_str for retryable_error in retryable_errors)

    def get_retry_delay(self, attempt: int) -> float:
        """Get delay before retry attempt.

        Args:
            attempt: Current attempt number (1-based).

        Returns:
            Delay in seconds.
        """
        base_delay = 1.0
        return base_delay * (self.backoff_factor ** (attempt - 1))

    def track_request_start(self) -> None:
        """Track the start of a network request."""
        self._active_connections += 1
        self._total_requests += 1

    def track_request_end(self, success: bool) -> None:
        """Track the end of a network request.

        Args:
            success: Whether the request was successful.
        """
        self._active_connections = max(0, self._active_connections - 1)
        if not success:
            self._failed_requests += 1

    def get_network_stats(self) -> Dict[str, Any]:
        """Get network usage statistics.

        Returns:
            Dictionary containing network statistics.
        """
        return {
            "active_connections": self._active_connections,
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "success_rate": (
                (self._total_requests - self._failed_requests)
                / max(1, self._total_requests)
            ),
            "connection_pool_utilization": self._active_connections
            / self.connection_pool_size,
        }

    def should_throttle_requests(self) -> bool:
        """Check if requests should be throttled due to high failure rate or usage.

        Returns:
            True if requests should be throttled.
        """
        stats = self.get_network_stats()

        # Throttle if failure rate is high or connection pool is saturated
        return (
            stats["success_rate"] < 0.7  # High failure rate
            or stats["connection_pool_utilization"] > 0.9  # Pool nearly full
        )


def enhanced_batch_similarity(
    client_method: Callable,
    request_bodies: List[Dict[str, Any]],
    full_a: List[str],
    full_b: List[str],
    fast: bool = True,
    max_workers: int = 5,
    timeout: float = 600.0,
    enable_optimization: bool = True,
) -> Any:
    """Enhanced batch similarity processing with resource optimization and network mgmt.

    This function extends the existing batch similarity logic with controlled
    parallel processing, resource optimization, and network management for
    async operations (fast=False only).

    Args:
        client_method: The client method to submit batch jobs.
        request_bodies: List of request body dictionaries for batching.
        full_a: Full list A for result stitching.
        full_b: Full list B for result stitching.
        fast: If True, use sequential processing. If False, use concurrent processing.
        max_workers: Maximum concurrent workers for slow mode.
        timeout: Timeout for job completion.
        enable_optimization: Whether to enable resource optimization.

    Returns:
        Stitched similarity matrix or result.
    """
    # Initialize resource management
    controller = BatchConcurrencyController(max_workers, enable_optimization)

    # Create job submitters with network management
    job_submitters = create_batch_job_submitters(client_method, request_bodies)

    # Process jobs with resource optimization and performance tracking
    total_items = len(full_a) + len(full_b)
    results = controller.process_with_controlled_concurrency(
        job_submitters,
        fast=fast,
        feature="similarity",
        total_items=total_items,
        batch_size=max(len(full_a), len(full_b)),
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
    enable_optimization: bool = True,
) -> Any:
    """Enhanced batch clustering processing with resource optimization and network mgmt.

    This function implements intelligent parallel clustering batching with
    clustering-specific chunk sizes, resource optimization, and result
    reconstruction for async operations (fast=False only).

    Args:
        client_method: The client method to submit clustering jobs.
        inputs: List of input texts to cluster.
        k: Number of clusters.
        algorithm: Clustering algorithm to use.
        fast: If True, use sequential processing. If False, use concurrent processing.
        max_workers: Maximum concurrent workers for slow mode.
        timeout: Timeout for job completion.
        enable_optimization: Whether to enable resource optimization.

    Returns:
        Reconstructed clustering response.
    """
    # Initialize resource management
    controller = BatchConcurrencyController(max_workers, enable_optimization)

    # Optimize chunk size based on system resources
    chunk_size = CLUSTERING_CHUNK_SIZE
    if enable_optimization and controller.optimizer:
        optimized_size, _ = controller.optimizer.optimize_batch_sizes(
            "clustering", len(inputs), chunk_size
        )
        chunk_size = optimized_size

    # Create chunks using optimized chunking strategy
    if chunk_size != CLUSTERING_CHUNK_SIZE:
        # Use optimized chunk size
        chunks = [inputs[i : i + chunk_size] for i in range(0, len(inputs), chunk_size)]
    else:
        # Use default clustering chunks
        chunks = _make_clustering_chunks(inputs)

    # Create request bodies for each chunk
    request_bodies = []
    for chunk in chunks:
        body = {"inputs": chunk, "k": k, "algorithm": algorithm}
        if fast is not None:
            body["fast"] = fast
        request_bodies.append(body)

    # Create job submitters with network management
    job_submitters = create_batch_job_submitters(client_method, request_bodies)

    # Process jobs with resource optimization and performance tracking
    results = controller.process_with_controlled_concurrency(
        job_submitters,
        fast=fast,
        feature="clustering",
        total_items=len(inputs),
        batch_size=chunk_size,
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


class ResourceMonitor:
    """Monitor system resources during batch processing for optimization."""

    def __init__(self):
        """Initialize the resource monitor."""
        self.has_psutil = HAS_PSUTIL
        self._initial_memory = None
        self._peak_memory = None
        self._start_time = None

    def start_monitoring(self) -> None:
        """Start monitoring resource usage."""
        self._start_time = time.time()
        if self.has_psutil:
            self._initial_memory = psutil.virtual_memory().percent
            self._peak_memory = self._initial_memory

    def update_peak_memory(self) -> None:
        """Update peak memory usage if current usage is higher."""
        if self.has_psutil:
            current_memory = psutil.virtual_memory().percent
            if self._peak_memory is None or current_memory > self._peak_memory:
                self._peak_memory = current_memory

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        if not self.has_psutil:
            return {"available": False, "message": "psutil not available"}

        memory = psutil.virtual_memory()
        return {
            "available": True,
            "percent_used": memory.percent,
            "available_gb": memory.available / (1024**3),
            "total_gb": memory.total / (1024**3),
            "initial_percent": self._initial_memory,
            "peak_percent": self._peak_memory,
        }

    def should_reduce_concurrency(self, threshold: float = 85.0) -> bool:
        """Check if concurrency should be reduced due to high memory usage."""
        if not self.has_psutil:
            return False

        current_memory = psutil.virtual_memory().percent
        return current_memory > threshold

    def get_recommended_batch_size(
        self, current_size: int, memory_threshold: float = 80.0
    ) -> int:
        """Get recommended batch size based on current memory usage."""
        if not self.has_psutil:
            return current_size

        memory_percent = psutil.virtual_memory().percent
        if memory_percent < memory_threshold:
            return current_size

        # Reduce batch size based on memory pressure
        reduction_factor = max(0.5, (100 - memory_percent) / (100 - memory_threshold))
        return max(100, int(current_size * reduction_factor))

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        elapsed_time = time.time() - self._start_time if self._start_time else 0
        memory_info = self.get_memory_usage()

        return {
            "elapsed_seconds": elapsed_time,
            "memory_info": memory_info,
            "memory_increase": (
                memory_info.get("peak_percent", 0)
                - memory_info.get("initial_percent", 0)
                if memory_info.get("available")
                else 0
            ),
        }


class BatchOptimizer:
    """Optimizer for batch processing performance and resource usage."""

    def __init__(self, enable_monitoring: bool = True):
        """Initialize the batch optimizer.

        Args:
            enable_monitoring: Whether to enable resource monitoring.
        """
        self.monitor = ResourceMonitor() if enable_monitoring else None
        self.enable_monitoring = enable_monitoring

    def optimize_batch_sizes(
        self,
        feature: str,
        input_count: int,
        default_batch_size: int,
        memory_threshold: float = 80.0,
    ) -> Tuple[int, int]:
        """Optimize batch size and concurrency based on system resources.

        Args:
            feature: Name of the feature being processed.
            input_count: Total number of inputs to process.
            default_batch_size: Default batch size for the feature.
            memory_threshold: Memory usage threshold for optimization.

        Returns:
            Tuple of (optimized_batch_size, recommended_max_workers).
        """
        if not self.monitor:
            return default_batch_size, 5

        # Get current memory usage
        memory_info = self.monitor.get_memory_usage()
        if not memory_info.get("available"):
            return default_batch_size, 5

        memory_percent = memory_info["percent_used"]
        available_gb = memory_info["available_gb"]

        # Optimize batch size based on memory
        if memory_percent > memory_threshold:
            # High memory usage - reduce batch size
            reduction_factor = max(
                0.3, (100 - memory_percent) / (100 - memory_threshold)
            )
            optimized_batch_size = max(100, int(default_batch_size * reduction_factor))
        elif available_gb > 8.0 and memory_percent < 50:
            # Plenty of memory - can increase batch size
            increase_factor = min(2.0, available_gb / 4.0)
            optimized_batch_size = min(
                int(default_batch_size * increase_factor),
                input_count,  # Don't exceed total input count
            )
        else:
            optimized_batch_size = default_batch_size

        # Optimize concurrency based on memory and CPU
        if memory_percent > 85:
            recommended_workers = 2  # Very conservative
        elif memory_percent > 70:
            recommended_workers = 3  # Somewhat conservative
        elif available_gb > 16.0:
            recommended_workers = min(
                8, max(5, int(available_gb / 2))
            )  # Scale with memory
        else:
            recommended_workers = 5  # Default

        return optimized_batch_size, recommended_workers

    def should_use_memory_optimization(self) -> bool:
        """Check if memory optimization should be enabled."""
        if not self.monitor:
            return False

        memory_info = self.monitor.get_memory_usage()
        if not memory_info.get("available"):
            return False

        # Enable optimization if memory usage is high or available memory is low
        return memory_info["percent_used"] > 70 or memory_info["available_gb"] < 4.0

    def cleanup_memory(self) -> None:
        """Perform memory cleanup operations."""
        # Force garbage collection
        gc.collect()

        # Additional cleanup for numpy arrays if available
        if HAS_NUMPY:
            # Clear numpy's internal caches
            try:
                np.core._clearcache()
            except AttributeError:
                pass  # Method not available in all numpy versions


class PerformanceBenchmark:
    """Benchmark utility to ensure batched operations are faster than manual."""

    def __init__(self):
        """Initialize the performance benchmark."""
        self.benchmarks = {}

    def start_benchmark(self, operation_id: str) -> None:
        """Start timing an operation.

        Args:
            operation_id: Unique identifier for the operation.
        """
        self.benchmarks[operation_id] = {
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
        }

    def end_benchmark(self, operation_id: str) -> float:
        """End timing an operation and return duration.

        Args:
            operation_id: Unique identifier for the operation.

        Returns:
            Duration in seconds.
        """
        if operation_id not in self.benchmarks:
            return 0.0

        benchmark = self.benchmarks[operation_id]
        benchmark["end_time"] = time.time()
        benchmark["duration"] = benchmark["end_time"] - benchmark["start_time"]

        return benchmark["duration"]

    def get_benchmark_results(self) -> Dict[str, Dict[str, Any]]:
        """Get all benchmark results.

        Returns:
            Dictionary of benchmark results.
        """
        return self.benchmarks.copy()

    def compare_performance(
        self,
        batched_duration: float,
        estimated_manual_duration: float,
        operation: str = "unknown",
    ) -> Dict[str, Any]:
        """Compare batched vs manual performance.

        Args:
            batched_duration: Duration of batched operation.
            estimated_manual_duration: Estimated duration of manual approach.
            operation: Name of the operation being compared.

        Returns:
            Performance comparison results.
        """
        if estimated_manual_duration <= 0:
            return {
                "operation": operation,
                "batched_faster": True,
                "speedup_factor": float("inf"),
                "time_saved": batched_duration,
                "recommendation": "Batching provides significant perf improvement",
            }

        speedup_factor = estimated_manual_duration / batched_duration
        time_saved = estimated_manual_duration - batched_duration

        return {
            "operation": operation,
            "batched_duration": batched_duration,
            "estimated_manual_duration": estimated_manual_duration,
            "batched_faster": speedup_factor > 1.0,
            "speedup_factor": speedup_factor,
            "time_saved": time_saved,
            "recommendation": (
                f"Batching is {speedup_factor:.1f}x faster, saving {time_saved:.1f}s"
                if speedup_factor > 1.0
                else f"Manual approach might be {1/speedup_factor:.1f}x faster"
            ),
        }

    def estimate_manual_duration(
        self, total_items: int, batch_size: int, per_request_overhead: float = 0.5
    ) -> float:
        """Estimate duration for manual chunking approach.

        Args:
            total_items: Total number of items to process.
            batch_size: Size of each manual batch.
            per_request_overhead: Overhead per request in seconds.

        Returns:
            Estimated duration in seconds.
        """
        num_batches = (total_items + batch_size - 1) // batch_size  # Ceiling division

        # Estimate sequential processing time
        # Assumes each batch takes base time + overhead, processed sequentially
        base_processing_time = 2.0  # Base processing time per batch
        total_overhead = num_batches * per_request_overhead
        total_processing = num_batches * base_processing_time

        return total_processing + total_overhead


class BatchConcurrencyController:
    """Controller for managing batch processing concurrency with optimization."""

    def __init__(
        self, max_concurrent_batches: int = 5, enable_optimization: bool = True
    ):
        """Initialize the concurrency controller.

        Args:
            max_concurrent_batches: Maximum number of concurrent batch operations.
            enable_optimization: Whether to enable resource-based optimization.
        """
        self.max_concurrent_batches = max_concurrent_batches
        self.enable_optimization = enable_optimization
        self.optimizer = (
            BatchOptimizer(enable_optimization) if enable_optimization else None
        )
        self.benchmark = PerformanceBenchmark()
        self._config = None

    def get_config(self, feature: str = "unknown", input_count: int = 0) -> Any:
        """Get the concurrent job configuration with resource optimization.

        Args:
            feature: Name of the feature being processed.
            input_count: Total number of inputs to process.

        Returns:
            ConcurrentJobConfig instance or None if concurrent processing unavailable.
        """
        if self._config is None or self.enable_optimization:
            (
                ConcurrentJobManager,
                ConcurrentJobConfig,
                AsyncJobProcessor,
                has_concurrent,
            ) = _get_concurrent_classes()

            if has_concurrent:
                max_workers = self.max_concurrent_batches

                # Apply resource-based optimization
                if self.optimizer and input_count > 0:
                    _, recommended_workers = self.optimizer.optimize_batch_sizes(
                        feature, input_count, 2000  # Default batch size
                    )
                    max_workers = min(max_workers, recommended_workers)

                self._config = ConcurrentJobConfig(
                    max_workers=max_workers,
                    timeout=600.0,
                    enable_progress=True,
                )
        return self._config

    def process_with_controlled_concurrency(
        self,
        job_submitters: List[Callable[[], Any]],
        fast: bool = True,
        feature: str = "unknown",
        total_items: int = 0,
        batch_size: int = 1000,
    ) -> List[Any]:
        """Process jobs with controlled concurrency and resource optimization.

        Args:
            job_submitters: List of job submitter functions.
            fast: If True, process sequentially. If False, use concurrent processing.
            feature: Name of the feature being processed.
            total_items: Total number of items being processed.
            batch_size: Size of each batch.

        Returns:
            List of job results.
        """
        # Start performance benchmark
        benchmark_id = f"{feature}_batch_{int(time.time())}"
        self.benchmark.start_benchmark(benchmark_id)

        # Start resource monitoring
        if self.optimizer and self.optimizer.monitor:
            self.optimizer.monitor.start_monitoring()

        try:
            # Check if memory optimization is needed
            if self.optimizer and self.optimizer.should_use_memory_optimization():
                self.optimizer.cleanup_memory()

            # Get optimized configuration
            config = self.get_config(feature, len(job_submitters))
            max_workers = config.max_workers if config else self.max_concurrent_batches

            result = process_batches_concurrently(
                job_submitters,
                max_workers=max_workers,
                timeout=600.0,
                fast=fast,
                feature=feature,
            )

            # Final cleanup if needed
            if self.optimizer and self.optimizer.should_use_memory_optimization():
                self.optimizer.cleanup_memory()

            return result

        finally:
            # End performance benchmark and compare
            actual_duration = self.benchmark.end_benchmark(benchmark_id)

            if total_items > 0:
                estimated_manual = self.benchmark.estimate_manual_duration(
                    total_items, batch_size
                )
                comparison = self.benchmark.compare_performance(
                    actual_duration, estimated_manual, feature
                )

                # Log performance comparison if batching is not faster
                if (
                    not comparison["batched_faster"]
                    and comparison["speedup_factor"] < 0.8
                ):
                    import warnings

                    warnings.warn(
                        f"Batched {feature} processing may be slower than manual: "
                        f"{comparison['recommendation']}",
                        UserWarning,
                    )

            # Log processing stats if monitoring is enabled
            if self.optimizer and self.optimizer.monitor:
                stats = self.optimizer.monitor.get_processing_stats()
                if (
                    stats["memory_increase"] > 20
                ):  # Log if memory increased significantly
                    import warnings

                    warnings.warn(
                        f"High memory usage during {feature} processing: "
                        f"{stats['memory_increase']:.1f}% increase",
                        UserWarning,
                    )
