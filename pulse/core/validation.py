"""Input validation utilities for Pulse API endpoints."""

from typing import List, Optional, Any, Dict


class ValidationLimits:
    """Input validation limits for different endpoints and modes."""

    # Embeddings limits
    EMBEDDINGS_SYNC_MAX = 200
    EMBEDDINGS_ASYNC_MAX = 1_000_000
    EMBEDDINGS_BATCH_SIZE = 5_000

    # Similarity limits
    SIMILARITY_SELF_SYNC_MAX = 500
    SIMILARITY_CROSS_SYNC_PRODUCT_MAX = 20000
    SIMILARITY_ASYNC_MAX = 44721

    # Themes limits
    THEMES_SYNC_MAX = 200
    THEMES_ASYNC_MAX = 500

    # Clustering limits - following similarity pattern
    CLUSTERING_SYNC_MAX = 500
    CLUSTERING_ASYNC_MAX = 44721
    CLUSTERING_AUTO_BATCH_THRESHOLD = 500

    # Sentiment limits
    SENTIMENT_SYNC_MAX = 200
    SENTIMENT_ASYNC_MAX = 1_000_000
    SENTIMENT_BATCH_SIZE = 2_000

    # Extractions limits
    EXTRACTIONS_SYNC_MAX = 200
    EXTRACTIONS_ASYNC_MAX = 1_000_000
    EXTRACTIONS_BATCH_SIZE = 2_000


# Error documentation structure for comprehensive error handling
ERROR_DOCUMENTATION = {
    "FAST_MODE_LIMIT_EXCEEDED": {
        "code": "BATCH_001",
        "message_template": (
            "Fast mode supports up to {limit} texts. "
            "Use slow mode for larger datasets or reduce input size."
        ),
        "resolution_steps": [
            "Set fast=False to enable slow mode processing",
            "Reduce input size to {limit} or fewer texts",
            "Consider processing data in smaller batches",
        ],
        "examples": {
            "sentiment": "client.analyze_sentiment(texts[:200], fast=True)",
            "embeddings": (
                "client.create_embeddings("
                "EmbeddingsRequest(inputs=texts[:200], fast=True))"
            ),
            "extractions": (
                "client.extract_elements(texts[:200], dictionary, fast=True)"
            ),
        },
    },
    "SLOW_MODE_LIMIT_EXCEEDED": {
        "code": "BATCH_002",
        "message_template": (
            "Maximum input limit is {limit:,} texts. "
            "Please reduce your dataset size."
        ),
        "resolution_steps": [
            "Split dataset into chunks of {limit:,} or fewer texts",
            "Process data in multiple separate requests",
            "Consider data sampling or filtering strategies",
        ],
        "examples": {
            "chunking": "for chunk in chunks(texts, {limit}): process_chunk(chunk)"
        },
    },
    "BATCH_JOB_FAILED": {
        "code": "BATCH_003",
        "message_template": (
            "Batch {batch_num} of {total_batches} failed: {error_message}"
        ),
        "resolution_steps": [
            "Check individual batch data for issues",
            "Retry with smaller batch sizes",
            "Verify network connectivity and API limits",
            "Review batch content for problematic inputs",
        ],
        "examples": {
            "retry": (
                "try: result = process_batch(batch) "
                "except BatchingError: retry_with_smaller_batch()"
            )
        },
    },
    "CONCURRENCY_LIMIT_EXCEEDED": {
        "code": "BATCH_004",
        "message_template": (
            "Too many concurrent jobs: {current_jobs}. " "Maximum allowed: {max_jobs}"
        ),
        "resolution_steps": [
            "Reduce concurrent job limit in configuration",
            "Wait for existing jobs to complete before starting new ones",
            "Consider sequential processing for resource-constrained environments",
        ],
    },
    "CLUSTERING_AUTO_BATCH_TRIGGERED": {
        "code": "BATCH_005",
        "message_template": (
            "Input size {input_count} exceeds threshold {threshold}. "
            "Automatic batching enabled."
        ),
        "resolution_steps": [
            "This is informational - batching will proceed automatically",
            "Results will be reconstructed to match non-batched clustering",
            "Consider reducing input size if processing time is too long",
        ],
    },
}


class PulseValidationError(Exception):
    """Custom validation error with detailed context."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        limit: Optional[int] = None,
        endpoint: Optional[str] = None,
        mode: Optional[str] = None,
    ):
        self.message = message
        self.field = field
        self.value = value
        self.limit = limit
        self.endpoint = endpoint
        self.mode = mode
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format a user-friendly error message."""
        parts = []

        if self.endpoint:
            parts.append(f"[{self.endpoint}]")

        if self.mode:
            parts.append(f"({self.mode} mode)")

        parts.append(self.message)

        if self.field and self.value is not None:
            parts.append(f"Field '{self.field}' has value {self.value}")

        if self.limit is not None:
            parts.append(f"Limit: {self.limit}")

        return " ".join(parts)


class BatchingError(PulseValidationError):
    """Specialized error for batching operations with structured error information."""

    def __init__(
        self,
        message: str,
        feature: str,
        input_count: int,
        limit: int,
        suggested_action: str,
        error_code: Optional[str] = None,
        batch_number: Optional[int] = None,
        total_batches: Optional[int] = None,
        resolution_steps: Optional[List[str]] = None,
        **kwargs,
    ):
        self.feature = feature
        self.input_count = input_count
        self.suggested_action = suggested_action
        self.error_code = error_code
        self.batch_number = batch_number
        self.total_batches = total_batches
        self.resolution_steps = resolution_steps or []

        super().__init__(message, limit=limit, endpoint=feature, **kwargs)

    def get_structured_info(self) -> Dict[str, Any]:
        """Get structured error information for programmatic handling."""
        return {
            "error_code": self.error_code,
            "feature": self.feature,
            "input_count": self.input_count,
            "limit": self.limit,
            "mode": self.mode,
            "suggested_action": self.suggested_action,
            "batch_number": self.batch_number,
            "total_batches": self.total_batches,
            "resolution_steps": self.resolution_steps,
        }


class BatchingErrorHelper:
    """Helper class for generating actionable error guidance and diagnostics."""

    @staticmethod
    def create_fast_mode_error(
        feature: str, input_count: int, limit: int
    ) -> BatchingError:
        """Create a fast mode limit exceeded error with structured information."""
        error_info = ERROR_DOCUMENTATION["FAST_MODE_LIMIT_EXCEEDED"]
        message = error_info["message_template"].format(limit=limit)

        resolution_steps = [
            step.format(limit=limit) for step in error_info["resolution_steps"]
        ]

        return BatchingError(
            message=message,
            feature=feature,
            input_count=input_count,
            limit=limit,
            suggested_action=f"Set fast=False or reduce input to {limit} texts",
            error_code=error_info["code"],
            mode="fast",
            resolution_steps=resolution_steps,
        )

    @staticmethod
    def create_slow_mode_error(
        feature: str, input_count: int, limit: int
    ) -> BatchingError:
        """Create a slow mode limit exceeded error with structured information."""
        error_info = ERROR_DOCUMENTATION["SLOW_MODE_LIMIT_EXCEEDED"]
        message = error_info["message_template"].format(limit=limit)

        resolution_steps = [
            step.format(limit=limit) for step in error_info["resolution_steps"]
        ]

        return BatchingError(
            message=message,
            feature=feature,
            input_count=input_count,
            limit=limit,
            suggested_action=f"Split dataset into chunks of {limit:,} or fewer texts",
            error_code=error_info["code"],
            mode="slow",
            resolution_steps=resolution_steps,
        )

    @staticmethod
    def create_batch_failure_error(
        feature: str,
        batch_num: int,
        total_batches: int,
        error_message: str,
        input_count: int = 0,
    ) -> BatchingError:
        """Create a batch job failure error with structured information."""
        error_info = ERROR_DOCUMENTATION["BATCH_JOB_FAILED"]
        message = error_info["message_template"].format(
            batch_num=batch_num,
            total_batches=total_batches,
            error_message=error_message,
        )

        return BatchingError(
            message=message,
            feature=feature,
            input_count=input_count,
            limit=0,  # Not applicable for batch failures
            suggested_action="Check batch data and retry with smaller batches",
            error_code=error_info["code"],
            batch_number=batch_num,
            total_batches=total_batches,
            resolution_steps=error_info["resolution_steps"],
        )

    @staticmethod
    def create_concurrency_error(current_jobs: int, max_jobs: int) -> BatchingError:
        """Create a concurrency limit exceeded error."""
        error_info = ERROR_DOCUMENTATION["CONCURRENCY_LIMIT_EXCEEDED"]
        message = error_info["message_template"].format(
            current_jobs=current_jobs, max_jobs=max_jobs
        )

        return BatchingError(
            message=message,
            feature="batching",
            input_count=current_jobs,
            limit=max_jobs,
            suggested_action=f"Reduce concurrent jobs to {max_jobs} or fewer",
            error_code=error_info["code"],
            resolution_steps=error_info["resolution_steps"],
        )

    @staticmethod
    def get_error_documentation(error_code: str) -> Optional[Dict[str, Any]]:
        """Get documentation for a specific error code."""
        for error_type, info in ERROR_DOCUMENTATION.items():
            if info["code"] == error_code:
                return info
        return None

    @staticmethod
    def generate_example_code(feature: str, error_code: str) -> Optional[str]:
        """Generate example code for handling a specific error."""
        error_info = BatchingErrorHelper.get_error_documentation(error_code)
        if not error_info or "examples" not in error_info:
            return None

        examples = error_info["examples"]

        # Return feature-specific example if available, otherwise return first example
        if feature in examples:
            return examples[feature]
        elif examples:
            return list(examples.values())[0]

        return None

    @staticmethod
    def diagnose_error(error: Exception) -> Dict[str, Any]:
        """Analyze error and provide diagnostic information."""
        if isinstance(error, BatchingError):
            return {
                "error_type": "BatchingError",
                "structured_info": error.get_structured_info(),
                "example_code": BatchingErrorHelper.generate_example_code(
                    error.feature, error.error_code
                ),
                "documentation": BatchingErrorHelper.get_error_documentation(
                    error.error_code
                ),
            }
        elif isinstance(error, PulseValidationError):
            return {
                "error_type": "PulseValidationError",
                "endpoint": error.endpoint,
                "mode": error.mode,
                "field": error.field,
                "value": error.value,
                "limit": error.limit,
            }
        else:
            return {"error_type": type(error).__name__, "message": str(error)}


def validate_embeddings_input(inputs: List[Any], fast: Optional[bool] = None) -> None:
    """
    Validate embeddings input according to sync/async limits.

    Args:
        inputs: List of input texts
        fast: Sync (True) or async (False/None) mode

    Raises:
        BatchingError: If validation fails with enhanced error information
    """
    if not inputs:
        raise PulseValidationError(
            "Input list cannot be empty", field="inputs", endpoint="embeddings"
        )

    input_count = len(inputs)

    if fast is True:
        # Fast mode - reject with clear error message
        if input_count > ValidationLimits.EMBEDDINGS_SYNC_MAX:
            raise BatchingErrorHelper.create_fast_mode_error(
                "embeddings", input_count, ValidationLimits.EMBEDDINGS_SYNC_MAX
            )
    else:
        # Slow mode
        if input_count > ValidationLimits.EMBEDDINGS_ASYNC_MAX:
            raise BatchingErrorHelper.create_slow_mode_error(
                "embeddings", input_count, ValidationLimits.EMBEDDINGS_ASYNC_MAX
            )


def validate_similarity_input(
    set: Optional[List[str]] = None,
    set_a: Optional[List[str]] = None,
    set_b: Optional[List[str]] = None,
    fast: Optional[bool] = None,
) -> None:
    """
    Validate similarity input according to sync/async limits.

    Args:
        set: Self-similarity input texts
        set_a: Cross-similarity set A
        set_b: Cross-similarity set B
        fast: Sync (True) or async (False/None) mode

    Raises:
        PulseValidationError: If validation fails
    """
    if set is not None:
        # Self-similarity mode
        input_count = len(set)

        if fast is True:
            # Sync mode
            if input_count > ValidationLimits.SIMILARITY_SELF_SYNC_MAX:
                raise PulseValidationError(
                    f"Too many inputs for self-similarity sync mode: {input_count}",
                    field="set",
                    value=input_count,
                    limit=ValidationLimits.SIMILARITY_SELF_SYNC_MAX,
                    endpoint="similarity",
                    mode="self-sync",
                )
        else:
            # Async mode
            if input_count > ValidationLimits.SIMILARITY_ASYNC_MAX:
                raise PulseValidationError(
                    f"Too many inputs for self-similarity async mode: {input_count}",
                    field="set",
                    value=input_count,
                    limit=ValidationLimits.SIMILARITY_ASYNC_MAX,
                    endpoint="similarity",
                    mode="self-async",
                )

    elif set_a is not None and set_b is not None:
        # Cross-similarity mode
        len_a = len(set_a)
        len_b = len(set_b)

        if fast is True:
            # Sync mode - check cross product limit
            cross_product = len_a * len_b
            if cross_product > ValidationLimits.SIMILARITY_CROSS_SYNC_PRODUCT_MAX:
                raise PulseValidationError(
                    f"Cross-product too large for sync mode: "
                    f"{len_a} × {len_b} = {cross_product}",
                    field="set_a × set_b",
                    value=cross_product,
                    limit=ValidationLimits.SIMILARITY_CROSS_SYNC_PRODUCT_MAX,
                    endpoint="similarity",
                    mode="cross-sync",
                )
        else:
            # Async mode - check individual set limits
            if len_a > ValidationLimits.SIMILARITY_ASYNC_MAX:
                raise PulseValidationError(
                    f"Too many inputs in set_a for async mode: {len_a}",
                    field="set_a",
                    value=len_a,
                    limit=ValidationLimits.SIMILARITY_ASYNC_MAX,
                    endpoint="similarity",
                    mode="cross-async",
                )

            if len_b > ValidationLimits.SIMILARITY_ASYNC_MAX:
                raise PulseValidationError(
                    f"Too many inputs in set_b for async mode: {len_b}",
                    field="set_b",
                    value=len_b,
                    limit=ValidationLimits.SIMILARITY_ASYNC_MAX,
                    endpoint="similarity",
                    mode="cross-async",
                )


def validate_themes_input(inputs: List[str], fast: Optional[bool] = None) -> None:
    """
    Validate themes input according to sync/async limits.

    Args:
        inputs: List of input texts
        fast: Sync (True) or async (False/None) mode

    Raises:
        PulseValidationError: If validation fails
    """
    if not inputs:
        raise PulseValidationError(
            "Input list cannot be empty", field="inputs", endpoint="themes"
        )

    input_count = len(inputs)

    if fast is True:
        # Sync mode
        if input_count > ValidationLimits.THEMES_SYNC_MAX:
            raise PulseValidationError(
                f"Too many inputs for sync mode: {input_count}",
                field="inputs",
                value=input_count,
                limit=ValidationLimits.THEMES_SYNC_MAX,
                endpoint="themes",
                mode="sync",
            )
    else:
        # Async mode
        if input_count > ValidationLimits.THEMES_ASYNC_MAX:
            raise PulseValidationError(
                f"Too many inputs for async mode: {input_count}",
                field="inputs",
                value=input_count,
                limit=ValidationLimits.THEMES_ASYNC_MAX,
                endpoint="themes",
                mode="async",
            )


def validate_clustering_input(inputs: List[str], fast: Optional[bool] = None) -> None:
    """
    Validate clustering input according to sync/async limits.

    Note: Clustering automatically triggers batching when input > 500 texts,
    similar to similarity analysis pattern.

    Args:
        inputs: List of input texts
        fast: Sync (True) or async (False/None) mode

    Raises:
        PulseValidationError: If validation fails
    """
    if not inputs:
        raise PulseValidationError(
            "Input list cannot be empty", field="inputs", endpoint="clustering"
        )

    input_count = len(inputs)

    if fast is True:
        # Sync mode
        if input_count > ValidationLimits.CLUSTERING_SYNC_MAX:
            raise PulseValidationError(
                f"Too many inputs for sync mode: {input_count}",
                field="inputs",
                value=input_count,
                limit=ValidationLimits.CLUSTERING_SYNC_MAX,
                endpoint="clustering",
                mode="sync",
            )
    else:
        # Async mode - follows similarity pattern with intelligent batching
        if input_count > ValidationLimits.CLUSTERING_ASYNC_MAX:
            raise PulseValidationError(
                f"Too many inputs for async mode: {input_count}",
                field="inputs",
                value=input_count,
                limit=ValidationLimits.CLUSTERING_ASYNC_MAX,
                endpoint="clustering",
                mode="async",
            )


def validate_sentiment_input(inputs: List[Any], fast: Optional[bool] = None) -> None:
    """
    Validate sentiment input according to sync/async limits.

    Args:
        inputs: List of input texts
        fast: Sync (True) or async (False/None) mode

    Raises:
        BatchingError: If validation fails with enhanced error information
    """
    if not inputs:
        raise PulseValidationError(
            "Input list cannot be empty", field="inputs", endpoint="sentiment"
        )

    input_count = len(inputs)

    if fast is True:
        # Fast mode - reject with clear error message
        if input_count > ValidationLimits.SENTIMENT_SYNC_MAX:
            raise BatchingErrorHelper.create_fast_mode_error(
                "sentiment", input_count, ValidationLimits.SENTIMENT_SYNC_MAX
            )
    else:
        # Slow mode
        if input_count > ValidationLimits.SENTIMENT_ASYNC_MAX:
            raise BatchingErrorHelper.create_slow_mode_error(
                "sentiment", input_count, ValidationLimits.SENTIMENT_ASYNC_MAX
            )


def validate_extractions_input(inputs: List[str], fast: Optional[bool] = None) -> None:
    """
    Validate extractions input according to sync/async limits.

    Args:
        inputs: List of input texts
        fast: Sync (True) or async (False/None) mode

    Raises:
        BatchingError: If validation fails with enhanced error information
    """
    if not inputs:
        raise PulseValidationError(
            "Input list cannot be empty", field="inputs", endpoint="extractions"
        )

    input_count = len(inputs)

    if fast is True:
        # Fast mode - reject with clear error message
        if input_count > ValidationLimits.EXTRACTIONS_SYNC_MAX:
            raise BatchingErrorHelper.create_fast_mode_error(
                "extractions", input_count, ValidationLimits.EXTRACTIONS_SYNC_MAX
            )
    else:
        # Slow mode
        if input_count > ValidationLimits.EXTRACTIONS_ASYNC_MAX:
            raise BatchingErrorHelper.create_slow_mode_error(
                "extractions", input_count, ValidationLimits.EXTRACTIONS_ASYNC_MAX
            )


def validate_cross_field_constraints(model_data: Dict[str, Any], endpoint: str) -> None:
    """
    Validate cross-field constraints for specific endpoints.

    Args:
        model_data: Dictionary of model field values
        endpoint: The endpoint being validated

    Raises:
        PulseValidationError: If validation fails
    """
    if endpoint == "themes":
        # Validate initialSets > 1 requires interactive=true
        initial_sets = model_data.get("initialSets")
        interactive = model_data.get("interactive")

        if initial_sets and initial_sets > 1 and not interactive:
            raise PulseValidationError(
                "initialSets > 1 requires interactive=true",
                field="initialSets",
                value=initial_sets,
                endpoint="themes",
            )

    elif endpoint == "extractions":
        # Validate expand_dictionary=false when type="themes"
        extraction_type = model_data.get("type")
        expand_dictionary = model_data.get("expand_dictionary")

        if extraction_type == "themes" and expand_dictionary:
            raise PulseValidationError(
                "expand_dictionary must be false when type is 'themes'",
                field="expand_dictionary",
                value=expand_dictionary,
                endpoint="extractions",
            )

    elif endpoint == "similarity":
        # Validate set vs set_a/set_b constraints
        set_val = model_data.get("set")
        set_a = model_data.get("set_a")
        set_b = model_data.get("set_b")

        if set_val is None and (set_a is None or set_b is None):
            raise PulseValidationError(
                "Provide 'set' for self-similarity or both 'set_a' and "
                "'set_b' for cross-similarity",
                endpoint="similarity",
            )

        if set_val is not None and (set_a is not None or set_b is not None):
            raise PulseValidationError(
                "Cannot provide both 'set' and 'set_a'/'set_b'", endpoint="similarity"
            )


def get_validation_summary() -> Dict[str, Dict[str, int]]:
    """
    Get a summary of all validation limits for documentation purposes.

    Returns:
        Dictionary mapping endpoints to their sync/async limits
    """
    return {
        "embeddings": {
            "sync_max": ValidationLimits.EMBEDDINGS_SYNC_MAX,
            "async_max": ValidationLimits.EMBEDDINGS_ASYNC_MAX,
        },
        "similarity": {
            "self_sync_max": ValidationLimits.SIMILARITY_SELF_SYNC_MAX,
            "cross_sync_product_max": (
                ValidationLimits.SIMILARITY_CROSS_SYNC_PRODUCT_MAX
            ),
            "async_max": ValidationLimits.SIMILARITY_ASYNC_MAX,
        },
        "themes": {
            "sync_max": ValidationLimits.THEMES_SYNC_MAX,
            "async_max": ValidationLimits.THEMES_ASYNC_MAX,
        },
        "clustering": {
            "sync_max": ValidationLimits.CLUSTERING_SYNC_MAX,
            "async_max": ValidationLimits.CLUSTERING_ASYNC_MAX,
        },
        "sentiment": {
            "sync_max": ValidationLimits.SENTIMENT_SYNC_MAX,
            "async_max": ValidationLimits.SENTIMENT_ASYNC_MAX,
        },
        "extractions": {
            "sync_max": ValidationLimits.EXTRACTIONS_SYNC_MAX,
            "async_max": ValidationLimits.EXTRACTIONS_ASYNC_MAX,
        },
    }


def validate_before_request(endpoint: str, **kwargs) -> None:
    """
    Main validation entry point for API requests.

    Args:
        endpoint: The API endpoint name
        **kwargs: Request parameters to validate

    Raises:
        PulseValidationError: If validation fails
        BatchingError: If batching-related validation fails
    """
    ValidationHelper.validate_request(endpoint, **kwargs)


class ValidationHelper:
    """Helper class for client-side validation with user-friendly error messages."""

    @staticmethod
    def validate_request(endpoint: str, **kwargs) -> None:
        """
        Validate a request for a specific endpoint.

        Args:
            endpoint: The API endpoint name
            **kwargs: Request parameters to validate

        Raises:
            PulseValidationError: If validation fails
        """
        if endpoint == "embeddings":
            ValidationHelper._validate_embeddings_request(**kwargs)
        elif endpoint == "similarity":
            ValidationHelper._validate_similarity_request(**kwargs)
        elif endpoint == "themes":
            ValidationHelper._validate_themes_request(**kwargs)
        elif endpoint == "clustering":
            ValidationHelper._validate_clustering_request(**kwargs)
        elif endpoint == "sentiment":
            ValidationHelper._validate_sentiment_request(**kwargs)
        elif endpoint == "extractions":
            ValidationHelper._validate_extractions_request(**kwargs)
        else:
            raise PulseValidationError(f"Unknown endpoint: {endpoint}")

    @staticmethod
    def _validate_embeddings_request(**kwargs) -> None:
        """Validate embeddings request parameters."""
        inputs = kwargs.get("inputs", [])
        fast = kwargs.get("fast")

        validate_embeddings_input(inputs, fast)

    @staticmethod
    def _validate_similarity_request(**kwargs) -> None:
        """Validate similarity request parameters."""
        set_val = kwargs.get("set")
        set_a = kwargs.get("set_a")
        set_b = kwargs.get("set_b")
        fast = kwargs.get("fast")

        # Validate cross-field constraints first
        validate_cross_field_constraints(kwargs, "similarity")

        # Then validate input limits
        validate_similarity_input(set_val, set_a, set_b, fast)

    @staticmethod
    def _validate_themes_request(**kwargs) -> None:
        """Validate themes request parameters."""
        inputs = kwargs.get("inputs", [])
        fast = kwargs.get("fast")

        # Validate cross-field constraints first
        validate_cross_field_constraints(kwargs, "themes")

        # Then validate input limits
        validate_themes_input(inputs, fast)

    @staticmethod
    def _validate_clustering_request(**kwargs) -> None:
        """Validate clustering request parameters."""
        inputs = kwargs.get("inputs", [])
        fast = kwargs.get("fast")

        validate_clustering_input(inputs, fast)

    @staticmethod
    def _validate_sentiment_request(**kwargs) -> None:
        """Validate sentiment request parameters."""
        inputs = kwargs.get("inputs", [])
        fast = kwargs.get("fast")

        validate_sentiment_input(inputs, fast)

    @staticmethod
    def _validate_extractions_request(**kwargs) -> None:
        """Validate extractions request parameters."""
        inputs = kwargs.get("inputs", [])
        fast = kwargs.get("fast")

        # Validate cross-field constraints first
        validate_cross_field_constraints(kwargs, "extractions")

        # Then validate input limits
        validate_extractions_input(inputs, fast)

    @staticmethod
    def get_limit_info(endpoint: str, mode: str = "sync") -> Dict[str, Any]:
        """
        Get limit information for an endpoint and mode.

        Args:
            endpoint: The API endpoint name
            mode: Either "sync" or "async"

        Returns:
            Dictionary with limit information
        """
        limits = get_validation_summary()

        if endpoint not in limits:
            return {"error": f"Unknown endpoint: {endpoint}"}

        endpoint_limits = limits[endpoint]

        if endpoint == "similarity":
            if mode == "sync":
                return {
                    "self_max": endpoint_limits["self_sync_max"],
                    "cross_product_max": endpoint_limits["cross_sync_product_max"],
                    "description": (
                        f"Self-similarity: max {endpoint_limits['self_sync_max']} "
                        f"texts. Cross-similarity: product of set sizes ≤ "
                        f"{endpoint_limits['cross_sync_product_max']}"
                    ),
                }
            else:
                return {
                    "max": endpoint_limits["async_max"],
                    "description": f"Max {endpoint_limits['async_max']} texts per set",
                }
        else:
            limit_key = f"{mode}_max"
            if limit_key in endpoint_limits:
                return {
                    "max": endpoint_limits[limit_key],
                    "description": (
                        f"Max {endpoint_limits[limit_key]} texts in {mode} mode"
                    ),
                }
            else:
                return {"error": f"Unknown mode '{mode}' for endpoint '{endpoint}'"}

    @staticmethod
    def suggest_optimization(
        endpoint: str, input_count: int, fast: Optional[bool] = None
    ) -> str:
        """
        Suggest optimization strategies when input limits are exceeded.

        Args:
            endpoint: The API endpoint name
            input_count: Number of inputs provided
            fast: Sync (True) or async (False/None) mode

        Returns:
            Optimization suggestion string
        """
        mode = "sync" if fast is True else "async"
        limit_info = ValidationHelper.get_limit_info(endpoint, mode)

        if "error" in limit_info:
            return limit_info["error"]

        suggestions = []

        if endpoint == "similarity" and mode == "sync":
            if "self_max" in limit_info and input_count > limit_info["self_max"]:
                suggestions.append(
                    f"• Switch to async mode (fast=False) to handle up to "
                    f"{ValidationLimits.SIMILARITY_ASYNC_MAX} texts"
                )
                suggestions.append(
                    f"• Split your {input_count} texts into chunks of "
                    f"{limit_info['self_max']} or fewer"
                )
        else:
            max_limit = limit_info.get("max", 0)
            if input_count > max_limit:
                if mode == "sync":
                    async_info = ValidationHelper.get_limit_info(endpoint, "async")
                    async_max = async_info.get("max", 0)
                    if input_count <= async_max:
                        suggestions.append(
                            f"• Switch to async mode (fast=False) to handle up to "
                            f"{async_max} texts"
                        )
                    else:
                        suggestions.append(
                            f"• Split your {input_count} texts into chunks of "
                            f"{async_max} or fewer"
                        )
                        suggestions.append(
                            "• Use async mode (fast=False) for larger batches"
                        )
                else:
                    chunk_size = max_limit
                    num_chunks = (input_count + chunk_size - 1) // chunk_size
                    suggestions.append(
                        f"• Split your {input_count} texts into {num_chunks} "
                        f"chunks of {chunk_size} or fewer"
                    )

                suggestions.append(
                    "• Consider using the batching utilities in pulse.core.batching"
                )

        if not suggestions:
            return "Input size is within limits."

        return "Suggestions to handle large input:\n" + "\n".join(suggestions)
