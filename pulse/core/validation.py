"""Input validation utilities for Pulse API endpoints."""

from typing import List, Optional, Any, Dict
from pydantic import ValidationError


class ValidationLimits:
    """Input validation limits for different endpoints and modes."""

    # Embeddings limits
    EMBEDDINGS_SYNC_MAX = 200
    EMBEDDINGS_ASYNC_MAX = 5000

    # Similarity limits
    SIMILARITY_SELF_SYNC_MAX = 500
    SIMILARITY_CROSS_SYNC_PRODUCT_MAX = 20000
    SIMILARITY_ASYNC_MAX = 44721

    # Themes limits
    THEMES_SYNC_MAX = 200
    THEMES_ASYNC_MAX = 500

    # Clustering limits
    CLUSTERING_SYNC_MAX = 500
    CLUSTERING_ASYNC_MAX = 44721

    # Sentiment limits
    SENTIMENT_SYNC_MAX = 200
    SENTIMENT_ASYNC_MAX = 5000

    # Extractions limits
    EXTRACTIONS_SYNC_MAX = 200
    EXTRACTIONS_ASYNC_MAX = 5000


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


def validate_embeddings_input(inputs: List[Any], fast: Optional[bool] = None) -> None:
    """
    Validate embeddings input according to sync/async limits.

    Args:
        inputs: List of input texts
        fast: Sync (True) or async (False/None) mode

    Raises:
        PulseValidationError: If validation fails
    """
    if not inputs:
        raise PulseValidationError(
            "Input list cannot be empty", field="inputs", endpoint="embeddings"
        )

    input_count = len(inputs)

    if fast is True:
        # Sync mode
        if input_count > ValidationLimits.EMBEDDINGS_SYNC_MAX:
            raise PulseValidationError(
                f"Too many inputs for sync mode: {input_count}",
                field="inputs",
                value=input_count,
                limit=ValidationLimits.EMBEDDINGS_SYNC_MAX,
                endpoint="embeddings",
                mode="sync",
            )
    else:
        # Async mode (fast=False or fast=None)
        if input_count > ValidationLimits.EMBEDDINGS_ASYNC_MAX:
            raise PulseValidationError(
                f"Too many inputs for async mode: {input_count}",
                field="inputs",
                value=input_count,
                limit=ValidationLimits.EMBEDDINGS_ASYNC_MAX,
                endpoint="embeddings",
                mode="async",
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
        # Async mode
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
        PulseValidationError: If validation fails
    """
    if not inputs:
        raise PulseValidationError(
            "Input list cannot be empty", field="inputs", endpoint="sentiment"
        )

    input_count = len(inputs)

    if fast is True:
        # Sync mode
        if input_count > ValidationLimits.SENTIMENT_SYNC_MAX:
            raise PulseValidationError(
                f"Too many inputs for sync mode: {input_count}",
                field="inputs",
                value=input_count,
                limit=ValidationLimits.SENTIMENT_SYNC_MAX,
                endpoint="sentiment",
                mode="sync",
            )
    else:
        # Async mode
        if input_count > ValidationLimits.SENTIMENT_ASYNC_MAX:
            raise PulseValidationError(
                f"Too many inputs for async mode: {input_count}",
                field="inputs",
                value=input_count,
                limit=ValidationLimits.SENTIMENT_ASYNC_MAX,
                endpoint="sentiment",
                mode="async",
            )


def validate_extractions_input(inputs: List[str], fast: Optional[bool] = None) -> None:
    """
    Validate extractions input according to sync/async limits.

    Args:
        inputs: List of input texts
        fast: Sync (True) or async (False/None) mode

    Raises:
        PulseValidationError: If validation fails
    """
    if not inputs:
        raise PulseValidationError(
            "Input list cannot be empty", field="inputs", endpoint="extractions"
        )

    input_count = len(inputs)

    if fast is True:
        # Sync mode
        if input_count > ValidationLimits.EXTRACTIONS_SYNC_MAX:
            raise PulseValidationError(
                f"Too many inputs for sync mode: {input_count}",
                field="inputs",
                value=input_count,
                limit=ValidationLimits.EXTRACTIONS_SYNC_MAX,
                endpoint="extractions",
                mode="sync",
            )
    else:
        # Async mode
        if input_count > ValidationLimits.EXTRACTIONS_ASYNC_MAX:
            raise PulseValidationError(
                f"Too many inputs for async mode: {input_count}",
                field="inputs",
                value=input_count,
                limit=ValidationLimits.EXTRACTIONS_ASYNC_MAX,
                endpoint="extractions",
                mode="async",
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


def validate_before_request(endpoint: str, **kwargs) -> None:
    """
    Convenience function to validate a request before sending to API.

    Args:
        endpoint: The API endpoint name
        **kwargs: Request parameters to validate

    Raises:
        PulseValidationError: If validation fails with user-friendly message
    """
    try:
        ValidationHelper.validate_request(endpoint, **kwargs)
    except ValidationError as e:
        # Enhance error message with optimization suggestions
        if "Too many inputs" in e.message:
            input_count = len(kwargs.get("inputs", kwargs.get("set", [])))
            fast = kwargs.get("fast")
            suggestions = ValidationHelper.suggest_optimization(
                endpoint, input_count, fast
            )
            enhanced_message = f"{e.message}\n\n{suggestions}"
            raise PulseValidationError(
                enhanced_message,
                field=e.field,
                value=e.value,
                limit=e.limit,
                endpoint=e.endpoint,
                mode=e.mode,
            ) from e
        else:
            raise
