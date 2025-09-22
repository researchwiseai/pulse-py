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


# Comprehensive error documentation structure for all batching scenarios
ERROR_DOCUMENTATION = {
    "FAST_MODE_LIMIT_EXCEEDED": {
        "code": "BATCH_001",
        "category": "limit_exceeded",
        "severity": "user_error",
        "message_template": (
            "Fast mode supports up to {limit} texts. "
            "Use slow mode for larger datasets or reduce input size."
        ),
        "resolution_steps": [
            "Set fast=False to enable slow mode processing",
            "Reduce input size to {limit} or fewer texts",
            "Consider processing data in smaller batches",
        ],
        "context_fields": ["feature", "input_count", "limit", "mode"],
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
        "related_errors": ["BATCH_002"],
        "documentation_url": "/docs/batching-errors#batch-001",
    },
    "SLOW_MODE_LIMIT_EXCEEDED": {
        "code": "BATCH_002",
        "category": "limit_exceeded",
        "severity": "user_error",
        "message_template": (
            "Maximum input limit is {limit:,} texts. "
            "Please reduce your dataset size."
        ),
        "resolution_steps": [
            "Split dataset into chunks of {limit:,} or fewer texts",
            "Process data in multiple separate requests",
            "Consider data sampling or filtering strategies",
        ],
        "context_fields": ["feature", "input_count", "limit", "mode"],
        "examples": {
            "chunking": "for chunk in chunks(texts, {limit}): process_chunk(chunk)",
            "sampling": "sampled_texts = random.sample(texts, {limit})",
        },
        "related_errors": ["BATCH_001"],
        "documentation_url": "/docs/batching-errors#batch-002",
    },
    "BATCH_JOB_FAILED": {
        "code": "BATCH_003",
        "category": "batch_failed",
        "severity": "runtime_error",
        "message_template": (
            "Batch {batch_num} of {total_batches} failed: {error_message}"
        ),
        "resolution_steps": [
            "Check individual batch data for issues",
            "Retry with smaller batch sizes",
            "Verify network connectivity and API limits",
            "Review batch content for problematic inputs",
        ],
        "context_fields": [
            "feature",
            "batch_num",
            "total_batches",
            "error_message",
            "input_count",
        ],
        "examples": {
            "retry": (
                "try:\n    result = process_batch(batch)\n"
                "except BatchingError as e:\n    if e.batch_number:\n"
                "        retry_batch_with_smaller_size(e.batch_number)"
            ),
            "inspect": (
                "failed_batch_data = batches[error.batch_number - 1]\n"
                "print(f'Failed batch contains {len(failed_batch_data)} items')"
            ),
        },
        "related_errors": ["BATCH_006", "BATCH_007"],
        "documentation_url": "/docs/batching-errors#batch-003",
    },
    "CONCURRENCY_LIMIT_EXCEEDED": {
        "code": "BATCH_004",
        "category": "concurrency_error",
        "severity": "configuration_error",
        "message_template": (
            "Too many concurrent jobs: {current_jobs}. Maximum allowed: {max_jobs}"
        ),
        "resolution_steps": [
            "Reduce concurrent job limit in configuration",
            "Wait for existing jobs to complete before starting new ones",
            "Consider sequential processing for resource-constrained environments",
        ],
        "context_fields": ["current_jobs", "max_jobs", "feature"],
        "examples": {
            "config": (
                "config = BatchingConfig(max_concurrent_jobs={max_jobs})\n"
                "client = CoreClient(batching_config=config)"
            ),
            "sequential": (
                "# Process batches sequentially\n"
                "for batch in batches:\n    result = process_batch(batch)"
            ),
        },
        "related_errors": ["BATCH_007"],
        "documentation_url": "/docs/batching-errors#batch-004",
    },
    "CLUSTERING_AUTO_BATCH_TRIGGERED": {
        "code": "BATCH_005",
        "category": "informational",
        "severity": "info",
        "message_template": (
            "Input size {input_count} exceeds threshold {threshold}. "
            "Automatic batching enabled."
        ),
        "resolution_steps": [
            "This is informational - batching will proceed automatically",
            "Results will be reconstructed to match non-batched clustering",
            "Consider reducing input size if processing time is too long",
        ],
        "context_fields": ["input_count", "threshold", "feature"],
        "examples": {
            "reduce_size": (
                "# Reduce input size if needed\n"
                "if len(texts) > 500:\n    texts = texts[:500]"
            )
        },
        "related_errors": [],
        "documentation_url": "/docs/batching-errors#batch-005",
    },
    "BATCH_TIMEOUT": {
        "code": "BATCH_006",
        "category": "timeout_error",
        "severity": "runtime_error",
        "message_template": ("Batch {batch_num} timed out after {timeout} seconds"),
        "resolution_steps": [
            "Increase timeout_per_batch in BatchingConfig",
            "Reduce batch sizes to process faster",
            "Check network connectivity",
            "Verify API endpoint performance",
        ],
        "context_fields": ["batch_num", "timeout", "feature", "total_batches"],
        "examples": {
            "increase_timeout": (
                "config = BatchingConfig(timeout_per_batch=600.0)\n"
                "client = CoreClient(batching_config=config)"
            ),
            "smaller_batches": (
                "# Use smaller batch sizes\n"
                "config = BatchingConfig(default_batch_sizes={'sentiment': 1000})"
            ),
        },
        "related_errors": ["BATCH_003"],
        "documentation_url": "/docs/batching-errors#batch-006",
    },
    "RESOURCE_EXHAUSTION": {
        "code": "BATCH_007",
        "category": "resource_error",
        "severity": "system_error",
        "message_template": (
            "Resource exhaustion: {resource_type} usage too high ({current_usage})"
        ),
        "resolution_steps": [
            "Reduce concurrent jobs to lower resource usage",
            "Process smaller batches to reduce memory footprint",
            "Monitor resource usage during processing",
            "Consider upgrading system resources",
        ],
        "context_fields": ["resource_type", "current_usage", "feature"],
        "examples": {
            "reduce_concurrency": (
                "config = BatchingConfig(max_concurrent_jobs=2)\n"
                "client = CoreClient(batching_config=config)"
            ),
            "monitor": (
                "import psutil\n"
                "memory_percent = psutil.virtual_memory().percent\n"
                "if memory_percent > 80:\n    # Reduce batch size"
            ),
        },
        "related_errors": ["BATCH_004"],
        "documentation_url": "/docs/batching-errors#batch-007",
    },
    "BATCH_ORDER_CORRUPTION": {
        "code": "BATCH_008",
        "category": "data_integrity_error",
        "severity": "critical_error",
        "message_template": (
            "Batch result order corruption detected in {feature}. "
            "Expected {expected_count} results, got {actual_count}"
        ),
        "resolution_steps": [
            "This indicates a serious bug - please report to support",
            "Retry the operation with sequential processing",
            "Verify input data integrity",
            "Check for concurrent modifications to input data",
        ],
        "context_fields": ["feature", "expected_count", "actual_count", "batch_num"],
        "examples": {
            "sequential_fallback": (
                "# Fallback to sequential processing\n"
                "config = BatchingConfig(max_concurrent_jobs=1)\n"
                "client = CoreClient(batching_config=config)"
            )
        },
        "related_errors": [],
        "documentation_url": "/docs/batching-errors#batch-008",
    },
    "INVALID_BATCH_CONFIGURATION": {
        "code": "BATCH_009",
        "category": "configuration_error",
        "severity": "user_error",
        "message_template": (
            "Invalid batching configuration: {config_field} = "
            "{config_value}. {constraint}"
        ),
        "resolution_steps": [
            "Check BatchingConfig parameter values",
            "Ensure all configuration values are within valid ranges",
            "Refer to documentation for valid configuration options",
        ],
        "context_fields": ["config_field", "config_value", "constraint"],
        "examples": {
            "valid_config": (
                "config = BatchingConfig(\n"
                "    max_concurrent_jobs=5,  # 1-10\n"
                "    timeout_per_batch=300.0,  # > 0\n"
                "    default_batch_sizes={'sentiment': 2000}  # > 0\n"
                ")"
            )
        },
        "related_errors": ["BATCH_004"],
        "documentation_url": "/docs/batching-errors#batch-009",
    },
    "BATCH_RESULT_AGGREGATION_FAILED": {
        "code": "BATCH_010",
        "category": "data_processing_error",
        "severity": "runtime_error",
        "message_template": (
            "Failed to aggregate results from {completed_batches} of "
            "{total_batches} batches: {error_message}"
        ),
        "resolution_steps": [
            "Check individual batch results for consistency",
            "Verify that all batches completed successfully",
            "Retry with smaller batch sizes",
            "Check for memory issues during aggregation",
        ],
        "context_fields": [
            "completed_batches",
            "total_batches",
            "error_message",
            "feature",
        ],
        "examples": {
            "debug_aggregation": (
                "# Debug batch results\n"
                "for i, batch_result in enumerate(batch_results):\n"
                "    print(f'Batch {i}: {len(batch_result)} items')"
            )
        },
        "related_errors": ["BATCH_003", "BATCH_007"],
        "documentation_url": "/docs/batching-errors#batch-010",
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
    """Specialized error for batching operations with comprehensive structured info."""

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
        context: Optional[Dict[str, Any]] = None,
        alternatives: Optional[List[str]] = None,
        severity: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs,
    ):
        self.feature = feature
        self.input_count = input_count
        self.suggested_action = suggested_action
        self.error_code = error_code
        self.batch_number = batch_number
        self.total_batches = total_batches
        self.resolution_steps = resolution_steps or []
        self.context = context or {}
        self.alternatives = alternatives or []
        self.severity = severity or "user_error"
        self.category = category or "unknown"

        # Add error-specific context
        self.context.update(
            {
                "feature": feature,
                "input_count": input_count,
                "limit": limit,
                "error_code": error_code,
                "batch_number": batch_number,
                "total_batches": total_batches,
            }
        )

        super().__init__(message, limit=limit, endpoint=feature, **kwargs)

    def get_structured_info(self) -> Dict[str, Any]:
        """Get comprehensive structured error information for programmatic handling."""
        return {
            "error_code": self.error_code,
            "category": self.category,
            "severity": self.severity,
            "feature": self.feature,
            "input_count": self.input_count,
            "limit": self.limit,
            "mode": self.mode,
            "suggested_action": self.suggested_action,
            "alternatives": self.alternatives,
            "batch_number": self.batch_number,
            "total_batches": self.total_batches,
            "resolution_steps": self.resolution_steps,
            "context": self.context,
            "timestamp": self._get_timestamp(),
            "is_retryable": self._is_retryable(),
            "expected_fix_time": self._get_expected_fix_time(),
        }

    def _get_timestamp(self) -> str:
        """Get current timestamp for error tracking."""
        from datetime import datetime

        return datetime.utcnow().isoformat() + "Z"

    def _is_retryable(self) -> bool:
        """Determine if this error is retryable."""
        retryable_codes = ["BATCH_003", "BATCH_006", "BATCH_007", "BATCH_010"]
        return self.error_code in retryable_codes

    def _get_expected_fix_time(self) -> str:
        """Get expected time to fix based on error type."""
        if self.error_code in ["BATCH_001", "BATCH_002"]:
            return "immediate"  # User can fix immediately
        elif self.error_code in ["BATCH_003", "BATCH_006"]:
            return "short"  # Retry or configuration change
        elif self.error_code in ["BATCH_004", "BATCH_007"]:
            return "medium"  # Resource or configuration adjustment
        else:
            return "unknown"

    def get_error_context_for_field(self, field: str) -> Any:
        """Get context value for a specific field."""
        return self.context.get(field)

    def add_context(self, key: str, value: Any) -> None:
        """Add additional context information."""
        self.context[key] = value

    def get_retry_strategy(self) -> Optional[Dict[str, Any]]:
        """Get retry strategy if error is retryable."""
        if not self._is_retryable():
            return None

        if self.error_code == "BATCH_003":  # Batch job failed
            return {
                "strategy": "exponential_backoff",
                "max_retries": 3,
                "base_delay": 1.0,
                "max_delay": 60.0,
                "modify_request": "reduce_batch_size",
            }
        elif self.error_code == "BATCH_006":  # Timeout
            return {
                "strategy": "linear_backoff",
                "max_retries": 2,
                "base_delay": 5.0,
                "modify_request": "increase_timeout",
            }
        elif self.error_code == "BATCH_007":  # Resource exhaustion
            return {
                "strategy": "wait_and_retry",
                "max_retries": 2,
                "base_delay": 30.0,
                "modify_request": "reduce_concurrency",
            }

        return {"strategy": "simple_retry", "max_retries": 1, "base_delay": 1.0}

    def format_for_logging(self) -> Dict[str, Any]:
        """Format error information for structured logging."""
        return {
            "error_type": "BatchingError",
            "error_code": self.error_code,
            "category": self.category,
            "severity": self.severity,
            "feature": self.feature,
            "message": str(self),
            "context": self.context,
            "is_retryable": self._is_retryable(),
            "timestamp": self._get_timestamp(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "type": "BatchingError",
            "message": str(self),
            "structured_info": self.get_structured_info(),
            "retry_strategy": self.get_retry_strategy(),
        }


class BatchingErrorCategory:
    """Error category classification for different batching error types."""

    LIMIT_EXCEEDED = "limit_exceeded"
    BATCH_FAILED = "batch_failed"
    CONCURRENCY_ERROR = "concurrency_error"
    TIMEOUT_ERROR = "timeout_error"
    NETWORK_ERROR = "network_error"
    RESOURCE_ERROR = "resource_error"
    CONFIGURATION_ERROR = "configuration_error"


class ErrorRecoveryStrategy:
    """Defines recovery strategies for different error types with specific guidance."""

    @staticmethod
    def handle_limit_exceeded(error: BatchingError) -> Dict[str, Any]:
        """Provide specific guidance for limit exceeded errors."""
        if error.mode == "fast":
            return {
                "primary_action": "Switch to slow mode",
                "alternative_actions": [
                    f"Reduce input size to {error.limit} or fewer texts",
                    "Process data in smaller chunks manually",
                ],
                "code_example": (
                    f"# Switch to slow mode\n"
                    f"result = client.{error.feature}(texts, fast=False)"
                ),
                "expected_outcome": ("Automatic batching will handle large datasets"),
            }
        else:
            return {
                "primary_action": "Split dataset into smaller chunks",
                "alternative_actions": [
                    "Filter or sample your data",
                    "Process data in multiple separate requests",
                ],
                "code_example": (
                    f"# Process in chunks\n"
                    f"for chunk in chunks(texts, {error.limit}):\n"
                    f"    result = client.{error.feature}(chunk)"
                ),
                "expected_outcome": "Each chunk will be processed successfully",
            }

    @staticmethod
    def handle_batch_failure(error: BatchingError) -> Dict[str, Any]:
        """Provide guidance for individual batch failures."""
        return {
            "primary_action": "Inspect failed batch data",
            "alternative_actions": [
                "Retry with smaller batch sizes",
                "Check for problematic input texts",
                "Verify network connectivity",
            ],
            "code_example": (
                "# Retry with smaller batches\n"
                "try:\n    result = process_batch(batch)\n"
                "except BatchingError as e:\n    if e.batch_number:\n"
                "        retry_batch_with_smaller_size(e.batch_number)"
            ),
            "expected_outcome": ("Individual batches will process successfully"),
        }

    @staticmethod
    def handle_concurrency_error(error: BatchingError) -> Dict[str, Any]:
        """Provide guidance for concurrency errors."""
        return {
            "primary_action": "Reduce concurrent job limit",
            "alternative_actions": [
                "Wait for existing jobs to complete",
                "Use sequential processing",
                "Increase system resources",
            ],
            "code_example": (
                f"# Configure lower concurrency\n"
                f"config = BatchingConfig(max_concurrent_jobs={error.limit})\n"
                f"client = CoreClient(batching_config=config)"
            ),
            "expected_outcome": "Jobs will run within concurrency limits",
        }

    @staticmethod
    def handle_timeout_error(error: Exception) -> Dict[str, Any]:
        """Provide guidance for timeout errors."""
        return {
            "primary_action": "Increase timeout duration",
            "alternative_actions": [
                "Reduce batch sizes",
                "Check network connectivity",
                "Retry with exponential backoff",
            ],
            "code_example": (
                "# Increase timeout\n"
                "config = BatchingConfig(timeout_per_batch=600.0)\n"
                "client = CoreClient(batching_config=config)"
            ),
            "expected_outcome": "Batches will have more time to complete",
        }

    @staticmethod
    def handle_network_error(error: Exception) -> Dict[str, Any]:
        """Provide guidance for network errors."""
        return {
            "primary_action": "Check network connectivity",
            "alternative_actions": [
                "Retry with exponential backoff",
                "Reduce concurrent requests",
                "Check API endpoint status",
            ],
            "code_example": (
                "# Retry with backoff\n"
                "import time\n"
                "for attempt in range(3):\n"
                "    try:\n        result = client.process(data)\n        break\n"
                "    except NetworkError:\n        time.sleep(2 ** attempt)"
            ),
            "expected_outcome": "Network issues will be resolved through retries",
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
            category=error_info["category"],
            severity=error_info["severity"],
            mode="fast",
            resolution_steps=resolution_steps,
            context={"limit": limit, "input_count": input_count},
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
            category=error_info["category"],
            severity=error_info["severity"],
            mode="slow",
            resolution_steps=resolution_steps,
            context={"limit": limit, "input_count": input_count},
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
            category=error_info["category"],
            severity=error_info["severity"],
            batch_number=batch_num,
            total_batches=total_batches,
            resolution_steps=error_info["resolution_steps"],
            context={
                "batch_num": batch_num,
                "total_batches": total_batches,
                "error_message": error_message,
            },
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
            category=error_info["category"],
            severity=error_info["severity"],
            resolution_steps=error_info["resolution_steps"],
            context={"current_jobs": current_jobs, "max_jobs": max_jobs},
        )

    @staticmethod
    def create_timeout_error(
        feature: str, timeout_seconds: float, batch_num: Optional[int] = None
    ) -> BatchingError:
        """Create a timeout error with structured information."""
        error_info = ERROR_DOCUMENTATION["BATCH_TIMEOUT"]
        message = error_info["message_template"].format(
            batch_num=batch_num or "operation", timeout=timeout_seconds
        )

        return BatchingError(
            message=message,
            feature=feature,
            input_count=0,
            limit=int(timeout_seconds),
            suggested_action="Increase timeout or reduce batch size",
            error_code=error_info["code"],
            category=error_info["category"],
            severity=error_info["severity"],
            batch_number=batch_num,
            resolution_steps=error_info["resolution_steps"],
            context={"timeout": timeout_seconds, "batch_num": batch_num},
        )

    @staticmethod
    def create_resource_error(
        feature: str, resource_type: str, current_usage: str
    ) -> BatchingError:
        """Create a resource exhaustion error with structured information."""
        error_info = ERROR_DOCUMENTATION["RESOURCE_EXHAUSTION"]
        message = error_info["message_template"].format(
            resource_type=resource_type, current_usage=current_usage
        )

        return BatchingError(
            message=message,
            feature=feature,
            input_count=0,
            limit=0,
            suggested_action=(
                f"Reduce {resource_type} usage or increase system resources"
            ),
            error_code=error_info["code"],
            category=error_info["category"],
            severity=error_info["severity"],
            resolution_steps=error_info["resolution_steps"],
            context={"resource_type": resource_type, "current_usage": current_usage},
        )

    @staticmethod
    def create_order_corruption_error(
        feature: str,
        expected_count: int,
        actual_count: int,
        batch_num: Optional[int] = None,
    ) -> BatchingError:
        """Create a batch result order corruption error."""
        error_info = ERROR_DOCUMENTATION["BATCH_ORDER_CORRUPTION"]
        message = error_info["message_template"].format(
            feature=feature, expected_count=expected_count, actual_count=actual_count
        )

        return BatchingError(
            message=message,
            feature=feature,
            input_count=expected_count,
            limit=actual_count,
            suggested_action="Report to support and retry with sequential processing",
            error_code=error_info["code"],
            category=error_info["category"],
            severity=error_info["severity"],
            batch_number=batch_num,
            resolution_steps=error_info["resolution_steps"],
            context={
                "expected_count": expected_count,
                "actual_count": actual_count,
                "batch_num": batch_num,
            },
        )

    @staticmethod
    def create_configuration_error(
        config_field: str, config_value: Any, constraint: str
    ) -> BatchingError:
        """Create an invalid configuration error."""
        error_info = ERROR_DOCUMENTATION["INVALID_BATCH_CONFIGURATION"]
        message = error_info["message_template"].format(
            config_field=config_field, config_value=config_value, constraint=constraint
        )

        return BatchingError(
            message=message,
            feature="configuration",
            input_count=0,
            limit=0,
            suggested_action=f"Fix {config_field} configuration value",
            error_code=error_info["code"],
            category=error_info["category"],
            severity=error_info["severity"],
            resolution_steps=error_info["resolution_steps"],
            context={
                "config_field": config_field,
                "config_value": config_value,
                "constraint": constraint,
            },
        )

    @staticmethod
    def create_aggregation_error(
        feature: str, completed_batches: int, total_batches: int, error_message: str
    ) -> BatchingError:
        """Create a result aggregation failure error."""
        error_info = ERROR_DOCUMENTATION["BATCH_RESULT_AGGREGATION_FAILED"]
        message = error_info["message_template"].format(
            completed_batches=completed_batches,
            total_batches=total_batches,
            error_message=error_message,
        )

        return BatchingError(
            message=message,
            feature=feature,
            input_count=completed_batches,
            limit=total_batches,
            suggested_action="Check batch results and retry with smaller batches",
            error_code=error_info["code"],
            category=error_info["category"],
            severity=error_info["severity"],
            total_batches=total_batches,
            resolution_steps=error_info["resolution_steps"],
            context={
                "completed_batches": completed_batches,
                "total_batches": total_batches,
                "error_message": error_message,
            },
        )

    @staticmethod
    def create_clustering_info(input_count: int, threshold: int) -> BatchingError:
        """Create an informational message for clustering auto-batching."""
        error_info = ERROR_DOCUMENTATION["CLUSTERING_AUTO_BATCH_TRIGGERED"]
        message = error_info["message_template"].format(
            input_count=input_count, threshold=threshold
        )

        return BatchingError(
            message=message,
            feature="clustering",
            input_count=input_count,
            limit=threshold,
            suggested_action="Batching will proceed automatically",
            error_code=error_info["code"],
            category=error_info["category"],
            severity=error_info["severity"],
            resolution_steps=error_info["resolution_steps"],
            context={"input_count": input_count, "threshold": threshold},
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
    def classify_error(error: Exception) -> str:
        """Classify an error into a specific category."""
        if isinstance(error, BatchingError):
            if error.error_code in ["BATCH_001", "BATCH_002"]:
                return BatchingErrorCategory.LIMIT_EXCEEDED
            elif error.error_code == "BATCH_003":
                return BatchingErrorCategory.BATCH_FAILED
            elif error.error_code == "BATCH_004":
                return BatchingErrorCategory.CONCURRENCY_ERROR
            elif error.error_code == "BATCH_006":
                return BatchingErrorCategory.TIMEOUT_ERROR
            elif error.error_code == "BATCH_007":
                return BatchingErrorCategory.RESOURCE_ERROR

        # Import here to avoid circular imports
        from .exceptions import NetworkError, TimeoutError

        if isinstance(error, NetworkError):
            return BatchingErrorCategory.NETWORK_ERROR
        elif isinstance(error, TimeoutError):
            return BatchingErrorCategory.TIMEOUT_ERROR
        elif isinstance(error, ValueError) and "configuration" in str(error).lower():
            return BatchingErrorCategory.CONFIGURATION_ERROR

        return "unknown"

    @staticmethod
    def suggest_fix(
        error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Suggest specific fixes based on error and context."""
        category = BatchingErrorHelper.classify_error(error)

        if isinstance(error, BatchingError):
            if category == BatchingErrorCategory.LIMIT_EXCEEDED:
                recovery = ErrorRecoveryStrategy.handle_limit_exceeded(error)
            elif category == BatchingErrorCategory.BATCH_FAILED:
                recovery = ErrorRecoveryStrategy.handle_batch_failure(error)
            elif category == BatchingErrorCategory.CONCURRENCY_ERROR:
                recovery = ErrorRecoveryStrategy.handle_concurrency_error(error)
            else:
                return ["Check error message and documentation"]

            suggestions = [recovery["primary_action"]]
            suggestions.extend(recovery["alternative_actions"])
            return suggestions

        elif category == BatchingErrorCategory.TIMEOUT_ERROR:
            recovery = ErrorRecoveryStrategy.handle_timeout_error(error)
            suggestions = [recovery["primary_action"]]
            suggestions.extend(recovery["alternative_actions"])
            return suggestions

        elif category == BatchingErrorCategory.NETWORK_ERROR:
            recovery = ErrorRecoveryStrategy.handle_network_error(error)
            suggestions = [recovery["primary_action"]]
            suggestions.extend(recovery["alternative_actions"])
            return suggestions

        return ["Check error message and documentation"]

    @staticmethod
    def diagnose_error(error: Exception) -> Dict[str, Any]:
        """Analyze error and provide comprehensive diagnostic information."""
        category = BatchingErrorHelper.classify_error(error)

        base_info = {
            "error_type": type(error).__name__,
            "category": category,
            "message": str(error),
            "suggestions": BatchingErrorHelper.suggest_fix(error),
        }

        if isinstance(error, BatchingError):
            structured_info = error.get_structured_info()
            recovery_strategy = None

            if category == BatchingErrorCategory.LIMIT_EXCEEDED:
                recovery_strategy = ErrorRecoveryStrategy.handle_limit_exceeded(error)
            elif category == BatchingErrorCategory.BATCH_FAILED:
                recovery_strategy = ErrorRecoveryStrategy.handle_batch_failure(error)
            elif category == BatchingErrorCategory.CONCURRENCY_ERROR:
                recovery_strategy = ErrorRecoveryStrategy.handle_concurrency_error(
                    error
                )

            base_info.update(
                {
                    "structured_info": structured_info,
                    "example_code": BatchingErrorHelper.generate_example_code(
                        error.feature, error.error_code or ""
                    ),
                    "documentation": BatchingErrorHelper.get_error_documentation(
                        error.error_code or ""
                    ),
                    "recovery_strategy": recovery_strategy,
                }
            )

        elif isinstance(error, PulseValidationError):
            base_info.update(
                {
                    "endpoint": error.endpoint,
                    "mode": error.mode,
                    "field": error.field,
                    "value": error.value,
                    "limit": error.limit,
                }
            )

        return base_info

    @staticmethod
    def create_structured_error_response(
        error_code: str, context: Dict[str, Any], feature: str = "unknown"
    ) -> Dict[str, Any]:
        """Create a structured error response for programmatic handling."""
        error_info = BatchingErrorHelper.get_error_documentation(error_code)
        if not error_info:
            return {
                "error_code": error_code,
                "message": "Unknown error",
                "category": "unknown",
                "severity": "unknown",
                "context": context,
                "resolution_steps": [],
                "alternatives": [],
            }

        # Format message with context
        try:
            message = error_info["message_template"].format(**context)
        except KeyError:
            message = error_info["message_template"]

        # Format resolution steps with context
        resolution_steps = []
        for step in error_info["resolution_steps"]:
            try:
                formatted_step = step.format(**context)
                resolution_steps.append(formatted_step)
            except KeyError:
                resolution_steps.append(step)

        return {
            "error_code": error_code,
            "message": message,
            "category": error_info["category"],
            "severity": error_info["severity"],
            "feature": feature,
            "context": context,
            "resolution_steps": resolution_steps,
            "alternatives": list(context.get("alternatives", [])),
            "examples": error_info.get("examples", {}),
            "related_errors": error_info.get("related_errors", []),
            "documentation_url": error_info.get("documentation_url", ""),
            "is_retryable": error_code
            in ["BATCH_003", "BATCH_006", "BATCH_007", "BATCH_010"],
            "expected_fix_time": BatchingErrorHelper._get_fix_time_for_code(error_code),
        }

    @staticmethod
    def _get_fix_time_for_code(error_code: str) -> str:
        """Get expected fix time for an error code."""
        immediate_fix = ["BATCH_001", "BATCH_002", "BATCH_009"]
        short_fix = ["BATCH_003", "BATCH_006"]
        medium_fix = ["BATCH_004", "BATCH_007"]
        long_fix = ["BATCH_008", "BATCH_010"]

        if error_code in immediate_fix:
            return "immediate"
        elif error_code in short_fix:
            return "short"
        elif error_code in medium_fix:
            return "medium"
        elif error_code in long_fix:
            return "long"
        else:
            return "unknown"

    @staticmethod
    def validate_error_context(error_code: str, context: Dict[str, Any]) -> List[str]:
        """Validate that context contains required fields for error code."""
        error_info = BatchingErrorHelper.get_error_documentation(error_code)
        if not error_info:
            return ["Unknown error code"]

        required_fields = error_info.get("context_fields", [])
        missing_fields = []

        for field in required_fields:
            if field not in context:
                missing_fields.append(field)

        return missing_fields

    @staticmethod
    def get_all_error_codes() -> List[str]:
        """Get list of all available error codes."""
        return [info["code"] for info in ERROR_DOCUMENTATION.values()]

    @staticmethod
    def get_errors_by_category(category: str) -> List[Dict[str, Any]]:
        """Get all errors in a specific category."""
        return [
            {"code": info["code"], "message_template": info["message_template"]}
            for info in ERROR_DOCUMENTATION.values()
            if info.get("category") == category
        ]

    @staticmethod
    def get_errors_by_severity(severity: str) -> List[Dict[str, Any]]:
        """Get all errors with a specific severity level."""
        return [
            {"code": info["code"], "message_template": info["message_template"]}
            for info in ERROR_DOCUMENTATION.values()
            if info.get("severity") == severity
        ]

    @staticmethod
    def format_error_for_user(error: Exception) -> str:
        """Format error information in a user-friendly way."""
        diagnosis = BatchingErrorHelper.diagnose_error(error)

        lines = [
            f"Error: {diagnosis['message']}",
            f"Category: {diagnosis['category']}",
            "",
            "Suggested fixes:",
        ]

        for i, suggestion in enumerate(diagnosis["suggestions"], 1):
            lines.append(f"  {i}. {suggestion}")

        if isinstance(error, BatchingError) and diagnosis.get("recovery_strategy"):
            recovery = diagnosis["recovery_strategy"]
            lines.extend(
                [
                    "",
                    "Recovery strategy:",
                    f"  Primary action: {recovery['primary_action']}",
                    f"  Expected outcome: {recovery['expected_outcome']}",
                    "",
                    "Example code:",
                    recovery["code_example"],
                ]
            )

        return "\n".join(lines)

    @staticmethod
    def format_error_for_api(error: Exception) -> Dict[str, Any]:
        """Format error for API response with consistent structure."""
        if isinstance(error, BatchingError):
            return {
                "error": {
                    "code": error.error_code,
                    "message": str(error),
                    "category": error.category,
                    "severity": error.severity,
                    "details": error.get_structured_info(),
                    "retry_strategy": error.get_retry_strategy(),
                }
            }
        else:
            return {
                "error": {
                    "code": "UNKNOWN",
                    "message": str(error),
                    "category": "unknown",
                    "severity": "unknown",
                    "details": {"type": type(error).__name__},
                }
            }


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
