"""Exceptions for Pulse Client API errors."""

from __future__ import annotations

from typing import Any, Optional

import httpx


class PulseAPIError(Exception):
    """Represents an error returned by the Pulse API."""

    def __init__(self, response: httpx.Response) -> None:
        self.status: int = response.status_code
        self.code: Optional[str] = None
        self.message: Optional[str] = None
        # Capture relevant headers for diagnostics (esp. AWS API Gateway)
        # Note: httpx.Headers is case-insensitive
        self.headers = dict(response.headers)
        # Common AWS API Gateway headers that help diagnose 401s
        self.aws_www_authenticate: Optional[str] = response.headers.get(
            "www-authenticate"
        ) or response.headers.get("x-amzn-remapped-www-authenticate")
        self.aws_request_id: Optional[str] = (
            response.headers.get("apigw-requestid")
            or response.headers.get("x-amzn-requestid")
            or response.headers.get("x-amz-apigw-id")
        )
        self.aws_error_type: Optional[str] = response.headers.get(
            "x-amzn-errortype"
        ) or response.headers.get("x-amzn-ErrorType")
        try:
            body: Any = response.json()
        except ValueError:
            body = response.text

        if isinstance(body, dict):
            self.code = body.get("code")
            self.message = body.get("message") or response.reason_phrase
            self.body = body
        else:
            self.body = body
            if isinstance(body, str):
                self.message = body
            else:
                self.message = response.reason_phrase

        # Enrich 401 Unauthorized errors with AWS API Gateway hints, when present.
        if self.status == 401:
            hints: list[str] = []
            if self.aws_www_authenticate:
                hints.append(f"auth={self.aws_www_authenticate}")
            if self.aws_error_type:
                hints.append(f"errorType={self.aws_error_type}")
            if self.aws_request_id:
                hints.append(f"requestId={self.aws_request_id}")
            if hints:
                # Preserve original message and append concise guidance.
                base_msg = self.message or "Unauthorized"
                self.message = f"{base_msg} | AWS API Gateway hint: " + ", ".join(hints)

        super().__init__(str(self))

    def __str__(self) -> str:  # pragma: no cover - simple formatting
        parts = [f"{self.status}"]
        if self.code is not None:
            parts.append(str(self.code))
        msg = self.message or ""
        return f"Pulse API Error {' '.join(parts)}: {msg}"

    @property
    def is_transient(self) -> bool:
        """Return True if this error is likely transient and should be retried."""
        # Rate limiting and server errors are typically transient
        if self.status == 429:  # Too Many Requests
            return True
        if 500 <= self.status < 600:  # Server errors
            return True
        # Token expiry is transient (SDK handles refresh)
        if self.status == 401 and self.message and "expired" in self.message.lower():
            return True
        return False

    @property
    def is_permanent(self) -> bool:
        """Return True if this error is permanent and should not be retried."""
        return not self.is_transient

    @property
    def error_category(self) -> str:
        """Return the error category for this exception."""
        if self.status == 401:
            return "authentication"
        elif self.status == 403:
            return "authorization"
        elif self.status == 429:
            return "rate_limiting"
        elif 400 <= self.status < 500:
            return "client_error"
        elif 500 <= self.status < 600:
            return "server_error"
        else:
            return "unknown"

    @property
    def recovery_hint(self) -> str:
        """Return a recovery hint for this error."""
        if self.status == 401:
            if self.message and "expired" in self.message.lower():
                return "Token expired - SDK will automatically refresh"
            return (
                "Check your PULSE_CLIENT_ID and PULSE_CLIENT_SECRET "
                "environment variables"
            )
        elif self.status == 403:
            return "Check your account permissions and subscription status"
        elif self.status == 429:
            retry_after = self.headers.get("retry-after", "60")
            return f"Rate limited - wait {retry_after} seconds before retrying"
        elif self.status == 400:
            return "Check your request data format and parameters"
        elif self.status == 422:
            return "Validate your input data - some fields may be invalid"
        elif 500 <= self.status < 600:
            return "Server error - retry with exponential backoff"
        else:
            return "Check the error message and API documentation"


class TimeoutError(Exception):
    """Error thrown when an HTTP request times out."""

    def __init__(self, url: str, timeout: float) -> None:
        super().__init__(f"Request to {url} timed out after {timeout}ms")
        self.url = url
        self.timeout = timeout


class NetworkError(Exception):
    """Error thrown when a network error occurs during a request."""

    def __init__(self, url: str, cause: Exception) -> None:
        super().__init__(f"Network error while requesting {url}: {cause}")
        self.url = url
        self.cause = cause

    @property
    def is_transient(self) -> bool:
        """Network errors are typically transient."""
        return True

    @property
    def error_category(self) -> str:
        """Return the error category."""
        return "network"

    @property
    def recovery_hint(self) -> str:
        """Return a recovery hint."""
        return "Check network connectivity and retry with exponential backoff"


# Error severity levels
class ErrorSeverity:
    """Error severity classification."""

    TRANSIENT = "transient"
    PERMANENT = "permanent"
    UNKNOWN = "unknown"


# Error categories
class ErrorCategory:
    """Error category classification."""

    NETWORK = "network"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMITING = "rate_limiting"
    CLIENT_ERROR = "client_error"
    SERVER_ERROR = "server_error"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


def classify_error(error: Exception) -> tuple[str, str, str]:
    """
    Classify an error and return (category, severity, recovery_hint).

    Args:
        error: The exception to classify

    Returns:
        Tuple of (category, severity, recovery_hint)
    """
    if isinstance(error, PulseAPIError):
        category = error.error_category
        severity = (
            ErrorSeverity.TRANSIENT if error.is_transient else ErrorSeverity.PERMANENT
        )
        hint = error.recovery_hint
    elif isinstance(error, NetworkError):
        category = error.error_category
        severity = ErrorSeverity.TRANSIENT
        hint = error.recovery_hint
    elif isinstance(error, TimeoutError):
        category = ErrorCategory.TIMEOUT
        severity = ErrorSeverity.TRANSIENT
        hint = "Increase timeout or break request into smaller chunks"
    elif isinstance(error, ValueError) and "Client Secret" in str(error):
        category = ErrorCategory.CONFIGURATION
        severity = ErrorSeverity.PERMANENT
        hint = "Set PULSE_CLIENT_SECRET environment variable"
    else:
        category = ErrorCategory.UNKNOWN
        severity = ErrorSeverity.UNKNOWN
        hint = "Check error message and documentation"

    return category, severity, hint


def should_retry_error(error: Exception) -> bool:
    """
    Determine if an error should be retried.

    Args:
        error: The exception to check

    Returns:
        True if the error should be retried
    """
    category, severity, _ = classify_error(error)
    return severity == ErrorSeverity.TRANSIENT
