"""Tests for error recovery functionality."""

from unittest.mock import Mock
import httpx

from pulse.core.exceptions import (
    PulseAPIError,
    NetworkError,
    TimeoutError,
    classify_error,
    should_retry_error,
    ErrorSeverity,
)


class TestPulseAPIError:
    """Test PulseAPIError classification and recovery hints."""

    def test_transient_rate_limiting_error(self):
        """Test that 429 errors are classified as transient."""
        response = Mock(spec=httpx.Response)
        response.status_code = 429
        response.headers = {"retry-after": "60"}
        response.reason_phrase = "Too Many Requests"
        response.json.return_value = {"message": "Rate limit exceeded"}

        error = PulseAPIError(response)

        assert error.is_transient is True
        assert error.is_permanent is False
        assert error.error_category == "rate_limiting"
        assert "wait 60 seconds" in error.recovery_hint

    def test_transient_server_error(self):
        """Test that 5xx errors are classified as transient."""
        response = Mock(spec=httpx.Response)
        response.status_code = 503
        response.headers = {}
        response.reason_phrase = "Service Unavailable"
        response.json.return_value = {"message": "Service temporarily unavailable"}

        error = PulseAPIError(response)

        assert error.is_transient is True
        assert error.error_category == "server_error"
        assert "retry with exponential backoff" in error.recovery_hint

    def test_permanent_auth_error(self):
        """Test that 401 errors (non-expired) are classified as permanent."""
        response = Mock(spec=httpx.Response)
        response.status_code = 401
        response.headers = {}
        response.reason_phrase = "Unauthorized"
        response.json.return_value = {"message": "Invalid credentials"}

        error = PulseAPIError(response)

        assert error.is_permanent is True
        assert error.is_transient is False
        assert error.error_category == "authentication"
        assert "PULSE_CLIENT_ID" in error.recovery_hint

    def test_transient_expired_token_error(self):
        """Test that expired token errors are classified as transient."""
        response = Mock(spec=httpx.Response)
        response.status_code = 401
        response.headers = {}
        response.reason_phrase = "Unauthorized"
        response.json.return_value = {"message": "Token has expired"}

        error = PulseAPIError(response)

        assert error.is_transient is True
        assert error.error_category == "authentication"
        assert "automatically refresh" in error.recovery_hint

    def test_permanent_client_error(self):
        """Test that 4xx client errors are classified as permanent."""
        response = Mock(spec=httpx.Response)
        response.status_code = 400
        response.headers = {}
        response.reason_phrase = "Bad Request"
        response.json.return_value = {"message": "Invalid request format"}

        error = PulseAPIError(response)

        assert error.is_permanent is True
        assert error.error_category == "client_error"
        assert "request data format" in error.recovery_hint


class TestNetworkError:
    """Test NetworkError classification."""

    def test_network_error_classification(self):
        """Test that network errors are classified as transient."""
        error = NetworkError(
            "https://api.example.com", ConnectionError("Connection failed")
        )

        assert error.is_transient is True
        assert error.error_category == "network"
        assert "network connectivity" in error.recovery_hint


class TestErrorClassification:
    """Test error classification helper functions."""

    def test_classify_api_error(self):
        """Test classification of API errors."""
        response = Mock(spec=httpx.Response)
        response.status_code = 429
        response.headers = {"retry-after": "30"}
        response.reason_phrase = "Too Many Requests"
        response.json.return_value = {"message": "Rate limit exceeded"}

        error = PulseAPIError(response)
        category, severity, hint = classify_error(error)

        assert category == "rate_limiting"
        assert severity == ErrorSeverity.TRANSIENT
        assert "wait 30 seconds" in hint

    def test_classify_network_error(self):
        """Test classification of network errors."""
        error = NetworkError(
            "https://api.example.com", ConnectionError("Connection failed")
        )
        category, severity, hint = classify_error(error)

        assert category == "network"
        assert severity == ErrorSeverity.TRANSIENT
        assert "network connectivity" in hint

    def test_classify_timeout_error(self):
        """Test classification of timeout errors."""
        error = TimeoutError("https://api.example.com", 30000)
        category, severity, hint = classify_error(error)

        assert category == "timeout"
        assert severity == ErrorSeverity.TRANSIENT
        assert "timeout" in hint

    def test_classify_configuration_error(self):
        """Test classification of configuration errors."""
        error = ValueError("Client Secret is required for OAuth2 authentication")
        category, severity, hint = classify_error(error)

        assert category == "configuration"
        assert severity == ErrorSeverity.PERMANENT
        assert "PULSE_CLIENT_SECRET" in hint

    def test_classify_unknown_error(self):
        """Test classification of unknown errors."""
        error = RuntimeError("Something went wrong")
        category, severity, hint = classify_error(error)

        assert category == "unknown"
        assert severity == ErrorSeverity.UNKNOWN
        assert "documentation" in hint

    def test_should_retry_transient_error(self):
        """Test that transient errors should be retried."""
        error = NetworkError(
            "https://api.example.com", ConnectionError("Connection failed")
        )
        assert should_retry_error(error) is True

    def test_should_not_retry_permanent_error(self):
        """Test that permanent errors should not be retried."""
        response = Mock(spec=httpx.Response)
        response.status_code = 400
        response.headers = {}
        response.reason_phrase = "Bad Request"
        response.json.return_value = {"message": "Invalid request"}

        error = PulseAPIError(response)
        assert should_retry_error(error) is False


class TestErrorRecoveryHints:
    """Test error recovery hint generation."""

    def test_rate_limiting_hint_with_retry_after(self):
        """Test rate limiting hint includes retry-after value."""
        response = Mock(spec=httpx.Response)
        response.status_code = 429
        response.headers = {"retry-after": "120"}
        response.reason_phrase = "Too Many Requests"
        response.json.return_value = {"message": "Rate limit exceeded"}

        error = PulseAPIError(response)
        assert "wait 120 seconds" in error.recovery_hint

    def test_rate_limiting_hint_without_retry_after(self):
        """Test rate limiting hint with default value."""
        response = Mock(spec=httpx.Response)
        response.status_code = 429
        response.headers = {}
        response.reason_phrase = "Too Many Requests"
        response.json.return_value = {"message": "Rate limit exceeded"}

        error = PulseAPIError(response)
        assert "wait 60 seconds" in error.recovery_hint

    def test_auth_error_hint(self):
        """Test authentication error hint."""
        response = Mock(spec=httpx.Response)
        response.status_code = 401
        response.headers = {}
        response.reason_phrase = "Unauthorized"
        response.json.return_value = {"message": "Invalid credentials"}

        error = PulseAPIError(response)
        assert "PULSE_CLIENT_ID" in error.recovery_hint
        assert "PULSE_CLIENT_SECRET" in error.recovery_hint

    def test_server_error_hint(self):
        """Test server error hint."""
        response = Mock(spec=httpx.Response)
        response.status_code = 500
        response.headers = {}
        response.reason_phrase = "Internal Server Error"
        response.json.return_value = {"message": "Internal server error"}

        error = PulseAPIError(response)
        assert "exponential backoff" in error.recovery_hint
