"""Debugging and introspection tools for the Pulse SDK.

This module provides comprehensive debugging capabilities including:
- Request/response logging with credential masking
- Performance timing and metrics collection
- Cache hit/miss statistics
- Authentication token status inspection
- Granular logging categories

Enable debug mode by setting the PULSE_DEBUG environment variable to 'true'.
"""

import json
import logging
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import httpx

__all__ = [
    "DebugConfig",
    "DebugStats",
    "TokenInfo",
    "enable_debug",
    "disable_debug",
    "get_debug_config",
    "get_debug_stats",
    "inspect_token",
    "clear_debug_stats",
    "debug_request",
    "mask_credentials",
]


@dataclass
class DebugConfig:
    """Configuration for debug mode."""

    enabled: bool = False
    log_requests: bool = True
    log_responses: bool = True
    log_timing: bool = True
    log_cache_stats: bool = True
    log_auth_status: bool = True
    mask_credentials: bool = True
    max_body_size: int = 10000  # Max characters to log for request/response bodies
    log_level: str = "DEBUG"
    categories: List[str] = field(default_factory=lambda: ["all"])


@dataclass
class RequestTiming:
    """Timing information for a request."""

    start_time: float
    end_time: float
    duration: float
    method: str
    url: str
    status_code: Optional[int] = None


@dataclass
class CacheStats:
    """Cache hit/miss statistics."""

    hits: int = 0
    misses: int = 0
    total_requests: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100


@dataclass
class DebugStats:
    """Comprehensive debugging statistics."""

    request_timings: List[RequestTiming] = field(default_factory=list)
    cache_stats: CacheStats = field(default_factory=CacheStats)
    total_requests: int = 0
    failed_requests: int = 0
    auth_refreshes: int = 0

    @property
    def average_request_time(self) -> float:
        """Calculate average request duration in seconds."""
        if not self.request_timings:
            return 0.0
        return sum(t.duration for t in self.request_timings) / len(self.request_timings)

    @property
    def success_rate(self) -> float:
        """Calculate request success rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        successful = self.total_requests - self.failed_requests
        return (successful / self.total_requests) * 100


@dataclass
class TokenInfo:
    """Information about authentication token status."""

    has_token: bool
    expires_at: Optional[float] = None
    expires_in: Optional[float] = None
    is_expired: bool = False
    token_type: Optional[str] = None
    masked_token: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Check if token is present and not expired."""
        return self.has_token and not self.is_expired


# Global debug configuration and stats
_debug_config = DebugConfig()
_debug_stats = DebugStats()
_logger = logging.getLogger("pulse.debug")


def _setup_logger() -> None:
    """Setup debug logger with appropriate formatting."""
    if not _logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        _logger.addHandler(handler)

    # Set log level based on config
    level = getattr(logging, _debug_config.log_level.upper(), logging.DEBUG)
    _logger.setLevel(level)


def enable_debug(
    log_requests: bool = True,
    log_responses: bool = True,
    log_timing: bool = True,
    log_cache_stats: bool = True,
    log_auth_status: bool = True,
    mask_credentials: bool = True,
    max_body_size: int = 10000,
    log_level: str = "DEBUG",
    categories: Optional[List[str]] = None,
) -> None:
    """Enable debug mode with specified configuration.

    Args:
        log_requests: Whether to log HTTP requests
        log_responses: Whether to log HTTP responses
        log_timing: Whether to log timing information
        log_cache_stats: Whether to log cache statistics
        log_auth_status: Whether to log authentication status
        mask_credentials: Whether to mask sensitive credentials in logs
        max_body_size: Maximum characters to log for request/response bodies
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        categories: List of debug categories to enable (default: ["all"])
    """
    global _debug_config

    _debug_config.enabled = True
    _debug_config.log_requests = log_requests
    _debug_config.log_responses = log_responses
    _debug_config.log_timing = log_timing
    _debug_config.log_cache_stats = log_cache_stats
    _debug_config.log_auth_status = log_auth_status
    _debug_config.mask_credentials = mask_credentials
    _debug_config.max_body_size = max_body_size
    _debug_config.log_level = log_level
    _debug_config.categories = categories or ["all"]

    _setup_logger()
    _logger.info("Debug mode enabled")


def disable_debug() -> None:
    """Disable debug mode."""
    global _debug_config

    _debug_config.enabled = False
    _logger.info("Debug mode disabled")


def get_debug_config() -> DebugConfig:
    """Get current debug configuration."""
    return _debug_config


def get_debug_stats() -> DebugStats:
    """Get current debug statistics."""
    return _debug_stats


def clear_debug_stats() -> None:
    """Clear all debug statistics."""
    global _debug_stats

    _debug_stats = DebugStats()
    _logger.info("Debug statistics cleared")


def mask_credentials(data: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
    """Mask sensitive credentials in data.

    Args:
        data: String or dictionary that may contain sensitive information

    Returns:
        Data with sensitive information masked
    """
    if not _debug_config.mask_credentials:
        return data

    # Patterns for sensitive data
    sensitive_patterns = [
        (r'("client_secret":\s*")[^"]*(")', r"\1***MASKED***\2"),
        (r'("client_id":\s*")[^"]*(")', r"\1***MASKED***\2"),
        (r'("access_token":\s*")[^"]*(")', r"\1***MASKED***\2"),
        (r'("refresh_token":\s*")[^"]*(")', r"\1***MASKED***\2"),
        (r'("code":\s*")[^"]*(")', r"\1***MASKED***\2"),
        (r'("code_verifier":\s*")[^"]*(")', r"\1***MASKED***\2"),
        (r"(Bearer\s+)[A-Za-z0-9\-._~+/]+=*", r"\1***MASKED***"),
        (r"(client_secret=)[^&\s]*", r"\1***MASKED***"),
        (r"(client_id=)[^&\s]*", r"\1***MASKED***"),
    ]

    if isinstance(data, str):
        masked_data = data
        for pattern, replacement in sensitive_patterns:
            masked_data = re.sub(pattern, replacement, masked_data, flags=re.IGNORECASE)
        return masked_data

    elif isinstance(data, dict):
        masked_data = {}
        for key, value in data.items():
            if key.lower() in [
                "client_secret",
                "client_id",
                "access_token",
                "refresh_token",
                "code",
                "code_verifier",
            ]:
                masked_data[key] = "***MASKED***"
            elif isinstance(value, (str, dict)):
                masked_data[key] = mask_credentials(value)
            else:
                masked_data[key] = value
        return masked_data

    return data


def _should_log_category(category: str) -> bool:
    """Check if a debug category should be logged."""
    if not _debug_config.enabled:
        return False

    return "all" in _debug_config.categories or category in _debug_config.categories


def _truncate_body(body: str, max_size: int) -> str:
    """Truncate body content if it exceeds max size."""
    if len(body) <= max_size:
        return body

    return body[:max_size] + f"... [truncated, total length: {len(body)}]"


@contextmanager
def debug_request(method: str, url: str, **kwargs):
    """Context manager for debugging HTTP requests.

    Args:
        method: HTTP method
        url: Request URL
        **kwargs: Additional request parameters

    Yields:
        Request timing object that gets populated during the request
    """
    timing = RequestTiming(
        start_time=time.time(),
        end_time=0.0,
        duration=0.0,
        method=method.upper(),
        url=url,
    )

    try:
        # Log request if enabled
        if (
            _debug_config.enabled
            and _debug_config.log_requests
            and _should_log_category("requests")
        ):
            _log_request(method, url, kwargs)

        yield timing

    finally:
        # Update timing
        timing.end_time = time.time()
        timing.duration = timing.end_time - timing.start_time

        # Record timing stats
        _debug_stats.request_timings.append(timing)
        _debug_stats.total_requests += 1

        # Log timing if enabled
        if (
            _debug_config.enabled
            and _debug_config.log_timing
            and _should_log_category("timing")
        ):
            _log_timing(timing)


def _log_request(method: str, url: str, kwargs: Dict[str, Any]) -> None:
    """Log HTTP request details."""
    # Parse URL to get host
    parsed_url = urlparse(url)
    safe_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
    if parsed_url.query:
        safe_url += f"?{parsed_url.query}"

    _logger.debug(f"→ {method.upper()} {safe_url}")

    # Log headers (masked)
    if "headers" in kwargs:
        masked_headers = mask_credentials(dict(kwargs["headers"]))
        _logger.debug(f"  Headers: {masked_headers}")

    # Log request body (masked and truncated)
    if "json" in kwargs:
        try:
            body_str = json.dumps(kwargs["json"], indent=2)
            masked_body = mask_credentials(body_str)
            truncated_body = _truncate_body(masked_body, _debug_config.max_body_size)
            _logger.debug(f"  Body: {truncated_body}")
        except Exception as e:
            _logger.debug(f"  Body: [Error serializing: {e}]")

    elif "data" in kwargs:
        masked_data = mask_credentials(str(kwargs["data"]))
        truncated_data = _truncate_body(masked_data, _debug_config.max_body_size)
        _logger.debug(f"  Data: {truncated_data}")


def _log_response(response: httpx.Response) -> None:
    """Log HTTP response details."""
    if not (
        _debug_config.enabled
        and _debug_config.log_responses
        and _should_log_category("responses")
    ):
        return

    _logger.debug(f"← {response.status_code} {response.reason_phrase}")

    # Log response headers (masked)
    masked_headers = mask_credentials(dict(response.headers))
    _logger.debug(f"  Headers: {masked_headers}")

    # Log response body (masked and truncated)
    try:
        if response.headers.get("content-type", "").startswith("application/json"):
            body_str = response.text
            masked_body = mask_credentials(body_str)
            truncated_body = _truncate_body(masked_body, _debug_config.max_body_size)
            _logger.debug(f"  Body: {truncated_body}")
        else:
            content_length = len(response.content)
            _logger.debug(f"  Body: [Binary content, {content_length} bytes]")
    except Exception as e:
        _logger.debug(f"  Body: [Error reading response: {e}]")


def _log_timing(timing: RequestTiming) -> None:
    """Log request timing information."""
    _logger.debug(
        f"⏱️  {timing.method} {timing.url} completed in {timing.duration:.3f}s"
    )


def inspect_token(auth_obj: Any) -> TokenInfo:
    """Inspect authentication token status.

    Args:
        auth_obj: Authentication object (ClientCredentialsAuth or
                 AuthorizationCodePKCEAuth)

    Returns:
        TokenInfo object with token status details
    """
    token_info = TokenInfo(has_token=False)

    try:
        # Check if auth object has token attributes
        if hasattr(auth_obj, "_access_token"):
            token = auth_obj._access_token
            token_info.has_token = token is not None

            if token:
                # Mask token for logging
                if len(token) > 10:
                    token_info.masked_token = f"{token[:4]}...{token[-4:]}"
                else:
                    token_info.masked_token = (
                        "***MASKED***"  # nosec B105 - Masking string, not a password
                    )

        # Check expiration
        if hasattr(auth_obj, "_expires_at"):
            expires_at = auth_obj._expires_at
            token_info.expires_at = expires_at

            if expires_at:
                current_time = time.time()
                token_info.expires_in = expires_at - current_time
                token_info.is_expired = current_time >= expires_at

        # Determine token type
        if hasattr(auth_obj, "__class__"):
            class_name = auth_obj.__class__.__name__
            if "ClientCredentials" in class_name:
                # nosec B105 - Token type identifier, not a password
                token_info.token_type = "client_credentials"
            elif "PKCE" in class_name:
                # nosec B105 - Token type identifier, not a password
                token_info.token_type = "authorization_code_pkce"
            else:
                # nosec B105 - Token type identifier, not a password
                token_info.token_type = "unknown"

    except Exception as e:
        _logger.warning(f"Error inspecting token: {e}")

    # Log token status if enabled
    if (
        _debug_config.enabled
        and _debug_config.log_auth_status
        and _should_log_category("auth")
    ):
        _log_token_status(token_info)

    return token_info


def _log_token_status(token_info: TokenInfo) -> None:
    """Log authentication token status."""
    status = "✅ Valid" if token_info.is_valid else "❌ Invalid"
    _logger.debug(f"🔐 Token Status: {status}")

    if token_info.has_token:
        _logger.debug(f"   Type: {token_info.token_type}")
        _logger.debug(f"   Token: {token_info.masked_token}")

        if token_info.expires_in is not None:
            if token_info.expires_in > 0:
                _logger.debug(f"   Expires in: {token_info.expires_in:.0f} seconds")
            else:
                _logger.debug("   Status: EXPIRED")
    else:
        _logger.debug("   Status: No token available")


def log_cache_hit(cache_key: str) -> None:
    """Log a cache hit event.

    Args:
        cache_key: The cache key that was hit
    """
    _debug_stats.cache_stats.hits += 1
    _debug_stats.cache_stats.total_requests += 1

    if (
        _debug_config.enabled
        and _debug_config.log_cache_stats
        and _should_log_category("cache")
    ):
        _logger.debug(f"💾 Cache HIT: {cache_key}")


def log_cache_miss(cache_key: str) -> None:
    """Log a cache miss event.

    Args:
        cache_key: The cache key that was missed
    """
    _debug_stats.cache_stats.misses += 1
    _debug_stats.cache_stats.total_requests += 1

    if (
        _debug_config.enabled
        and _debug_config.log_cache_stats
        and _should_log_category("cache")
    ):
        _logger.debug(f"💾 Cache MISS: {cache_key}")


def log_auth_refresh() -> None:
    """Log an authentication token refresh event."""
    _debug_stats.auth_refreshes += 1

    if (
        _debug_config.enabled
        and _debug_config.log_auth_status
        and _should_log_category("auth")
    ):
        _logger.debug("🔄 Authentication token refreshed")


def log_request_failure(error: Exception) -> None:
    """Log a request failure.

    Args:
        error: The exception that caused the failure
    """
    _debug_stats.failed_requests += 1

    if _debug_config.enabled and _should_log_category("errors"):
        _logger.debug(f"❌ Request failed: {error}")


# Auto-enable debug mode if PULSE_DEBUG environment variable is set
if os.getenv("PULSE_DEBUG", "").lower() in ("true", "1", "yes", "on"):
    enable_debug()
