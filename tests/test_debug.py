"""Tests for the debug module."""

import os
import time
from unittest.mock import Mock, patch

import pytest

from pulse.debug import (
    enable_debug,
    disable_debug,
    get_debug_config,
    get_debug_stats,
    clear_debug_stats,
    mask_credentials,
    inspect_token,
    log_cache_hit,
    log_cache_miss,
    log_auth_refresh,
    log_request_failure,
    debug_request,
)


class TestDebugConfig:
    """Test debug configuration functionality."""

    def test_enable_debug_default_settings(self):
        """Test enabling debug with default settings."""
        enable_debug()
        config = get_debug_config()

        assert config.enabled is True
        assert config.log_requests is True
        assert config.log_responses is True
        assert config.log_timing is True
        assert config.log_cache_stats is True
        assert config.log_auth_status is True
        assert config.mask_credentials is True
        assert config.max_body_size == 10000
        assert config.log_level == "DEBUG"
        assert config.categories == ["all"]

    def test_enable_debug_custom_settings(self):
        """Test enabling debug with custom settings."""
        enable_debug(
            log_requests=False,
            log_responses=False,
            log_timing=False,
            log_cache_stats=False,
            log_auth_status=False,
            mask_credentials=False,
            max_body_size=5000,
            log_level="INFO",
            categories=["requests", "auth"],
        )
        config = get_debug_config()

        assert config.enabled is True
        assert config.log_requests is False
        assert config.log_responses is False
        assert config.log_timing is False
        assert config.log_cache_stats is False
        assert config.log_auth_status is False
        assert config.mask_credentials is False
        assert config.max_body_size == 5000
        assert config.log_level == "INFO"
        assert config.categories == ["requests", "auth"]

    def test_disable_debug(self):
        """Test disabling debug mode."""
        enable_debug()
        assert get_debug_config().enabled is True

        disable_debug()
        assert get_debug_config().enabled is False


class TestDebugStats:
    """Test debug statistics functionality."""

    def setup_method(self):
        """Clear stats before each test."""
        clear_debug_stats()

    def test_clear_debug_stats(self):
        """Test clearing debug statistics."""
        stats = get_debug_stats()
        stats.total_requests = 5
        stats.failed_requests = 1
        stats.auth_refreshes = 2

        clear_debug_stats()
        new_stats = get_debug_stats()

        assert new_stats.total_requests == 0
        assert new_stats.failed_requests == 0
        assert new_stats.auth_refreshes == 0

    def test_cache_stats(self):
        """Test cache hit/miss statistics."""
        enable_debug()

        log_cache_hit("test_key_1")
        log_cache_hit("test_key_2")
        log_cache_miss("test_key_3")

        stats = get_debug_stats()
        assert stats.cache_stats.hits == 2
        assert stats.cache_stats.misses == 1
        assert stats.cache_stats.total_requests == 3
        assert stats.cache_stats.hit_rate == 66.66666666666666

    def test_auth_refresh_logging(self):
        """Test authentication refresh logging."""
        enable_debug()

        log_auth_refresh()
        log_auth_refresh()

        stats = get_debug_stats()
        assert stats.auth_refreshes == 2

    def test_request_failure_logging(self):
        """Test request failure logging."""
        enable_debug()

        error = Exception("Test error")
        log_request_failure(error)

        stats = get_debug_stats()
        assert stats.failed_requests == 1

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        stats = get_debug_stats()
        stats.total_requests = 10
        stats.failed_requests = 2

        assert stats.success_rate == 80.0

    def test_success_rate_no_requests(self):
        """Test success rate with no requests."""
        stats = get_debug_stats()
        assert stats.success_rate == 0.0


class TestCredentialMasking:
    """Test credential masking functionality."""

    def test_mask_credentials_string(self):
        """Test masking credentials in strings."""
        enable_debug(mask_credentials=True)

        test_data = '{"client_secret": "secret123", "access_token": "token456"}'
        masked = mask_credentials(test_data)

        assert "secret123" not in masked
        assert "token456" not in masked
        assert "***MASKED***" in masked

    def test_mask_credentials_dict(self):
        """Test masking credentials in dictionaries."""
        enable_debug(mask_credentials=True)

        test_data = {
            "client_secret": "secret123",
            "client_id": "client456",
            "access_token": "token789",
            "other_field": "safe_value",
        }
        masked = mask_credentials(test_data)

        assert masked["client_secret"] == "***MASKED***"
        assert masked["client_id"] == "***MASKED***"
        assert masked["access_token"] == "***MASKED***"
        assert masked["other_field"] == "safe_value"

    def test_mask_credentials_bearer_token(self):
        """Test masking Bearer tokens in strings."""
        enable_debug(mask_credentials=True)

        test_data = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        masked = mask_credentials(test_data)

        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in masked
        assert "Bearer ***MASKED***" in masked

    def test_mask_credentials_disabled(self):
        """Test that masking is disabled when configured."""
        enable_debug(mask_credentials=False)

        test_data = '{"client_secret": "secret123"}'
        masked = mask_credentials(test_data)

        assert masked == test_data


class TestTokenInspection:
    """Test token inspection functionality."""

    def test_inspect_token_with_valid_token(self):
        """Test inspecting a valid token."""
        mock_auth = Mock()
        mock_auth._access_token = "valid_token_12345"
        mock_auth._expires_at = time.time() + 3600  # 1 hour from now
        mock_auth.__class__.__name__ = "ClientCredentialsAuth"

        token_info = inspect_token(mock_auth)

        assert token_info.has_token is True
        assert token_info.is_expired is False
        assert token_info.is_valid is True
        assert token_info.token_type == "client_credentials"
        assert token_info.masked_token == "vali...2345"
        assert token_info.expires_in > 3500  # Should be close to 3600

    def test_inspect_token_with_expired_token(self):
        """Test inspecting an expired token."""
        mock_auth = Mock()
        mock_auth._access_token = "expired_token"
        mock_auth._expires_at = time.time() - 3600  # 1 hour ago
        mock_auth.__class__.__name__ = "AuthorizationCodePKCEAuth"

        token_info = inspect_token(mock_auth)

        assert token_info.has_token is True
        assert token_info.is_expired is True
        assert token_info.is_valid is False
        assert token_info.token_type == "authorization_code_pkce"

    def test_inspect_token_no_token(self):
        """Test inspecting auth object with no token."""
        mock_auth = Mock()
        mock_auth._access_token = None

        token_info = inspect_token(mock_auth)

        assert token_info.has_token is False
        assert token_info.is_valid is False

    def test_inspect_token_short_token(self):
        """Test inspecting a short token."""
        mock_auth = Mock()
        mock_auth._access_token = "short"
        mock_auth._expires_at = time.time() + 3600

        token_info = inspect_token(mock_auth)

        assert token_info.masked_token == "***MASKED***"


class TestDebugRequest:
    """Test debug request context manager."""

    def setup_method(self):
        """Clear stats before each test."""
        clear_debug_stats()

    def test_debug_request_timing(self):
        """Test request timing measurement."""
        enable_debug()

        with debug_request("GET", "https://example.com") as timing:
            time.sleep(0.1)  # Simulate request duration

        assert timing.method == "GET"
        assert timing.url == "https://example.com"
        assert timing.duration >= 0.1
        assert timing.start_time > 0
        assert timing.end_time > timing.start_time

        stats = get_debug_stats()
        assert len(stats.request_timings) == 1
        assert stats.total_requests == 1

    def test_debug_request_with_exception(self):
        """Test debug request handling with exception."""
        enable_debug()

        with pytest.raises(ValueError):
            with debug_request("POST", "https://example.com") as timing:
                raise ValueError("Test error")

        # Timing should still be recorded even with exception
        assert timing.duration > 0
        stats = get_debug_stats()
        assert len(stats.request_timings) == 1

    def test_average_request_time(self):
        """Test average request time calculation."""
        enable_debug()

        with debug_request("GET", "https://example.com"):
            time.sleep(0.1)

        with debug_request("POST", "https://example.com"):
            time.sleep(0.2)

        stats = get_debug_stats()
        avg_time = stats.average_request_time
        assert 0.1 <= avg_time <= 0.2


class TestEnvironmentVariableActivation:
    """Test automatic debug activation via environment variable."""

    def test_pulse_debug_true(self):
        """Test that PULSE_DEBUG=true enables debug mode."""
        with patch.dict(os.environ, {"PULSE_DEBUG": "true"}):
            # Re-import the module to trigger environment check
            import importlib
            from pulse import debug

            importlib.reload(debug)

            config = debug.get_debug_config()
            assert config.enabled is True

    def test_pulse_debug_false(self):
        """Test that PULSE_DEBUG=false doesn't enable debug mode."""
        with patch.dict(os.environ, {"PULSE_DEBUG": "false"}):
            # Re-import the module to trigger environment check
            import importlib
            from pulse import debug

            importlib.reload(debug)

            config = debug.get_debug_config()
            assert config.enabled is False

    def test_pulse_debug_not_set(self):
        """Test that debug mode is disabled when PULSE_DEBUG is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Re-import the module to trigger environment check
            import importlib
            from pulse import debug

            importlib.reload(debug)

            config = debug.get_debug_config()
            assert config.enabled is False


class TestLoggingCategories:
    """Test granular logging categories."""

    def setup_method(self):
        """Setup for each test."""
        clear_debug_stats()

    def test_category_filtering_all(self):
        """Test that 'all' category enables all logging."""
        enable_debug(categories=["all"])

        # This would normally log if categories are working
        log_cache_hit("test_key")

        stats = get_debug_stats()
        assert stats.cache_stats.hits == 1

    def test_category_filtering_specific(self):
        """Test specific category filtering."""
        enable_debug(categories=["cache"])

        # Cache logging should work
        log_cache_hit("test_key")

        stats = get_debug_stats()
        assert stats.cache_stats.hits == 1

    def test_category_filtering_excluded(self):
        """Test that excluded categories don't log."""
        enable_debug(categories=["requests"])  # Only requests, not cache

        # Cache logging should still update stats but not log
        log_cache_hit("test_key")

        stats = get_debug_stats()
        assert stats.cache_stats.hits == 1
