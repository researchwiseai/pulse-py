"""Tests for async authentication classes and factory methods."""

import os
import pytest
import httpx
from unittest.mock import patch, AsyncMock

from pulse.async_auth import (
    AsyncClientCredentialsAuth,
    AsyncAuthorizationCodePKCEAuth,
    async_auto_auth,
)
from pulse.core.async_client import AsyncCoreClient


class TestAsyncClientCredentialsAuth:
    """Test AsyncClientCredentialsAuth class."""

    def test_init_with_params(self):
        """Test initialization with direct parameters."""
        auth = AsyncClientCredentialsAuth(
            client_id="test_id",
            client_secret="test_secret",
            audience="test_audience",
            token_url="https://test.com/token",
        )

        assert auth.client_id == "test_id"
        assert auth.client_secret == "test_secret"
        assert auth.audience == "test_audience"
        assert auth.token_url == "https://test.com/token"

    def test_init_with_env_vars(self):
        """Test initialization with environment variables."""
        with patch.dict(
            os.environ,
            {
                "PULSE_CLIENT_ID": "env_id",
                "PULSE_CLIENT_SECRET": "env_secret",
                "PULSE_AUDIENCE": "env_audience",
                "PULSE_TOKEN_URL": "https://env.com/token",
            },
        ):
            auth = AsyncClientCredentialsAuth()

            assert auth.client_id == "env_id"
            assert auth.client_secret == "env_secret"
            assert auth.audience == "env_audience"
            assert auth.token_url == "https://env.com/token"

    def test_init_missing_client_secret(self):
        """Test that missing client secret raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Client Secret is required"):
                AsyncClientCredentialsAuth(client_id="test")

    def test_sync_auth_flow_raises_error(self):
        """Test that sync_auth_flow raises RuntimeError."""
        auth = AsyncClientCredentialsAuth(client_id="test", client_secret="test")

        with pytest.raises(RuntimeError, match="Cannot use async authentication class"):
            list(auth.sync_auth_flow(httpx.Request("GET", "https://example.com")))

    @pytest.mark.asyncio
    async def test_async_auth_flow(self):
        """Test async_auth_flow adds authorization header."""
        auth = AsyncClientCredentialsAuth(client_id="test", client_secret="test")

        # Mock the token refresh to avoid actual HTTP calls
        auth._access_token = "test_token"
        auth._expires_at = 9999999999  # Far future

        request = httpx.Request("GET", "https://pulse.researchwiseai.com/v1/test")

        async_gen = auth.async_auth_flow(request)
        modified_request = await async_gen.__anext__()

        assert modified_request.headers["Authorization"] == "Bearer test_token"

    @pytest.mark.asyncio
    async def test_refresh_token(self):
        """Test token refresh functionality."""
        auth = AsyncClientCredentialsAuth(
            client_id="test_id", client_secret="test_secret", audience="test_audience"
        )

        # Mock the HTTP response
        from unittest.mock import Mock

        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "new_token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            await auth._refresh_token()

            assert auth._access_token == "new_token"
            assert auth._expires_at > 0

            # Verify the request was made correctly
            mock_client.post.assert_called_once_with(
                auth.token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": "test_id",
                    "client_secret": "test_secret",
                    "audience": "test_audience",
                },
            )


class TestAsyncAuthorizationCodePKCEAuth:
    """Test AsyncAuthorizationCodePKCEAuth class."""

    def test_init_with_params(self):
        """Test initialization with direct parameters."""
        auth = AsyncAuthorizationCodePKCEAuth(
            code="test_code",
            code_verifier="test_verifier",
            client_id="test_id",
            redirect_uri="http://localhost:8888/callback",
        )

        assert auth.code == "test_code"
        assert auth.code_verifier == "test_verifier"
        assert auth.client_id == "test_id"
        assert auth.redirect_uri == "http://localhost:8888/callback"

    def test_sync_auth_flow_raises_error(self):
        """Test that sync_auth_flow raises RuntimeError."""
        auth = AsyncAuthorizationCodePKCEAuth(
            code="test_code", code_verifier="test_verifier"
        )

        with pytest.raises(RuntimeError, match="Cannot use async authentication class"):
            list(auth.sync_auth_flow(httpx.Request("GET", "https://example.com")))

    @pytest.mark.asyncio
    async def test_refresh_token(self):
        """Test token refresh functionality."""
        auth = AsyncAuthorizationCodePKCEAuth(
            code="test_code",
            code_verifier="test_verifier",
            client_id="test_id",
            redirect_uri="http://localhost:8888/callback",
        )

        # Mock the HTTP response
        from unittest.mock import Mock

        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "new_token",
            "expires_in": 3600,
            "refresh_token": "refresh_token",
        }
        mock_response.raise_for_status.return_value = None

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            await auth._refresh_token()

            assert auth._access_token == "new_token"
            assert auth._refresh_token_value == "refresh_token"
            assert auth._expires_at > 0


class TestAsyncCoreClientFactoryMethods:
    """Test AsyncCoreClient factory methods."""

    def test_with_client_credentials_async(self):
        """Test with_client_credentials_async factory method."""
        client = AsyncCoreClient.with_client_credentials_async(
            client_id="test_id", client_secret="test_secret", audience="test_audience"
        )

        assert isinstance(client.client.auth, AsyncClientCredentialsAuth)
        assert client.client.auth.client_id == "test_id"
        assert client.client.auth.client_secret == "test_secret"
        assert client.client.auth.audience == "test_audience"

    def test_with_client_credentials_async_env_vars(self):
        """Test with_client_credentials_async with environment variables."""
        with patch.dict(
            os.environ,
            {"PULSE_CLIENT_ID": "env_id", "PULSE_CLIENT_SECRET": "env_secret"},
        ):
            client = AsyncCoreClient.with_client_credentials_async()

            assert isinstance(client.client.auth, AsyncClientCredentialsAuth)
            assert client.client.auth.client_id == "env_id"
            assert client.client.auth.client_secret == "env_secret"

    def test_with_client_credentials_async_missing_id(self):
        """Test with_client_credentials_async raises error for missing client_id."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Client ID must be provided"):
                AsyncCoreClient.with_client_credentials_async(client_secret="secret")

    def test_with_client_credentials_async_missing_secret(self):
        """Test with_client_credentials_async raises error for missing secret."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Client secret must be provided"):
                AsyncCoreClient.with_client_credentials_async(client_id="id")

    def test_with_pkce_async(self):
        """Test with_pkce_async factory method."""
        client = AsyncCoreClient.with_pkce_async(
            code="test_code",
            code_verifier="test_verifier",
            client_id="test_id",
            redirect_uri="http://localhost:8888/callback",
        )

        assert isinstance(client.client.auth, AsyncAuthorizationCodePKCEAuth)
        assert client.client.auth.code == "test_code"
        assert client.client.auth.code_verifier == "test_verifier"
        assert client.client.auth.client_id == "test_id"
        assert client.client.auth.redirect_uri == "http://localhost:8888/callback"

    def test_with_pkce_async_env_vars(self):
        """Test with_pkce_async with environment variables."""
        with patch.dict(
            os.environ,
            {
                "PULSE_CLIENT_ID": "env_id",
                "PULSE_REDIRECT_URI": "http://env:8888/callback",
            },
        ):
            client = AsyncCoreClient.with_pkce_async(
                code="test_code", code_verifier="test_verifier"
            )

            assert isinstance(client.client.auth, AsyncAuthorizationCodePKCEAuth)
            assert client.client.auth.client_id == "env_id"
            assert client.client.auth.redirect_uri == "http://env:8888/callback"

    def test_with_pkce_async_missing_client_id(self):
        """Test with_pkce_async raises error for missing client_id."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Client ID must be provided"):
                AsyncCoreClient.with_pkce_async(
                    code="code",
                    code_verifier="verifier",
                    redirect_uri="http://localhost:8888/callback",
                )

    def test_with_pkce_async_missing_redirect_uri(self):
        """Test with_pkce_async raises error for missing redirect_uri."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Redirect URI must be provided"):
                AsyncCoreClient.with_pkce_async(
                    code="code", code_verifier="verifier", client_id="id"
                )


class TestAsyncAutoAuth:
    """Test async_auto_auth function."""

    def test_async_auto_auth_with_client_credentials(self):
        """Test async_auto_auth returns AsyncClientCredentialsAuth when env vars set."""
        with patch.dict(
            os.environ,
            {"PULSE_CLIENT_ID": "test_id", "PULSE_CLIENT_SECRET": "test_secret"},
        ):
            auth = async_auto_auth()
            assert isinstance(auth, AsyncClientCredentialsAuth)

    def test_async_auto_auth_without_credentials(self):
        """Test async_auto_auth returns AsyncAuthorizationCodePKCEAuth when no env."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(
                AsyncAuthorizationCodePKCEAuth, "_get_code", return_value="test_code"
            ):
                auth = async_auto_auth()
                assert isinstance(auth, AsyncAuthorizationCodePKCEAuth)
