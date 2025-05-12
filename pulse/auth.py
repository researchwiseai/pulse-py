"""Authentication module for future OAuth2 credentials."""


import os
import time
import httpx

__all__ = [
    "ClientCredentialsAuth",
    "AuthorizationCodePKCEAuth",
]


def _throw_client_id_error() -> str:
    """Raise an error if client_id is not provided."""
    raise ValueError(
        "Client ID is required for OAuth2 authentication. "
        "Please set the PULSE_CLIENT_ID environment variable."
    )


def _throw_client_secret_error() -> str:
    """Raise an error if client_secret is not provided."""
    raise ValueError(
        "Client Secret is required for OAuth2 authentication. "
        "Please set the PULSE_CLIENT_SECRET environment variable."
    )


def _throw_organization_error() -> str:
    """Raise an error if organization is not provided."""
    raise ValueError(
        "Organization ID is required for OAuth2 authentication. "
        "Please set the PULSE_ORG_ID environment variable."
    )


def _throw_redirect_uri_error() -> str:
    """Raise an error if redirect_uri is not provided."""
    raise ValueError(
        "Redirect URI is required for OAuth2 authentication. "
        "Please set the PULSE_REDIRECT_URI environment variable."
    )


class _BaseOAuth2Auth(httpx.Auth):
    """Base HTTPX Auth class for OAuth2 bearer token authentication."""

    def __init__(
        self,
        token_url: str | None,
        client_id: str | None,
        audience: str | None,
        organization: str | None,
    ) -> None:
        self.token_url = (
            token_url
            or os.getenv("PULSE_TOKEN_URL")
            or "https://research-wise-ai-eu.eu.auth0.com/oauth/token"
        )
        self.client_id = (
            client_id or os.getenv("PULSE_CLIENT_ID") or _throw_client_id_error()
        )
        self.audience = (
            audience
            or os.getenv("PULSE_API_URL")
            or "https://core.researchwiseai.com/pulse/v1"
        )
        self.organization = (
            organization or os.getenv("PULSE_ORG_ID") or _throw_organization_error()
        )
        self._access_token: str | None = None
        self._expires_at: float = 0.0

    def auth_flow(self, request: httpx.Request) -> httpx.Request:
        # Only attach token to core.researchwiseai.com endpoints
        host = request.url.host or ""
        if host.endswith("core.researchwiseai.com"):
            token = self._get_token()
            request.headers["Authorization"] = f"Bearer {token}"
        yield request

    def _get_token(self) -> str:
        now = time.time()
        # Refresh token if missing or expired
        if self._access_token is None or now >= self._expires_at:
            self._refresh_token()
        return self._access_token  # type: ignore

    def _refresh_token(self) -> None:
        """Refresh the access token. To be implemented by subclasses."""
        raise NotImplementedError


class ClientCredentialsAuth(_BaseOAuth2Auth):
    """OAuth2 Client Credentials flow authentication."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        organization: str | None = None,
        token_url: str | None = None,
        audience: str | None = None,
    ) -> None:
        super().__init__(token_url, client_id, audience, organization)
        self.client_secret = (
            client_secret
            or os.getenv("PULSE_CLIENT_SECRET")
            or _throw_client_secret_error()
        )

    def _refresh_token(self) -> None:
        data: dict[str, str] = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "audience": self.audience,
            "organization": self.organization,
        }
        resp = httpx.post(self.token_url, data=data)
        resp.raise_for_status()
        payload = resp.json()
        # Expecting standard OAuth2 response with access_token and expires_in
        self._access_token = payload.get("access_token")
        expires_in = payload.get("expires_in", 3600)
        # Subtract a buffer to account for clock skew
        self._expires_at = time.time() + float(expires_in) - 60.0


class AuthorizationCodePKCEAuth(_BaseOAuth2Auth):
    """OAuth2 Authorization Code flow with PKCE authentication."""

    def __init__(
        self,
        code: str,
        code_verifier: str,
        client_id: str | None = None,
        redirect_uri: str | None = None,
        organization: str | None = None,
        token_url: str | None = None,
        audience: str | None = None,
    ) -> None:
        super().__init__(token_url, client_id, audience, organization)
        self.code = code
        self.redirect_uri = (
            redirect_uri
            or os.getenv("PULSE_REDIRECT_URI")
            or _throw_redirect_uri_error()
        )
        self.code_verifier = code_verifier
        self._refresh_token_value: str | None = None

    def _refresh_token(self) -> None:
        data: dict[str, str] = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "code": self.code,
            "redirect_uri": self.redirect_uri,
            "code_verifier": self.code_verifier,
        }
        if self.audience:
            data["audience"] = self.audience
        resp = httpx.post(self.token_url, data=data)
        resp.raise_for_status()
        payload = resp.json()
        self._access_token = payload.get("access_token")
        # Store refresh token if provided
        self._refresh_token_value = payload.get("refresh_token")
        expires_in = payload.get("expires_in", 3600)
        self._expires_at = time.time() + float(expires_in) - 60.0
