"""Authentication module for future OAuth2 credentials."""


import time
import httpx

__all__ = [
    "ClientCredentialsAuth",
    "AuthorizationCodePKCEAuth",
]


class _BaseOAuth2Auth(httpx.Auth):
    """Base HTTPX Auth class for OAuth2 bearer token authentication."""

    def __init__(
        self,
        token_url: str,
        client_id: str,
        scope: str | None = None,
        audience: str | None = None,
    ) -> None:
        self.token_url = token_url
        self.client_id = client_id
        self.scope = scope
        self.audience = audience
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
        client_id: str,
        client_secret: str,
        token_url: str = "https://research-wise-ai-eu.eu.auth0.com/oauth/token",
        audience: str | None = None,
        scope: str | None = None,
        organization: str | None = None,
    ) -> None:
        super().__init__(token_url, client_id, scope, audience)
        self.client_secret = client_secret
        self.organization = organization

    def _refresh_token(self) -> None:
        data: dict[str, str] = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        if self.scope:
            data["scope"] = self.scope
        if self.audience:
            data["audience"] = self.audience
        if self.organization:
            data["organization"] = self.organization
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
        token_url: str,
        client_id: str,
        code: str,
        redirect_uri: str,
        code_verifier: str,
        scope: str | None = None,
        audience: str | None = None,
    ) -> None:
        super().__init__(token_url, client_id, scope, audience)
        self.code = code
        self.redirect_uri = redirect_uri
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
        if self.scope:
            data["scope"] = self.scope
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
