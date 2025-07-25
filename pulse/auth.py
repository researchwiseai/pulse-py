"""Authentication module for future OAuth2 credentials."""

import base64
import hashlib
import os
import secrets
import time
import webbrowser
from urllib.parse import parse_qs, urlencode, urlparse
import httpx

from pulse.config import DEFAULT_SCOPES, PROD_AUTH_DOMAIN, PROD_BASE_URL, PROD_CLIENT_ID

__all__ = [
    "ClientCredentialsAuth",
    "AuthorizationCodePKCEAuth",
]


def _throw_client_secret_error() -> str:
    """Raise an error if client_secret is not provided."""
    raise ValueError(
        "Client Secret is required for OAuth2 authentication. "
        "Please set the PULSE_CLIENT_SECRET environment variable."
    )


class _BaseOAuth2Auth(httpx.Auth):
    """Base HTTPX Auth class for OAuth2 bearer token authentication."""

    def __init__(
        self,
        token_url: str | None,
        client_id: str | None,
        audience: str | None,
    ) -> None:
        self.token_url = (
            token_url
            or os.getenv("PULSE_TOKEN_URL")
            or f"https://{PROD_AUTH_DOMAIN}/oauth/token"
        )
        self.client_id = client_id or os.getenv("PULSE_CLIENT_ID") or PROD_CLIENT_ID
        self.audience = audience or os.getenv("PULSE_API_URL") or PROD_BASE_URL
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
        token_url: str | None = None,
        audience: str | None = None,
    ) -> None:
        super().__init__(token_url, client_id, audience)
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
        code: str | None = None,
        code_verifier: str | None = None,
        client_id: str | None = None,
        redirect_uri: str | None = None,
        token_url: str | None = None,
        audience: str | None = None,
        scope: str | None = None,
        organization: str | None = None,
    ) -> None:
        super().__init__(token_url, client_id, audience)

        # Scenarios:
        # No options provided => Code is generated and user is prompted
        # Client ID and redirect_uri are optional

        # Code, code_verifier, client ID and redirect_uri are provided => Take over
        # Scope, audience, and token_url are optional

        self.organization = organization
        self.redirect_uri = (
            redirect_uri
            or os.getenv("PULSE_REDIRECT_URI")
            or "http://localhost:8888/callback"
        )
        self.code_verifier = code_verifier
        self.scope = scope
        # Track whether audience or scope were explicitly provided
        self._provided_audience = audience is not None
        self._provided_env_audience = os.getenv("PULSE_API_URL") is not None
        self._provided_scope = scope is not None
        self._provided_env_scope = os.getenv("PULSE_SCOPE") is not None
        self._refresh_token_value: str | None = None

        self.code = code or self._get_code()

    def _get_code(self) -> str:
        # Generate PKCE code verifier and challenge
        self.code_verifier = secrets.token_urlsafe(64)
        code_challenge = (
            base64.urlsafe_b64encode(
                hashlib.sha256(self.code_verifier.encode()).digest()
            )
            .decode()
            .rstrip("=")
        )
        # Retrieve configuration from environment

        # redirect_uri = "http://localhost:8888/callback"
        authorize_url = f"https://{PROD_AUTH_DOMAIN}/authorize"

        params: dict[str, str] = {
            "response_type": "code",
            "client_id": self.client_id or PROD_CLIENT_ID,
            "audience": self.audience or PROD_BASE_URL,
            "redirect_uri": self.redirect_uri,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "scope": self.scope or DEFAULT_SCOPES,
        }

        if self.organization:
            params["organization"] = self.organization

        auth_url = f"{authorize_url}?{urlencode(params)}"

        print("Please authenticate by visiting the following URL in your browser:")
        print(auth_url)
        try:
            webbrowser.open(auth_url)
        except Exception:
            pass

        # Attempt to receive the authorization code via a local HTTP callback
        code: str | None = None
        try:
            parsed_uri = urlparse(self.redirect_uri)
            host = parsed_uri.hostname or "localhost"
            port = parsed_uri.port or 8888
            path = parsed_uri.path or "/callback"
            from http.server import HTTPServer, BaseHTTPRequestHandler

            class _CallbackHandler(BaseHTTPRequestHandler):
                def do_GET(self_inner):
                    qs = parse_qs(urlparse(self_inner.path).query)
                    if "code" in qs:
                        self_inner.server.code = qs["code"][0]
                    self_inner.send_response(200)
                    self_inner.send_header("Content-Type", "text/html")
                    self_inner.end_headers()
                    self_inner.wfile.write(
                        b"<html><body>You may now close this window.</body></html>"
                    )

                def log_message(self_inner, format, *args):
                    return

            httpd = HTTPServer((host, port), _CallbackHandler)
            print(f"Listening for callback at http://{host}:{port}{path} ...")
            httpd.handle_request()
            code = getattr(httpd, "code", None)
        except Exception:
            code = None
        # Fallback to manual prompt if no code received
        if not code:
            code_input = input(
                "Enter the full redirect URL or authorization code: "
            ).strip()
            try:
                parsed = urlparse(code_input)
                qs = parse_qs(parsed.query)
                code = qs.get("code", [code_input])[0]
            except Exception:
                code = code_input

        return code

    def _refresh_token(self) -> None:
        # Build token request payload
        data: dict[str, str] = {
            "grant_type": "authorization_code",
            "client_id": self.client_id or PROD_CLIENT_ID,
            "code": self.code,
            "redirect_uri": self.redirect_uri,
            "code_verifier": self.code_verifier,
        }
        # Include audience if provided via init or environment
        if self._provided_audience or self._provided_env_audience:
            data["audience"] = self.audience  # type: ignore
        # Include scope if provided via init or environment
        if self._provided_scope or self._provided_env_scope:
            data["scope"] = self.scope  # type: ignore
        resp = httpx.post(self.token_url, data=data)
        resp.raise_for_status()
        payload = resp.json()
        self._access_token = payload.get("access_token")
        # Store refresh token if provided
        self._refresh_token_value = payload.get("refresh_token")
        expires_in = payload.get("expires_in", 3600)
        self._expires_at = time.time() + float(expires_in) - 60.0


def _get_default_auth() -> AuthorizationCodePKCEAuth:
    """Interactive PKCE Authorization Code flow for CLI usage."""
    return AuthorizationCodePKCEAuth()


def _in_jupyter_notebook() -> bool:
    """Return True if running in a Jupyter notebook (ZMQInteractiveShell)."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    try:
        ip = get_ipython()
        if ip is None:
            return False
        return ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def auto_auth():
    """
    Automatically retrieves and returns the default authentication credentials.

    This function delegates to `_get_default_auth()` to determine and
    provide the appropriate default authentication mechanism.

    Returns:
        The default authentication credentials or handler.
    """
    if (
        len(os.environ.get("PULSE_CLIENT_SECRET", "")) > 0
        and len(os.environ.get("PULSE_CLIENT_ID", "")) > 0
    ):
        return ClientCredentialsAuth()
    return _get_default_auth()
