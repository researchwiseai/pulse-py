# Authentication

The SDK supports two OAuth2 flows via HTTPX-compatible auth classes. If you do not provide an auth object, the client attempts to create one automatically based on your environment.

## Auth Classes (module: `pulse.auth`)

### `ClientCredentialsAuth`
OAuth2 Client Credentials flow. Suitable for server-to-server usage.

Constructor parameters:
- `client_id: str | None` – OAuth client ID. Falls back to `PULSE_CLIENT_ID` or `PROD_CLIENT_ID`.
- `client_secret: str | None` – OAuth client secret. Falls back to `PULSE_CLIENT_SECRET`. Required for this flow.
- `token_url: str | None` – Token endpoint. Falls back to `PULSE_TOKEN_URL` or `https://{PROD_AUTH_DOMAIN}/oauth/token`.
- `audience: str | None` – Audience for the token. Falls back to `PULSE_API_URL` or `PROD_BASE_URL`.

Behavior:
- Fetches an access token on demand and refreshes when expired (uses `expires_in` minus a 60s buffer).
- Adds `Authorization: Bearer <token>` to requests targeting `*.core.researchwiseai.com`.

Example:
```python
from pulse.auth import ClientCredentialsAuth
from pulse.core.client import CoreClient

auth = ClientCredentialsAuth(
    client_id="...",
    client_secret="...",
    token_url="https://YOUR_DOMAIN/oauth/token",
    audience="https://core.researchwiseai.com/pulse/v1",
)
client = CoreClient(auth=auth)
```

### `AuthorizationCodePKCEAuth`
OAuth2 Authorization Code flow with PKCE. Good for interactive or user-facing contexts.

Constructor parameters:
- `code: str | None` – Authorization code. Optional when using the interactive prompt.
- `code_verifier: str | None` – PKCE verifier. Optional when using the interactive prompt (auto-generated).
- `client_id: str | None` – OAuth client ID; falls back to `PULSE_CLIENT_ID` or `PROD_CLIENT_ID`.
- `redirect_uri: str | None` – Redirect URI; falls back to `PULSE_REDIRECT_URI` or `http://localhost:8888/callback`.
- `token_url: str | None` – Token endpoint; falls back to `PULSE_TOKEN_URL`.
- `audience: str | None` – Audience; falls back to `PULSE_API_URL` or `PROD_BASE_URL`.
- `scope: str | None` – Space‑separated scopes; falls back to `PULSE_SCOPE` or `DEFAULT_SCOPES`.
- `organization: str | None` – Optional org parameter for the authorization request.

Behavior:
- If `code` is not provided, opens a browser to begin the PKCE flow and listens on the redirect URI port for a one‑time callback; falls back to a manual prompt for the redirect URL or code.
- Exchanges the authorization code for an access token (and optional refresh token) and refreshes when near expiry.
- Adds `Authorization: Bearer <token>` only for hosts matching `*.core.researchwiseai.com`.

Example (interactive):
```python
from pulse.auth import AuthorizationCodePKCEAuth
from pulse.core.client import CoreClient

auth = AuthorizationCodePKCEAuth(
    client_id="...",
    redirect_uri="http://localhost:8888/callback",
)
client = CoreClient(auth=auth)
```

## Auto Authentication

`pulse.auth.auto_auth()` selects a default auth automatically:
- If both `PULSE_CLIENT_ID` and `PULSE_CLIENT_SECRET` are set, it returns `ClientCredentialsAuth()`.
- Otherwise, it returns `AuthorizationCodePKCEAuth()` and will run an interactive PKCE flow.

`CoreClient()` uses `auto_auth()` when constructed without a preconfigured `httpx.Client` and without an explicit `auth` parameter.

## Client Shortcuts

The core client provides convenience constructors that wire up auth for you:

### `CoreClient.with_client_credentials(...)`
Parameters (resolved in order: explicit argument → environment variable → default):
- `client_id`, `client_secret` (env: `PULSE_CLIENT_ID`, `PULSE_CLIENT_SECRET`) – required.
- `audience` (env: `PULSE_AUDIENCE`) – optional.
- `token_url` (env: `PULSE_TOKEN_URL`) – default Auth0 token URL if omitted.
- `base_url` (env: `PULSE_BASE_URL`) – defaults to `PROD_BASE_URL`.
- `organization` – reserved for future use.

Returns a ready‑to‑use `CoreClient` with `ClientCredentialsAuth`.

### `CoreClient.with_pkce(...)`
Parameters (resolved in order: explicit argument → environment variable → default):
- Required: `code` and `code_verifier` (the auth code and PKCE verifier).
- `client_id` (env: `PULSE_CLIENT_ID`) – required.
- `redirect_uri` (env: `PULSE_REDIRECT_URI`) – required.
- `base_url` (env: `PULSE_BASE_URL`) – defaults to `PROD_BASE_URL`.
- `token_url` (env: `PULSE_TOKEN_URL`) – default Auth0 token URL if omitted.
- `scope` (env: `PULSE_SCOPE`) – optional.

Returns a ready‑to‑use `CoreClient` configured with `AuthorizationCodePKCEAuth`.

