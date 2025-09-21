# Configuration

The SDK ships with sane defaults and can be configured programmatically or via environment variables.

## Module: `pulse.config`

- `BASE_URL` (str): Default API base URL (`https://pulse.researchwiseai.com/v1`)
- `AUDIENCE` (str): Default OAuth2 audience (`https://core.researchwiseai.com/pulse/v1`)
- `AUTH_DOMAIN` (str): Default Auth0 domain (`research-wise-ai-eu.eu.auth0.com`)
- `CLIENT_ID` (str): Default Auth0 client ID
- `DEFAULT_SCOPES` (str): Default OAuth scopes used by PKCE (`"openid profile email"`)
- `DEFAULT_TIMEOUT` (float): Default HTTP timeout in seconds (30.0)
- `DEFAULT_RETRIES` (int): Default retry attempts for transient errors (3)

These values can be overridden via environment variables (`PULSE_BASE_URL`, `PULSE_AUDIENCE`, etc.) or by passing explicit parameters to client constructors.

## Environment Variables

The client and auth classes resolve configuration from environment variables when available.

- `PULSE_BASE_URL`: Overrides the API base URL used by `CoreClient` constructors.
- `PULSE_CLIENT_ID`: OAuth2 client ID.
- `PULSE_CLIENT_SECRET`: OAuth2 client secret (Client Credentials flow).
- `PULSE_TOKEN_URL`: OAuth2 token endpoint.
- `PULSE_AUDIENCE`: OAuth2 audience; defaults to the API base URL.
- `PULSE_API_URL`: Alias used by the auth module for the audience; if set, it is preferred.
- `PULSE_REDIRECT_URI`: Redirect URI used by PKCE.
- `PULSE_SCOPE`: Space‑separated OAuth scopes.

These are optional unless the chosen auth flow requires them (see Authentication).
