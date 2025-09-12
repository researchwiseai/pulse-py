# Configuration

The SDK ships with sane defaults and can be configured programmatically or via environment variables.

## Module: `pulse.config`

- `DEV_BASE_URL` (str): `https://dev.core.researchwiseai.com/pulse/v1`
- `PROD_BASE_URL` (str): `https://core.researchwiseai.com/pulse/v1`
- `PROD_CLIENT_ID` (str): Default Auth0 client ID for production.
- `PROD_AUTH_DOMAIN` (str): Default Auth0 domain for production.
- `DEFAULT_SCOPES` (str): Default OAuth scopes used by PKCE (`"openid profile email"`).
- `DEFAULT_TIMEOUT` (float): Default HTTP timeout in seconds (30.0).
- `DEFAULT_RETRIES` (int): Default retry attempts for transient errors (3).

You generally do not need to modify these unless you are targeting a different environment.

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

