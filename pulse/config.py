"""Configuration for Pulse Client.

Individual values can be overridden via `PULSE_BASE_URL`, `PULSE_AUDIENCE`,
`PULSE_TOKEN_URL`, and `PULSE_AUTH_DOMAIN` environment variables.
"""

import os

# Production defaults - can be overridden via environment variables
BASE_URL = os.getenv("PULSE_BASE_URL", "https://pulse.researchwiseai.com/v1")
AUDIENCE = os.getenv("PULSE_AUDIENCE", "https://core.researchwiseai.com/pulse/v1")
AUTH_DOMAIN = os.getenv("PULSE_AUTH_DOMAIN", "research-wise-ai-eu.eu.auth0.com")

CLIENT_ID = "9LJJxxJjm90HjKW5cWTyFNZ2o0mF0pZs"
DEFAULT_SCOPES = "openid profile email"
DEFAULT_TIMEOUT = 30.0  # seconds
DEFAULT_RETRIES = 3
