"""Configuration for Pulse Client.

Supports environment-based toggling via `PULSE_ENV`:
  - prod (default): production endpoints
  - staging/dev: staging API and dev audience/auth domain

Individual values can be overridden via `PULSE_BASE_URL`, `PULSE_AUDIENCE`, and
`PULSE_AUTH_DOMAIN` environment variables.
"""

import os

ENV = os.getenv("PULSE_ENV", "prod").lower()

if ENV in ("staging", "dev"):
    # Staging / Dev defaults
    BASE_URL = os.getenv(
        "PULSE_BASE_URL", "https://staging.pulse.researchwiseai.com/v1"
    )
    AUDIENCE = os.getenv(
        "PULSE_AUDIENCE", "https://dev.core.researchwiseai.com/pulse/v1"
    )
    AUTH_DOMAIN = os.getenv("PULSE_AUTH_DOMAIN", "wise-dev.eu.auth0.com")
else:
    # Production defaults
    BASE_URL = os.getenv("PULSE_BASE_URL", "https://pulse.researchwiseai.com/v1")
    AUDIENCE = os.getenv("PULSE_AUDIENCE", "https://core.researchwiseai.com/pulse/v1")
    AUTH_DOMAIN = os.getenv("PULSE_AUTH_DOMAIN", "research-wise-ai-eu.eu.auth0.com")

CLIENT_ID = "9LJJxxJjm90HjKW5cWTyFNZ2o0mF0pZs"
DEFAULT_SCOPES = "openid profile email"
DEFAULT_TIMEOUT = 30.0  # seconds
DEFAULT_RETRIES = 3
