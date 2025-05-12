"""Configuration for Pulse Client."""

from typing import Literal

DEV_BASE_URL = "https://dev.core.researchwiseai.com/pulse/v1"
PROD_BASE_URL = "https://core.researchwiseai.com/pulse/v1"
PROD_CLIENT_ID = "9LJJxxJjm90HjKW5cWTyFNZ2o0mF0pZs"
PROD_AUTH_DOMAIN = "research-wise-ai-eu.eu.auth0.com"
DEFAULT_SCOPES = "openid profile email"
DEFAULT_TIMEOUT = 30.0  # seconds
DEFAULT_RETRIES = 3

Environment = Literal["dev", "prod"]
