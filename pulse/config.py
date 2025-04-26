"""Configuration for Pulse Client."""

from typing import Literal

DEV_BASE_URL = "https://dev.core.researchwiseai.com/pulse/v1"
PROD_BASE_URL = "https://core.researchwiseai.com/pulse/v1"
DEFAULT_TIMEOUT = 30.0  # seconds
DEFAULT_RETRIES = 3

Environment = Literal["dev", "prod"]
