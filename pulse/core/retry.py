from __future__ import annotations

import time
from typing import Callable, Iterable

import httpx

DEFAULT_RETRY_STATUSES = {429, 500, 502, 503, 504}
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF = 0.5


def retry_request(
    func: Callable[[], httpx.Response],
    *,
    retry_statuses: Iterable[int] = DEFAULT_RETRY_STATUSES,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF,
) -> httpx.Response:
    """Execute ``func`` retrying on network or transient server errors."""
    attempt = 0
    delay = backoff
    while True:
        try:
            response = func()
            if response.status_code not in retry_statuses:
                return response
        except httpx.TransportError:
            response = None
        attempt += 1
        if attempt >= max_attempts:
            if response is not None:
                return response
            raise
        time.sleep(delay)
        delay *= 2
