from __future__ import annotations

import time
from typing import Callable, Iterable, Optional

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
    last_exc: Optional[BaseException] = None
    while True:
        try:
            response = func()
            if response.status_code not in retry_statuses:
                return response
            # Successful request (even if it's a retryable status); clear last exception
            last_exc = None
        except httpx.TransportError as exc:
            response = None
            last_exc = exc
        attempt += 1
        if attempt >= max_attempts:
            if response is not None:
                return response
            # Re-raise the last transport error if available; otherwise raise a
            # descriptive RuntimeError so callers get a meaningful failure.
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(
                "Request failed after retries with no response or exception captured"
            )
        time.sleep(delay)
        delay *= 2
