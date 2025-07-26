"""Exceptions for Pulse Client API errors."""

from __future__ import annotations

from typing import Any, Optional

import httpx


class PulseAPIError(Exception):
    """Represents an error returned by the Pulse API."""

    def __init__(self, response: httpx.Response) -> None:
        self.status: int = response.status_code
        self.code: Optional[str] = None
        self.message: Optional[str] = None
        try:
            body: Any = response.json()
        except ValueError:
            body = response.text

        if isinstance(body, dict):
            self.code = body.get("code")
            self.message = body.get("message") or response.reason_phrase
            self.body = body
        else:
            self.body = body
            if isinstance(body, str):
                self.message = body
            else:
                self.message = response.reason_phrase

        super().__init__(str(self))

    def __str__(self) -> str:  # pragma: no cover - simple formatting
        parts = [f"{self.status}"]
        if self.code is not None:
            parts.append(str(self.code))
        msg = self.message or ""
        return f"Pulse API Error {' '.join(parts)}: {msg}"


class TimeoutError(Exception):
    """Error thrown when an HTTP request times out."""

    def __init__(self, url: str, timeout: float) -> None:
        super().__init__(f"Request to {url} timed out after {timeout}ms")
        self.url = url
        self.timeout = timeout


class NetworkError(Exception):
    """Error thrown when a network error occurs during a request."""

    def __init__(self, url: str, cause: Exception) -> None:
        super().__init__(f"Network error while requesting {url}: {cause}")
        self.url = url
        self.cause = cause
