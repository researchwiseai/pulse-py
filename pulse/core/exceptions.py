"""Exceptions for Pulse Client API errors."""


class PulseAPIError(Exception):
    """Represents an error returned by the Pulse API."""

    def __init__(self, response):
        self.status_code = response.status_code
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        super().__init__(f"Status code: {self.status_code}, Detail: {detail}")
        self.detail = detail
