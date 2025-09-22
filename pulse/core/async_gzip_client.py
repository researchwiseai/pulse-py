"""Async HTTPX client that transparently gzip-compresses request content."""

import httpx
import gzip


class AsyncGzipClient(httpx.AsyncClient):
    """Async HTTPX client that compresses request content with gzip when provided."""

    def build_request(self, method: str, url: str, **kwargs) -> httpx.Request:
        """Build request with gzip compression when content is provided."""
        # Only compress when the user explicitly passed `content`
        if "content" in kwargs and kwargs["content"]:
            original = kwargs["content"]
            # Ensure content is bytes
            if isinstance(original, str):
                original = original.encode("utf-8")
            compressed = gzip.compress(original)

            # Update the kwargs for the request
            kwargs["content"] = compressed
            headers = kwargs.setdefault("headers", {})
            headers["Content-Encoding"] = "gzip"
            headers["Content-Length"] = str(len(compressed))

        return super().build_request(method, url, **kwargs)

    async def __aenter__(self) -> "AsyncGzipClient":
        """Async context manager entry."""
        await super().__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with proper resource cleanup."""
        await super().__aexit__(exc_type, exc_val, exc_tb)
