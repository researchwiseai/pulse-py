"""HTTPX client that transparently gzip-compresses request content."""
import httpx
import gzip


class GzipClient(httpx.Client):
    """HTTPX client that compresses request content with gzip when provided."""

    def build_request(self, method: str, url: str, **kwargs) -> httpx.Request:
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
