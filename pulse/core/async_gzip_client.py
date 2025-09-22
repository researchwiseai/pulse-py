"""Async HTTPX client that transparently gzip-compresses request content with optimized connection pooling."""

import httpx
import gzip
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class AsyncConnectionPoolConfig:
    """Configuration for async connection pool optimization."""

    max_keepalive_connections: int = 20
    max_connections: int = 100
    keepalive_expiry: float = 30.0
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    write_timeout: float = 10.0
    pool_timeout: float = 5.0

    @classmethod
    def for_batch_operations(
        cls, concurrent_jobs: int = 5
    ) -> "AsyncConnectionPoolConfig":
        """Create optimized config for batch operations.

        Args:
            concurrent_jobs: Expected number of concurrent jobs

        Returns:
            Optimized connection pool configuration
        """
        if concurrent_jobs <= 5:
            return cls(
                max_keepalive_connections=15,
                max_connections=30,
                keepalive_expiry=60.0,
            )
        elif concurrent_jobs <= 15:
            return cls(
                max_keepalive_connections=25,
                max_connections=60,
                keepalive_expiry=90.0,
            )
        else:
            return cls(
                max_keepalive_connections=40,
                max_connections=120,
                keepalive_expiry=120.0,
                connect_timeout=15.0,
                read_timeout=60.0,
            )

    @classmethod
    def for_high_throughput(cls) -> "AsyncConnectionPoolConfig":
        """Create config optimized for high throughput operations.

        Returns:
            High throughput connection pool configuration
        """
        return cls(
            max_keepalive_connections=50,
            max_connections=200,
            keepalive_expiry=300.0,
            connect_timeout=20.0,
            read_timeout=120.0,
            write_timeout=30.0,
            pool_timeout=10.0,
        )


class AsyncGzipClient(httpx.AsyncClient):
    """Async HTTPX client that compresses request content with gzip and optimized connection pooling."""

    def __init__(
        self,
        *,
        pool_config: Optional[AsyncConnectionPoolConfig] = None,
        enable_compression: bool = True,
        compression_level: int = 6,
        **kwargs,
    ):
        """Initialize AsyncGzipClient with connection pool optimization.

        Args:
            pool_config: Connection pool configuration. Uses defaults if None.
            enable_compression: Whether to enable gzip compression.
            compression_level: Gzip compression level (1-9, higher = better compression).
            **kwargs: Additional arguments passed to httpx.AsyncClient
        """
        self.enable_compression = enable_compression
        self.compression_level = compression_level

        # Apply connection pool configuration
        if pool_config is None:
            pool_config = AsyncConnectionPoolConfig()

        # Set up connection limits
        limits = httpx.Limits(
            max_keepalive_connections=pool_config.max_keepalive_connections,
            max_connections=pool_config.max_connections,
            keepalive_expiry=pool_config.keepalive_expiry,
        )

        # Set up timeouts
        timeout = httpx.Timeout(
            connect=pool_config.connect_timeout,
            read=pool_config.read_timeout,
            write=pool_config.write_timeout,
            pool=pool_config.pool_timeout,
        )

        # Set optimized headers for batch operations
        default_headers = {
            "Connection": "keep-alive",
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": "pulse-sdk-async/1.0",
        }

        # Merge with user-provided headers
        headers = kwargs.pop("headers", {})
        default_headers.update(headers)

        super().__init__(
            limits=limits, timeout=timeout, headers=default_headers, **kwargs
        )

        # Track connection statistics
        self._connection_stats = {
            "requests_made": 0,
            "connections_created": 0,
            "connections_reused": 0,
            "compression_ratio": 0.0,
            "total_bytes_sent": 0,
            "total_bytes_compressed": 0,
        }

    def build_request(self, method: str, url: str, **kwargs) -> httpx.Request:
        """Build request with gzip compression when content is provided."""
        self._connection_stats["requests_made"] += 1

        # Only compress when the user explicitly passed `content` and compression is enabled
        if self.enable_compression and "content" in kwargs and kwargs["content"]:
            original = kwargs["content"]
            # Ensure content is bytes
            if isinstance(original, str):
                original = original.encode("utf-8")

            original_size = len(original)
            compressed = gzip.compress(original, compresslevel=self.compression_level)
            compressed_size = len(compressed)

            # Update compression statistics
            self._connection_stats["total_bytes_sent"] += original_size
            self._connection_stats["total_bytes_compressed"] += compressed_size

            # Calculate running compression ratio
            if self._connection_stats["total_bytes_sent"] > 0:
                self._connection_stats["compression_ratio"] = (
                    self._connection_stats["total_bytes_compressed"]
                    / self._connection_stats["total_bytes_sent"]
                )

            # Update the kwargs for the request
            kwargs["content"] = compressed
            headers = kwargs.setdefault("headers", {})
            headers["Content-Encoding"] = "gzip"
            headers["Content-Length"] = str(compressed_size)

        return super().build_request(method, url, **kwargs)

    async def __aenter__(self) -> "AsyncGzipClient":
        """Async context manager entry."""
        await super().__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with proper resource cleanup."""
        await super().__aexit__(exc_type, exc_val, exc_tb)

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection and compression statistics.

        Returns:
            Dictionary with connection statistics
        """
        return self._connection_stats.copy()

    def reset_stats(self) -> None:
        """Reset connection statistics."""
        self._connection_stats = {
            "requests_made": 0,
            "connections_created": 0,
            "connections_reused": 0,
            "compression_ratio": 0.0,
            "total_bytes_sent": 0,
            "total_bytes_compressed": 0,
        }

    @classmethod
    def create_for_batch_operations(
        cls,
        base_url: str,
        concurrent_jobs: int = 5,
        auth: Optional[httpx.Auth] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> "AsyncGzipClient":
        """Create AsyncGzipClient optimized for batch operations.

        Args:
            base_url: Base URL for the client
            concurrent_jobs: Expected number of concurrent jobs
            auth: Optional authentication
            timeout: Optional timeout override
            **kwargs: Additional arguments

        Returns:
            AsyncGzipClient optimized for batch operations
        """
        pool_config = AsyncConnectionPoolConfig.for_batch_operations(concurrent_jobs)

        # Handle timeout parameter separately to avoid conflicts
        client_kwargs = kwargs.copy()
        if timeout is not None:
            # Override the pool config timeout with explicit timeout
            pool_config.read_timeout = timeout
            pool_config.connect_timeout = min(timeout / 3, 10.0)

        return cls(
            base_url=base_url, pool_config=pool_config, auth=auth, **client_kwargs
        )

    @classmethod
    def create_for_high_throughput(
        cls,
        base_url: str,
        auth: Optional[httpx.Auth] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> "AsyncGzipClient":
        """Create AsyncGzipClient optimized for high throughput operations.

        Args:
            base_url: Base URL for the client
            auth: Optional authentication
            timeout: Optional timeout override
            **kwargs: Additional arguments

        Returns:
            AsyncGzipClient optimized for high throughput
        """
        pool_config = AsyncConnectionPoolConfig.for_high_throughput()

        # Handle timeout parameter separately to avoid conflicts
        client_kwargs = kwargs.copy()
        if timeout is not None:
            # Override the pool config timeout with explicit timeout
            pool_config.read_timeout = timeout
            pool_config.connect_timeout = min(timeout / 3, 20.0)

        return cls(
            base_url=base_url, pool_config=pool_config, auth=auth, **client_kwargs
        )
