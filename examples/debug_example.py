#!/usr/bin/env python3
"""
Example demonstrating the Pulse SDK debugging and introspection tools.

This script shows how to:
1. Enable debug mode with various configurations
2. Monitor HTTP requests and responses
3. Track authentication token status
4. Collect performance metrics and cache statistics
5. Use granular logging categories

Run with: python examples/debug_example.py
"""

import os
import time
from pulse.debug import (
    enable_debug,
    disable_debug,
    get_debug_config,
    get_debug_stats,
    clear_debug_stats,
)
from pulse.core.client import CoreClient
from pulse.analysis.analyzer import Analyzer


def demonstrate_basic_debug():
    """Demonstrate basic debug functionality."""
    print("=== Basic Debug Mode ===")

    # Enable debug mode with default settings
    enable_debug()

    config = get_debug_config()
    print(f"Debug enabled: {config.enabled}")
    print(f"Log requests: {config.log_requests}")
    print(f"Log responses: {config.log_responses}")
    print(f"Mask credentials: {config.mask_credentials}")
    print()


def demonstrate_custom_debug_config():
    """Demonstrate custom debug configuration."""
    print("=== Custom Debug Configuration ===")

    # Enable debug with custom settings
    enable_debug(
        log_requests=True,
        log_responses=False,  # Disable response logging
        log_timing=True,
        log_cache_stats=True,
        log_auth_status=True,
        mask_credentials=True,
        max_body_size=5000,  # Smaller body size limit
        log_level="INFO",
        categories=["requests", "timing", "auth"],  # Only specific categories
    )

    config = get_debug_config()
    print(f"Custom config - Log responses: {config.log_responses}")
    print(f"Custom config - Max body size: {config.max_body_size}")
    print(f"Custom config - Categories: {config.categories}")
    print()


def demonstrate_auth_inspection():
    """Demonstrate authentication token inspection."""
    print("=== Authentication Token Inspection ===")

    try:
        # Create a client (this will attempt to authenticate)
        client = CoreClient.with_client_credentials()

        # Inspect the authentication status
        token_info = client.debug_auth_status()

        print(f"Has token: {token_info.has_token}")
        print(f"Token type: {token_info.token_type}")
        print(f"Is valid: {token_info.is_valid}")
        print(f"Is expired: {token_info.is_expired}")
        print(f"Masked token: {token_info.masked_token}")

        if token_info.expires_in:
            print(f"Expires in: {token_info.expires_in:.0f} seconds")

        client.close()

    except Exception as e:
        print(f"Authentication failed (expected in demo): {e}")

    print()


def demonstrate_request_debugging():
    """Demonstrate HTTP request debugging."""
    print("=== HTTP Request Debugging ===")

    # Clear previous stats
    clear_debug_stats()

    try:
        # Make some API calls to generate debug output
        client = CoreClient.with_client_credentials()

        # This will log the request/response details
        texts = ["Hello world", "This is a test"]
        from pulse.core.models import EmbeddingsRequest

        request = EmbeddingsRequest(inputs=texts, fast=True)

        print("Making embeddings request...")
        response = client.create_embeddings(request)
        print(f"Response received with {len(response.results)} embeddings")

        client.close()

    except Exception as e:
        print(f"API call failed (expected in demo): {e}")

    # Show debug statistics
    stats = get_debug_stats()
    print("\nDebug Statistics:")
    print(f"Total requests: {stats.total_requests}")
    print(f"Failed requests: {stats.failed_requests}")
    print(f"Success rate: {stats.success_rate:.1f}%")
    print(f"Auth refreshes: {stats.auth_refreshes}")

    if stats.request_timings:
        print(f"Average request time: {stats.average_request_time:.3f}s")
        print("Request timings:")
        for timing in stats.request_timings[-3:]:  # Show last 3
            print(f"  {timing.method} {timing.url} - {timing.duration:.3f}s")

    print()


def demonstrate_cache_debugging():
    """Demonstrate cache hit/miss debugging."""
    print("=== Cache Statistics Debugging ===")

    # Clear previous stats
    clear_debug_stats()

    try:
        # Use analyzer with caching enabled
        texts = [
            "I love this product, it's amazing!",
            "This is terrible, worst purchase ever.",
            "It's okay, nothing special.",
            "Absolutely fantastic, highly recommend!",
        ]

        print("Running analysis with caching...")
        with Analyzer(dataset=texts, cache_dir=".pulse_cache") as analyzer:
            # Run the same analysis twice to see cache hits
            print("First run (cache misses expected):")
            analyzer.run()

            print("Second run (cache hits expected):")
            analyzer.run()

        # Show cache statistics
        stats = get_debug_stats()
        cache_stats = stats.cache_stats

        print("\nCache Statistics:")
        print(f"Total cache requests: {cache_stats.total_requests}")
        print(f"Cache hits: {cache_stats.hits}")
        print(f"Cache misses: {cache_stats.misses}")
        print(f"Hit rate: {cache_stats.hit_rate:.1f}%")

    except Exception as e:
        print(f"Cache demo failed (expected without proper auth): {e}")

    print()


def demonstrate_granular_logging():
    """Demonstrate granular logging categories."""
    print("=== Granular Logging Categories ===")

    # Enable only specific categories
    enable_debug(categories=["timing", "cache"])

    print("Enabled categories: timing, cache")
    print("This will log timing and cache events, but not requests/responses")

    # Simulate some operations
    from pulse.debug import log_cache_hit, log_cache_miss, debug_request

    with debug_request("GET", "https://example.com/test"):
        time.sleep(0.1)

    log_cache_hit("example_key")
    log_cache_miss("another_key")

    print("Check the logs above - only timing and cache events should appear")
    print()


def demonstrate_environment_activation():
    """Demonstrate environment variable activation."""
    print("=== Environment Variable Activation ===")

    # Show current environment variable
    pulse_debug = os.getenv("PULSE_DEBUG", "not set")
    print(f"PULSE_DEBUG environment variable: {pulse_debug}")

    if pulse_debug.lower() in ("true", "1", "yes", "on"):
        print("Debug mode was automatically enabled via environment variable")
    else:
        print("To auto-enable debug mode, set: export PULSE_DEBUG=true")

    print()


def demonstrate_credential_masking():
    """Demonstrate credential masking in logs."""
    print("=== Credential Masking ===")

    from pulse.debug import mask_credentials

    # Test data with sensitive information
    sensitive_data = {
        "client_id": "my_client_id_12345",
        "client_secret": "super_secret_key_67890",
        "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "safe_field": "this_is_safe_to_log",
    }

    print("Original data:")
    print(f"  client_id: {sensitive_data['client_id']}")
    print(f"  client_secret: {sensitive_data['client_secret']}")
    print(f"  access_token: {sensitive_data['access_token']}")
    print(f"  safe_field: {sensitive_data['safe_field']}")

    masked_data = mask_credentials(sensitive_data)

    print("\nMasked data:")
    print(f"  client_id: {masked_data['client_id']}")
    print(f"  client_secret: {masked_data['client_secret']}")
    print(f"  access_token: {masked_data['access_token']}")
    print(f"  safe_field: {masked_data['safe_field']}")

    print()


def main():
    """Run all debug demonstrations."""
    print("Pulse SDK Debug Tools Demonstration")
    print("=" * 50)
    print()

    demonstrate_basic_debug()
    demonstrate_custom_debug_config()
    demonstrate_environment_activation()
    demonstrate_credential_masking()
    demonstrate_auth_inspection()
    demonstrate_request_debugging()
    demonstrate_cache_debugging()
    demonstrate_granular_logging()

    # Clean up
    disable_debug()
    print("=== Debug Mode Disabled ===")
    print("Debug demonstration complete!")


if __name__ == "__main__":
    main()
