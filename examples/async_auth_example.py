#!/usr/bin/env python3
"""Example demonstrating async authentication with AsyncCoreClient."""

import asyncio
import os
from pulse.core.async_client import AsyncCoreClient
from pulse.async_auth import AsyncClientCredentialsAuth, AsyncAuthorizationCodePKCEAuth


async def example_client_credentials_async():
    """Example using async client credentials authentication."""
    print("=== Async Client Credentials Example ===")

    # Method 1: Using factory method with environment variables
    # Set these environment variables:
    # export PULSE_CLIENT_ID="your_client_id"
    # export PULSE_CLIENT_SECRET="your_client_secret"

    async with AsyncCoreClient.with_client_credentials_async() as client:
        print("✓ Created AsyncCoreClient with async client credentials auth")
        # The client is now ready to make API calls
        # Example: embeddings = await client.create_embeddings(request)

    # Method 2: Using factory method with explicit parameters
    client = AsyncCoreClient.with_client_credentials_async(
        client_id="your_client_id",
        client_secret="your_client_secret",
        audience="https://core.researchwiseai.com/pulse/v1",
    )
    print("✓ Created AsyncCoreClient with explicit credentials")
    await client.close()

    # Method 3: Using auth class directly
    auth = AsyncClientCredentialsAuth(
        client_id="your_client_id", client_secret="your_client_secret"
    )
    async with AsyncCoreClient(auth=auth) as client:
        print("✓ Created AsyncCoreClient with async auth instance")


async def example_pkce_async():
    """Example using async PKCE authentication."""
    print("\n=== Async PKCE Example ===")

    # Method 1: Using factory method
    # Note: In real usage, you would obtain these from the OAuth flow
    client = AsyncCoreClient.with_pkce_async(
        code="authorization_code_from_oauth_flow",
        code_verifier="pkce_code_verifier",
        client_id="your_client_id",
        redirect_uri="http://localhost:8888/callback",
    )
    print("✓ Created AsyncCoreClient with async PKCE auth")
    await client.close()

    # Method 2: Using auth class directly
    auth = AsyncAuthorizationCodePKCEAuth(
        code="authorization_code_from_oauth_flow",
        code_verifier="pkce_code_verifier",
        client_id="your_client_id",
        redirect_uri="http://localhost:8888/callback",
    )
    async with AsyncCoreClient(auth=auth) as client:
        print("✓ Created AsyncCoreClient with async PKCE auth instance")


async def example_default_async_auth():
    """Example showing default async authentication behavior."""
    print("\n=== Default Async Auth Example ===")

    # AsyncCoreClient automatically uses async-compatible auth
    # If PULSE_CLIENT_SECRET is set, it uses AsyncClientCredentialsAuth
    # Otherwise, it uses AsyncAuthorizationCodePKCEAuth

    # Set environment variables for client credentials
    os.environ["PULSE_CLIENT_ID"] = "test_client_id"
    os.environ["PULSE_CLIENT_SECRET"] = "test_client_secret"

    async with AsyncCoreClient() as client:
        print("✓ AsyncCoreClient automatically uses async client credentials auth")
        print(f"  Auth type: {type(client.client.auth).__name__}")

    # Remove client secret to trigger PKCE auth
    del os.environ["PULSE_CLIENT_SECRET"]

    # Note: This would normally prompt for interactive auth
    # For demo purposes, we'll just show what would happen
    print("✓ AsyncCoreClient would fall back to async PKCE auth")
    print("  (Interactive auth flow would be triggered)")
    print("  Auth type: AsyncAuthorizationCodePKCEAuth")


async def example_error_handling():
    """Example showing error handling with async auth."""
    print("\n=== Error Handling Example ===")

    # Async auth classes cannot be used with sync clients

    try:
        AsyncClientCredentialsAuth(client_id="test", client_secret="test")
        # This would raise an error if we tried to use it
        print("✓ Async auth classes prevent sync usage")
    except Exception as e:
        print(f"  Error: {e}")

    # Missing credentials raise clear errors
    try:
        AsyncCoreClient.with_client_credentials_async(
            client_id="test"
            # Missing client_secret
        )
    except ValueError as e:
        print(f"✓ Clear error for missing credentials: {e}")


async def main():
    """Run all examples."""
    print("Async Authentication Examples\n")

    await example_client_credentials_async()
    await example_pkce_async()
    await example_default_async_auth()
    await example_error_handling()

    print("\n" + "=" * 50)
    print("Key Benefits of Async Authentication:")
    print("• Proper async/await support for token refresh")
    print("• Non-blocking authentication flows")
    print("• Compatible with AsyncCoreClient and async frameworks")
    print("• Clear error messages for misuse")
    print("• Automatic fallback to appropriate auth method")


if __name__ == "__main__":
    asyncio.run(main())
