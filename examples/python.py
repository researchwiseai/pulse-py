#!/usr/bin/env python3
import os

from pulse.auth import ClientCredentialsAuth

# Attempt to import the PulseClient from pulse_sdk.
# If pulse-sdk is not installed or the class name is different,
# this will raise an ImportError.
try:
    from pulse.starters import (
        sentiment_analysis,
    )  # Replace PulseClient if the class name is different
except ImportError:
    print("Error: Failed to import PulseClient from pulse_sdk.")
    print(
        "Please ensure that the 'pulse-sdk' is installed (e.g.,"
        + " 'pip install pulse-sdk')"
    )
    print(
        "and that 'PulseClient' is the correct name of the main "
        + "client class in the SDK."
    )
    exit(1)


def demonstrate_pulse_sdk_client_credentials():
    """
    Demonstrates initializing and using the pulse-sdk with the
    OAuth2 Client Credentials flow.

    This script expects the following environment variables to be set:
    - PULSE_CLIENT_ID: Your application's client ID.
    - PULSE_CLIENT_SECRET: Your application's client secret.
    - PULSE_API_BASE_URL (Optional): Override the default API base URL.
    """
    print("Pulse SDK Client Credentials Flow Demonstration")
    print("---------------------------------------------\n")

    # Retrieve credentials and optional base URL from environment variables
    client_id = ""
    client_secret = ""
    # Example: PULSE_API_BASE_URL="https://api.your-pulse-instance.com"
    os.getenv("PULSE_API_BASE_URL")

    if not client_id:
        print("Error: PULSE_CLIENT_ID environment variable is not set.")
        print("Please set it before running the script.")
        return

    if not client_secret:
        print("Error: PULSE_CLIENT_SECRET environment variable is not set.")
        print("Please set it before running the script.")
        return

    print("Attempting to initialize PulseClient...")
    print(f"Using Client ID: {client_id[:4]}****")  # Print partial ID for confirmation

    auth = ClientCredentialsAuth(
        client_id=client_id, client_secret=client_secret, organization=""
    )

    print("PulseClient initialized successfully.\n")

    sentiment = sentiment_analysis(texts=["AI is making my life easier."], auth=auth)

    print("Sentiment analysis result:")
    print(sentiment)


if __name__ == "__main__":
    demonstrate_pulse_sdk_client_credentials()
