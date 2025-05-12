#!/usr/bin/env python3


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


def main():
    sentiment = sentiment_analysis(input_data=["AI is making my life easier."])

    print("Sentiment Analysis Results:")
    for result in sentiment:
        print(result.summary())
        print("-" * 40)


if __name__ == "__main__":
    main()
