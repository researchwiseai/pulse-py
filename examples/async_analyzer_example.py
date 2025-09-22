"""Example demonstrating AsyncAnalyzer usage."""

import asyncio
from pulse.analysis.async_analyzer import AsyncAnalyzer
from pulse.analysis.processes import ThemeGeneration, SentimentProcess, ThemeAllocation


async def main():
    """Demonstrate AsyncAnalyzer with various processes."""

    # Sample data
    texts = [
        "I absolutely love this product! It's amazing.",
        "This is the best service I've ever used.",
        "Not bad, but could be improved.",
        "Terrible experience, very disappointed.",
        "Outstanding quality and great customer support.",
        "Average product, nothing special.",
    ]

    # Define processes to run
    processes = [
        ThemeGeneration(min_themes=2, max_themes=4),
        SentimentProcess(),
        ThemeAllocation(),  # This will depend on ThemeGeneration
    ]

    # Create AsyncAnalyzer with async context manager
    async with AsyncAnalyzer(
        dataset=texts,
        processes=processes,
        fast=True,  # Use fast mode for demo
        use_cache=False,  # Disable caching for demo
    ) as analyzer:
        print("Running async analysis...")

        # Run all processes asynchronously
        results = await analyzer.run()

        print("\n=== Results ===")

        # Access theme generation results
        if hasattr(results, "theme_generation"):
            themes = results.theme_generation
            print(f"\nGenerated {len(themes.themes)} themes:")
            for i, theme in enumerate(themes.themes):
                print(f"  {i+1}. {theme.shortLabel}: {theme.description}")
                print(f"     Representatives: {', '.join(theme.representatives[:3])}")

        # Access sentiment results
        if hasattr(results, "sentiment"):
            sentiment = results.sentiment
            print("\nSentiment Analysis:")
            for i, (text, result) in enumerate(zip(texts, sentiment.results)):
                sentiment_info = f"{result.sentiment} ({result.confidence:.2f})"
                print(f"  {i+1}. '{text[:50]}...' -> {sentiment_info}")

        # Access theme allocation results
        if hasattr(results, "theme_allocation"):
            allocation = results.theme_allocation
            print("\nTheme Allocation:")
            for i, (text, theme_idx) in enumerate(zip(texts, allocation.assignments)):
                theme_name = (
                    allocation.themes[theme_idx]
                    if theme_idx < len(allocation.themes)
                    else "Unknown"
                )
                print(f"  {i+1}. '{text[:50]}...' -> {theme_name}")

        print("\nAnalysis completed successfully!")


async def demonstrate_caching():
    """Demonstrate caching functionality."""
    import tempfile
    import os

    texts = ["Great product!", "Not so good", "Amazing service!"]
    processes = [SentimentProcess()]

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, "async_cache")

        print("\n=== Caching Demo ===")

        # First run - will hit the API
        print("First run (will call API)...")
        async with AsyncAnalyzer(
            dataset=texts,
            processes=processes,
            cache_dir=cache_dir,
            use_cache=True,
            fast=True,
        ) as analyzer:
            start_time = asyncio.get_event_loop().time()
            await analyzer.run()
            first_duration = asyncio.get_event_loop().time() - start_time

        # Second run - should use cache
        print("Second run (should use cache)...")
        async with AsyncAnalyzer(
            dataset=texts,
            processes=processes,
            cache_dir=cache_dir,
            use_cache=True,
            fast=True,
        ) as analyzer:
            start_time = asyncio.get_event_loop().time()
            await analyzer.run()
            second_duration = asyncio.get_event_loop().time() - start_time

        print(f"First run took: {first_duration:.3f}s")
        print(f"Second run took: {second_duration:.3f}s")
        print("Second run should be faster due to caching!")


if __name__ == "__main__":
    # Note: This example requires valid API credentials
    # Set PULSE_CLIENT_ID and PULSE_CLIENT_SECRET environment variables

    print("AsyncAnalyzer Example")
    print("====================")

    try:
        # Run the main example
        asyncio.run(main())

        # Run the caching demo
        asyncio.run(demonstrate_caching())

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: This example requires valid API credentials.")
        print("Set PULSE_CLIENT_ID and PULSE_CLIENT_SECRET environment variables.")
