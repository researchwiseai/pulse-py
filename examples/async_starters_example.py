"""Example demonstrating async starter functions."""

import asyncio
from pulse.async_starters import (
    generate_themes_async,
    sentiment_analysis_async,
    theme_allocation_async,
    compare_similarity_async,
    cluster_analysis_async,
    extract_elements_async,
    summarize_async,
    estimate_usage_async,
)


async def main():
    """Demonstrate async starter functions."""

    # Sample data
    sample_texts = [
        "The food was excellent but the service was slow",
        "Great atmosphere and friendly staff",
        "Terrible experience, would not recommend",
        "Amazing food quality and quick service",
        "The restaurant was too noisy and crowded",
    ]

    print("=== Async Starter Functions Demo ===\n")

    # 1. Sentiment Analysis
    print("1. Sentiment Analysis:")
    try:
        sentiment_result = await sentiment_analysis_async(sample_texts)
        print(f"   Analyzed {len(sentiment_result.results)} texts")
        for i, result in enumerate(sentiment_result.results[:2]):  # Show first 2
            print(
                f"   Text {i+1}: {result.sentiment} "
                f"(confidence: {result.confidence:.2f})"
            )
    except Exception as e:
        print(f"   Error: {e}")
    print()

    # 2. Theme Generation
    print("2. Theme Generation:")
    try:
        themes_result = await generate_themes_async(
            sample_texts, min_themes=2, max_themes=4
        )
        print(f"   Generated {len(themes_result.themes)} themes:")
        for theme in themes_result.themes[:3]:  # Show first 3
            print(f"   - {theme.shortLabel}: {theme.description}")
    except Exception as e:
        print(f"   Error: {e}")
    print()

    # 3. Theme Allocation
    print("3. Theme Allocation:")
    try:
        allocation_result = await theme_allocation_async(
            sample_texts, themes=["Food Quality", "Service", "Atmosphere"]
        )
        print(f"   Allocated {len(allocation_result.texts)} texts to themes")
        for i, (text, theme) in enumerate(
            zip(allocation_result.texts[:2], allocation_result.allocated_themes[:2])
        ):
            print(f"   Text {i+1}: '{text[:30]}...' -> {theme}")
    except Exception as e:
        print(f"   Error: {e}")
    print()

    # 4. Similarity Comparison
    print("4. Similarity Comparison:")
    try:
        similarity_result = await compare_similarity_async(
            sample_texts[:3], flatten=True  # Use first 3 texts
        )
        print(f"   Computed similarity for {len(sample_texts[:3])} texts")
        sim_rows = len(similarity_result.similarity)
        sim_cols = (
            len(similarity_result.similarity[0]) if similarity_result.similarity else 0
        )
        print(f"   Similarity matrix shape: {sim_rows}x{sim_cols}")
    except Exception as e:
        print(f"   Error: {e}")
    print()

    # 5. Clustering
    print("5. Clustering:")
    try:
        cluster_result = await cluster_analysis_async(
            sample_texts, k=2, algorithm="kmeans"
        )
        print(
            f"   Clustered {len(cluster_result.inputs)} texts into "
            f"{len(set(cluster_result.labels))} clusters"
        )
        for i, (text, label) in enumerate(
            zip(cluster_result.inputs[:2], cluster_result.labels[:2])
        ):
            print(f"   Text {i+1}: Cluster {label} - '{text[:30]}...'")
    except Exception as e:
        print(f"   Error: {e}")
    print()

    # 6. Element Extraction
    print("6. Element Extraction:")
    try:
        extraction_result = await extract_elements_async(
            sample_texts,
            dictionary=["food", "service", "staff", "atmosphere"],
            type="named-entities",
        )
        print(f"   Extracted elements from {len(extraction_result.inputs)} texts")
        total_extractions = sum(
            len(result.extractions) for result in extraction_result.results
        )
        print(f"   Total extractions found: {total_extractions}")
    except Exception as e:
        print(f"   Error: {e}")
    print()

    # 7. Summarization
    print("7. Summarization:")
    try:
        summary_result = await summarize_async(
            sample_texts,
            question="What are the main points about this restaurant?",
            length="short",
        )
        print(f"   Generated summary from {len(summary_result.inputs)} texts")
        if summary_result.summaries:
            print(f"   Summary: {summary_result.summaries[0].summary[:100]}...")
    except Exception as e:
        print(f"   Error: {e}")
    print()

    # 8. Usage Estimation
    print("8. Usage Estimation:")
    try:
        usage_result = await estimate_usage_async("sentiment", sample_texts)
        print("   Estimated usage for sentiment analysis:")
        print(f"   Total credits: {usage_result.total}")
        if usage_result.records:
            print(
                f"   First record: {usage_result.records[0].quantity} units at "
                f"{usage_result.records[0].rate} credits/unit"
            )
    except Exception as e:
        print(f"   Error: {e}")
    print()

    print("=== Demo Complete ===")


async def concurrent_demo():
    """Demonstrate concurrent execution of async starters."""
    print("\n=== Concurrent Execution Demo ===\n")

    sample_texts = [
        "Great product, fast delivery",
        "Poor quality, disappointed",
        "Excellent customer service",
    ]

    try:
        # Run multiple operations concurrently
        tasks = [
            sentiment_analysis_async(sample_texts),
            estimate_usage_async("sentiment", sample_texts),
            compare_similarity_async(sample_texts, flatten=True),
        ]

        print(
            "Running sentiment analysis, usage estimation, and "
            "similarity comparison concurrently..."
        )
        results = await asyncio.gather(*tasks, return_exceptions=True)

        print("Concurrent operations completed:")
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"   Task {i+1}: Error - {result}")
            else:
                print(f"   Task {i+1}: Success - {type(result).__name__}")

    except Exception as e:
        print(f"   Concurrent execution error: {e}")

    print("\n=== Concurrent Demo Complete ===")


if __name__ == "__main__":
    # Note: This example requires valid authentication credentials
    # Set PULSE_CLIENT_ID and PULSE_CLIENT_SECRET environment variables

    print("This example demonstrates async starter functions.")
    print("Note: Requires valid authentication credentials in environment variables.")
    print("Set PULSE_CLIENT_ID and PULSE_CLIENT_SECRET to run with real API calls.\n")

    # Run the main demo
    asyncio.run(main())

    # Run the concurrent demo
    asyncio.run(concurrent_demo())
