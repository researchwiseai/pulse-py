"""Example demonstrating AsyncWorkflow DSL usage."""

import asyncio
import os
from pulse.async_dsl import AsyncWorkflow


async def basic_async_workflow_example():
    """Basic async workflow example with theme generation and sentiment analysis."""

    # Sample data
    comments = [
        "The food was absolutely delicious and the service was outstanding!",
        "The restaurant was too noisy and the wait time was excessive.",
        "Great atmosphere and friendly staff, will definitely come back.",
        "The prices are reasonable and the portions are generous.",
        "Poor customer service and the food was cold when it arrived.",
    ]

    # Create async workflow
    async with AsyncWorkflow() as workflow:
        # Register data source
        workflow.source("comments", comments)

        # Build pipeline
        workflow.theme_generation(
            source="comments", min_themes=2, max_themes=5, fast=True
        ).sentiment(source="comments").theme_allocation(inputs="comments")

        # Execute workflow
        print("Running async workflow...")
        results = await workflow.run()

        # Access results
        print(f"Generated themes: {results.theme_generation.themes}")
        print(f"Sentiment analysis: {len(results.sentiment.sentiments)} sentiments")
        print(
            f"Theme allocation: {len(results.theme_allocation.assignments)} assignments"
        )


async def advanced_async_workflow_example():
    """Advanced async workflow with custom monitoring and multiple data sources."""

    # Sample data
    reviews = [
        "Excellent food quality and presentation",
        "Service was slow but friendly",
        "Noisy environment, hard to have conversation",
        "Great value for money",
        "Clean and well-maintained facilities",
    ]

    predefined_themes = [
        "Food Quality",
        "Service",
        "Atmosphere",
        "Value",
        "Cleanliness",
    ]

    # Async monitoring callbacks
    async def on_run_start():
        print("🚀 Starting async workflow execution...")

    async def on_process_start(process_id: str):
        print(f"⚡ Starting process: {process_id}")

    async def on_process_end(process_id: str, result):
        print(f"✅ Completed process: {process_id}")

    async def on_run_end():
        print("🎉 Workflow execution completed!")

    # Create workflow with monitoring
    async with AsyncWorkflow() as workflow:
        # Register data sources
        workflow.source("reviews", reviews)
        workflow.source("themes", predefined_themes)

        # Add monitoring
        workflow.monitor(
            on_run_start=on_run_start,
            on_process_start=on_process_start,
            on_process_end=on_process_end,
            on_run_end=on_run_end,
        )

        # Build complex pipeline
        workflow.sentiment(source="reviews", name="review_sentiment").theme_allocation(
            inputs="reviews",
            themes_from="themes",
            single_label=False,
            threshold=0.3,
            name="theme_assignment",
        ).similarity(source="reviews", flatten=True, name="review_similarity").cluster(
            source="reviews", k=3, algorithm="kmeans", name="review_clusters"
        )

        # Execute with custom client
        print("Running advanced async workflow...")
        results = await workflow.run(fast=True)

        # Display results
        print(
            f"\nSentiment Results: {len(results.review_sentiment.sentiments)} analyzed"
        )
        assignments_count = len(results.theme_assignment.assignments)
        print(f"Theme Assignments: {assignments_count} texts assigned")
        similarity_matrix = results.review_similarity.similarity
        print(
            f"Similarity Matrix: {len(similarity_matrix)} x "
            f"{len(similarity_matrix[0])}"
        )
        print(f"Clusters: {len(results.review_clusters.clusters)} clusters found")


async def concurrent_workflows_example():
    """Example of running multiple async workflows concurrently."""

    # Different datasets
    restaurant_reviews = [
        "Amazing food and great service",
        "Too expensive for the quality",
        "Perfect for a romantic dinner",
    ]

    hotel_reviews = [
        "Clean rooms and comfortable beds",
        "Noisy neighbors kept us awake",
        "Excellent location and amenities",
    ]

    product_reviews = [
        "High quality and durable product",
        "Difficult to use and poor instructions",
        "Great value for the price",
    ]

    async def analyze_dataset(name: str, data: list[str]) -> dict:
        """Analyze a single dataset."""
        async with AsyncWorkflow() as workflow:
            workflow.source("data", data)
            workflow.theme_generation(source="data", fast=True)
            workflow.sentiment(source="data")

            results = await workflow.run()
            return {
                "name": name,
                "themes": results.theme_generation.themes,
                "sentiments": [s.sentiment for s in results.sentiment.sentiments],
            }

    # Run workflows concurrently
    print("Running concurrent async workflows...")

    tasks = [
        analyze_dataset("Restaurant", restaurant_reviews),
        analyze_dataset("Hotel", hotel_reviews),
        analyze_dataset("Product", product_reviews),
    ]

    results = await asyncio.gather(*tasks)

    # Display results
    for result in results:
        print(f"\n{result['name']} Analysis:")
        print(f"  Themes: {[theme.shortLabel for theme in result['themes']]}")
        print(f"  Sentiments: {result['sentiments']}")


async def workflow_from_config_example():
    """Example of loading workflow from configuration file."""

    # Create a temporary config file
    import tempfile
    import json

    config = {
        "pipeline": [
            {"theme_generation": {"min_themes": 2, "max_themes": 4, "fast": True}},
            {"sentiment": {"fast": True}},
            {"theme_allocation": {"single_label": True, "threshold": 0.5}},
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    try:
        # Load workflow from file
        workflow = AsyncWorkflow.from_file(config_path)

        # Add data source
        texts = [
            "Excellent customer service experience",
            "Product quality exceeded expectations",
            "Delivery was delayed but item was perfect",
        ]
        workflow.source("dataset", texts)

        # Execute loaded workflow
        print("Running workflow loaded from config...")
        async with workflow:
            results = await workflow.run()

            themes = [theme.shortLabel for theme in results.theme_generation.themes]
            print(f"Themes: {themes}")
            print(f"Sentiments: {[s.sentiment for s in results.sentiment.sentiments]}")
            print(f"Assignments: {results.theme_allocation.assignments}")

    finally:
        os.unlink(config_path)


async def main():
    """Run all async workflow examples."""
    print("=== Pulse SDK AsyncWorkflow Examples ===\n")

    # Check for credentials
    if not os.getenv("PULSE_CLIENT_ID") or not os.getenv("PULSE_CLIENT_SECRET"):
        print(
            "⚠️  Pulse credentials not found. Set PULSE_CLIENT_ID and "
            "PULSE_CLIENT_SECRET environment variables."
        )
        print("Running examples with mock data only...\n")

    try:
        print("1. Basic Async Workflow")
        print("-" * 30)
        await basic_async_workflow_example()

        print("\n2. Advanced Async Workflow with Monitoring")
        print("-" * 45)
        await advanced_async_workflow_example()

        print("\n3. Concurrent Workflows")
        print("-" * 25)
        await concurrent_workflows_example()

        print("\n4. Workflow from Configuration")
        print("-" * 35)
        await workflow_from_config_example()

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("This is expected if running without valid API credentials.")


if __name__ == "__main__":
    asyncio.run(main())
