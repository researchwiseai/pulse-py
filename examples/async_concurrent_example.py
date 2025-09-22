"""Example demonstrating async concurrent processing utilities."""

import asyncio
import time
from pulse.core.async_client import AsyncCoreClient
from pulse.core.async_concurrent import (
    gather_jobs,
    submit_and_gather_jobs,
    wait_for_any_job,
    AsyncJobManager,
    AsyncConnectionPoolManager,
)
from pulse.core.models import EmbeddingsRequest


async def basic_concurrent_jobs_example():
    """Basic example of processing multiple jobs concurrently."""
    print("=== Basic Concurrent Jobs Example ===")

    async with AsyncCoreClient.with_client_credentials_async() as client:
        # Create multiple embedding requests
        requests = [
            EmbeddingsRequest(inputs=[f"Sample text {i} for embedding"], fast=False)
            for i in range(5)
        ]

        # Submit jobs without waiting for results
        print("Submitting jobs...")
        jobs = []
        for i, request in enumerate(requests):
            job = await client.create_embeddings(request, await_job_result=False)
            jobs.append(job)
            print(f"Submitted job {i+1}: {job.id}")

        # Gather results concurrently with controlled concurrency
        print("\nGathering results concurrently (max 3 concurrent)...")
        start_time = time.time()

        results = await gather_jobs(jobs, max_concurrent=3)

        end_time = time.time()
        print(f"Completed {len(results)} jobs in {end_time - start_time:.2f} seconds")

        # Display results
        for i, result in enumerate(results):
            if result and "embeddings" in result:
                print(f"Job {i+1}: Got {len(result['embeddings'])} embeddings")


async def submit_and_gather_example():
    """Example of submitting and gathering jobs in one operation."""
    print("\n=== Submit and Gather Example ===")

    async with AsyncCoreClient.with_client_credentials_async() as client:
        # Create job submitters with rate limiting
        async def create_embedding_submitter(text, job_num):
            async def submitter():
                request = EmbeddingsRequest(
                    inputs=[f"{text} - job {job_num}"], fast=False
                )
                return await client.create_embeddings(request, await_job_result=False)

            return submitter

        # Create multiple job submitters
        submitters = []
        for i in range(8):
            submitter = await create_embedding_submitter(f"Concurrent text {i}", i + 1)
            submitters.append(submitter)

        print(f"Processing {len(submitters)} jobs with rate limiting...")
        start_time = time.time()

        # Submit and gather with rate limiting and concurrency control
        results = await submit_and_gather_jobs(
            submitters,
            max_concurrent=4,
            rate_limit_delay=0.1,  # 100ms between submissions
            timeout=60.0,
        )

        end_time = time.time()
        print(f"Completed in {end_time - start_time:.2f} seconds")

        # Display results
        successful_results = [r for r in results if r is not None and "embeddings" in r]
        print(
            f"Successfully processed {len(successful_results)} out of "
            f"{len(results)} jobs"
        )


async def wait_for_any_example():
    """Example of waiting for any job to complete first."""
    print("\n=== Wait for Any Job Example ===")

    async with AsyncCoreClient.with_client_credentials_async() as client:
        # Submit multiple jobs with different expected completion times
        jobs = []
        job_descriptions = [
            ("Short text", "fast job"),
            (
                "This is a longer text that might take more time to process",
                "medium job",
            ),
            ("Very short", "quick job"),
        ]

        print("Submitting jobs with different processing times...")
        for text, description in job_descriptions:
            request = EmbeddingsRequest(inputs=[text], fast=False)
            job = await client.create_embeddings(request, await_job_result=False)
            jobs.append(job)
            print(f"Submitted {description}: {job.id}")

        # Wait for the first job to complete
        print("\nWaiting for first job to complete...")
        start_time = time.time()

        completed_job, result = await wait_for_any_job(jobs, timeout=60.0)

        end_time = time.time()
        print(f"First job completed in {end_time - start_time:.2f} seconds")
        print(f"Completed job ID: {completed_job.id}")

        if result and "embeddings" in result:
            print(f"Result: {len(result['embeddings'])} embeddings")


async def advanced_job_manager_example():
    """Advanced example using AsyncJobManager directly."""
    print("\n=== Advanced Job Manager Example ===")

    # Create optimized configuration for high concurrency
    config = AsyncConnectionPoolManager.get_recommended_config(concurrent_jobs=10)
    print(
        f"Using config: max_concurrent={config.max_concurrent_jobs}, "
        f"rate_limit={config.rate_limit_delay}s"
    )

    # Create optimized client
    optimized_client = AsyncConnectionPoolManager.create_optimized_client(
        "https://pulse.researchwiseai.com/v1", config=config
    )

    async with AsyncCoreClient(client=optimized_client) as client:
        # Create job manager with custom config
        manager = AsyncJobManager(config)

        # Create job submitters for batch processing
        async def create_batch_submitter(batch_id, batch_size=3):
            async def submitter():
                texts = [f"Batch {batch_id} text {i}" for i in range(batch_size)]
                request = EmbeddingsRequest(inputs=texts, fast=False)
                return await client.create_embeddings(request, await_job_result=False)

            return submitter

        # Create multiple batches
        batch_count = 6
        submitters = []
        for i in range(batch_count):
            submitter = await create_batch_submitter(i + 1)
            submitters.append(submitter)

        print(f"Processing {batch_count} batches with advanced job manager...")

        try:
            # Use batch processing with retry logic
            start_time = time.time()
            results = await manager.batch_process_with_retry(
                submitters, max_retries=2, retry_delay=1.0
            )
            end_time = time.time()

            print(
                f"Successfully processed {len(results)} batches in "
                f"{end_time - start_time:.2f} seconds"
            )

            # Display batch results
            for i, result in enumerate(results):
                if result and "embeddings" in result:
                    print(f"Batch {i+1}: {len(result['embeddings'])} embeddings")

        except Exception as e:
            print(f"Batch processing failed: {e}")


async def error_handling_example():
    """Example demonstrating error handling in concurrent processing."""
    print("\n=== Error Handling Example ===")

    async with AsyncCoreClient.with_client_credentials_async() as client:
        # Create a mix of valid and invalid requests
        async def create_mixed_submitter(job_id, should_fail=False):
            async def submitter():
                if should_fail:
                    # Create an invalid request that will fail
                    request = EmbeddingsRequest(inputs=[""], fast=False)  # Empty input
                else:
                    request = EmbeddingsRequest(
                        inputs=[f"Valid text {job_id}"], fast=False
                    )

                return await client.create_embeddings(request, await_job_result=False)

            return submitter

        # Create submitters with some that will fail
        submitters = []
        for i in range(6):
            should_fail = i % 3 == 0  # Every 3rd job will fail
            submitter = await create_mixed_submitter(i + 1, should_fail)
            submitters.append(submitter)

        print("Processing jobs with mixed success/failure (return_exceptions=True)...")

        # Process with exception handling
        results = await submit_and_gather_jobs(
            submitters,
            max_concurrent=3,
            return_exceptions=True,  # Return exceptions instead of raising
        )

        # Analyze results
        successes = [r for r in results if isinstance(r, dict) and "embeddings" in r]
        failures = [r for r in results if isinstance(r, Exception)]

        print(f"Results: {len(successes)} successful, {len(failures)} failed")

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Job {i+1}: Failed - {type(result).__name__}: {result}")
            elif result and "embeddings" in result:
                print(f"Job {i+1}: Success - {len(result['embeddings'])} embeddings")


async def performance_comparison_example():
    """Example comparing sequential vs concurrent processing."""
    print("\n=== Performance Comparison Example ===")

    async with AsyncCoreClient.with_client_credentials_async() as client:
        # Create test requests
        requests = [
            EmbeddingsRequest(inputs=[f"Performance test text {i}"], fast=False)
            for i in range(6)
        ]

        # Sequential processing
        print("Sequential processing...")
        start_time = time.time()
        sequential_results = []
        for i, request in enumerate(requests):
            result = await client.create_embeddings(request)
            sequential_results.append(result)
            print(f"  Completed job {i+1}")
        sequential_time = time.time() - start_time

        # Concurrent processing
        print("\nConcurrent processing...")
        start_time = time.time()

        # Submit all jobs
        jobs = []
        for request in requests:
            job = await client.create_embeddings(request, await_job_result=False)
            jobs.append(job)

        # Gather results concurrently
        await gather_jobs(jobs, max_concurrent=3)
        concurrent_time = time.time() - start_time

        # Compare performance
        print("\nPerformance Comparison:")
        print(f"Sequential: {sequential_time:.2f} seconds")
        print(f"Concurrent: {concurrent_time:.2f} seconds")
        print(f"Speedup: {sequential_time / concurrent_time:.2f}x")
        print(f"Both processed {len(requests)} jobs successfully")


async def main():
    """Run all concurrent processing examples."""
    print("Async Concurrent Processing Examples")
    print("=" * 50)

    try:
        await basic_concurrent_jobs_example()
        await submit_and_gather_example()
        await wait_for_any_example()
        await advanced_job_manager_example()
        await error_handling_example()
        await performance_comparison_example()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"\nExample failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
