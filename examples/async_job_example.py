#!/usr/bin/env python3
"""
Example demonstrating AsyncJob usage patterns.

This example shows how to use the AsyncJob class for manual job control
in async applications.
"""

import asyncio
from pulse.core.async_jobs import AsyncJob


async def main():
    """Demonstrate AsyncJob usage patterns."""

    # Example job data (normally this would come from an API response)
    job_data = {
        "jobId": "example-job-123",
        "jobStatus": "pending",
        "message": None,
        "resultUrl": None,
    }

    # Create AsyncJob instance
    job = AsyncJob.model_validate(job_data)

    # In a real application, you would set the client from the API response
    # job._client = async_client

    print(f"Created AsyncJob: {job.id} with status: {job.status}")

    # Example 1: Manual polling
    print("\n=== Manual Polling Example ===")
    print("You can manually refresh job status:")
    print("refreshed_job = await job.refresh()")

    # Example 2: Wait for completion
    print("\n=== Wait for Completion Example ===")
    print("You can wait for job completion:")
    print("result = await job.wait(timeout=180.0)")

    # Example 3: Wait with callback
    print("\n=== Wait with Callback Example ===")
    print("You can wait with status callbacks:")

    async def status_callback(status):
        print(f"Job status changed to: {status}")

    print("result = await job.wait_with_callback(status_callback)")

    # Example 4: Wait for specific status
    print("\n=== Wait for Specific Status Example ===")
    print("You can wait for a specific status:")
    print("queued_job = await job.wait_for_status('queued')")

    # Example 5: Job cancellation
    print("\n=== Job Cancellation Example ===")
    print("You can attempt to cancel a job:")
    print("cancelled = await job.cancel()")

    # Example 6: Concurrent job handling
    print("\n=== Concurrent Jobs Example ===")
    print("You can handle multiple jobs concurrently:")
    print(
        """
async def handle_multiple_jobs(jobs):
    # Wait for all jobs to complete
    results = await asyncio.gather(*[job.wait() for job in jobs])
    return results

# Or wait for first completion
async def wait_for_first_completion(jobs):
    done, pending = await asyncio.wait(
        [job.wait() for job in jobs],
        return_when=asyncio.FIRST_COMPLETED
    )
    # Cancel remaining jobs
    for job in pending:
        job.cancel()
    return done
    """
    )

    print("\n=== AsyncJob Features Summary ===")
    print("✅ Async refresh() with retry logic")
    print("✅ Async wait() with timeout support")
    print("✅ Job cancellation with cancel()")
    print("✅ Status callbacks with wait_with_callback()")
    print("✅ Wait for specific status with wait_for_status()")
    print("✅ Proper asyncio timeout handling")
    print("✅ Compatible with asyncio.gather() and asyncio.wait()")


if __name__ == "__main__":
    asyncio.run(main())
