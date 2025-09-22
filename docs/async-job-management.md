# Async Job Management and Concurrency

This guide covers advanced async job management patterns and concurrency best practices for the Pulse SDK.

## Overview

The Pulse SDK's async job management system provides:

- **Manual job control**: Submit jobs without waiting for completion
- **Concurrent processing**: Handle multiple jobs simultaneously with controlled concurrency
- **Advanced monitoring**: Track job progress with callbacks and status updates
- **Error handling**: Robust error recovery and retry mechanisms
- **Resource optimization**: Efficient connection pooling and rate limiting

## Job Lifecycle

### Basic Job Flow

```python
from pulse.core.async_client import AsyncCoreClient
from pulse.core.models import SentimentRequest

async def basic_job_flow():
    async with AsyncCoreClient.with_client_credentials_async() as client:
        request = SentimentRequest(inputs=["Sample text"], fast=True)

        # Submit job without waiting
        job = await client.analyze_sentiment(request, await_job_result=False)
        print(f"Job submitted: {job.id} (status: {job.status})")

        # Manual polling
        while job.status in ["pending", "queued", "running"]:
            await asyncio.sleep(1)  # Wait 1 second
            await job.refresh()
            print(f"Job status: {job.status}")

        # Get final result
        if job.status == "completed":
            result = await job.get_result()
            return result
        else:
            raise Exception(f"Job failed with status: {job.status}")
```

### Automatic Job Waiting

```python
async def automatic_job_waiting():
    async with AsyncCoreClient.with_client_credentials_async() as client:
        request = SentimentRequest(inputs=["Sample text"], fast=True)

        # Automatic waiting (default behavior)
        result = await client.analyze_sentiment(request)
        # Job is automatically polled until completion
        return result
```

## Advanced Job Control

### Job Monitoring with Callbacks

```python
async def job_with_monitoring():
    async def status_callback(status: str):
        print(f"Job status changed to: {status}")
        if status == "running":
            print("Job is now processing...")

    async with AsyncCoreClient.with_client_credentials_async() as client:
        request = SentimentRequest(inputs=["Sample text"], fast=False)
        job = await client.analyze_sentiment(request, await_job_result=False)

        # Wait with status callbacks
        result = await job.wait_with_callback(
            callback=status_callback,
            timeout=120.0,
            poll_interval=2.0
        )

        return result
```

### Waiting for Specific Status

```python
async def wait_for_specific_status():
    async with AsyncCoreClient.with_client_credentials_async() as client:
        request = SentimentRequest(inputs=["Sample text"], fast=False)
        job = await client.analyze_sentiment(request, await_job_result=False)

        # Wait for job to start running
        await job.wait_for_status("running", timeout=30.0)
        print("Job is now running!")

        # Wait for completion
        result = await job.wait(timeout=120.0)
        return result
```

### Job Cancellation

```python
async def job_cancellation_example():
    async with AsyncCoreClient.with_client_credentials_async() as client:
        request = SentimentRequest(inputs=["Sample text"], fast=False)
        job = await client.analyze_sentiment(request, await_job_result=False)

        try:
            # Wait with short timeout
            result = await job.wait(timeout=5.0)
            return result
        except asyncio.TimeoutError:
            print("Job taking too long, cancelling...")
            cancelled = await job.cancel()
            if cancelled:
                print("Job cancelled successfully")
            else:
                print("Job could not be cancelled (may have completed)")
            return None
```

## Concurrent Job Processing

### Basic Concurrent Jobs

```python
from pulse.core.async_concurrent import gather_jobs

async def basic_concurrent_jobs():
    async with AsyncCoreClient.with_client_credentials_async() as client:
        # Submit multiple jobs
        jobs = []
        texts_batches = [["text1", "text2"], ["text3", "text4"], ["text5", "text6"]]

        for batch in texts_batches:
            request = SentimentRequest(inputs=batch, fast=True)
            job = await client.analyze_sentiment(request, await_job_result=False)
            jobs.append(job)
            print(f"Submitted job: {job.id}")

        # Gather results with controlled concurrency
        print("Gathering results...")
        results = await gather_jobs(jobs, max_concurrent=2)

        print(f"Completed {len(results)} jobs")
        return results
```

### Submit and Gather Pattern

```python
from pulse.core.async_concurrent import submit_and_gather_jobs

async def submit_and_gather_example():
    async with AsyncCoreClient.with_client_credentials_async() as client:
        # Create job submitters
        async def create_job_submitter(batch_id: int, texts: list):
            async def submitter():
                request = SentimentRequest(inputs=texts, fast=True)
                return await client.analyze_sentiment(request, await_job_result=False)
            return submitter

        # Create submitters for multiple batches
        submitters = []
        for i, batch in enumerate([["text1"], ["text2"], ["text3"], ["text4"]]):
            submitter = await create_job_submitter(i, batch)
            submitters.append(submitter)

        # Submit and gather with rate limiting
        results = await submit_and_gather_jobs(
            submitters,
            max_concurrent=2,
            rate_limit_delay=0.5,  # 500ms between submissions
            timeout=60.0
        )

        return results
```

### Wait for Any Job Completion

```python
from pulse.core.async_concurrent import wait_for_any_job

async def wait_for_any_example():
    async with AsyncCoreClient.with_client_credentials_async() as client:
        # Submit multiple jobs with different processing times
        jobs = []
        job_descriptions = [
            (["Short text"], "quick job"),
            (["This is a much longer text that will take more time to process"], "slow job"),
            (["Medium length text"], "medium job"),
        ]

        for texts, description in job_descriptions:
            request = SentimentRequest(inputs=texts, fast=False)
            job = await client.analyze_sentiment(request, await_job_result=False)
            jobs.append(job)
            print(f"Submitted {description}: {job.id}")

        # Wait for first completion
        print("Waiting for first job to complete...")
        completed_job, result = await wait_for_any_job(jobs, timeout=60.0)

        print(f"First completed job: {completed_job.id}")

        # Cancel remaining jobs if needed
        for job in jobs:
            if job.id != completed_job.id and job.status in ["pending", "queued", "running"]:
                await job.cancel()

        return result
```

## Advanced Concurrency Patterns

### Job Manager with Retry Logic

```python
from pulse.core.async_concurrent import AsyncJobManager, AsyncConnectionPoolManager

async def advanced_job_manager():
    # Create optimized configuration
    config = AsyncConnectionPoolManager.get_recommended_config(concurrent_jobs=10)
    manager = AsyncJobManager(config)

    # Create optimized client
    optimized_client = AsyncConnectionPoolManager.create_optimized_client(
        "https://pulse.researchwiseai.com/v1",
        config=config
    )

    async with AsyncCoreClient(client=optimized_client) as client:
        # Create job submitters
        async def create_batch_submitter(batch_id: int):
            async def submitter():
                texts = [f"Batch {batch_id} text {i}" for i in range(3)]
                request = SentimentRequest(inputs=texts, fast=True)
                return await client.analyze_sentiment(request, await_job_result=False)
            return submitter

        # Create multiple batch submitters
        submitters = [await create_batch_submitter(i) for i in range(8)]

        # Process with retry logic
        try:
            results = await manager.batch_process_with_retry(
                submitters,
                max_retries=2,
                retry_delay=1.0,
                timeout=120.0
            )

            print(f"Successfully processed {len(results)} batches")
            return results

        except Exception as e:
            print(f"Batch processing failed: {e}")
            return []
```

### Producer-Consumer Pattern

```python
import asyncio
from asyncio import Queue

async def producer_consumer_pattern():
    # Create queue for job coordination
    job_queue = Queue(maxsize=20)
    result_queue = Queue()

    async def producer(texts: list):
        """Submit jobs to the queue."""
        async with AsyncCoreClient.with_client_credentials_async() as client:
            for i, text in enumerate(texts):
                request = SentimentRequest(inputs=[text], fast=True)
                job = await client.analyze_sentiment(request, await_job_result=False)
                await job_queue.put((i, job))
                print(f"Produced job {i}: {job.id}")

            # Signal completion
            await job_queue.put(None)

    async def consumer(consumer_id: int):
        """Process jobs from the queue."""
        results = []
        while True:
            item = await job_queue.get()
            if item is None:
                # Put sentinel back for other consumers
                await job_queue.put(None)
                break

            job_index, job = item
            try:
                result = await job.wait(timeout=60.0)
                results.append((job_index, result))
                print(f"Consumer {consumer_id} completed job {job_index}")
            except Exception as e:
                print(f"Consumer {consumer_id} failed job {job_index}: {e}")
            finally:
                job_queue.task_done()

        return results

    # Sample data
    texts = [f"Sample text {i}" for i in range(10)]

    # Start producer and multiple consumers
    producer_task = asyncio.create_task(producer(texts))
    consumer_tasks = [
        asyncio.create_task(consumer(i))
        for i in range(3)  # 3 consumers
    ]

    # Wait for producer to finish
    await producer_task

    # Wait for all jobs to be processed
    await job_queue.join()

    # Gather results from all consumers
    consumer_results = await asyncio.gather(*consumer_tasks)

    # Combine and sort results
    all_results = []
    for consumer_result in consumer_results:
        all_results.extend(consumer_result)

    all_results.sort(key=lambda x: x[0])  # Sort by job index
    return [result for _, result in all_results]
```

### Rate-Limited Processing

```python
import asyncio
from asyncio import Semaphore

async def rate_limited_processing():
    # Control concurrent operations
    semaphore = Semaphore(3)  # Max 3 concurrent operations

    async def process_with_rate_limit(text: str, delay: float = 0.1):
        async with semaphore:
            # Add delay between operations
            await asyncio.sleep(delay)

            async with AsyncCoreClient.with_client_credentials_async() as client:
                request = SentimentRequest(inputs=[text], fast=True)
                return await client.analyze_sentiment(request)

    # Process multiple texts with rate limiting
    texts = [f"Text {i}" for i in range(10)]

    tasks = [process_with_rate_limit(text) for text in texts]
    results = await asyncio.gather(*tasks)

    return results
```

## Error Handling and Recovery

### Robust Error Handling

```python
async def robust_job_processing():
    async def safe_job_execution(job_data: dict):
        """Execute a job with comprehensive error handling."""
        try:
            async with AsyncCoreClient.with_client_credentials_async() as client:
                request = SentimentRequest(inputs=job_data['texts'], fast=True)

                # Submit job
                job = await client.analyze_sentiment(request, await_job_result=False)

                # Wait with timeout
                result = await asyncio.wait_for(job.wait(), timeout=60.0)

                return {
                    'job_id': job_data['id'],
                    'status': 'success',
                    'result': result,
                    'error': None
                }

        except asyncio.TimeoutError:
            return {
                'job_id': job_data['id'],
                'status': 'timeout',
                'result': None,
                'error': 'Job timed out after 60 seconds'
            }
        except Exception as e:
            return {
                'job_id': job_data['id'],
                'status': 'error',
                'result': None,
                'error': str(e)
            }

    # Sample job data
    job_data_list = [
        {'id': 1, 'texts': ['Great product!']},
        {'id': 2, 'texts': ['Poor quality']},
        {'id': 3, 'texts': ['Average experience']},
    ]

    # Process all jobs with error handling
    results = await asyncio.gather(*[
        safe_job_execution(job_data)
        for job_data in job_data_list
    ])

    # Analyze results
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']

    print(f"Successful jobs: {len(successful)}")
    print(f"Failed jobs: {len(failed)}")

    for failure in failed:
        print(f"Job {failure['job_id']} failed: {failure['error']}")

    return successful, failed
```

### Retry Logic with Exponential Backoff

```python
import random

async def job_with_retry():
    async def execute_with_retry(
        operation,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0
    ):
        """Execute operation with exponential backoff retry."""
        for attempt in range(max_retries + 1):
            try:
                return await operation()
            except Exception as e:
                if attempt == max_retries:
                    raise e

                # Calculate delay with exponential backoff and jitter
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, delay * 0.1)
                total_delay = delay + jitter

                print(f"Attempt {attempt + 1} failed: {e}")
                print(f"Retrying in {total_delay:.2f} seconds...")
                await asyncio.sleep(total_delay)

    async def unreliable_operation():
        """Simulate an operation that might fail."""
        async with AsyncCoreClient.with_client_credentials_async() as client:
            request = SentimentRequest(inputs=["Sample text"], fast=True)
            return await client.analyze_sentiment(request)

    # Execute with retry logic
    try:
        result = await execute_with_retry(unreliable_operation)
        print("Operation succeeded!")
        return result
    except Exception as e:
        print(f"Operation failed after all retries: {e}")
        return None
```

## Performance Optimization

### Connection Pool Optimization

```python
from pulse.core.async_concurrent import AsyncConnectionPoolManager

async def optimized_concurrent_processing():
    # Get recommended configuration for high concurrency
    config = AsyncConnectionPoolManager.get_recommended_config(
        concurrent_jobs=20,
        target_latency_ms=500
    )

    print(f"Using configuration:")
    print(f"  Max concurrent jobs: {config.max_concurrent_jobs}")
    print(f"  Rate limit delay: {config.rate_limit_delay}s")
    print(f"  Connection pool size: {config.connection_pool_size}")

    # Create optimized HTTP client
    optimized_client = AsyncConnectionPoolManager.create_optimized_client(
        "https://pulse.researchwiseai.com/v1",
        config=config
    )

    async with AsyncCoreClient(client=optimized_client) as client:
        # Create many concurrent jobs
        jobs = []
        for i in range(50):
            request = SentimentRequest(inputs=[f"Text {i}"], fast=True)
            job = await client.analyze_sentiment(request, await_job_result=False)
            jobs.append(job)

        # Process with optimized settings
        results = await gather_jobs(
            jobs,
            max_concurrent=config.max_concurrent_jobs
        )

        print(f"Processed {len(results)} jobs with optimized configuration")
        return results
```

### Memory-Efficient Batch Processing

```python
async def memory_efficient_processing(large_dataset: list, batch_size: int = 100):
    """Process large datasets without loading everything into memory."""

    async def process_batch(batch: list):
        async with AsyncCoreClient.with_client_credentials_async() as client:
            request = SentimentRequest(inputs=batch, fast=True)
            return await client.analyze_sentiment(request)

    results = []

    # Process in batches to control memory usage
    for i in range(0, len(large_dataset), batch_size):
        batch = large_dataset[i:i + batch_size]

        print(f"Processing batch {i // batch_size + 1}/{(len(large_dataset) + batch_size - 1) // batch_size}")

        # Process current batch
        batch_result = await process_batch(batch)
        results.append(batch_result)

        # Optional: Add delay between batches to avoid rate limits
        await asyncio.sleep(0.1)

    return results
```

## Monitoring and Observability

### Job Progress Tracking

```python
import time
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class JobMetrics:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    start_time: float = 0
    end_time: float = 0

async def monitored_job_processing():
    metrics = JobMetrics(start_time=time.time())

    async def track_job(job, job_id: str):
        """Track individual job progress."""
        try:
            result = await job.wait(timeout=120.0)
            metrics.completed += 1
            print(f"✅ Job {job_id} completed")
            return result
        except asyncio.TimeoutError:
            await job.cancel()
            metrics.cancelled += 1
            print(f"⏰ Job {job_id} cancelled (timeout)")
            return None
        except Exception as e:
            metrics.failed += 1
            print(f"❌ Job {job_id} failed: {e}")
            return None

    async with AsyncCoreClient.with_client_credentials_async() as client:
        # Submit jobs
        jobs = []
        for i in range(10):
            request = SentimentRequest(inputs=[f"Text {i}"], fast=True)
            job = await client.analyze_sentiment(request, await_job_result=False)
            jobs.append((job, f"job_{i}"))
            metrics.submitted += 1

        print(f"📤 Submitted {metrics.submitted} jobs")

        # Track all jobs
        tracking_tasks = [track_job(job, job_id) for job, job_id in jobs]
        results = await asyncio.gather(*tracking_tasks)

        metrics.end_time = time.time()

        # Print final metrics
        duration = metrics.end_time - metrics.start_time
        print(f"\n📊 Final Metrics:")
        print(f"  Submitted: {metrics.submitted}")
        print(f"  Completed: {metrics.completed}")
        print(f"  Failed: {metrics.failed}")
        print(f"  Cancelled: {metrics.cancelled}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Success Rate: {metrics.completed / metrics.submitted * 100:.1f}%")

        return [r for r in results if r is not None]
```

### Real-time Progress Updates

```python
async def real_time_progress_monitoring():
    """Monitor job progress with real-time updates."""

    class ProgressMonitor:
        def __init__(self, total_jobs: int):
            self.total_jobs = total_jobs
            self.completed_jobs = 0
            self.failed_jobs = 0
            self.start_time = time.time()

        async def update_progress(self, job_id: str, status: str):
            if status == "completed":
                self.completed_jobs += 1
            elif status == "failed":
                self.failed_jobs += 1

            elapsed = time.time() - self.start_time
            progress = (self.completed_jobs + self.failed_jobs) / self.total_jobs * 100

            print(f"Progress: {progress:.1f}% ({self.completed_jobs + self.failed_jobs}/{self.total_jobs}) "
                  f"- Elapsed: {elapsed:.1f}s - Job {job_id}: {status}")

    monitor = ProgressMonitor(total_jobs=5)

    async def monitored_job(job_id: str, texts: list):
        try:
            async with AsyncCoreClient.with_client_credentials_async() as client:
                request = SentimentRequest(inputs=texts, fast=True)
                result = await client.analyze_sentiment(request)
                await monitor.update_progress(job_id, "completed")
                return result
        except Exception as e:
            await monitor.update_progress(job_id, "failed")
            raise e

    # Create and run monitored jobs
    job_data = [
        ("job_1", ["Great product!"]),
        ("job_2", ["Poor quality"]),
        ("job_3", ["Average experience"]),
        ("job_4", ["Excellent service"]),
        ("job_5", ["Terrible support"]),
    ]

    tasks = [monitored_job(job_id, texts) for job_id, texts in job_data]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return results
```

## Best Practices Summary

### 1. Job Submission
- Use `await_job_result=False` for manual job control
- Submit jobs in batches to avoid overwhelming the API
- Implement proper error handling for job submission

### 2. Concurrency Control
- Always use `max_concurrent` parameter to limit concurrent operations
- Use `gather_jobs()` for controlled job gathering
- Implement rate limiting with `rate_limit_delay`

### 3. Error Handling
- Implement timeout handling for all job operations
- Use retry logic with exponential backoff for transient failures
- Handle job cancellation gracefully

### 4. Resource Management
- Always use async context managers for client lifecycle
- Configure connection pools for high-concurrency scenarios
- Monitor memory usage with large datasets

### 5. Monitoring
- Track job metrics for performance analysis
- Implement progress monitoring for long-running operations
- Log job failures for debugging

### 6. Performance Optimization
- Use optimized connection pool configurations
- Process data in batches for memory efficiency
- Consider producer-consumer patterns for complex workflows

These patterns provide a solid foundation for building robust, scalable async applications with the Pulse SDK.
