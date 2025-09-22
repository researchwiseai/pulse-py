# Async/Await Patterns

The Pulse SDK provides comprehensive async/await support for all major components, enabling efficient concurrent processing and seamless integration with async Python applications.

## Overview

The async API maintains the same intuitive interface as the sync version while providing:

- **Non-blocking operations**: Perfect for web applications and concurrent processing
- **Manual job control**: Full control over job submission and polling
- **Concurrent processing**: Built-in utilities for managing multiple operations
- **Resource efficiency**: Optimized connection pooling and resource management

## Quick Start

```python
import asyncio
from pulse.core.async_client import AsyncCoreClient
from pulse.core.models import SentimentRequest

async def main():
    async with AsyncCoreClient.with_client_credentials_async() as client:
        request = SentimentRequest(inputs=["Great product!", "Not so good"], fast=True)
        result = await client.analyze_sentiment(request)
        print(f"Analyzed {len(result.results)} texts")

asyncio.run(main())
```

## Core Components

### AsyncCoreClient

The `AsyncCoreClient` provides direct async access to the Pulse API:

```python
from pulse.core.async_client import AsyncCoreClient
from pulse.core.models import EmbeddingsRequest

async def example():
    # Automatic authentication from environment
    async with AsyncCoreClient.with_client_credentials_async() as client:
        request = EmbeddingsRequest(inputs=["Hello world"], fast=True)

        # Automatic job waiting (default)
        result = await client.create_embeddings(request)

        # Manual job control
        job = await client.create_embeddings(request, await_job_result=False)
        result = await job.wait(timeout=60.0)
```

### AsyncAnalyzer

High-level workflows with caching and dependency management:

```python
from pulse.analysis.async_analyzer import AsyncAnalyzer
from pulse.analysis.processes import ThemeGeneration, SentimentProcess

async def workflow_example():
    processes = [
        ThemeGeneration(min_themes=2, max_themes=4),
        SentimentProcess(),
    ]

    async with AsyncAnalyzer(
        dataset=["Text 1", "Text 2", "Text 3"],
        processes=processes,
        fast=True
    ) as analyzer:
        results = await analyzer.run()
        return results
```

### AsyncWorkflow DSL

Declarative pipeline building:

```python
from pulse.async_dsl import AsyncWorkflow

async def dsl_example():
    async with AsyncWorkflow() as workflow:
        workflow.source("texts", ["Sample text 1", "Sample text 2"])
        workflow.theme_generation(source="texts", fast=True)
        workflow.sentiment(source="texts", fast=True)

        results = await workflow.run()
        return results
```

### Async Starters

Simple one-line functions:

```python
from pulse.async_starters import sentiment_analysis_async, generate_themes_async

async def starters_example():
    texts = ["Great service!", "Poor quality"]

    # Run operations concurrently
    sentiment_task = sentiment_analysis_async(texts, fast=True)
    themes_task = generate_themes_async(texts, min_themes=1, max_themes=3, fast=True)

    sentiment_result, themes_result = await asyncio.gather(sentiment_task, themes_task)
    return sentiment_result, themes_result
```

## Concurrent Processing

### Basic Concurrency

```python
import asyncio
from pulse.core.async_client import AsyncCoreClient
from pulse.core.models import SentimentRequest

async def concurrent_example():
    async with AsyncCoreClient.with_client_credentials_async() as client:
        # Create multiple requests
        requests = [
            SentimentRequest(inputs=[f"Text batch {i}"], fast=True)
            for i in range(5)
        ]

        # Process concurrently
        tasks = [client.analyze_sentiment(req) for req in requests]
        results = await asyncio.gather(*tasks)

        return results
```

### Advanced Job Management

```python
from pulse.core.async_concurrent import gather_jobs, submit_and_gather_jobs

async def advanced_concurrent():
    async with AsyncCoreClient.with_client_credentials_async() as client:
        # Submit jobs without waiting
        jobs = []
        for i in range(10):
            request = SentimentRequest(inputs=[f"Text {i}"], fast=True)
            job = await client.analyze_sentiment(request, await_job_result=False)
            jobs.append(job)

        # Gather with controlled concurrency
        results = await gather_jobs(jobs, max_concurrent=3)
        return results

async def batch_processing():
    async def create_job_submitter(batch_id):
        async def submitter():
            request = SentimentRequest(inputs=[f"Batch {batch_id} text"], fast=True)
            return await client.analyze_sentiment(request, await_job_result=False)
        return submitter

    # Create job submitters
    submitters = [await create_job_submitter(i) for i in range(5)]

    # Submit and gather with rate limiting
    results = await submit_and_gather_jobs(
        submitters,
        max_concurrent=2,
        rate_limit_delay=0.1,  # 100ms between submissions
        timeout=60.0
    )
    return results
```

## Error Handling

### Basic Error Handling

```python
async def safe_operation():
    try:
        async with AsyncCoreClient.with_client_credentials_async() as client:
            request = SentimentRequest(inputs=["Test text"], fast=True)
            result = await asyncio.wait_for(
                client.analyze_sentiment(request),
                timeout=30.0
            )
            return result
    except asyncio.TimeoutError:
        print("Operation timed out")
    except Exception as e:
        print(f"Error: {e}")
        return None
```

### Concurrent Error Handling

```python
async def safe_concurrent_operations():
    async def safe_task(operation_name, coro):
        try:
            result = await coro
            return {'name': operation_name, 'success': True, 'result': result}
        except Exception as e:
            return {'name': operation_name, 'success': False, 'error': str(e)}

    async with AsyncCoreClient.with_client_credentials_async() as client:
        tasks = [
            safe_task("sentiment", client.analyze_sentiment(SentimentRequest(inputs=["Text"], fast=True))),
            safe_task("themes", client.generate_themes(ThemesRequest(inputs=["Text"], fast=True))),
        ]

        results = await asyncio.gather(*tasks)

        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        return successful, failed
```

## Resource Management

### Proper Cleanup

```python
class AsyncResourceManager:
    def __init__(self):
        self.client = None
        self.active_jobs = []

    async def __aenter__(self):
        self.client = AsyncCoreClient.with_client_credentials_async()
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cancel active jobs
        for job in self.active_jobs:
            try:
                await job.cancel()
            except:
                pass

        # Close client
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)

    async def submit_job(self, request):
        job = await self.client.analyze_sentiment(request, await_job_result=False)
        self.active_jobs.append(job)
        return job

# Usage
async with AsyncResourceManager() as manager:
    job = await manager.submit_job(SentimentRequest(inputs=["Text"], fast=True))
    result = await job.wait()
```

## Performance Optimization

### Connection Pool Configuration

```python
from pulse.core.async_concurrent import AsyncConnectionPoolManager

async def optimized_connection_example():
    # Get recommended configuration for high concurrency
    config = AsyncConnectionPoolManager.get_recommended_config(concurrent_jobs=20)

    # Create optimized client
    optimized_client = AsyncConnectionPoolManager.create_optimized_client(
        "https://pulse.researchwiseai.com/v1",
        config=config
    )

    async with AsyncCoreClient(client=optimized_client) as client:
        # Use optimized client for high-concurrency operations
        pass
```

### Batch Processing

```python
from pulse.core.async_concurrent import AsyncJobManager

async def optimized_batch_processing():
    config = AsyncConnectionPoolManager.get_recommended_config(concurrent_jobs=10)
    manager = AsyncJobManager(config)

    # Create batch submitters
    async def create_batch_submitter(batch_data):
        async def submitter():
            request = SentimentRequest(inputs=batch_data, fast=True)
            return await client.analyze_sentiment(request, await_job_result=False)
        return submitter

    batches = [["text1", "text2"], ["text3", "text4"], ["text5", "text6"]]
    submitters = [await create_batch_submitter(batch) for batch in batches]

    # Process with retry logic
    results = await manager.batch_process_with_retry(
        submitters,
        max_retries=2,
        retry_delay=1.0
    )

    return results
```

## Migration from Sync to Async

### Component Mapping

| Sync Component | Async Component | Notes |
|----------------|-----------------|-------|
| `CoreClient` | `AsyncCoreClient` | Use `with_client_credentials_async()` |
| `Analyzer` | `AsyncAnalyzer` | Same interface, add `await` |
| `Workflow` | `AsyncWorkflow` | Same DSL, async execution |
| `sentiment_analysis()` | `sentiment_analysis_async()` | Add `_async` suffix |
| `generate_themes()` | `generate_themes_async()` | Add `_async` suffix |

### Migration Steps

1. **Replace imports**:
   ```python
   # Before
   from pulse.core.client import CoreClient
   from pulse.starters import sentiment_analysis

   # After
   from pulse.core.async_client import AsyncCoreClient
   from pulse.async_starters import sentiment_analysis_async
   ```

2. **Update client creation**:
   ```python
   # Before
   with CoreClient.with_client_credentials() as client:
       result = client.analyze_sentiment(request)

   # After
   async with AsyncCoreClient.with_client_credentials_async() as client:
       result = await client.analyze_sentiment(request)
   ```

3. **Add async/await keywords**:
   ```python
   # Before
   result = sentiment_analysis(texts)

   # After
   async def example():
       result = await sentiment_analysis_async(texts)
       return result
   ```

4. **Update function definitions**:
   ```python
   # Before
   def process_texts(texts):
       return sentiment_analysis(texts)

   # After
   async def process_texts(texts):
       return await sentiment_analysis_async(texts)
   ```

## Best Practices

### 1. Always Use Context Managers

```python
# ✅ Good
async with AsyncCoreClient.with_client_credentials_async() as client:
    result = await client.analyze_sentiment(request)

# ❌ Bad - no automatic cleanup
client = AsyncCoreClient.with_client_credentials_async()
result = await client.analyze_sentiment(request)
```

### 2. Control Concurrency

```python
# ✅ Good - controlled concurrency
results = await gather_jobs(jobs, max_concurrent=3)

# ❌ Bad - unlimited concurrency may overwhelm API
async def bad_example():
    results = await asyncio.gather(*[job.wait() for job in jobs])
    return results
```

### 3. Handle Timeouts

```python
# ✅ Good
async def good_example():
    try:
        result = await asyncio.wait_for(operation, timeout=60.0)
    except asyncio.TimeoutError:
        print("Operation timed out")
    return result
```

### 4. Use Appropriate Error Handling

```python
# ✅ Good - handle exceptions in concurrent operations
async def error_handling_example():
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            print(f"Task failed: {result}")
    return results
```

### 5. Choose the Right Abstraction

- **AsyncCoreClient**: When you need fine-grained control
- **AsyncAnalyzer**: For complex workflows with caching
- **AsyncWorkflow**: For declarative pipeline building
- **Async Starters**: For simple, one-off operations

### 6. Leverage Caching

```python
# ✅ Good - enable caching for repeated operations
async def caching_example():
    async with AsyncAnalyzer(
        dataset=texts,
        processes=processes,
        use_cache=True,
        cache_dir="./analysis_cache"
    ) as analyzer:
        results = await analyzer.run()
    return results
```

## Common Patterns

### Producer-Consumer Pattern

```python
import asyncio
from asyncio import Queue

async def producer(queue: Queue, texts: list):
    for text in texts:
        await queue.put(text)
    await queue.put(None)  # Sentinel

async def consumer(queue: Queue, client: AsyncCoreClient):
    results = []
    while True:
        text = await queue.get()
        if text is None:
            break

        request = SentimentRequest(inputs=[text], fast=True)
        result = await client.analyze_sentiment(request)
        results.append(result)
        queue.task_done()

    return results

async def producer_consumer_example():
    queue = Queue(maxsize=10)
    texts = ["Text 1", "Text 2", "Text 3"]

    async with AsyncCoreClient.with_client_credentials_async() as client:
        # Start producer and consumer
        producer_task = asyncio.create_task(producer(queue, texts))
        consumer_task = asyncio.create_task(consumer(queue, client))

        # Wait for completion
        await producer_task
        await queue.join()
        results = await consumer_task

        return results
```

### Rate-Limited Processing

```python
import asyncio
from asyncio import Semaphore

async def rate_limited_processing(texts: list, rate_limit: int = 5):
    semaphore = Semaphore(rate_limit)

    async def process_text(text: str):
        async with semaphore:
            async with AsyncCoreClient.with_client_credentials_async() as client:
                request = SentimentRequest(inputs=[text], fast=True)
                return await client.analyze_sentiment(request)

    tasks = [process_text(text) for text in texts]
    results = await asyncio.gather(*tasks)
    return results
```

## Troubleshooting

### Common Issues

1. **"RuntimeError: This event loop is already running"**
   - Don't call `asyncio.run()` inside an already running event loop
   - Use `await` instead of `asyncio.run()` in Jupyter notebooks

2. **"Session is closed" errors**
   - Always use async context managers
   - Don't reuse clients after they've been closed

3. **High memory usage with many concurrent operations**
   - Use `max_concurrent` parameter to limit concurrency
   - Consider batch processing for large datasets

4. **Timeout errors**
   - Increase timeout values for slow operations
   - Use `fast=True` for quicker processing when accuracy trade-offs are acceptable

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

async with AsyncCoreClient.with_client_credentials_async() as client:
    # Debug information will be logged
    result = await client.analyze_sentiment(request)
```

## Examples

See the comprehensive examples in:
- `examples/async_api.ipynb` - Jupyter notebook with interactive examples
- `examples/async_*_example.py` - Individual component examples
- `tests/test_async_*.py` - Test files with usage patterns
