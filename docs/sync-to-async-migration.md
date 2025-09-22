# Sync to Async Migration Guide

This guide helps you migrate existing sync Pulse SDK code to use the new async/await functionality.

## Overview

The Pulse SDK's async API provides the same functionality as the sync API with these key benefits:

- **Non-blocking operations**: Perfect for web applications and concurrent processing
- **Better resource utilization**: Efficient handling of I/O-bound operations
- **Concurrent processing**: Process multiple operations simultaneously
- **Scalability**: Handle higher loads with better performance

## Component Migration Map

| Sync Component | Async Component | Import Change |
|----------------|-----------------|---------------|
| `CoreClient` | `AsyncCoreClient` | `pulse.core.async_client` |
| `Analyzer` | `AsyncAnalyzer` | `pulse.analysis.async_analyzer` |
| `Workflow` | `AsyncWorkflow` | `pulse.async_dsl` |
| `ClientCredentialsAuth` | `AsyncClientCredentialsAuth` | `pulse.async_auth` |
| `AuthorizationCodePKCEAuth` | `AsyncAuthorizationCodePKCEAuth` | `pulse.async_auth` |
| Starter functions | Async starter functions | `pulse.async_starters` |

## Step-by-Step Migration

### 1. Update Imports

**Before (Sync):**
```python
from pulse.core.client import CoreClient
from pulse.analysis.analyzer import Analyzer
from pulse.dsl import Workflow
from pulse.starters import sentiment_analysis, generate_themes
from pulse.auth import ClientCredentialsAuth
```

**After (Async):**
```python
from pulse.core.async_client import AsyncCoreClient
from pulse.analysis.async_analyzer import AsyncAnalyzer
from pulse.async_dsl import AsyncWorkflow
from pulse.async_starters import sentiment_analysis_async, generate_themes_async
from pulse.async_auth import AsyncClientCredentialsAuth
```

### 2. Update Function Signatures

**Before (Sync):**
```python
def analyze_customer_feedback(reviews):
    with CoreClient.with_client_credentials() as client:
        request = SentimentRequest(inputs=reviews, fast=True)
        return client.analyze_sentiment(request)
```

**After (Async):**
```python
async def analyze_customer_feedback(reviews):
    async with AsyncCoreClient.with_client_credentials_async() as client:
        request = SentimentRequest(inputs=reviews, fast=True)
        return await client.analyze_sentiment(request)
```

### 3. Update Client Creation

**Before (Sync):**
```python
# Method 1: Context manager
with CoreClient.with_client_credentials() as client:
    result = client.analyze_sentiment(request)

# Method 2: Manual creation
auth = ClientCredentialsAuth(client_id="...", client_secret="...")
client = CoreClient(auth=auth)
result = client.analyze_sentiment(request)
client.close()
```

**After (Async):**
```python
# Method 1: Async context manager
async with AsyncCoreClient.with_client_credentials_async() as client:
    result = await client.analyze_sentiment(request)

# Method 2: Manual creation
auth = AsyncClientCredentialsAuth(client_id="...", client_secret="...")
client = AsyncCoreClient(auth=auth)
result = await client.analyze_sentiment(request)
await client.close()
```

### 4. Update Starter Functions

**Before (Sync):**
```python
def process_reviews(reviews):
    # Generate themes
    themes = generate_themes(reviews, min_themes=2, max_themes=5)

    # Analyze sentiment
    sentiment = sentiment_analysis(reviews)

    # Allocate to themes
    allocation = theme_allocation(reviews, themes=[t.shortLabel for t in themes.themes])

    return themes, sentiment, allocation
```

**After (Async):**
```python
async def process_reviews(reviews):
    # Generate themes
    themes = await generate_themes_async(reviews, min_themes=2, max_themes=5)

    # Analyze sentiment
    sentiment = await sentiment_analysis_async(reviews)

    # Allocate to themes
    allocation = await theme_allocation_async(reviews, themes=[t.shortLabel for t in themes.themes])

    return themes, sentiment, allocation
```

### 5. Update Analyzer Workflows

**Before (Sync):**
```python
def run_analysis_workflow(dataset):
    processes = [
        ThemeGeneration(min_themes=2, max_themes=4),
        SentimentProcess(),
        ThemeAllocation(),
    ]

    with Analyzer(dataset=dataset, processes=processes) as analyzer:
        return analyzer.run()
```

**After (Async):**
```python
async def run_analysis_workflow(dataset):
    processes = [
        ThemeGeneration(min_themes=2, max_themes=4),
        SentimentProcess(),
        ThemeAllocation(),
    ]

    async with AsyncAnalyzer(dataset=dataset, processes=processes) as analyzer:
        return await analyzer.run()
```

### 6. Update DSL Workflows

**Before (Sync):**
```python
def create_workflow(texts):
    with Workflow() as workflow:
        workflow.source("texts", texts)
        workflow.theme_generation(source="texts", min_themes=2, max_themes=4)
        workflow.sentiment(source="texts")
        workflow.theme_allocation(inputs="texts")

        return workflow.run()
```

**After (Async):**
```python
async def create_workflow(texts):
    async with AsyncWorkflow() as workflow:
        workflow.source("texts", texts)
        workflow.theme_generation(source="texts", min_themes=2, max_themes=4)
        workflow.sentiment(source="texts")
        workflow.theme_allocation(inputs="texts")

        return await workflow.run()
```

## Advanced Migration Patterns

### 1. Concurrent Processing

One of the biggest benefits of async is the ability to process multiple operations concurrently.

**Before (Sync - Sequential):**
```python
def process_multiple_datasets(datasets):
    results = []
    for dataset in datasets:
        result = sentiment_analysis(dataset)
        results.append(result)
    return results
```

**After (Async - Concurrent):**
```python
import asyncio

async def process_multiple_datasets(datasets):
    # Process all datasets concurrently
    tasks = [sentiment_analysis_async(dataset) for dataset in datasets]
    results = await asyncio.gather(*tasks)
    return results
```

### 2. Manual Job Control

Async provides better control over job management.

**Before (Sync):**
```python
def process_with_job_control(texts):
    with CoreClient.with_client_credentials() as client:
        request = SentimentRequest(inputs=texts, fast=False)
        # Sync client automatically waits for job completion
        return client.analyze_sentiment(request)
```

**After (Async):**
```python
async def process_with_job_control(texts):
    async with AsyncCoreClient.with_client_credentials_async() as client:
        request = SentimentRequest(inputs=texts, fast=False)

        # Option 1: Automatic waiting (same as sync)
        result = await client.analyze_sentiment(request)

        # Option 2: Manual job control
        job = await client.analyze_sentiment(request, await_job_result=False)
        print(f"Job submitted: {job.id}")
        result = await job.wait(timeout=120.0)

        return result
```

### 3. Error Handling

**Before (Sync):**
```python
def safe_analysis(texts):
    try:
        with CoreClient.with_client_credentials() as client:
            request = SentimentRequest(inputs=texts, fast=True)
            return client.analyze_sentiment(request)
    except Exception as e:
        print(f"Analysis failed: {e}")
        return None
```

**After (Async):**
```python
import asyncio

async def safe_analysis(texts):
    try:
        async with AsyncCoreClient.with_client_credentials_async() as client:
            request = SentimentRequest(inputs=texts, fast=True)
            # Add timeout for better control
            return await asyncio.wait_for(
                client.analyze_sentiment(request),
                timeout=60.0
            )
    except asyncio.TimeoutError:
        print("Analysis timed out")
        return None
    except Exception as e:
        print(f"Analysis failed: {e}")
        return None
```

### 4. Batch Processing

**Before (Sync):**
```python
def process_large_dataset(texts, batch_size=10):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = sentiment_analysis(batch)
        results.append(result)
    return results
```

**After (Async):**
```python
from pulse.core.async_concurrent import submit_and_gather_jobs

async def process_large_dataset(texts, batch_size=10, max_concurrent=3):
    async with AsyncCoreClient.with_client_credentials_async() as client:
        # Create job submitters for each batch
        async def create_batch_submitter(batch):
            async def submitter():
                request = SentimentRequest(inputs=batch, fast=True)
                return await client.analyze_sentiment(request, await_job_result=False)
            return submitter

        # Create submitters for all batches
        submitters = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            submitter = await create_batch_submitter(batch)
            submitters.append(submitter)

        # Process with controlled concurrency
        results = await submit_and_gather_jobs(
            submitters,
            max_concurrent=max_concurrent,
            rate_limit_delay=0.1
        )

        return results
```

## Web Framework Integration

### FastAPI

**Before (Sync with thread pool):**
```python
from fastapi import FastAPI
from concurrent.futures import ThreadPoolExecutor
import asyncio

app = FastAPI()
executor = ThreadPoolExecutor()

@app.post("/analyze")
async def analyze_endpoint(texts: list[str]):
    # Run sync code in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        lambda: sentiment_analysis(texts)
    )
    return result
```

**After (Native Async):**
```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/analyze")
async def analyze_endpoint(texts: list[str]):
    # Native async - no thread pool needed
    result = await sentiment_analysis_async(texts)
    return result
```

### Django (Async Views)

**Before (Sync):**
```python
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def analyze_view(request):
    data = json.loads(request.body)
    texts = data.get('texts', [])

    result = sentiment_analysis(texts)
    return JsonResponse({'result': result.dict()})
```

**After (Async):**
```python
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
async def analyze_view(request):
    data = json.loads(request.body)
    texts = data.get('texts', [])

    result = await sentiment_analysis_async(texts)
    return JsonResponse({'result': result.dict()})
```

## Performance Considerations

### Memory Usage

Async operations can use more memory due to maintaining multiple coroutines. Use these strategies:

```python
# ✅ Good - Control concurrency
from pulse.core.async_concurrent import gather_jobs

async def memory_efficient_processing(large_dataset):
    async with AsyncCoreClient.with_client_credentials_async() as client:
        # Submit jobs
        jobs = []
        for batch in large_dataset:
            request = SentimentRequest(inputs=batch, fast=True)
            job = await client.analyze_sentiment(request, await_job_result=False)
            jobs.append(job)

        # Process with limited concurrency
        results = await gather_jobs(jobs, max_concurrent=5)
        return results

# ❌ Bad - Unlimited concurrency
async def memory_intensive_processing(large_dataset):
    tasks = [sentiment_analysis_async(batch) for batch in large_dataset]
    return await asyncio.gather(*tasks)  # May use too much memory
```

### Connection Pooling

```python
from pulse.core.async_concurrent import AsyncConnectionPoolManager

async def optimized_connection_example():
    # Configure optimized connection pool
    config = AsyncConnectionPoolManager.get_recommended_config(concurrent_jobs=10)
    optimized_client = AsyncConnectionPoolManager.create_optimized_client(
        "https://pulse.researchwiseai.com/v1",
        config=config
    )

    async with AsyncCoreClient(client=optimized_client) as client:
        # Use optimized client for better performance
        pass
```

## Testing Migration

### Unit Tests

**Before (Sync):**
```python
import unittest
from pulse.starters import sentiment_analysis

class TestAnalysis(unittest.TestCase):
    def test_sentiment_analysis(self):
        texts = ["Great product!", "Poor quality"]
        result = sentiment_analysis(texts)
        self.assertEqual(len(result.results), 2)
```

**After (Async):**
```python
import unittest
import asyncio
from pulse.async_starters import sentiment_analysis_async

class TestAnalysis(unittest.TestCase):
    def test_sentiment_analysis(self):
        async def run_test():
            texts = ["Great product!", "Poor quality"]
            result = await sentiment_analysis_async(texts)
            self.assertEqual(len(result.results), 2)

        asyncio.run(run_test())
```

### pytest-asyncio

```python
import pytest
from pulse.async_starters import sentiment_analysis_async

@pytest.mark.asyncio
async def test_sentiment_analysis():
    texts = ["Great product!", "Poor quality"]
    result = await sentiment_analysis_async(texts)
    assert len(result.results) == 2
```

## Common Migration Issues

### 1. Event Loop Already Running

**Problem:**
```python
# This will fail in Jupyter notebooks or async contexts
asyncio.run(sentiment_analysis_async(texts))
```

**Solution:**
```python
async def jupyter_solution():
    # In Jupyter or async context, use await directly
    result = await sentiment_analysis_async(texts)
    return result

# Or create a new event loop
import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
result = loop.run_until_complete(sentiment_analysis_async(texts))
loop.close()
```

### 2. Mixing Sync and Async

**Problem:**
```python
async def mixed_function():
    # This won't work - can't call sync from async context efficiently
    sync_result = sentiment_analysis(texts)  # Blocks event loop
    async_result = await sentiment_analysis_async(texts)
    return sync_result, async_result
```

**Solution:**
```python
async def async_only_function():
    # Use only async functions
    result1 = await sentiment_analysis_async(texts1)
    result2 = await sentiment_analysis_async(texts2)
    return result1, result2

# Or run sync code in thread pool if necessary
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def mixed_function_fixed():
    executor = ThreadPoolExecutor()
    loop = asyncio.get_event_loop()

    sync_result = await loop.run_in_executor(
        executor,
        lambda: sentiment_analysis(texts)
    )
    async_result = await sentiment_analysis_async(texts)
    return sync_result, async_result
```

### 3. Resource Cleanup

**Problem:**
```python
# Forgetting to close resources
client = AsyncCoreClient.with_client_credentials_async()
result = await client.analyze_sentiment(request)
# Client not closed - resource leak
```

**Solution:**
```python
# Always use context managers
async with AsyncCoreClient.with_client_credentials_async() as client:
    result = await client.analyze_sentiment(request)
# Client automatically closed
```

## Migration Checklist

- [ ] Update all imports to async versions
- [ ] Add `async` keyword to function definitions
- [ ] Add `await` keyword before all API calls
- [ ] Replace `with` with `async with` for context managers
- [ ] Update error handling for async patterns
- [ ] Add timeout handling where appropriate
- [ ] Consider concurrent processing opportunities
- [ ] Update tests to use async patterns
- [ ] Verify resource cleanup with context managers
- [ ] Test performance improvements

## Gradual Migration Strategy

You can migrate gradually by running both sync and async code in the same application:

```python
# Keep existing sync functions
def legacy_analysis(texts):
    return sentiment_analysis(texts)

# Add new async functions
async def new_analysis(texts):
    return await sentiment_analysis_async(texts)

# Bridge function to call async from sync context
def bridge_analysis(texts):
    return asyncio.run(new_analysis(texts))

# Use bridge function during migration
result = bridge_analysis(texts)
```

## Performance Benefits

After migration, you should see:

- **Reduced latency** for concurrent operations
- **Better resource utilization** in I/O-bound applications
- **Improved scalability** for web applications
- **More responsive** user interfaces

Example performance improvement:

```python
import time

# Before (Sync - Sequential)
start = time.time()
results = []
for dataset in datasets:  # 5 datasets
    result = sentiment_analysis(dataset)  # ~2s each
    results.append(result)
# Total time: ~10 seconds

# After (Async - Concurrent)
async def concurrent_processing():
    start = time.time()
    tasks = [sentiment_analysis_async(dataset) for dataset in datasets]
    results = await asyncio.gather(*tasks)
    # Total time: ~2-3 seconds (depending on concurrency limits)
    return results
```

The async version can be 3-5x faster for concurrent operations while using the same API interface.
