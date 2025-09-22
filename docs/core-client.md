# Core Client

Module: `pulse.core.client`

`CoreClient` is a synchronous, HTTPX-based client for the Pulse REST API. It exposes endpoints for embeddings, similarity, themes, sentiment, extractions, clustering, and summaries. Methods support both synchronous (fast) and asynchronous (job) modes where applicable.

## Async Support

For async/await applications, use `AsyncCoreClient` from `pulse.core.async_client`:

```python
import asyncio
from pulse.core.async_client import AsyncCoreClient
from pulse.core.models import SentimentRequest

async def main():
    async with AsyncCoreClient.with_client_credentials_async() as client:
        request = SentimentRequest(inputs=["Great product!", "Poor quality"], fast=True)
        result = await client.analyze_sentiment(request)
        print([r.sentiment for r in result.results])

asyncio.run(main())
```

The async client provides the same interface as the sync client but with full async/await support, concurrent processing capabilities, and advanced job management. See the [Async Patterns Guide](async-patterns.md) for comprehensive documentation.

## Enhanced Automatic Batching

The Core Client features comprehensive automatic batching capabilities that handle large-scale data processing transparently. When processing large datasets, the client automatically:

- **Splits large requests** into optimal batch sizes
- **Processes batches in parallel** (up to 5 concurrent jobs)
- **Preserves result order** to match input order
- **Aggregates usage metrics** across all batches
- **Handles failures gracefully** with retry logic

### Batching Limits and Behavior

| Feature | Fast Mode Limit | Slow Mode Limit | Auto-Batch Size | Concurrent Jobs |
|---------|----------------|-----------------|-----------------|-----------------|
| Sentiment Analysis | 200 texts | 1,000,000 texts | 2,000 texts | 5 |
| Embeddings | 200 texts | 1,000,000 texts | 5,000 texts | 5 |
| Element Extraction | 200 texts | 1,000,000 texts | 2,000 texts | 5 |
| Clustering | 500 texts (auto-batch) | 44,721 texts | Variable | 5 |
| Similarity | 500 texts (self) / 20k product (cross) | 44,721 texts | Variable | 5 |

### Batching Configuration

Configure batching behavior using `BatchingConfig`:

```python
from pulse.core.concurrent import BatchingConfig
from pulse.core.client import CoreClient

# Custom batching configuration
config = BatchingConfig(
    max_concurrent_jobs=3,           # Reduce concurrency for resource-constrained systems
    timeout_per_batch=600.0,         # 10 minutes per batch
    default_batch_sizes={
        'sentiment': 1500,           # Smaller sentiment batches
        'embeddings': 3000,          # Smaller embedding batches
        'extractions': 1500          # Smaller extraction batches
    },
    retry_failed_batches=True,       # Retry failed batches
    max_retries=3                    # Maximum retry attempts
)

client = CoreClient.with_client_credentials(batching_config=config)
```

### Batching Error Handling

The client provides comprehensive error handling for batching operations:

```python
from pulse.core.exceptions import BatchingError

try:
    # Process large dataset
    texts = ["text"] * 100000
    result = client.analyze_sentiment(texts, fast=False)

except BatchingError as e:
    print(f"Batching error: {e.error_code}")
    print(f"Suggested action: {e.suggested_action}")

    # Get structured error information
    error_info = e.get_structured_info()
    print(f"Feature: {error_info['feature']}")
    print(f"Input count: {error_info['input_count']}")
    print(f"Limit: {error_info['limit']}")

    # Check if error is retryable
    if error_info['is_retryable']:
        retry_strategy = e.get_retry_strategy()
        print(f"Retry strategy: {retry_strategy}")
```

For detailed error handling, see the [Batching Error Reference](batching-errors.md).

## Constructing a Client

```python
from pulse.core.client import CoreClient

# Default construction uses gzip client + auto_auth()
client = CoreClient()  # base_url defaults to PROD; set auth env vars or pass auth explicitly

# Provide explicit base_url/timeout
client = CoreClient(base_url="https://your-custom-endpoint.com/v1", timeout=30.0)

# Provide a fully-configured HTTPX client (you manage auth and base_url)
import httpx
client = CoreClient(client=httpx.Client(base_url="https://..."))

# Use convenience auth-aware helpers
client = CoreClient.with_client_credentials()  # resolves args from env
client = CoreClient.with_pkce(code="...", code_verifier="...")
```

Constructor parameters:
- `base_url: str` – API base URL. Defaults to `PROD_BASE_URL`. Env override: `PULSE_BASE_URL` (via helpers).
- `timeout: float` – Request timeout in seconds. Defaults to `DEFAULT_TIMEOUT`.
- `client: httpx.Client | None` – Use your own client (you manage auth and compression).
- `auth: httpx.Auth | None` – HTTPX auth (e.g., `ClientCredentialsAuth`, `AuthorizationCodePKCEAuth`). Ignored if `client` is provided.

Notes:
- When not passing `client`, `CoreClient` uses an internal `GzipClient` (gzip-compresses raw `content=` bodies) and `auto_auth()` to resolve auth.
- Requests retry transient errors (429/5xx) with exponential backoff.

## Return Types and Job Handling

Most endpoints support synchronous and asynchronous execution. Common flags:
- `fast: bool` – If `True`, request synchronous processing; if the server responds with 202 Accepted while `fast=True`, the client raises `PulseAPIError`.
- `await_job_result: bool` – If `False`, return a `Job` handle instead of blocking. Call `job.wait()` or `job.result()` to retrieve the final JSON payload.

## Methods

### `create_embeddings(request: EmbeddingsRequest, *, await_job_result: bool = True) -> EmbeddingsResponse | Job`
Generate dense vector embeddings with automatic batching for large datasets.

Parameters:
- `request: EmbeddingsRequest` – Request model. Fields:
  - `inputs: list[str]` – Input strings for embedding generation
  - `fast: bool | None` – Processing mode:
    - `True`: Synchronous processing (limit: 200 texts)
    - `False`: Asynchronous processing with automatic batching (limit: 1,000,000 texts)
- `await_job_result: bool` – Return `Job` when false and server responds 202

**Batching Behavior:**
- **Fast mode (`fast=True`)**: Processes up to 200 texts synchronously. Exceeding this limit raises `BatchingError` with code `BATCH_001`.
- **Slow mode (`fast=False`)**: Automatically batches large datasets into 5,000-text chunks, processes up to 5 batches concurrently, and returns embeddings in the same order as input texts.

Examples:
```python
from pulse.core.client import CoreClient
from pulse.core.models import EmbeddingsRequest

client = CoreClient()

# Small dataset - fast mode
resp = client.create_embeddings(EmbeddingsRequest(inputs=["hello", "world"], fast=True))
for doc in resp.embeddings:
    print(doc.text, len(doc.vector))

# Large dataset - automatic batching
large_texts = ["document text"] * 25000
request = EmbeddingsRequest(inputs=large_texts, fast=False)
resp = client.create_embeddings(request)
print(f"Generated {len(resp.embeddings)} embeddings")

# Verify order preservation
assert resp.embeddings[0].text == large_texts[0]
assert resp.embeddings[-1].text == large_texts[-1]

# Handle batching limits
try:
    huge_dataset = ["text"] * 1500000  # Exceeds 1M limit
    request = EmbeddingsRequest(inputs=huge_dataset, fast=False)
    resp = client.create_embeddings(request)
except BatchingError as e:
    if e.error_code == "BATCH_002":
        # Split into smaller chunks
        chunk_size = 1000000
        all_embeddings = []

        for i in range(0, len(huge_dataset), chunk_size):
            chunk = huge_dataset[i:i + chunk_size]
            request = EmbeddingsRequest(inputs=chunk, fast=False)
            resp = client.create_embeddings(request)
            all_embeddings.extend(resp.embeddings)
```

**Performance Optimization:**
```python
# Optimize for large embedding generation
config = BatchingConfig(
    max_concurrent_jobs=5,
    default_batch_sizes={'embeddings': 5000},
    timeout_per_batch=600.0  # Longer timeout for embeddings
)
client = CoreClient.with_client_credentials(batching_config=config)

# Process 500k texts efficiently
large_corpus = ["document"] * 500000
request = EmbeddingsRequest(inputs=large_corpus, fast=False)
result = client.create_embeddings(request)

# Usage metrics are aggregated across all batches
print(f"Total usage: {result.usage}")
```

### `compare_similarity(request: SimilarityRequest, *, await_job_result: bool = True) -> SimilarityResponse | Job`
Compute cosine similarity between strings with advanced text splitting capabilities.

Supply either self-similarity (`set`) or cross-similarity (`set_a` and `set_b`). Enhanced `split` settings support sentence/newline/word unit splitting with multiple aggregation methods and sliding window processing.

Parameters (via `SimilarityRequest`):
- `set: list[str] | None` – Single set (self-similarity). Minimum length 2, maximum 500 (sync) or 44,721 (async).
- `set_a, set_b: list[str] | None` – Cross-similarity sets. Cross-product limited to 20,000 for sync mode.
- `fast: bool | None` – Synchronous or asynchronous.
- `flatten: bool` – Flattened values or matrix.
- `version: str | None` – Model version pin.
- `split: Split | None` – Text splitting configuration with unit/aggregation options.

Text Splitting Options:
- **Units**: `sentence`, `newline`, `word`
- **Aggregation**: `mean`, `max`, `top2`, `top3`
- **Window Processing**: `window_size` and `stride_size` for sliding windows

Example:
```python
from pulse.core.models import SimilarityRequest, Split, UnitAgg

# Self-similarity (matrix)
resp = client.compare_similarity(SimilarityRequest(set=["a", "b", "c"], fast=True, flatten=False))
print(resp.similarity)  # NxN matrix

# Cross-similarity with sentence splitting and max aggregation
sp = Split(set_a=UnitAgg(unit="sentence", agg="max", window_size=2, stride_size=1))
resp = client.compare_similarity(SimilarityRequest(
    set_a=["First sentence. Second sentence."],
    set_b=["Another sentence. Final sentence."],
    split=sp,
    fast=True
))

# Word-level splitting with top2 aggregation
word_split = Split(set_a=UnitAgg(unit="word", agg="top2"))
resp = client.compare_similarity(SimilarityRequest(
    set_a=["hello world example"],
    set_b=["example test case"],
    split=word_split,
    fast=True
))
```

### `batch_similarity(... ) -> list[list[float]]`
Batch large similarity requests under the 10k item limit. Called automatically when `fast=False` and input exceeds limits. You can invoke it directly to manage large inputs.

Keyword parameters:
- `set: list[str] | None`, `set_a: list[str] | None`, `set_b: list[str] | None`
- `flatten: bool = False`
- `version: str | None = None`
- `split: Any | None = None`

Returns a full similarity matrix (`list[list[float]]`).

### `generate_themes(texts: list[str], min_themes=2, max_themes=50, fast=False, *, context=None, version=None, prune=None, interactive=None, initial_sets=None, await_job_result=True) -> ThemesResponse | ThemeSetsResponse | Job`
Cluster texts into latent themes with enhanced functionality.

Parameters:
- `texts: list[str]` – At least 2 non‑empty strings. If fewer than 2 provided, returns an empty `ThemesResponse` without calling the API.
- `min_themes: int` – Minimum cluster count.
- `max_themes: int` – Maximum cluster count.
- `fast: bool` – Synchronous (default) or asynchronous.
- `context: Any | None` – Optional context string to guide clustering.
- `version: str | None` – Model version pin. Use "2025-09-01" for ThemeSetsResponse format.
- `prune: int | None` – Drop N lowest‑frequency themes (0-25).
- `interactive: bool | None` – Enable interactive theme generation mode.
- `initial_sets: int | None` – Number of initial theme sets (1-3). Requires interactive=True when > 1.
- `await_job_result: bool` – Return a `Job` when false.

Returns:
- `ThemesResponse` for standard theme generation
- `ThemeSetsResponse` when version="2025-09-01" is specified (contains multiple theme sets)

Example:
```python
# Standard themes
resp = client.generate_themes(["food was great", "service slow", "loved the vibe"], fast=True)
for th in resp.themes:
    print(th.shortLabel, th.representatives)

# Interactive themes with multiple sets
resp = client.generate_themes(
    ["food was great", "service slow", "loved the vibe"],
    version="2025-09-01",
    interactive=True,
    initial_sets=2,
    fast=True
)
for i, theme_set in enumerate(resp.themeSets):
    print(f"Theme Set {i+1}:")
    for th in theme_set:
        print(f"  {th.shortLabel}: {th.description}")
```

### `analyze_sentiment(texts: list[str], *, version: str | None = None, fast: bool = True, await_job_result: bool = True) -> SentimentResponse | Job`
Classify sentiment for each input text with automatic batching for large datasets.

Parameters:
- `texts: list[str]` – Input texts for sentiment analysis
- `version: str | None` – Model version pin
- `fast: bool` – Processing mode:
  - `True`: Synchronous processing (limit: 200 texts)
  - `False`: Asynchronous processing with automatic batching (limit: 1,000,000 texts)
- `await_job_result: bool` – Return Job when false

**Batching Behavior:**
- **Fast mode (`fast=True`)**: Processes up to 200 texts synchronously. Exceeding this limit raises `BatchingError` with code `BATCH_001`.
- **Slow mode (`fast=False`)**: Automatically batches large datasets into 2,000-text chunks, processes up to 5 batches concurrently, and aggregates results while preserving order.

Examples:
```python
# Small dataset - fast mode
resp = client.analyze_sentiment(["love it", "not great"], fast=True)
print([r.sentiment for r in resp.results])

# Large dataset - automatic batching
large_texts = ["text sample"] * 50000
resp = client.analyze_sentiment(large_texts, fast=False)
print(f"Processed {len(resp.results)} sentiments")

# Handle batching errors
try:
    too_many_texts = ["text"] * 500
    resp = client.analyze_sentiment(too_many_texts, fast=True)  # Will fail
except BatchingError as e:
    if e.error_code == "BATCH_001":
        # Switch to slow mode for automatic batching
        resp = client.analyze_sentiment(too_many_texts, fast=False)
```

**Performance Optimization:**
```python
# For maximum throughput with large datasets
config = BatchingConfig(
    max_concurrent_jobs=5,
    default_batch_sizes={'sentiment': 2000},
    timeout_per_batch=300.0
)
client = CoreClient.with_client_credentials(batching_config=config)

# Process 1 million texts efficiently
massive_dataset = ["text"] * 1000000
result = client.analyze_sentiment(massive_dataset, fast=False)
```

### `extract_elements(inputs: list[str], dictionary: list[str], *, type="named-entities", expand_dictionary=False, expand_dictionary_limit=None, version=None, fast=None, await_job_result=True) -> ExtractionsResponse | Job`
Extract elements from texts with enhanced type control, dictionary expansion, and automatic batching for large datasets.

Parameters:
- `inputs: list[str]` – Input texts for element extraction
- `dictionary: list[str]` – Dictionary terms to extract (3-200 terms)
- `type: str` – Extraction type: "named-entities" (default) or "themes"
- `expand_dictionary: bool` – Expand dictionary entries with synonyms (must be False for type="themes")
- `expand_dictionary_limit: int | None` – Limit number of dictionary expansions
- `version: str | None` – Model version pin
- `fast: bool | None` – Processing mode:
  - `True`: Synchronous processing (limit: 200 texts)
  - `False`: Asynchronous processing with automatic batching (limit: 1,000,000 texts)
- `await_job_result: bool` – Return Job when false

**Batching Behavior:**
- **Fast mode (`fast=True`)**: Processes up to 200 texts synchronously. Exceeding this limit raises `BatchingError` with code `BATCH_001`.
- **Slow mode (`fast=False`)**: Automatically batches large datasets into 2,000-text chunks, processes up to 5 batches concurrently, and preserves extraction result order.

**Extraction Types:**
- **named-entities**: Uses named entity recognition prompts for precise extraction
- **themes**: Uses theme-based prompts for conceptual extraction (requires expand_dictionary=False)

Examples:
```python
# Small dataset - named entity extraction with dictionary expansion
resp = client.extract_elements(
    inputs=["The food was great, but service was slow."],
    dictionary=["food", "service", "quality"],
    type="named-entities",
    expand_dictionary=True,
    expand_dictionary_limit=10,
    fast=True,
)
print(resp.columns)
print(resp.matrix)

# Large dataset - automatic batching
large_reviews = ["Customer review text"] * 10000
dictionary = ["satisfaction", "quality", "service", "price", "experience"]

resp = client.extract_elements(
    inputs=large_reviews,
    dictionary=dictionary,
    type="named-entities",
    expand_dictionary=True,
    fast=False  # Enables automatic batching
)

print(f"Processed {len(resp.matrix)} reviews")
print(f"Extracted {len(resp.columns)} element types")

# Theme-based extraction (no dictionary expansion)
resp = client.extract_elements(
    inputs=["Customer satisfaction survey responses"],
    dictionary=["satisfaction", "experience", "recommendation"],
    type="themes",
    expand_dictionary=False,
    fast=True,
)

# Handle batching errors
try:
    massive_dataset = ["text"] * 500
    resp = client.extract_elements(
        inputs=massive_dataset,
        dictionary=["entity1", "entity2"],
        fast=True  # Will fail with BATCH_001
    )
except BatchingError as e:
    if e.error_code == "BATCH_001":
        # Switch to slow mode for automatic batching
        resp = client.extract_elements(
            inputs=massive_dataset,
            dictionary=["entity1", "entity2"],
            fast=False
        )
```

**Performance Optimization:**
```python
# Optimize for large-scale element extraction
config = BatchingConfig(
    max_concurrent_jobs=5,
    default_batch_sizes={'extractions': 2000},
    timeout_per_batch=450.0  # Longer timeout for complex extractions
)
client = CoreClient.with_client_credentials(batching_config=config)

# Process large document corpus
documents = ["document text"] * 100000
dictionary = ["concept1", "concept2", "concept3", "concept4", "concept5"]

result = client.extract_elements(
    inputs=documents,
    dictionary=dictionary,
    type="named-entities",
    expand_dictionary=True,
    expand_dictionary_limit=5,
    fast=False
)

# Results maintain order and structure
assert len(result.matrix) == len(documents)
print(f"Extraction matrix shape: {len(result.matrix)} x {len(result.columns)}")
```

### `cluster_texts(inputs: list[str], *, k: int, algorithm: str = "kmeans", fast: bool | None = None, await_job_result: bool = True) -> ClusteringResponse | Job`
Cluster texts using embeddings with multiple algorithm options and intelligent automatic batching.

Parameters:
- `inputs: list[str]` – Input texts for clustering (minimum 2 texts)
- `k: int` – Desired number of clusters (1-50)
- `algorithm: str` – Clustering algorithm (default: "kmeans")
- `fast: bool | None` – Processing mode (auto-determined based on input size)
- `await_job_result: bool` – Return Job when false

**Intelligent Batching Behavior:**
- **Small datasets (≤500 texts)**: Processed directly without batching
- **Large datasets (>500 texts)**: Automatically triggers intelligent parallel batching with result reconstruction
- **Maximum limit**: 44,721 texts (based on similarity matrix constraints)
- **Batching strategy**: Uses similarity-style matrix processing with up to 5 concurrent jobs
- **Result consistency**: Clustering results are identical to non-batched clustering for the same input

**Available Algorithms:**
- **kmeans**: Standard k-means clustering (default)
- **skmeans**: Spherical k-means (normalized vectors)
- **agglomerative**: Hierarchical agglomerative clustering
- **hdbscan**: Density-based clustering with noise detection

Examples:
```python
# Small dataset - no batching
resp = client.cluster_texts(
    inputs=["text1", "text2", "text3", "text4"],
    k=2,
    algorithm="kmeans",
    fast=True
)
print(f"Algorithm used: {resp.algorithm}")
for cluster in resp.clusters:
    print(f"Cluster {cluster.clusterId}: {cluster.items}")

# Large dataset - automatic intelligent batching
large_corpus = ["document text"] * 2000
resp = client.cluster_texts(
    inputs=large_corpus,
    k=10,
    algorithm="kmeans"
    # fast mode auto-determined based on size
)

print(f"Clustered {len(large_corpus)} texts into {len(resp.clusters)} clusters")

# Verify clustering quality with large datasets
cluster_sizes = [len(cluster.items) for cluster in resp.clusters]
print(f"Cluster sizes: {cluster_sizes}")

# Different algorithms with large datasets
algorithms = ["kmeans", "skmeans", "agglomerative", "hdbscan"]
corpus = ["varied document content"] * 1500

for algo in algorithms:
    resp = client.cluster_texts(
        inputs=corpus,
        k=8,
        algorithm=algo
    )
    print(f"{algo}: {len(resp.clusters)} clusters generated")

# Handle clustering limits
try:
    massive_corpus = ["text"] * 50000  # Exceeds 44,721 limit
    resp = client.cluster_texts(inputs=massive_corpus, k=20)
except BatchingError as e:
    if e.error_code == "BATCH_002":
        # Reduce dataset size
        reduced_corpus = massive_corpus[:44000]
        resp = client.cluster_texts(inputs=reduced_corpus, k=20)
        print(f"Processed reduced dataset: {len(reduced_corpus)} texts")
```

**Performance Optimization:**
```python
# Configure for optimal clustering performance
config = BatchingConfig(
    max_concurrent_jobs=5,
    timeout_per_batch=900.0  # Longer timeout for complex clustering
)
client = CoreClient.with_client_credentials(batching_config=config)

# Process large document collection
documents = ["research paper abstract"] * 10000
result = client.cluster_texts(
    inputs=documents,
    k=25,
    algorithm="kmeans"
)

# Analyze clustering results
print(f"Clustering completed:")
print(f"  Total documents: {len(documents)}")
print(f"  Clusters generated: {len(result.clusters)}")
print(f"  Algorithm used: {result.algorithm}")

# Verify result consistency
total_clustered = sum(len(cluster.items) for cluster in result.clusters)
assert total_clustered == len(documents), "All documents should be clustered"
```

**Batching Information Messages:**
When clustering datasets >500 texts, you may see informational message `BATCH_005`:
```
Input size 2000 exceeds threshold 500. Automatic batching enabled.
```
This is normal behavior and indicates that intelligent batching is handling your large dataset automatically.

### `generate_summary(inputs: list[str], question: str, *, length: str | None = None, preset: str | None = None, fast: bool | None = None, await_job_result: bool = True) -> SummariesResponse | Job`
Summarize inputs following a guiding question.

Parameters:
- `inputs: list[str]`
- `question: str`
- `length: str | None` – One of `bullet-points`, `short`, `medium`, `long`.
- `preset: str | None` – One of `five-point`, `ten-point`, `one-tweet`, `three-tweets`, `one-para`, `exec`, `two-pager`, `one-pager`.
- `fast: bool | None`
- `await_job_result: bool`

### `estimate_usage(feature: str, inputs: list[str]) -> UsageEstimateResponse`
Estimate credit usage for a feature without authentication.

Parameters:
- `feature: str` – Feature to estimate: "embeddings", "sentiment", "themes", "extractions", "summaries", "clustering", "similarity".
- `inputs: list[str]` – Input texts for estimation.

Returns:
- `UsageEstimateResponse` with usage estimation details.

Note: This endpoint does not require authentication and can be used for planning and budgeting API usage.

Example:
```python
# Estimate usage for theme generation
estimate = client.estimate_usage(
    feature="themes",
    inputs=["sample text 1", "sample text 2", "sample text 3"]
)
print(f"Estimated usage: {estimate.usage}")

# Estimate usage for large similarity computation
estimate = client.estimate_usage(
    feature="similarity",
    inputs=["text"] * 100  # 100 texts for self-similarity
)
print(f"Estimated credits: {estimate.usage}")
```

### `get_job_status(job_id: str) -> Job`
Poll job status by ID. Returns a `Job` object which can be `.wait()`ed. Useful when you stored a job id and want to resume later.

### `close() -> None`
Close underlying HTTP connections.

## Exceptions

All non‑successful responses raise `pulse.core.exceptions.PulseAPIError` with useful context (status, code, message).
