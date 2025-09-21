# Core Client

Module: `pulse.core.client`

`CoreClient` is a synchronous, HTTPX-based client for the Pulse REST API. It exposes endpoints for embeddings, similarity, themes, sentiment, extractions, clustering, and summaries. Methods support both synchronous (fast) and asynchronous (job) modes where applicable.

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
Generate dense vector embeddings.

Parameters:
- `request: EmbeddingsRequest` – Request model. Fields:
  - `inputs: list[str]` – 1..2000 input strings.
  - `fast: bool | None` – If true, synchronous; otherwise async.
- `await_job_result: bool` – Return `Job` when false and server responds 202.

Example:
```python
from pulse.core.client import CoreClient
from pulse.core.models import EmbeddingsRequest

client = CoreClient()
resp = client.create_embeddings(EmbeddingsRequest(inputs=["hello", "world"], fast=True))
for doc in resp.embeddings:
    print(doc.text, len(doc.vector))
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

### `generate_themes(texts: list[str], min_themes=2, max_themes=50, fast=True, *, context=None, version=None, prune=None, interactive=None, initial_sets=None, await_job_result=True) -> ThemesResponse | ThemeSetsResponse | Job`
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
Classify sentiment for each input text.

Parameters:
- `texts: list[str]`
- `version: str | None` – Model version pin.
- `fast: bool` – Synchronous; large inputs are chunked automatically when needed.
- `await_job_result: bool`

Example:
```python
resp = client.analyze_sentiment(["love it", "not great"], fast=True)
print([r.sentiment for r in resp.results])
```

### `extract_elements(inputs: list[str], dictionary: list[str], *, type="named-entities", expand_dictionary=False, expand_dictionary_limit=None, version=None, fast=None, await_job_result=True) -> ExtractionsResponse | Job`
Extract elements from texts with enhanced type control and dictionary expansion.

Parameters:
- `inputs: list[str]` – Input texts (1-200 sync, 1-5,000 async).
- `dictionary: list[str]` – Dictionary terms to extract (3-200 terms).
- `type: str` – Extraction type: "named-entities" (default) or "themes".
- `expand_dictionary: bool` – Expand dictionary entries with synonyms (must be False for type="themes").
- `expand_dictionary_limit: int | None` – Limit number of dictionary expansions.
- `version: str | None` – Model version pin.
- `fast: bool | None` – Synchronous/asynchronous.
- `await_job_result: bool` – Return Job when false.

Extraction Types:
- **named-entities**: Uses named entity recognition prompts for precise extraction
- **themes**: Uses theme-based prompts for conceptual extraction (requires expand_dictionary=False)

Example:
```python
# Named entity extraction with dictionary expansion
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

# Theme-based extraction (no dictionary expansion)
resp = client.extract_elements(
    inputs=["Customer satisfaction survey responses"],
    dictionary=["satisfaction", "experience", "recommendation"],
    type="themes",
    expand_dictionary=False,
    fast=True,
)
```

### `cluster_texts(inputs: list[str], *, k: int, algorithm: str = "kmeans", fast: bool | None = None, await_job_result: bool = True) -> ClusteringResponse | Job`
Cluster texts using embeddings with multiple algorithm options.

Parameters:
- `inputs: list[str]` – Input texts (2-500 sync, 2-44,721 async).
- `k: int` – Desired number of clusters (1-50).
- `algorithm: str` – Clustering algorithm (default: "kmeans").
- `fast: bool | None` – Synchronous/asynchronous.
- `await_job_result: bool` – Return Job when false.

Available Algorithms:
- **kmeans**: Standard k-means clustering (default)
- **skmeans**: Spherical k-means (normalized vectors)
- **agglomerative**: Hierarchical agglomerative clustering
- **hdbscan**: Density-based clustering with noise detection

Example:
```python
# Standard k-means clustering
resp = client.cluster_texts(
    inputs=["text1", "text2", "text3", "text4"],
    k=2,
    algorithm="kmeans",
    fast=True
)
print(f"Algorithm used: {resp.algorithm}")
for cluster in resp.clusters:
    print(f"Cluster {cluster.clusterId}: {cluster.items}")

# Spherical k-means for normalized similarity
resp = client.cluster_texts(
    inputs=["document about AI", "machine learning paper", "cooking recipe", "food blog"],
    k=2,
    algorithm="skmeans",
    fast=True
)

# HDBSCAN for density-based clustering
resp = client.cluster_texts(
    inputs=["similar text 1", "similar text 2", "outlier text", "another similar text"],
    k=2,
    algorithm="hdbscan",
    fast=True
)
```

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
