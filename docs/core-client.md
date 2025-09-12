# Core Client

Module: `pulse.core.client`

`CoreClient` is a synchronous, HTTPX-based client for the Pulse REST API. It exposes endpoints for embeddings, similarity, themes, sentiment, extractions, clustering, and summaries. Methods support both synchronous (fast) and asynchronous (job) modes where applicable.

## Constructing a Client

```python
from pulse.core.client import CoreClient

# Default construction uses gzip client + auto_auth()
client = CoreClient()  # base_url defaults to PROD; set auth env vars or pass auth explicitly

# Provide explicit base_url/timeout
client = CoreClient(base_url="https://dev.core.researchwiseai.com/pulse/v1", timeout=30.0)

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
Compute cosine similarity between strings.

Supply either self-similarity (`set`) or cross-similarity (`set_a` and `set_b`). Optional `split` settings allow sentence/newline unit splitting with mean/max aggregation.

Parameters (via `SimilarityRequest`):
- `set: list[str] | None` – Single set (self-similarity). Minimum length 2.
- `set_a, set_b: list[str] | None` – Cross-similarity sets.
- `fast: bool | None` – Synchronous or asynchronous.
- `flatten: bool` – Flattened values or matrix.
- `version: str | None` – Model version pin.
- `split: Split | None` – Global or per‑set unit/agg.

Example:
```python
from pulse.core.models import SimilarityRequest, Split, UnitAgg

# Self-similarity (matrix)
resp = client.compare_similarity(SimilarityRequest(set=["a", "b", "c"], fast=True, flatten=False))
print(resp.similarity)  # NxN matrix

# Cross-similarity with splitting
sp = Split(set_a=UnitAgg(unit="sentence", agg="mean"))
resp = client.compare_similarity(SimilarityRequest(set_a=["a"], set_b=["a b", "c"], split=sp, fast=True))
```

### `batch_similarity(... ) -> list[list[float]]`
Batch large similarity requests under the 10k item limit. Called automatically when `fast=False` and input exceeds limits. You can invoke it directly to manage large inputs.

Keyword parameters:
- `set: list[str] | None`, `set_a: list[str] | None`, `set_b: list[str] | None`
- `flatten: bool = False`
- `version: str | None = None`
- `split: Any | None = None`

Returns a full similarity matrix (`list[list[float]]`).

### `generate_themes(texts: list[str], min_themes=2, max_themes=50, fast=True, *, context=None, version=None, prune=None, await_job_result=True) -> ThemesResponse | Job`
Cluster texts into latent themes.

Parameters:
- `texts: list[str]` – At least 2 non‑empty strings. If fewer than 2 provided, returns an empty `ThemesResponse` without calling the API.
- `min_themes: int` – Minimum cluster count.
- `max_themes: int` – Maximum cluster count.
- `fast: bool` – Synchronous (default) or asynchronous.
- `context: Any | None` – Optional context string to guide clustering.
- `version: str | None` – Model version pin.
- `prune: int | None` – Drop N lowest‑frequency themes.
- `await_job_result: bool` – Return a `Job` when false.

Example:
```python
resp = client.generate_themes(["food was great", "service slow", "loved the vibe"], fast=True)
for th in resp.themes:
    print(th.shortLabel, th.representatives)
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

### `extract_elements(texts: list[str] | None, categories: list[str] | list[Theme] | None, *, version=None, dictionary=None, expand_dictionary=None, use_ner=None, use_llm=None, threshold=None, fast=None, await_job_result=True) -> ExtractionsResponse | Job`
Extract elements for the given categories from texts. Accepts category names as strings or `Theme` objects.

Parameters:
- `texts: list[str]` – Input texts (alias `inputs` supported for backward compatibility).
- `categories: list[str] | list[Theme]` – Categories (alias `themes` supported for backward compatibility).
- `version: str | None` – Model version pin.
- `dictionary: dict[str, list[str]] | None` – Category→terms mapping to bias extraction.
- `expand_dictionary: bool | None` – Expand dictionary entries with synonyms.
- `use_ner: bool | None` – Enable NER.
- `use_llm: bool | None` – Enable LLM assistance.
- `threshold: float | None` – Score threshold.
- `fast: bool | None` – Synchronous/asynchronous.
- `await_job_result: bool`

Example:
```python
resp = client.extract_elements(
    texts=["The food was great, but service was slow."],
    categories=["food", "service"],
    dictionary={"food": ["food"], "service": ["service"]},
    use_ner=True,
    fast=True,
)
print(resp.columns)
print(resp.matrix)
```

### `cluster_texts(inputs: list[str], *, k: int, algorithm: str | None = None, fast: bool | None = None, await_job_result: bool = True) -> ClusteringResponse | Job`
Cluster texts using embeddings.

Parameters:
- `inputs: list[str]`
- `k: int` – Desired number of clusters.
- `algorithm: str | None` – One of `kmeans`, `skmeans`, `agglomerative`, `hdbscan`.
- `fast: bool | None`
- `await_job_result: bool`

### `generate_summary(inputs: list[str], question: str, *, length: str | None = None, preset: str | None = None, fast: bool | None = None, await_job_result: bool = True) -> SummariesResponse | Job`
Summarize inputs following a guiding question.

Parameters:
- `inputs: list[str]`
- `question: str`
- `length: str | None` – One of `bullet-points`, `short`, `medium`, `long`.
- `preset: str | None` – One of `five-point`, `ten-point`, `one-tweet`, `three-tweets`, `one-para`, `exec`, `two-pager`, `one-pager`.
- `fast: bool | None`
- `await_job_result: bool`

### `get_job_status(job_id: str) -> Job`
Poll job status by ID. Returns a `Job` object which can be `.wait()`ed. Useful when you stored a job id and want to resume later.

### `close() -> None`
Close underlying HTTP connections.

## Exceptions

All non‑successful responses raise `pulse.core.exceptions.PulseAPIError` with useful context (status, code, message).

