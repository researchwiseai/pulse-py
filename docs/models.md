# Data Models

Pydantic models define request and response payloads. These appear under `pulse.core.models` and are used across the Core Client and Analyzer results.

## Embeddings

- `EmbeddingsRequest`
  - `inputs: list[str]` (1..2000)
  - `fast: bool | None`
- `EmbeddingDocument`
  - `id: str | None`
  - `text: str`
  - `vector: list[float]`
- `EmbeddingsResponse`
  - `embeddings: list[EmbeddingDocument]`
  - `requestId: str | None`

## Similarity

- `UnitAgg`
  - `unit: Literal["sentence", "newline", "word"]` – Text splitting unit
  - `agg: Literal["mean", "max", "top2", "top3"]` (default `"mean"`) – Aggregation method
  - `window_size: int` (default 1, ≥1) – Window size for sliding window processing
  - `stride_size: int` (default 1, ≥1) – Stride size for sliding window processing
- `Split`
  - `set_a: UnitAgg | None` – Splitting configuration for set_a
  - `set_b: UnitAgg | None` – Splitting configuration for set_b
- `SimilarityRequest`
  - `set: list[str] | None` (2-44,721 items) – Self-similarity input texts
  - `set_a: list[str] | None`, `set_b: list[str] | None` – Cross-similarity sets
  - `fast: bool | None` – Synchronous or asynchronous
  - `flatten: bool` (default False) – Return flattened results
  - `version: str | None` – API version
  - `split: Split | None` – Text splitting configuration
- `SimilarityResponse`
  - `scenario: Literal["self", "cross"]`
  - `mode: Literal["matrix", "flattened"]`
  - `n: int`
  - `flattened: list[float]`
  - `matrix: list[list[float]] | None`
  - `requestId: str | None`
  - Property: `similarity -> list[list[float]]` – Returns a full matrix regardless of mode.

## Themes

- `ThemesRequest`
  - `inputs: list[str]` (2-500 items) – Input texts
  - `minThemes: int | None` (≥1) – Minimum number of themes
  - `maxThemes: int | None` (≤50) – Maximum number of themes
  - `context: str | None` – Context to steer theme generation
  - `version: str | None` – API version (use "2025-09-01" for ThemeSetsResponse)
  - `prune: int | None` (0-25) – Pruning threshold
  - `interactive: bool | None` – Enable interactive mode
  - `initialSets: int | None` (1-3) – Number of initial theme sets (requires interactive=True when >1)
  - `fast: bool | None` – Synchronous or asynchronous
- `Theme`
  - `shortLabel: str` – Concise 2-4 word name
  - `label: str` – Descriptive title
  - `description: str` – 1-2 sentence summary
  - `representatives: list[str]` (exactly two strings) – Representative input strings
- `ThemesResponse`
  - `themes: list[Theme]`
  - `requestId: str | None`
- `ThemeSetsResponse` (version 2025-09-01)
  - `themeSets: list[list[Theme]]` (max 3 sets) – Multiple theme sets
  - `requestId: str | None`

## Sentiment

- `SentimentResult`
  - `sentiment: Literal["positive", "negative", "neutral", "mixed"]`
  - `confidence: float`
- `SentimentResponse`
  - `results: list[SentimentResult]`
  - `requestId: str | None`
  - Property: `sentiments -> list[str]` convenience accessor for labels.

## Extractions

- `ExtractionsRequest`
  - `inputs: list[str]` (1-5,000 items) – Input texts
  - `dictionary: list[str]` (3-200 terms) – Dictionary terms to extract
  - `type: Literal["named-entities", "themes"]` (default "named-entities") – Extraction type
  - `expand_dictionary: bool` (default False) – Expand dictionary with synonyms
  - `expand_dictionary_limit: int | None` – Limit for dictionary expansions
  - `version: str | None` – API version
  - `fast: bool | None` – Synchronous or asynchronous
  - Deprecated fields: `category`, `texts`, `categories`, `use_ner`, `use_llm`, `threshold`
- `ExtractionsResponse`
  - `columns: list[{ category: str, term: str }]`
  - `matrix: list[list[str]]` – Rows per input, columns aligned to `columns`.
  - `requestId: str | None`

## Clustering

- `ClusteringRequest`
  - `inputs: list[str]` (2-44,721 items) – Input texts
  - `k: int` (1-50) – Number of clusters
  - `algorithm: Literal["kmeans", "skmeans", "agglomerative", "hdbscan"]` (default "kmeans") – Clustering algorithm
  - `fast: bool | None` – Synchronous or asynchronous
- `Cluster`
  - `clusterId: int` – Cluster identifier
  - `items: list[str]` – Items assigned to this cluster
- `ClusteringResponse`
  - `algorithm: str` – Algorithm used for clustering
  - `clusters: list[Cluster]` – List of cluster groups
  - `requestId: str | None`

## Summaries

- `SummariesRequest`
  - `inputs: list[str]`
  - `question: str`
  - `length: Literal["bullet-points", "short", "medium", "long"] | None`
  - `preset: Literal["five-point","ten-point","one-tweet","three-tweets","one-para","exec","two-pager","one-pager"] | None`
  - `fast: bool | None`
- `SummariesResponse`
  - `summary: str`
  - `requestId: str | None`

## Usage Estimation

- `UsageEstimateRequest`
  - `feature: Literal["embeddings", "sentiment", "themes", "extractions", "summaries", "clustering", "similarity"]` – Feature to estimate
  - `inputs: list[str]` (≥1) – Input texts for estimation
- `UsageEstimateResponse`
  - `usage: dict[str, Any]` – Estimated usage information

## Usage Tracking

- `UsageRecord`
  - `feature: str` – Name of the feature
  - `quantity: int` – Quantity consumed (replaces deprecated `units` field)
  - Property: `units -> int` – Backward compatibility accessor for `quantity`
- `UsageReport`
  - `total: int` – Total units consumed
  - `records: list[UsageRecord]` – Per-feature usage records

## Error Handling

- `ErrorDetail`
  - `code: str | None` – Error code
  - `message: str` – Error message
  - `path: list[str | int] | None` – Path to the field that caused the error
  - `field: str | None` – Field name that caused the error
  - `location: Literal["body", "query", "header", "path"] | None` – Location of the error
- `ErrorResponse`
  - `code: str` – Error code
  - `message: str` – Error message
  - `errors: list[ErrorDetail] | None` – Detailed field-level error information
  - `meta: dict[str, Any] | None` – Additional metadata about the error
