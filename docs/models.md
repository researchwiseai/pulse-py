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
  - `unit: Literal["sentence", "newline"]`
  - `agg: Literal["mean", "max"]` (default `"mean"`)
- `Split`
  - `unit: Literal["sentence", "newline"] | None`
  - `agg: Literal["mean", "max"] | None`
  - `set_a: UnitAgg | None`, `set_b: UnitAgg | None`
- `SimilarityRequest`
  - `set: list[str] | None`
  - `set_a: list[str] | None`, `set_b: list[str] | None`
  - `fast: bool | None`
  - `flatten: bool` (default False)
  - `version: str | None`
  - `split: Split | None`
- `SimilarityResponse`
  - `scenario: Literal["self", "cross"]`
  - `mode: Literal["matrix", "flattened"]`
  - `n: int`
  - `flattened: list[float]`
  - `matrix: list[list[float]] | None`
  - `requestId: str | None`
  - Property: `similarity -> list[list[float]]` – Returns a full matrix regardless of mode.

## Themes

- `Theme`
  - `shortLabel: str`
  - `label: str`
  - `description: str`
  - `representatives: list[str]` (exactly two strings)
- `ThemesResponse`
  - `themes: list[Theme]`
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
  - `texts: list[str]`
  - `categories: list[str]`
  - `dictionary: dict[str, list[str]] | None`
  - `expand_dictionary: bool | None`
  - `use_ner: bool | None`
  - `use_llm: bool | None`
  - `threshold: float | None`
- `ExtractionsResponse`
  - `columns: list[{ category: str, term: str }]`
  - `matrix: list[list[str]]` – Rows per input, columns aligned to `columns`.
  - `requestId: str | None`

## Clustering

- `ClusteringRequest`
  - `inputs: list[str]`
  - `k: int` (1..50)
  - `algorithm: Literal["kmeans", "skmeans", "agglomerative", "hdbscan"] | None`
  - `fast: bool | None`
- `Cluster`
  - `clusterId: int`
  - `items: list[str]`
- `ClusteringResponse`
  - `algorithm: str`
  - `clusters: list[Cluster]`
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

