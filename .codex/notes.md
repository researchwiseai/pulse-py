# Refactor Changelog

This document summarizes the major refactoring to align Pulse client models and flows with the OpenAPI specification.

## Core Models
- Introduced `EmbeddingDocument` and retained `EmbeddingsResponse` for vector embeddings
  - Fields: `text`, `vector`, optional `id`, optional `requestId`
- Overhauled `SimilarityResponse` to spec fields:
  - `scenario`: self or cross
  - `mode`: matrix or flattened
  - `n`: number of inputs
  - `flattened`: list of floats
  - `matrix`: optional full 2D list
  - optional `requestId`
  - Added `@model_validator` to translate legacy `similarity` field
  - Added `.similarity` alias property for backward compatibility
- Replaced legacy string list in `ThemesResponse` with spec model:
  - Defined `Theme` model with `shortLabel`, `label`, `description`, `representatives`
  - `ThemesResponse` now holds `themes: List[Theme]` and optional `requestId`
- Standardized `SentimentResponse`:
  - `SentimentResult` model with `sentiment` enum and `confidence`
  - `SentimentResponse` holds `results: List[SentimentResult]` + optional `requestId`
  - Legacy `sentiments` field auto-mapped via `@model_validator`
- `ExtractionsResponse` unchanged except optional `requestId`

## CoreClient Updates
- `create_embeddings`: fast-mode placeholder yields empty vectors + `requestId=None`
- `compare_similarity`: error & async placeholders follow spec shape
- `generate_themes`: returns `ThemesResponse` with real `Theme` objects or fast-sync placeholder
- `analyze_sentiment`: returns `SentimentResponse` with structured `SentimentResult` or placeholder
- `extract_elements`: placeholder + structured responses

## Analysis Result Wrappers
- `ThemeGenerationResult`:
  - `.themes` returns list of `shortLabel`s
  - `to_dataframe()` outputs theme metadata DataFrame (no assignments)
- Removed legacy `.assignments` getter from generation result

## DSL / Processes
- `ThemeGeneration`: emits spec-based `ThemesResponse`
- `ThemeAllocation`: now
  - computes assignments via cosine similarity between texts and theme labels
  - no longer depends on a precomputed `assignments` field
- `ThemeExtraction`, `SentimentProcess`, `Cluster` adapt seamlessly to new schema

## Tests & Examples
- Updated dummy clients in tests to return spec-based models (`Theme`, `SentimentResult`)
- Overhauled all `test_*` files to assert on new fields & types
- Example notebooks (if any) must be updated to use new `to_dataframe()` and `.themes` lists of `Theme`

---
All existing `pytest` runs are green. Future work:
- Migrate example notebooks to new models
- Update high-level DSL docs to reflect metadata-based theme outputs
- Remove any remaining legacy code/comments after full QA
