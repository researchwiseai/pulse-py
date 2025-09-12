# Starters

The `pulse.starters` module provides convenient helpers for common tasks. Each function accepts either a list of strings or a path to a text/CSV/TSV/Excel file where the first column is treated as the text input.

Supported file types: `.txt`, `.csv`, `.tsv`, `.xls`, `.xlsx`.

## Loading Inputs

`get_strings(source: list[str] | str) -> list[str]` loads strings from a list or file path.

## Sentiment

```python
from pulse.starters import sentiment_analysis

resp = sentiment_analysis(["great", "bad"])  # or a file path
print([r.sentiment for r in resp.results])
```

Parameters:
- `input_data: list[str] | str`
- `version: str | None = None`
- `await_job_result: bool = True` – If False returns a `Job`.
- `auth: pulse.auth._BaseOAuth2Auth | None = None` – Optional auth.
- `client: CoreClient | None = None` – Existing client; else one is created.

Notes:
- Automatically sets `fast=True` when `len(texts) <= 200`.

## Theme Allocation

```python
from pulse.starters import theme_allocation

res = theme_allocation(["food good, service slow"], themes=["Food", "Service"])
print(res.assign_single())
```

Parameters:
- `input_data: list[str] | str`
- `themes: list[str] | None` – If None, the helper runs a `ThemeGeneration` step automatically.
- `auth: pulse.auth._BaseOAuth2Auth | None = None`
- `client: CoreClient | None = None`

Returns: `ThemeAllocationResult` with helpers like `assign_single()`, `assign_multi()`, `heatmap()`, and `to_dataframe()`.

## Clustering

```python
from pulse.starters import cluster_analysis

resp = cluster_analysis(["good", "bad", "ok"], k=2)
print(resp.clusters)
```

Parameters:
- `input_data: list[str] | str`
- `k: int` – Number of clusters.
- `algorithm: str | None = None`
- `await_job_result: bool = True`
- `auth: pulse.auth._BaseOAuth2Auth | None = None`
- `client: CoreClient | None = None`

## Summaries

```python
from pulse.starters import summarize

resp = summarize(["Great food, slow service"], question="What do diners say?", length="short")
print(resp.summary)
```

Parameters:
- `input_data: list[str] | str`
- `question: str`
- `length: str | None = None` – One of `bullet-points`, `short`, `medium`, `long`.
- `preset: str | None = None` – One of `five-point`, `ten-point`, `one-tweet`, `three-tweets`, `one-para`, `exec`, `two-pager`, `one-pager`.
- `await_job_result: bool = True`
- `auth: pulse.auth._BaseOAuth2Auth | None = None`
- `client: CoreClient | None = None`

