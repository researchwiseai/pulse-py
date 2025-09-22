# Analyzer and Processes

The Analyzer orchestrates one or more processes against a dataset, with simple caching and rich result helpers for downstream analysis and visualization.

Module: `pulse.analysis.analyzer`, `pulse.analysis.processes`, `pulse.analysis.results`

## Async Support

For async/await applications, use `AsyncAnalyzer` from `pulse.analysis.async_analyzer`:

```python
import asyncio
from pulse.analysis.async_analyzer import AsyncAnalyzer
from pulse.analysis.processes import ThemeGeneration, SentimentProcess

async def main():
    texts = ["I love pizza", "I hate rain"]
    processes = [ThemeGeneration(min_themes=2), SentimentProcess()]

    async with AsyncAnalyzer(dataset=texts, processes=processes, cache_dir=".pulse_cache") as analyzer:
        results = await analyzer.run()

    print(results.theme_generation.to_dataframe())
    print(results.sentiment.summary())

asyncio.run(main())
```

The async analyzer provides the same interface as the sync version but with full async/await support. See the [Async Patterns Guide](async-patterns.md) for detailed documentation.

## Analyzer

```python
from pulse.analysis.analyzer import Analyzer
from pulse.analysis.processes import ThemeGeneration, SentimentProcess

texts = ["I love pizza", "I hate rain"]
processes = [ThemeGeneration(min_themes=2), SentimentProcess()]
with Analyzer(dataset=texts, processes=processes, cache_dir=".pulse_cache") as az:
    results = az.run()

print(results.theme_generation.to_dataframe())
print(results.sentiment.summary())
```

Constructor parameters:
- `dataset: Sequence[str] | pandas.Series` – Input data.
- `processes: Sequence[Process] | None` – Ordered list of process instances (see below).
- `fast: bool | None` – Default fast/async flag used by processes when not specified on the process.
- `cache_dir: str | None` – If set and `use_cache=True`, stores per‑process results in `diskcache`.
- `use_cache: bool` – Enable/disable caching (default True).
- `client: CoreClient | None` – `CoreClient` instance to use; defaults to `CoreClient(auth=auth)`.
- `auth: pulse.auth._BaseOAuth2Auth | None` – Auth to pass to an internally constructed `CoreClient`.

Methods:
- `run() -> AnalysisResult` – Executes processes in order, returning a container exposing each process result as an attribute using the process id / alias.
- `clear_cache()` – Clears on‑disk cache (if enabled).
- `close()` – Closes the underlying client and cache.

Dependency resolution:
- Some processes depend on others (e.g., `ThemeAllocation` depends on `ThemeGeneration`). The Analyzer automatically inserts missing dependencies with sensible defaults.

## Processes (module: `pulse.analysis.processes`)

All processes implement a common `Process` protocol: `id` (str), `depends_on` (tuple of ids), and `run(ctx)`.

### `ThemeGeneration`
Clusters texts into latent themes.

Parameters:
- `min_themes: int = 2`, `max_themes: int = 50` – Bounds for clustering.
- `context: Any = None` – Optional guiding context.
- `version: str | None = None` – Model version pin.
- `prune: int | None = None` – Drop N lowest‑frequency themes.
- `fast: bool | None = None` – Overrides Analyzer’s default fast flag.
- `await_job_result: bool = True` – Return `Job` when False.

Result wrapper: `ThemeGenerationResult` (see below).

### `SentimentProcess`
Classifies sentiment for each text.

Parameters:
- `fast: bool | None = None`

Result wrapper: `SentimentResult`.

### `ThemeAllocation`
Allocates each text to themes by computing similarity between texts and theme labels, then choosing the most similar themes.

Parameters:
- `themes: list[str] | None = None` – Use explicit themes; otherwise use themes from the most recent `ThemeGeneration` or a specified source.
- `single_label: bool = True` – If true, assign only top theme above threshold.
- `fast: bool | None = None`
- `threshold: float = 0.5` – Minimum similarity for assignment.

Returns a raw dict in Analyzer (`{"themes", "assignments", "similarity"}`), which the Analyzer wraps as `ThemeAllocationResult`.

### `ThemeExtraction`
Extracts elements matching themes from inputs.

Parameters:
- `themes: list[str] | None = None`
- `version: str | None = None`
- `fast: bool | None = None`
- `dictionary: dict[str, list[str]] | None = None`
- `expand_dictionary: bool | None = None`
- `use_ner: bool | None = None`
- `use_llm: bool | None = None`
- `threshold: float | None = None`

Result wrapper: `ThemeExtractionResult`.

### `Cluster`
Computes a similarity matrix suitable for downstream clustering (performed client‑side).

Parameters:
- `fast: bool | None = None`

Result wrapper: `ClusterResult`.

## Result Helpers (module: `pulse.analysis.results`)

### `ThemeGenerationResult`
- `themes: list[Theme]` – Structured themes.
- `to_dataframe() -> pandas.DataFrame` – Columns: `shortLabel, label, description, representative_1, representative_2`.

### `SentimentResult`
- `sentiments: list[SentimentResultModel]` – Sentiment label + confidence per input.
- `to_dataframe() -> pandas.DataFrame` – Columns: `text, sentiment, confidence`.
- `summary() -> pandas.Series` – Counts per sentiment label.
- `plot_distribution(**kwargs)` – Matplotlib bar chart of the distribution.

### `ThemeAllocationResult`
- `assign_single(threshold: float | None = None) -> pandas.Series` – Single best theme per text above threshold (or None).
- `assign_multi(k: int | None = None) -> pandas.DataFrame` – Top‑k theme labels per text, ordered by similarity.
- `bar_chart(**kwargs)` – Horizontal bar chart of assignment counts.
- `heatmap(**kwargs)` – Heatmap of the similarity matrix (uses seaborn if available, falls back to matplotlib).
- `to_dataframe() -> pandas.DataFrame` – Long format: `text, theme, score`.

### `ClusterResult`
- `matrix` – NumPy array view of the similarity matrix.
- `kmeans(n_clusters, **kwargs)` – Runs scikit‑learn KMeans on the matrix.
- `dbscan(eps=0.4, min_samples=5, **kwargs)` – Runs scikit‑learn DBSCAN on a distance transform.
- `plot_scatter(**kwargs)` – 2D PCA scatter plot.
- `dendrogram(**kwargs)` – Hierarchical clustering dendrogram (SciPy).

### `ThemeExtractionResult`
- `extractions: list[list[list[str]]]` – Nested extractions per text per theme.
- `to_dataframe() -> pandas.DataFrame` – Long format: `text, category, extraction`.

## Optional Dependencies for Plots/ML

Some helper methods use third‑party libraries:
- `matplotlib` – plotting utilities across results
- `seaborn` – heatmap in `ThemeAllocationResult.heatmap()` (falls back if missing)
- `numpy` – matrix handling in `ClusterResult`
- `scikit‑learn` – `kmeans`, `dbscan`, and `plot_scatter` (PCA)
- `scipy` – `dendrogram()`

Install these as needed, for example:
```bash
pip install matplotlib seaborn numpy scikit-learn scipy
```
