"""High-level orchestrator for running processes."""

from typing import Sequence, Optional, Union, Any
import pandas as pd

from pulse.core.client import CoreClient
from pulse.auth import OAuth2Credentials
from pulse.analysis.processes import Process
from pulse.analysis.results import (
    ThemeGenerationResult,
    SentimentResult,
    ThemeAllocationResult,
    ClusterResult,
    ThemeExtractionResult,
)


class Analyzer:
    """High-level orchestrator for Pulse API processes with caching."""

    def __init__(
        self,
        dataset: Union[Sequence[str], pd.Series],
        processes: Optional[Sequence[Process]] = None,
        *,
        fast: Optional[bool] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        client: Optional[CoreClient] = None,
        auth: Optional[OAuth2Credentials] = None,
    ) -> None:
        # Dataset as pandas Series
        if isinstance(dataset, pd.Series):
            self.dataset = dataset
        else:
            self.dataset = pd.Series(dataset)
        # Processes to execute
        self.processes = list(processes) if processes else []
        # Automatically include any dependent processes
        self._resolve_dependencies()
        # Fast/slow flag per process
        self.fast = fast if fast is not None else False
        # Persistent caching setup
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        if use_cache and cache_dir:
            from diskcache import Cache

            self._cache = Cache(cache_dir)
        else:
            self._cache = None
        # Core client and auth
        self.client = client or CoreClient()
        self.auth = auth
        # In-memory results
        self.results: dict[str, Any] = {}

    def _resolve_dependencies(self) -> None:
        """Automatically include any processes that are
        dependencies of specified processes."""
        from pulse.analysis.processes import ThemeGeneration

        existing_ids = {p.id for p in self.processes}
        resolved: list[Process] = []
        for proc in self.processes:
            for dep in getattr(proc, "depends_on", ()):
                if dep not in existing_ids:
                    if dep == ThemeGeneration.id:
                        resolved.append(ThemeGeneration())
                        existing_ids.add(dep)
                    else:
                        raise RuntimeError(f"Missing dependency process '{dep}'")
            resolved.append(proc)
        self.processes = resolved

    def run(self) -> "AnalysisResult":
        """Run the configured processes (with simple caching and wrapping results)."""
        results: dict[str, Any] = {}
        texts = self.dataset.tolist()
        for process in self.processes:
            key = self._make_cache_key(process) if self._cache is not None else None
            if self.use_cache and self._cache is not None and key in self._cache:
                wrapped = self._cache[key]
            else:
                raw = process.run(self)
                # Wrap raw response in high-level result based on original process id
                orig_id = getattr(process, "_orig_id", process.id)
                if orig_id == "theme_generation":
                    wrapped = ThemeGenerationResult(raw, texts)
                elif orig_id == "sentiment":
                    wrapped = SentimentResult(raw, texts)
                elif orig_id == "theme_allocation":
                    wrapped = ThemeAllocationResult(
                        texts,
                        raw["themes"],
                        raw["assignments"],
                        process.single_label,
                        process.threshold,
                        similarity=raw.get("similarity"),
                    )
                elif orig_id == "cluster":
                    wrapped = ClusterResult(raw, texts)
                elif orig_id == "theme_extraction":
                    wrapped = ThemeExtractionResult(raw, texts, process.themes)
                else:
                    wrapped = raw
                if self.use_cache and self._cache is not None:
                    self._cache[key] = wrapped
            results[process.id] = wrapped
            # expose partial results for downstream dependencies
            self.results = results
        self.results = results
        return AnalysisResult(results)

    def clear_cache(self) -> None:
        """Clear the on-disk cache, if enabled."""
        if self._cache is not None:
            self._cache.clear()

    def close(self) -> None:
        """Close underlying HTTP client and persistent cache."""
        try:
            self.client.close()
        except Exception:
            pass
        if self._cache:
            try:
                self._cache.close()
            except Exception:
                pass

    def __enter__(self):  # type: ignore
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        self.close()

    def _make_cache_key(self, process: Process) -> str:
        import pickle
        import hashlib

        # data to hash: dataset values, process id, process attributes
        data = (
            tuple(self.dataset.tolist()),
            process.id,
            tuple(
                sorted(
                    (k, getattr(process, k))
                    for k in vars(process)
                    if not k.startswith("_")
                )
            ),
        )
        pickled = pickle.dumps(data)
        return hashlib.sha256(pickled).hexdigest()


class AnalysisResult:
    """Container for analysis results, exposing process outcomes as attributes."""

    def __init__(self, results: dict[str, Any]) -> None:
        self._results = results

    def __getattr__(self, name: str) -> Any:
        if name in self._results:
            return self._results[name]
        raise AttributeError(f"No result for process '{name}'")
