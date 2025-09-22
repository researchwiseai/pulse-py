"""High-level async orchestrator for running processes."""

from typing import Sequence, Optional, Union, Any
import pandas as pd

from pulse.core.async_client import AsyncCoreClient
from pulse.auth import _BaseOAuth2Auth
from pulse.analysis.processes import Process
from pulse.analysis.results import (
    ThemeGenerationResult,
    SentimentResult,
    ThemeAllocationResult,
    ClusterResult,
    ThemeExtractionResult,
)


class AsyncAnalyzer:
    """High-level async orchestrator for Pulse API processes with caching."""

    def __init__(
        self,
        dataset: Union[Sequence[str], pd.Series],
        processes: Optional[Sequence[Process]] = None,
        *,
        fast: Optional[bool] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        client: Optional[AsyncCoreClient] = None,
        auth: Optional[_BaseOAuth2Auth] = None,
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
        self.client = client or AsyncCoreClient(auth=auth)
        self._client_owned = client is None  # Track if we own the client
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

    async def run(self) -> "AsyncAnalysisResult":
        """Run the configured processes asynchronously with caching and wrapping."""
        results: dict[str, Any] = {}
        texts = self.dataset.tolist()
        for process in self.processes:
            key = self._make_cache_key(process) if self._cache is not None else None
            if self.use_cache and self._cache is not None and key in self._cache:
                from pulse.debug import log_cache_hit

                log_cache_hit(key)
                wrapped = self._cache[key]
            else:
                from pulse.debug import log_cache_miss

                if key:
                    log_cache_miss(key)
                raw = await self._run_process_async(process)
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
        return AsyncAnalysisResult(results)

    async def _run_process_async(self, process: Process) -> Any:
        """Run a single process asynchronously by adapting sync to async client."""
        # Create an async context that mimics the sync context but uses async client
        async_ctx = AsyncProcessContext(
            dataset=self.dataset,
            fast=self.fast,
            client=self.client,
            results=self.results,
        )

        # Adapt the sync process.run() method to work with async client
        return await self._adapt_process_to_async(process, async_ctx)

    async def _adapt_process_to_async(
        self, process: Process, ctx: "AsyncProcessContext"
    ) -> Any:
        """Adapt sync process methods to use async client methods."""
        process_id = process.id

        if process_id == "theme_generation":
            return await self._run_theme_generation_async(process, ctx)
        elif process_id == "sentiment":
            return await self._run_sentiment_async(process, ctx)
        elif process_id == "theme_allocation":
            return await self._run_theme_allocation_async(process, ctx)
        elif process_id == "similarity":
            return await self._run_similarity_async(process, ctx)
        elif process_id == "theme_extraction":
            return await self._run_theme_extraction_async(process, ctx)
        elif process_id == "cluster":
            return await self._run_cluster_async(process, ctx)
        else:
            # Fallback: try to run the sync process with async client
            # This may not work for all processes but provides a fallback
            return process.run(ctx)

    async def _run_theme_generation_async(
        self, process: Any, ctx: "AsyncProcessContext"
    ) -> Any:
        """Run theme generation process asynchronously."""
        import random

        texts = ctx.dataset.tolist()
        fast_flag = process.fast if process.fast is not None else ctx.fast

        # sample randomly according to fast flag
        sample_size = 200 if fast_flag else 1000
        if len(texts) > sample_size:
            texts = random.sample(
                texts, sample_size
            )  # nosec B311 - Used for data sampling, not cryptographic purposes

        return await ctx.client.generate_themes(
            texts,
            min_themes=process.min_themes,
            max_themes=process.max_themes,
            fast=process.fast or ctx.fast,
            context=process.context,
            version=process.version,
            prune=process.prune,
            interactive=process.interactive,
            initial_sets=process.initial_sets,
            await_job_result=process.await_job_result,
        )

    async def _run_sentiment_async(
        self, process: Any, ctx: "AsyncProcessContext"
    ) -> Any:
        """Run sentiment analysis process asynchronously."""
        texts = ctx.dataset.tolist()
        return await ctx.client.analyze_sentiment(texts, fast=process.fast or ctx.fast)

    async def _run_theme_allocation_async(
        self, process: Any, ctx: "AsyncProcessContext"
    ) -> dict[str, Any]:
        """Run theme allocation process asynchronously."""
        from pulse.core.models import Theme as ThemeModel

        texts = list(ctx.dataset)
        # Determine raw themes list (static strings or ThemeModel instances)
        if process.themes is not None:
            raw_themes = list(process.themes)
        else:
            alias = getattr(process, "_themes_from_alias", "theme_generation")
            tg = ctx.results.get(alias)
            if tg is not None:
                raw_themes = list(tg.themes)
            else:
                src = getattr(ctx, "sources", {})
                if alias in src:
                    raw_themes = list(src[alias])
                else:
                    raise RuntimeError(f"{alias} result not available for allocation")
        # Prepare labels for output and texts for similarity input
        if raw_themes and isinstance(raw_themes[0], ThemeModel):
            labels = [t.shortLabel for t in raw_themes]
            sim_texts = [" ".join(t.representatives) for t in raw_themes]
        else:
            labels = list(raw_themes)
            sim_texts = list(raw_themes)
        fast_flag = process.fast if process.fast is not None else ctx.fast

        resp = await ctx.client.compare_similarity(
            set_a=texts,
            set_b=sim_texts,
            fast=fast_flag,
            flatten=False,
        )
        # normalize similarity matrix from response or raw matrix
        similarity = getattr(resp, "similarity", resp)

        # If single_label=True, then assign each input to its most similar theme
        # as long as it is over the threshold. If single_label=False, then we
        # assign it to all themes that it has a similarity score over the
        # threshold.

        # compute raw assignments: best matching theme index for each text
        assignments: list[int]
        if similarity is not None:
            assignments = []
            for sim_row in similarity:
                # find index of maximum similarity
                best_idx = max(range(len(sim_row)), key=lambda i: sim_row[i])
                assignments.append(best_idx)
        else:
            raise RuntimeError("No similarity matrix available for allocation")
        return {
            "themes": labels,
            "assignments": assignments,
            "similarity": similarity,
        }

    async def _run_similarity_async(
        self, process: Any, ctx: "AsyncProcessContext"
    ) -> Any:
        """Run similarity process asynchronously."""
        texts = list(ctx.dataset)

        # Use provided sets or default to dataset
        set_a = process.set_a or texts
        set_b = process.set_b

        if set_b is None:
            # Self-similarity
            return await ctx.client.compare_similarity(
                set=set_a,
                split=process.split,
                flatten=process.flatten,
                version=process.version,
                fast=process.fast or ctx.fast,
                await_job_result=process.await_job_result,
            )
        else:
            # Cross-similarity
            return await ctx.client.compare_similarity(
                set_a=set_a,
                set_b=set_b,
                split=process.split,
                flatten=process.flatten,
                version=process.version,
                fast=process.fast or ctx.fast,
                await_job_result=process.await_job_result,
            )

    async def _run_theme_extraction_async(
        self, process: Any, ctx: "AsyncProcessContext"
    ) -> Any:
        """Run theme extraction process asynchronously."""
        texts = list(ctx.dataset)

        # Determine dictionary - use provided dictionary or themes
        if process.dictionary is not None:
            used_dictionary = list(process.dictionary)
        elif process.themes is not None:
            used_dictionary = list(process.themes)
        else:
            # Get themes from previous process
            alias = getattr(process, "_themes_from_alias", "theme_generation")
            prev = ctx.results.get(alias)
            if prev is not None:
                used_dictionary = prev.themes
            else:
                # fallback to named source
                src = getattr(ctx, "sources", {})
                if alias in src:
                    used_dictionary = list(src[alias])
                else:
                    raise RuntimeError(f"{alias} result not available for extraction")

        return await ctx.client.extract_elements(
            inputs=texts,
            dictionary=used_dictionary,
            type=process.type,
            expand_dictionary=process.expand_dictionary,
            expand_dictionary_limit=process.expand_dictionary_limit,
            version=process.version,
            fast=process.fast or ctx.fast,
            await_job_result=process.await_job_result,
            # Pass deprecated parameters for backward compatibility
            use_ner=process.use_ner,
            use_llm=process.use_llm,
            threshold=process.threshold,
        )

    async def _run_cluster_async(self, process: Any, ctx: "AsyncProcessContext") -> Any:
        """Run clustering process asynchronously."""
        texts = list(ctx.dataset)
        return await ctx.client.cluster_texts(
            inputs=texts,
            k=process.k,
            algorithm=process.algorithm,
            fast=process.fast or ctx.fast,
            await_job_result=process.await_job_result,
        )

    def clear_cache(self) -> None:
        """Clear the on-disk cache, if enabled."""
        if self._cache is not None:
            self._cache.clear()

    async def close(self) -> None:
        """Close underlying HTTP client and persistent cache."""
        try:
            if self._client_owned:
                await self.client.close()
        except Exception as e:
            # Log but don't raise - cleanup should be best effort
            import logging

            logging.getLogger(__name__).debug(f"Error closing client: {e}")
        if self._cache:
            try:
                self._cache.close()
            except Exception as e:
                # Log but don't raise - cleanup should be best effort
                import logging

                logging.getLogger(__name__).debug(f"Error closing cache: {e}")

    async def __aenter__(self) -> "AsyncAnalyzer":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with proper resource cleanup."""
        await self.close()

    def _make_cache_key(self, process: Process) -> str:
        # nosec B403 - Used for cache key generation, not deserializing untrusted data
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


class AsyncProcessContext:
    """Context object for async process execution."""

    def __init__(
        self,
        dataset: pd.Series,
        fast: bool,
        client: AsyncCoreClient,
        results: dict[str, Any],
    ) -> None:
        self.dataset = dataset
        self.fast = fast
        self.client = client
        self.results = results


class AsyncAnalysisResult:
    """Container for async analysis results, exposing process outcomes as attributes."""

    def __init__(self, results: dict[str, Any]) -> None:
        self._results = results

    def __getattr__(self, name: str) -> Any:
        if name in self._results:
            return self._results[name]
        raise AttributeError(f"No result for process '{name}'")
