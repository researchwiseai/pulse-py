"""Async DSL builder for custom workflows in the Pulse client."""

from collections import defaultdict
import os
import json
from typing import Any, Callable, Dict, List, Awaitable

import pandas as pd
from pulse.analysis.processes import (
    ThemeGeneration,
    ThemeAllocation,
    ThemeExtraction,
    SentimentProcess,
    SimilarityProcess,
    Cluster,
)
from pulse.analysis.async_analyzer import AsyncAnalyzer
from pulse.core.async_client import AsyncCoreClient
from pulse.core.models import SentimentResponse as CoreSentimentResponse


# Helpers to flatten and reconstruct nested inputs (reused from sync DSL)
def _flatten_and_shape(x: Any):
    shape: List[int] = []

    def _get_shape(a: Any, lvl: int = 0):
        nonlocal shape
        if isinstance(a, list):
            if len(shape) <= lvl:
                shape.append(len(a))
            else:
                shape[lvl] = max(shape[lvl], len(a))
            if a:
                _get_shape(a[0], lvl + 1)

    def _flatten(a: Any) -> List[Any]:
        if isinstance(a, list):
            out: List[Any] = []
            for v in a:
                out.extend(_flatten(v))
            return out
        return [a]

    _get_shape(x)
    flat = _flatten(x)
    return shape, flat


def _reconstruct(flat: List[Any], shape: List[int]):
    it = iter(flat)

    def _build(level: int):
        if level >= len(shape):
            return next(it)
        return [_build(level + 1) for _ in range(shape[level])]

    return _build(0)


class AsyncWorkflow:
    """
    Async workflow builder for composing sequences of Processes.

    Supports method chaining and provides a simple DAG representation with
    async execution.
    """

    def __init__(self) -> None:
        # Registered named data sources for DSL (alias -> data)
        self._sources: Dict[str, Any] = {}
        # Internal list of process nodes
        self._processes: List[Any] = []
        # Counters for aliasing duplicate process IDs
        self._id_counts: Dict[str, int] = defaultdict(int)
        # Client for async operations
        self._client: AsyncCoreClient | None = None
        self._client_owned: bool = False

    def source(self, name: str, data: Any) -> "AsyncWorkflow":
        """
        Register a named data source for subsequent steps.

        e.g. wf.source('comments', comments_list)
        """
        if name in self._sources:
            raise ValueError(f"Source '{name}' already registered")
        self._sources[name] = data
        return self

    def _add_process(self, process: Any, name: str | None = None) -> None:
        orig_id = process.id
        # increment counter for this process type
        count = self._id_counts.get(orig_id, 0) + 1
        self._id_counts[orig_id] = count
        # preserve original process id for result wrapping
        setattr(process, "_orig_id", orig_id)
        if name:
            # user-specified alias: must be unique among sources and processes
            if name in self._sources or name in [p.id for p in self._processes]:
                raise ValueError(f"Process name '{name}' already registered")
            setattr(process, "id", name)
        elif count > 1:
            # auto-aliased numbered id (e.g. sentiment_2)
            alias = f"{orig_id}_{count}"
            setattr(process, "id", alias)
        # first occurrence retains original id
        self._processes.append(process)

    def theme_generation(
        self,
        *,
        min_themes: int = 2,
        max_themes: int = 10,
        context: Any = None,
        version: str | None = None,
        prune: int | None = None,
        interactive: bool | None = None,
        initial_sets: int | None = None,
        fast: bool | None = None,
        source: str | None = None,
        name: str | None = None,
    ) -> "AsyncWorkflow":
        """Add a theme generation step to the workflow."""
        process = ThemeGeneration(
            min_themes=min_themes,
            max_themes=max_themes,
            context=context,
            version=version,
            prune=prune,
            interactive=interactive,
            initial_sets=initial_sets,
            fast=fast,
        )
        self._add_process(process, name=name)
        # determine input source for texts
        alias = source or "dataset"
        # allow text source from named sources or prior process outputs
        if (
            alias != "dataset"
            and alias not in self._sources
            and alias not in [p.id for p in self._processes]
        ):
            raise ValueError(f"Unknown source for theme_generation: '{alias}'")
        setattr(process, "_inputs", [alias])
        return self

    def theme_allocation(
        self,
        *,
        themes: list[str] | None = None,
        fast: bool | None = None,
        single_label: bool = True,
        threshold: float = 0.5,
        inputs: str | None = None,
        themes_from: str | None = None,
        name: str | None = None,
    ) -> "AsyncWorkflow":
        """Add a theme allocation step with explicit input wiring."""
        # auto-inject theme_generation for dynamic themes if not already present
        text_alias = inputs or "dataset"
        if themes is None and themes_from is None:
            # validate text source alias
            if (
                text_alias != "dataset"
                and text_alias not in self._sources
                and text_alias not in [p.id for p in self._processes]
            ):
                raise ValueError(
                    f"Unknown inputs source for theme_allocation: '{text_alias}'"
                )
            # inject default theme_generation on the same texts
            if not any(
                getattr(p, "_orig_id", p.id) == "theme_generation"
                for p in self._processes
            ):
                self.theme_generation(source=text_alias)
        process = ThemeAllocation(
            themes=themes,
            single_label=single_label,
            threshold=threshold,
            fast=fast,
        )
        self._add_process(process, name=name)
        # wire text inputs
        inp = text_alias
        if (
            inp != "dataset"
            and inp not in self._sources
            and inp not in [p.id for p in self._processes]
        ):
            raise ValueError(f"Unknown inputs source for theme_allocation: '{inp}'")
        setattr(process, "_inputs", [inp])
        # wire themes list if dynamic
        if themes is None:
            if themes_from:
                alias = themes_from
                if alias not in self._sources and alias not in [
                    p.id for p in self._processes
                ]:
                    raise ValueError(
                        f"Unknown themes source for theme_allocation: '{alias}'"
                    )
            else:
                # find last theme_generation alias
                alias = next(
                    (
                        p.id
                        for p in reversed(self._processes[:-1])
                        if getattr(p, "_orig_id", p.id) == "theme_generation"
                    ),
                    None,
                )
            if not alias:
                raise ValueError("No theme_generation found for theme_allocation")
            setattr(process, "_themes_from_alias", alias)
        return self

    def theme_extraction(
        self,
        *,
        themes: list[str] | None = None,
        dictionary: list[str] | None = None,
        type: str = "named-entities",
        expand_dictionary: bool = False,
        expand_dictionary_limit: int | None = None,
        version: str | None = None,
        fast: bool | None = None,
        inputs: str | None = None,
        themes_from: str | None = None,
        name: str | None = None,
    ) -> "AsyncWorkflow":
        """Add a theme extraction step with type control and explicit input wiring."""
        process = ThemeExtraction(
            themes=themes,
            dictionary=dictionary,
            type=type,
            expand_dictionary=expand_dictionary,
            expand_dictionary_limit=expand_dictionary_limit,
            version=version,
            fast=fast,
        )
        self._add_process(process, name=name)
        # wire text inputs
        inp = inputs or "dataset"
        if (
            inp != "dataset"
            and inp not in self._sources
            and inp not in [p.id for p in self._processes]
        ):
            raise ValueError(f"Unknown inputs source for theme_extraction: '{inp}'")
        setattr(process, "_inputs", [inp])
        # wire themes list if dynamic
        if themes is None:
            if themes_from:
                alias = themes_from
                if alias not in self._sources and alias not in [
                    p.id for p in self._processes
                ]:
                    raise ValueError(
                        f"Unknown themes source for theme_extraction: '{alias}'"
                    )
            else:
                alias = next(
                    (
                        p.id
                        for p in reversed(self._processes[:-1])
                        if getattr(p, "_orig_id", p.id) == "theme_generation"
                    ),
                    None,
                )
            if not alias:
                raise ValueError("No theme_generation found for theme_extraction")
            setattr(process, "_themes_from_alias", alias)
        return self

    def sentiment(
        self,
        *,
        fast: bool | None = None,
        source: str | None = None,
        name: str | None = None,
    ) -> "AsyncWorkflow":
        """Add a sentiment analysis step with optional source override."""
        process = SentimentProcess(fast=fast)
        self._add_process(process, name=name)
        # determine input source
        alias = source or "dataset"
        if (
            alias != "dataset"
            and alias not in self._sources
            and alias not in [p.id for p in self._processes]
        ):
            raise ValueError(f"Unknown source for sentiment: '{alias}'")
        setattr(process, "_inputs", [alias])
        return self

    def similarity(
        self,
        *,
        set_a: list[str] | None = None,
        set_b: list[str] | None = None,
        split: dict[str, Any] | None = None,
        flatten: bool = False,
        version: str | None = None,
        source: str | None = None,
        fast: bool | None = None,
        name: str | None = None,
    ) -> "AsyncWorkflow":
        """Add a similarity computation step with text splitting support."""
        process = SimilarityProcess(
            set_a=set_a,
            set_b=set_b,
            split=split,
            flatten=flatten,
            version=version,
            fast=fast,
        )
        self._add_process(process, name=name)
        # determine input source
        alias = source or "dataset"
        if (
            alias != "dataset"
            and alias not in self._sources
            and alias not in [p.id for p in self._processes]
        ):
            raise ValueError(f"Unknown source for similarity: '{alias}'")
        setattr(process, "_inputs", [alias])
        return self

    def cluster(
        self,
        *,
        k: int = 2,
        algorithm: str = "kmeans",
        source: str | None = None,
        fast: bool | None = None,
        name: str | None = None,
    ) -> "AsyncWorkflow":
        """Add a clustering step with algorithm selection."""
        process = Cluster(k=k, algorithm=algorithm, fast=fast)
        self._add_process(process, name=name)
        # determine input source for clustering
        alias = source or "dataset"
        if (
            alias != "dataset"
            and alias not in self._sources
            and alias not in [p.id for p in self._processes]
        ):
            raise ValueError(f"Unknown source for cluster: '{alias}'")
        setattr(process, "_inputs", [alias])
        return self

    def monitor(
        self,
        on_run_start: Callable[[], Awaitable[None]] | None = None,
        on_process_start: Callable[[str], Awaitable[None]] | None = None,
        on_process_end: Callable[[str, Any], Awaitable[None]] | None = None,
        on_run_end: Callable[[], Awaitable[None]] | None = None,
    ) -> "AsyncWorkflow":
        """
        Register async lifecycle callbacks for observability:
          • on_run_start(): called once before any processes run
          • on_process_start(process_id): called before each process
          • on_process_end(process_id, result): called after each process
          • on_run_end(): called once after all processes have run
        """
        self._monitors = {
            "on_run_start": on_run_start,
            "on_process_start": on_process_start,
            "on_process_end": on_process_end,
            "on_run_end": on_run_end,
        }
        return self

    @classmethod
    def from_file(cls, file_path: str) -> "AsyncWorkflow":
        """
        Load workflow definition from a JSON or YAML file.

        The file must define a top-level 'pipeline' list of single-key mappings.
        """
        wf = cls()
        ext = os.path.splitext(file_path)[1].lower()
        with open(file_path, "r") as f:
            if ext in (".yml", ".yaml"):
                try:
                    import yaml

                    config = yaml.safe_load(f)
                except ImportError as e:
                    raise ImportError("PyYAML is required to parse YAML files") from e
            elif ext == ".json":
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config type: {file_path}")
        pipeline = config.get("pipeline", [])
        for step in pipeline:
            if not isinstance(step, dict) or len(step) != 1:
                raise ValueError(f"Invalid pipeline step: {step}")
            name, params = next(iter(step.items()))
            if not hasattr(wf, name):
                raise ValueError(f"Unknown pipeline step: {name}")
            if params is None:
                params = {}
            getattr(wf, name)(**params)
        return wf

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the workflow asynchronously.
        If any named sources were registered via .source(), runs in DSL mode.
        Otherwise, delegates to the existing AsyncAnalyzer engine.
        """
        # Extract client and fast flag for DSL
        client = kwargs.get("client", None)
        fast = kwargs.get("fast", None)
        # Dataset positional argument
        dataset = args[0] if args else None
        # DSL mode if any sources registered
        if self._sources:
            # Register default dataset source if provided
            if dataset is not None and "dataset" not in self._sources:
                self._sources["dataset"] = dataset
            return await self._run_dsl_async(client=client, fast=fast)
        # Linear mode: use AsyncAnalyzer
        async with AsyncAnalyzer(
            dataset=dataset, processes=self._processes, **kwargs
        ) as analyzer:
            return await analyzer.run()

    async def _run_dsl_async(
        self, client: AsyncCoreClient | None = None, fast: bool | None = None
    ) -> Any:
        """
        Internal runner for advanced DSL mode with named sources and DAG execution.
        """
        # Lazy import to avoid circular dependencies
        from pulse.analysis.results import (
            ThemeGenerationResult,
            SentimentResult,
            ThemeAllocationResult,
            ClusterResult,
            ThemeExtractionResult,
        )

        # Default client
        if client is None:
            self._client = AsyncCoreClient()
            self._client_owned = True
        else:
            self._client = client
            self._client_owned = False

        try:
            # Initialize context streams
            sources: Dict[str, Any] = dict(self._sources)
            # Results mapping for wrapper objects
            results: Dict[str, Any] = {}
            # Lifecycle: on_run_start callback
            on_run_start = getattr(self, "_monitors", {}).get("on_run_start")
            if on_run_start:
                await on_run_start()
            # Execute processes in declaration order
            for process in self._processes:
                # Lifecycle: on_process_start callback
                on_process_start = getattr(self, "_monitors", {}).get(
                    "on_process_start"
                )
                if on_process_start:
                    await on_process_start(process.id)
                # Validate and get dataset input
                inputs = getattr(process, "_inputs", ["dataset"])
                if not inputs:
                    raise RuntimeError(f"No input source for process '{process.id}'")
                ds_alias = inputs[0]
                if ds_alias not in sources:
                    raise ValueError(
                        f"Source '{ds_alias}' not found for process '{process.id}'"
                    )
                ds_data = sources[ds_alias]

                # Build async context
                class AsyncCtx:
                    pass

                ctx = AsyncCtx()
                ctx.client = self._client
                # fast flag per process, fallback to DSL-level
                ctx.fast = (
                    process.fast
                    if getattr(process, "fast", None) is not None
                    else (fast if fast is not None else False)
                )
                # Dataset as pandas Series
                if isinstance(ds_data, pd.Series):
                    ctx.dataset = ds_data
                else:
                    ctx.dataset = pd.Series(ds_data)
                ctx.results = results
                # expose named and generated sources to processes
                ctx.sources = sources
                # Run and wrap result
                raw = await self._run_process_async(process, ctx)
                orig = getattr(process, "_orig_id", process.id)
                if orig == "theme_generation":
                    wrapped = ThemeGenerationResult(raw, ctx.dataset.tolist())
                    # make themes available as data source
                    sources[process.id] = wrapped.themes
                elif orig == "sentiment":
                    # Support nested input: flatten, call, reconstruct
                    data_in = ds_data
                    try:
                        shape, flat_texts = _flatten_and_shape(data_in)
                        # call sentiment on flat list
                        ctx.dataset = pd.Series(flat_texts)
                        flat_raw = await self._run_process_async(process, ctx)
                        flat_sents = flat_raw.sentiments
                        nested = _reconstruct(flat_sents, shape)
                        # wrap nested sentiments
                        raw2 = CoreSentimentResponse(sentiments=nested)
                        wrapped = SentimentResult(raw2, flat_texts)
                        sources[process.id] = nested
                    except Exception:
                        # fallback to default behavior
                        wrapped = SentimentResult(raw, ctx.dataset.tolist())
                        sources[process.id] = wrapped.sentiments
                elif orig == "theme_allocation":
                    wrapped = ThemeAllocationResult(
                        ctx.dataset.tolist(),
                        raw["themes"],
                        raw["assignments"],
                        process.single_label,
                        process.threshold,
                        similarity=raw.get("similarity"),
                    )
                elif orig == "similarity":
                    # For similarity, just store the raw response
                    wrapped = raw
                    sources[process.id] = raw
                elif orig == "cluster":
                    wrapped = ClusterResult(raw, ctx.dataset.tolist())
                elif orig == "theme_extraction":
                    wrapped = ThemeExtractionResult(
                        raw, ctx.dataset.tolist(), process.themes
                    )
                    # make extracted elements available as data source
                    sources[process.id] = wrapped.extractions
                else:
                    wrapped = raw
                # Store for downstream
                results[process.id] = wrapped
                # Lifecycle: on_process_end callback
                on_process_end = getattr(self, "_monitors", {}).get("on_process_end")
                if on_process_end:
                    await on_process_end(process.id, wrapped)
            # Lifecycle: on_run_end callback
            on_run_end = getattr(self, "_monitors", {}).get("on_run_end")
            if on_run_end:
                await on_run_end()
            # Return a results container
            return type("AsyncDSLResult", (), results)()
        finally:
            # Clean up client if we own it
            if self._client_owned and self._client:
                await self._client.close()

    async def _run_process_async(self, process: Any, ctx: Any) -> Any:
        """Run a single process asynchronously."""
        process_id = getattr(process, "_orig_id", process.id)

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
            raise RuntimeError(f"Unknown async process type: {process_id}")

    async def _run_theme_generation_async(self, process: Any, ctx: Any) -> Any:
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
            await_job_result=getattr(process, "await_job_result", True),
        )

    async def _run_sentiment_async(self, process: Any, ctx: Any) -> Any:
        """Run sentiment analysis process asynchronously."""
        texts = ctx.dataset.tolist()
        return await ctx.client.analyze_sentiment(texts, fast=process.fast or ctx.fast)

    async def _run_theme_allocation_async(
        self, process: Any, ctx: Any
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

    async def _run_similarity_async(self, process: Any, ctx: Any) -> Any:
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
                await_job_result=getattr(process, "await_job_result", True),
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
                await_job_result=getattr(process, "await_job_result", True),
            )

    async def _run_theme_extraction_async(self, process: Any, ctx: Any) -> Any:
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
            await_job_result=getattr(process, "await_job_result", True),
        )

    async def _run_cluster_async(self, process: Any, ctx: Any) -> Any:
        """Run clustering process asynchronously."""
        texts = list(ctx.dataset)
        return await ctx.client.cluster_texts(
            inputs=texts,
            k=process.k,
            algorithm=process.algorithm,
            fast=process.fast or ctx.fast,
            await_job_result=getattr(process, "await_job_result", True),
        )

    def graph(self) -> Dict[str, List[str]]:
        """
        Return a simple adjacency list representing the workflow DAG.
        """
        edges: Dict[str, List[str]] = {}
        id_to_aliases: Dict[str, List[str]] = defaultdict(list)
        for p in self._processes:
            orig = getattr(p, "_orig_id", p.id)
            id_to_aliases[orig].append(p.id)
        # Build adjacency: include both declared depends_on and wired inputs
        proc_ids = [p.id for p in self._processes]
        for p in self._processes:
            alias = p.id
            # collect static dependencies based on orig_id.depends_on
            deps: List[str] = []
            for dep in getattr(p, "depends_on", ()):  # type: ignore[attr-defined]
                deps.extend(id_to_aliases.get(dep, []))
            # collect dynamic inputs from DSL wiring (skip 'dataset')
            for inp in getattr(p, "_inputs", []):
                if inp != "dataset" and inp in proc_ids:
                    deps.append(inp)
            # collect theme-source wiring
            theme_src = getattr(p, "_themes_from_alias", None)
            if theme_src and theme_src in proc_ids:
                deps.append(theme_src)
            # remove duplicates preserving order
            seen = set()
            cleaned: List[str] = []
            for d in deps:
                if d not in seen:
                    seen.add(d)
                    cleaned.append(d)
            edges[alias] = cleaned
        return edges

    async def close(self) -> None:
        """Close underlying HTTP client if owned by this workflow."""
        if self._client_owned and self._client:
            await self._client.close()

    async def __aenter__(self) -> "AsyncWorkflow":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with proper resource cleanup."""
        await self.close()
