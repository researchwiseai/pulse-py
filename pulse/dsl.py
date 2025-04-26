"""DSL builder for custom workflows in the Pulse client."""

from collections import defaultdict
import os
import json
from typing import Any, Dict, List

import pandas as pd
from pulse.analysis.processes import (
    ThemeGeneration,
    ThemeAllocation,
    ThemeExtraction,
    SentimentProcess,
    Cluster,
)
from pulse.analysis.analyzer import Analyzer
from pulse.core.client import CoreClient
from pulse.core.models import SentimentResponse as CoreSentimentResponse


# Helpers to flatten and reconstruct nested inputs
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


class Workflow:
    """
    Workflow builder for composing sequences of Processes.

    Supports method chaining and provides a simple DAG representation.
    """

    def __init__(self) -> None:
        # Registered named data sources for DSL (alias -> data)
        self._sources: Dict[str, Any] = {}
        # Internal list of process nodes
        self._processes: List[Any] = []
        # Counters for aliasing duplicate process IDs
        self._id_counts: Dict[str, int] = defaultdict(int)

    def source(self, name: str, data: Any) -> "Workflow":
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
        fast: bool | None = None,
        source: str | None = None,
        name: str | None = None,
    ) -> "Workflow":
        """Add a theme generation step to the workflow."""
        process = ThemeGeneration(
            min_themes=min_themes,
            max_themes=max_themes,
            context=context,
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
        single_label: bool = True,
        threshold: float = 0.5,
        inputs: str | None = None,
        themes_from: str | None = None,
        name: str | None = None,
    ) -> "Workflow":
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
        version: str | None = None,
        fast: bool | None = None,
        inputs: str | None = None,
        themes_from: str | None = None,
        name: str | None = None,
    ) -> "Workflow":
        """Add a theme extraction step with explicit input wiring."""
        process = ThemeExtraction(
            themes=themes,
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
    ) -> "Workflow":
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

    def cluster(
        self,
        *,
        k: int = 2,
        source: str | None = None,
        fast: bool | None = None,
        name: str | None = None,
    ) -> "Workflow":
        """Add a clustering step with optional source override."""
        process = Cluster(fast=fast)
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

    @classmethod
    def from_file(cls, file_path: str) -> "Workflow":
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

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the workflow.
        If any named sources were registered via .source(), runs in DSL mode.
        Otherwise, delegates to the existing Analyzer engine.
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
            return self._run_dsl(client=client, fast=fast)
        # Linear mode: use Analyzer
        return Analyzer(dataset=dataset, processes=self._processes, **kwargs).run()

    def _run_dsl(
        self, client: CoreClient | None = None, fast: bool | None = None
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
        client = client or CoreClient()
        # Initialize context streams
        sources: Dict[str, Any] = dict(self._sources)
        # Results mapping for wrapper objects
        results: Dict[str, Any] = {}
        # Execute processes in declaration order
        for process in self._processes:
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

            # Build context
            class Ctx:
                pass

            ctx = Ctx()
            ctx.client = client
            # fast flag per process, fallback to DSL-level
            ctx.fast = (
                process.fast
                if getattr(process, "fast", None) is not None
                else (fast if fast is not None else True)
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
            raw = process.run(ctx)
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
                    flat_raw = process.run(ctx)
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
        # Return a results container
        return type("DSLResult", (), results)()

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
