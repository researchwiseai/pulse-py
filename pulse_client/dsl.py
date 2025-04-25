"""DSL builder for custom workflows in the Pulse client."""

from collections import defaultdict
import json
import os
import warnings
from typing import Any, Dict, List, Sequence, Union

import pandas as pd
from pulse_client.analysis.processes import (
    ThemeGeneration,
    ThemeAllocation,
    ThemeExtraction,
    SentimentProcess,
    Cluster,
)
from pulse_client.analysis.analyzer import Analyzer
from pulse_client.analysis.results import (
    ThemeGenerationResult,
    ThemeAllocationResult,
    ThemeExtractionResult,
    SentimentResult,
    ClusterResult,
)
from pulse_client.core.client import CoreClient


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

    def _add_process(self, process: Any) -> None:
        orig_id = process.id
        count = self._id_counts[orig_id] + 1
        self._id_counts[orig_id] = count
        setattr(process, "_orig_id", orig_id)
        if count > 1:
            alias = f"{orig_id}_{count}"
            setattr(process, "id", alias)
        self._processes.append(process)

    def theme_generation(
        self,
        *,
        min_themes: int = 2,
        max_themes: int = 10,
        context: Any = None,
        fast: bool | None = None,
        source: str | None = None,
    ) -> "Workflow":
        """Add a theme generation step to the workflow."""
        process = ThemeGeneration(
            min_themes=min_themes,
            max_themes=max_themes,
            context=context,
            fast=fast,
        )
        self._add_process(process)
        # determine input source for texts
        alias = source or "dataset"
        if alias != "dataset" and alias not in self._sources:
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
    ) -> "Workflow":
        """Add a theme allocation step with explicit input wiring."""
        process = ThemeAllocation(
            themes=themes,
            single_label=single_label,
            threshold=threshold,
        )
        self._add_process(process)
        # wire text inputs
        inp = inputs or "dataset"
        if (
            inp != "dataset"
            and inp not in self._sources
            and inp not in [p.id for p in self._processes]
        ):
            raise ValueError(f"Unknown inputs source for theme_allocation: '{inp}'")
        setattr(process, "_inputs", [inp])
        # wire themes list if dynamic
        if themes is None:
            # determine alias for theme source
            if themes_from:
                alias = themes_from
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
    ) -> "Workflow":
        """Add a theme extraction step with explicit input wiring."""
        process = ThemeExtraction(
            themes=themes,
            version=version,
            fast=fast,
        )
        self._add_process(process)
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
    ) -> "Workflow":
        """Add a sentiment analysis step with optional source override."""
        process = SentimentProcess(fast=fast)
        self._add_process(process)
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
    ) -> "Workflow":
        """Add a clustering step with optional source override."""
        process = Cluster(fast=fast)
        self._add_process(process)
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
        from pulse_client.analysis.results import (
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
            # Run and wrap result
            raw = process.run(ctx)
            orig = getattr(process, "_orig_id", process.id)
            if orig == "theme_generation":
                wrapped = ThemeGenerationResult(raw, ctx.dataset.tolist())
                # make themes available as data source
                sources[process.id] = wrapped.themes
            elif orig == "sentiment":
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
        for p in self._processes:
            alias = p.id
            orig = getattr(p, "_orig_id", p.id)
            deps: List[str] = []
            for dep in getattr(p, "depends_on", ()):  # type: ignore[attr-defined]
                deps.extend(id_to_aliases.get(dep, []))
            edges[alias] = deps
        return edges
