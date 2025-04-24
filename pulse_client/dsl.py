"""DSL builder for custom workflows in the Pulse client."""
from collections import defaultdict
import json
import os
import warnings
from typing import Any, Dict, List, Sequence, Union

from pulse_client.analysis.processes import (
    ThemeGeneration,
    ThemeAllocation,
    ThemeExtraction,
    SentimentProcess,
    Cluster,
)
from pulse_client.analysis.analyzer import Analyzer


class Workflow:
    """
    Workflow builder for composing sequences of Processes.

    Supports method chaining and provides a simple DAG representation.
    """

    def __init__(self) -> None:
        self._processes: List[Any] = []
        self._id_counts: Dict[str, int] = defaultdict(int)

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
    ) -> "Workflow":
        """Add a theme generation step to the workflow."""
        process = ThemeGeneration(
            min_themes=min_themes,
            max_themes=max_themes,
            context=context,
            fast=fast,
        )
        self._add_process(process)
        return self

    def theme_allocation(
        self,
        *,
        themes: list[str] | None = None,
        single_label: bool = True,
        threshold: float = 0.5,
    ) -> "Workflow":
        """Add a theme allocation step."""
        process = ThemeAllocation(
            themes=themes,
            single_label=single_label,
            threshold=threshold,
        )
        self._add_process(process)
        return self

    def theme_extraction(
        self,
        *,
        themes: list[str] | None = None,
        version: str | None = None,
        fast: bool | None = None,
    ) -> "Workflow":
        """Add a theme extraction step."""
        process = ThemeExtraction(
            themes=themes,
            version=version,
            fast=fast,
        )
        self._add_process(process)
        return self

    def sentiment(
        self,
        *,
        fast: bool | None = None,
        source: str | None = None,
    ) -> "Workflow":
        """Add a sentiment analysis step."""
        if source not in (None, "dataset"):
            warnings.warn("DSL v1 does not support source override; ignoring 'source'")
        process = SentimentProcess(fast=fast)
        self._add_process(process)
        return self

    def cluster(
        self,
        *,
        k: int = 2,
        source: str | None = None,
        fast: bool | None = None,
    ) -> "Workflow":
        """Add a clustering step (k parameter not used by underlying API)."""
        if source is not None:
            warnings.warn("DSL v1 does not support source override; ignoring 'source'")
        process = Cluster(fast=fast)
        self._add_process(process)
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

    def run(
        self,
        dataset: Union[Sequence[str], Any],
        **kwargs: Any,
    ) -> Any:
        """
        Execute the workflow on the given dataset.

        Delegates execution to the Analyzer engine.
        """
        analyzer = Analyzer(dataset=dataset, processes=self._processes, **kwargs)
        return analyzer.run()

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