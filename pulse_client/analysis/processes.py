"""Built-in Process primitives for Analyzer."""

from typing import Any, Tuple

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


class Process(Protocol):
    """Process primitive protocol."""

    id: str
    depends_on: Tuple[str, ...]

    def run(self, ctx: Any) -> Any:
        ...


class ThemeGeneration:
    """Process: cluster texts into latent themes."""

    id = "theme_generation"
    depends_on: Tuple[str, ...] = ()

    def __init__(
        self,
        min_themes: int = 2,
        max_themes: int = 10,
        context: Any = None,
        fast: bool | None = None,
    ):
        self.min_themes = min_themes
        self.max_themes = max_themes
        self.context = context
        self.fast = fast

    def run(self, ctx: Any) -> Any:
        texts = ctx.dataset.tolist()
        fast = self.fast if self.fast is not None else ctx.fast
        return ctx.client.generate_themes(
            texts, min_themes=self.min_themes, max_themes=self.max_themes, fast=fast
        )


class SentimentProcess:
    """Process: classify sentiment for texts."""

    id = "sentiment"
    depends_on: Tuple[str, ...] = ()

    def __init__(self, fast: bool | None = None):
        self.fast = fast

    def run(self, ctx: Any) -> Any:
        texts = ctx.dataset.tolist()
        fast = self.fast if self.fast is not None else ctx.fast
        return ctx.client.analyze_sentiment(texts, fast=fast)


class ThemeAllocation:
    """Process: allocate themes to texts based on generation results."""

    id = "theme_allocation"
    depends_on: Tuple[str, ...] = ("theme_generation",)

    def __init__(
        self,
        themes: list[str] | None = None,
        single_label: bool = True,
        threshold: float = 0.5,
    ):
        self.themes = themes
        self.single_label = single_label
        self.threshold = threshold

    def run(self, ctx: Any) -> dict[str, Any]:
        """
        Allocate themes to texts using similarity to theme labels.
        Returns raw dict including themes, single assignments, and similarity matrix.
        """
        texts = list(ctx.dataset)
        # Determine themes list (static or from another process)
        if self.themes is not None:
            themes = list(self.themes)
        else:
            alias = getattr(self, "_themes_from_alias", "theme_generation")
            tg = ctx.results.get(alias)
            if tg is None:
                raise RuntimeError(f"{alias} result not available for allocation")
            themes = tg.themes
        # base single assignment indices
        assignments = tg.assignments
        # Compute similarity between texts and theme labels
        merged = texts + themes
        fast = ctx.fast
        # request full matrix
        resp = ctx.client.compare_similarity(merged, fast=fast, flatten=False)
        full_sim = resp.similarity  # (n+m)x(n+m)
        n = len(texts)
        m = len(themes)
        # cross-similarity: top-left block cross texts->themes
        sim_matrix = [row[n : n + m] for row in full_sim[:n]]
        return {"themes": themes, "assignments": assignments, "similarity": sim_matrix}


class ThemeExtraction:
    """Process: extract elements matching themes from input strings."""

    id = "theme_extraction"
    depends_on: Tuple[str, ...] = ("theme_generation",)

    def __init__(
        self,
        themes: list[str] | None = None,
        version: str | None = None,
        fast: bool | None = None,
    ):
        self.themes = themes
        self.version = version
        self.fast = fast

    def run(self, ctx: Any) -> Any:
        texts = list(ctx.dataset)
        # Determine themes list (static or from another process)
        if self.themes is not None:
            used_themes = list(self.themes)
        else:
            alias = getattr(self, "_themes_from_alias", "theme_generation")
            prev = ctx.results.get(alias)
            if prev is None:
                raise RuntimeError(f"{alias} result not available for extraction")
            used_themes = prev.themes
        self.themes = used_themes
        self.themes = used_themes
        fast_flag = self.fast if self.fast is not None else ctx.fast
        return ctx.client.extract_elements(
            inputs=texts, themes=used_themes, version=self.version, fast=fast_flag
        )


class Cluster:
    """Process: compute similarity matrix for clustering."""

    id = "cluster"
    depends_on: Tuple[str, ...] = ()

    def __init__(self, fast: bool | None = None):
        self.fast = fast

    def run(self, ctx: Any) -> Any:
        """Compute similarity matrix for clustering (cached for later use)."""
        texts = list(ctx.dataset)
        fast = self.fast if self.fast is not None else ctx.fast
        # request full matrix (flatten=False for NxN)
        resp = ctx.client.compare_similarity(texts, fast=fast, flatten=False)
        # resp.similarity is List[List[float]]
        return resp.similarity
