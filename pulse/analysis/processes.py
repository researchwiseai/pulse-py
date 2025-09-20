"""Built-in Process primitives for Analyzer."""

from typing import Any, Tuple
from pulse.core.models import Theme as ThemeModel

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol
import random


class Process(Protocol):
    """Process primitive protocol."""

    id: str
    depends_on: Tuple[str, ...]

    def run(self, ctx: Any) -> Any: ...


class ThemeGeneration:
    """Process: cluster texts into latent themes."""

    id = "theme_generation"
    depends_on: Tuple[str, ...] = ()

    def __init__(
        self,
        min_themes: int = 2,
        max_themes: int = 50,
        context: Any = None,
        version: str | None = None,
        prune: int | None = None,
        interactive: bool | None = None,
        initial_sets: int | None = None,
        fast: bool | None = None,
        await_job_result: bool = True,
    ):
        self.min_themes = min_themes
        self.max_themes = max_themes
        self.context = context
        self.version = version
        self.prune = prune
        self.interactive = interactive
        self.initial_sets = initial_sets
        self.fast = fast
        self.await_job_result = await_job_result

    def run(self, ctx: Any) -> Any:
        texts = ctx.dataset.tolist()
        fast_flag = self.fast if self.fast is not None else ctx.fast

        # sample randomly according to fast flag
        sample_size = 200 if fast_flag else 1000
        if len(texts) > sample_size:
            texts = random.sample(
                texts, sample_size
            )  # nosec B311 - Used for data sampling, not cryptographic purposes

        return ctx.client.generate_themes(
            texts,
            min_themes=self.min_themes,
            max_themes=self.max_themes,
            fast=self.fast or ctx.fast,
            context=self.context,
            version=self.version,
            prune=self.prune,
            interactive=self.interactive,
            initial_sets=self.initial_sets,
            await_job_result=self.await_job_result,
        )


class SentimentProcess:
    """Process: classify sentiment for texts."""

    id = "sentiment"
    depends_on: Tuple[str, ...] = ()

    def __init__(self, fast: bool | None = None):
        self.fast = fast

    def run(self, ctx: Any) -> Any:
        texts = ctx.dataset.tolist()
        return ctx.client.analyze_sentiment(texts, fast=self.fast or ctx.fast)


class ThemeAllocation:
    """Process: allocate themes to texts based on generation results."""

    id = "theme_allocation"
    depends_on: Tuple[str, ...] = ("theme_generation",)

    def __init__(
        self,
        themes: list[str] | None = None,
        single_label: bool = True,
        fast: bool | None = None,
        threshold: float = 0.5,
    ):
        self.themes = themes
        self.single_label = single_label
        self.threshold = threshold
        self.fast = fast

    def run(self, ctx: Any) -> dict[str, Any]:
        """
        Allocate themes to texts using similarity to theme labels.
        Returns raw dict including themes, single assignments, and similarity matrix.
        """
        texts = list(ctx.dataset)
        # Determine raw themes list (static strings or ThemeModel instances)
        if self.themes is not None:
            raw_themes = list(self.themes)
        else:
            alias = getattr(self, "_themes_from_alias", "theme_generation")
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
        fast_flag = self.fast if self.fast is not None else ctx.fast

        resp = ctx.client.compare_similarity(
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


class SimilarityProcess:
    """Process: compute similarity between texts with optional splitting."""

    id = "similarity"
    depends_on: Tuple[str, ...] = ()

    def __init__(
        self,
        set_a: list[str] | None = None,
        set_b: list[str] | None = None,
        split: dict[str, Any] | None = None,
        flatten: bool = False,
        version: str | None = None,
        fast: bool | None = None,
        await_job_result: bool = True,
    ):
        self.set_a = set_a
        self.set_b = set_b
        self.split = split
        self.flatten = flatten
        self.version = version
        self.fast = fast
        self.await_job_result = await_job_result

    def run(self, ctx: Any) -> Any:
        """Compute similarity with optional text splitting."""
        texts = list(ctx.dataset)

        # Use provided sets or default to dataset
        set_a = self.set_a or texts
        set_b = self.set_b

        if set_b is None:
            # Self-similarity
            return ctx.client.compare_similarity(
                set=set_a,
                split=self.split,
                flatten=self.flatten,
                version=self.version,
                fast=self.fast or ctx.fast,
                await_job_result=self.await_job_result,
            )
        else:
            # Cross-similarity
            return ctx.client.compare_similarity(
                set_a=set_a,
                set_b=set_b,
                split=self.split,
                flatten=self.flatten,
                version=self.version,
                fast=self.fast or ctx.fast,
                await_job_result=self.await_job_result,
            )


class ThemeExtraction:
    """Process: extract elements matching themes from input strings."""

    id = "theme_extraction"
    depends_on: Tuple[str, ...] = ("theme_generation",)

    def __init__(
        self,
        themes: list[str] | None = None,
        dictionary: list[str] | None = None,
        type: str = "named-entities",
        expand_dictionary: bool = False,
        expand_dictionary_limit: int | None = None,
        version: str | None = None,
        fast: bool | None = None,
        await_job_result: bool = True,
        # Deprecated parameters for backward compatibility
        use_ner: bool | None = None,
        use_llm: bool | None = None,
        threshold: float | None = None,
    ):
        self.themes = themes
        self.dictionary = dictionary
        self.type = type
        self.expand_dictionary = expand_dictionary
        self.expand_dictionary_limit = expand_dictionary_limit
        self.version = version
        self.fast = fast
        self.await_job_result = await_job_result
        # Deprecated parameters
        self.use_ner = use_ner
        self.use_llm = use_llm
        self.threshold = threshold

    def run(self, ctx: Any) -> Any:
        texts = list(ctx.dataset)

        # Determine dictionary - use provided dictionary or themes
        if self.dictionary is not None:
            used_dictionary = list(self.dictionary)
        elif self.themes is not None:
            used_dictionary = list(self.themes)
        else:
            # Get themes from previous process
            alias = getattr(self, "_themes_from_alias", "theme_generation")
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

        return ctx.client.extract_elements(
            inputs=texts,
            dictionary=used_dictionary,
            type=self.type,
            expand_dictionary=self.expand_dictionary,
            expand_dictionary_limit=self.expand_dictionary_limit,
            version=self.version,
            fast=self.fast or ctx.fast,
            await_job_result=self.await_job_result,
            # Pass deprecated parameters for backward compatibility
            use_ner=self.use_ner,
            use_llm=self.use_llm,
            threshold=self.threshold,
        )


class Cluster:
    """Process: cluster texts using various algorithms."""

    id = "cluster"
    depends_on: Tuple[str, ...] = ()

    def __init__(
        self,
        k: int = 2,
        algorithm: str = "kmeans",
        fast: bool | None = None,
        await_job_result: bool = True,
    ):
        self.k = k
        self.algorithm = algorithm
        self.fast = fast
        self.await_job_result = await_job_result

    def run(self, ctx: Any) -> Any:
        """Cluster texts using the specified algorithm."""
        texts = list(ctx.dataset)
        return ctx.client.cluster_texts(
            inputs=texts,
            k=self.k,
            algorithm=self.algorithm,
            fast=self.fast or ctx.fast,
            await_job_result=self.await_job_result,
        )
