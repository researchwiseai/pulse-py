"""Result helper classes for analysis processes."""

from typing import Any, Optional, Sequence
import pandas as pd
from pulse_client.core.models import (
    ThemesResponse,
    SentimentResponse,
    ExtractionsResponse,
)


class ThemeGenerationResult:
    """Results of theme generation with helper methods."""

    def __init__(self, response: ThemesResponse, texts: Sequence[str]) -> None:
        self._response = response
        self._texts = list(texts)

    @property
    def themes(self) -> list[str]:
        """List of theme shortLabels."""
        return [theme.shortLabel for theme in self._response.themes]

    # legacy assignments removed; assignment is handled by ThemeAllocation process

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert theme metadata to a pandas DataFrame with columns:
        [shortLabel, label, description, representative_1, representative_2]
        """
        data = []
        for theme in self._response.themes:
            data.append(
                {
                    "shortLabel": theme.shortLabel,
                    "label": theme.label,
                    "description": theme.description,
                    "representative_1": theme.representatives[0],
                    "representative_2": theme.representatives[1],
                }
            )
        return pd.DataFrame(data)


class SentimentResult:
    """Results of sentiment analysis with helper methods."""

    def __init__(self, response: SentimentResponse, texts: Sequence[str]) -> None:
        self._response = response
        self._texts = list(texts)

    @property
    def sentiments(self) -> list[Any]:
        """List of sentiment labels for each text."""
        return self._response.sentiments

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame with text and sentiment."""
        return pd.DataFrame(
            {
                "text": self._texts,
                "sentiment": self._response.sentiments,
            }
        )

    def summary(self) -> pd.Series:
        """Return a summary of sentiment counts as a pandas Series."""
        series = pd.Series(self._response.sentiments)
        return series.value_counts()

    def plot_distribution(self, **kwargs) -> Any:
        """Plot the distribution of sentiment labels using matplotlib."""
        series = self.summary()
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.bar(series.index.astype(str), series.values, **kwargs)
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        return ax


class ThemeAllocationResult:
    """Results of theme allocation with helper methods."""

    def __init__(
        self,
        texts: Sequence[str],
        themes: Sequence[str],
        assignments: Sequence[int],
        single_label: bool = True,
        threshold: float = 0.5,
        similarity: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self._texts = list(texts)
        self._themes = list(themes)
        self._assignments = list(assignments)
        self._single_label = single_label
        self._threshold = threshold
        # similarity matrix shape (n_texts x n_themes)
        self._similarity = similarity or []  # type: Sequence[Sequence[float]]

    def assign_single(self, threshold: Optional[float] = None) -> pd.Series:
        """Return a Series mapping each text to its single theme label.
        Applies threshold if provided."""
        thr = self._threshold if threshold is None else threshold
        labels = []
        for idx, assign in enumerate(self._assignments):
            if self._similarity:
                sim_row = self._similarity[idx]
                # find best index
                best_idx = max(range(len(sim_row)), key=lambda i: sim_row[i])
                best_val = sim_row[best_idx]
                if best_val >= thr:
                    labels.append(self._themes[best_idx])
                else:
                    labels.append(None)
            else:
                # fallback to single assignment
                labels.append(self._themes[assign])
        return pd.Series(labels, index=self._texts, name="theme")

    def assign_multi(self, k: int = 1) -> pd.DataFrame:
        """Return a DataFrame of top-k theme labels per text, based on similarity."""
        if not self._similarity:
            # fallback: replicate single assignment
            data = {
                f"theme_{j+1}": [self._themes[a] for a in self._assignments]
                for j in range(k)
            }
        else:
            data = {}
            for j in range(k):
                col = []
                for sim_row in self._similarity:
                    # sorted indices by similarity descending
                    sorted_idx = sorted(
                        range(len(sim_row)), key=lambda i: sim_row[i], reverse=True
                    )
                    if j < len(sorted_idx):
                        col.append(self._themes[sorted_idx[j]])
                    else:
                        col.append(None)
                data[f"theme_{j+1}"] = col
        return pd.DataFrame(data, index=self._texts)

    def heatmap(self, **kwargs) -> Any:
        """Plot a bar chart of theme assignment counts using matplotlib."""
        counts = pd.Series(self._assignments).value_counts().sort_index()
        labels = [self._themes[i] for i in counts.index]
        values = counts.values
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.barh(labels, values, **kwargs)
        ax.set_xlabel("Count")
        ax.set_ylabel("Theme")
        return ax


class ClusterResult:
    """Results of clustering with helper methods."""

    def __init__(
        self, similarity_matrix: Sequence[Sequence[float]], texts: Sequence[str]
    ) -> None:
        import numpy as np

        self._matrix = np.array(similarity_matrix)
        self._texts = list(texts)

    @property
    def matrix(self) -> Any:
        """Return the raw similarity matrix as NumPy array."""
        return self._matrix

    def kmeans(self, n_clusters: int, **kwargs) -> Any:
        """Perform KMeans clustering on the similarity matrix."""
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=n_clusters, **kwargs)
        labels = model.fit_predict(self._matrix)
        return labels

    def dbscan(self, eps: float = 0.5, min_samples: int = 5, **kwargs) -> Any:
        """Perform DBSCAN clustering on the similarity matrix."""
        from sklearn.cluster import DBSCAN

        model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        labels = model.fit_predict(self._matrix)
        return labels

    def plot_scatter(self, **kwargs) -> Any:
        """Plot a 2D scatter of items via PCA reduction of the similarity matrix."""
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        coords = PCA(n_components=2).fit_transform(self._matrix)
        fig, ax = plt.subplots()
        ax.scatter(coords[:, 0], coords[:, 1], **kwargs)
        for i, txt in enumerate(self._texts):
            ax.annotate(txt, (coords[i, 0], coords[i, 1]))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        return ax

    def dendrogram(self, **kwargs) -> Any:
        """
        Plot a hierarchical clustering dendrogram based on the similarity matrix.
        Converts similarity to distances (1 - similarity) and uses SciPy linkage.
        """
        from scipy.cluster.hierarchy import linkage, dendrogram as _dendrogram
        from scipy.spatial.distance import squareform
        import matplotlib.pyplot as _plt

        # Convert similarity to distance matrix
        dist_matrix = 1 - self._matrix
        # Condense distance matrix
        condensed = squareform(dist_matrix, checks=False)
        # Compute linkage (Ward method)
        Z = linkage(condensed, method=kwargs.pop("method", "ward"))
        # Plot dendrogram
        fig, ax = _plt.subplots()
        _dendrogram(Z, labels=self._texts, ax=ax, **kwargs)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Distance")
        return ax


class ThemeExtractionResult:
    """Results of theme extraction with helper methods."""

    def __init__(
        self,
        response: ExtractionsResponse,
        texts: Sequence[str],
        themes: Sequence[str],
    ) -> None:
        self._response = response
        self._texts = list(texts)
        self._themes = list(themes)

    @property
    def extractions(self) -> list[list[list[str]]]:
        """Nested list of extracted elements per text per theme."""
        return self._response.extractions

    def to_dataframe(self) -> pd.DataFrame:
        """Convert extraction results to a DataFrame.
        Columns: text, theme, extraction."""
        rows: list[dict[str, str]] = []
        for i, text in enumerate(self._texts):
            for j, theme in enumerate(self._themes):
                try:
                    items = self._response.extractions[i][j]
                except (IndexError, TypeError):
                    continue
                for item in items:
                    rows.append({"text": text, "theme": theme, "extraction": item})
        return pd.DataFrame(rows)
