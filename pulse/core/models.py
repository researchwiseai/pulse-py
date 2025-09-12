"""Pydantic models for Pulse API responses."""

from typing import Any, List, Optional, Literal, Dict
from pydantic import BaseModel, Field, model_validator, ConfigDict


class UsageRecord(BaseModel):
    """Single usage record for a feature."""

    feature: str = Field(..., description="Name of the feature")
    units: int = Field(..., description="Units consumed for the feature")


class UsageReport(BaseModel):
    """Usage summary returned by the API."""

    total: int = Field(..., description="Total units consumed")
    records: List[UsageRecord] = Field(
        default_factory=list, description="Per-feature usage records"
    )


class UsageModel(BaseModel):
    """Mixin for responses that include usage information."""

    usage: Optional[UsageReport] = Field(
        None, description="Usage information for the request"
    )

    @property
    def usage_total(self) -> Optional[int]:
        """Return total units consumed if usage info is available."""
        return self.usage.total if self.usage else None

    def usage_records_by_feature(self) -> Dict[str, UsageRecord]:
        """Return usage records keyed by feature name."""
        if not self.usage:
            return {}
        return {record.feature: record for record in self.usage.records}


class EmbeddingDocument(BaseModel):
    """Single embedding document as returned by the embeddings API."""

    id: Optional[str] = Field(None, description="Optional document identifier")
    text: str = Field(..., description="Input text for this embedding")
    vector: List[float] = Field(..., description="Dense vector encoding of the text")


class EmbeddingsRequest(BaseModel):
    """Request model for generating embeddings."""

    inputs: List[str] = Field(
        ..., min_length=1, max_length=2000, description="Input texts"
    )
    fast: Optional[bool] = Field(
        None, description="Synchronous (True) or asynchronous (False)"
    )

    model_config = ConfigDict(populate_by_name=True)


class EmbeddingsResponse(UsageModel):
    """Response model for batch embeddings."""

    embeddings: List[EmbeddingDocument] = Field(
        ..., description="List of embedding documents (text + vector)"
    )
    requestId: Optional[str] = Field(None, description="Unique request identifier")


class SimilarityResponse(UsageModel):
    """Response model for cosine similarity computations."""

    scenario: Literal["self", "cross"] = Field(
        ..., description="Self-similarity or cross-similarity scenario"
    )
    mode: Literal["matrix", "flattened"] = Field(
        ..., description="Representation mode: matrix or flattened"
    )
    n: int = Field(..., description="Number of input texts (for self-similarity)")
    flattened: List[float] = Field(..., description="Flattened similarity values")
    matrix: Optional[List[List[float]]] = Field(
        None, description="Full similarity matrix"
    )
    requestId: Optional[str] = Field(None, description="Unique request identifier")

    @property
    def similarity(self) -> List[List[float]]:
        """
        Return the full similarity matrix. If `matrix` is provided, use it.
        Otherwise reconstruct from `flattened` based on the `scenario`.
        """
        if self.matrix:
            return self.matrix

        flat = self.flattened

        if self.scenario == "self":
            # flattened upper triangle (with or without diagonal)
            n = self.n
            total = len(flat)
            full_tri_len = n * (n + 1) // 2
            no_diag_tri_len = n * (n - 1) // 2

            # init zero matrix
            mat = [[0.0] * n for _ in range(n)]
            idx = 0

            if total == full_tri_len:
                # includes diagonal
                for i in range(n):
                    for j in range(i, n):
                        mat[i][j] = flat[idx]
                        mat[j][i] = flat[idx]
                        idx += 1
            elif total == no_diag_tri_len:
                # excludes diagonal: assume diagonal = 1
                for i in range(n):
                    mat[i][i] = 1.0
                for i in range(n):
                    for j in range(i + 1, n):
                        mat[i][j] = flat[idx]
                        mat[j][i] = flat[idx]
                        idx += 1
            else:
                raise ValueError(
                    f"Unexpected length {total} for self-similarity with n={n}"
                )

            return mat

        elif self.scenario == "cross":
            # flattened full cross-matrix of shape (n x m)
            n = self.n
            total = len(flat)
            if n <= 0 or total % n != 0:
                raise ValueError(
                    f"Cannot reshape flattened length {total} into {n} rows"
                )
            m = total // n
            return [flat[i * m : (i + 1) * m] for i in range(n)]

        else:
            # unknown scenario
            return []


class UnitAgg(BaseModel):
    """Unit and aggregation options for text splitting."""

    unit: Literal["sentence", "newline"]
    agg: Literal["mean", "max"] = Field("mean")


class Split(BaseModel):
    """Split configuration for similarity requests."""

    unit: Optional[Literal["sentence", "newline"]] = None
    agg: Optional[Literal["mean", "max"]] = None
    set_a: Optional[UnitAgg] = Field(None, alias="set_a")
    set_b: Optional[UnitAgg] = Field(None, alias="set_b")

    model_config = ConfigDict(populate_by_name=True)


class SimilarityRequest(BaseModel):
    """Request model for computing similarities."""

    set: Optional[List[str]] = Field(None, min_length=2)
    set_a: Optional[List[str]] = Field(None, alias="set_a")
    set_b: Optional[List[str]] = Field(None, alias="set_b")
    fast: Optional[bool] = None
    flatten: bool = False
    version: Optional[str] = None
    split: Optional[Split] = None

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _check_sets(cls, data: Any) -> "SimilarityRequest":
        if data.set is None and (data.set_a is None or data.set_b is None):
            raise ValueError("Provide `set` or both `set_a` and `set_b`.")
        if data.set is not None and (data.set_a is not None or data.set_b is not None):
            raise ValueError("Cannot provide both `set` and `set_a`/`set_b`.")
        return data


class Theme(BaseModel):
    """Single theme metadata as returned by the API."""

    shortLabel: str = Field(..., description="Concise name for dashboard display")
    label: str = Field(..., description="Descriptive title of the theme")
    description: str = Field(..., description="One-sentence summary of the theme")
    representatives: List[str] = Field(
        ..., min_length=2, max_length=2, description="Two representative input strings"
    )


class ThemesResponse(UsageModel):
    """Response model for thematic clustering."""

    themes: List[Theme] = Field(..., description="List of cluster metadata objects")
    requestId: Optional[str] = Field(None, description="Unique request identifier")


class SentimentResult(BaseModel):
    """Single sentiment classification result."""

    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        ..., description="Sentiment category"
    )
    confidence: float = Field(..., description="Confidence score between 0 and 1")


class SentimentResponse(UsageModel):
    """Response model for sentiment analysis."""

    results: List[SentimentResult] = Field(
        ..., description="Sentiment results for each input string"
    )
    requestId: Optional[str] = Field(None, description="Unique request identifier")

    @model_validator(mode="before")
    def _normalize_legacy(cls, values: dict) -> dict:
        """
        Allow legacy 'sentiments' field input by mapping into results list,
        mapping shorthand labels to full values.
        """
        if "sentiments" in values:
            sens = values.pop("sentiments") or []
            # map shorthand to full labels
            mapping = {"pos": "positive", "neg": "negative", "neu": "neutral"}
            mapped = [mapping.get(s, s) for s in sens]
            values["results"] = [{"sentiment": s, "confidence": 0.0} for s in mapped]
        return values

    @property
    def sentiments(self) -> List[str]:
        """
        Convenience property extracting sentiment labels only.
        """
        return [r.sentiment for r in self.results]


class ExtractionsRequest(BaseModel):
    """Request model for text element extraction."""

    texts: List[str] = Field(..., min_length=1, description="Input texts")
    categories: List[str] = Field(
        ..., min_length=1, description="Categories to extract elements for"
    )
    dictionary: Optional[dict[str, List[str]]] = Field(
        None, description="Optional mapping of category to search terms"
    )
    expand_dictionary: Optional[bool] = Field(
        None, description="Expand dictionary entries with synonyms"
    )
    use_ner: Optional[bool] = Field(None, description="Enable named-entity recognition")
    use_llm: Optional[bool] = Field(None, description="Enable LLM powered extraction")
    threshold: Optional[float] = Field(
        None, description="Score threshold for extraction"
    )

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    def _normalize_legacy(cls, values: dict) -> dict:
        if "inputs" in values and "texts" not in values:
            values["texts"] = values.pop("inputs")
        if "themes" in values and "categories" not in values:
            values["categories"] = values.pop("themes")
        # drop deprecated fields from older API versions
        values.pop("version", None)
        values.pop("fast", None)
        if isinstance(values.get("dictionary"), bool):
            values.pop("dictionary")
        return values


class ExtractionsResponse(UsageModel):
    """Response model for text element extraction."""

    class ExtractionColumn(BaseModel):
        category: str
        term: str

    columns: List[ExtractionColumn] = Field(..., description="Column metadata")
    matrix: List[List[str]] = Field(..., description="Extraction results matrix")
    requestId: Optional[str] = Field(None, description="Unique request identifier")


class JobSubmissionResponse(BaseModel):
    """Initial response model for async job submission (202 Accepted)."""

    jobId: str = Field(..., description="Unique job identifier")


class JobStatusResponse(BaseModel):
    """Polling response model for job status endpoint."""

    jobId: str = Field(..., description="Unique job identifier")
    jobStatus: Literal["pending", "completed", "error", "failed"] = Field(
        ..., description="Current job status"
    )
    resultUrl: Optional[str] = Field(
        None, description="URL to fetch job result upon completion"
    )
    message: Optional[str] = Field(
        None, description="Error message if jobStatus is error or failed"
    )


class ClusteringRequest(BaseModel):
    """Request model for text clustering."""

    inputs: List[str] = Field(..., min_length=2, description="Input texts")
    k: int = Field(..., ge=1, le=50, description="Number of clusters")
    algorithm: Optional[Literal["kmeans", "skmeans", "agglomerative", "hdbscan"]] = (
        Field(None, description="Clustering algorithm")
    )
    fast: Optional[bool] = Field(
        None, description="Synchronous (True) or asynchronous (False)"
    )


class Cluster(BaseModel):
    """Single cluster grouping."""

    clusterId: int = Field(..., description="Cluster identifier")
    items: List[str] = Field(..., description="Items assigned to this cluster")


class ClusteringResponse(UsageModel):
    """Response model for clustering request."""

    algorithm: str = Field(..., description="Algorithm used for clustering")
    clusters: List[Cluster] = Field(..., description="List of cluster groups")
    requestId: Optional[str] = Field(None, description="Unique request identifier")


class SummariesRequest(BaseModel):
    """Request model for text summarization."""

    inputs: List[str] = Field(..., min_length=1, description="Input texts")
    question: str = Field(..., description="Question to guide the summary")
    length: Optional[Literal["bullet-points", "short", "medium", "long"]] = Field(
        None, description="Desired summary length"
    )
    preset: Optional[
        Literal[
            "five-point",
            "ten-point",
            "one-tweet",
            "three-tweets",
            "one-para",
            "exec",
            "two-pager",
            "one-pager",
        ]
    ] = Field(None, description="Predefined summary style")
    fast: Optional[bool] = Field(
        None, description="Synchronous (True) or asynchronous (False)"
    )


class SummariesResponse(UsageModel):
    """Response model for text summarization."""

    summary: str = Field(..., description="Generated summary text")
    requestId: Optional[str] = Field(None, description="Unique request identifier")
