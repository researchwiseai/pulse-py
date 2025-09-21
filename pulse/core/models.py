"""Pydantic models for Pulse API responses."""

from typing import Any, List, Optional, Literal, Dict, Union
from pydantic import BaseModel, Field, model_validator, ConfigDict


class UsageRecord(BaseModel):
    """Single usage record for a feature."""

    feature: str = Field(..., description="Name of the feature")
    quantity: int = Field(..., description="Quantity consumed for the feature")

    model_config = ConfigDict(populate_by_name=True)

    @property
    def units(self) -> int:
        """Backward compatibility property for units field."""
        return self.quantity


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
    """Request model for generating embeddings.

    Allow arbitrary input element types to defer strict validation to the API,
    so client tests can verify server-side error handling.
    """

    inputs: List[Any] = Field(
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

    unit: Literal["sentence", "newline", "word"] = Field(
        ..., description="Text splitting unit"
    )
    agg: Literal["mean", "max", "top2", "top3"] = Field(
        "mean", description="Aggregation method"
    )
    window_size: int = Field(
        1, ge=1, description="Window size for sliding window processing"
    )
    stride_size: int = Field(
        1, ge=1, description="Stride size for sliding window processing"
    )


class Split(BaseModel):
    """Split configuration for similarity requests."""

    set_a: Optional[UnitAgg] = Field(
        None, description="Splitting configuration for set_a"
    )
    set_b: Optional[UnitAgg] = Field(
        None, description="Splitting configuration for set_b"
    )

    model_config = ConfigDict(populate_by_name=True)


class SimilarityRequest(BaseModel):
    """Request model for computing similarities."""

    set: Optional[List[str]] = Field(
        None, min_length=2, max_length=44721, description="Self-similarity input texts"
    )
    set_a: Optional[List[str]] = Field(
        None, min_length=1, max_length=2000000000, description="Cross-similarity set A"
    )
    set_b: Optional[List[str]] = Field(
        None, min_length=1, max_length=2000000000, description="Cross-similarity set B"
    )
    fast: Optional[bool] = Field(
        None, description="Synchronous (True) or asynchronous (False)"
    )
    flatten: bool = Field(False, description="Return flattened results")
    version: Optional[str] = Field(None, description="API version")
    split: Optional[Split] = Field(None, description="Text splitting configuration")

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _check_sets(cls, data: Any) -> "SimilarityRequest":
        if data.set is None and (data.set_a is None or data.set_b is None):
            raise ValueError("Provide `set` or both `set_a` and `set_b`.")
        if data.set is not None and (data.set_a is not None or data.set_b is not None):
            raise ValueError("Cannot provide both `set` and `set_a`/`set_b`.")
        return data


class ThemesRequest(BaseModel):
    """Request model for theme generation."""

    inputs: List[str] = Field(
        ..., min_length=2, max_length=500, description="Input texts"
    )
    minThemes: Optional[int] = Field(None, ge=1, description="Minimum number of themes")
    maxThemes: Optional[int] = Field(
        None, le=50, description="Maximum number of themes"
    )
    context: Optional[str] = Field(
        None, description="Context to steer theme generation"
    )
    version: Optional[str] = Field(None, description="API version")
    prune: Optional[int] = Field(None, ge=0, le=25, description="Pruning threshold")
    interactive: Optional[bool] = Field(None, description="Enable interactive mode")
    initialSets: Optional[int] = Field(
        None, ge=1, le=3, description="Number of initial theme sets"
    )
    fast: Optional[bool] = Field(
        None, description="Synchronous (True) or asynchronous (False)"
    )

    @model_validator(mode="after")
    def validate_initial_sets(self) -> "ThemesRequest":
        """Validate that initialSets > 1 requires interactive=true."""
        if self.initialSets and self.initialSets > 1 and not self.interactive:
            raise ValueError("initialSets > 1 requires interactive=true")
        return self


class Theme(BaseModel):
    """Single theme metadata as returned by the API."""

    shortLabel: str = Field(..., description="Concise name for dashboard display")
    label: str = Field(..., description="Descriptive title of the theme")
    description: str = Field(..., description="One-sentence summary of the theme")
    representatives: List[str] = Field(
        ..., min_length=1, max_length=10, description="Representative input strings"
    )


class ThemesResponse(UsageModel):
    """Response model for thematic clustering."""

    themes: List[Theme] = Field(..., description="List of cluster metadata objects")
    requestId: Optional[str] = Field(None, description="Unique request identifier")


class ThemeSetsResponse(UsageModel):
    """Response model for themes with multiple theme sets (version 2025-09-01)."""

    themeSets: List[List[Theme]] = Field(
        ..., max_length=3, description="List of theme sets"
    )
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

    inputs: List[str] = Field(
        ..., min_length=1, max_length=5000, description="Input texts"
    )
    dictionary: List[str] = Field(
        ..., min_length=3, max_length=200, description="Dictionary terms to extract"
    )
    type: Literal["named-entities", "themes"] = Field(
        "named-entities", description="Extraction type"
    )
    expand_dictionary: bool = Field(
        False, description="Expand dictionary entries with synonyms"
    )
    expand_dictionary_limit: Optional[int] = Field(
        None, description="Limit for dictionary expansions"
    )
    version: Optional[str] = Field(None, description="API version")
    fast: Optional[bool] = Field(
        None, description="Synchronous (True) or asynchronous (False)"
    )
    # Deprecated fields maintained for backward compatibility
    category: Optional[str] = Field(None, description="Deprecated: use type instead")
    texts: Optional[List[str]] = Field(
        None, description="Deprecated: use inputs instead"
    )
    categories: Optional[List[str]] = Field(
        None, description="Deprecated: use dictionary instead"
    )
    use_ner: Optional[bool] = Field(None, description="Deprecated field")
    use_llm: Optional[bool] = Field(None, description="Deprecated field")
    threshold: Optional[float] = Field(None, description="Deprecated field")

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def validate_themes_constraints(self) -> "ExtractionsRequest":
        """Validate that expand_dictionary=false when type='themes'."""
        if self.type == "themes" and self.expand_dictionary:
            raise ValueError("expand_dictionary must be false when type is 'themes'")
        return self

    @model_validator(mode="before")
    def _normalize_legacy(cls, values: dict) -> dict:
        # Handle legacy field mappings
        if "texts" in values and "inputs" not in values:
            values["inputs"] = values.pop("texts")
        if "themes" in values and "dictionary" not in values:
            values["dictionary"] = values.pop("themes")
        if "categories" in values and "dictionary" not in values:
            values["dictionary"] = values.pop("categories")

        # Handle legacy dictionary format
        if isinstance(values.get("dictionary"), dict):
            # Convert dict format to list format
            dict_terms = values.get("dictionary", {})
            if dict_terms:
                # Flatten all terms from the dictionary
                all_terms = []
                for term_list in dict_terms.values():
                    all_terms.extend(term_list)
                values["dictionary"] = all_terms
        elif isinstance(values.get("dictionary"), bool):
            values.pop("dictionary")

        # Remove deprecated fields that are no longer used
        for deprecated_field in ["use_ner", "use_llm", "threshold"]:
            values.pop(deprecated_field, None)

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

    jobId: str = Field(..., alias="job_id", description="Unique job identifier")

    model_config = ConfigDict(populate_by_name=True)


class JobStatusResponse(BaseModel):
    """Polling response model for job status endpoint."""

    jobId: str = Field(..., alias="job_id", description="Unique job identifier")
    jobStatus: Literal["pending", "queued", "completed", "error", "failed"] = Field(
        ..., alias="job_status", description="Current job status"
    )
    resultUrl: Optional[str] = Field(
        None, alias="result_url", description="URL to fetch job result upon completion"
    )
    message: Optional[str] = Field(
        None, description="Error message if jobStatus is error or failed"
    )

    model_config = ConfigDict(populate_by_name=True)


class ClusteringRequest(BaseModel):
    """Request model for text clustering."""

    inputs: List[str] = Field(
        ..., min_length=2, max_length=44721, description="Input texts"
    )
    k: int = Field(..., ge=1, le=50, description="Number of clusters")
    algorithm: Literal["kmeans", "skmeans", "agglomerative", "hdbscan"] = Field(
        "kmeans", description="Clustering algorithm"
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


class UsageEstimateRequest(BaseModel):
    """Request model for usage estimation."""

    feature: Literal[
        "embeddings",
        "sentiment",
        "themes",
        "extractions",
        "summaries",
        "clustering",
        "similarity",
    ] = Field(..., description="Feature to estimate usage for")
    inputs: List[str] = Field(
        ..., min_length=1, description="Input texts for estimation"
    )


class UsageEstimateResponse(BaseModel):
    """Response model for usage estimation."""

    usage: Dict[str, Any] = Field(..., description="Estimated usage information")


class ErrorDetail(BaseModel):
    """Detailed error information for field-level validation errors."""

    code: Optional[str] = Field(None, description="Error code")
    message: str = Field(..., description="Error message")
    path: Optional[List[Union[str, int]]] = Field(
        None, description="Path to the field that caused the error"
    )
    field: Optional[str] = Field(None, description="Field name that caused the error")
    location: Optional[Literal["body", "query", "header", "path"]] = Field(
        None, description="Location of the error in the request"
    )


class ErrorResponse(BaseModel):
    """Enhanced error response model matching new API structure."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    errors: Optional[List[ErrorDetail]] = Field(
        None, description="Detailed field-level error information"
    )
    meta: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata about the error"
    )
