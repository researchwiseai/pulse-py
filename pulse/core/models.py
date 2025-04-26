"""Pydantic models for Pulse API responses."""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator


class EmbeddingDocument(BaseModel):
    """Single embedding document as returned by the embeddings API."""

    id: Optional[str] = Field(None, description="Optional document identifier")
    text: str = Field(..., description="Input text for this embedding")
    vector: List[float] = Field(..., description="Dense vector encoding of the text")


class EmbeddingsResponse(BaseModel):
    """Response model for batch embeddings."""

    embeddings: List[EmbeddingDocument] = Field(
        ..., description="List of embedding documents (text + vector)"
    )
    requestId: Optional[str] = Field(None, description="Unique request identifier")


class SimilarityResponse(BaseModel):
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


class Theme(BaseModel):
    """Single theme metadata as returned by the API."""

    shortLabel: str = Field(..., description="Concise name for dashboard display")
    label: str = Field(..., description="Descriptive title of the theme")
    description: str = Field(..., description="One-sentence summary of the theme")
    representatives: List[str] = Field(
        ..., min_length=2, max_length=2, description="Two representative input strings"
    )


class ThemesResponse(BaseModel):
    """Response model for thematic clustering."""

    themes: List[Theme] = Field(..., description="List of cluster metadata objects")
    requestId: Optional[str] = Field(None, description="Unique request identifier")


class SentimentResult(BaseModel):
    """Single sentiment classification result."""

    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        ..., description="Sentiment category"
    )
    confidence: float = Field(..., description="Confidence score between 0 and 1")


class SentimentResponse(BaseModel):
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


class ExtractionsResponse(BaseModel):
    """Response model for text element extraction."""

    extractions: List[List[List[str]]] = Field(
        ..., description="3D array of extracted elements, shape [inputs][themes][k]"
    )
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
