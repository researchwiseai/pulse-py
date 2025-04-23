"""Pydantic models for Pulse API responses."""

from pydantic import BaseModel
from typing import List, Any

class EmbeddingsResponse(BaseModel):
     embeddings: List[List[float]]

class SimilarityResponse(BaseModel):
     similarity: List[List[float]]

class ThemesResponse(BaseModel):
     themes: List[str]
     assignments: List[int]

class SentimentResponse(BaseModel):
     sentiments: List[Any]