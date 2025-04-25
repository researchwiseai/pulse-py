"""End-to-end tests for CoreClient using pytest-vcr.

All HTTP interactions are recorded and replayed; no manual mocks.
"""
import pytest

from pulse_client.core.client import CoreClient
from pulse_client.core.models import (
    EmbeddingsResponse,
    SimilarityResponse,
    ThemesResponse,
    SentimentResponse,
)
from pulse_client.core.exceptions import PulseAPIError

pytestmark = pytest.mark.vcr(record_mode="new_episodes")


@pytest.fixture(autouse=True)
def disable_sleep(monkeypatch):
    import time

    monkeypatch.setattr(time, "sleep", lambda x: None)


def test_create_embeddings_fast():
    client = CoreClient()
    resp = client.create_embeddings(["a", "b"], fast=True)
    assert isinstance(resp, EmbeddingsResponse)
    assert hasattr(resp, "embeddings")
    assert isinstance(resp.embeddings, list)
    assert all(isinstance(row, list) for row in resp.embeddings)


def test_compare_similarity_fast():
    client = CoreClient()
    resp = client.compare_similarity(["x", "y"], fast=True, flatten=False)
    assert isinstance(resp, SimilarityResponse)
    assert hasattr(resp, "similarity")
    assert isinstance(resp.similarity, list)
    assert all(isinstance(row, list) for row in resp.similarity)


def test_generate_themes_fast():
    client = CoreClient()
    resp = client.generate_themes(["x", "y"], min_themes=1, max_themes=3, fast=True)
    assert isinstance(resp, ThemesResponse)
    assert hasattr(resp, "themes")
    assert isinstance(resp.themes, list)
    assert hasattr(resp, "assignments")
    assert isinstance(resp.assignments, list)


def test_analyze_sentiment_fast():
    client = CoreClient()
    resp = client.analyze_sentiment(["happy", "sad"], fast=True)
    assert isinstance(resp, SentimentResponse)
    assert hasattr(resp, "sentiments")
    assert isinstance(resp.sentiments, list)


def test_error_raises():
    client = CoreClient()
    with pytest.raises(PulseAPIError):
        client.create_embeddings([], fast=True)
