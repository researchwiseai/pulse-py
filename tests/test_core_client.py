"""End-to-end tests for CoreClient using pytest-vcr.

All HTTP interactions are recorded and replayed; no manual mocks.
"""
import pytest

from pulse_client.core.client import CoreClient
from pulse_client.core.exceptions import PulseAPIError

pytestmark = pytest.mark.vcr(record_mode="new_episodes")


@pytest.fixture(autouse=True)
def disable_sleep(monkeypatch):
    import time

    monkeypatch.setattr(time, "sleep", lambda x: None)


def test_create_embeddings_fast():
    client = CoreClient()
    # fast=True should error on queued 202 response
    with pytest.raises(PulseAPIError):
        client.create_embeddings(["a", "b"], fast=True)


def test_compare_similarity_fast():
    client = CoreClient()
    # fast=True should error on queued 202 response
    with pytest.raises(PulseAPIError):
        client.compare_similarity(["x", "y"], fast=True, flatten=False)


def test_generate_themes_fast():
    client = CoreClient()
    # fast=True should error on queued 202 response
    with pytest.raises(PulseAPIError):
        client.generate_themes(["x", "y"], min_themes=1, max_themes=3, fast=True)


def test_analyze_sentiment_fast():
    client = CoreClient()
    # fast=True should error on queued 202 response
    with pytest.raises(PulseAPIError):
        client.analyze_sentiment(["happy", "sad"], fast=True)


def test_error_raises():
    client = CoreClient()
    with pytest.raises(PulseAPIError):
        client.create_embeddings([], fast=True)
