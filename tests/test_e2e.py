"""End-to-end tests for CoreClient against the real Pulse API (recorded via VCR)."""

import pytest

from pulse_client.core.client import CoreClient
from pulse_client.core.models import EmbeddingDocument


@pytest.fixture(autouse=True)
def disable_sleep(monkeypatch):
    import time

    monkeypatch.setattr(time, "sleep", lambda x: None)


@pytest.mark.vcr()
def test_create_embeddings_e2e():
    client = CoreClient(base_url="https://dev.core.researchwiseai.com/pulse/v1")
    # Skip if API not reachable
    try:
        resp = client.create_embeddings(["test e2e", "pulse client"], fast=False)
    except Exception as exc:
        pytest.skip(f"Skipping E2E create_embeddings: {exc}")
    assert hasattr(resp, "embeddings"), "Response has no embeddings field"
    assert isinstance(resp.embeddings, list)
    # embeddings should be parsed as EmbeddingDocument instances
    assert all(isinstance(doc, EmbeddingDocument) for doc in resp.embeddings)


@pytest.mark.vcr()
def test_compare_similarity_e2e():
    client = CoreClient(base_url="https://dev.core.researchwiseai.com/pulse/v1")
    try:
        # pass 'set' keyword due to keyword-only parameters
        resp = client.compare_similarity(
            set=["alpha", "beta"], fast=False, flatten=False
        )
    except Exception as exc:
        pytest.skip(f"Skipping E2E compare_similarity: {exc}")
    assert hasattr(resp, "similarity"), "Response has no similarity field"
    assert isinstance(resp.similarity, list)
    assert all(isinstance(row, list) for row in resp.similarity)


@pytest.mark.vcr()
def test_generate_themes_e2e():
    client = CoreClient(base_url="https://dev.core.researchwiseai.com/pulse/v1")

    resp = client.generate_themes(
        ["alpha", "beta"], min_themes=1, max_themes=3, fast=False
    )

    assert hasattr(resp, "themes"), "Response has no themes field"
    assert isinstance(resp.themes, list)
    # requestId present or None for placeholder
    assert hasattr(resp, "requestId")
    assert resp.requestId is None or isinstance(resp.requestId, str)


@pytest.mark.vcr()
def test_analyze_sentiment_e2e():
    client = CoreClient(base_url="https://dev.core.researchwiseai.com/pulse/v1")

    resp = client.analyze_sentiment(["happy", "sad"], fast=False)

    assert hasattr(resp, "sentiments"), "Response has no sentiments field"
    assert isinstance(resp.sentiments, list)


# @pytest.mark.vcr()
# def test_extract_elements_e2e():
#     client = CoreClient(base_url="https://dev.core.researchwiseai.com/pulse/v1")

#     resp = client.extract_elements(
#         [
#             "The food was tasty and the service was great",
#             "The waiter never smiled, otherwise it was fine"
#         ],
#         ["Food", "Service", "Drinks"],
#         fast=True
#     )

#     assert hasattr(resp, "extractions"), "Response has no extractions field"
#     assert isinstance(resp.extractions, list)
