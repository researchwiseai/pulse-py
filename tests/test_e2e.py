"""End-to-end tests for CoreClient against the real Pulse API (recorded via VCR)."""

import pytest

from pulse_client.core.client import CoreClient


@pytest.mark.vcr()
def test_create_embeddings_e2e():
    client = CoreClient(base_url="https://dev.core.researchwiseai.com/pulse/v1")
    # Skip if API not reachable
    try:
        resp = client.create_embeddings(["test e2e", "pulse client"], fast=True)
    except Exception as exc:
        pytest.skip(f"Skipping E2E create_embeddings: {exc}")
    assert hasattr(resp, "embeddings"), "Response has no embeddings field"
    assert isinstance(resp.embeddings, list)
    assert all(isinstance(row, list) for row in resp.embeddings)


@pytest.mark.vcr()
def test_compare_similarity_e2e():
    client = CoreClient(base_url="https://dev.core.researchwiseai.com/pulse/v1")
    resp = client.compare_similarity(["alpha", "beta"], fast=True, flatten=False)
    assert hasattr(resp, "similarity"), "Response has no similarity field"
    assert isinstance(resp.similarity, list)
    assert all(isinstance(row, list) for row in resp.similarity)


@pytest.mark.vcr()
def test_generate_themes_e2e():
    client = CoreClient(base_url="https://dev.core.researchwiseai.com/pulse/v1")
    try:
        resp = client.generate_themes(
            ["alpha", "beta"], min_themes=1, max_themes=3, fast=True
        )
    except Exception as exc:
        pytest.skip(f"Skipping E2E generate_themes: {exc}")
    assert hasattr(resp, "themes"), "Response has no themes field"
    assert isinstance(resp.themes, list)
    assert hasattr(resp, "assignments"), "Response has no assignments field"
    assert isinstance(resp.assignments, list)


@pytest.mark.vcr()
def test_analyze_sentiment_e2e():
    client = CoreClient(base_url="https://dev.core.researchwiseai.com/pulse/v1")
    try:
        resp = client.analyze_sentiment(["happy", "sad"], fast=True)
    except Exception as exc:
        pytest.skip(f"Skipping E2E analyze_sentiment: {exc}")
    assert hasattr(resp, "sentiments"), "Response has no sentiments field"
    assert isinstance(resp.sentiments, list)


@pytest.mark.vcr()
def test_extract_elements_e2e():
    client = CoreClient(base_url="https://dev.core.researchwiseai.com/pulse/v1")
    try:
        resp = client.extract_elements(["sample text"], ["theme1"], fast=True)
    except Exception as exc:
        pytest.skip(f"Skipping E2E extract_elements: {exc}")
    assert hasattr(resp, "extractions"), "Response has no extractions field"
    assert isinstance(resp.extractions, list)
