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
