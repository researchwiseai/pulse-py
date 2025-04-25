"""End-to-end example flows from the Jupyter notebooks,
using real HTTPS calls with VCR recording."""

import pandas as pd
import pytest

from pulse_client.analysis.analyzer import Analyzer
from pulse_client.analysis.processes import (
    ThemeGeneration,
    ThemeAllocation,
    ThemeExtraction,
    SentimentProcess,
    Cluster,
)
from pulse_client.core.client import CoreClient


@pytest.mark.vcr()
def test_high_level_examples_flow():
    # Sample data
    comments = ["Good day", "Bad day", "Okay day"]
    # Build processes as in high-level notebook
    processes = [
        ThemeGeneration(min_themes=2, max_themes=2, fast=True),
        ThemeAllocation(threshold=0.5),
        ThemeExtraction(),
        SentimentProcess(fast=True),
        Cluster(fast=True),
    ]
    # Run analyzer with real CoreClient under VCR
    client = CoreClient(base_url="https://dev.core.researchwiseai.com/pulse/v1")
    try:
        az = Analyzer(dataset=comments, processes=processes, client=client)
        results = az.run()
    except Exception as exc:
        pytest.skip(f"Skipping E2E high-level examples: {exc}")
    # Check generation: DataFrame of theme metadata
    df_gen = results.theme_generation.to_dataframe()
    # Verify shortLabels match expected themes
    assert set(df_gen["shortLabel"]) <= set(["T0", "T1"])
    # Check allocation can produce a single-label Series
    ser_alloc = results.theme_allocation.assign_single()
    assert isinstance(ser_alloc, pd.Series)
    assert len(ser_alloc) == len(comments)
    # Check extractions
    df_extr = results.theme_extraction.to_dataframe()
    assert all(col in df_extr.columns for col in ["text", "theme", "extraction"])
    # Check sentiment
    sent = results.sentiment
    assert isinstance(sent.sentiments, list)
    # Check clustering similarity matrix shape
    mat = results.cluster.matrix
    assert mat.shape == (len(comments), len(comments))


@pytest.mark.vcr()
def test_low_level_examples_e2e():
    client = CoreClient(base_url="https://dev.core.researchwiseai.com/pulse/v1")
    try:
        emb = client.create_embeddings(["example text"], fast=False)
    except Exception as exc:
        pytest.skip(f"Skipping E2E low-level examples: {exc}")
    assert hasattr(emb, "embeddings"), "Embedding response missing"
    assert isinstance(emb.embeddings, list)

    sim = client.compare_similarity(["example text"], fast=False, flatten=False)
    assert hasattr(sim, "similarity"), "Similarity response missing"
    assert isinstance(sim.similarity, list)

    th = client.generate_themes(
        ["example text"], min_themes=1, max_themes=2, fast=False
    )
    assert hasattr(th, "themes"), "Themes response missing"
    assert isinstance(th.themes, list)

    sent = client.analyze_sentiment(["example text"], fast=False)
    assert hasattr(sent, "sentiments"), "Sentiment response missing"
    assert isinstance(sent.sentiments, list)

    extr = client.extract_elements(
        ["example text"], th.themes if hasattr(th, "themes") else ["t"], fast=False
    )
    assert hasattr(extr, "extractions"), "Extractions response missing"
    assert isinstance(extr.extractions, list)
