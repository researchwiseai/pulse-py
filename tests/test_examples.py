"""End-to-end example flows from the Jupyter notebooks, using stub clients."""

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
from pulse_client.core.models import (
    ThemesResponse,
    SentimentResponse,
    SimilarityResponse,
    ExtractionsResponse,
)


class DummyHighLevelClient:
    """Stub for high-level example: returns predictable responses."""

    def generate_themes(self, texts, min_themes, max_themes, fast):
        # return two themes, one assignment per text
        themes = [f"T{i}" for i in range(2)]
        assignments = [i % 2 for i in range(len(texts))]
        return ThemesResponse(themes=themes, assignments=assignments)

    def analyze_sentiment(self, texts, fast):
        # assign 'pos' or 'neg' alternately
        sentiments = ["pos" if i % 2 == 0 else "neg" for i in range(len(texts))]
        return SentimentResponse(sentiments=sentiments)

    def compare_similarity(self, texts, fast=True, flatten=False):
        # identity matrix
        n = len(texts)
        sim = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        return SimilarityResponse(similarity=sim)

    def extract_elements(self, inputs, themes, version=None, fast=True):
        # one extraction per text/theme pair
        n, m = len(inputs), len(themes)
        extractions = [
            [[f"{inputs[i]}_{themes[j]}"] for j in range(m)] for i in range(n)
        ]
        return ExtractionsResponse(extractions=extractions)


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
    # Run analyzer with DummyHighLevelClient
    az = Analyzer(dataset=comments, processes=processes, client=DummyHighLevelClient())
    results = az.run()
    # Check generation
    df_gen = results.theme_generation.to_dataframe()
    assert set(df_gen["theme"]) <= set(["T0", "T1"])
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
        emb = client.create_embeddings(["example text"], fast=True)
    except Exception as exc:
        pytest.skip(f"Skipping E2E low-level examples: {exc}")
    assert hasattr(emb, "embeddings"), "Embedding response missing"
    assert isinstance(emb.embeddings, list)

    sim = client.compare_similarity(["example text"], fast=True, flatten=False)
    assert hasattr(sim, "similarity"), "Similarity response missing"
    assert isinstance(sim.similarity, list)

    th = client.generate_themes(["example text"], min_themes=1, max_themes=2, fast=True)
    assert hasattr(th, "themes"), "Themes response missing"
    assert isinstance(th.themes, list)

    sent = client.analyze_sentiment(["example text"], fast=True)
    assert hasattr(sent, "sentiments"), "Sentiment response missing"
    assert isinstance(sent.sentiments, list)

    extr = client.extract_elements(
        ["example text"], th.themes if hasattr(th, "themes") else ["t"], fast=True
    )
    assert hasattr(extr, "extractions"), "Extractions response missing"
    assert isinstance(extr.extractions, list)
