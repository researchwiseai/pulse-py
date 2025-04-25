"""Tests for the DSL Workflow builder and execution."""

import json
import pandas as pd
import pytest  # noqa: F401 (imported for pytest fixtures, if needed)

from pulse_client.dsl import Workflow
from pulse_client.analysis.analyzer import AnalysisResult
from pulse_client.core.models import (
    ThemesResponse,
    SentimentResponse,
    SimilarityResponse,
    ExtractionsResponse,
)


class DummyDSLClient:
    """Stub client for DSL tests with predictable outputs."""

    def __init__(self):
        self.called = {}

    def generate_themes(self, texts, min_themes, max_themes, fast):
        self.called["generate_themes"] = dict(
            texts=list(texts), min_themes=min_themes, max_themes=max_themes, fast=fast
        )
        assignments = [0 for _ in texts]
        return ThemesResponse(themes=["T1", "T2"], assignments=assignments)

    def extract_elements(self, inputs, themes, version, fast):
        self.called["extract_elements"] = dict(
            inputs=list(inputs), themes=list(themes), version=version, fast=fast
        )
        extractions = []
        for txt in inputs:
            row = []
            for th in themes:
                row.append([f"{txt}-{th}"])
            extractions.append(row)
        return ExtractionsResponse(extractions=extractions)

    def analyze_sentiment(self, texts, fast):
        self.called["analyze_sentiment"] = dict(texts=list(texts), fast=fast)
        return SentimentResponse(sentiments=["neu" for _ in texts])

    def compare_similarity(self, texts, fast=True, flatten=True):
        self.called["compare_similarity"] = dict(
            texts=list(texts), fast=fast, flatten=flatten
        )
        size = len(texts)
        sim = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
        return SimilarityResponse(similarity=sim)


def test_workflow_graph_and_aliasing():
    wf = (
        Workflow()
        .theme_generation(min_themes=1, max_themes=2)
        .theme_extraction()
        .sentiment(fast=False)
        .sentiment(fast=True)
    )
    graph = wf.graph()
    assert set(graph.keys()) == {
        "theme_generation",
        "theme_extraction",
        "sentiment",
        "sentiment_2",
    }
    assert graph["theme_generation"] == []
    assert graph["theme_extraction"] == ["theme_generation"]
    assert graph["sentiment"] == []
    assert graph["sentiment_2"] == []


def test_workflow_run_and_results():
    texts = ["a", "b"]
    wf = (
        Workflow()
        .theme_generation(min_themes=3, max_themes=3, fast=False)
        .theme_extraction()
        .sentiment(fast=True)
        .cluster()
    )
    client = DummyDSLClient()
    results = wf.run(texts, client=client)
    assert isinstance(results, AnalysisResult)
    tg = results.theme_generation
    assert tg.themes == ["T1", "T2"]
    te = results.theme_extraction
    ex_list = te.extractions
    assert len(ex_list) == len(texts)
    df = te.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    sent = results.sentiment
    assert sent.sentiments == ["neu", "neu"]
    cl = results.cluster
    mat = cl.matrix
    assert mat.shape == (2, 2)


def test_from_file_json(tmp_path):
    config = {
        "pipeline": [
            {"theme_generation": {"min_themes": 1, "max_themes": 1, "fast": True}},
            {"sentiment": {"fast": False}},
        ]
    }
    file_path = tmp_path / "pipe.json"
    file_path.write_text(json.dumps(config))
    wf = Workflow.from_file(str(file_path))
    graph = wf.graph()
    assert set(graph.keys()) == {"theme_generation", "sentiment"}
    client = DummyDSLClient()
    results = wf.run(["x"], client=client)
    assert hasattr(results, "theme_generation")
    assert hasattr(results, "sentiment")
