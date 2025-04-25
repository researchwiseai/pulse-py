"""Tests for Analyzer and built-in processes."""

import pytest

# from pydantic import BaseModel  # Unused import removed

from pulse_client.analysis.analyzer import Analyzer, AnalysisResult
from pulse_client.analysis.processes import ThemeGeneration, SentimentProcess
from pulse_client.core.models import ThemesResponse


class DummyClient:
    """Stub CoreClient with predictable responses."""

    def __init__(self):
        self.called = {}

    def generate_themes(self, texts, min_themes, max_themes, fast):
        from pulse_client.core.models import Theme

        self.called["generate_themes"] = dict(
            texts=texts, min_themes=min_themes, max_themes=max_themes, fast=fast
        )
        # Return spec-based ThemesResponse with Theme objects
        themeA = Theme(
            shortLabel="A",
            label="Label A",
            description="Desc A",
            representatives=["rA1", "rA2"],
        )
        themeB = Theme(
            shortLabel="B",
            label="Label B",
            description="Desc B",
            representatives=["rB1", "rB2"],
        )
        return ThemesResponse(themes=[themeA, themeB], requestId=None)

    def analyze_sentiment(self, texts, fast):
        from pulse_client.core.models import (
            SentimentResult as CoreSentimentResult,
            SentimentResponse as CoreSentimentResponse,
        )

        self.called["analyze_sentiment"] = dict(texts=texts, fast=fast)
        # return spec-based sentiment results
        results = [
            CoreSentimentResult(sentiment="positive", confidence=0.9),
            CoreSentimentResult(sentiment="negative", confidence=0.8),
        ]
        return CoreSentimentResponse(results=results, requestId=None)


def test_analyzer_no_processes():
    az = Analyzer(dataset=["x", "y"], processes=[], client=DummyClient())
    res = az.run()
    assert isinstance(res, AnalysisResult)
    with pytest.raises(AttributeError):
        _ = res.theme_generation


def test_theme_generation_process():
    client = DummyClient()
    proc = ThemeGeneration(min_themes=3, max_themes=5, fast=False)
    az = Analyzer(dataset=["t1", "t2"], processes=[proc], fast=True, client=client)
    res = az.run()
    # check that generate_themes was called with fast override False
    assert client.called["generate_themes"] == {
        "texts": ["t1", "t2"],
        "min_themes": 3,
        "max_themes": 5,
        "fast": False,
    }
    # result attribute name matches process id
    tg = res.theme_generation
    from pulse_client.analysis.results import ThemeGenerationResult

    assert isinstance(tg, ThemeGenerationResult)
    # shortLabels should match dummy
    assert tg.themes == ["A", "B"]


def test_sentiment_process():
    client = DummyClient()
    proc = SentimentProcess(fast=True)
    az = Analyzer(dataset=["s1", "s2", "s3"], processes=[proc], client=client)
    res = az.run()
    assert client.called["analyze_sentiment"] == {
        "texts": ["s1", "s2", "s3"],
        "fast": True,
    }
    sent = res.sentiment
    from pulse_client.analysis.results import SentimentResult as AnalysisSentimentResult

    assert isinstance(sent, AnalysisSentimentResult)
    # expect spec-based sentiment labels
    assert sent.sentiments == ["positive", "negative"]
