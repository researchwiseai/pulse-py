"""End-to-end DSL workflow test with a dummy client to simulate all steps."""

from pulse_client.dsl import Workflow
from pulse_client.core.jobs import Job
from pulse_client.core.models import (
    Theme,
    ThemesResponse,
    SentimentResponse as CoreSentimentResponse,
    SentimentResult as CoreSentimentResult,
    SimilarityResponse,
    ExtractionsResponse,
)


class DummyDSLClient:
    """Stub CoreClient for DSL end-to-end test."""

    def generate_themes(self, texts, min_themes, max_themes, fast):
        # Return spec-based Theme objects
        th1 = Theme(
            shortLabel="T1",
            label="Label T1",
            description="Desc T1",
            representatives=["r1", "r2"],
        )
        th2 = Theme(
            shortLabel="T2",
            label="Label T2",
            description="Desc T2",
            representatives=["r3", "r4"],
        )
        return ThemesResponse(themes=[th1, th2], requestId=None)

    def theme_allocation(self, inputs, themes, fast, flatten=False):
        # Not used directly by DSL, allocation uses analyzer + processes
        pass

    def compare_similarity(self, texts, fast=True, flatten=True):
        size = len(texts)
        sim = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
        return SimilarityResponse(similarity=sim)

    def extract_elements(self, inputs, themes, version=None, fast=True):
        extractions = [[[f"{t}-{th}" for th in themes] for t in inputs]]
        # reshape: one row per text
        # for simplicity, return at least correct structure
        return ExtractionsResponse(extractions=extractions)

    def analyze_sentiment(self, texts, fast):
        # Return spec-based sentiment "positive" for each
        results = [
            CoreSentimentResult(sentiment="positive", confidence=1.0) for _ in texts
        ]
        return CoreSentimentResponse(results=results, requestId=None)


def test_dsl_end_to_end_with_dummy():
    comments = ["a", "b"]
    existing = ["X", "Y"]
    wf = (
        Workflow()
        .source("comments", comments)
        .source("themes", existing)
        .theme_generation(min_themes=2, max_themes=3, fast=True, source="comments")
        .theme_allocation(inputs="comments", themes_from="theme_generation")
        .theme_extraction(inputs="comments", themes_from="theme_generation")
        .sentiment(source="comments", fast=True)
        .sentiment(source="theme_extraction", fast=True)
        .cluster(source="comments", fast=True)
    )
    client = DummyDSLClient()
    results = wf.run(client=client)
    # verify results types and not raw Job
    for step in [
        "theme_generation",
        "theme_allocation",
        "theme_extraction",
        "sentiment",
        "sentiment_2",
        "cluster",
    ]:
        assert hasattr(results, step), f"Missing DSL step: {step}"
        res = getattr(results, step)
        assert not isinstance(res, Job), f"Step {step} returned raw Job"
        # check minimal behavior
        if hasattr(res, "themes"):
            assert isinstance(res.themes, list)
        if hasattr(res, "sentiments"):
            assert isinstance(res.sentiments, list)
