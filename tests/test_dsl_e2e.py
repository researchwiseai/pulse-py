"""End-to-end DSL workflow test with a dummy client to simulate all steps."""

import pytest

from pulse.dsl import Workflow
from pulse.core.jobs import Job
from pulse.core.client import CoreClient


@pytest.fixture(autouse=True)
def disable_sleep(monkeypatch):
    import time

    monkeypatch.setattr(time, "sleep", lambda x: None)


@pytest.mark.vcr()
def test_dsl_end_to_end():
    comments = ["everything was very tasty", "it was a little too noisy"]
    existing = ["Food Quality", "Service", "Environment"]
    wf = (
        Workflow()
        .source("comments", comments)
        .source("themes", existing)
        .theme_allocation(inputs="comments", themes_from="themes")
    )
    client = CoreClient()
    results = wf.run(client=client)
    # verify results types and not raw Job
    for step in [
        "theme_allocation",
    ]:
        assert hasattr(results, step), f"Missing DSL step: {step}"
        res = getattr(results, step)
        assert not isinstance(res, Job), f"Step {step} returned raw Job"
        # check minimal behavior
        if hasattr(res, "themes"):
            assert isinstance(res.themes, list)
        if hasattr(res, "sentiments"):
            assert isinstance(res.sentiments, list)
