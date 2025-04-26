"""Tests for persistent on-disk caching in Analyzer."""

# import os      # Unused
# from pathlib import Path  # Unused
# import shutil  # Unused

import pytest

from pulse.analysis.analyzer import Analyzer

# from pulse.analysis.processes import Process  # Unused


class DummyProcess:
    """A dummy process that counts how many times run() is called."""

    id = "dummy"
    depends_on = ()

    def __init__(self):
        # internal counter, excluded from cache key (_ prefix)
        self._call_count = 0

    @property
    def call_count(self):
        return self._call_count

    def run(self, ctx):  # type: ignore
        self._call_count += 1
        return f"result_{self._call_count}"


@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path / "cache")


def test_persistent_caching(tmp_path, cache_dir):
    # First run: process.run called once
    texts = ["x", "y"]
    proc = DummyProcess()
    az1 = Analyzer(dataset=texts, processes=[proc], cache_dir=cache_dir, use_cache=True)
    res1 = az1.run()
    assert proc.call_count == 1
    assert res1.dummy == "result_1"

    # Second run on the same Analyzer: should not call run again (cached)
    res2 = az1.run()
    assert proc.call_count == 1
    assert res2.dummy == "result_1"

    # After clearing cache, run() should be called again
    az1.clear_cache()
    res3 = az1.run()
    assert proc.call_count == 2
    assert res3.dummy == "result_2"
    az1.close()


def test_disable_caching(tmp_path, cache_dir):
    # When use_cache=False, run() is always called
    texts = ["a"]
    proc = DummyProcess()
    az = Analyzer(dataset=texts, processes=[proc], cache_dir=cache_dir, use_cache=False)
    az.run()
    az.run()
    # call_count should increment twice
    assert proc.call_count == 2
    az.close()
