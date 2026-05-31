"""Benchmark test configuration.

Wall-clock performance assertions are intentionally serial-only.  Pytest's
benchmark plugin disables itself under xdist because timings are unreliable
when workers contend for CPU and importlib metadata caches.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip benchmark timing assertions in xdist worker processes."""
    if not (hasattr(config, "workerinput") or os.environ.get("PYTEST_XDIST_WORKER")):
        return

    skip_xdist = pytest.mark.skip(reason="benchmark timing tests run serially, not under xdist")
    for item in items:
        if "tests/benchmarks" in item.path.as_posix():
            item.add_marker(skip_xdist)
