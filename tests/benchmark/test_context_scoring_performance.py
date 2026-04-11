"""Benchmark tests for batch_score_messages: Rust vs Python context scoring.

Scoring formula: score = 0.4 * (priority / 100) + 0.6 * (1 - age / max_age)
"""

from __future__ import annotations

import random
import time
from typing import List, Tuple

import pytest

from victor.processing.native._base import _NATIVE_AVAILABLE


def _python_score_messages(
    priorities: List[int], timestamps: List[float]
) -> List[Tuple[int, float]]:
    """Pure Python scoring reference."""
    if not priorities:
        return []
    max_ts = max(timestamps)
    max_age = max(max_ts - t for t in timestamps) or 1e-9
    scored = []
    for i, (pri, ts) in enumerate(zip(priorities, timestamps)):
        score = (pri / 100.0) * 0.4 + (1.0 - ((max_ts - ts) / max_age)) * 0.6
        scored.append((i, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _generate_test_data(n, seed=42):
    rng = random.Random(seed)
    now = time.time()
    priorities = [rng.choice([0, 25, 50, 75, 100]) for _ in range(n)]
    timestamps = [now - rng.uniform(0, 86400) for _ in range(n)]
    return priorities, timestamps


class TestBatchScoreCorrectness:

    def test_empty_list(self):
        from victor.processing.native.context_fitter import batch_score_messages
        assert batch_score_messages([], []) == []

    def test_single_message(self):
        from victor.processing.native.context_fitter import batch_score_messages
        result = batch_score_messages([50], [1000.0])
        assert len(result) == 1
        assert result[0][0] == 0
        assert abs(result[0][1] - 0.8) < 0.01

    def test_correctness_vs_python_100(self):
        from victor.processing.native.context_fitter import batch_score_messages
        priorities, timestamps = _generate_test_data(100)
        rust_result = batch_score_messages(priorities, timestamps)
        python_result = _python_score_messages(priorities, timestamps)
        assert len(rust_result) == len(python_result)
        for (ri, rs), (pi, ps) in zip(rust_result, python_result):
            assert ri == pi
            assert abs(rs - ps) < 0.01

    def test_sorted_descending(self):
        from victor.processing.native.context_fitter import batch_score_messages
        priorities, timestamps = _generate_test_data(50)
        result = batch_score_messages(priorities, timestamps)
        scores = [s for _, s in result]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_max_age_zero(self):
        from victor.processing.native.context_fitter import batch_score_messages
        now = time.time()
        result = batch_score_messages([50, 75, 100], [now, now, now])
        assert len(result) == 3
        assert result[0][0] == 2  # priority=100 first

    def test_priority_weight(self):
        from victor.processing.native.context_fitter import batch_score_messages
        now = time.time()
        result = batch_score_messages([25, 100], [now, now])
        assert result[0][0] == 1

    def test_recency_weight(self):
        from victor.processing.native.context_fitter import batch_score_messages
        now = time.time()
        result = batch_score_messages([50, 50], [now - 3600, now])
        assert result[0][0] == 1


def _time_fn(func, *args, iterations=10):
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        times.append((time.perf_counter() - start) * 1000)
    return sum(times) / len(times)


class TestBatchScorePerformance:

    @pytest.mark.skipif(not _NATIVE_AVAILABLE, reason="Rust native not available")
    def test_benchmark_1k(self):
        from victor.processing.native.context_fitter import batch_score_messages
        p, t = _generate_test_data(1000)
        py_ms = _time_fn(_python_score_messages, p, t)
        rs_ms = _time_fn(batch_score_messages, p, t)
        speedup = py_ms / rs_ms if rs_ms > 0 else float("inf")
        print(f"\n1K: Python={py_ms:.2f}ms Rust={rs_ms:.2f}ms Speedup={speedup:.1f}x")
        assert speedup >= 1.5

    @pytest.mark.skipif(not _NATIVE_AVAILABLE, reason="Rust native not available")
    def test_benchmark_10k(self):
        from victor.processing.native.context_fitter import batch_score_messages
        p, t = _generate_test_data(10000)
        py_ms = _time_fn(_python_score_messages, p, t)
        rs_ms = _time_fn(batch_score_messages, p, t)
        speedup = py_ms / rs_ms if rs_ms > 0 else float("inf")
        print(f"\n10K: Python={py_ms:.2f}ms Rust={rs_ms:.2f}ms Speedup={speedup:.1f}x")
        assert speedup >= 2.0
