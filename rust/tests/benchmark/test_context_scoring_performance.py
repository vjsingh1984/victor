"""Benchmark: batch_score_messages Rust vs Python."""

import random
import time
from typing import List, Tuple
import pytest
from victor.processing.native._base import _NATIVE_AVAILABLE


def _python_score(priorities, timestamps):
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


def _gen(n, seed=42):
    rng = random.Random(seed)
    now = time.time()
    return ([rng.choice([0, 25, 50, 75, 100]) for _ in range(n)],
            [now - rng.uniform(0, 86400) for _ in range(n)])


class TestBatchScoreCorrectness:
    def test_empty(self):
        from victor.processing.native.context_fitter import batch_score_messages
        assert batch_score_messages([], []) == []

    def test_single(self):
        from victor.processing.native.context_fitter import batch_score_messages
        r = batch_score_messages([50], [1000.0])
        assert len(r) == 1 and r[0][0] == 0 and abs(r[0][1] - 0.8) < 0.01

    def test_correctness_100(self):
        from victor.processing.native.context_fitter import batch_score_messages
        p, t = _gen(100)
        rust = batch_score_messages(p, t)
        py = _python_score(p, t)
        for (ri, rs), (pi, ps) in zip(rust, py):
            assert ri == pi and abs(rs - ps) < 0.01

    def test_sorted_desc(self):
        from victor.processing.native.context_fitter import batch_score_messages
        r = batch_score_messages(*_gen(50))
        scores = [s for _, s in r]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    def test_max_age_zero(self):
        from victor.processing.native.context_fitter import batch_score_messages
        now = time.time()
        r = batch_score_messages([50, 75, 100], [now, now, now])
        assert r[0][0] == 2

    def test_priority_weight(self):
        from victor.processing.native.context_fitter import batch_score_messages
        now = time.time()
        assert batch_score_messages([25, 100], [now, now])[0][0] == 1

    def test_recency_weight(self):
        from victor.processing.native.context_fitter import batch_score_messages
        now = time.time()
        assert batch_score_messages([50, 50], [now - 3600, now])[0][0] == 1


def _time_fn(func, *args, n=10):
    times = []
    for _ in range(n):
        s = time.perf_counter()
        func(*args)
        times.append((time.perf_counter() - s) * 1000)
    return sum(times) / len(times)


class TestBatchScorePerformance:
    @pytest.mark.skipif(not _NATIVE_AVAILABLE, reason="No Rust")
    def test_1k(self):
        from victor.processing.native.context_fitter import batch_score_messages
        p, t = _gen(1000)
        py = _time_fn(_python_score, p, t)
        rs = _time_fn(batch_score_messages, p, t)
        sp = py / rs if rs > 0 else float("inf")
        print(f"\n1K: Py={py:.2f}ms Rust={rs:.2f}ms {sp:.1f}x")
        assert sp >= 1.5

    @pytest.mark.skipif(not _NATIVE_AVAILABLE, reason="No Rust")
    def test_10k(self):
        from victor.processing.native.context_fitter import batch_score_messages
        p, t = _gen(10000)
        py = _time_fn(_python_score, p, t)
        rs = _time_fn(batch_score_messages, p, t)
        sp = py / rs if rs > 0 else float("inf")
        print(f"\n10K: Py={py:.2f}ms Rust={rs:.2f}ms {sp:.1f}x")
        assert sp >= 2.0
