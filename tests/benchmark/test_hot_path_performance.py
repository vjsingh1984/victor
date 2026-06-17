"""Benchmark tests for Rust hot path acceleration.

These benchmarks are non-blocking validation for the native tokenizer,
EmbeddingIndex, and context fitting paths. They skip automatically when
the native extension is unavailable.
"""

from __future__ import annotations

import random
import time
from typing import Callable

import pytest


def is_rust_available() -> bool:
    try:
        import victor_native

        return hasattr(victor_native, "count_tokens_fast")
    except ImportError:
        return False


def time_function(func: Callable, iterations: int = 100) -> float:
    """Average time in milliseconds per call."""
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1000


class TestTokenCountingPerformance:
    SAMPLE_TEXTS = [
        "Hello world",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "The quick brown fox jumps over the lazy dog. " * 50,
        "import os\nimport sys\n\n" + "\n".join(f"x_{i} = {i}" for i in range(200)),
    ]

    @pytest.mark.skipif(not is_rust_available(), reason="Rust extensions not available")
    def test_rust_count_tokens_fast(self):
        import victor_native

        for text in self.SAMPLE_TEXTS:
            result = victor_native.count_tokens_fast(text)
            assert result > 0
            assert isinstance(result, int)

    @pytest.mark.skipif(not is_rust_available(), reason="Rust extensions not available")
    def test_rust_vs_python_token_counting_speed(self):
        import victor_native

        text = "The quick brown fox jumps over the lazy dog. " * 100

        rust_time = time_function(lambda: victor_native.count_tokens_fast(text), iterations=1000)
        python_time = time_function(lambda: len(text.split()) * 13 // 10, iterations=1000)

        assert rust_time >= 0
        assert python_time >= 0

    @pytest.mark.skipif(not is_rust_available(), reason="Rust extensions not available")
    def test_batch_token_counting(self):
        import victor_native

        tokenizer = victor_native.BpeTokenizer(
            "test", [(b"hello", 0), (b"world", 1)], [("<|endoftext|>", 100)]
        )
        texts = [f"word_{i} " * 50 for i in range(100)]

        batch_time = time_function(lambda: tokenizer.count_tokens_batch(texts), iterations=10)
        seq_time = time_function(lambda: [tokenizer.count_tokens(t) for t in texts], iterations=10)

        assert batch_time >= 0
        assert seq_time >= 0


class TestEmbeddingSimilarityPerformance:
    DIM = 384
    N_TOOLS = 50

    @staticmethod
    def _random_vector(dim: int) -> list[float]:
        return [random.gauss(0, 1) for _ in range(dim)]

    @pytest.mark.skipif(not is_rust_available(), reason="Rust extensions not available")
    def test_embedding_index_query(self):
        import victor_native

        vectors = [self._random_vector(self.DIM) for _ in range(self.N_TOOLS)]
        labels = [f"tool_{i}" for i in range(self.N_TOOLS)]
        index = victor_native.EmbeddingIndex(vectors, labels)

        query = self._random_vector(self.DIM)
        results = index.query(query, k=10, threshold=0.0)

        assert len(results) == 10
        assert all(isinstance(result, tuple) and len(result) == 2 for result in results)

    @pytest.mark.skipif(not is_rust_available(), reason="Rust extensions not available")
    def test_embedding_index_vs_per_tool_loop(self):
        import numpy as np
        import victor_native

        vectors = [self._random_vector(self.DIM) for _ in range(self.N_TOOLS)]
        labels = [f"tool_{i}" for i in range(self.N_TOOLS)]
        index = victor_native.EmbeddingIndex(vectors, labels)

        np_vectors = [np.array(v, dtype=np.float32) for v in vectors]
        query_list = self._random_vector(self.DIM)
        query_np = np.array(query_list, dtype=np.float32)

        rust_time = time_function(
            lambda: index.query(query_list, k=10, threshold=0.0),
            iterations=500,
        )

        def numpy_loop():
            similarities = []
            for vector in np_vectors:
                norm_q = np.linalg.norm(query_np)
                norm_v = np.linalg.norm(vector)
                if norm_q > 0 and norm_v > 0:
                    score = float(np.dot(query_np, vector) / (norm_q * norm_v))
                else:
                    score = 0.0
                similarities.append(score)
            return sorted(enumerate(similarities), key=lambda item: -item[1])[:10]

        python_time = time_function(numpy_loop, iterations=500)

        assert rust_time >= 0
        assert python_time >= 0


class TestContextFittingPerformance:
    @pytest.mark.skipif(not is_rust_available(), reason="Rust extensions not available")
    def test_fit_context_smart(self):
        import victor_native

        messages = [
            victor_native.MessageSlot(
                index,
                random.randint(50, 500),
                random.randint(0, 100),
                role,
                index / 100,
            )
            for index, role in enumerate(["system"] + ["user", "assistant"] * 50)
        ]

        result = victor_native.fit_context(messages, budget=4000, strategy="smart")
        assert len(result.kept_indices) > 0
        assert result.total_tokens <= 4000

    @pytest.mark.skipif(not is_rust_available(), reason="Rust extensions not available")
    def test_truncate_message(self):
        import victor_native

        content = "\n".join(f"Line {index}: " + "x" * 80 for index in range(200))

        result = victor_native.truncate_message(content, 50, 5)
        assert "[...truncated" in result
        assert "Line 0:" in result
        assert "Line 199:" in result
