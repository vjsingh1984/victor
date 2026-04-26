"""Tests for KNNFewShot input-aware few-shot selection (DSPy-inspired)."""

from unittest.mock import MagicMock


class TestKNNFewShotSelection:
    def _make_trace(self, task="coding", tools=None, failures=None, score=0.8):
        t = MagicMock()
        t.task_type = task
        t.success = True
        t.completion_score = score
        t.tool_calls = 5
        t.tool_failures = failures or {}
        details = []
        for tool in tools or ["read"]:
            d = MagicMock()
            d.tool_name = tool
            details.append(d)
        t.tool_call_details = details
        return t

    def test_method_exists(self):
        from victor.framework.rl.learners.strategies.miprov2_strategy import MIPROv2Strategy

        s = MIPROv2Strategy()
        assert hasattr(s, "select_similar_traces")

    def test_select_returns_top_k(self):
        from victor.framework.rl.learners.strategies.miprov2_strategy import MIPROv2Strategy

        s = MIPROv2Strategy(max_examples=2)
        traces = [self._make_trace() for _ in range(5)]
        # Without embeddings available, falls back to top-scoring
        result = s.select_similar_traces(traces, "fix the bug", top_k=2)
        assert len(result) <= 2

    def test_fallback_without_embeddings(self):
        from victor.framework.rl.learners.strategies.miprov2_strategy import MIPROv2Strategy

        s = MIPROv2Strategy()
        traces = [self._make_trace() for _ in range(3)]
        # Should return traces without error even if embeddings unavailable
        result = s.select_similar_traces(traces, "query", top_k=2)
        assert len(result) > 0

    def test_reflect_accepts_query_kwarg(self):
        from victor.framework.rl.learners.strategies.miprov2_strategy import MIPROv2Strategy

        s = MIPROv2Strategy()
        traces = [self._make_trace(score=0.9)]
        # Should not raise with query parameter
        result = s.reflect(traces, "FEW_SHOT_EXAMPLES", "current text", query="fix auth bug")
        assert isinstance(result, str)

    def test_reflect_honors_min_completion_score(self):
        from victor.framework.rl.learners.strategies.miprov2_strategy import MIPROv2Strategy

        s = MIPROv2Strategy(max_examples=3, min_completion_score=0.8)
        traces = [
            self._make_trace(task="coding", tools=["read"], score=0.95),
            self._make_trace(task="coding", tools=["edit"], score=0.65),
        ]

        result = s.reflect(traces, "FEW_SHOT_EXAMPLES", "current text")

        assert "score=0.9" in result
        assert "score=0.7" not in result

    def test_reflect_deduplicates_examples_when_diversity_enabled(self):
        from victor.framework.rl.learners.strategies.miprov2_strategy import MIPROv2Strategy

        s = MIPROv2Strategy(max_examples=3, example_diversity=True)
        traces = [
            self._make_trace(task="coding", tools=["read", "edit"], score=0.95),
            self._make_trace(task="coding", tools=["read", "edit"], score=0.9),
            self._make_trace(task="debug", tools=["search", "read"], score=0.85),
        ]

        result = s.reflect(traces, "FEW_SHOT_EXAMPLES", "current text")

        assert result.count("Example ") == 2

    def test_reflect_truncates_example_block(self):
        from victor.framework.rl.learners.strategies.miprov2_strategy import MIPROv2Strategy

        s = MIPROv2Strategy(max_examples=3, max_example_chars=50)
        traces = [
            self._make_trace(task="coding", tools=["read", "edit", "search", "shell"], score=0.95),
            self._make_trace(task="debug", tools=["grep", "read", "edit"], score=0.9),
        ]

        result = s.reflect(traces, "FEW_SHOT_EXAMPLES", "current text")

        block = result.split("--- Few-shot demonstrations ---\n", 1)[1]
        assert len(block) <= 50
