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
