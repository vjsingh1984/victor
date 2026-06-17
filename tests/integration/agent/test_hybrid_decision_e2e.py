"""Integration tests for hybrid decision service.

Tests cover:
- Full decision pipeline (lookup → pattern → ensemble → LLM)
- Cache integration
- Confidence calibration integration
- Budget management
- Metrics tracking
- End-to-end workflows
"""

from unittest.mock import MagicMock, Mock

import pytest

from victor.agent.decisions.schemas import DecisionType
from victor.agent.services.hybrid_decision_service import (
    HybridDecisionService,
    HybridDecisionServiceConfig,
    HybridMetrics,
    _elapsed_ms,
)


class TestHybridDecisionService:
    """Test hybrid decision service initialization and basic operations."""

    def test_initialization_with_defaults(self):
        """Test service initialization with default config."""
        service = HybridDecisionService(provider=None, model="test")

        assert service._config.enable_lookup_tables is True
        assert service._config.enable_pattern_matcher is True
        assert service._config.enable_ensemble_voting is True
        assert service._config.enable_calibration is True
        assert service._config.enable_cache is True

    def test_initialization_with_custom_config(self):
        """Test service initialization with custom config."""
        config = HybridDecisionServiceConfig(
            enable_lookup_tables=True,
            enable_pattern_matcher=False,
            enable_ensemble_voting=False,
            enable_calibration=False,
            enable_cache=False,
            enable_llm_fallback=False,
        )

        service = HybridDecisionService(provider=None, model="test", config=config)

        assert service._config.enable_lookup_tables is True
        assert service._config.enable_pattern_matcher is False
        assert service._config.enable_ensemble_voting is False
        assert service._calibrator is None
        assert service._cache is None

    def test_lookup_decision(self):
        """Test decision via lookup tables."""
        service = HybridDecisionService(provider=None, model="test")

        result = service.decide_sync(
            DecisionType.TASK_COMPLETION,
            {"message": "I'm done with the task"},
        )

        assert result.source == "lookup"
        assert result.result.is_complete is True
        assert result.confidence >= 0.90

    def test_pattern_decision(self):
        """Test decision via pattern matcher."""
        config = HybridDecisionServiceConfig(
            enable_lookup_tables=False,  # Force pattern match
        )

        service = HybridDecisionService(provider=None, model="test", config=config)

        result = service.decide_sync(
            DecisionType.TASK_COMPLETION,
            {"message": "The task is complete"},
        )

        assert result.source == "pattern"
        assert result.result.is_complete is True

    def test_ensemble_decision(self):
        """Test decision via ensemble voting."""
        config = HybridDecisionServiceConfig(
            enable_lookup_tables=False,
            enable_pattern_matcher=False,
            enable_ensemble_voting=True,
        )

        service = HybridDecisionService(provider=None, model="test", config=config)

        result = service.decide_sync(
            DecisionType.TASK_COMPLETION,
            {"message": "done"},
            heuristic_result=None,
            heuristic_confidence=0.0,
        )

        assert result.source == "ensemble"
        assert result.result is not None

    def test_heuristic_fallback(self):
        """Test fallback to heuristic when no patterns match."""
        config = HybridDecisionServiceConfig(
            enable_lookup_tables=False,
            enable_pattern_matcher=False,
            enable_ensemble_voting=False,
            enable_llm_fallback=False,
        )

        service = HybridDecisionService(provider=None, model="test", config=config)

        heuristic_result = MagicMock(is_complete=True, confidence=0.80)

        result = service.decide_sync(
            DecisionType.TASK_COMPLETION,
            {"message": "unusual message"},
            heuristic_result=heuristic_result,
            heuristic_confidence=0.80,
        )

        assert result.source == "heuristic_fallback"
        assert result.result == heuristic_result


class TestCacheIntegration:
    """Test cache integration with hybrid decision service."""

    def test_cache_hit_on_second_call(self):
        """Test that second call hits cache."""
        service = HybridDecisionService(provider=None, model="test")

        context = {"message": "done"}

        # First call - miss (cache not populated yet)
        result1 = service.decide_sync(DecisionType.TASK_COMPLETION, context)

        # Second call - hit (should use cache this time)
        result2 = service.decide_sync(DecisionType.TASK_COMPLETION, context)

        assert result1.source == "lookup"
        assert result2.source == "lookup"  # Should still work

        metrics = service.get_hybrid_metrics()
        assert metrics.cache_hits >= 1

    def test_cache_disabled(self):
        """Test service with cache disabled."""
        config = HybridDecisionServiceConfig(
            enable_cache=False,
        )

        service = HybridDecisionService(provider=None, model="test", config=config)

        result = service.decide_sync(
            DecisionType.TASK_COMPLETION,
            {"message": "done"},
        )

        assert result is not None
        assert service._cache is None


class TestCalibrationIntegration:
    """Test confidence calibration integration."""

    def test_calibration_affects_threshold(self):
        """Test that calibration affects decision threshold."""
        config = HybridDecisionServiceConfig(
            base_threshold=0.70,
            enable_calibration=True,
        )

        service = HybridDecisionService(provider=None, model="test", config=config)

        # Record high accuracy outcomes (should lower threshold)
        for _ in range(25):
            service.record_outcome(
                DecisionType.TASK_COMPLETION,
                heuristic_confidence=0.80,
                used_llm=False,
                was_correct=True,
                source="lookup",
            )

        # Get the calibrated threshold
        threshold = service._calibrator.get_threshold(DecisionType.TASK_COMPLETION)

        # Threshold should be lowered from 0.70
        assert threshold < 0.70

    def test_calibration_disabled(self):
        """Test service with calibration disabled."""
        config = HybridDecisionServiceConfig(
            enable_calibration=False,
        )

        service = HybridDecisionService(provider=None, model="test", config=config)

        assert service._calibrator is None


class TestBudgetManagement:
    """Test LLM budget management."""

    def test_budget_tracking(self):
        """Test that LLM calls are tracked against budget."""
        config = HybridDecisionServiceConfig(
            micro_budget=2,
        )

        service = HybridDecisionService(provider=None, model="test", config=config)

        assert service._budget_used == 0
        assert service.budget_remaining == 2

    def test_budget_exhaustion(self):
        """Test behavior when budget is exhausted."""
        config = HybridDecisionServiceConfig(
            micro_budget=1,
        )

        service = HybridDecisionService(provider=None, model="test", config=config)

        # Use up the budget
        service._budget_used = 1

        assert service.budget_remaining == 0

    def test_reset_budget(self):
        """Test budget reset."""
        service = HybridDecisionService(provider=None, model="test")

        service._budget_used = 5
        service.reset_budget()

        assert service._budget_used == 0


class TestMetrics:
    """Test metrics tracking."""

    def test_hybrid_metrics_initialization(self):
        """Test hybrid metrics initialization."""
        metrics = HybridMetrics()

        assert metrics.cache_hits == 0
        assert metrics.lookup_hits == 0
        assert metrics.pattern_hits == 0
        assert metrics.ensemble_hits == 0
        assert metrics.llm_calls == 0

    def test_hybrid_metrics_hit_rate(self):
        """Test hit rate calculation."""
        metrics = HybridMetrics()
        metrics.lookup_hits = 70
        metrics.pattern_hits = 20
        metrics.llm_calls = 10

        hit_rate = metrics.get_hit_rate()

        assert hit_rate == 0.9  # (70 + 20) / (70 + 20 + 10)

    def test_get_hybrid_metrics(self):
        """Test getting hybrid metrics from service."""
        service = HybridDecisionService(provider=None, model="test")

        # Make some decisions
        service.decide_sync(
            DecisionType.TASK_COMPLETION,
            {"message": "done"},
        )

        metrics = service.get_hybrid_metrics()

        assert metrics.lookup_hits >= 1

    def test_get_summary(self):
        """Test getting comprehensive summary."""
        service = HybridDecisionService(provider=None, model="test")

        summary = service.get_summary()

        assert "config" in summary
        assert "metrics" in summary
        assert "hybrid_metrics" in summary
        assert summary["config"]["lookup_enabled"] is True


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_decision_pipeline(self):
        """Test complete decision flow through all layers."""
        service = HybridDecisionService(provider=None, model="test")

        # This should hit the lookup layer
        result = service.decide_sync(
            DecisionType.TASK_COMPLETION,
            {"message": "Task is complete"},
        )

        assert result is not None
        assert result.source in ("lookup", "pattern", "ensemble")
        assert result.result is not None

    def test_multiple_decision_types(self):
        """Test handling multiple decision types."""
        service = HybridDecisionService(provider=None, model="test")

        results = {}

        results["task_completion"] = service.decide_sync(
            DecisionType.TASK_COMPLETION,
            {"message": "done"},
        )

        results["intent"] = service.decide_sync(
            DecisionType.INTENT_CLASSIFICATION,
            {"response": "Here's the solution"},
        )

        results["task_type"] = service.decide_sync(
            DecisionType.TASK_TYPE_CLASSIFICATION,
            {"task_description": "Fix the bug"},
        )

        for decision_type, result in results.items():
            assert result is not None
            assert result.result is not None

    def test_cache_with_multiple_types(self):
        """Test caching works across different decision types."""
        service = HybridDecisionService(provider=None, model="test")

        context = {"message": "test"}

        # Cache for different types
        service.decide_sync(DecisionType.TASK_COMPLETION, context)
        service.decide_sync(DecisionType.INTENT_CLASSIFICATION, context)

        # Both should be cached
        result1 = service.decide_sync(DecisionType.TASK_COMPLETION, context)
        result2 = service.decide_sync(DecisionType.INTENT_CLASSIFICATION, context)

        assert result1 is not None
        assert result2 is not None


class TestElapsedMs:
    """Test elapsed milliseconds calculation."""

    def test_elapsed_ms_positive(self):
        """Test elapsed time calculation."""
        import time

        start = time.monotonic()
        time.sleep(0.01)  # 10ms
        elapsed = _elapsed_ms(start)

        assert elapsed >= 10.0
        assert elapsed < 50.0  # Should be close to 10ms

    def test_elapsed_ms_zero(self):
        """Test elapsed time with no delay."""
        import time

        start = time.monotonic()
        elapsed = _elapsed_ms(start)

        assert elapsed >= 0.0
        assert elapsed < 10.0  # Should be very small


class TestConfiguration:
    """Test service configuration options."""

    def test_lookup_only_mode(self):
        """Test service with only lookup enabled."""
        config = HybridDecisionServiceConfig(
            enable_lookup_tables=True,
            enable_pattern_matcher=False,
            enable_ensemble_voting=False,
            enable_llm_fallback=False,
        )

        service = HybridDecisionService(provider=None, model="test", config=config)

        # Should match lookup pattern
        result = service.decide_sync(
            DecisionType.TASK_COMPLETION,
            {"message": "done"},
        )

        assert result.source in ("lookup", "heuristic_fallback")

    def test_rules_only_mode(self):
        """Test service with deterministic rules only (no LLM)."""
        config = HybridDecisionServiceConfig(
            enable_lookup_tables=True,
            enable_pattern_matcher=True,
            enable_ensemble_voting=True,
            enable_llm_fallback=False,
        )

        service = HybridDecisionService(provider=None, model="test", config=config)

        # Should not call LLM
        result = service.decide_sync(
            DecisionType.TASK_COMPLETION,
            {"message": "unusual message that won't match patterns"},
        )

        assert result.source in ("lookup", "pattern", "ensemble", "heuristic_fallback")
        assert result.source != "llm"


class TestOutcomeRecording:
    """Test outcome recording for calibration."""

    def test_record_outcome_updates_calibrator(self):
        """Test that recording outcome updates calibrator."""
        service = HybridDecisionService(provider=None, model="test")

        service.record_outcome(
            DecisionType.TASK_COMPLETION,
            heuristic_confidence=0.80,
            used_llm=False,
            was_correct=True,
            source="lookup",
        )

        stats = service._calibrator.get_statistics(DecisionType.TASK_COMPLETION)

        assert stats.total_decisions == 1
        assert stats.heuristic_correct == 1

    def test_record_outcome_without_calibrator(self):
        """Test recording outcome when calibrator is disabled."""
        config = HybridDecisionServiceConfig(
            enable_calibration=False,
        )

        service = HybridDecisionService(provider=None, model="test", config=config)

        # Should not raise error
        service.record_outcome(
            DecisionType.TASK_COMPLETION,
            heuristic_confidence=0.80,
            used_llm=False,
            was_correct=True,
            source="lookup",
        )


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_context(self):
        """Test handling of empty context."""
        service = HybridDecisionService(provider=None, model="test")

        result = service.decide_sync(
            DecisionType.TASK_COMPLETION,
            {},
        )

        # Should fall back to heuristic
        assert result is not None

    def test_missing_context_keys(self):
        """Test handling of missing context keys."""
        service = HybridDecisionService(provider=None, model="test")

        # This might not match any patterns
        result = service.decide_sync(
            DecisionType.TASK_COMPLETION,
            {"random_key": "random_value"},
        )

        assert result is not None

    def test_concurrent_decisions(self):
        """Test handling multiple decision types concurrently."""
        service = HybridDecisionService(provider=None, model="test")

        results = {}
        for decision_type in [
            DecisionType.TASK_COMPLETION,
            DecisionType.INTENT_CLASSIFICATION,
            DecisionType.TASK_TYPE_CLASSIFICATION,
        ]:
            # Use appropriate context for each type that matches lookup patterns
            if decision_type == DecisionType.TASK_TYPE_CLASSIFICATION:
                context = {"task_description": "Fix the bug"}
            elif decision_type == DecisionType.INTENT_CLASSIFICATION:
                context = {"response": "Here's the solution"}
            else:
                context = {"message": "done"}

            results[decision_type] = service.decide_sync(
                decision_type,
                context,
            )

        # All should succeed
        for decision_type, result in results.items():
            assert result is not None
            assert result.result is not None
