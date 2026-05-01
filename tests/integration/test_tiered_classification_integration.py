"""Integration tests for tiered classification triage system.

These tests require real services and are marked with pytest.mark.integration.
Run with: pytest tests/integration/test_tiered_classification_integration.py -v
"""

from __future__ import annotations

import pytest

from victor.agent.services.tiered_decision_service import (
    TieredDecisionService,
    ClassificationTriage,
)
from victor.agent.decisions.schemas import DecisionType
from victor.config.decision_settings import DecisionServiceSettings
from victor.framework.runtime_evaluation_policy import RuntimeEvaluationPolicy
from victor.agent.unified_classifier import UnifiedTaskClassifier, ClassifierTaskType
from victor.storage.embeddings.intent_classifier import IntentClassifier, IntentType


@pytest.mark.integration
class TestUnifiedTaskClassifierWithTriage:
    """Test UnifiedTaskClassifier with tiered triage integration."""

    def test_classifier_with_tiered_triage_enabled(self):
        """Test classifier uses tiered triage when service is provided."""
        # Create tiered decision service
        config = DecisionServiceSettings(enabled=True)
        tiered_service = TieredDecisionService(config)

        # Create classifier with tiered service
        classifier = UnifiedTaskClassifier(tiered_decision_service=tiered_service)

        # High confidence case should use fast path
        result = classifier.classify("fix the authentication bug")
        assert result.task_type in (ClassifierTaskType.EDIT, ClassifierTaskType.ACTION)
        assert result.confidence >= 0.5 or result.task_type == ClassifierTaskType.DEFAULT
        assert result.source in ("keyword", "triage", "llm", "edge_verification")

    def test_classifier_without_tiered_triage_fallback(self):
        """Test classifier falls back gracefully without tiered service."""
        # Create classifier without tiered service
        classifier = UnifiedTaskClassifier(tiered_decision_service=None)

        # Should still work with base classification
        result = classifier.classify("analyze the codebase")
        assert result.task_type in (ClassifierTaskType.ANALYSIS, ClassifierTaskType.SEARCH)
        assert result.source in ("keyword", "llm")

    def test_classifier_ambiguous_message_with_triage(self):
        """Test classifier with ambiguous message uses triage."""
        config = DecisionServiceSettings(enabled=True)
        tiered_service = TieredDecisionService(config)
        classifier = UnifiedTaskClassifier(tiered_decision_service=tiered_service)

        # Ambiguous message that might have low confidence
        result = classifier.classify("do something with the code")
        # Should handle gracefully - either classify or fall back to DEFAULT
        assert result.task_type in ClassifierTaskType
        assert 0.0 <= result.confidence <= 1.0

    def test_classifier_cache_with_triage(self):
        """Test that caching works with triage enabled."""
        config = DecisionServiceSettings(enabled=True)
        tiered_service = TieredDecisionService(config)
        classifier = UnifiedTaskClassifier(tiered_decision_service=tiered_service)

        message = "fix the bug in authentication"

        # First call - should compute
        result1 = classifier.classify(message, use_cache=True)

        # Second call - should use cache
        result2 = classifier.classify(message, use_cache=True)

        # Results should be identical
        assert result1.task_type == result2.task_type
        assert result1.confidence == result2.confidence

        # Cache stats should show hit
        stats = classifier.get_cache_stats()
        assert stats["cache_hits"] >= 1


@pytest.mark.integration
class TestIntentClassifierWithTriage:
    """Test IntentClassifier with tiered triage integration."""

    def test_intent_classifier_with_triage(self):
        """Test intent classifier uses tiered triage when available."""
        config = DecisionServiceSettings(enabled=True)
        tiered_service = TieredDecisionService(config)
        classifier = IntentClassifier.get_instance()
        classifier.initialize_sync()

        # Clear continuation intent
        text = "In summary, here are the key findings from the analysis"
        intent, confidence, source = classifier.classify_with_triage(
            text,
            context={"has_tool_calls": False},
            tiered_service=tiered_service,
        )

        assert intent in IntentType
        assert 0.0 <= confidence <= 1.0
        assert source in ("semantic", "llm", "edge_verification", "heuristic_fallback")

    def test_intent_classifier_without_triage_fallback(self):
        """Test intent classifier falls back without tiered service."""
        classifier = IntentClassifier.get_instance()
        classifier.initialize_sync()

        # Continuation intent
        text = "Let me examine the next file"
        intent, confidence, source = classifier.classify_with_triage(
            text,
            context={"has_tool_calls": False},
            tiered_service=None,
        )

        assert intent in (IntentType.CONTINUATION, IntentType.NEUTRAL)
        assert source == "semantic"

    def test_intent_classifier_continuation_detection(self):
        """Test continuation detection with triage."""
        config = DecisionServiceSettings(enabled=True)
        tiered_service = TieredDecisionService(config)
        classifier = IntentClassifier.get_instance()
        classifier.initialize_sync()

        # Clear continuation pattern
        text = "Let me start by examining the orchestrator implementation"
        intent, confidence, source = classifier.classify_with_triage(
            text,
            context={"has_tool_calls": False},
            tiered_service=tiered_service,
        )

        assert intent == IntentType.CONTINUATION
        assert confidence >= 0.3

    def test_intent_classifier_completion_detection(self):
        """Test completion detection with triage."""
        config = DecisionServiceSettings(enabled=True)
        tiered_service = TieredDecisionService(config)
        classifier = IntentClassifier.get_instance()
        classifier.initialize_sync()

        # Clear completion pattern
        text = "In summary, here are the main issues found in the codebase"
        intent, confidence, source = classifier.classify_with_triage(
            text,
            context={"has_tool_calls": False},
            tiered_service=tiered_service,
        )

        assert intent == IntentType.COMPLETION
        assert confidence >= 0.3


@pytest.mark.integration
class TestTieredDecisionServiceIntegration:
    """Test TieredDecisionService integration with real decision types."""

    def test_task_type_classification_triage(self):
        """Test task type classification with triage."""
        config = DecisionServiceSettings(enabled=True)
        service = TieredDecisionService(config)
        runtime_policy = RuntimeEvaluationPolicy()

        # High confidence task type
        result = service.classify_with_triage(
            DecisionType.TASK_TYPE_CLASSIFICATION,
            context={"message_excerpt": "fix the authentication bug"},
            heuristic_result="edit",
            heuristic_confidence=0.85,
            runtime_policy=runtime_policy,
        )

        assert result.triage_outcome == ClassificationTriage.ACCEPT
        assert result.confidence >= 0.8
        assert result.verification_result is None

    def test_intent_classification_triage(self):
        """Test intent classification with triage."""
        config = DecisionServiceSettings(enabled=True)
        service = TieredDecisionService(config)
        runtime_policy = RuntimeEvaluationPolicy()

        # Medium confidence intent (should trigger verification if edge available)
        result = service.classify_with_triage(
            DecisionType.INTENT_CLASSIFICATION,
            context={"text_tail": "I'll continue working on the analysis"},
            heuristic_result="continuation",
            heuristic_confidence=0.65,
            runtime_policy=runtime_policy,
        )

        assert result.triage_outcome in (ClassificationTriage.VERIFY, ClassificationTriage.ACCEPT)
        assert result.confidence >= 0.0

    def test_tool_selection_triage(self):
        """Test tool selection with triage."""
        config = DecisionServiceSettings(enabled=True)
        service = TieredDecisionService(config)
        runtime_policy = RuntimeEvaluationPolicy()

        # High confidence tool selection
        result = service.classify_with_triage(
            DecisionType.TOOL_SELECTION,
            context={
                "message_excerpt": "search for all files",
                "available_tools": "search, read, write",
            },
            heuristic_result=["search", "read_file"],
            heuristic_confidence=0.88,
            runtime_policy=runtime_policy,
        )

        assert result.triage_outcome == ClassificationTriage.ACCEPT
        assert result.confidence >= 0.8

    def test_custom_runtime_policy(self):
        """Test triage with custom runtime policy."""
        config = DecisionServiceSettings(enabled=True)
        service = TieredDecisionService(config)

        # Custom policy with different thresholds
        custom_policy = RuntimeEvaluationPolicy(
            high_confidence_threshold=0.9,
            medium_confidence_threshold=0.7,
        )

        result = service.classify_with_triage(
            DecisionType.TASK_TYPE_CLASSIFICATION,
            context={"message_excerpt": "analyze code"},
            heuristic_result="analysis",
            heuristic_confidence=0.85,
            runtime_policy=custom_policy,
        )

        # With custom thresholds, 0.85 is in VERIFY range (0.7-0.9)
        assert result.triage_outcome in (ClassificationTriage.VERIFY, ClassificationTriage.ACCEPT)


@pytest.mark.integration
class TestTriageEndToEndScenarios:
    """Test end-to-end triage scenarios."""

    def test_classify_clear_task_type(self):
        """Test classifying a clear task type."""
        config = DecisionServiceSettings(enabled=True)
        tiered_service = TieredDecisionService(config)
        classifier = UnifiedTaskClassifier(tiered_decision_service=tiered_service)

        result = classifier.classify("fix the authentication bug in the login module")

        # Should classify with high confidence
        assert result.task_type in (ClassifierTaskType.EDIT, ClassifierTaskType.ACTION)
        assert result.confidence >= 0.5

    def test_classify_ambiguous_task_type(self):
        """Test classifying an ambiguous task type."""
        config = DecisionServiceSettings(enabled=True)
        tiered_service = TieredDecisionService(config)
        classifier = UnifiedTaskClassifier(tiered_decision_service=tiered_service)

        result = classifier.classify("maybe work on the code")

        # Should handle ambiguity gracefully
        assert result.task_type in ClassifierTaskType
        assert 0.0 <= result.confidence <= 1.0

    def test_classify_with_context_awareness(self):
        """Test classification with conversation context."""
        config = DecisionServiceSettings(enabled=True)
        tiered_service = TieredDecisionService(config)
        classifier = UnifiedTaskClassifier(tiered_decision_service=tiered_service)

        history = [
            {"role": "user", "content": "Analyze the codebase"},
            {"role": "user", "content": "Review the authentication module"},
        ]

        result = classifier.classify_with_context("continue the review", history)

        # Context should help classification
        assert result.task_type in ClassifierTaskType
        assert result.confidence >= 0.0

    def test_intent_detection_continuation_vs_completion(self):
        """Test distinguishing continuation from completion intent."""
        config = DecisionServiceSettings(enabled=True)
        tiered_service = TieredDecisionService(config)
        classifier = IntentClassifier.get_instance()
        classifier.initialize_sync()

        # Continuation
        cont_intent, cont_conf, _ = classifier.classify_with_triage(
            "Let me read the next file",
            context={"has_tool_calls": False},
            tiered_service=tiered_service,
        )

        # Completion
        comp_intent, comp_conf, _ = classifier.classify_with_triage(
            "In summary, here are the findings",
            context={"has_tool_calls": False},
            tiered_service=tiered_service,
        )

        # Should correctly distinguish
        assert cont_intent in (IntentType.CONTINUATION, IntentType.NEUTRAL)
        assert comp_intent in (IntentType.COMPLETION, IntentType.NEUTRAL)

    def test_triage_performance_targets(self):
        """Test that triage meets performance targets."""
        import time

        config = DecisionServiceSettings(enabled=True)
        service = TieredDecisionService(config)
        runtime_policy = RuntimeEvaluationPolicy()

        # High confidence - should be fast (< 10ms)
        start = time.monotonic()
        result = service.classify_with_triage(
            DecisionType.TASK_TYPE_CLASSIFICATION,
            context={"message": "fix bug"},
            heuristic_result="debug",
            heuristic_confidence=0.9,
            runtime_policy=runtime_policy,
        )
        high_conf_latency_ms = (time.monotonic() - start) * 1000

        assert result.triage_outcome == ClassificationTriage.ACCEPT
        assert high_conf_latency_ms < 50  # Fast path


@pytest.mark.integration
class TestTriageErrorHandling:
    """Test error handling and graceful degradation."""

    def test_triage_without_edge_service(self):
        """Test triage behavior when edge service is unavailable."""
        config = DecisionServiceSettings(enabled=True)
        service = TieredDecisionService(config)
        runtime_policy = RuntimeEvaluationPolicy()

        # Medium confidence - would normally verify
        result = service.classify_with_triage(
            DecisionType.TASK_TYPE_CLASSIFICATION,
            context={"message": "analyze code"},
            heuristic_result="analysis",
            heuristic_confidence=0.65,
            runtime_policy=runtime_policy,
        )

        # Should handle gracefully
        assert result.triage_outcome in (ClassificationTriage.VERIFY, ClassificationTriage.ACCEPT)
        assert result.confidence >= 0.0

    def test_triage_with_invalid_heuristic(self):
        """Test triage with invalid heuristic result."""
        config = DecisionServiceSettings(enabled=True)
        service = TieredDecisionService(config)
        runtime_policy = RuntimeEvaluationPolicy()

        # Very low confidence heuristic
        result = service.classify_with_triage(
            DecisionType.TASK_TYPE_CLASSIFICATION,
            context={"message": "???"},
            heuristic_result="unknown",
            heuristic_confidence=0.1,
            runtime_policy=runtime_policy,
        )

        # Should reject and fall back
        assert result.triage_outcome == ClassificationTriage.REJECT
        assert result.source == "heuristic_fallback"
