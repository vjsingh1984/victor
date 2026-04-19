"""Tests for deterministic decision rules.

Tests cover:
- Lookup table accuracy for all supported decision types
- Pattern matching with regex patterns
- Ensemble voting with multiple signals
- Edge cases and error handling
"""

import pytest

from victor.agent.decisions.schemas import (
    DecisionType,
    ErrorClassDecision,
    IntentDecision,
    TaskCompletionDecision,
    TaskTypeDecision,
    ToolNecessityDecision,
)
from victor.agent.services.deterministic_decision_rules import (
    EnsembleVoter,
    LookupTables,
    LookupResult,
    PatternMatcher,
)


class TestLookupTables:
    """Test O(1) lookup table decision making."""

    def test_task_completion_lookup_done(self):
        """Test task completion lookup for 'done' patterns."""
        context = {"message": "I'm done with the task"}

        result = LookupTables.lookup(DecisionType.TASK_COMPLETION, context)

        assert result is not None
        assert isinstance(result.decision, TaskCompletionDecision)
        assert result.decision.is_complete is True
        assert result.decision.phase == "done"
        assert result.confidence >= 0.90
        assert "matched" in result.reason.lower()

    def test_task_completion_lookup_working(self):
        """Test task completion lookup for 'working' patterns."""
        context = {"message": "I'm working on it"}

        result = LookupTables.lookup(DecisionType.TASK_COMPLETION, context)

        assert result is not None
        assert result.decision.is_complete is False
        assert result.decision.phase == "working"
        assert result.confidence >= 0.85

    def test_task_completion_lookup_stuck(self):
        """Test task completion lookup for 'stuck' patterns."""
        context = {"message": "I'm stuck on this error"}

        result = LookupTables.lookup(DecisionType.TASK_COMPLETION, context)

        assert result is not None
        assert result.decision.is_complete is False
        assert result.decision.phase == "stuck"
        assert result.confidence >= 0.90

    def test_task_completion_lookup_finalizing(self):
        """Test task completion lookup for 'finalizing' patterns."""
        context = {"message": "Almost done, one more step"}

        result = LookupTables.lookup(DecisionType.TASK_COMPLETION, context)

        assert result is not None
        assert result.decision.is_complete is False
        assert result.decision.phase == "finalizing"
        assert result.confidence >= 0.85

    def test_task_completion_word_by_word_lookup(self):
        """Test word-by-word lookup for longer phrases."""
        context = {"message": "The implementation is complete"}

        result = LookupTables.lookup(DecisionType.TASK_COMPLETION, context)

        assert result is not None
        assert result.decision.is_complete is True
        assert result.decision.phase == "done"

    def test_intent_classification_lookup_completion(self):
        """Test intent classification for completion intent."""
        context = {"response": "Here's the code you requested"}

        result = LookupTables.lookup(DecisionType.INTENT_CLASSIFICATION, context)

        assert result is not None
        assert isinstance(result.decision, IntentDecision)
        assert result.decision.intent == "completion"
        assert result.confidence >= 0.85

    def test_intent_classification_lookup_continuation(self):
        """Test intent classification for continuation intent."""
        context = {"response": "Now let me implement the fix"}

        result = LookupTables.lookup(DecisionType.INTENT_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.intent == "continuation"
        assert result.confidence >= 0.85

    def test_intent_classification_lookup_asking_input(self):
        """Test intent classification for asking input."""
        context = {"response": "Would you like me to add tests?"}

        result = LookupTables.lookup(DecisionType.INTENT_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.intent == "asking_input"
        assert result.confidence >= 0.90

    def test_intent_classification_lookup_stuck_loop(self):
        """Test intent classification for stuck loop."""
        context = {"response": "I'm stuck on this problem"}

        result = LookupTables.lookup(DecisionType.INTENT_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.intent == "stuck_loop"
        assert result.confidence >= 0.90

    def test_task_type_lookup_analysis(self):
        """Test task type classification for analysis."""
        context = {"task_description": "Analyze the performance issue"}

        result = LookupTables.lookup(DecisionType.TASK_TYPE_CLASSIFICATION, context)

        assert result is not None
        assert isinstance(result.decision, TaskTypeDecision)
        assert result.decision.task_type == "analysis"
        assert result.confidence >= 0.90

    def test_task_type_lookup_action(self):
        """Test task type classification for action."""
        context = {"task_description": "Fix the authentication bug"}

        result = LookupTables.lookup(DecisionType.TASK_TYPE_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.task_type == "action"
        assert result.confidence >= 0.90

    def test_task_type_lookup_generation(self):
        """Test task type classification for generation."""
        context = {"task_description": "Generate a REST API client"}

        result = LookupTables.lookup(DecisionType.TASK_TYPE_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.task_type == "generation"
        assert result.confidence >= 0.90

    def test_task_type_lookup_search(self):
        """Test task type classification for search."""
        context = {"task_description": "Find all uses of the function"}

        result = LookupTables.lookup(DecisionType.TASK_TYPE_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.task_type == "search"
        assert result.confidence >= 0.90

    def test_task_type_lookup_edit(self):
        """Test task type classification for edit."""
        context = {"task_description": "Edit the configuration file"}

        result = LookupTables.lookup(DecisionType.TASK_TYPE_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.task_type == "edit"
        assert result.confidence >= 0.90

    def test_error_type_lookup_permanent(self):
        """Test error classification for permanent errors."""
        context = {"error_message": "File not found: config.yaml"}

        result = LookupTables.lookup(DecisionType.ERROR_CLASSIFICATION, context)

        assert result is not None
        assert isinstance(result.decision, ErrorClassDecision)
        assert result.decision.error_type == "permanent"
        assert result.confidence >= 0.90

    def test_error_type_lookup_transient(self):
        """Test error classification for transient errors."""
        context = {"error_message": "Connection timeout after 30s"}

        result = LookupTables.lookup(DecisionType.ERROR_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.error_type == "transient"
        assert result.confidence >= 0.90

    def test_error_type_lookup_retryable(self):
        """Test error classification for retryable errors."""
        context = {"error_message": "500 Internal Server Error"}

        result = LookupTables.lookup(DecisionType.ERROR_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.error_type == "retryable"
        assert result.confidence >= 0.85

    def test_tool_necessity_lookup_needs_tools(self):
        """Test tool necessity for tasks that need tools."""
        context = {"message": "Fix the bug in login.py"}

        result = LookupTables.lookup(DecisionType.TOOL_NECESSITY, context)

        assert result is not None
        assert isinstance(result.decision, ToolNecessityDecision)
        assert result.decision.requires_tools is True
        assert result.confidence >= 0.90

    def test_tool_necessity_lookup_no_tools(self):
        """Test tool necessity for tasks that don't need tools."""
        context = {"message": "Explain how the authentication works"}

        result = LookupTables.lookup(DecisionType.TOOL_NECESSITY, context)

        assert result is not None
        assert result.decision.requires_tools is False
        assert result.confidence >= 0.80

    def test_lookup_with_nested_context(self):
        """Test lookup with nested context dict."""
        context = {"nested": {"message": "Task is complete"}}

        result = LookupTables.lookup(DecisionType.TASK_COMPLETION, context)

        assert result is not None
        assert result.decision.is_complete is True

    def test_lookup_with_missing_text(self):
        """Test lookup with no text in context."""
        context = {"some_other_key": "value"}

        result = LookupTables.lookup(DecisionType.TASK_COMPLETION, context)

        assert result is None

    def test_lookup_unsupported_decision_type(self):
        """Test lookup for unsupported decision type."""
        context = {"message": "test"}

        result = LookupTables.lookup(DecisionType.STAGE_DETECTION, context)

        assert result is None


class TestPatternMatcher:
    """Test regex pattern matching for decisions."""

    def test_task_completion_pattern_done(self):
        """Test task completion pattern for done status."""
        context = {"message": "I've completed the implementation"}

        result = PatternMatcher.match(DecisionType.TASK_COMPLETION, context)

        assert result is not None
        assert result.decision.is_complete is True
        assert result.decision.phase == "done"
        assert result.confidence >= 0.90

    def test_task_completion_pattern_working(self):
        """Test task completion pattern for working status."""
        context = {"message": "Let's proceed with the next step"}

        result = PatternMatcher.match(DecisionType.TASK_COMPLETION, context)

        assert result is not None
        assert result.decision.is_complete is False
        assert result.decision.phase == "working"

    def test_task_completion_pattern_stuck(self):
        """Test task completion pattern for stuck status."""
        context = {"message": "I'm stuck and cannot proceed"}

        result = PatternMatcher.match(DecisionType.TASK_COMPLETION, context)

        assert result is not None
        assert result.decision.is_complete is False
        assert result.decision.phase == "stuck"
        assert result.confidence >= 0.90

    def test_task_completion_pattern_finalizing(self):
        """Test task completion pattern for finalizing status."""
        context = {"message": "Almost there, just one more thing"}

        result = PatternMatcher.match(DecisionType.TASK_COMPLETION, context)

        assert result is not None
        assert result.decision.is_complete is False
        assert result.decision.phase == "finalizing"

    def test_intent_pattern_completion(self):
        """Test intent pattern for completion."""
        context = {"response": "The solution is ready to deploy"}

        result = PatternMatcher.match(DecisionType.INTENT_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.intent == "completion"

    def test_intent_pattern_continuation(self):
        """Test intent pattern for continuation."""
        context = {"response": "Next, I'll implement the tests"}

        result = PatternMatcher.match(DecisionType.INTENT_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.intent == "continuation"

    def test_intent_pattern_asking_input(self):
        """Test intent pattern for asking input."""
        context = {"response": "Which approach would you prefer?"}

        result = PatternMatcher.match(DecisionType.INTENT_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.intent == "asking_input"
        assert result.confidence >= 0.85

    def test_intent_pattern_stuck_loop(self):
        """Test intent pattern for stuck loop."""
        context = {"response": "I'm having trouble with this"}

        result = PatternMatcher.match(DecisionType.INTENT_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.intent == "stuck_loop"

    def test_task_type_pattern_analysis(self):
        """Test task type pattern for analysis."""
        context = {"task_description": "Investigate the memory leak"}

        result = PatternMatcher.match(DecisionType.TASK_TYPE_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.task_type == "analysis"
        assert result.confidence >= 0.88

    def test_task_type_pattern_action(self):
        """Test task type pattern for action."""
        context = {"task_description": "Implement the user authentication"}

        result = PatternMatcher.match(DecisionType.TASK_TYPE_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.task_type == "action"

    def test_task_type_pattern_generation(self):
        """Test task type pattern for generation."""
        context = {"task_description": "Generate a comprehensive test suite"}

        result = PatternMatcher.match(DecisionType.TASK_TYPE_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.task_type == "generation"

    def test_task_type_pattern_search(self):
        """Test task type pattern for search."""
        context = {"task_description": "Find where this function is called"}

        result = PatternMatcher.match(DecisionType.TASK_TYPE_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.task_type == "search"

    def test_task_type_pattern_edit(self):
        """Test task type pattern for edit."""
        context = {"task_description": "Refactor this module for better performance"}

        result = PatternMatcher.match(DecisionType.TASK_TYPE_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.task_type == "edit"

    def test_error_pattern_permanent(self):
        """Test error pattern for permanent errors."""
        context = {"error_message": "Permission denied: access to file denied"}

        result = PatternMatcher.match(DecisionType.ERROR_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.error_type == "permanent"
        assert result.confidence >= 0.90

    def test_error_pattern_transient(self):
        """Test error pattern for transient errors."""
        context = {"error_message": "Network connection failed"}

        result = PatternMatcher.match(DecisionType.ERROR_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.error_type == "transient"

    def test_error_pattern_retryable(self):
        """Test error pattern for retryable errors."""
        context = {"error_message": "503 Service Unavailable"}

        result = PatternMatcher.match(DecisionType.ERROR_CLASSIFICATION, context)

        assert result is not None
        assert result.decision.error_type == "retryable"

    def test_pattern_case_insensitive(self):
        """Test that pattern matching is case-insensitive."""
        context = {"message": "I'M DONE WITH THE TASK"}

        result = PatternMatcher.match(DecisionType.TASK_COMPLETION, context)

        assert result is not None
        assert result.decision.is_complete is True

    def test_pattern_with_missing_text(self):
        """Test pattern with no text in context."""
        context = {"other_key": "value"}

        result = PatternMatcher.match(DecisionType.TASK_COMPLETION, context)

        assert result is None


class TestEnsembleVoter:
    """Test ensemble voting for combining signals."""

    def test_ensemble_voter_initialization(self):
        """Test ensemble voter initialization."""
        voter = EnsembleVoter()

        assert voter.weights["keyword"] == 0.3
        assert voter.weights["semantic"] == 0.4
        assert voter.weights["context"] == 0.2
        assert voter.weights["heuristic"] == 0.1

    def test_ensemble_voter_custom_weights(self):
        """Test ensemble voter with custom weights."""
        custom_weights = {"keyword": 0.5, "semantic": 0.3, "context": 0.1, "heuristic": 0.1}
        voter = EnsembleVoter(weights=custom_weights)

        assert voter.weights == custom_weights

    def test_ensemble_voter_invalid_weights(self):
        """Test ensemble voter with invalid weights."""
        invalid_weights = {"keyword": 0.5, "semantic": 0.3}  # Sum != 1.0

        with pytest.raises(ValueError, match="must sum to 1.0"):
            EnsembleVoter(weights=invalid_weights)

    def test_ensemble_vote_with_keyword_only(self):
        """Test ensemble voting with only keyword result."""
        voter = EnsembleVoter()

        keyword_result = LookupResult(
            decision=TaskCompletionDecision(is_complete=True, confidence=0.90, phase="done"),
            confidence=0.90,
            reason="Keyword matched",
            matched_pattern="done",
        )

        result = voter.vote(
            DecisionType.TASK_COMPLETION,
            {"message": "done"},
            keyword_result=keyword_result,
        )

        assert result is not None
        assert result.decision.is_complete is True
        # Confidence should be boosted slightly due to ensemble
        assert result.confidence >= 0.90

    def test_ensemble_vote_with_multiple_signals(self):
        """Test ensemble voting with multiple signals."""
        voter = EnsembleVoter()

        keyword_result = LookupResult(
            decision=TaskCompletionDecision(is_complete=True, confidence=0.85, phase="done"),
            confidence=0.85,
            reason="Keyword matched",
            matched_pattern="complete",
        )

        semantic_result = LookupResult(
            decision=TaskCompletionDecision(is_complete=True, confidence=0.88, phase="done"),
            confidence=0.88,
            reason="Semantic match",
            matched_pattern="semantic",
        )

        result = voter.vote(
            DecisionType.TASK_COMPLETION,
            {"message": "Task is complete"},
            keyword_result=keyword_result,
            semantic_result=semantic_result,
        )

        assert result is not None
        assert result.decision.is_complete is True
        # Confidence should be boosted due to agreement
        assert result.confidence >= 0.88
        assert "Ensemble" in result.reason

    def test_ensemble_vote_with_heuristic_fallback(self):
        """Test ensemble voting with heuristic fallback."""
        voter = EnsembleVoter()

        heuristic_decision = TaskCompletionDecision(
            is_complete=False, confidence=0.70, phase="working"
        )

        result = voter.vote(
            DecisionType.TASK_COMPLETION,
            {"message": "working on it"},
            heuristic_result=heuristic_decision,
            heuristic_confidence=0.70,
        )

        assert result is not None
        assert result.decision.is_complete is False
        assert "heuristic" in result.matched_pattern.lower()

    def test_ensemble_vote_no_signals(self):
        """Test ensemble voting with no signals."""
        voter = EnsembleVoter()

        result = voter.vote(
            DecisionType.TASK_COMPLETION,
            {"message": "test"},
        )

        assert result is None

    def test_ensemble_confidence_boost(self):
        """Test that ensemble voting boosts confidence appropriately."""
        voter = EnsembleVoter()

        # Create multiple agreeing results
        results = []
        for i in range(3):
            results.append(
                LookupResult(
                    decision=TaskCompletionDecision(
                        is_complete=True, confidence=0.85, phase="done"
                    ),
                    confidence=0.85,
                    reason=f"Signal {i}",
                    matched_pattern=f"pattern{i}",
                )
            )

        result = voter.vote(
            DecisionType.TASK_COMPLETION,
            {"message": "done"},
            keyword_result=results[0],
            semantic_result=results[1],
        )

        # Should have boosted confidence (max boost is 15%)
        assert result.confidence > 0.85
        assert result.confidence <= 0.98  # Max cap

    def test_ensemble_confidence_cap(self):
        """Test that ensemble confidence is capped at 0.98."""
        voter = EnsembleVoter()

        # Create high-confidence results
        keyword_result = LookupResult(
            decision=TaskCompletionDecision(is_complete=True, confidence=0.95, phase="done"),
            confidence=0.95,
            reason="Keyword",
            matched_pattern="done",
        )

        semantic_result = LookupResult(
            decision=TaskCompletionDecision(is_complete=True, confidence=0.95, phase="done"),
            confidence=0.95,
            reason="Semantic",
            matched_pattern="done",
        )

        result = voter.vote(
            DecisionType.TASK_COMPLETION,
            {"message": "done"},
            keyword_result=keyword_result,
            semantic_result=semantic_result,
        )

        # Should be capped at 0.98 even with boost
        assert result.confidence <= 0.98


class TestIntegration:
    """Integration tests for deterministic decision rules."""

    def test_lookup_fallback_to_pattern(self):
        """Test that pattern matcher catches what lookup misses."""
        context = {"message": "I have completed the task successfully"}

        # Try lookup first
        lookup_result = LookupTables.lookup(DecisionType.TASK_COMPLETION, context)

        # If lookup doesn't match, pattern should
        pattern_result = PatternMatcher.match(DecisionType.TASK_COMPLETION, context)

        # At least one should match
        assert lookup_result is not None or pattern_result is not None

    def test_high_coverage_completion_patterns(self):
        """Test that lookup + pattern provides high coverage."""
        test_phrases = [
            "I'm done",
            "Task is complete",
            "Finished the work",
            "Successfully implemented",
            "I've completed the task",
            "The implementation is done",
            "All finished",
            "Ready to go",
        ]

        matches = 0
        for phrase in test_phrases:
            context = {"message": phrase}
            lookup_result = LookupTables.lookup(DecisionType.TASK_COMPLETION, context)
            pattern_result = PatternMatcher.match(DecisionType.TASK_COMPLETION, context)

            if lookup_result or pattern_result:
                matches += 1

        # Should match at least 80% of test phrases
        coverage = matches / len(test_phrases)
        assert coverage >= 0.8, f"Coverage {coverage:.2%} below 80% threshold"

    def test_high_coverage_intent_patterns(self):
        """Test high coverage for intent classification."""
        test_phrases = [
            "Here's the solution",
            "Now let me continue",
            "Would you like me to proceed?",
            "I'm stuck on this",
            "The code is ready",
            "Next, I'll implement",
        ]

        matches = 0
        for phrase in test_phrases:
            context = {"response": phrase}
            lookup_result = LookupTables.lookup(DecisionType.INTENT_CLASSIFICATION, context)
            pattern_result = PatternMatcher.match(DecisionType.INTENT_CLASSIFICATION, context)

            if lookup_result or pattern_result:
                matches += 1

        coverage = matches / len(test_phrases)
        assert coverage >= 0.7, f"Coverage {coverage:.2%} below 70% threshold"

    def test_error_classification_coverage(self):
        """Test error classification coverage."""
        error_messages = [
            "File not found",
            "Connection timeout",
            "500 Internal Server Error",
            "Permission denied",
            "Rate limit exceeded",
            "SyntaxError: invalid syntax",
        ]

        matches = 0
        for error_msg in error_messages:
            context = {"error_message": error_msg}
            lookup_result = LookupTables.lookup(DecisionType.ERROR_CLASSIFICATION, context)
            pattern_result = PatternMatcher.match(DecisionType.ERROR_CLASSIFICATION, context)

            if lookup_result or pattern_result:
                matches += 1

        coverage = matches / len(error_messages)
        assert coverage >= 0.8, f"Coverage {coverage:.2%} below 80% threshold"
