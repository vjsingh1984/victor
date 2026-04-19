"""Tests for confidence calibrator.

Tests cover:
- Threshold initialization and retrieval
- Decision outcome recording
- Adaptive threshold calibration
- Hysteresis and oscillation prevention
- Statistics tracking and accuracy calculation
- Different calibration strategies
"""

import pytest

from victor.agent.decisions.schemas import DecisionType
from victor.agent.services.confidence_calibrator import (
    CalibrationStrategy,
    ConfidenceCalibrator,
    DecisionRecord,
    TypeStatistics,
    create_confidence_calibrator,
)


class TestConfidenceCalibrator:
    """Test confidence calibrator initialization and basic operations."""

    def test_initialization_with_defaults(self):
        """Test calibrator initialization with default parameters."""
        calibrator = ConfidenceCalibrator()

        assert calibrator._strategy == CalibrationStrategy.ADAPTIVE
        assert calibrator._base_threshold == 0.70
        assert calibrator._target_accuracy == 0.95
        assert calibrator._min_threshold == 0.50
        assert calibrator._max_threshold == 0.95

    def test_initialization_with_custom_params(self):
        """Test calibrator initialization with custom parameters."""
        calibrator = ConfidenceCalibrator(
            strategy=CalibrationStrategy.CONSERVATIVE,
            base_threshold=0.80,
            target_accuracy=0.90,
            min_threshold=0.60,
            max_threshold=0.99,
        )

        assert calibrator._strategy == CalibrationStrategy.CONSERVATIVE
        assert calibrator._base_threshold == 0.80
        assert calibrator._target_accuracy == 0.90
        assert calibrator._min_threshold == 0.60
        assert calibrator._max_threshold == 0.99

    def test_get_threshold_initializes_new_type(self):
        """Test that getting threshold initializes a new decision type."""
        calibrator = ConfidenceCalibrator(base_threshold=0.75)

        threshold = calibrator.get_threshold(DecisionType.TASK_COMPLETION)

        assert threshold == 0.75
        assert DecisionType.TASK_COMPLETION in calibrator._thresholds

    def test_get_threshold_returns_existing(self):
        """Test that getting threshold returns existing value."""
        calibrator = ConfidenceCalibrator(base_threshold=0.75)
        calibrator._thresholds[DecisionType.TASK_COMPLETION] = 0.85

        threshold = calibrator.get_threshold(DecisionType.TASK_COMPLETION)

        assert threshold == 0.85

    def test_conservative_strategy_always_uses_max_threshold(self):
        """Test that conservative strategy uses max threshold."""
        calibrator = ConfidenceCalibrator(
            strategy=CalibrationStrategy.CONSERVATIVE,
            max_threshold=0.90,
        )

        should_use_llm = calibrator.should_use_llm(
            DecisionType.TASK_COMPLETION,
            heuristic_confidence=0.85,
        )

        assert should_use_llm is True  # 0.85 < 0.90

    def test_aggressive_strategy_always_uses_min_threshold(self):
        """Test that aggressive strategy uses min threshold."""
        calibrator = ConfidenceCalibrator(
            strategy=CalibrationStrategy.AGGRESSIVE,
            min_threshold=0.50,
        )

        should_use_llm = calibrator.should_use_llm(
            DecisionType.TASK_COMPLETION,
            heuristic_confidence=0.60,
        )

        assert should_use_llm is False  # 0.60 >= 0.50

    def test_balanced_strategy_uses_base_threshold(self):
        """Test that balanced strategy uses base threshold."""
        calibrator = ConfidenceCalibrator(
            strategy=CalibrationStrategy.BALANCED,
            base_threshold=0.70,
        )

        should_use_llm = calibrator.should_use_llm(
            DecisionType.TASK_COMPLETION,
            heuristic_confidence=0.65,
        )

        assert should_use_llm is True  # 0.65 < 0.70


class TestDecisionRecording:
    """Test decision outcome recording and statistics tracking."""

    def test_record_outcome_creates_statistics(self):
        """Test that recording an outcome creates statistics."""
        calibrator = ConfidenceCalibrator()

        calibrator.record_outcome(
            decision_type=DecisionType.TASK_COMPLETION,
            heuristic_confidence=0.80,
            used_llm=False,
            was_correct=True,
        )

        assert DecisionType.TASK_COMPLETION in calibrator._statistics
        stats = calibrator._statistics[DecisionType.TASK_COMPLETION]
        assert stats.total_decisions == 1
        assert stats.heuristic_correct == 1
        assert stats.heuristic_incorrect == 0

    def test_record_multiple_outcomes(self):
        """Test recording multiple outcomes."""
        calibrator = ConfidenceCalibrator()

        for i in range(10):
            calibrator.record_outcome(
                decision_type=DecisionType.TASK_COMPLETION,
                heuristic_confidence=0.80,
                used_llm=False,
                was_correct=i % 2 == 0,  # Alternate correct/incorrect
            )

        stats = calibrator.get_statistics(DecisionType.TASK_COMPLETION)
        assert stats.total_decisions == 10
        assert stats.heuristic_correct == 5
        assert stats.heuristic_incorrect == 5

    def test_record_llm_outcome(self):
        """Test recording LLM decision outcomes."""
        calibrator = ConfidenceCalibrator()

        calibrator.record_outcome(
            decision_type=DecisionType.TASK_COMPLETION,
            heuristic_confidence=0.50,
            used_llm=True,
            was_correct=True,
            llm_confidence=0.90,
        )

        stats = calibrator.get_statistics(DecisionType.TASK_COMPLETION)
        assert stats.llm_correct == 1
        assert stats.llm_incorrect == 0

    def test_get_heuristic_accuracy(self):
        """Test calculating heuristic accuracy."""
        calibrator = ConfidenceCalibrator()

        # Record 10 decisions: 8 correct, 2 incorrect
        for i in range(10):
            calibrator.record_outcome(
                decision_type=DecisionType.TASK_COMPLETION,
                heuristic_confidence=0.80,
                used_llm=False,
                was_correct=i < 8,
            )

        stats = calibrator.get_statistics(DecisionType.TASK_COMPLETION)
        accuracy = stats.get_heuristic_accuracy()

        assert accuracy == 0.8

    def test_get_llm_accuracy(self):
        """Test calculating LLM accuracy."""
        calibrator = ConfidenceCalibrator()

        # Record 5 LLM decisions: 4 correct, 1 incorrect
        for i in range(5):
            calibrator.record_outcome(
                decision_type=DecisionType.TASK_COMPLETION,
                heuristic_confidence=0.50,
                used_llm=True,
                was_correct=i < 4,
                llm_confidence=0.90,
            )

        stats = calibrator.get_statistics(DecisionType.TASK_COMPLETION)
        accuracy = stats.get_llm_accuracy()

        assert accuracy == 0.8

    def test_recent_decisions_limit(self):
        """Test that recent decisions are limited to maxlen."""
        calibrator = ConfidenceCalibrator()

        # Record 150 decisions (more than maxlen=100)
        for i in range(150):
            calibrator.record_outcome(
                decision_type=DecisionType.TASK_COMPLETION,
                heuristic_confidence=0.80,
                used_llm=False,
                was_correct=True,
            )

        stats = calibrator.get_statistics(DecisionType.TASK_COMPLETION)
        assert len(stats.recent_decisions) == 100
        assert stats.total_decisions == 150  # Total count is not limited


class TestAdaptiveCalibration:
    """Test adaptive threshold calibration."""

    def test_threshold_lowered_on_high_accuracy(self):
        """Test that threshold is lowered when heuristic accuracy is high."""
        calibrator = ConfidenceCalibrator(
            strategy=CalibrationStrategy.ADAPTIVE,
            base_threshold=0.70,
            target_accuracy=0.95,
            adjustment_step=0.10,
        )

        # Record 25 correct heuristic decisions (100% accuracy)
        for _ in range(25):
            calibrator.record_outcome(
                decision_type=DecisionType.TASK_COMPLETION,
                heuristic_confidence=0.80,
                used_llm=False,
                was_correct=True,
            )

        # Threshold should be lowered
        new_threshold = calibrator.get_threshold(DecisionType.TASK_COMPLETION)
        assert new_threshold < 0.70
        assert new_threshold >= calibrator._min_threshold

    def test_threshold_raised_on_low_accuracy(self):
        """Test that threshold is raised when heuristic accuracy is low."""
        calibrator = ConfidenceCalibrator(
            strategy=CalibrationStrategy.ADAPTIVE,
            base_threshold=0.60,
            target_accuracy=0.95,
            adjustment_step=0.10,
        )

        # Record 25 decisions: only 15 correct (60% accuracy)
        for i in range(25):
            calibrator.record_outcome(
                decision_type=DecisionType.TASK_COMPLETION,
                heuristic_confidence=0.80,
                used_llm=False,
                was_correct=i < 15,
            )

        # Threshold should be raised
        new_threshold = calibrator.get_threshold(DecisionType.TASK_COMPLETION)
        assert new_threshold > 0.60
        assert new_threshold <= calibrator._max_threshold

    def test_threshold_not_changed_within_tolerance(self):
        """Test that threshold doesn't change when accuracy is within tolerance."""
        calibrator = ConfidenceCalibrator(
            strategy=CalibrationStrategy.ADAPTIVE,
            base_threshold=0.70,
            target_accuracy=0.95,
            hysteresis=0.06,  # Larger than adjustment_step (0.05) to prevent changes
        )

        # Record 25 decisions: 23 correct (92% accuracy, within 5% tolerance)
        for i in range(25):
            calibrator.record_outcome(
                decision_type=DecisionType.TASK_COMPLETION,
                heuristic_confidence=0.80,
                used_llm=False,
                was_correct=i < 23,
            )

        # Threshold should not change significantly
        new_threshold = calibrator.get_threshold(DecisionType.TASK_COMPLETION)
        assert abs(new_threshold - 0.70) < 0.01

    def test_hysteresis_prevents_oscillation(self):
        """Test that hysteresis prevents frequent threshold changes."""
        calibrator = ConfidenceCalibrator(
            strategy=CalibrationStrategy.ADAPTIVE,
            base_threshold=0.70,
            hysteresis=0.10,  # Large hysteresis
            adjustment_step=0.05,
        )

        # Record enough decisions to trigger calibration
        for _ in range(25):
            calibrator.record_outcome(
                decision_type=DecisionType.TASK_COMPLETION,
                heuristic_confidence=0.80,
                used_llm=False,
                was_correct=True,
            )

        # Threshold should not change due to hysteresis
        # (adjustment_step=0.05 < hysteresis=0.10)
        new_threshold = calibrator.get_threshold(DecisionType.TASK_COMPLETION)
        assert new_threshold == 0.70

    def test_threshold_clamped_to_min_max(self):
        """Test that threshold is clamped to min/max values."""
        calibrator = ConfidenceCalibrator(
            strategy=CalibrationStrategy.ADAPTIVE,
            base_threshold=0.70,
            min_threshold=0.60,
            max_threshold=0.80,
            adjustment_step=0.20,  # Large step to test clamping
        )

        # Test max clamping: very high accuracy should lower threshold
        for _ in range(30):
            calibrator.record_outcome(
                decision_type=DecisionType.TASK_COMPLETION,
                heuristic_confidence=0.99,
                used_llm=False,
                was_correct=True,
            )

        threshold = calibrator.get_threshold(DecisionType.TASK_COMPLETION)
        assert threshold >= calibrator._min_threshold
        assert threshold <= calibrator._max_threshold

    def test_minimum_decisions_before_calibration(self):
        """Test that calibration requires minimum decisions."""
        calibrator = ConfidenceCalibrator(
            strategy=CalibrationStrategy.ADAPTIVE,
            base_threshold=0.70,
            min_decisions_for_calibration=20,
        )

        # Record only 15 decisions (below minimum)
        for _ in range(15):
            calibrator.record_outcome(
                decision_type=DecisionType.TASK_COMPLETION,
                heuristic_confidence=0.80,
                used_llm=False,
                was_correct=True,
            )

        # Threshold should not change
        threshold = calibrator.get_threshold(DecisionType.TASK_COMPLETION)
        assert threshold == 0.70


class TestStatistics:
    """Test statistics tracking and retrieval."""

    def test_get_statistics_creates_if_not_exists(self):
        """Test that get_statistics creates if doesn't exist."""
        calibrator = ConfidenceCalibrator()

        stats = calibrator.get_statistics(DecisionType.TASK_COMPLETION)

        assert isinstance(stats, TypeStatistics)
        assert stats.total_decisions == 0

    def test_get_all_statistics(self):
        """Test getting statistics for all decision types."""
        calibrator = ConfidenceCalibrator()

        # Record decisions for multiple types
        calibrator.record_outcome(
            DecisionType.TASK_COMPLETION,
            0.80,
            False,
            True,
        )
        calibrator.record_outcome(
            DecisionType.INTENT_CLASSIFICATION,
            0.70,
            False,
            True,
        )

        all_stats = calibrator.get_all_statistics()

        assert len(all_stats) == 2
        assert DecisionType.TASK_COMPLETION in all_stats
        assert DecisionType.INTENT_CLASSIFICATION in all_stats

    def test_reset_specific_type(self):
        """Test resetting statistics for a specific type."""
        calibrator = ConfidenceCalibrator()

        calibrator.record_outcome(
            DecisionType.TASK_COMPLETION,
            0.80,
            False,
            True,
        )
        calibrator.record_outcome(
            DecisionType.INTENT_CLASSIFICATION,
            0.70,
            False,
            True,
        )

        calibrator.reset_statistics(DecisionType.TASK_COMPLETION)

        assert DecisionType.TASK_COMPLETION not in calibrator._statistics
        assert DecisionType.INTENT_CLASSIFICATION in calibrator._statistics

    def test_reset_all_types(self):
        """Test resetting statistics for all types."""
        calibrator = ConfidenceCalibrator()

        calibrator.record_outcome(
            DecisionType.TASK_COMPLETION,
            0.80,
            False,
            True,
        )

        calibrator.reset_statistics()

        assert len(calibrator._statistics) == 0
        assert len(calibrator._thresholds) == 0


class TestSummary:
    """Test summary generation."""

    def test_get_summary(self):
        """Test getting calibrator summary."""
        calibrator = ConfidenceCalibrator(
            strategy=CalibrationStrategy.ADAPTIVE,
            base_threshold=0.70,
        )

        # Record some decisions
        for i in range(10):
            calibrator.record_outcome(
                DecisionType.TASK_COMPLETION,
                0.80,
                False,
                i < 8,
            )

        summary = calibrator.get_summary()

        assert summary["strategy"] == "adaptive"
        assert summary["base_threshold"] == 0.70
        assert "decision_types" in summary
        assert "task_completion" in summary["decision_types"]

        type_summary = summary["decision_types"]["task_completion"]
        assert type_summary["total_decisions"] == 10
        assert type_summary["recent_accuracy"] == 0.8


class TestFactoryFunction:
    """Test factory function."""

    def test_create_with_default_strategy(self):
        """Test creating calibrator with default strategy."""
        calibrator = create_confidence_calibrator()

        assert calibrator._strategy == CalibrationStrategy.ADAPTIVE

    def test_create_with_custom_strategy(self):
        """Test creating calibrator with custom strategy."""
        calibrator = create_confidence_calibrator(strategy="conservative")

        assert calibrator._strategy == CalibrationStrategy.CONSERVATIVE

    def test_create_with_invalid_strategy(self):
        """Test creating calibrator with invalid strategy falls back to adaptive."""
        calibrator = create_confidence_calibrator(strategy="invalid")

        assert calibrator._strategy == CalibrationStrategy.ADAPTIVE

    def test_create_with_custom_params(self):
        """Test creating calibrator with custom parameters."""
        calibrator = create_confidence_calibrator(
            strategy="aggressive",
            base_threshold=0.60,
            target_accuracy=0.90,
        )

        assert calibrator._strategy == CalibrationStrategy.AGGRESSIVE
        assert calibrator._base_threshold == 0.60
        assert calibrator._target_accuracy == 0.90


class TestDecisionRecord:
    """Test DecisionRecord dataclass."""

    def test_decision_record_creation(self):
        """Test creating a decision record."""
        record = DecisionRecord(
            decision_type=DecisionType.TASK_COMPLETION,
            timestamp=123456.0,
            heuristic_confidence=0.80,
            used_llm=False,
            was_correct=True,
            llm_confidence=None,
            source="lookup",
        )

        assert record.decision_type == DecisionType.TASK_COMPLETION
        assert record.heuristic_confidence == 0.80
        assert record.used_llm is False
        assert record.was_correct is True
        assert record.source == "lookup"


class TestTypeStatistics:
    """Test TypeStatistics class."""

    def test_initial_statistics(self):
        """Test initial statistics values."""
        stats = TypeStatistics()

        assert stats.total_decisions == 0
        assert stats.heuristic_correct == 0
        assert stats.heuristic_incorrect == 0
        assert stats.llm_correct == 0
        assert stats.llm_incorrect == 0
        assert stats.recent_accuracy == 0.0

    def test_add_decision_updates_counts(self):
        """Test that adding a decision updates counts."""
        stats = TypeStatistics()

        record = DecisionRecord(
            decision_type=DecisionType.TASK_COMPLETION,
            timestamp=0.0,
            heuristic_confidence=0.80,
            used_llm=False,
            was_correct=True,
        )
        stats.add_decision(record)

        assert stats.total_decisions == 1
        assert stats.heuristic_correct == 1

    def test_add_decision_with_unknown_outcome(self):
        """Test that decisions with unknown outcomes are counted but not scored."""
        stats = TypeStatistics()

        record = DecisionRecord(
            decision_type=DecisionType.TASK_COMPLETION,
            timestamp=0.0,
            heuristic_confidence=0.80,
            used_llm=False,
            was_correct=None,  # Unknown outcome
        )
        stats.add_decision(record)

        assert stats.total_decisions == 1
        assert stats.heuristic_correct == 0
        assert stats.heuristic_incorrect == 0

    def test_recent_decisions_maxlen(self):
        """Test that recent decisions has a max length."""
        stats = TypeStatistics()

        # Add 150 decisions
        for i in range(150):
            record = DecisionRecord(
                decision_type=DecisionType.TASK_COMPLETION,
                timestamp=float(i),
                heuristic_confidence=0.80,
                used_llm=False,
                was_correct=True,
            )
            stats.add_decision(record)

        # Should only keep 100
        assert len(stats.recent_decisions) == 100
        assert stats.total_decisions == 150


class TestIntegration:
    """Integration tests for confidence calibrator."""

    def test_full_calibration_cycle(self):
        """Test a full calibration cycle from high to low accuracy."""
        calibrator = ConfidenceCalibrator(
            strategy=CalibrationStrategy.ADAPTIVE,
            base_threshold=0.70,
            target_accuracy=0.95,
            adjustment_step=0.10,
            hysteresis=0.01,
        )

        # Phase 1: High accuracy (all correct)
        for _ in range(25):
            calibrator.record_outcome(
                DecisionType.TASK_COMPLETION,
                0.80,
                False,
                True,
            )

        threshold_after_high = calibrator.get_threshold(DecisionType.TASK_COMPLETION)
        assert threshold_after_high < 0.70

        # Phase 2: Low accuracy (many incorrect)
        for i in range(25):
            calibrator.record_outcome(
                DecisionType.TASK_COMPLETION,
                0.80,
                False,
                i < 10,  # Only 10/25 correct
            )

        threshold_after_low = calibrator.get_threshold(DecisionType.TASK_COMPLETION)
        assert threshold_after_low > threshold_after_high

    def test_multiple_decision_types_independent(self):
        """Test that multiple decision types are calibrated independently."""
        calibrator = ConfidenceCalibrator(
            strategy=CalibrationStrategy.ADAPTIVE,
            base_threshold=0.70,
        )

        # Task completion: high accuracy
        for _ in range(25):
            calibrator.record_outcome(
                DecisionType.TASK_COMPLETION,
                0.80,
                False,
                True,
            )

        # Intent classification: low accuracy
        for i in range(25):
            calibrator.record_outcome(
                DecisionType.INTENT_CLASSIFICATION,
                0.80,
                False,
                i < 10,
            )

        tc_threshold = calibrator.get_threshold(DecisionType.TASK_COMPLETION)
        ic_threshold = calibrator.get_threshold(DecisionType.INTENT_CLASSIFICATION)

        # Task completion should have lower threshold (more aggressive)
        # Intent classification should have higher threshold (more conservative)
        assert tc_threshold < ic_threshold
