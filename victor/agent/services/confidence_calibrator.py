"""Confidence calibrator for adaptive decision thresholds.

Tracks accuracy history per decision type and adjusts thresholds dynamically
to optimize the trade-off between fast heuristic decisions and LLM fallback.

Key features:
- Tracks recent accuracy (default: last 100 decisions per type)
- Adaptive thresholds that adjust based on performance
- Threshold adjustment logic with hysteresis to prevent oscillation
- Per-decision-type calibration with configurable parameters
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from victor.agent.decisions.schemas import DecisionType

logger = logging.getLogger(__name__)


class CalibrationStrategy(str, Enum):
    """Strategies for threshold calibration."""

    CONSERVATIVE = "conservative"  # High threshold, prefer LLM
    BALANCED = "balanced"  # Medium threshold
    AGGRESSIVE = "aggressive"  # Low threshold, prefer heuristics
    ADAPTIVE = "adaptive"  # Automatically adjust based on accuracy


@dataclass
class DecisionRecord:
    """Record of a single decision and its outcome."""

    decision_type: DecisionType
    timestamp: float
    heuristic_confidence: float
    used_llm: bool
    was_correct: Optional[bool] = None  # None if outcome not yet known
    llm_confidence: Optional[float] = None
    source: str = "unknown"  # "lookup", "pattern", "ensemble", "llm", "heuristic"


@dataclass
class TypeStatistics:
    """Statistics for a specific decision type."""

    total_decisions: int = 0
    heuristic_correct: int = 0
    heuristic_incorrect: int = 0
    llm_correct: int = 0
    llm_incorrect: int = 0
    recent_accuracy: float = 0.0
    recent_decisions: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_decision(self, record: DecisionRecord) -> None:
        """Add a decision record and update statistics."""
        self.total_decisions += 1
        self.recent_decisions.append(record)

        # Update correctness counts
        if record.was_correct is not None:
            if record.used_llm:
                if record.was_correct:
                    self.llm_correct += 1
                else:
                    self.llm_incorrect += 1
            else:
                if record.was_correct:
                    self.heuristic_correct += 1
                else:
                    self.heuristic_incorrect += 1

        # Recalculate recent accuracy
        self._recalculate_accuracy()

    def _recalculate_accuracy(self) -> None:
        """Recalculate accuracy based on recent decisions."""
        if not self.recent_decisions:
            self.recent_accuracy = 0.0
            return

        # Only count decisions with known outcomes
        known_outcomes = [d for d in self.recent_decisions if d.was_correct is not None]

        if not known_outcomes:
            self.recent_accuracy = 0.0
            return

        correct = sum(1 for d in known_outcomes if d.was_correct)
        self.recent_accuracy = correct / len(known_outcomes)

    def get_heuristic_accuracy(self) -> float:
        """Get accuracy of heuristic-only decisions."""
        heuristic_decisions = [
            d for d in self.recent_decisions
            if not d.used_llm and d.was_correct is not None
        ]

        if not heuristic_decisions:
            return 0.0

        correct = sum(1 for d in heuristic_decisions if d.was_correct)
        return correct / len(heuristic_decisions)

    def get_llm_accuracy(self) -> float:
        """Get accuracy of LLM decisions."""
        llm_decisions = [
            d for d in self.recent_decisions
            if d.used_llm and d.was_correct is not None
        ]

        if not llm_decisions:
            return 0.0

        correct = sum(1 for d in llm_decisions if d.was_correct)
        return correct / len(llm_decisions)


class ConfidenceCalibrator:
    """Calibrates confidence thresholds based on historical accuracy.

    Tracks decision outcomes and adjusts thresholds to maintain target
    accuracy while minimizing LLM calls. Uses hysteresis to prevent
    threshold oscillation.

    Usage:
        calibrator = ConfidenceCalibrator()

        # Before making a decision
        should_use_llm = calibrator.should_use_llm(
            decision_type=DecisionType.TASK_COMPLETION,
            heuristic_confidence=0.85,
        )

        # After getting outcome
        calibrator.record_outcome(
            decision_type=DecisionType.TASK_COMPLETION,
            heuristic_confidence=0.85,
            used_llm=should_use_llm,
            was_correct=True,
        )
    """

    def __init__(
        self,
        strategy: CalibrationStrategy = CalibrationStrategy.ADAPTIVE,
        base_threshold: float = 0.70,
        target_accuracy: float = 0.95,
        min_threshold: float = 0.50,
        max_threshold: float = 0.95,
        adjustment_step: float = 0.05,
        hysteresis: float = 0.03,
        min_decisions_for_calibration: int = 20,
    ) -> None:
        """Initialize the confidence calibrator.

        Args:
            strategy: Calibration strategy to use
            base_threshold: Initial confidence threshold
            target_accuracy: Target accuracy for heuristic decisions
            min_threshold: Minimum allowed threshold (aggressive)
            max_threshold: Maximum allowed threshold (conservative)
            adjustment_step: How much to adjust threshold per calibration
            hysteresis: Minimum change before adjusting threshold (prevents oscillation)
            min_decisions_for_calibration: Minimum decisions before calibrating
        """
        self._strategy = strategy
        self._base_threshold = base_threshold
        self._target_accuracy = target_accuracy
        self._min_threshold = min_threshold
        self._max_threshold = max_threshold
        self._adjustment_step = adjustment_step
        self._hysteresis = hysteresis
        self._min_decisions_for_calibration = min_decisions_for_calibration

        # Per-decision-type thresholds and statistics
        self._thresholds: Dict[DecisionType, float] = {}
        self._statistics: Dict[DecisionType, TypeStatistics] = {}

        # Last calibration time per type
        self._last_calibration: Dict[DecisionType, float] = {}

        logger.info(
            "ConfidenceCalibrator initialized: strategy=%s, base_threshold=%.2f, target_accuracy=%.2f",
            strategy.value,
            base_threshold,
            target_accuracy,
        )

    def should_use_llm(
        self,
        decision_type: DecisionType,
        heuristic_confidence: float,
    ) -> bool:
        """Determine if LLM should be used based on calibrated threshold.

        Args:
            decision_type: Type of decision being made
            heuristic_confidence: Confidence of heuristic prediction

        Returns:
            True if LLM should be used, False if heuristic is sufficient
        """
        threshold = self._get_threshold(decision_type)

        # For non-adaptive strategies, use fixed thresholds
        if self._strategy == CalibrationStrategy.CONSERVATIVE:
            return heuristic_confidence < self._max_threshold
        elif self._strategy == CalibrationStrategy.AGGRESSIVE:
            return heuristic_confidence < self._min_threshold
        elif self._strategy == CalibrationStrategy.BALANCED:
            return heuristic_confidence < self._base_threshold

        # Adaptive strategy: use calibrated threshold
        should_use = heuristic_confidence < threshold

        logger.debug(
            "Decision %s: heuristic_confidence=%.2f, threshold=%.2f, use_llm=%s",
            decision_type.value,
            heuristic_confidence,
            threshold,
            should_use,
        )

        return should_use

    def record_outcome(
        self,
        decision_type: DecisionType,
        heuristic_confidence: float,
        used_llm: bool,
        was_correct: bool,
        llm_confidence: Optional[float] = None,
        source: str = "unknown",
    ) -> None:
        """Record the outcome of a decision for calibration.

        Args:
            decision_type: Type of decision that was made
            heuristic_confidence: Original heuristic confidence
            used_llm: Whether LLM was used
            was_correct: Whether the decision was correct
            llm_confidence: Confidence of LLM prediction (if used)
            source: Source of the decision ("lookup", "pattern", "ensemble", etc.)
        """
        record = DecisionRecord(
            decision_type=decision_type,
            timestamp=time.monotonic(),
            heuristic_confidence=heuristic_confidence,
            used_llm=used_llm,
            was_correct=was_correct,
            llm_confidence=llm_confidence,
            source=source,
        )

        # Get or create statistics for this decision type
        if decision_type not in self._statistics:
            self._statistics[decision_type] = TypeStatistics()

        stats = self._statistics[decision_type]
        stats.add_decision(record)

        logger.debug(
            "Recorded outcome for %s: used_llm=%s, correct=%s, recent_accuracy=%.2f",
            decision_type.value,
            used_llm,
            was_correct,
            stats.recent_accuracy,
        )

        # Calibrate threshold if we have enough data
        if stats.total_decisions >= self._min_decisions_for_calibration:
            self._calibrate_threshold(decision_type, stats)

    def get_threshold(self, decision_type: DecisionType) -> float:
        """Get the current threshold for a decision type.

        Args:
            decision_type: Type of decision

        Returns:
            Current confidence threshold
        """
        return self._get_threshold(decision_type)

    def get_statistics(self, decision_type: DecisionType) -> TypeStatistics:
        """Get statistics for a decision type.

        Args:
            decision_type: Type of decision

        Returns:
            Statistics object for the decision type
        """
        if decision_type not in self._statistics:
            self._statistics[decision_type] = TypeStatistics()

        return self._statistics[decision_type]

    def get_all_statistics(self) -> Dict[DecisionType, TypeStatistics]:
        """Get statistics for all decision types.

        Returns:
            Dictionary mapping decision types to their statistics
        """
        return self._statistics.copy()

    def reset_statistics(self, decision_type: Optional[DecisionType] = None) -> None:
        """Reset statistics for a decision type or all types.

        Args:
            decision_type: Specific type to reset, or None to reset all
        """
        if decision_type:
            if decision_type in self._statistics:
                del self._statistics[decision_type]
            if decision_type in self._thresholds:
                del self._thresholds[decision_type]
            if decision_type in self._last_calibration:
                del self._last_calibration[decision_type]
        else:
            self._statistics.clear()
            self._thresholds.clear()
            self._last_calibration.clear()

        logger.info("Reset statistics for %s", decision_type or "all types")

    def _get_threshold(self, decision_type: DecisionType) -> float:
        """Get the current threshold for a decision type."""
        if decision_type not in self._thresholds:
            # Initialize with base threshold
            self._thresholds[decision_type] = self._base_threshold

        return self._thresholds[decision_type]

    def _calibrate_threshold(
        self,
        decision_type: DecisionType,
        stats: TypeStatistics,
    ) -> None:
        """Calibrate threshold based on recent accuracy.

        Adjusts threshold to maintain target accuracy while minimizing
        LLM calls. Uses hysteresis to prevent oscillation.

        Args:
            decision_type: Type of decision to calibrate
            stats: Current statistics for the decision type
        """
        current_threshold = self._get_threshold(decision_type)
        heuristic_accuracy = stats.get_heuristic_accuracy()

        # Need at least min_decisions_for_calibration total decisions to calibrate
        if stats.total_decisions < self._min_decisions_for_calibration:
            return

        # Also need at least 10 heuristic decisions with known outcomes
        heuristic_decisions = [
            d for d in stats.recent_decisions
            if not d.used_llm and d.was_correct is not None
        ]

        if len(heuristic_decisions) < 10:
            return

        # Calculate new threshold based on accuracy
        new_threshold = self._calculate_new_threshold(
            current_threshold,
            heuristic_accuracy,
            stats,
        )

        # Apply hysteresis: only change if difference is significant
        if abs(new_threshold - current_threshold) < self._hysteresis:
            logger.debug(
                "Threshold change below hysteresis threshold for %s: %.3f < %.3f",
                decision_type.value,
                abs(new_threshold - current_threshold),
                self._hysteresis,
            )
            return

        # Clamp to allowed range
        new_threshold = max(self._min_threshold, min(self._max_threshold, new_threshold))

        # Update threshold
        old_threshold = current_threshold
        self._thresholds[decision_type] = new_threshold
        self._last_calibration[decision_type] = time.monotonic()

        logger.info(
            "Calibrated threshold for %s: %.2f -> %.2f (heuristic_accuracy=%.2f, target=%.2f)",
            decision_type.value,
            old_threshold,
            new_threshold,
            heuristic_accuracy,
            self._target_accuracy,
        )

    def _calculate_new_threshold(
        self,
        current_threshold: float,
        heuristic_accuracy: float,
        stats: TypeStatistics,
    ) -> float:
        """Calculate new threshold based on accuracy and strategy.

        Args:
            current_threshold: Current threshold
            heuristic_accuracy: Recent accuracy of heuristic decisions
            stats: Statistics for the decision type

        Returns:
            New threshold value
        """
        # If heuristic accuracy is above target, we can lower threshold
        # (be more aggressive with heuristics)
        if heuristic_accuracy >= self._target_accuracy:
            # Lower threshold to use heuristics more
            return max(
                self._min_threshold,
                current_threshold - self._adjustment_step,
            )

        # If heuristic accuracy is below target, raise threshold
        # (be more conservative, use LLM more)
        if heuristic_accuracy < self._target_accuracy - 0.05:  # 5% tolerance
            # Raise threshold to use LLM more
            return min(
                self._max_threshold,
                current_threshold + self._adjustment_step,
            )

        # Accuracy is within tolerance, keep current threshold
        return current_threshold

    def get_summary(self) -> Dict[str, any]:
        """Get a summary of calibrator state.

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "strategy": self._strategy.value,
            "base_threshold": self._base_threshold,
            "target_accuracy": self._target_accuracy,
            "min_threshold": self._min_threshold,
            "max_threshold": self._max_threshold,
            "decision_types": {},
        }

        for decision_type, stats in self._statistics.items():
            threshold = self._get_threshold(decision_type)
            summary["decision_types"][decision_type.value] = {
                "threshold": threshold,
                "total_decisions": stats.total_decisions,
                "recent_accuracy": stats.recent_accuracy,
                "heuristic_accuracy": stats.get_heuristic_accuracy(),
                "llm_accuracy": stats.get_llm_accuracy(),
            }

        return summary


def create_confidence_calibrator(
    strategy: str = "adaptive",
    **kwargs,
) -> ConfidenceCalibrator:
    """Factory function to create a confidence calibrator.

    Args:
        strategy: Calibration strategy ("conservative", "balanced", "aggressive", "adaptive")
        **kwargs: Additional arguments to pass to ConfidenceCalibrator

    Returns:
        Configured ConfidenceCalibrator instance
    """
    try:
        strategy_enum = CalibrationStrategy(strategy.lower())
    except ValueError:
        logger.warning("Invalid strategy '%s', using 'adaptive'", strategy)
        strategy_enum = CalibrationStrategy.ADAPTIVE

    return ConfidenceCalibrator(strategy=strategy_enum, **kwargs)
