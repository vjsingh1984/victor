"""Completion signal fusion for the agentic loop DECIDE phase.

Wave 3: aggregates the four completion signals (fulfillment, requirement,
keyword, confidence) into a single FuserResult with velocity tracking.
Velocity prevents premature COMPLETE on score backslide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

_WEIGHTS = {
    "fulfillment": 0.35,
    "requirement": 0.30,
    "keyword": 0.20,
    "confidence": 0.15,
}

_DEFAULT_COMPLETION_THRESHOLD = 0.80
_DEFAULT_BACKSLIDE_THRESHOLD = -0.10


@dataclass
class CompletionSignalFuserConfig:
    """Typed configuration for CompletionSignalFuser with weight sum validation.

    Weights (fulfillment + requirement + keyword + confidence) must sum to 1.0 ±0.01.
    """

    fulfillment_weight: float = 0.35
    requirement_weight: float = 0.30
    keyword_weight: float = 0.20
    confidence_weight: float = 0.15
    completion_threshold: float = 0.80
    backslide_threshold: float = -0.10

    def __post_init__(self) -> None:
        total = (
            self.fulfillment_weight
            + self.requirement_weight
            + self.keyword_weight
            + self.confidence_weight
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"CompletionSignalFuserConfig weights must sum to 1.0 (±0.01), "
                f"got {total:.4f}. "
                f"Adjust fulfillment+requirement+keyword+confidence weights."
            )


@dataclass
class FuserResult:
    """Aggregated completion signal from CompletionSignalFuser.

    Attributes:
        score:        Weighted fusion score (0.0–1.0)
        decision:     "complete" | "continue" | "retry"
        reason:       Human-readable explanation
        velocity:     score[n] – score[n-1]; 0.0 on first turn
        signals_used: Per-signal scores for observability
    """

    score: float
    decision: str
    reason: str
    velocity: float = 0.0
    signals_used: Dict[str, float] = field(default_factory=dict)


class CompletionSignalFuser:
    """Aggregates completion signals using weighted fusion with velocity tracking.

    Args:
        completion_threshold: Minimum score to yield ``"complete"`` (default 0.80).
        backslide_threshold:  Velocity below this blocks ``"complete"`` even when
                               score exceeds the threshold (default -0.10).
        weights:              Per-signal weight overrides; missing keys fall back to
                               built-in defaults.
    """

    def __init__(
        self,
        completion_threshold: float = _DEFAULT_COMPLETION_THRESHOLD,
        backslide_threshold: float = _DEFAULT_BACKSLIDE_THRESHOLD,
        weights: Optional[Dict[str, float]] = None,
        config: Optional["CompletionSignalFuserConfig"] = None,
    ) -> None:
        if config is not None:
            # Typed config takes precedence over individual kwargs
            self._completion_threshold = config.completion_threshold
            self._backslide_threshold = config.backslide_threshold
            self._weights = {
                "fulfillment": config.fulfillment_weight,
                "requirement": config.requirement_weight,
                "keyword": config.keyword_weight,
                "confidence": config.confidence_weight,
            }
        else:
            self._completion_threshold = completion_threshold
            self._backslide_threshold = backslide_threshold
            self._weights = dict(_WEIGHTS)
            if weights:
                self._weights.update(weights)

    def fuse(
        self,
        fulfillment: float,
        requirement: float,
        keyword: float,
        confidence: float,
        score_history: List[float],
    ) -> FuserResult:
        """Compute a fused completion decision.

        Args:
            fulfillment:   FulfillmentDetector score (0.0–1.0)
            requirement:   RequirementValidator satisfaction score (0.0–1.0)
            keyword:       ContextAwareKeywordDetector score (0.0–1.0)
            confidence:    Calibrated confidence score (0.0–1.0)
            score_history: Ordered list of previous fused scores (most recent last).

        Returns:
            FuserResult with score, decision, velocity, and per-signal breakdown.
        """
        signals: Dict[str, float] = {
            "fulfillment": max(0.0, min(1.0, fulfillment)),
            "requirement": max(0.0, min(1.0, requirement)),
            "keyword": max(0.0, min(1.0, keyword)),
            "confidence": max(0.0, min(1.0, confidence)),
        }

        score = sum(signals[k] * self._weights.get(k, 0.0) for k in signals)
        total_weight = sum(self._weights.get(k, 0.0) for k in signals)
        if total_weight > 0:
            score = score / total_weight

        velocity = 0.0
        if score_history:
            velocity = score - score_history[-1]

        backslide = velocity < self._backslide_threshold

        if score >= self._completion_threshold and not backslide:
            decision = "complete"
            reason = (
                f"Fused score {score:.2f} ≥ threshold {self._completion_threshold:.2f}"
            )
        elif score >= self._completion_threshold and backslide:
            decision = "continue"
            reason = (
                f"Score {score:.2f} high but velocity {velocity:.2f} indicates backslide "
                f"(threshold {self._backslide_threshold:.2f}); deferring completion"
            )
        else:
            decision = "continue"
            reason = (
                f"Fused score {score:.2f} < threshold {self._completion_threshold:.2f}"
            )

        return FuserResult(
            score=score,
            decision=decision,
            reason=reason,
            velocity=velocity,
            signals_used=signals,
        )
