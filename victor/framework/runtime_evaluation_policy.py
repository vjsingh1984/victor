# Copyright 2026 Vijaykumar Singh <singhvijay@users.noreply.github.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Canonical runtime evaluation-policy object for live agent execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class ClarificationDecision:
    """Typed clarification policy outcome for a perception result."""

    requires_clarification: bool = False
    reason: Optional[str] = None
    prompt: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Return compatibility mapping for existing perception callers."""
        return {
            "needs_clarification": self.requires_clarification,
            "clarification_reason": self.reason,
            "clarification_prompt": self.prompt,
        }


@dataclass(frozen=True)
class RuntimeEvaluationPolicy:
    """Shared thresholds and wording for runtime evaluation decisions."""

    clarification_confidence_threshold: float = 0.45
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.5
    low_confidence_retry_limit: int = 2

    default_clarification_prompt: str = (
        "Please clarify the target file, component, or bug before I continue."
    )
    fallback_clarification_reason: str = "task details are incomplete"
    empty_request_reason: str = "request is empty"
    empty_request_prompt: str = "What would you like me to do?"
    underspecified_target_reason: str = "target artifact or scope is underspecified"
    underspecified_target_prompt: str = "Which file, component, or bug should I target first?"
    confirmation_reason: str = "task intent requires confirmation"
    confirmation_prompt: str = (
        "Should I modify files directly, or keep this read-only and provide guidance first?"
    )

    high_confidence_reason: str = "High confidence in perception"
    medium_confidence_reason: str = "Medium confidence - continue"
    low_confidence_reason: str = "Low confidence - retry"
    retry_exhausted_reason_template: str = (
        "Low confidence retry budget exhausted after {retry_count} retries"
    )

    @classmethod
    def from_config(
        cls,
        config: Optional[Mapping[str, Any]] = None,
    ) -> "RuntimeEvaluationPolicy":
        """Build a policy object from a flat runtime config mapping."""
        if not config:
            return cls()

        allowed = {field.name for field in fields(cls)}
        overrides = {key: value for key, value in config.items() if key in allowed}
        return cls(**overrides)

    def to_config(self) -> Dict[str, Any]:
        """Return config-compatible representation of the policy."""
        return asdict(self)

    def empty_request_decision(self, confidence: float = 0.0) -> ClarificationDecision:
        """Return the canonical clarification response for empty requests."""
        return ClarificationDecision(
            requires_clarification=True,
            reason=self.empty_request_reason,
            prompt=self.empty_request_prompt,
            confidence=confidence,
        )

    def underspecified_target_decision(self, confidence: float) -> ClarificationDecision:
        """Return the canonical clarification response for missing targets."""
        return ClarificationDecision(
            requires_clarification=True,
            reason=self.underspecified_target_reason,
            prompt=self.underspecified_target_prompt,
            confidence=confidence,
        )

    def confirmation_required_decision(self, confidence: float) -> ClarificationDecision:
        """Return the canonical clarification response for confirmation-needed tasks."""
        return ClarificationDecision(
            requires_clarification=True,
            reason=self.confirmation_reason,
            prompt=self.confirmation_prompt,
            confidence=confidence,
        )

    def get_clarification_decision(self, perception: Optional[Any]) -> ClarificationDecision:
        """Normalize clarification policy into one typed runtime decision."""
        if not getattr(perception, "needs_clarification", False):
            return ClarificationDecision(
                requires_clarification=False,
                confidence=float(getattr(perception, "confidence", 0.0) or 0.0),
            )

        reason = getattr(perception, "clarification_reason", None) or self.fallback_clarification_reason
        prompt = getattr(perception, "clarification_prompt", None) or self.default_clarification_prompt
        confidence = float(getattr(perception, "confidence", 0.0) or 0.0)
        return ClarificationDecision(
            requires_clarification=True,
            reason=reason,
            prompt=prompt,
            confidence=confidence,
        )

    def get_confidence_evaluation(self, confidence: float) -> Any:
        """Emit the canonical confidence-band evaluation without mutating retry state."""
        from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult

        if confidence >= self.high_confidence_threshold:
            return EvaluationResult(
                decision=EvaluationDecision.COMPLETE,
                score=confidence,
                reason=self.high_confidence_reason,
            )

        if confidence >= self.medium_confidence_threshold:
            return EvaluationResult(
                decision=EvaluationDecision.CONTINUE,
                score=confidence,
                reason=self.medium_confidence_reason,
            )

        return EvaluationResult(
            decision=EvaluationDecision.RETRY,
            score=confidence,
            reason=self.low_confidence_reason,
        )

    def apply_retry_budget(
        self,
        evaluation: Any,
        state: Dict[str, Any],
        *,
        retry_limit: Optional[int] = None,
        low_confidence_threshold: Optional[float] = None,
    ) -> Any:
        """Apply the canonical retry-budget policy to low-confidence retry results."""
        from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult

        if getattr(evaluation, "should_complete", False) or getattr(
            evaluation, "should_continue", False
        ):
            if "low_confidence_retries" in state:
                state["low_confidence_retries"] = 0
            return evaluation

        threshold = (
            self.medium_confidence_threshold
            if low_confidence_threshold is None
            else low_confidence_threshold
        )
        if not getattr(evaluation, "should_retry", False) or evaluation.score >= threshold:
            return evaluation

        effective_retry_limit = self.low_confidence_retry_limit if retry_limit is None else retry_limit
        effective_retry_limit = max(int(effective_retry_limit), 0)
        retry_count = int(state.get("low_confidence_retries", 0))
        if retry_count >= effective_retry_limit:
            return EvaluationResult(
                decision=EvaluationDecision.FAIL,
                score=evaluation.score,
                reason=self.retry_exhausted_reason_template.format(retry_count=retry_count),
                metrics=dict(evaluation.metrics),
                metadata={
                    **dict(evaluation.metadata),
                    "low_confidence_retry_exhausted": True,
                    "low_confidence_retries": retry_count,
                    "low_confidence_retry_limit": effective_retry_limit,
                },
            )

        retry_count += 1
        state["low_confidence_retries"] = retry_count
        return EvaluationResult(
            decision=EvaluationDecision.RETRY,
            score=evaluation.score,
            reason=evaluation.reason,
            metrics=dict(evaluation.metrics),
            metadata={
                **dict(evaluation.metadata),
                "low_confidence_retries": retry_count,
                "low_confidence_retry_limit": effective_retry_limit,
            },
        )

    def evaluate_confidence_progress(
        self,
        confidence: float,
        state: Dict[str, Any],
        *,
        retry_limit: Optional[int] = None,
    ) -> Any:
        """Apply the canonical confidence-band policy for live-loop evaluation."""
        evaluation = self.get_confidence_evaluation(confidence)

        if getattr(evaluation, "should_complete", False) or getattr(
            evaluation, "should_continue", False
        ):
            if "low_confidence_retries" in state:
                state["low_confidence_retries"] = 0
            return evaluation

        return self.apply_retry_budget(
            evaluation,
            state,
            retry_limit=retry_limit,
            low_confidence_threshold=self.medium_confidence_threshold,
        )
