# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Domain events for evaluation and analytics.

This module defines event classes for breaking circular dependencies between
the orchestrator and evaluation components. Events follow the domain event pattern
to enable loose coupling and event-driven communication.

Key Events:
- EvaluationCompletedEvent: Published when evaluation cycle completes
- RLFeedbackEvent: Published when RL feedback signal is generated
- AnalyticsFlushedEvent: Published when analytics are flushed

Example:
    from victor.observability.events import EvaluationCompletedEvent
    from victor.core.events.backends import get_observability_bus

    # Publish event
    bus = get_observability_bus()
    await bus.emit(
        topic="evaluation.completed",
        data={
            "success": True,
            "quality_score": 0.9,
            "tool_calls_used": 5,
        }
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from victor.core.events.protocols import MessagingEvent


class EvaluationEventType(str, Enum):
    """Types of evaluation events."""

    EVALUATION_COMPLETED = "evaluation.completed"
    """Evaluation cycle completed successfully."""

    EVALUATION_FAILED = "evaluation.failed"
    """Evaluation cycle failed."""

    RL_FEEDBACK_GENERATED = "evaluation.rl_feedback"
    """RL feedback signal generated."""

    ANALYTICS_FLUSHED = "evaluation.analytics_flushed"
    """Analytics flushed to storage."""

    OUTCOME_RECORDED = "evaluation.outcome_recorded"
    """Intelligent outcome recorded for learning."""


class ClassificationEventType(str, Enum):
    """Types of classification events."""

    TASK_CLASSIFIED = "classification.task"
    """Task classification completed."""

    INTENT_CLASSIFIED = "classification.intent"
    """Intent classification completed."""

    COMPLEXITY_CLASSIFIED = "classification.complexity"
    """Task complexity classification completed."""

    CLASSIFICATION_ENSEMBLE = "classification.ensemble"
    """Ensemble classification result available."""


@dataclass
class EvaluationCompletedEvent:
    """Event published when evaluation cycle completes.

    This event breaks the circular dependency between orchestrator and
    evaluation coordinator by providing evaluation results via events
    instead of direct method calls.

    Attributes:
        success: Whether evaluation succeeded
        quality_score: Quality score (0.0 to 1.0)
        tool_calls_used: Number of tool calls in session
        tokens_used: Token usage statistics
        session_id: Unique session identifier
        timestamp: Event timestamp

    Example:
        event = EvaluationCompletedEvent(
            success=True,
            quality_score=0.9,
            tool_calls_used=5,
            tokens_used={"input": 1000, "output": 500},
            session_id="session_123",
        )
    """

    success: bool
    quality_score: float
    tool_calls_used: int
    tokens_used: dict[str, int]
    session_id: str
    timestamp: float = field(default_factory=time.time)

    def to_messaging_event(self) -> MessagingEvent:
        """Convert to MessagingEvent for bus emission."""
        return MessagingEvent(
            topic=EvaluationEventType.EVALUATION_COMPLETED.value,
            data={
                "success": self.success,
                "quality_score": self.quality_score,
                "tool_calls_used": self.tool_calls_used,
                "tokens_used": self.tokens_used,
                "session_id": self.session_id,
                "timestamp": self.timestamp,
            },
            source="evaluation_coordinator",
            correlation_id=self.session_id,
        )


@dataclass
class RLFeedbackEvent:
    """Event published when RL feedback signal is generated.

    This event breaks circular dependency by providing RL feedback via
    events instead of direct method calls to RL coordinator.

    Attributes:
        session_id: Session identifier
        reward: Reward signal for RL
        outcome_type: Type of outcome (success, failure, timeout, etc.)
        provider: Provider used
        model: Model used
        latency_ms: Request latency in milliseconds
        tool_calls: Number of tool calls
        timestamp: Event timestamp

    Example:
        event = RLFeedbackEvent(
            session_id="session_123",
            reward=1.0,
            outcome_type="success",
            provider="anthropic",
            model="claude-sonnet-4-5",
            latency_ms=1500,
            tool_calls=5,
        )
    """

    session_id: str
    reward: float
    outcome_type: str
    provider: str
    model: str
    latency_ms: int
    tool_calls: int
    timestamp: float = field(default_factory=time.time)

    def to_messaging_event(self) -> MessagingEvent:
        """Convert to MessagingEvent for bus emission."""
        return MessagingEvent(
            topic=EvaluationEventType.RL_FEEDBACK_GENERATED.value,
            data={
                "session_id": self.session_id,
                "reward": self.reward,
                "outcome_type": self.outcome_type,
                "provider": self.provider,
                "model": self.model,
                "latency_ms": self.latency_ms,
                "tool_calls": self.tool_calls,
                "timestamp": self.timestamp,
            },
            source="evaluation_coordinator",
            correlation_id=self.session_id,
        )


@dataclass
class AnalyticsFlushedEvent:
    """Event published when analytics are flushed.

    This event notifies listeners that analytics data has been persisted.

    Attributes:
        records_written: Number of records written
        storage_backend: Storage backend used
        flush_duration_ms: Time taken to flush
        timestamp: Event timestamp

    Example:
        event = AnalyticsFlushedEvent(
            records_written=100,
            storage_backend="sqlite",
            flush_duration_ms=50,
        )
    """

    records_written: int
    storage_backend: str
    flush_duration_ms: int
    timestamp: float = field(default_factory=time.time)

    def to_messaging_event(self) -> MessagingEvent:
        """Convert to MessagingEvent for bus emission."""
        return MessagingEvent(
            topic=EvaluationEventType.ANALYTICS_FLUSHED.value,
            data={
                "records_written": self.records_written,
                "storage_backend": self.storage_backend,
                "flush_duration_ms": self.flush_duration_ms,
                "timestamp": self.timestamp,
            },
            source="evaluation_coordinator",
        )


@dataclass
class OutcomeRecordedEvent:
    """Event published when intelligent outcome is recorded.

    This event is used for Q-learning and reinforcement learning.

    Attributes:
        session_id: Session identifier
        success: Whether task succeeded
        quality_score: Quality assessment
        provider: Provider used
        model: Model used
        tool_calls: Number of tool calls
        outcome_data: Additional outcome data
        timestamp: Event timestamp

    Example:
        event = OutcomeRecordedEvent(
            session_id="session_123",
            success=True,
            quality_score=0.9,
            provider="anthropic",
            model="claude-sonnet-4-5",
            tool_calls=5,
            outcome_data={"latency_ms": 1500},
        )
    """

    session_id: str
    success: bool
    quality_score: float
    provider: str
    model: str
    tool_calls: int
    outcome_data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_messaging_event(self) -> MessagingEvent:
        """Convert to MessagingEvent for bus emission."""
        return MessagingEvent(
            topic=EvaluationEventType.OUTCOME_RECORDED.value,
            data={
                "session_id": self.session_id,
                "success": self.success,
                "quality_score": self.quality_score,
                "provider": self.provider,
                "model": self.model,
                "tool_calls": self.tool_calls,
                "outcome_data": self.outcome_data,
                "timestamp": self.timestamp,
            },
            source="evaluation_coordinator",
            correlation_id=self.session_id,
        )


# =============================================================================
# Classification Events
# =============================================================================


@dataclass
class TaskClassifiedEvent:
    """Event published when task classification completes.

    This event breaks the circular dependency between semantic_selector and
    unified_classifier by providing classification results via events instead
    of direct imports.

    Attributes:
        query: The original query/prompt
        task_type: The classified task type (enum value as string)
        confidence: Confidence score (0.0 to 1.0)
        is_action_task: Whether this is an action-oriented task
        is_analysis_task: Whether this is an analysis-oriented task
        is_generation_task: Whether this is a generation-oriented task
        method: Classification method used (keyword, semantic, ensemble)
        timestamp: Event timestamp

    Example:
        event = TaskClassifiedEvent(
            query="Create a new user authentication system",
            task_type="CREATE",
            confidence=0.92,
            is_action_task=True,
            is_analysis_task=False,
            is_generation_task=True,
            method="semantic",
        )
    """

    query: str
    task_type: str
    confidence: float
    is_action_task: bool
    is_analysis_task: bool
    is_generation_task: bool
    method: str = "keyword"
    timestamp: float = field(default_factory=time.time)

    def to_messaging_event(self) -> MessagingEvent:
        """Convert to MessagingEvent for bus emission."""
        return MessagingEvent(
            topic=ClassificationEventType.TASK_CLASSIFIED.value,
            data={
                "query": self.query,
                "task_type": self.task_type,
                "confidence": self.confidence,
                "is_action_task": self.is_action_task,
                "is_analysis_task": self.is_analysis_task,
                "is_generation_task": self.is_generation_task,
                "method": self.method,
                "timestamp": self.timestamp,
            },
            source="task_classifier",
        )


@dataclass
class IntentClassifiedEvent:
    """Event published when intent classification completes.

    This event provides intent classification results without direct imports.

    Attributes:
        query: The original query/prompt
        intent_type: The classified intent type (enum value as string)
        confidence: Confidence score (0.0 to 1.0)
        requires_confirmation: Whether user confirmation is required
        can_write_files: Whether this intent allows file writes
        timestamp: Event timestamp

    Example:
        event = IntentClassifiedEvent(
            query="Delete all test files",
            intent_type="DESTRUCTIVE",
            confidence=0.95,
            requires_confirmation=True,
            can_write_files=True,
        )
    """

    query: str
    intent_type: str
    confidence: float
    requires_confirmation: bool
    can_write_files: bool
    timestamp: float = field(default_factory=time.time)

    def to_messaging_event(self) -> MessagingEvent:
        """Convert to MessagingEvent for bus emission."""
        return MessagingEvent(
            topic=ClassificationEventType.INTENT_CLASSIFIED.value,
            data={
                "query": self.query,
                "intent_type": self.intent_type,
                "confidence": self.confidence,
                "requires_confirmation": self.requires_confirmation,
                "can_write_files": self.can_write_files,
                "timestamp": self.timestamp,
            },
            source="intent_classifier",
        )


@dataclass
class ComplexityClassifiedEvent:
    """Event published when task complexity classification completes.

    This event provides complexity assessment without direct imports.

    Attributes:
        query: The original query/prompt
        complexity: The complexity level (SIMPLE, COMPLEX, GENERATION)
        confidence: Confidence score (0.0 to 1.0)
        estimated_steps: Estimated number of steps to complete
        tool_budget: Recommended tool budget
        timestamp: Event timestamp

    Example:
        event = ComplexityClassifiedEvent(
            query="Build a complete REST API",
            complexity="COMPLEX",
            confidence=0.88,
            estimated_steps=15,
            tool_budget=20,
        )
    """

    query: str
    complexity: str
    confidence: float
    estimated_steps: int
    tool_budget: int
    timestamp: float = field(default_factory=time.time)

    def to_messaging_event(self) -> MessagingEvent:
        """Convert to MessagingEvent for bus emission."""
        return MessagingEvent(
            topic=ClassificationEventType.COMPLEXITY_CLASSIFIED.value,
            data={
                "query": self.query,
                "complexity": self.complexity,
                "confidence": self.confidence,
                "estimated_steps": self.estimated_steps,
                "tool_budget": self.tool_budget,
                "timestamp": self.timestamp,
            },
            source="complexity_classifier",
        )


@dataclass
class ClassificationEnsembleEvent:
    """Event published when ensemble classification result is available.

    This event combines multiple classification results (task, intent, complexity)
    into a single ensemble result, breaking circular dependencies between
    task_analyzer and multiple classifiers.

    Attributes:
        query: The original query/prompt
        session_id: Session identifier
        task_type: Task classification result (as dict)
        intent_type: Intent classification result (as dict, optional)
        complexity: Complexity classification result (as dict)
        unified_type: Unified task type (enum value as string)
        confidence: Overall ensemble confidence
        tool_budget: Recommended tool budget
        requires_confirmation: Whether confirmation is required
        matched_patterns: List of matched patterns
        timestamp: Event timestamp

    Example:
        event = ClassificationEnsembleEvent(
            query="Create a user authentication system",
            session_id="session_123",
            task_type={"type": "CREATE", "confidence": 0.9},
            intent_type={"type": "CONSTRUCTIVE", "confidence": 0.85},
            complexity={"level": "COMPLEX", "confidence": 0.88, "tool_budget": 15},
            unified_type="CODING_GENERATION",
            confidence=0.88,
            tool_budget=15,
            requires_confirmation=False,
            matched_patterns=["create", "system"],
        )
    """

    query: str
    session_id: str
    task_type: dict[str, Any]
    intent_type: Optional[dict[str, Any]] = None
    complexity: Optional[dict[str, Any]] = None
    unified_type: str = "UNKNOWN"
    confidence: float = 0.5
    tool_budget: int = 5
    requires_confirmation: bool = False
    matched_patterns: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_messaging_event(self) -> MessagingEvent:
        """Convert to MessagingEvent for bus emission."""
        return MessagingEvent(
            topic=ClassificationEventType.CLASSIFICATION_ENSEMBLE.value,
            data={
                "query": self.query,
                "session_id": self.session_id,
                "task_type": self.task_type,
                "intent_type": self.intent_type,
                "complexity": self.complexity,
                "unified_type": self.unified_type,
                "confidence": self.confidence,
                "tool_budget": self.tool_budget,
                "requires_confirmation": self.requires_confirmation,
                "matched_patterns": self.matched_patterns,
                "timestamp": self.timestamp,
            },
            source="task_analyzer",
            correlation_id=self.session_id,
        )


# =============================================================================
# Event Factory Functions
# =============================================================================


def create_evaluation_completed_event(
    success: bool,
    quality_score: float,
    tool_calls_used: int,
    tokens_used: dict[str, int],
    session_id: str,
) -> EvaluationCompletedEvent:
    """Create an evaluation completed event.

    Args:
        success: Whether evaluation succeeded
        quality_score: Quality score (0.0 to 1.0)
        tool_calls_used: Number of tool calls
        tokens_used: Token usage statistics
        session_id: Session identifier

    Returns:
        EvaluationCompletedEvent instance
    """
    return EvaluationCompletedEvent(
        success=success,
        quality_score=quality_score,
        tool_calls_used=tool_calls_used,
        tokens_used=tokens_used,
        session_id=session_id,
    )


def create_rl_feedback_event(
    session_id: str,
    reward: float,
    outcome_type: str,
    provider: str,
    model: str,
    latency_ms: int,
    tool_calls: int,
) -> RLFeedbackEvent:
    """Create an RL feedback event.

    Args:
        session_id: Session identifier
        reward: Reward signal
        outcome_type: Type of outcome
        provider: Provider name
        model: Model name
        latency_ms: Latency in milliseconds
        tool_calls: Number of tool calls

    Returns:
        RLFeedbackEvent instance
    """
    return RLFeedbackEvent(
        session_id=session_id,
        reward=reward,
        outcome_type=outcome_type,
        provider=provider,
        model=model,
        latency_ms=latency_ms,
        tool_calls=tool_calls,
    )


def create_analytics_flushed_event(
    records_written: int,
    storage_backend: str,
    flush_duration_ms: int,
) -> AnalyticsFlushedEvent:
    """Create an analytics flushed event.

    Args:
        records_written: Number of records written
        storage_backend: Storage backend name
        flush_duration_ms: Flush duration in milliseconds

    Returns:
        AnalyticsFlushedEvent instance
    """
    return AnalyticsFlushedEvent(
        records_written=records_written,
        storage_backend=storage_backend,
        flush_duration_ms=flush_duration_ms,
    )


def create_outcome_recorded_event(
    session_id: str,
    success: bool,
    quality_score: float,
    provider: str,
    model: str,
    tool_calls: int,
    outcome_data: Optional[dict[str, Any]] = None,
) -> OutcomeRecordedEvent:
    """Create an outcome recorded event.

    Args:
        session_id: Session identifier
        success: Whether outcome was successful
        quality_score: Quality score
        provider: Provider name
        model: Model name
        tool_calls: Number of tool calls
        outcome_data: Additional outcome data

    Returns:
        OutcomeRecordedEvent instance
    """
    return OutcomeRecordedEvent(
        session_id=session_id,
        success=success,
        quality_score=quality_score,
        provider=provider,
        model=model,
        tool_calls=tool_calls,
        outcome_data=outcome_data or {},
    )


def create_task_classified_event(
    query: str,
    task_type: str,
    confidence: float,
    is_action_task: bool = False,
    is_analysis_task: bool = False,
    is_generation_task: bool = False,
    method: str = "keyword",
) -> TaskClassifiedEvent:
    """Create a task classified event.

    Args:
        query: The original query/prompt
        task_type: The classified task type (enum value as string)
        confidence: Confidence score (0.0 to 1.0)
        is_action_task: Whether this is an action-oriented task
        is_analysis_task: Whether this is an analysis-oriented task
        is_generation_task: Whether this is a generation-oriented task
        method: Classification method used

    Returns:
        TaskClassifiedEvent instance
    """
    return TaskClassifiedEvent(
        query=query,
        task_type=task_type,
        confidence=confidence,
        is_action_task=is_action_task,
        is_analysis_task=is_analysis_task,
        is_generation_task=is_generation_task,
        method=method,
    )


def create_intent_classified_event(
    query: str,
    intent_type: str,
    confidence: float,
    requires_confirmation: bool = False,
    can_write_files: bool = False,
) -> IntentClassifiedEvent:
    """Create an intent classified event.

    Args:
        query: The original query/prompt
        intent_type: The classified intent type (enum value as string)
        confidence: Confidence score (0.0 to 1.0)
        requires_confirmation: Whether user confirmation is required
        can_write_files: Whether this intent allows file writes

    Returns:
        IntentClassifiedEvent instance
    """
    return IntentClassifiedEvent(
        query=query,
        intent_type=intent_type,
        confidence=confidence,
        requires_confirmation=requires_confirmation,
        can_write_files=can_write_files,
    )


def create_complexity_classified_event(
    query: str,
    complexity: str,
    confidence: float,
    estimated_steps: int,
    tool_budget: int,
) -> ComplexityClassifiedEvent:
    """Create a complexity classified event.

    Args:
        query: The original query/prompt
        complexity: The complexity level (SIMPLE, COMPLEX, GENERATION)
        confidence: Confidence score (0.0 to 1.0)
        estimated_steps: Estimated number of steps to complete
        tool_budget: Recommended tool budget

    Returns:
        ComplexityClassifiedEvent instance
    """
    return ComplexityClassifiedEvent(
        query=query,
        complexity=complexity,
        confidence=confidence,
        estimated_steps=estimated_steps,
        tool_budget=tool_budget,
    )


def create_classification_ensemble_event(
    query: str,
    session_id: str,
    task_type: dict[str, Any],
    intent_type: Optional[dict[str, Any]] = None,
    complexity: Optional[dict[str, Any]] = None,
    unified_type: str = "UNKNOWN",
    confidence: float = 0.5,
    tool_budget: int = 5,
    requires_confirmation: bool = False,
    matched_patterns: Optional[list[str]] = None,
) -> ClassificationEnsembleEvent:
    """Create a classification ensemble event.

    Args:
        query: The original query/prompt
        session_id: Session identifier
        task_type: Task classification result (as dict)
        intent_type: Intent classification result (as dict, optional)
        complexity: Complexity classification result (as dict)
        unified_type: Unified task type (enum value as string)
        confidence: Overall ensemble confidence
        tool_budget: Recommended tool budget
        requires_confirmation: Whether confirmation is required
        matched_patterns: List of matched patterns

    Returns:
        ClassificationEnsembleEvent instance
    """
    return ClassificationEnsembleEvent(
        query=query,
        session_id=session_id,
        task_type=task_type,
        intent_type=intent_type,
        complexity=complexity,
        unified_type=unified_type,
        confidence=confidence,
        tool_budget=tool_budget,
        requires_confirmation=requires_confirmation,
        matched_patterns=matched_patterns or [],
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Event types
    "EvaluationEventType",
    "ClassificationEventType",
    # Event classes - Evaluation
    "EvaluationCompletedEvent",
    "RLFeedbackEvent",
    "AnalyticsFlushedEvent",
    "OutcomeRecordedEvent",
    # Event classes - Classification
    "TaskClassifiedEvent",
    "IntentClassifiedEvent",
    "ComplexityClassifiedEvent",
    "ClassificationEnsembleEvent",
    # Factory functions - Evaluation
    "create_evaluation_completed_event",
    "create_rl_feedback_event",
    "create_analytics_flushed_event",
    "create_outcome_recorded_event",
    # Factory functions - Classification
    "create_task_classified_event",
    "create_intent_classified_event",
    "create_complexity_classified_event",
    "create_classification_ensemble_event",
]
