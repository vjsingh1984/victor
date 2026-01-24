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

"""Reinforcement Learning - Framework-level adaptive learning infrastructure.

This module provides Victor's RL system for learning optimal parameters
from experience across all verticals.

Architecture:
┌──────────────────────────────────────────────────────────────────────┐
│                    HIGH-LEVEL API                                     │
│  RLManager, LearnerType, convenience functions                        │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ wraps
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    RLCoordinator (Singleton)                          │
│  ├─ Learner registry                                                  │
│  ├─ Unified SQLite storage (~/.victor/rl_data/rl.db)                 │
│  ├─ Telemetry collection                                              │
│  └─ Cross-vertical learning                                           │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ manages
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Specialized Learners                               │
│  ├─ ToolSelectorLearner        ├─ ContinuationPatienceLearner        │
│  ├─ ContinuationPromptLearner  ├─ SemanticThresholdLearner           │
│  ├─ ModelSelectorLearner       ├─ GroundingThresholdLearner          │
│  ├─ ModeTransitionLearner      ├─ PromptTemplateLearner              │
│  ├─ QualityWeightsLearner      └─ CacheEvictionLearner               │
└──────────────────────────────────────────────────────────────────────┘

Key Components:
- RLCoordinator: Central coordinator for all learners
- BaseLearner: Abstract base class for all RL learners
- RLManager: High-level API for using RL in applications
- LearnerType: Enum of available learner types

Usage:
    from victor.framework.rl import RLManager, LearnerType, get_rl_coordinator

    # Option 1: Use high-level RLManager
    rl = RLManager()
    rl.record_success(
        learner=LearnerType.TOOL_SELECTOR,
        provider="anthropic",
        model="claude-3-opus",
        task_type="analysis",
    )

    rec = rl.get_recommendation(
        learner=LearnerType.CONTINUATION_PATIENCE,
        provider="deepseek",
        model="deepseek-chat",
    )

    # Option 2: Use coordinator directly
    coordinator = get_rl_coordinator()
    coordinator.record_outcome(
        learner_name="continuation_patience",
        outcome=RLOutcome(...),
        vertical="coding",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.framework.rl.coordinator import RLCoordinator, get_rl_coordinator

if TYPE_CHECKING:
    from victor.framework.agent import Agent
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator

logger = logging.getLogger(__name__)


class LearnerType(str, Enum):
    """Available learner types in the RL system.

    Each learner optimizes a different aspect of agent behavior.
    """

    # Tool-related learners
    TOOL_SELECTOR = "tool_selector"
    """Optimizes tool selection based on task context."""

    CACHE_EVICTION = "cache_eviction"
    """Learns optimal cache eviction strategies."""

    # Continuation learners
    CONTINUATION_PATIENCE = "continuation_patience"
    """Adjusts patience for model continuation attempts."""

    CONTINUATION_PROMPTS = "continuation_prompts"
    """Optimizes continuation prompt templates."""

    # Quality learners
    GROUNDING_THRESHOLD = "grounding_threshold"
    """Adjusts grounding verification thresholds."""

    SEMANTIC_THRESHOLD = "semantic_threshold"
    """Optimizes semantic similarity thresholds."""

    QUALITY_WEIGHTS = "quality_weights"
    """Learns quality assessment weights."""

    # Model/mode learners
    MODEL_SELECTOR = "model_selector"
    """Recommends optimal model for task type."""

    MODE_TRANSITION = "mode_transition"
    """Learns optimal mode transitions."""

    # Prompt learners
    PROMPT_TEMPLATE = "prompt_template"
    """Optimizes prompt template selection."""


# Import config after LearnerType to avoid circular import
from victor.framework.rl.config import BaseRLConfig, DEFAULT_ACTIVE_LEARNERS, DEFAULT_PATIENCE_MAP


@dataclass
class LearnerStats:
    """Statistics for a specific learner.

    Attributes:
        name: Learner name
        total_records: Total number of recorded outcomes
        success_rate: Ratio of successful outcomes
        last_updated: Unix timestamp of last update
        parameters: Current parameter values
    """

    name: str
    total_records: int = 0
    success_rate: float = 0.0
    last_updated: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RLStats:
    """Aggregate statistics for the RL system.

    Attributes:
        total_outcomes: Total outcomes recorded across all learners
        active_learners: Number of active learners
        learner_stats: Per-learner statistics
        database_path: Path to SQLite database
    """

    total_outcomes: int = 0
    active_learners: int = 0
    learner_stats: Dict[str, LearnerStats] = field(default_factory=dict)
    database_path: Optional[str] = None


class RLManager:
    """High-level manager for reinforcement learning operations.

    RLManager provides a user-friendly interface to Victor's RL system,
    wrapping the RLCoordinator with simplified methods.

    Attributes:
        coordinator: Underlying RLCoordinator instance

    Example:
        # Create from agent
        rl = RLManager.from_agent(agent)

        # Record outcome
        rl.record_success(
            learner=LearnerType.TOOL_SELECTOR,
            context={"tool": "code_search"},
        )

        # Get recommendation
        rec = rl.get_recommendation(
            learner=LearnerType.CONTINUATION_PATIENCE,
            provider="anthropic",
        )
    """

    def __init__(self, coordinator: Optional[RLCoordinator] = None):
        """Initialize RLManager.

        Args:
            coordinator: Optional RLCoordinator. If not provided,
                uses the global singleton.
        """
        self._coordinator = coordinator or get_rl_coordinator()

    @classmethod
    def from_agent(cls, agent: "Agent") -> "RLManager":
        """Create RLManager from an Agent instance.

        Args:
            agent: Agent instance

        Returns:
            RLManager configured for the agent
        """
        return cls()

    @classmethod
    def from_orchestrator(cls, orchestrator: "AgentOrchestrator") -> "RLManager":
        """Create RLManager from an AgentOrchestrator.

        Args:
            orchestrator: AgentOrchestrator instance

        Returns:
            RLManager configured for the orchestrator
        """
        return cls()

    @property
    def coordinator(self) -> RLCoordinator:
        """Get underlying RLCoordinator."""
        return self._coordinator

    # =========================================================================
    # Recording Outcomes
    # =========================================================================

    def record_outcome(
        self,
        learner: Union[LearnerType, str],
        outcome: RLOutcome,
        *,
        vertical: str = "general",
    ) -> None:
        """Record an outcome for a specific learner.

        Args:
            learner: Learner type or name
            outcome: RLOutcome with success/metrics data
            vertical: Vertical context (coding, devops, etc.)
        """
        learner_name = learner.value if isinstance(learner, LearnerType) else learner
        self._coordinator.record_outcome(
            learner_name=learner_name,
            outcome=outcome,
            vertical=vertical,
        )

    def record_success(
        self,
        learner: Union[LearnerType, str],
        *,
        provider: str = "unknown",
        model: str = "unknown",
        task_type: str = "general",
        quality_score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        vertical: str = "coding",
    ) -> None:
        """Record a successful outcome.

        Convenience method for recording successful outcomes.

        Args:
            learner: Learner type or name
            provider: LLM provider name
            model: Model name
            task_type: Type of task
            quality_score: Quality score 0.0-1.0
            metadata: Additional metadata
            vertical: Vertical context
        """
        outcome = RLOutcome(
            provider=provider,
            model=model,
            task_type=task_type,
            success=True,
            quality_score=quality_score,
            metadata=metadata or {},
            vertical=vertical,
        )
        self.record_outcome(learner, outcome, vertical=vertical)

    def record_failure(
        self,
        learner: Union[LearnerType, str],
        *,
        provider: str = "unknown",
        model: str = "unknown",
        task_type: str = "general",
        quality_score: float = 0.0,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        vertical: str = "coding",
    ) -> None:
        """Record a failed outcome.

        Convenience method for recording failed outcomes.

        Args:
            learner: Learner type or name
            provider: LLM provider name
            model: Model name
            task_type: Type of task
            quality_score: Quality score 0.0-1.0
            error: Error message
            metadata: Additional metadata
            vertical: Vertical context
        """
        meta = metadata or {}
        if error:
            meta["error"] = error
        outcome = RLOutcome(
            provider=provider,
            model=model,
            task_type=task_type,
            success=False,
            quality_score=quality_score,
            metadata=meta,
            vertical=vertical,
        )
        self.record_outcome(learner, outcome, vertical=vertical)

    # =========================================================================
    # Getting Recommendations
    # =========================================================================

    def get_recommendation(
        self,
        learner: Union[LearnerType, str],
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        task_type: Optional[str] = None,
        vertical: str = "general",
        **context: Any,
    ) -> Optional[RLRecommendation]:
        """Get a recommendation from a specific learner.

        Args:
            learner: Learner type or name
            provider: LLM provider name
            model: Model name
            task_type: Type of task
            vertical: Vertical context
            **context: Additional context for recommendation

        Returns:
            RLRecommendation or None if no recommendation available
        """
        learner_name = learner.value if isinstance(learner, LearnerType) else learner

        # Ensure required parameters are not None
        if provider is None:
            provider = "anthropic"  # Default provider
        if model is None:
            model = "default"  # Default model
        if task_type is None:
            task_type = "default"  # Default task type

        return self._coordinator.get_recommendation(
            learner_name=learner_name,
            provider=provider,
            model=model,
            task_type=task_type,
        )

    def get_tool_recommendation(
        self,
        task_type: str,
        *,
        available_tools: Optional[List[str]] = None,
        vertical: str = "coding",
    ) -> Optional[List[str]]:
        """Get recommended tools for a task type.

        Convenience method for tool selection recommendations.

        Args:
            task_type: Type of task (analysis, implementation, etc.)
            available_tools: List of available tool names
            vertical: Vertical context

        Returns:
            List of recommended tool names or None
        """
        rec = self.get_recommendation(
            LearnerType.TOOL_SELECTOR,
            task_type=task_type,
            vertical=vertical,
            available_tools=available_tools,
        )
        if rec and isinstance(rec.value, list):
            return rec.value
        return None

    def get_patience_recommendation(
        self,
        provider: str,
        model: str,
        *,
        task_type: str = "general",
    ) -> Optional[int]:
        """Get recommended continuation patience.

        Args:
            provider: LLM provider name
            model: Model name
            task_type: Type of task

        Returns:
            Recommended patience value or None
        """
        rec = self.get_recommendation(
            LearnerType.CONTINUATION_PATIENCE,
            provider=provider,
            model=model,
            task_type=task_type,
        )
        if rec and isinstance(rec.value, (int, float)):
            return int(rec.value)
        return None

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> RLStats:
        """Get aggregate RL statistics.

        Returns:
            RLStats with system-wide statistics
        """
        stats = RLStats(
            database_path=(
                str(self._coordinator.db_path) if hasattr(self._coordinator, "db_path") else None
            ),
        )

        # Get learner stats if available
        if hasattr(self._coordinator, "get_all_stats"):
            learner_data = self._coordinator.get_all_stats()
            for name, data in learner_data.items():
                stats.learner_stats[name] = LearnerStats(
                    name=name,
                    total_records=data.get("total_records", 0),
                    success_rate=data.get("success_rate", 0.0),
                    last_updated=data.get("last_updated"),
                    parameters=data.get("parameters", {}),
                )
                stats.total_outcomes += data.get("total_records", 0)
            stats.active_learners = len(stats.learner_stats)

        return stats

    def get_learner_stats(self, learner: Union[LearnerType, str]) -> Optional[LearnerStats]:
        """Get statistics for a specific learner.

        Args:
            learner: Learner type or name

        Returns:
            LearnerStats or None if learner not found
        """
        learner_name = learner.value if isinstance(learner, LearnerType) else learner
        stats = self.get_stats()
        return stats.learner_stats.get(learner_name)

    # =========================================================================
    # Learner Management
    # =========================================================================

    def list_learners(self) -> List[str]:
        """List all registered learners.

        Returns:
            List of learner names
        """
        if hasattr(self._coordinator, "list_learners"):
            return self._coordinator.list_learners()
        return [lt.value for lt in LearnerType]

    def reset_learner(
        self,
        learner: Union[LearnerType, str],
        *,
        vertical: Optional[str] = None,
    ) -> None:
        """Reset a learner's learned parameters.

        Args:
            learner: Learner type or name
            vertical: Optional vertical to reset (None = all)
        """
        learner_name = learner.value if isinstance(learner, LearnerType) else learner
        if hasattr(self._coordinator, "reset_learner"):
            self._coordinator.reset_learner(learner_name, vertical=vertical)

    def __repr__(self) -> str:
        return f"RLManager(learners={len(self.list_learners())})"


# =============================================================================
# Convenience Functions
# =============================================================================


def create_outcome(
    success: bool,
    *,
    provider: str = "unknown",
    model: str = "unknown",
    task_type: str = "general",
    quality_score: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    vertical: str = "coding",
) -> RLOutcome:
    """Create an RLOutcome.

    Convenience function for creating outcomes.

    Args:
        success: Whether the action was successful
        provider: LLM provider name
        model: Model name
        task_type: Type of task
        quality_score: Quality score 0.0-1.0 (default 1.0 for success, 0.0 for failure)
        metadata: Additional metadata
        vertical: Vertical context

    Returns:
        RLOutcome instance
    """
    if quality_score is None:
        quality_score = 1.0 if success else 0.0

    return RLOutcome(
        provider=provider,
        model=model,
        task_type=task_type,
        success=success,
        quality_score=quality_score,
        metadata=metadata or {},
        vertical=vertical,
    )


def record_tool_success(
    tool_name: str,
    task_type: str,
    *,
    provider: str = "unknown",
    model: str = "unknown",
    duration_ms: Optional[float] = None,
    vertical: str = "coding",
) -> None:
    """Record a successful tool execution.

    Convenience function for recording tool successes.

    Args:
        tool_name: Name of the tool
        task_type: Type of task
        provider: LLM provider name
        model: Model name
        duration_ms: Execution duration in milliseconds
        vertical: Vertical context
    """
    coordinator = get_rl_coordinator()
    metadata: Dict[str, Any] = {"tool": tool_name}
    if duration_ms is not None:
        metadata["duration_ms"] = duration_ms

    coordinator.record_outcome(
        learner_name=LearnerType.TOOL_SELECTOR.value,
        outcome=RLOutcome(
            provider=provider,
            model=model,
            task_type=task_type,
            success=True,
            quality_score=1.0,
            metadata=metadata,
            vertical=vertical,
        ),
        vertical=vertical,
    )


__all__ = [
    # Manager
    "RLManager",
    # Types
    "LearnerType",
    "LearnerStats",
    "RLStats",
    # Core infrastructure
    "RLOutcome",
    "RLRecommendation",
    "RLCoordinator",
    "BaseLearner",
    "get_rl_coordinator",
    # Configuration
    "BaseRLConfig",
    "DEFAULT_PATIENCE_MAP",
    "DEFAULT_ACTIVE_LEARNERS",
    # Convenience functions
    "create_outcome",
    "record_tool_success",
]
