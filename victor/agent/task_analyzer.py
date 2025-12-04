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

"""Unified Task Analyzer combining multiple classification systems.

This module provides a single entry point for task analysis, combining:
- Complexity classification (tool budgeting)
- Task type classification (routing)
- Action authorization (permissions)
- Intent classification (continuation detection)

Design Principles:
- Facade pattern: Provides unified interface to multiple classifiers
- Lazy loading: Classifiers initialized on first use
- Composable: Individual results available or combined analysis
- Cache-friendly: Results can be cached for performance

Example Usage:
    analyzer = TaskAnalyzer()

    # Full analysis
    result = analyzer.analyze("Refactor the authentication module")
    print(f"Complexity: {result.complexity}")
    print(f"Task Type: {result.task_type}")
    print(f"Can Write: {result.can_write_files}")

    # Quick checks
    if analyzer.is_simple_query("What files are in src/"):
        # Skip full analysis for simple queries
        pass
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.agent.action_authorizer import ActionAuthorizer, ActionIntent
from victor.agent.complexity_classifier import (
    ComplexityClassifier,
    TaskClassification,
    TaskComplexity,
)

if TYPE_CHECKING:
    from victor.embeddings.task_classifier import TaskType
    from victor.embeddings.intent_classifier import IntentType

logger = logging.getLogger(__name__)


@dataclass
class TaskAnalysis:
    """Combined result of all task analysis.

    Provides a unified view of task classification from all subsystems.
    """

    # Complexity analysis (for tool budgeting)
    complexity: TaskComplexity
    tool_budget: int
    complexity_confidence: float

    # Task type (for routing)
    task_type: Optional["TaskType"] = None
    task_type_confidence: float = 0.0
    has_file_context: bool = False

    # Action authorization (for permissions)
    action_intent: ActionIntent = ActionIntent.AMBIGUOUS
    can_write_files: bool = False
    requires_confirmation: bool = False

    # Intent classification (for continuation)
    continuation_needed: Optional[bool] = None
    intent_type: Optional["IntentType"] = None

    # Metadata
    matched_patterns: List[str] = field(default_factory=list)
    analysis_details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_simple(self) -> bool:
        """Check if this is a simple task."""
        return self.complexity == TaskComplexity.SIMPLE

    @property
    def is_complex(self) -> bool:
        """Check if this is a complex task."""
        return self.complexity == TaskComplexity.COMPLEX

    @property
    def is_generation(self) -> bool:
        """Check if this is a generation task."""
        return self.complexity == TaskComplexity.GENERATION

    def should_force_completion(self, tool_calls: int) -> bool:
        """Check if completion should be forced after N tool calls.

        Args:
            tool_calls: Current number of tool calls made

        Returns:
            True if completion should be forced
        """
        return tool_calls >= self.tool_budget


class TaskAnalyzer:
    """Unified task analyzer combining multiple classification systems.

    This class provides a single interface for analyzing user messages,
    combining results from:
    - ComplexityClassifier: Determines SIMPLE/MEDIUM/COMPLEX/GENERATION
    - TaskTypeClassifier: Determines EDIT/SEARCH/CREATE/ANALYZE/DESIGN
    - ActionAuthorizer: Determines if file writes are authorized
    - IntentClassifier: Determines if model needs to continue

    Example:
        analyzer = TaskAnalyzer()
        result = analyzer.analyze("Implement user authentication")

        if result.is_complex:
            # Use full tool budget
            pass
        if result.can_write_files:
            # Allow file modifications
            pass
    """

    def __init__(self) -> None:
        """Initialize analyzer with lazy-loaded classifiers."""
        self._complexity_classifier: Optional[ComplexityClassifier] = None
        self._action_authorizer: Optional[ActionAuthorizer] = None
        self._task_classifier: Optional[Any] = None  # Lazy import
        self._intent_classifier: Optional[Any] = None  # Lazy import

    @property
    def complexity_classifier(self) -> ComplexityClassifier:
        """Get or create complexity classifier."""
        if self._complexity_classifier is None:
            self._complexity_classifier = ComplexityClassifier()
        return self._complexity_classifier

    @property
    def action_authorizer(self) -> ActionAuthorizer:
        """Get or create action authorizer."""
        if self._action_authorizer is None:
            self._action_authorizer = ActionAuthorizer()
        return self._action_authorizer

    @property
    def task_classifier(self) -> Any:
        """Get or create task type classifier (lazy import)."""
        if self._task_classifier is None:
            try:
                from victor.embeddings.task_classifier import TaskTypeClassifier

                self._task_classifier = TaskTypeClassifier()
            except ImportError:
                logger.warning("TaskTypeClassifier not available")
                self._task_classifier = None
        return self._task_classifier

    @property
    def intent_classifier(self) -> Any:
        """Get or create intent classifier (lazy import)."""
        if self._intent_classifier is None:
            try:
                from victor.embeddings.intent_classifier import IntentClassifier

                self._intent_classifier = IntentClassifier()
            except ImportError:
                logger.warning("IntentClassifier not available")
                self._intent_classifier = None
        return self._intent_classifier

    def analyze(
        self,
        message: str,
        include_task_type: bool = True,
        include_intent: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskAnalysis:
        """Perform full task analysis on a message.

        Args:
            message: User message to analyze
            include_task_type: Whether to run task type classification
            include_intent: Whether to run intent classification
            context: Optional context (file paths, history, etc.)

        Returns:
            TaskAnalysis with combined results
        """
        context = context or {}

        # 1. Complexity classification (always run - lightweight)
        complexity_result = self.complexity_classifier.classify(message)

        # 2. Action authorization (always run - lightweight)
        action_result = self.action_authorizer.detect(message)

        # Build base analysis
        analysis = TaskAnalysis(
            complexity=complexity_result.complexity,
            tool_budget=complexity_result.tool_budget,
            complexity_confidence=complexity_result.confidence,
            action_intent=action_result.intent,
            can_write_files=action_result.intent == ActionIntent.WRITE_ALLOWED,
            requires_confirmation=action_result.intent == ActionIntent.AMBIGUOUS,
            matched_patterns=complexity_result.matched_patterns,
            analysis_details={
                "complexity_hint": complexity_result.prompt_hint,
                "action_signals": action_result.matched_signals,
            },
        )

        # 3. Task type classification (optional - uses embeddings)
        if include_task_type and self.task_classifier:
            try:
                task_result = self.task_classifier.classify(message)
                analysis.task_type = task_result.task_type
                analysis.task_type_confidence = task_result.confidence
                analysis.has_file_context = task_result.has_file_context
                analysis.analysis_details["task_type_matches"] = task_result.top_matches
            except Exception as e:
                logger.warning(f"Task type classification failed: {e}")

        # 4. Intent classification (optional - uses embeddings)
        if include_intent and self.intent_classifier:
            try:
                intent_result = self.intent_classifier.classify(message)
                analysis.intent_type = intent_result.intent_type
                analysis.continuation_needed = intent_result.needs_continuation
                analysis.analysis_details["intent_confidence"] = intent_result.confidence
            except Exception as e:
                logger.warning(f"Intent classification failed: {e}")

        return analysis

    def classify_complexity(self, message: str) -> TaskClassification:
        """Quick complexity classification only.

        Args:
            message: User message

        Returns:
            TaskClassification result
        """
        return self.complexity_classifier.classify(message)

    def check_write_authorization(self, message: str) -> bool:
        """Quick check if message authorizes file writes.

        Args:
            message: User message

        Returns:
            True if file writes are authorized
        """
        result = self.action_authorizer.detect(message)
        return result.intent == ActionIntent.WRITE_ALLOWED

    def is_simple_query(self, message: str) -> bool:
        """Quick check if message is a simple query.

        Args:
            message: User message

        Returns:
            True if this is a simple query
        """
        result = self.complexity_classifier.classify(message)
        return result.complexity == TaskComplexity.SIMPLE

    def is_generation_task(self, message: str) -> bool:
        """Quick check if message is a generation task.

        Args:
            message: User message

        Returns:
            True if this is a generation task
        """
        result = self.complexity_classifier.classify(message)
        return result.complexity == TaskComplexity.GENERATION

    def get_tool_budget(self, message: str) -> int:
        """Get recommended tool budget for a message.

        Args:
            message: User message

        Returns:
            Recommended tool call budget
        """
        result = self.complexity_classifier.classify(message)
        return result.tool_budget


# =============================================================================
# Global Instance
# =============================================================================

_analyzer: Optional[TaskAnalyzer] = None


def get_task_analyzer() -> TaskAnalyzer:
    """Get or create the global task analyzer.

    Returns:
        Global TaskAnalyzer instance
    """
    global _analyzer
    if _analyzer is None:
        _analyzer = TaskAnalyzer()
    return _analyzer


def reset_task_analyzer() -> None:
    """Reset the global task analyzer (for testing)."""
    global _analyzer
    _analyzer = None
