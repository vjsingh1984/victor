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

"""Unified Task Analyzer combining multiple classification systems."""

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
from victor.agent.unified_classifier import (
    UnifiedTaskClassifier,
    ClassificationResult,
    TaskType as UnifiedTaskType,
)

if TYPE_CHECKING:
    from victor.embeddings.task_classifier import TaskType
    from victor.embeddings.intent_classifier import IntentType

# Import protocols for type hints (available at runtime since protocols.py has no heavy deps)
from victor.core.protocols import TaskClassifierProtocol, IntentClassifierProtocol

logger = logging.getLogger(__name__)


@dataclass
class TaskAnalysis:
    complexity: TaskComplexity
    tool_budget: int
    complexity_confidence: float
    task_type: Optional["TaskType"] = None
    task_type_confidence: float = 0.0
    has_file_context: bool = False
    unified_task_type: UnifiedTaskType = UnifiedTaskType.DEFAULT
    unified_confidence: float = 0.0
    is_action_task: bool = False
    is_analysis_task: bool = False
    is_generation_task: bool = False
    needs_execution: bool = False
    negated_keywords: List[str] = field(default_factory=list)
    action_intent: ActionIntent = ActionIntent.AMBIGUOUS
    can_write_files: bool = False
    requires_confirmation: bool = False
    continuation_needed: Optional[bool] = None
    intent_type: Optional["IntentType"] = None
    matched_patterns: List[str] = field(default_factory=list)
    analysis_details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_simple(self) -> bool:
        return self.complexity == TaskComplexity.SIMPLE

    @property
    def is_complex(self) -> bool:
        return self.complexity == TaskComplexity.COMPLEX

    @property
    def is_generation(self) -> bool:
        return self.complexity == TaskComplexity.GENERATION

    def should_force_completion(self, tool_calls: int) -> bool:
        return tool_calls >= self.tool_budget


class TaskAnalyzer:

    def __init__(self):
        self._complexity_classifier = None
        self._action_authorizer = None
        self._unified_classifier = None
        self._task_classifier = None
        self._intent_classifier = None

    @property
    def complexity_classifier(self) -> ComplexityClassifier:
        if not self._complexity_classifier:
            self._complexity_classifier = ComplexityClassifier()
        return self._complexity_classifier

    @property
    def action_authorizer(self) -> ActionAuthorizer:
        if not self._action_authorizer:
            self._action_authorizer = ActionAuthorizer()
        return self._action_authorizer

    @property
    def unified_classifier(self) -> UnifiedTaskClassifier:
        if not self._unified_classifier:
            self._unified_classifier = UnifiedTaskClassifier(
                task_analyzer=self, enable_semantic=True
            )
        return self._unified_classifier

    @property
    def task_classifier(self) -> Optional[TaskClassifierProtocol]:
        if self._task_classifier is None:
            try:
                from victor.embeddings.task_classifier import TaskTypeClassifier

                self._task_classifier = TaskTypeClassifier.get_instance()
            except ImportError:
                logger.warning("TaskTypeClassifier not available")
        return self._task_classifier

    @property
    def intent_classifier(self) -> Optional[IntentClassifierProtocol]:
        if self._intent_classifier is None:
            try:
                from victor.embeddings.intent_classifier import IntentClassifier

                self._intent_classifier = IntentClassifier.get_instance()
            except ImportError:
                logger.warning("IntentClassifier not available")
        return self._intent_classifier

    def analyze(
        self,
        message: str,
        include_task_type: bool = True,
        include_intent: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskAnalysis:
        context = context or {}
        history = context.get("history", [])

        complexity_result = self.complexity_classifier.classify(message)
        action_result = self.action_authorizer.detect(message)
        unified_result = (
            self.unified_classifier.classify_with_context(message, history)
            if history
            else self.unified_classifier.classify(message)
        )

        analysis = TaskAnalysis(
            complexity=complexity_result.complexity,
            tool_budget=complexity_result.tool_budget,
            complexity_confidence=complexity_result.confidence,
            unified_task_type=unified_result.task_type,
            unified_confidence=unified_result.confidence,
            is_action_task=unified_result.is_action_task,
            is_analysis_task=unified_result.is_analysis_task,
            is_generation_task=unified_result.is_generation_task,
            needs_execution=unified_result.needs_execution,
            negated_keywords=[m.keyword for m in unified_result.negated_keywords],
            action_intent=action_result.intent,
            can_write_files=action_result.intent == ActionIntent.WRITE_ALLOWED,
            requires_confirmation=action_result.intent == ActionIntent.AMBIGUOUS,
            matched_patterns=complexity_result.matched_patterns,
            analysis_details={
                "complexity_hint": complexity_result.prompt_hint,
                "action_signals": action_result.matched_signals,
                "unified_source": unified_result.source,
                "unified_matched_keywords": [m.keyword for m in unified_result.matched_keywords],
            },
        )

        if include_task_type and self.task_classifier:
            try:
                task_result = self.task_classifier.classify(message)
                analysis.task_type = task_result.task_type
                analysis.task_type_confidence = task_result.confidence
                analysis.has_file_context = task_result.has_file_context
                analysis.analysis_details["task_type_matches"] = task_result.top_matches
            except Exception as e:
                logger.warning(f"Task type classification failed: {e}")

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
        return self.complexity_classifier.classify(message)

    def check_write_authorization(self, message: str) -> bool:
        return self.action_authorizer.detect(message).intent == ActionIntent.WRITE_ALLOWED

    def is_simple_query(self, message: str) -> bool:
        return self.complexity_classifier.classify(message).complexity == TaskComplexity.SIMPLE

    def is_generation_task(self, message: str) -> bool:
        return self.complexity_classifier.classify(message).complexity == TaskComplexity.GENERATION

    def get_tool_budget(self, message: str) -> int:
        return self.complexity_classifier.classify(message).tool_budget

    def classify_unified(
        self, message: str, history: Optional[List[Dict[str, Any]]] = None
    ) -> ClassificationResult:
        return (
            self.unified_classifier.classify_with_context(message, history)
            if history
            else self.unified_classifier.classify(message)
        )

    def is_analysis_task(self, message: str) -> bool:
        return self.unified_classifier.classify(message).is_analysis_task

    def is_action_task(self, message: str) -> bool:
        return self.unified_classifier.classify(message).is_action_task

    def get_negated_keywords(self, message: str) -> List[str]:
        return [m.keyword for m in self.unified_classifier.classify(message).negated_keywords]


# Global instance (legacy - prefer DI container)
_analyzer: Optional[TaskAnalyzer] = None


def get_task_analyzer() -> TaskAnalyzer:
    """Get or create the task analyzer.

    Resolution order:
    1. Check DI container (preferred)
    2. Fall back to module-level singleton (legacy)

    Returns:
        TaskAnalyzer instance
    """
    global _analyzer

    # Try DI container first
    try:
        from victor.core.container import get_container
        from victor.agent.protocols import TaskAnalyzerProtocol

        container = get_container()
        if container.is_registered(TaskAnalyzerProtocol):
            return container.get(TaskAnalyzerProtocol)
    except Exception:
        pass  # Fall back to legacy singleton

    # Legacy fallback
    if not _analyzer:
        _analyzer = TaskAnalyzer()
    return _analyzer


def reset_task_analyzer() -> None:
    """Reset the global task analyzer (for testing).

    Note: This only resets the legacy module-level singleton. If using DI
    container, use reset_container() as well.
    """
    global _analyzer
    _analyzer = None
