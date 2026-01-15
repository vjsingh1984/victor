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
from victor.framework.task import (
    TaskComplexityService as ComplexityClassifier,
    TaskClassification,
    TaskComplexity,
)
from victor.agent.unified_classifier import (
    UnifiedTaskClassifier,
    ClassificationResult,
    ClassifierTaskType as UnifiedTaskType,
)

if TYPE_CHECKING:
    from victor.storage.embeddings.task_classifier import TaskType
    from victor.storage.embeddings.intent_classifier import IntentType
    from victor.agent.mode_workflow_team_coordinator import ModeWorkflowTeamCoordinator
    from victor.protocols.coordination import CoordinationSuggestion

# Import protocols for type hints (available at runtime since protocols.py has no heavy deps)
from victor.core.protocols import TaskClassifierProtocol, IntentClassifierProtocol
from victor.observability.events import create_classification_ensemble_event

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
    # Coordination suggestions (team/workflow recommendations)
    coordination_suggestion: Optional["CoordinationSuggestion"] = None

    @property
    def is_simple(self) -> bool:
        return self.complexity == TaskComplexity.SIMPLE

    @property
    def is_complex(self) -> bool:
        return self.complexity == TaskComplexity.COMPLEX

    @property
    def is_generation(self) -> bool:
        return self.complexity == TaskComplexity.GENERATION

    @property
    def has_team_suggestion(self) -> bool:
        """Check if there are team suggestions."""
        if not self.coordination_suggestion:
            return False
        return len(self.coordination_suggestion.team_recommendations) > 0

    @property
    def should_spawn_team(self) -> bool:
        """Check if a team should be auto-spawned based on suggestions."""
        if not self.coordination_suggestion:
            return False
        return self.coordination_suggestion.should_spawn_team

    @property
    def primary_team(self) -> Optional[str]:
        """Get the primary team recommendation if available."""
        if not self.coordination_suggestion:
            return None
        rec = self.coordination_suggestion.primary_team
        return rec.team_name if rec else None

    def should_force_completion(self, tool_calls: int) -> bool:
        return tool_calls >= self.tool_budget


class TaskAnalyzer:

    def __init__(
        self,
        coordinator: Optional["ModeWorkflowTeamCoordinator"] = None,
    ):
        """Initialize task analyzer.

        Args:
            coordinator: Optional ModeWorkflowTeamCoordinator for team/workflow suggestions
        """
        self._complexity_classifier = None
        self._action_authorizer = None
        self._unified_classifier = None
        self._task_classifier = None
        self._intent_classifier = None
        self._coordinator = coordinator
        self._last_complexity = None  # Track last analyzed complexity for continuation strategy

    def set_coordinator(self, coordinator: "ModeWorkflowTeamCoordinator") -> None:
        """Set the coordinator for team/workflow suggestions.

        Args:
            coordinator: ModeWorkflowTeamCoordinator instance
        """
        self._coordinator = coordinator

    @property
    def complexity_classifier(self) -> ComplexityClassifier:
        if not self._complexity_classifier:
            self._complexity_classifier = ComplexityClassifier()
        return self._complexity_classifier

    def _get_complexity_hint(self, complexity: "TaskComplexity") -> str:
        """Get prompt hint for a complexity level via enricher."""
        from victor.framework.enrichment.strategies import get_complexity_hint

        return get_complexity_hint(complexity)

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
                from victor.storage.embeddings.task_classifier import TaskTypeClassifier

                self._task_classifier = TaskTypeClassifier.get_instance()
            except ImportError:
                logger.warning("TaskTypeClassifier not available")
        return self._task_classifier

    @property
    def intent_classifier(self) -> Optional[IntentClassifierProtocol]:
        if self._intent_classifier is None:
            try:
                from victor.storage.embeddings.intent_classifier import IntentClassifier

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

        # Store last complexity for continuation strategy
        self._last_complexity = complexity_result.complexity

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
                "complexity_hint": self._get_complexity_hint(complexity_result.complexity),
                "action_signals": action_result.matched_signals,
                "unified_source": unified_result.source,
                "unified_matched_keywords": [m.keyword for m in unified_result.matched_keywords],
            },
        )

        if include_task_type and self.task_classifier:
            try:
                task_result = self.task_classifier.classify_sync(message)
                analysis.task_type = task_result.task_type
                analysis.task_type_confidence = task_result.confidence
                analysis.has_file_context = task_result.has_file_context
                analysis.analysis_details["task_type_matches"] = task_result.top_matches
            except Exception as e:
                logger.warning(f"Task type classification failed: {e}")

        if include_intent and self.intent_classifier:
            try:
                intent_result = self.intent_classifier.classify_intent_sync(message)
                analysis.intent_type = intent_result.intent_type
                analysis.continuation_needed = intent_result.needs_continuation
                analysis.analysis_details["intent_confidence"] = intent_result.confidence
            except Exception as e:
                logger.warning(f"Intent classification failed: {e}")

        # Publish ensemble classification event (non-blocking, best-effort)
        try:
            from victor.core.events.backends import get_observability_bus

            bus = get_observability_bus()

            # Build event data from analysis results
            task_type_dict = None
            if analysis.task_type:
                task_type_dict = {
                    "type": analysis.task_type.value if hasattr(analysis.task_type, 'value') else str(analysis.task_type),
                    "confidence": analysis.task_type_confidence,
                }

            intent_type_dict = None
            if analysis.intent_type:
                intent_type_dict = {
                    "type": analysis.intent_type.value if hasattr(analysis.intent_type, 'value') else str(analysis.intent_type),
                    "confidence": analysis.analysis_details.get("intent_confidence", 0.0),
                }

            complexity_dict = {
                "level": analysis.complexity.value if hasattr(analysis.complexity, 'value') else str(analysis.complexity),
                "confidence": analysis.complexity_confidence,
                "tool_budget": analysis.tool_budget,
            }

            event = create_classification_ensemble_event(
                query=message,
                session_id=f"task_analyzer_{id(message)}",  # Use message id as pseudo-session
                task_type=task_type_dict or {},
                intent_type=intent_type_dict,
                complexity=complexity_dict,
                unified_type=analysis.unified_task_type.value if hasattr(analysis.unified_task_type, 'value') else str(analysis.unified_task_type),
                confidence=analysis.unified_confidence,
                tool_budget=analysis.tool_budget,
                requires_confirmation=analysis.requires_confirmation,
                matched_patterns=analysis.matched_patterns,
            )

            import asyncio

            # Fire and forget - don't wait for event emission
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(bus.emit(event.to_messaging_event()))
                else:
                    # Sync context, emit synchronously
                    pass  # Event bus requires async context
            except RuntimeError:
                # No event loop, skip event emission
                pass
        except Exception:
            # Event bus not available or other error - don't break classification
            pass

        return analysis

    def analyze_with_suggestions(
        self,
        message: str,
        mode: str = "build",
        include_task_type: bool = True,
        include_intent: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskAnalysis:
        """Analyze task and get coordination suggestions.

        This extends the base analyze() method by adding coordination suggestions
        from the ModeWorkflowTeamCoordinator based on task type and complexity.

        Args:
            message: User message to analyze
            mode: Current agent mode (explore, plan, build)
            include_task_type: Include semantic task type classification
            include_intent: Include intent classification
            context: Optional context dict with history, etc.

        Returns:
            TaskAnalysis with coordination_suggestion populated
        """
        # Get base analysis
        analysis = self.analyze(
            message=message,
            include_task_type=include_task_type,
            include_intent=include_intent,
            context=context,
        )

        # Add coordination suggestions if coordinator is available
        if self._coordinator:
            try:
                # Map complexity to string
                complexity_str = analysis.complexity.value.lower()

                # Get task type string
                task_type_str = analysis.unified_task_type.value.lower()

                # Get suggestions from coordinator
                suggestion = self._coordinator.suggest_for_task(
                    task_type=task_type_str,
                    complexity=complexity_str,
                    mode=mode,
                )
                analysis.coordination_suggestion = suggestion

                # Log if there are team suggestions for high complexity
                if suggestion.has_team_suggestion:
                    logger.debug(
                        f"Task analysis with suggestions: task={task_type_str}, "
                        f"complexity={complexity_str}, mode={mode}, "
                        f"teams={len(suggestion.team_recommendations)}, "
                        f"action={suggestion.action.value}"
                    )
            except Exception as e:
                logger.warning(f"Failed to get coordination suggestions: {e}")

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

    def detect_intent(self, message: str) -> Any:
        """Detect user intent from message using action authorizer.

        This method uses the ActionAuthorizer to detect whether the user
        intends to read, write, display, or has ambiguous intent.

        Args:
            message: User message to analyze

        Returns:
            IntentClassification with intent (ActionIntent), confidence, and prompt_guard
        """
        return self.action_authorizer.detect(message)

    # =========================================================================
    # Task Classification Methods (moved from AgentOrchestrator)
    # =========================================================================

    def classify_task_keywords(self, user_message: str) -> Dict[str, Any]:
        """Classify task type based on keywords in the user message.

        Uses UnifiedTaskClassifier for robust classification with:
        - Negation detection (handles "don't analyze", "skip the review")
        - Confidence scoring for better decisions
        - Weighted keyword matching

        Args:
            user_message: The user's input message

        Returns:
            Dictionary with:
            - is_action_task: bool - True if task requires action (create/execute/run)
            - is_analysis_task: bool - True if task requires analysis/exploration
            - needs_execution: bool - True if task specifically requires execution
            - coarse_task_type: str - "analysis", "action", or "default"
            - confidence: float - Classification confidence (0.0-1.0)
            - source: str - Classification source ("keyword", "context", "ensemble")
            - task_type: str - Detailed task type
        """
        result = self.unified_classifier.classify(user_message)

        # Log negated keywords for debugging
        if result.negated_keywords:
            negated_strs = [f"{m.keyword}" for m in result.negated_keywords]
            logger.debug(f"Negated keywords detected: {negated_strs}")

        return result.to_legacy_dict()

    def classify_task_with_context(
        self, user_message: str, history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Classify task with conversation context for improved accuracy.

        Uses conversation history to boost classification confidence when
        the current message is ambiguous but context suggests a task type.

        Args:
            user_message: The user's input message
            history: Optional conversation history for context boosting

        Returns:
            Dictionary with classification results (same as classify_task_keywords
            but with potential context boosting applied)
        """
        if history:
            result = self.unified_classifier.classify_with_context(user_message, history)
        else:
            result = self.unified_classifier.classify(user_message)

        if result.context_signals:
            logger.debug(f"Context signals applied: {result.context_signals}")

        return result.to_legacy_dict()

    def extract_required_files_from_prompt(self, user_message: str) -> List[str]:
        """Extract file paths mentioned in user prompt for task completion tracking.

        Looks for patterns like:
        - /absolute/path/to/file.py
        - ./relative/path.py
        - victor/agent/orchestrator.py
        - *.py (wildcards not returned)

        Args:
            user_message: The user's prompt text

        Returns:
            List of file paths mentioned in the prompt
        """
        import re

        required_files: List[str] = []

        # Pattern for file paths (absolute, relative, or module-style)
        # Matches paths with at least one / and a file extension
        # Handles: "path/file.py", path/file.py, path/file.py. (sentence end)
        file_path_pattern = re.compile(
            r"(?:^|\s|[\"'\-])"  # Start of string, whitespace, quote, or dash (for bullet lists)
            r"((?:\.{0,2}/)?"  # Optional ./ or ../ or /
            r"[\w./-]+/"  # At least one directory component
            r"[\w.-]+\.[a-z]{1,10})"  # Filename with extension
            r"(?:\s|[\"']|$|[,;:.\)]|\Z)",  # End: space, quote, EOL, punctuation (incl. period, paren)
            re.IGNORECASE,
        )

        for match in file_path_pattern.finditer(user_message):
            path = match.group(1)
            # Skip wildcards and patterns
            if "*" not in path and "?" not in path:
                required_files.append(path)

        # Also look for explicit "read", "analyze", "audit" + file patterns
        explicit_pattern = re.compile(
            r"(?:read|analyze|audit|check|review|examine)\s+" r"([/\w.-]+(?:/[\w.-]+)+)",
            re.IGNORECASE,
        )
        for match in explicit_pattern.finditer(user_message):
            path = match.group(1)
            if path not in required_files and "*" not in path:
                required_files.append(path)

        logger.debug(f"Extracted required files from prompt: {required_files}")
        return required_files

    def extract_required_outputs_from_prompt(self, user_message: str) -> List[str]:
        """Extract output requirements from user prompt.

        Looks for patterns indicating required output format:
        - "create a findings table"
        - "provide top-3 fixes"
        - "6-10 findings"
        - "must output"

        Args:
            user_message: The user's prompt text

        Returns:
            List of required output types (e.g., ["findings table", "top-3 fixes"])
        """
        import re

        required_outputs: List[str] = []
        message_lower = user_message.lower()

        # Check for findings table requirement
        if re.search(r"findings?\s*table|table\s+of\s+findings?", message_lower):
            required_outputs.append("findings table")

        # Check for top-N fixes requirement
        if re.search(r"top[-\s]?\d+\s+fix(es)?|recommend\s+\d+\s+fix(es)?", message_lower):
            required_outputs.append("top-3 fixes")

        # Check for summary requirement
        if re.search(r"summary\s+of|provide\s+summary|create\s+summary", message_lower):
            required_outputs.append("summary")

        # Check for numbered findings requirement (e.g., "6-10 findings")
        if re.search(r"\d+[-\u2013]\d+\s+findings?", message_lower):
            required_outputs.append("findings table")

        logger.debug(f"Extracted required outputs from prompt: {required_outputs}")
        return required_outputs


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
