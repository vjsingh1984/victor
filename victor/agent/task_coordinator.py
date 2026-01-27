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

"""Task coordination component for task preparation and guidance.

This module provides a centralized interface for task coordination operations:
- Task preparation with complexity detection
- Intent-based prompt guards
- Task-specific guidance and budget adjustments

Extracted from CRITICAL-001 Phase 2D: Extract TaskCoordinator
"""

import logging
from typing import Any, Optional, TYPE_CHECKING

from victor.agent.unified_task_tracker import TrackerTaskType
from victor.core.events import ObservabilityBus
from victor.core.events.emit_helper import emit_event_sync

if TYPE_CHECKING:
    from victor.agent.task_analyzer import TaskAnalyzer
    from victor.agent.unified_task_tracker import UnifiedTaskTracker
    from victor.agent.conversation_controller import ConversationController
    from victor.agent.prompt_builder import SystemPromptBuilder
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


class TaskCoordinator:
    """Coordinates task preparation, intent detection, and guidance.

    This component provides a semantic interface for task coordination,
    consolidating methods that were previously in AgentOrchestrator.

    Architecture:
    - TaskCoordinator: High-level coordinator for task operations
    - TaskAnalyzer: Complexity classification and intent detection
    - UnifiedTaskTracker: Tool budget management
    - ConversationController: Message injection
    - PromptBuilder: Task hint retrieval

    Responsibilities:
    - Prepare tasks with complexity detection
    - Apply intent-based prompt guards
    - Provide task-specific guidance
    - Adjust tool budgets based on complexity

    Design Pattern:
    - Coordinator/Facade: Simplifies task coordination interface
    - Delegation: Delegates to TaskAnalyzer, UnifiedTaskTracker, etc.

    Extracted from CRITICAL-001 Phase 2D.
    """

    def __init__(
        self,
        task_analyzer: "TaskAnalyzer",
        unified_tracker: "UnifiedTaskTracker",
        prompt_builder: "SystemPromptBuilder",
        settings: "Settings",
        event_bus: Optional[ObservabilityBus] = None,
    ):
        """Initialize TaskCoordinator.

        Args:
            task_analyzer: Task complexity classifier and intent detector
            unified_tracker: Tool budget and iteration tracker
            prompt_builder: System prompt builder for task hints
            settings: Application settings
            event_bus: Optional ObservabilityBus instance. If None, uses DI container.
        """
        self.task_analyzer = task_analyzer
        self._event_bus = event_bus or self._get_default_bus()
        self.unified_tracker = unified_tracker
        self.prompt_builder = prompt_builder
        self.settings = settings
        self._current_intent = None
        self._temperature = getattr(settings, "temperature", 0.7)
        self._tool_budget = getattr(settings, "tool_budget", 15)
        self._observed_files: list[str] = []
        self._reminder_manager = None

    def _get_default_bus(self) -> Optional[ObservabilityBus]:
        """Get default ObservabilityBus from DI container.

        Returns:
            ObservabilityBus instance or None if unavailable
        """
        try:
            from victor.core.events import get_observability_bus

            return get_observability_bus()
        except Exception:
            return None

    # =====================================================================
    # Task Preparation
    # =====================================================================

    def prepare_task(
        self,
        user_message: str,
        unified_task_type: "TrackerTaskType",
        conversation_controller: "ConversationController",
    ) -> tuple[Any, int]:
        """Prepare task-specific guidance and budget adjustments.

        Uses granular task classification for hint lookup, then falls back
        to unified task type if needed. Adjusts tool budget based on
        task complexity.

        Args:
            user_message: The user's input message
            unified_task_type: Unified task type classification
            conversation_controller: Conversation controller for message injection

        Returns:
            Tuple of (task_classification, complexity_tool_budget)
        """
        from victor.agent.prompt_builder import get_task_type_hint
        from victor.storage.embeddings.task_classifier import TaskTypeClassifier
        from victor.framework.task import TaskComplexity, DEFAULT_BUDGETS

        # Get granular task type for more specific hints
        granular_task_type = None
        try:
            classifier = TaskTypeClassifier.get_instance()
            if classifier._initialized:
                result = classifier.classify_sync(user_message)
                granular_task_type = result.task_type.value
        except Exception:
            pass  # Fall back to unified type

        # Try granular type first (e.g., "refactor", "bug_fix", "documentation")
        task_hint = None
        hint_source = None
        if granular_task_type:
            task_hint = get_task_type_hint(
                granular_task_type,
                prompt_contributors=self.prompt_builder.prompt_contributors,
            )
            if task_hint:
                hint_source = granular_task_type

        # Fall back to unified type if no granular hint found
        if not task_hint:
            task_hint = get_task_type_hint(
                unified_task_type.value,
                prompt_contributors=self.prompt_builder.prompt_contributors,
            )
            if task_hint:
                hint_source = unified_task_type.value

        if task_hint:
            conversation_controller.add_message("system", task_hint.strip())
            logger.debug(f"Injected task hint for task type: {hint_source}")

        # Classify task complexity and adjust tool budget
        task_classification = self.task_analyzer.classify_complexity(user_message)
        complexity_tool_budget = DEFAULT_BUDGETS.get(task_classification.complexity, 15)

        if task_classification.complexity == TaskComplexity.SIMPLE:
            # Override with simpler budget for simple tasks
            current_max = self.unified_tracker.config.get("max_total_iterations", 50)
            new_max = min(complexity_tool_budget, current_max)
            self.unified_tracker.set_tool_budget(new_max)
            logger.info(
                f"Task complexity: {task_classification.complexity.value}, "
                f"adjusted max_iterations to {complexity_tool_budget}"
            )
        elif task_classification.complexity == TaskComplexity.GENERATION:
            # Generation tasks should complete in 1-2 tool calls
            current_max = self.unified_tracker.config.get("max_total_iterations", 50)
            new_max = min(complexity_tool_budget + 1, current_max)
            self.unified_tracker.set_tool_budget(new_max)
            logger.info(
                f"Generation task detected, limiting iterations to {complexity_tool_budget + 1}"
            )
        else:
            logger.info(
                f"Task complexity: {task_classification.complexity.value}, "
                f"confidence: {task_classification.confidence:.2f}"
            )

        # Update reminder manager with task complexity and hint
        if self._reminder_manager:
            from victor.framework.enrichment.strategies import get_complexity_hint

            self._reminder_manager.update_state(
                task_complexity=task_classification.complexity.value,
                task_hint=get_complexity_hint(task_classification.complexity),
                tool_budget=complexity_tool_budget,
            )

        # Emit STATE event for task preparation
        if self._event_bus:
            emit_event_sync(
                self._event_bus,
                topic="task.prepared",
                data={
                    "unified_task_type": unified_task_type.value,
                    "complexity": task_classification.complexity.value,
                    "confidence": task_classification.confidence,
                    "tool_budget": complexity_tool_budget,
                    "category": "state",  # Preserve for observability
                },
                source="TaskCoordinator",
            )

        return task_classification, complexity_tool_budget

    # =====================================================================
    # Intent Detection
    # =====================================================================

    def apply_intent_guard(
        self, user_message: str, conversation_controller: "ConversationController"
    ) -> None:
        """Detect intent and inject prompt guards for read-only tasks.

        Analyzes the user's message to determine their intent (DISPLAY_ONLY,
        READ_ONLY, WRITE_ALLOWED, etc.) and injects appropriate prompt guards
        to prevent unwanted write operations.

        Args:
            user_message: The user's input message
            conversation_controller: Conversation controller for message injection
        """
        from victor.agent.action_authorizer import ActionIntent

        intent_result = self.task_analyzer.detect_intent(user_message)
        self._current_intent = intent_result.intent

        if intent_result.intent in (ActionIntent.DISPLAY_ONLY, ActionIntent.READ_ONLY):
            if intent_result.prompt_guard:
                conversation_controller.add_message("system", intent_result.prompt_guard.strip())
                logger.info(f"Intent: {intent_result.intent.value}, injected prompt guard")
        elif intent_result.intent == ActionIntent.WRITE_ALLOWED:
            logger.info("Intent: write_allowed, no prompt guard needed")

    # =====================================================================
    # Task Guidance
    # =====================================================================

    def apply_task_guidance(
        self,
        user_message: str,
        unified_task_type: "TrackerTaskType",
        is_analysis_task: bool,
        is_action_task: bool,
        needs_execution: bool,
        max_exploration_iterations: int,
        conversation_controller: "ConversationController",
    ) -> None:
        """Apply guidance and budget tweaks for analysis/action tasks.

        Provides task-specific system prompts and adjusts agent parameters
        (temperature, tool budget) based on task characteristics.

        Args:
            user_message: The user's input message
            unified_task_type: Unified task type classification
            is_analysis_task: Whether this is an analysis task
            is_action_task: Whether this is an action-oriented task
            needs_execution: Whether the task requires execution
            max_exploration_iterations: Maximum exploration iterations allowed
            conversation_controller: Conversation controller for message injection
        """
        if is_analysis_task:
            # Increase temperature for more creative analysis
            analysis_temp = min(self._temperature + 0.2, 1.0)
            logger.info(
                f"Analysis task: increasing temperature {self._temperature:.1f} -> {analysis_temp:.1f}"
            )
            self._temperature = analysis_temp

            conversation_controller.add_message(
                "system",
                "ANALYSIS APPROACH: Work through the codebase one module at a time. "
                "For each module: 1) List its files, 2) Read 2-3 key files, 3) Note observations. "
                "After examining 3-4 modules, provide your summary. Keep responses concise.",
            )

            # Increase tool budget for comprehensive analysis
            original_budget = self._tool_budget
            self._tool_budget = max(self._tool_budget, 200)
            if self._tool_budget != original_budget:
                logger.info(
                    f"Analysis task: increased tool_budget from {original_budget} to {self._tool_budget}"
                )

            conversation_controller.add_message(
                "system",
                "This is an ANALYSIS task requiring thorough exploration of the codebase. "
                "You MUST systematically examine multiple modules and files using tools like "
                "read_file, list_directory, and code_search. "
                "DO NOT stop after examining just a few files. "
                "Continue using tools until you have gathered comprehensive information about "
                "all major components of the codebase. "
                "Only provide your final analysis AFTER you have examined all relevant modules.",
            )

        if is_action_task:
            logger.info(
                f"Detected action-oriented task - allowing up to {max_exploration_iterations} exploration iterations"
            )

            if needs_execution:
                conversation_controller.add_message(
                    "system",
                    "This is an action-oriented task requiring execution. "
                    "Follow this workflow: "
                    "1. CREATE the file/script with write_file or edit_files "
                    "2. EXECUTE it immediately with execute_bash (don't skip this step!) "
                    "3. SHOW the output to the user. "
                    "Minimize exploration and proceed directly to create→execute→show results.",
                )
            else:
                conversation_controller.add_message(
                    "system",
                    "This is an action-oriented task (create/write/build). "
                    "Minimize exploration and proceed directly to creating what was requested. "
                    "Only explore if absolutely necessary to complete the task.",
                )

    # =====================================================================
    # Property Accessors
    # =====================================================================

    @property
    def current_intent(self) -> Any:
        """Get the current detected intent."""
        return self._current_intent

    @property
    def temperature(self) -> float:
        """Get the current temperature setting."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set the temperature setting."""
        self._temperature = value

    @property
    def tool_budget(self) -> int:
        """Get the current tool budget."""
        return self._tool_budget

    @tool_budget.setter
    def tool_budget(self, value: int) -> None:
        """Set the tool budget."""
        self._tool_budget = value

    @property
    def observed_files(self) -> list[Any]:
        """Get the list of observed files."""
        return self._observed_files

    @observed_files.setter
    def observed_files(self, value: list[Any]) -> None:
        """Set the list of observed files."""
        self._observed_files = value

    def set_reminder_manager(self, reminder_manager: Any) -> None:
        """Set the reminder manager reference.

        Args:
            reminder_manager: ReminderManager instance
        """
        self._reminder_manager = reminder_manager
