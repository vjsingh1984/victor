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

"""System prompt coordinator for agent orchestration.

This module provides the SystemPromptCoordinator which extracts
system-prompt-related business logic from the orchestrator:

- System prompt building with adapter hints
- Dynamic parallel read budget calculation
- Shell variant resolution
- Task keyword classification
- RL prompt-used event emission

Design Pattern: Coordinator (SRP Compliance)
Extracted from AgentOrchestrator to improve modularity and testability
as part of the Phase 6 business logic extraction initiative.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.context_compactor import ParallelReadBudget
    from victor.agent.task_analyzer import TaskAnalyzer
    from victor.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


class SystemPromptCoordinator:
    """Coordinates system prompt building and task classification.

    This coordinator encapsulates business logic previously embedded in
    the AgentOrchestrator:

    - Building system prompts with parallel read budget hints
    - Resolving shell tool aliases to correct variants
    - Classifying tasks by keywords and conversation context
    - Emitting RL prompt-used events

    All dependencies are injected via constructor (DI pattern).

    Args:
        prompt_builder: The system prompt builder instance.
        get_context_window: Callable returning the model's context window size.
        provider_name: Name of the current LLM provider.
        model_name: Name of the current model.
        get_tools: Callable returning the tool registry (live reference).
        get_mode_controller: Callable returning the mode controller (live reference).
        task_analyzer: The task analyzer for keyword classification.
        session_id: Optional session ID for RL event metadata.
    """

    def __init__(
        self,
        prompt_builder: Any,
        get_context_window: Callable[[], int],
        provider_name: str,
        model_name: str,
        get_tools: Callable[[], Optional["ToolRegistry"]],
        get_mode_controller: Callable[[], Optional[object]],
        task_analyzer: "TaskAnalyzer",
        session_id: str = "",
    ) -> None:
        self._prompt_builder = prompt_builder
        self._get_context_window = get_context_window
        self._provider_name = provider_name
        self._model_name = model_name
        self._get_tools = get_tools
        self._get_mode_controller = get_mode_controller
        self._task_analyzer = task_analyzer
        self._session_id = session_id

    def build_system_prompt(self) -> str:
        """Build system prompt with dynamic parallel read budget.

        Includes a parallel read budget hint for models with >= 32K context
        windows to help the LLM plan efficient parallel file reads.

        Returns:
            The final system prompt string.
        """
        from victor.agent.context_compactor import calculate_parallel_read_budget

        base_prompt: str = self._prompt_builder.build()

        # Calculate dynamic parallel read budget based on model context window
        context_window = self._get_context_window()
        budget: ParallelReadBudget = calculate_parallel_read_budget(context_window)

        # Inject dynamic budget hint for models with reasonable context
        # Only add for models with >= 32K context (smaller models benefit
        # from sequential reads)
        if context_window >= 32768:
            budget_hint = budget.to_prompt_hint()
            final_prompt = f"{base_prompt}\n\n{budget_hint}"
        else:
            final_prompt = base_prompt

        # Emit prompt_used event for RL learning
        self._emit_prompt_used_event(final_prompt)

        return final_prompt

    def _emit_prompt_used_event(self, prompt: str) -> None:
        """Emit PROMPT_USED event for RL prompt template learner.

        Args:
            prompt: The final system prompt that was built.
        """
        try:
            from victor.framework.rl.hooks import (
                get_rl_hooks,
                RLEvent,
                RLEventType,
            )

            hooks = get_rl_hooks()
            if hooks is None:
                return

            # Determine prompt style based on provider type
            is_local = self._provider_name.lower() in {
                "ollama",
                "lmstudio",
                "vllm",
            }
            prompt_style = "detailed" if is_local else "structured"

            # Calculate prompt characteristics
            prompt_lower = prompt.lower()
            has_examples = "example" in prompt_lower or "e.g." in prompt_lower
            has_thinking = "step by step" in prompt_lower or "think" in prompt_lower
            has_constraints = "must" in prompt_lower or "always" in prompt_lower

            event = RLEvent(
                type=RLEventType.PROMPT_USED,
                success=True,
                quality_score=0.5,
                provider=self._provider_name,
                model=self._model_name,
                task_type="general",
                metadata={
                    "prompt_style": prompt_style,
                    "prompt_length": len(prompt),
                    "has_examples": has_examples,
                    "has_thinking_prompt": has_thinking,
                    "has_constraints": has_constraints,
                    "session_id": self._session_id,
                },
            )
            hooks.emit(event)
            logger.debug(f"Emitted prompt_used event: style={prompt_style}")

        except Exception as e:
            # RL hook failure should never block prompt building
            logger.debug(f"Failed to emit prompt_used event: {e}")

    def resolve_shell_variant(self, tool_name: str) -> str:
        """Resolve shell aliases to the appropriate enabled shell variant.

        Args:
            tool_name: The tool name to resolve.

        Returns:
            The resolved tool name.
        """
        from victor.agent.shell_resolver import resolve_shell_variant

        return resolve_shell_variant(tool_name, self._get_tools(), self._get_mode_controller())

    def classify_task_keywords(self, user_message: str) -> Dict[str, Any]:
        """Classify task type based on keywords in the user message.

        Args:
            user_message: The user's input message.

        Returns:
            Dictionary with classification results.
        """
        return self._task_analyzer.classify_task_keywords(user_message)

    def classify_task_with_context(
        self,
        user_message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Classify task with conversation context for improved accuracy.

        Args:
            user_message: The user's input message.
            history: Optional conversation history for context boosting.

        Returns:
            Dictionary with classification results.
        """
        return self._task_analyzer.classify_task_with_context(user_message, history)
