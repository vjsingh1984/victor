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

"""System prompt coordinator (DEPRECATED).

.. deprecated::
    Superseded by ``victor.agent.prompt_pipeline.UnifiedPromptPipeline``.
    This module is a thin backward-compat wrapper. New code should use
    UnifiedPromptPipeline directly.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.task_analyzer import TaskAnalyzer
    from victor.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


class SystemPromptCoordinator:
    """Backward-compatible wrapper — delegates to UnifiedPromptPipeline.

    Kept for callers in component_assembler.py, coordinator_factory.py,
    and coordinators/__init__.py.
    """

    def __init__(
        self,
        prompt_builder: Any = None,
        get_context_window: Optional[Callable[[], int]] = None,
        provider_name: str = "",
        model_name: str = "",
        get_tools: Optional[Callable[[], Optional["ToolRegistry"]]] = None,
        get_mode_controller: Optional[Callable[[], Optional[object]]] = None,
        task_analyzer: Optional["TaskAnalyzer"] = None,
        session_id: str = "",
    ):
        self._prompt_builder = prompt_builder
        self._get_context_window = get_context_window or (lambda: 128000)
        self._provider_name = provider_name
        self._model_name = model_name
        self._get_tools = get_tools
        self._get_mode_controller = get_mode_controller
        self._task_analyzer = task_analyzer
        self._session_id = session_id

    def build_system_prompt(self) -> str:
        """Build system prompt with dynamic parallel read budget."""
        if self._prompt_builder is None:
            return ""

        base_prompt: str = self._prompt_builder.build()

        context_window = self._get_context_window()
        if context_window >= 32768:
            try:
                from victor.agent.context_compactor import calculate_parallel_read_budget

                budget = calculate_parallel_read_budget(context_window)
                base_prompt = f"{base_prompt}\n\n{budget.to_prompt_hint()}"
            except Exception:
                pass

        self._emit_prompt_used_event(base_prompt)
        return base_prompt

    def _emit_prompt_used_event(self, prompt: str) -> None:
        """Emit PROMPT_USED event for RL prompt template learner."""
        try:
            from victor.framework.rl.hooks import get_rl_hooks, RLEvent, RLEventType

            hooks = get_rl_hooks()
            if hooks is None:
                return

            is_local = self._provider_name.lower() in {"ollama", "lmstudio", "vllm"}
            prompt_lower = prompt.lower()

            event = RLEvent(
                type=RLEventType.PROMPT_USED,
                success=True,
                quality_score=0.5,
                provider=self._provider_name,
                model=self._model_name,
                task_type="general",
                metadata={
                    "prompt_style": "detailed" if is_local else "structured",
                    "prompt_length": len(prompt),
                    "has_examples": "example" in prompt_lower or "e.g." in prompt_lower,
                    "has_thinking_prompt": "step by step" in prompt_lower,
                    "has_constraints": "must" in prompt_lower or "always" in prompt_lower,
                    "session_id": self._session_id,
                },
            )
            hooks.emit(event)
        except Exception as e:
            logger.debug("Failed to emit prompt_used event: %s", e)

    def resolve_shell_variant(self, tool_name: str) -> str:
        """Resolve shell aliases to the appropriate enabled shell variant."""
        try:
            from victor.agent.shell_resolver import resolve_shell_variant

            return resolve_shell_variant(
                tool_name,
                self._get_tools() if self._get_tools else None,
                self._get_mode_controller() if self._get_mode_controller else None,
            )
        except Exception:
            return tool_name

    def classify_task_keywords(self, user_message: str) -> Dict[str, Any]:
        """Classify task type based on keywords."""
        if self._task_analyzer:
            try:
                # Try the canonical method name first, fall back to alternate
                method = getattr(
                    self._task_analyzer,
                    "classify_task_keywords",
                    getattr(self._task_analyzer, "classify_keywords", None),
                )
                if method:
                    return method(user_message)
            except Exception:
                pass
        return {"task_type": "default", "confidence": 0.0}

    def classify_task_with_context(
        self, user_message: str, history: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """Classify task type with conversation history context."""
        if self._task_analyzer:
            try:
                method = getattr(
                    self._task_analyzer,
                    "classify_task_with_context",
                    getattr(self._task_analyzer, "classify_with_context", None),
                )
                if method:
                    return method(user_message, history)
            except Exception:
                pass
        return {"task_type": "default", "confidence": 0.0}
