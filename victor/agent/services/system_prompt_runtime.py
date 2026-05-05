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

"""Service-owned system prompt compatibility runtime.

..deprecated::
    Superseded by ``victor.agent.prompt_pipeline.UnifiedPromptPipeline``.
    This class is now a thin compatibility wrapper. New code should use
    UnifiedPromptPipeline directly.

Migration Notes (2026-05-04):
- SystemPromptCoordinator now wraps UnifiedPromptPipeline
- PromptRuntimeSupport removed - use UnifiedPromptPipeline instead
- All prompt coordination should go through UnifiedPromptPipeline

For state-passed orchestration boundaries, prefer
``victor.agent.coordinators.SystemPromptStatePassedCoordinator`` or the
matching ``OrchestrationFacade.system_prompt_state_passed`` surface.

The legacy `victor.agent.coordinators.system_prompt_coordinator` module now
re-exports this implementation for compatibility.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SystemPromptCoordinator:
    """Backward-compatible wrapper over UnifiedPromptPipeline.

    .. deprecated::
        Use UnifiedPromptPipeline directly. This class remains only for
        backward compatibility with existing imports.

    Migration path:
        - Old: system_prompt_coordinator.build_system_prompt()
        - New: unified_prompt_pipeline.build_system_prompt(project_context)
    """

    def __init__(
        self,
        prompt_builder: Any = None,
        get_context_window: Optional[Callable[[], int]] = None,
        provider_name: str = "",
        model_name: str = "",
        get_tools: Optional[Callable[[], Optional[Any]]] = None,
        get_mode_controller: Optional[Callable[[], Optional[object]]] = None,
        task_analyzer: Optional[Any] = None,
        session_id: str = "",
        *,
        _emit_deprecation_warning: bool = True,
    ):
        if _emit_deprecation_warning:
            warnings.warn(
                "SystemPromptCoordinator is deprecated and will be removed in v1.0.0. "
                "Use UnifiedPromptPipeline from victor.agent.prompt_pipeline instead. "
                "This compatibility wrapper now delegates to UnifiedPromptPipeline.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Store parameters for compatibility but don't use them
        self._prompt_builder = prompt_builder
        self._get_context_window = get_context_window
        self._provider_name = provider_name
        self._model_name = model_name
        self._get_tools = get_tools
        self._get_mode_controller = get_mode_controller
        self._task_analyzer = task_analyzer
        self._session_id = session_id

        # Create UnifiedPromptPipeline if we have the required dependencies
        self._pipeline = None
        if prompt_builder is not None:
            try:
                from victor.agent.prompt_pipeline import UnifiedPromptPipeline

                # Import optional dependencies
                from victor.agent.content_registry import ContentRegistry
                from victor.agent.optimization_injector import OptimizationInjector

                # Create dummy registry and optimizer if not provided
                registry = ContentRegistry() if True else None
                optimizer = OptimizationInjector() if True else None

                self._pipeline = UnifiedPromptPipeline(
                    provider=None,  # Will detect tier as NO_CACHE
                    builder=prompt_builder,
                    registry=registry,
                    optimizer=optimizer,
                    task_analyzer=task_analyzer,
                    get_context_window=get_context_window or (lambda: 128000),
                    session_id=session_id,
                )
                logger.debug("SystemPromptCoordinator: delegated to UnifiedPromptPipeline")
            except Exception as e:
                logger.debug("SystemPromptCoordinator: failed to create UnifiedPromptPipeline: %s", e)

    def build_system_prompt(self) -> str:
        """Build system prompt using UnifiedPromptPipeline.

        .. deprecated::
            Use unified_prompt_pipeline.build_system_prompt() directly.
        """
        if self._pipeline is not None:
            return self._pipeline.build_system_prompt()

        # Fallback to prompt_builder if pipeline creation failed
        if self._prompt_builder is not None:
            return self._prompt_builder.build()

        return ""

    def resolve_shell_variant(self, tool_name: str) -> str:
        """Resolve shell variant using UnifiedPromptPipeline.

        .. deprecated::
            Use unified_prompt_pipeline.resolve_shell_variant() directly.
        """
        if self._pipeline is not None:
            return self._pipeline.resolve_shell_variant(tool_name)

        return tool_name

    def classify_task_keywords(self, user_message: str) -> Dict[str, Any]:
        """Classify task using UnifiedPromptPipeline.

        .. deprecated::
            Use unified_prompt_pipeline.classify_task_keywords() directly.
        """
        if self._pipeline is not None:
            return self._pipeline.classify_task_keywords(user_message)

        return {"task_type": "default", "confidence": 0.0}

    def classify_task_with_context(
        self, user_message: str, history: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """Classify task with context using UnifiedPromptPipeline.

        .. deprecated::
            Use unified_prompt_pipeline.classify_task_with_context() directly.
        """
        if self._pipeline is not None:
            return self._pipeline.classify_task_with_context(user_message, history)

        return {"task_type": "default", "confidence": 0.0}
