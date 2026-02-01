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

"""Prompting builder for orchestrator initialization.

Part of HIGH-005: Initialization Complexity reduction.
"""

import logging
from typing import Any, Optional, TYPE_CHECKING

from victor.agent.builders.base import FactoryAwareBuilder
from victor.agent.coordinators.response_coordinator import ResponseCoordinator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.orchestrator_factory import OrchestratorFactory


class PromptingBuilder(FactoryAwareBuilder):
    """Build response sanitization, prompt, and mode components."""

    def __init__(self, settings: Any, factory: Optional["OrchestratorFactory"] = None):
        """Initialize the builder.

        Args:
            settings: Application settings
            factory: Optional OrchestratorFactory instance
        """
        super().__init__(settings, factory)

    def build(  # type: ignore[override]
        self,
        orchestrator: "AgentOrchestrator",
        model: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build prompting components and attach them to orchestrator."""
        factory = self._ensure_factory()
        components: dict[str, Any] = {}

        # Response sanitizer for cleaning model output (via factory - DI with fallback)
        orchestrator.sanitizer = factory.create_sanitizer()
        components["sanitizer"] = orchestrator.sanitizer

        # ResponseCoordinator: Coordinates response processing and sanitization (E2 extraction)
        orchestrator._response_coordinator = ResponseCoordinator(
            sanitizer=orchestrator.sanitizer,
            tool_adapter=None,  # Will be set after tool_adapter is initialized
            tool_registry=None,  # Will be set after tools are initialized
        )
        components["response_coordinator"] = orchestrator._response_coordinator

        # System prompt builder with vertical prompt contributors (via factory)
        orchestrator.prompt_builder = factory.create_system_prompt_builder(
            provider_name=orchestrator.provider_name,
            model=model,
            tool_adapter=orchestrator.tool_adapter,
            tool_calling_caps=orchestrator._tool_calling_caps_internal,
        )
        components["prompt_builder"] = orchestrator.prompt_builder

        # Load project context from .victor/init.md (via factory - DI with fallback)
        orchestrator.project_context = factory.create_project_context()
        components["project_context"] = orchestrator.project_context

        # Initialize PromptBuilderCoordinator for prompt building operations (SOLID refactoring)
        from victor.agent.coordinators.prompt_coordinator import PromptBuilderCoordinator

        orchestrator._prompt_coordinator = PromptBuilderCoordinator(
            tool_calling_caps=orchestrator._tool_calling_caps_internal,
            enable_rl_events=True,
        )
        components["prompt_coordinator"] = orchestrator._prompt_coordinator

        # ModeCoordinator: Unified mode management (consolidates scattered mode logic)
        from victor.agent.coordinators.mode_coordinator import ModeCoordinator

        orchestrator._mode_coordinator = ModeCoordinator(
            mode_controller=orchestrator.mode_controller,
            tool_registry=getattr(orchestrator, "tools", None),
        )
        components["mode_coordinator"] = orchestrator._mode_coordinator

        # Build system prompt using adapter hints
        base_system_prompt = orchestrator._build_system_prompt_with_adapter()

        # Inject project context if available
        if orchestrator.project_context.content:
            orchestrator._system_prompt = (
                base_system_prompt
                + "\n\n"
                + orchestrator.project_context.get_system_prompt_addition()
            )
            logger.info(f"Loaded project context from {orchestrator.project_context.context_file}")
        else:
            orchestrator._system_prompt = base_system_prompt

        orchestrator._system_added = False
        components["system_prompt"] = orchestrator._system_prompt

        self._register_components(components)
        return components
