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

"""Concrete post-switch hook implementations.

These hooks extract the inline post-switch logic from AgentOrchestrator._apply_post_switch_hooks()
into SRP-compliant, testable, reusable components.

Each hook implements the PostSwitchHook protocol and handles one specific concern:
- ExplorationSettingsHook: Update exploration multiplier and patience
- PromptBuilderHook: Reinitialize prompt builder with new capabilities
- SystemPromptHook: Rebuild system prompt with adapter hints
- ToolBudgetHook: Update tool budget based on new adapter

Usage:
    from victor.agent.post_switch_hooks import (
        ExplorationSettingsHook,
        PromptBuilderHook,
        SystemPromptHook,
        ToolBudgetHook,
    )
    from victor.agent.provider_switch_coordinator import HookPriority

    # Register hooks with coordinator
    coordinator.register_hook(
        ExplorationSettingsHook(unified_tracker, get_caps),
        priority=HookPriority.EARLY,
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from victor.agent.provider_switch_coordinator import SwitchContext

if TYPE_CHECKING:
    from victor.agent.unified_tracker import UnifiedTracker
    from victor.agent.tool_calling.base import ToolCallingCapabilities
    from victor.agent.prompts.system_prompt_builder import SystemPromptBuilder
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


class ExplorationSettingsHook:
    """Update exploration settings after provider/model switch.

    Sets model-specific exploration multiplier and continuation patience
    on the unified tracker.

    Priority: HookPriority.EARLY (10)
    """

    name = "exploration_settings"

    def __init__(
        self,
        unified_tracker: "UnifiedTracker",
        get_capabilities: Callable[[], "ToolCallingCapabilities"],
    ):
        """Initialize the hook.

        Args:
            unified_tracker: Tracker to update
            get_capabilities: Callable that returns current tool calling capabilities
        """
        self._tracker = unified_tracker
        self._get_capabilities = get_capabilities

    def execute(self, context: SwitchContext) -> None:
        """Apply exploration settings from new model's capabilities."""
        try:
            caps = self._get_capabilities()
            self._tracker.set_model_exploration_settings(
                exploration_multiplier=caps.exploration_multiplier,
                continuation_patience=caps.continuation_patience,
            )
            logger.debug(
                "Updated exploration settings: multiplier=%s, patience=%s",
                caps.exploration_multiplier,
                caps.continuation_patience,
            )
        except Exception as e:
            logger.warning("Failed to update exploration settings: %s", e)


class PromptBuilderHook:
    """Reinitialize prompt builder after provider/model switch.

    Creates a new SystemPromptBuilder with the new provider/model capabilities.

    Priority: HookPriority.NORMAL (20)
    """

    name = "prompt_builder"

    def __init__(
        self,
        set_prompt_builder: Callable[["SystemPromptBuilder"], None],
        get_provider_name: Callable[[], str],
        get_model: Callable[[], str],
        get_tool_adapter: Callable[[], Any],
        get_capabilities: Callable[[], "ToolCallingCapabilities"],
        get_prompt_contributors: Callable[[], list],
    ):
        """Initialize the hook.

        Args:
            set_prompt_builder: Setter for the prompt builder
            get_provider_name: Callable returning current provider name
            get_model: Callable returning current model name
            get_tool_adapter: Callable returning current tool adapter
            get_capabilities: Callable returning current capabilities
            get_prompt_contributors: Callable returning prompt contributors from vertical
        """
        self._set_prompt_builder = set_prompt_builder
        self._get_provider_name = get_provider_name
        self._get_model = get_model
        self._get_tool_adapter = get_tool_adapter
        self._get_capabilities = get_capabilities
        self._get_prompt_contributors = get_prompt_contributors

    def execute(self, context: SwitchContext) -> None:
        """Reinitialize prompt builder with new capabilities."""
        try:
            from victor.agent.prompts.system_prompt_builder import SystemPromptBuilder

            prompt_contributors = self._get_prompt_contributors()

            builder = SystemPromptBuilder(
                provider_name=self._get_provider_name(),
                model=self._get_model(),
                tool_adapter=self._get_tool_adapter(),
                capabilities=self._get_capabilities(),
                prompt_contributors=prompt_contributors,
            )
            self._set_prompt_builder(builder)
            logger.debug("Reinitialized prompt builder for %s", context.new_model)
        except Exception as e:
            logger.warning("Failed to reinitialize prompt builder: %s", e)


class SystemPromptHook:
    """Rebuild system prompt after provider/model switch.

    Rebuilds the system prompt using the new adapter hints and project context.

    Priority: HookPriority.LATE (30)
    """

    name = "system_prompt"

    def __init__(
        self,
        build_system_prompt: Callable[[], str],
        set_system_prompt: Callable[[str], None],
        get_project_context: Callable[[], Any],
    ):
        """Initialize the hook.

        Args:
            build_system_prompt: Callable that builds base system prompt
            set_system_prompt: Setter for the system prompt
            get_project_context: Callable returning project context
        """
        self._build_system_prompt = build_system_prompt
        self._set_system_prompt = set_system_prompt
        self._get_project_context = get_project_context

    def execute(self, context: SwitchContext) -> None:
        """Rebuild system prompt with new adapter hints."""
        try:
            base_prompt = self._build_system_prompt()
            project_context = self._get_project_context()

            if project_context and project_context.content:
                full_prompt = base_prompt + "\n\n" + project_context.get_system_prompt_addition()
            else:
                full_prompt = base_prompt

            self._set_system_prompt(full_prompt)
            logger.debug("Rebuilt system prompt for %s", context.new_model)
        except Exception as e:
            logger.warning("Failed to rebuild system prompt: %s", e)


class ToolBudgetHook:
    """Update tool budget after provider/model switch.

    Updates tool budget based on new adapter's recommendation, respecting
    sticky user budget overrides.

    Priority: HookPriority.LAST (40)
    """

    name = "tool_budget"

    def __init__(
        self,
        get_capabilities: Callable[[], "ToolCallingCapabilities"],
        get_settings: Callable[[], "Settings"],
        set_tool_budget: Callable[[int], None],
        get_sticky_budget: Callable[[], bool],
    ):
        """Initialize the hook.

        Args:
            get_capabilities: Callable returning current capabilities
            get_settings: Callable returning settings
            set_tool_budget: Setter for tool budget
            get_sticky_budget: Callable returning whether user budget is sticky
        """
        self._get_capabilities = get_capabilities
        self._get_settings = get_settings
        self._set_tool_budget = set_tool_budget
        self._get_sticky_budget = get_sticky_budget

    def execute(self, context: SwitchContext) -> None:
        """Update tool budget based on new adapter's recommendation."""
        try:
            # Check for sticky user budget override
            if context.respect_sticky_budget and self._get_sticky_budget():
                logger.debug("Skipping tool budget reset (sticky user override)")
                return

            caps = self._get_capabilities()
            settings = self._get_settings()

            default_budget = max(caps.recommended_tool_budget, 50)
            budget = getattr(settings, "tool_call_budget", default_budget)
            self._set_tool_budget(budget)

            logger.debug("Updated tool budget to %d", budget)
        except Exception as e:
            logger.warning("Failed to update tool budget: %s", e)


def create_standard_hooks(
    unified_tracker: "UnifiedTracker",
    get_capabilities: Callable[[], "ToolCallingCapabilities"],
    set_prompt_builder: Callable[["SystemPromptBuilder"], None],
    get_provider_name: Callable[[], str],
    get_model: Callable[[], str],
    get_tool_adapter: Callable[[], Any],
    get_prompt_contributors: Callable[[], list],
    build_system_prompt: Callable[[], str],
    set_system_prompt: Callable[[str], None],
    get_project_context: Callable[[], Any],
    get_settings: Callable[[], "Settings"],
    set_tool_budget: Callable[[int], None],
    get_sticky_budget: Callable[[], bool],
) -> list:
    """Create the standard set of post-switch hooks.

    This factory creates all four standard hooks with proper dependencies.
    Use this to easily set up the orchestrator's post-switch hooks.

    Returns:
        List of (hook, priority) tuples ready for registration
    """
    from victor.agent.provider_switch_coordinator import HookPriority

    return [
        (
            ExplorationSettingsHook(unified_tracker, get_capabilities),
            HookPriority.EARLY,
        ),
        (
            PromptBuilderHook(
                set_prompt_builder=set_prompt_builder,
                get_provider_name=get_provider_name,
                get_model=get_model,
                get_tool_adapter=get_tool_adapter,
                get_capabilities=get_capabilities,
                get_prompt_contributors=get_prompt_contributors,
            ),
            HookPriority.NORMAL,
        ),
        (
            SystemPromptHook(
                build_system_prompt=build_system_prompt,
                set_system_prompt=set_system_prompt,
                get_project_context=get_project_context,
            ),
            HookPriority.LATE,
        ),
        (
            ToolBudgetHook(
                get_capabilities=get_capabilities,
                get_settings=get_settings,
                set_tool_budget=set_tool_budget,
                get_sticky_budget=get_sticky_budget,
            ),
            HookPriority.LAST,
        ),
    ]


__all__ = [
    "ExplorationSettingsHook",
    "PromptBuilderHook",
    "SystemPromptHook",
    "ToolBudgetHook",
    "create_standard_hooks",
]
