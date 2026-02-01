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

"""Mode coordinator for unified mode management.

This coordinator consolidates scattered mode logic from the orchestrator
into a single, focused coordinator following SOLID principles.

Design Patterns:
    - Facade Pattern: Simplifies access to mode controller functionality
    - Delegation Pattern: Delegates to AgentModeController for core logic
    - SRP: Focused only on mode management and queries

Usage:
    from victor.agent.coordinators.mode_coordinator import ModeCoordinator
    from victor.agent.mode_controller import AgentModeController, AgentMode

    mode_controller = AgentModeController(initial_mode=AgentMode.BUILD)
    coordinator = ModeCoordinator(mode_controller=mode_controller)

    # Check tool access
    if coordinator.is_tool_allowed("bash"):
        # Tool is allowed in current mode

    # Get tool priority
    priority = coordinator.get_tool_priority("edit_files")

    # Switch modes
    coordinator.switch_mode(AgentMode.PLAN)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from victor.agent.mode_controller import AgentMode, AgentModeController, OperationalModeConfig

logger = logging.getLogger(__name__)


class ModeCoordinator:
    """Coordinator for unified mode management.

    This coordinator provides a unified API for mode-related queries
    that were previously scattered across the orchestrator. It delegates
    to AgentModeController for core logic while providing a cleaner
    abstraction layer.

    Responsibilities:
    - Provide unified API for mode queries
    - Check tool access based on current mode
    - Get tool priority adjustments
    - Resolve shell variants based on mode
    - Manage mode switches

    The coordinator follows the Facade pattern, simplifying access to
    mode controller functionality while delegating core logic.
    """

    def __init__(
        self,
        mode_controller: Optional[AgentModeController] = None,
        tool_registry: Optional[Any] = None,
    ) -> None:
        """Initialize the mode coordinator.

        Args:
            mode_controller: AgentModeController instance (creates default if None)
            tool_registry: Optional tool registry for shell variant resolution
        """
        self._mode_controller = mode_controller or AgentModeController()
        self._tool_registry = tool_registry

    # ========================================================================
    # Mode Queries
    # ========================================================================

    def get_current_mode(self) -> AgentMode:
        """Get the current agent mode.

        Returns:
            Current AgentMode
        """
        return self._mode_controller.current_mode

    def get_mode_config(self) -> OperationalModeConfig:
        """Get the current mode configuration.

        Returns:
            OperationalModeConfig for the current mode
        """
        return self._mode_controller.config

    def get_mode_status(self) -> dict[str, Any]:
        """Get comprehensive status of current mode.

        Returns:
            Dictionary with mode status including:
            - mode: Current mode string
            - name: Mode display name
            - description: Mode description
            - write_confirmation_required: Whether writes require confirmation
            - verbose_planning: Whether planning is verbose
        """
        return self._mode_controller.get_status()

    def get_available_modes(self) -> list[dict[str, str]]:
        """Get list of available modes.

        Returns:
            List of mode info dictionaries with:
            - mode: Mode string (e.g., "build", "plan", "explore")
            - name: Display name
            - description: Mode description
            - current: Whether this is the current mode
        """
        return self._mode_controller.get_mode_list()

    # ========================================================================
    # Tool Access Control
    # ========================================================================

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed in the current mode.

        This consolidates scattered tool access checks from the orchestrator:
        - Lines 3257-3263: Shell variant resolution
        - Lines 5847-5859: Tool access restrictions

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if the tool is allowed in current mode
        """
        return self._mode_controller.is_tool_allowed(tool_name)

    def get_tool_priority(self, tool_name: str) -> float:
        """Get priority adjustment for a tool in current mode.

        Higher priority tools should be selected more frequently.
        Returns 1.0 for tools without specific priority adjustments.

        Args:
            tool_name: Name of the tool

        Returns:
            Priority multiplier (1.0 = no adjustment)
        """
        return self._mode_controller.get_tool_priority(tool_name)

    def get_exploration_multiplier(self) -> float:
        """Get exploration multiplier for current mode.

        The exploration multiplier controls how much exploration
        is allowed before requiring action. Higher values allow
        more exploration.

        Returns:
            Exploration multiplier (e.g., 5.0 for BUILD, 10.0 for PLAN)
        """
        return self._mode_controller.config.exploration_multiplier

    # ========================================================================
    # Shell Variant Resolution
    # ========================================================================

    def resolve_shell_variant(self, tool_name: str) -> str:
        """Resolve shell aliases to the appropriate enabled shell variant.

        This consolidates shell variant resolution logic from the orchestrator
        (lines 3257-3278) into a single method.

        LLMs often hallucinate shell tool names like 'run', 'bash', 'execute'.
        These map to 'shell' canonically, but in non-BUILD modes we may need
        to use 'shell_readonly' instead.

        Resolution logic:
        1. Check if BUILD mode allows all tools → use full 'shell'
        2. Check if full 'shell' is enabled → use it
        3. Fall back to 'shell_readonly' if enabled
        4. Otherwise return canonical name (will fail validation)

        Args:
            tool_name: The hallucinated or alias tool name (e.g., "bash", "run")

        Returns:
            Resolved shell variant: "shell", "shell_readonly", or canonical name
        """
        from victor.tools.tool_names import ToolNames

        # Check mode controller for BUILD mode (allows all tools including shell)
        mc = self._mode_controller
        if mc is not None:
            config = mc.config
            # If mode allows all tools and shell isn't explicitly disallowed, use full shell
            if config.allow_all_tools and "shell" not in config.disallowed_tools:
                logger.debug(f"Resolved '{tool_name}' to 'shell' (mode allows all tools)")
                return ToolNames.SHELL

        # Check if full shell is enabled first
        if self._tool_registry and self._tool_registry.is_tool_enabled(ToolNames.SHELL):
            logger.debug(f"Resolved '{tool_name}' to 'shell' (shell enabled)")
            return ToolNames.SHELL

        # Fall back to shell_readonly if enabled
        if self._tool_registry and self._tool_registry.is_tool_enabled(ToolNames.SHELL_READONLY):
            logger.debug(f"Resolved '{tool_name}' to 'shell_readonly' (readonly mode)")
            return ToolNames.SHELL_READONLY

        # Neither enabled - return canonical name (will fail validation)
        from victor.tools.tool_names import get_canonical_name

        canonical = get_canonical_name(tool_name)
        logger.debug(f"No shell variant enabled for '{tool_name}', using canonical '{canonical}'")
        return canonical

    # ========================================================================
    # Mode Switching
    # ========================================================================

    def switch_mode(self, new_mode: AgentMode) -> bool:
        """Switch to a new mode.

        Args:
            new_mode: The mode to switch to

        Returns:
            True if switch was successful
        """
        return self._mode_controller.switch_mode(new_mode)

    def previous_mode(self) -> Optional[AgentMode]:
        """Switch to the previous mode.

        Returns:
            The mode switched to, or None if no history
        """
        return self._mode_controller.previous_mode()

    # ========================================================================
    # System Prompt
    # ========================================================================

    def get_system_prompt_addition(self) -> str:
        """Get additional system prompt text for current mode.

        Returns:
            Additional prompt text to append to system prompt
        """
        return self._mode_controller.get_system_prompt_addition()

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def mode_controller(self) -> AgentModeController:
        """Get the underlying mode controller."""
        return self._mode_controller

    @property
    def current_mode_name(self) -> str:
        """Get the current mode name as a string."""
        return self._mode_controller.current_mode.value


__all__ = [
    "ModeCoordinator",
]
