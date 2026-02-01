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

"""Mode controller adapter for DI integration.

Phase 1.1: Integrate AgentModeController with protocol.

This adapter wraps AgentModeController to provide DI-compatible interface
and implements ExtendedModeControllerProtocol for full functionality.

Design Pattern: Adapter
- Wraps existing AgentModeController
- Implements protocol for DI registration
- Provides consistent interface for orchestrator
- Enables testing via mock injection
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional
from collections.abc import Callable

if TYPE_CHECKING:
    from victor.agent.mode_controller import AgentMode, AgentModeController, OperationalModeConfig

logger = logging.getLogger(__name__)


class ModeControllerAdapter:
    """Adapter wrapping AgentModeController for DI integration.

    This adapter implements ExtendedModeControllerProtocol by delegating
    to an underlying AgentModeController instance. It provides:

    - Full mode management delegation
    - Protocol-compatible interface
    - DI container registration support
    - Consistent API for orchestrator usage

    Usage:
        # Via DI container (preferred)
        controller = container.get(ModeControllerProtocol)

        # Direct instantiation (for testing)
        adapter = ModeControllerAdapter(AgentModeController())

    Attributes:
        _controller: Underlying AgentModeController instance
    """

    def __init__(self, controller: "AgentModeController") -> None:
        """Initialize the adapter with an AgentModeController.

        Args:
            controller: The AgentModeController instance to wrap
        """
        self._controller = controller
        logger.debug(
            f"ModeControllerAdapter initialized with mode: {controller.current_mode.value}"
        )

    # =========================================================================
    # ModeControllerProtocol Implementation
    # =========================================================================

    @property
    def current_mode(self) -> "AgentMode":
        """Get the current agent mode.

        Returns:
            Current AgentMode (BUILD, PLAN, or EXPLORE)
        """
        return self._controller.current_mode

    @property
    def config(self) -> "OperationalModeConfig":
        """Get the current mode configuration.

        Returns:
            OperationalModeConfig for the current mode
        """
        return self._controller.config

    def switch_mode(self, new_mode: "AgentMode") -> bool:
        """Switch to a new agent mode.

        Args:
            new_mode: The mode to switch to

        Returns:
            True if switch was successful
        """
        return self._controller.switch_mode(new_mode)

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed in the current mode.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if the tool is allowed
        """
        return self._controller.is_tool_allowed(tool_name)

    def get_exploration_multiplier(self) -> float:
        """Get the exploration limit multiplier for current mode.

        Returns:
            Multiplier value (BUILD=5.0, PLAN=10.0, EXPLORE=20.0)
        """
        return self._controller.config.exploration_multiplier

    def get_tool_priority(self, tool_name: str) -> float:
        """Get priority adjustment for a tool in current mode.

        Args:
            tool_name: Name of the tool

        Returns:
            Priority multiplier (1.0 = no adjustment)
        """
        return self._controller.get_tool_priority(tool_name)

    # =========================================================================
    # ExtendedModeControllerProtocol Implementation
    # =========================================================================

    def previous_mode(self) -> Optional["AgentMode"]:
        """Switch to the previous mode in history.

        Returns:
            The previous mode, or None if no history
        """
        return self._controller.previous_mode()

    def register_callback(self, callback: Callable[["AgentMode", "AgentMode"], None]) -> None:
        """Register a callback for mode changes.

        Args:
            callback: Function called with (old_mode, new_mode) on transitions
        """
        self._controller.register_callback(callback)

    def get_system_prompt_addition(self) -> str:
        """Get additional system prompt text for current mode.

        Returns:
            Mode-specific system prompt addition
        """
        return self._controller.get_system_prompt_addition()

    def get_status(self) -> dict[str, Any]:
        """Get current mode status information.

        Returns:
            Dictionary with mode information
        """
        return self._controller.get_status()

    def get_mode_list(self) -> list[dict[str, str]]:
        """Get list of available modes.

        Returns:
            List of mode info dictionaries
        """
        return self._controller.get_mode_list()

    @property
    def sandbox_dir(self) -> Optional[str]:
        """Get sandbox directory for restricted modes.

        Returns:
            Sandbox directory path or None
        """
        return self._controller.config.sandbox_dir

    @property
    def allow_sandbox_edits(self) -> bool:
        """Check if sandbox edits are allowed.

        Returns:
            True if edits allowed in sandbox
        """
        return self._controller.config.allow_sandbox_edits

    @property
    def require_write_confirmation(self) -> bool:
        """Check if write confirmation is required.

        Returns:
            True if writes require confirmation
        """
        return self._controller.config.require_write_confirmation

    @property
    def max_files_per_operation(self) -> int:
        """Get maximum files allowed per operation.

        Returns:
            Max files limit (0 = unlimited)
        """
        return self._controller.config.max_files_per_operation


def create_mode_controller_adapter() -> ModeControllerAdapter:
    """Factory function for creating ModeControllerAdapter.

    This function is used for DI container registration. It creates
    a new AgentModeController and wraps it in an adapter.

    Returns:
        New ModeControllerAdapter instance
    """
    from victor.agent.mode_controller import AgentModeController, AgentMode

    controller = AgentModeController(initial_mode=AgentMode.BUILD)
    return ModeControllerAdapter(controller)
