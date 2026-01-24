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

"""Tool Access Coordinator - Manages tool access control.

This module extracts tool access control logic from ToolCoordinator,
following SRP (Single Responsibility Principle).

Responsibilities:
- Tool enable/disable management
- Mode-based access control
- Session-based tool filtering
- Tool availability checking

Design Philosophy:
- Single Responsibility: Only handles access control decisions
- Layered: Supports mode, session, and registry layers
- Protocol-based: Implements access control protocol
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Set

if TYPE_CHECKING:
    from victor.tools.registry import ToolRegistry

from victor.agent.coordinators.base_config import BaseCoordinatorConfig

logger = logging.getLogger(__name__)


@dataclass
class ToolAccessConfig(BaseCoordinatorConfig):
    """Configuration for ToolAccessCoordinator.

    Inherits common configuration from BaseCoordinatorConfig:
        enabled: Whether the coordinator is enabled
        timeout: Default timeout in seconds for operations
        max_retries: Maximum number of retry attempts for failed operations
        retry_enabled: Whether retry logic is enabled
        log_level: Logging level for coordinator messages
        enable_metrics: Whether to collect metrics

    Attributes:
        default_allow_all: Default behavior when no mode controller
        strict_mode: Reject unknown tools when True
    """

    default_allow_all: bool = True
    strict_mode: bool = False


@dataclass
class AccessDecision:
    """Decision for tool access.

    Attributes:
        tool_name: Name of tool being checked
        allowed: Whether access is allowed
        reason: Reason for denial (if denied)
        layer: Which layer made the decision (mode, session, registry)
    """

    tool_name: str
    allowed: bool
    reason: Optional[str] = None
    layer: str = "unknown"


@dataclass
class ToolAccessContext:
    """Context for access control decisions.

    Attributes:
        session_enabled_tools: Tools explicitly enabled for session
        current_mode: Current agent mode (build, plan, explore)
        disallowed_tools: Tools explicitly disallowed by mode
    """

    session_enabled_tools: Optional[Set[str]] = None
    current_mode: Optional[str] = None
    disallowed_tools: Set[str] = field(default_factory=set)


class ToolAccessCoordinator:
    """Coordinator for tool access control.

    Extracts access control logic from ToolCoordinator following SRP.
    This coordinator is responsible only for determining whether
    a tool can be used based on mode, session, and registry state.

    Example:
        coordinator = ToolAccessCoordinator(
            tool_registry=registry,
            config=ToolAccessConfig(default_allow_all=True)
        )

        # Set session tools
        coordinator.set_enabled_tools({"read", "grep", "ls"})

        # Check access
        decision = coordinator.check_access("write", context)
        if not decision.allowed:
            logger.warning(f"Tool denied: {decision.reason}")

        # Get all enabled tools
        enabled = coordinator.get_enabled_tools(context)
    """

    def __init__(
        self,
        tool_registry: "ToolRegistry",
        config: Optional[ToolAccessConfig] = None,
        mode_controller: Optional[Any] = None,
    ) -> None:
        """Initialize the access coordinator.

        Args:
            tool_registry: Tool registry for available tools
            config: Access control configuration
            mode_controller: Optional mode controller for mode-based checks
        """
        self._registry = tool_registry
        self._config = config or ToolAccessConfig()
        self._mode_controller = mode_controller

        # Session-level tool restrictions
        self._session_enabled_tools: Optional[Set[str]] = None

        logger.debug(
            f"ToolAccessCoordinator initialized with "
            f"default_allow_all={self._config.default_allow_all}"
        )

    # =====================================================================
    # Access Control Queries
    # =====================================================================

    def is_tool_enabled(
        self,
        tool_name: str,
        context: Optional[ToolAccessContext] = None,
    ) -> bool:
        """Check if a specific tool is enabled.

        Args:
            tool_name: Name of tool to check
            context: Optional access context

        Returns:
            True if tool is enabled
        """
        decision = self.check_access(tool_name, context)
        return decision.allowed

    def get_enabled_tools(
        self,
        context: Optional[ToolAccessContext] = None,
    ) -> Set[str]:
        """Get currently enabled tool names.

        Args:
            context: Optional access context

        Returns:
            Set of enabled tool names for this session
        """
        # Build context if not provided
        if context is None:
            context = self._build_context()

        # Check mode controller for BUILD mode (allows all tools)
        if self._mode_controller:
            config = self._mode_controller.config
            if config.allow_all_tools:
                all_tools = self.get_available_tools()
                enabled: Set[str] = all_tools - config.disallowed_tools
                return enabled

        # Return session-set tools
        if context.session_enabled_tools:
            return context.session_enabled_tools

        # Fall back to all available tools
        return self.get_available_tools()

    def get_available_tools(self) -> Set[str]:
        """Get all registered tool names.

        Returns:
            Set of tool names available in registry
        """
        if self._registry:
            return set(self._registry.list_tools())
        return set()

    # =====================================================================
    # Access Control Operations
    # =====================================================================

    def check_access(
        self,
        tool_name: str,
        context: Optional[ToolAccessContext] = None,
    ) -> AccessDecision:
        """Check access to a tool with detailed decision.

        Args:
            tool_name: Name of tool to check
            context: Optional access context

        Returns:
            AccessDecision with detailed access information
        """
        # Build context if not provided
        if context is None:
            context = self._build_context()

        # Layer 1: Mode controller restrictions
        if self._mode_controller:
            config = self._mode_controller.config
            if tool_name in config.disallowed_tools:
                return AccessDecision(
                    tool_name=tool_name,
                    allowed=False,
                    reason="Tool disallowed by current mode",
                    layer="mode",
                )
            if config.allow_all_tools and tool_name in self.get_available_tools():
                return AccessDecision(
                    tool_name=tool_name,
                    allowed=True,
                    layer="mode",
                )

        # Layer 2: Session restrictions
        if context.session_enabled_tools:
            if tool_name not in context.session_enabled_tools:
                return AccessDecision(
                    tool_name=tool_name,
                    allowed=False,
                    reason="Tool not in session-enabled set",
                    layer="session",
                )

        # Layer 3: Registry check
        if self._registry and tool_name in self._registry.list_tools():
            return AccessDecision(
                tool_name=tool_name,
                allowed=True,
                layer="registry",
            )

        # Strict mode: reject unknown tools
        if self._config.strict_mode:
            return AccessDecision(
                tool_name=tool_name,
                allowed=False,
                reason="Unknown tool and strict mode enabled",
                layer="strict",
            )

        # Default: allow
        if self._config.default_allow_all:
            return AccessDecision(
                tool_name=tool_name,
                allowed=True,
                layer="default",
            )

        return AccessDecision(
            tool_name=tool_name,
            allowed=False,
            reason="Tool not found and default_allow_all is False",
            layer="default",
        )

    def set_enabled_tools(self, tools: Set[str]) -> None:
        """Set which tools are enabled for this session.

        Args:
            tools: Set of tool names to enable
        """
        self._session_enabled_tools = tools
        logger.info(f"Session enabled tools: {sorted(tools)}")

        # Propagate to tool_selector if available
        if self._mode_controller is not None and hasattr(self._mode_controller, "tool_selector"):
            selector = self._mode_controller.tool_selector
            if selector is not None and hasattr(selector, "set_enabled_tools"):
                selector.set_enabled_tools(tools)

    def clear_session_restrictions(self) -> None:
        """Clear session-level tool restrictions."""
        self._session_enabled_tools = None
        logger.debug("Session tool restrictions cleared")

    # =====================================================================
    # Mode Controller Integration
    # =====================================================================

    def set_mode_controller(self, mode_controller: Any) -> None:
        """Set the mode controller for mode-based access control.

        Args:
            mode_controller: ModeController instance
        """
        self._mode_controller = mode_controller
        logger.debug(f"Mode controller set: {type(mode_controller).__name__}")

    # =====================================================================
    # Internal Helpers
    # =====================================================================

    def _build_context(self) -> ToolAccessContext:
        """Build ToolAccessContext from current state.

        Returns:
            ToolAccessContext with session and mode information
        """
        disallowed: Set[str] = set()
        mode_name = None

        if self._mode_controller:
            mode_name = getattr(self._mode_controller.config, "mode_name", None)
            disallowed = getattr(self._mode_controller.config, "disallowed_tools", set())

        return ToolAccessContext(
            session_enabled_tools=self._session_enabled_tools,
            current_mode=mode_name,
            disallowed_tools=disallowed,
        )


def create_tool_access_coordinator(
    tool_registry: "ToolRegistry",
    default_allow_all: bool = True,
    mode_controller: Optional[Any] = None,
) -> ToolAccessCoordinator:
    """Factory function to create a ToolAccessCoordinator.

    Args:
        tool_registry: Tool registry for available tools
        default_allow_all: Default behavior when no mode controller
        mode_controller: Optional mode controller for mode-based checks

    Returns:
        Configured ToolAccessCoordinator instance
    """
    config = ToolAccessConfig(default_allow_all=default_allow_all)

    return ToolAccessCoordinator(
        tool_registry=tool_registry,
        config=config,
        mode_controller=mode_controller,
    )


__all__ = [
    "ToolAccessCoordinator",
    "ToolAccessConfig",
    "AccessDecision",
    "ToolAccessContext",
    "create_tool_access_coordinator",
]
