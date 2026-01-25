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

"""Tool-related protocols.

This module contains protocols related to tool management, execution, and selection.
These protocols define contracts for:

- Tool registry and management
- Tool execution and validation
- Tool pipeline coordination
- Tool selection and routing

Usage:
    from victor.protocols.agent_tools import (
        ToolRegistryProtocol,
        ToolExecutorProtocol,
        ToolPipelineProtocol,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from victor.agent.tool_pipeline import ToolCallResult
    from victor.tools.enums import CostTier


# =============================================================================
# Tool Protocols
# =============================================================================


@runtime_checkable
class ToolRegistryProtocol(Protocol):
    """Protocol for tool registry.

    Manages tool registration, lookup, and cost tiers.
    """

    def register(self, tool: Any) -> None:
        """Register a tool with the registry."""
        ...

    def get(self, name: str) -> Optional[Any]:
        """Get a tool by name."""
        ...

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        ...

    def get_tool_cost(self, name: str) -> CostTier:
        """Get the cost tier for a tool."""
        ...

    def register_before_hook(self, hook: Any) -> None:
        """Register a hook to run before tool execution."""
        ...


@runtime_checkable
class ToolPipelineProtocol(Protocol):
    """Protocol for tool execution pipeline.

    Coordinates tool execution flow, budget enforcement, and caching.
    """

    @property
    def calls_used(self) -> int:
        """Number of tool calls used in current session."""
        ...

    @property
    def budget(self) -> int:
        """Maximum tool calls allowed."""
        ...

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCallResult:
        """Execute a tool call.

        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        ...

    def is_budget_exhausted(self) -> bool:
        """Check if tool budget is exhausted."""
        ...


@runtime_checkable
class ToolExecutorProtocol(Protocol):
    """Protocol for tool execution - DIP compliant.

    Handles individual tool execution with validation and context support.
    This protocol enables dependency inversion by allowing consumers to
    depend on the abstraction rather than concrete implementations.

    The protocol provides:
    - Synchronous and asynchronous execution methods
    - Argument validation before execution
    - Optional context passing for execution environment

    Usage:
        from victor.protocols.agent_tools import ToolExecutorProtocol

        def run_tool(executor: ToolExecutorProtocol, tool: str, args: dict[str, Any]) -> Any:
            if executor.validate_arguments(tool, args):
                return await executor.aexecute(tool, args)
            raise ValueError(f"Invalid arguments for {tool}")

        # Mock in tests
        mock_executor = MagicMock(spec=ToolExecutorProtocol)
        mock_executor.validate_arguments.return_value = True
    """

    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[Any] = None,
    ) -> Any:
        """Execute a tool synchronously.

        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments as dictionary
            context: Optional execution context (e.g., workspace, session info)

        Returns:
            Tool execution result
        """
        ...

    async def aexecute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[Any] = None,
    ) -> Any:
        """Execute a tool asynchronously.

        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments as dictionary
            context: Optional execution context (e.g., workspace, session info)

        Returns:
            Tool execution result
        """
        ...

    def validate_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> bool:
        """Validate tool arguments before execution.

        Checks that the provided arguments match the tool's expected schema.
        Should be called before execute() or aexecute() to ensure valid input.

        Args:
            tool_name: Name of tool to validate against
            arguments: Arguments to validate

        Returns:
            True if arguments are valid for the tool, False otherwise
        """
        ...


# =============================================================================
# Cache Protocols
# =============================================================================


@runtime_checkable
class ToolCacheProtocol(Protocol):
    """Protocol for tool result caching."""

    def get(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Any]:
        """Get cached result for a tool call."""
        ...

    def set(self, tool_name: str, arguments: Dict[str, Any], result: Any) -> None:
        """Cache a tool result."""
        ...

    def invalidate(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Invalidate a cached result."""
        ...


# =============================================================================
# Output Formatting Protocols
# =============================================================================


@runtime_checkable
class ToolOutputFormatterProtocol(Protocol):
    """Protocol for tool output formatting."""

    def format(
        self,
        tool_name: str,
        result: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format tool output for LLM consumption.

        Args:
            tool_name: Name of the tool
            result: Raw tool result
            context: Optional formatting context

        Returns:
            Formatted output string
        """
        ...


@runtime_checkable
class ResponseSanitizerProtocol(Protocol):
    """Protocol for response sanitization."""

    def sanitize(self, response: str) -> str:
        """Sanitize model response."""
        ...


# =============================================================================
# Utility Protocols
# =============================================================================


@runtime_checkable
class ArgumentNormalizerProtocol(Protocol):
    """Protocol for argument normalization."""

    def normalize(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize tool arguments.

        Handles malformed arguments, type coercion, etc.
        """
        ...


@runtime_checkable
class ProjectContextProtocol(Protocol):
    """Protocol for project context loading."""

    @property
    def content(self) -> Optional[str]:
        """Get loaded project context content."""
        ...

    def load(self) -> None:
        """Load project context from file."""
        ...

    def get_system_prompt_addition(self) -> str:
        """Get context as system prompt addition."""
        ...


# =============================================================================
# Infrastructure Service Protocols
# =============================================================================


@runtime_checkable
class ToolDependencyGraphProtocol(Protocol):
    """Protocol for tool dependency graph.

    Manages tool dependencies for intelligent execution ordering.
    """

    def add_dependency(self, tool: str, depends_on: str) -> None:
        """Add a dependency relationship between tools.

        Args:
            tool: Tool name that has a dependency
            depends_on: Tool that must be executed first
        """
        ...

    def get_dependencies(self, tool: str) -> List[str]:
        """Get dependencies for a tool.

        Args:
            tool: Tool name

        Returns:
            List of tool names that this tool depends on
        """
        ...

    def get_execution_order(self, tools: List[str]) -> List[str]:
        """Get optimal execution order for a list of tools.

        Args:
            tools: List of tool names

        Returns:
            Ordered list respecting dependencies
        """
        ...


@runtime_checkable
class ToolPluginRegistryProtocol(Protocol):
    """Protocol for tool plugin registry.

    Manages dynamic tool loading from plugins.
    """

    def register_plugin(self, plugin_path: str) -> None:
        """Register a plugin directory.

        Args:
            plugin_path: Path to plugin directory
        """
        ...

    def discover_tools(self) -> List[Any]:
        """Discover and load tools from registered plugins.

        Returns:
            List of discovered tool instances
        """
        ...

    def reload_plugins(self) -> None:
        """Reload all registered plugins."""
        ...


__all__ = [
    # Tool protocols
    "ToolRegistryProtocol",
    "ToolPipelineProtocol",
    "ToolExecutorProtocol",
    # Cache protocols
    "ToolCacheProtocol",
    # Output formatting protocols
    "ToolOutputFormatterProtocol",
    "ResponseSanitizerProtocol",
    # Utility protocols
    "ArgumentNormalizerProtocol",
    "ProjectContextProtocol",
    # Infrastructure service protocols
    "ToolDependencyGraphProtocol",
    "ToolPluginRegistryProtocol",
]
