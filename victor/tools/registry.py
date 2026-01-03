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

"""Tool registry for managing available tools."""

from typing import Any, Callable, Dict, List, Optional, Union

from victor.core.registry import BaseRegistry
from victor.tools.enums import CostTier


class HookError(Exception):
    """Raised when a critical hook fails."""

    def __init__(self, hook_name: str, original_error: Exception, tool_name: str = ""):
        self.hook_name = hook_name
        self.original_error = original_error
        self.tool_name = tool_name
        super().__init__(
            f"Critical hook '{hook_name}' failed for tool '{tool_name}': {original_error}"
        )


class Hook:
    """Tool execution hook with metadata.

    Hooks can be marked as critical, meaning their failure will prevent
    tool execution (useful for safety checks, validation, etc.).
    """

    def __init__(
        self,
        callback: Callable,
        name: str = "",
        critical: bool = False,
        description: str = "",
    ):
        """Initialize hook.

        Args:
            callback: The hook function to call
            name: Human-readable name for the hook
            critical: If True, hook failure blocks tool execution
            description: Description of what the hook does
        """
        self.callback = callback
        self.name = name or callback.__name__
        self.critical = critical
        self.description = description

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the hook callback."""
        return self.callback(*args, **kwargs)


class ToolRegistry(BaseRegistry[str, Any]):
    """Registry for managing available tools.

    Extends BaseRegistry to provide tool-specific functionality including:
    - Hook system for pre/post execution callbacks
    - Tool enable/disable state management
    - Cost tier filtering
    - JSON schema generation

    The generic type is Any to avoid circular import with BaseTool,
    but values are always BaseTool instances.
    """

    def __init__(self) -> None:
        """Initialize tool registry."""
        super().__init__()
        # Import here to avoid circular dependency
        from victor.tools.base import BaseTool, ToolResult

        self._BaseTool = BaseTool
        self._ToolResult = ToolResult
        # Note: self._items is inherited from BaseRegistry, aliased to _tools for compatibility
        self._tool_enabled: Dict[str, bool] = {}  # Track enabled/disabled state
        self._before_hooks: List[Union[Hook, Callable[[str, Dict[str, Any]], None]]] = []
        self._after_hooks: List[Union[Hook, Callable]] = []

    @property
    def _tools(self) -> Dict[str, Any]:
        """Alias for _items to maintain backward compatibility."""
        return self._items

    @_tools.setter
    def _tools(self, value: Dict[str, Any]) -> None:
        """Setter for _tools alias."""
        self._items = value

    def _wrap_hook(
        self, hook: Union[Hook, Callable], critical: bool = False, name: str = ""
    ) -> Hook:
        """Wrap a callable into a Hook object if needed."""
        if isinstance(hook, Hook):
            return hook
        return Hook(
            callback=hook, name=name or getattr(hook, "__name__", "hook"), critical=critical
        )

    def register_before_hook(
        self,
        hook: Union[Hook, Callable[[str, Dict[str, Any]], None]],
        critical: bool = False,
        name: str = "",
    ) -> None:
        """Register a hook to be called before a tool is executed.

        Args:
            hook: Hook instance or callable that takes (tool_name, arguments)
            critical: If True, hook failure will block tool execution
            name: Human-readable name for the hook (used for error messages)
        """
        wrapped = self._wrap_hook(hook, critical=critical, name=name)
        self._before_hooks.append(wrapped)

    def register_after_hook(
        self,
        hook: Union[Hook, Callable],
        critical: bool = False,
        name: str = "",
    ) -> None:
        """Register a hook to be called after a tool is executed.

        Args:
            hook: Hook instance or callable that takes (tool_result,)
            critical: If True, hook failure will raise an error
            name: Human-readable name for the hook (used for error messages)
        """
        wrapped = self._wrap_hook(hook, critical=critical, name=name)
        self._after_hooks.append(wrapped)

    def register(self, *args, enabled: bool = True, **kwargs) -> None:  # type: ignore[override]
        """Register a tool - supports both signatures for flexibility.

        Signatures:
            register(tool)           - Auto-extracts name from tool object (common case)
            register(key, value)     - Explicit key-value (LSP-compatible with BaseRegistry)

        This unified method provides both backwards compatibility and LSP compliance.
        The first form is preferred for tool registration; the second allows
        ToolRegistry to be used where BaseRegistry is expected.

        Args:
            *args: Either (tool,) or (key, value)
            enabled: Whether the tool is enabled by default (default: True)

        Examples:
            registry.register(my_tool)                    # Auto-extract name
            registry.register("custom_name", my_tool)     # Explicit name
            registry.register(decorated_function)         # From @tool decorator
        """
        if len(args) == 1:
            # Single argument: register(tool) - extract name automatically
            tool = args[0]
            if hasattr(tool, "Tool"):  # It's a decorated function
                tool_instance = tool.Tool
                super().register(tool_instance.name, tool_instance)
                self._tool_enabled[tool_instance.name] = enabled
            elif isinstance(tool, self._BaseTool):  # It's a BaseTool instance
                super().register(tool.name, tool)
                self._tool_enabled[tool.name] = enabled
            else:
                raise TypeError(
                    "Can only register BaseTool instances or functions decorated with @tool"
                )
        elif len(args) == 2:
            # Two arguments: register(key, value) - LSP-compatible
            key, value = args
            super().register(key, value)
            self._tool_enabled[key] = enabled
        else:
            raise TypeError(
                f"register() takes 1 or 2 positional arguments but {len(args)} were given"
            )

    # Alias for backwards compatibility with code using register_tool()
    def register_tool(self, tool: Any, enabled: bool = True) -> None:
        """Alias for register(tool). Kept for backwards compatibility."""
        self.register(tool, enabled=enabled)

    def register_dict(self, tool_dict: Dict[str, Any], enabled: bool = True) -> None:
        """Register a tool from a dictionary definition.

        Used primarily for MCP tool definitions that come as dictionaries.

        Args:
            tool_dict: Dictionary with 'name', 'description', and 'parameters' keys
            enabled: Whether the tool is enabled by default (default: True)
        """
        from victor.tools.base import BaseTool, ToolResult

        name = tool_dict.get("name", "")
        description = tool_dict.get("description", "")
        parameters = tool_dict.get("parameters", {"type": "object", "properties": {}})

        # Create a wrapper tool that stores the dictionary definition
        # This is a placeholder - actual execution is handled by mcp_call
        class DictTool(BaseTool):
            @property
            def name(self) -> str:
                return name

            @property
            def description(self) -> str:
                return description

            @property
            def parameters(self) -> Dict[str, Any]:
                return parameters

            async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
                # MCP tools are executed via mcp_call, not directly
                return ToolResult(
                    success=False,
                    output=None,
                    error="MCP tools should be called via mcp_call",
                )

        super().register(name, DictTool())
        self._tool_enabled[name] = enabled

    def unregister(self, name: str) -> bool:  # type: ignore[override]
        """Unregister a tool.

        Args:
            name: Tool name to unregister

        Returns:
            True if the tool was found and removed, False otherwise
        """
        self._tool_enabled.pop(name, None)
        return super().unregister(name)

    def enable_tool(self, name: str) -> bool:
        """Enable a tool by name.

        Args:
            name: Tool name to enable

        Returns:
            True if tool exists and was enabled, False otherwise
        """
        if name in self._tools:
            self._tool_enabled[name] = True
            return True
        return False

    def disable_tool(self, name: str) -> bool:
        """Disable a tool by name.

        Args:
            name: Tool name to disable

        Returns:
            True if tool exists and was disabled, False otherwise
        """
        if name in self._tools:
            self._tool_enabled[name] = False
            return True
        return False

    def is_tool_enabled(self, name: str) -> bool:
        """Check if a tool is enabled.

        Args:
            name: Tool name

        Returns:
            True if tool is enabled, False otherwise
        """
        return self._tool_enabled.get(name, False)

    def set_tool_states(self, tool_states: Dict[str, bool]) -> None:
        """Set enabled/disabled states for multiple tools.

        Args:
            tool_states: Dictionary mapping tool names to enabled state
        """
        for name, enabled in tool_states.items():
            if name in self._tools:
                self._tool_enabled[name] = enabled

    def get_tool_states(self) -> Dict[str, bool]:
        """Get enabled/disabled states for all tools.

        Returns:
            Dictionary mapping tool names to enabled state
        """
        return self._tool_enabled.copy()

    def get(self, name: str) -> Optional[Any]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return super().get(name)

    def list_tools(self, only_enabled: bool = True) -> list:
        """List all registered tools.

        Args:
            only_enabled: If True, only return enabled tools (default: True)

        Returns:
            List of tool instances
        """
        if only_enabled:
            return [
                tool for name, tool in self._tools.items() if self._tool_enabled.get(name, False)
            ]
        return list(self._tools.values())

    def get_tool_schemas(self, only_enabled: bool = True) -> list[Dict[str, Any]]:
        """Get JSON schemas for all tools.

        Args:
            only_enabled: If True, only return schemas for enabled tools (default: True)

        Returns:
            List of tool JSON schemas
        """
        if only_enabled:
            return [
                tool.to_json_schema()
                for name, tool in self._tools.items()
                if self._tool_enabled.get(name, False)
            ]
        return [tool.to_json_schema() for tool in self._tools.values()]

    def get_tool_cost(self, name: str) -> Optional[CostTier]:
        """Get the cost tier for a tool.

        Args:
            name: Tool name

        Returns:
            CostTier enum value or None if tool not found
        """
        tool = self.get(name)
        if tool:
            return tool.cost_tier
        return None

    def get_tools_by_cost(self, max_tier: CostTier = CostTier.HIGH, only_enabled: bool = True):
        """Get tools filtered by maximum cost tier.

        Args:
            max_tier: Maximum cost tier to include
            only_enabled: If True, only return enabled tools

        Returns:
            List of tools at or below the specified cost tier
        """
        tools = self.list_tools(only_enabled=only_enabled)
        return [t for t in tools if t.cost_tier.weight <= max_tier.weight]

    def get_cost_summary(self, only_enabled: bool = True) -> Dict[str, List[str]]:
        """Get a summary of tools grouped by cost tier.

        Args:
            only_enabled: If True, only include enabled tools

        Returns:
            Dictionary mapping cost tier names to lists of tool names
        """
        summary: Dict[str, List[str]] = {tier.value: [] for tier in CostTier}
        for tool in self.list_tools(only_enabled=only_enabled):
            summary[tool.cost_tier.value].append(tool.name)
        return summary

    async def execute(self, name: str, _exec_ctx: Dict[str, Any], **kwargs: Any):
        """Execute a tool by name.

        Args:
            name: Tool name
            _exec_ctx: Framework execution context (reserved name to avoid collision
                      with tool parameters). Contains shared resources.
            **kwargs: Tool parameters

        Returns:
            ToolResult with execution outcome
        """
        from victor.tools.base import ToolResult

        # Trigger before-execution hooks
        for hook in self._before_hooks:
            hook(name, kwargs)

        tool = self.get(name)
        if tool is None:
            result = ToolResult(
                success=False,
                output=None,
                error=f"Tool '{name}' not found",
                metadata=None,
            )
        elif not self.is_tool_enabled(name):
            result = ToolResult(
                success=False,
                output=None,
                error=f"Tool '{name}' is disabled",
                metadata=None,
            )
        else:
            # Use detailed validation for better error messages
            validation = tool.validate_parameters_detailed(**kwargs)
            if not validation.valid:
                error_msg = f"Invalid parameters for tool '{name}': " + "; ".join(validation.errors)
                result = ToolResult(
                    success=False,
                    output=None,
                    error=error_msg,
                    metadata={"invalid_params": validation.invalid_params},
                )
            else:
                try:
                    result = await tool.execute(_exec_ctx, **kwargs)
                except Exception as e:
                    result = ToolResult(
                        success=False,
                        output=None,
                        error=f"Tool execution failed: {str(e)}",
                        metadata={"exception": type(e).__name__},
                    )

        # Trigger after-execution hooks
        for hook in self._after_hooks:
            hook(result)

        return result
