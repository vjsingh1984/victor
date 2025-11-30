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

"""Base tool framework for CodingAgent."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Tool parameter definition."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (string, number, boolean, etc.)")
    description: str = Field(..., description="Parameter description")
    enum: Optional[list[str]] = Field(default=None, description="Allowed values for enum types")
    required: bool = Field(default=True, description="Whether parameter is required")


class ToolResult(BaseModel):
    """Result from tool execution."""

    success: bool = Field(..., description="Whether execution succeeded")
    output: Any = Field(..., description="Tool output data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class BaseTool(ABC):
    """Abstract base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema for tool parameters."""
        pass

    @staticmethod
    def convert_parameters_to_schema(parameters: List[ToolParameter]) -> Dict[str, Any]:
        """Convert list of ToolParameter objects to JSON Schema format.

        Args:
            parameters: List of ToolParameter objects

        Returns:
            JSON Schema dictionary
        """
        properties = {}
        required = []

        for param in parameters:
            param_schema: Dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }

            if param.enum:
                param_schema["enum"] = param.enum

            properties[param.name] = param_schema

            if param.required:
                required.append(param.name)

        schema = {
            "type": "object",
            "properties": properties,
        }

        if required:
            schema["required"] = required

        return schema

    @abstractmethod
    async def execute(self, context: Dict[str, Any], **kwargs: Any) -> ToolResult:
        """Execute the tool.

        Args:
            context: A dictionary of shared resources, e.g. {'code_manager': ...}.
            **kwargs: Tool parameters

        Returns:
            ToolResult with execution outcome
        """
        pass

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert tool to JSON Schema format.

        Returns:
            JSON Schema representation
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate_parameters(self, **kwargs: Any) -> bool:
        """Validate provided parameters against schema.

        Args:
            **kwargs: Parameters to validate

        Returns:
            True if valid, False otherwise
        """
        # Basic validation - can be extended with jsonschema
        required_params = self.parameters.get("required", [])
        properties = self.parameters.get("properties", {})

        # Check required parameters
        for param in required_params:
            if param not in kwargs:
                return False

        # Check parameter types (basic check)
        for param, value in kwargs.items():
            if param in properties:
                expected_type = properties[param].get("type")
                if expected_type and not self._check_type(value, expected_type):
                    return False

        return True

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type.

        Args:
            value: Value to check
            expected_type: Expected JSON Schema type

        Returns:
            True if types match
        """
        type_mapping: Dict[str, Union[type[Any], tuple[type[Any], ...]]] = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, allow it

        return isinstance(value, expected_python_type)


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self) -> None:
        """Initialize tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._tool_enabled: Dict[str, bool] = {}  # Track enabled/disabled state
        self._before_hooks: List[Callable[[str, Dict[str, Any]], None]] = []
        self._after_hooks: List[Callable[[ToolResult], None]] = []

    def register_before_hook(self, hook: Callable[[str, Dict[str, Any]], None]) -> None:
        """Register a hook to be called before a tool is executed."""
        self._before_hooks.append(hook)

    def register_after_hook(self, hook: Callable[[ToolResult], None]) -> None:
        """Register a hook to be called after a tool is executed."""
        self._after_hooks.append(hook)

    def register(self, tool: Any, enabled: bool = True) -> None:
        """Register a tool.

        Can register a BaseTool instance or a function decorated with @tool.

        Args:
            tool: Tool instance or decorated function to register
            enabled: Whether the tool is enabled by default (default: True)
        """
        if hasattr(tool, "Tool"):  # It's a decorated function
            tool_instance = tool.Tool
            self._tools[tool_instance.name] = tool_instance
            self._tool_enabled[tool_instance.name] = enabled
        elif isinstance(tool, BaseTool):  # It's a class instance
            self._tools[tool.name] = tool
            self._tool_enabled[tool.name] = enabled
        else:
            raise TypeError(
                "Can only register BaseTool instances or functions decorated with @tool"
            )

    def register_dict(self, tool_dict: Dict[str, Any], enabled: bool = True) -> None:
        """Register a tool from a dictionary definition.

        Used primarily for MCP tool definitions that come as dictionaries.

        Args:
            tool_dict: Dictionary with 'name', 'description', and 'parameters' keys
            enabled: Whether the tool is enabled by default (default: True)
        """
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

            async def execute(self, context: Dict[str, Any], **kwargs: Any) -> ToolResult:
                # MCP tools are executed via mcp_call, not directly
                return ToolResult(
                    success=False,
                    output=None,
                    error="MCP tools should be called via mcp_call",
                )

        self._tools[name] = DictTool()
        self._tool_enabled[name] = enabled

    def unregister(self, name: str) -> None:
        """Unregister a tool.

        Args:
            name: Tool name to unregister
        """
        self._tools.pop(name, None)
        self._tool_enabled.pop(name, None)

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

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def list_tools(self, only_enabled: bool = True) -> list[BaseTool]:
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

    async def execute(self, name: str, context: Dict[str, Any], **kwargs: Any) -> ToolResult:
        """Execute a tool by name.

        Args:
            name: Tool name
            context: A dictionary of shared resources.
            **kwargs: Tool parameters

        Returns:
            ToolResult with execution outcome
        """
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
        elif not tool.validate_parameters(**kwargs):
            result = ToolResult(
                success=False,
                output=None,
                error=f"Invalid parameters for tool '{name}'",
                metadata=None,
            )
        else:
            try:
                result = await tool.execute(context, **kwargs)
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
