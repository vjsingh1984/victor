"""Base tool framework for CodingAgent."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Tool parameter definition."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (string, number, boolean, etc.)")
    description: str = Field(..., description="Parameter description")
    enum: Optional[list[str]] = Field(None, description="Allowed values for enum types")
    required: bool = Field(True, description="Whether parameter is required")


class ToolResult(BaseModel):
    """Result from tool execution."""

    success: bool = Field(..., description="Whether execution succeeded")
    output: Any = Field(..., description="Tool output data")
    error: Optional[str] = Field(None, description="Error message if failed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


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
            param_schema = {
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
        type_mapping = {
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
        self._before_hooks: list = []
        self._after_hooks: list = []

    def register_before_hook(self, hook: Callable[[str, Dict[str, Any]], None]) -> None:
        """Register a hook to be called before a tool is executed."""
        self._before_hooks.append(hook)

    def register_after_hook(self, hook: Callable[[ToolResult], None]) -> None:
        """Register a hook to be called after a tool is executed."""
        self._after_hooks.append(hook)


    def register(self, tool: Any) -> None:
        """Register a tool.

        Can register a BaseTool instance or a function decorated with @tool.

        Args:
            tool: Tool instance or decorated function to register
        """
        if hasattr(tool, "Tool"):  # It's a decorated function
            tool_instance = tool.Tool
            self._tools[tool_instance.name] = tool_instance
        elif isinstance(tool, BaseTool):  # It's a class instance
            self._tools[tool.name] = tool
        else:
            raise TypeError("Can only register BaseTool instances or functions decorated with @tool")

    def unregister(self, name: str) -> None:
        """Unregister a tool.

        Args:
            name: Tool name to unregister
        """
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """List all registered tools.

        Returns:
            List of tool instances
        """
        return list(self._tools.values())

    def get_tool_schemas(self) -> list[Dict[str, Any]]:
        """Get JSON schemas for all tools.

        Returns:
            List of tool JSON schemas
        """
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
            )
        elif not tool.validate_parameters(**kwargs):
            result = ToolResult(
                success=False,
                output=None,
                error=f"Invalid parameters for tool '{name}'",
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
