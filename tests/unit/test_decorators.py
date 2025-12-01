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

"""Tests for decorators module."""

import pytest
from typing import Dict, Any

from victor.tools.decorators import tool, _create_tool_class
from victor.tools.base import BaseTool, ToolResult


class TestToolDecorator:
    """Tests for @tool decorator."""

    def test_tool_decorator_basic_function(self):
        """Test @tool decorator on a basic function."""

        @tool
        def simple_function(param1: str, param2: int = 5):
            """A simple test function.

            Args:
                param1: First parameter.
                param2: Second parameter.
            """
            return f"{param1}-{param2}"

        # Check that decorator creates Tool attribute (which is an instance)
        assert hasattr(simple_function, "Tool")
        assert isinstance(simple_function.Tool, BaseTool)

        # Function should still be callable
        result = simple_function("test", 10)
        assert result == "test-10"

    def test_tool_decorator_with_context(self):
        """Test @tool decorator with context parameter."""

        @tool
        async def function_with_context(context: Dict[str, Any], value: str):
            """Function that uses context.

            Args:
                context: The tool context.
                value: A value parameter.
            """
            return {"context_used": context is not None, "value": value}

        assert hasattr(function_with_context, "Tool")

    def test_tool_decorator_preserves_metadata(self):
        """Test that decorator preserves function metadata."""

        @tool
        def metadata_function(param: str):
            """Test function for metadata.

            Args:
                param: A parameter.
            """
            return param

        # Wrapper should preserve function name
        assert metadata_function.__name__ == "metadata_function"

    @pytest.mark.asyncio
    async def test_tool_execution_success(self):
        """Test successful tool execution."""

        @tool
        async def async_tool(value: int):
            """Async tool function.

            Args:
                value: An integer value.
            """
            return value * 2

        # Tool is already an instance
        tool_obj = async_tool.Tool

        # Execute tool
        result = await tool_obj.execute({}, value=5)

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == 10
        assert result.error is None

    @pytest.mark.asyncio
    async def test_tool_execution_with_exception(self):
        """Test tool execution with exception."""

        @tool
        async def failing_tool(value: int):
            """Tool that raises exception.

            Args:
                value: An integer value.
            """
            raise ValueError("Test error")

        tool_obj = failing_tool.Tool
        result = await tool_obj.execute({}, value=5)

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.output is None
        assert "Test error" in result.error
        assert result.metadata["exception"] == "ValueError"

    @pytest.mark.asyncio
    async def test_tool_execution_sync_function(self):
        """Test tool execution with synchronous function."""

        @tool
        def sync_tool(value: str):
            """Synchronous tool function.

            Args:
                value: A string value.
            """
            return value.upper()

        tool_obj = sync_tool.Tool
        result = await tool_obj.execute({}, value="hello")

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == "HELLO"

    @pytest.mark.asyncio
    async def test_tool_execution_with_context_injection(self):
        """Test that context is injected when function has context parameter."""

        @tool
        async def context_aware_tool(context: Dict[str, Any], value: str):
            """Tool that uses context.

            Args:
                context: The tool context.
                value: A value.
            """
            return {"has_context": context is not None, "value": value}

        tool_obj = context_aware_tool.Tool
        test_context = {"test_key": "test_value"}
        result = await tool_obj.execute(test_context, value="data")

        assert result.success is True
        assert result.output["has_context"] is True
        assert result.output["value"] == "data"

    def test_tool_properties(self):
        """Test that tool properties are accessible."""

        @tool
        def property_test_tool(param1: str, param2: int = 10):
            """A tool for testing properties.

            This is the long description.

            Args:
                param1: First parameter.
                param2: Second parameter with default.
            """
            return None

        tool_obj = property_test_tool.Tool

        # Test name property
        assert tool_obj.name == "property_test_tool"

        # Test description property
        assert "A tool for testing properties" in tool_obj.description
        assert "long description" in tool_obj.description

        # Test parameters property
        params = tool_obj.parameters
        assert params["type"] == "object"
        assert "param1" in params["properties"]
        assert "param2" in params["properties"]
        assert "param1" in params["required"]
        assert "param2" not in params["required"]  # Has default


class TestCreateToolClass:
    """Tests for _create_tool_class function."""

    def test_create_tool_class_with_typed_parameters(self):
        """Test creating tool class with type annotations."""

        def typed_function(str_param: str, int_param: int, float_param: float, bool_param: bool):
            """Function with typed parameters.

            Args:
                str_param: A string.
                int_param: An integer.
                float_param: A float.
                bool_param: A boolean.
            """
            return None

        tool_obj = _create_tool_class(typed_function)

        params = tool_obj.parameters
        assert params["properties"]["str_param"]["type"] == "string"
        assert params["properties"]["int_param"]["type"] == "number"
        assert params["properties"]["float_param"]["type"] == "number"
        assert params["properties"]["bool_param"]["type"] == "boolean"

    def test_create_tool_class_with_no_docstring(self):
        """Test creating tool class with no docstring."""

        def no_docstring_function(param):
            return param

        tool_obj = _create_tool_class(no_docstring_function)

        assert "No description provided" in tool_obj.description

    def test_create_tool_class_with_var_args(self):
        """Test creating tool class with *args and **kwargs parameters."""

        def var_args_function(*args, **kwargs):
            """Function with variable arguments.

            Args:
                args: Variable positional arguments.
                kwargs: Variable keyword arguments.
            """
            return None

        tool_obj = _create_tool_class(var_args_function)

        # *args and **kwargs should be skipped
        params = tool_obj.parameters
        # Should not include args/kwargs in properties
        assert len(params["properties"]) == 0

    def test_create_tool_class_with_keyword_only_params(self):
        """Test creating tool class with keyword-only parameters."""

        def keyword_only_function(normal_param: str, *, keyword_only: int):
            """Function with keyword-only parameter.

            Args:
                normal_param: A normal parameter.
                keyword_only: A keyword-only parameter.
            """
            return None

        tool_obj = _create_tool_class(keyword_only_function)

        params = tool_obj.parameters
        assert "normal_param" in params["properties"]
        assert "keyword_only" in params["properties"]

    def test_create_tool_class_inherits_base_tool(self):
        """Test that created tool instance is a BaseTool."""

        def test_function(param: str):
            """Test function.

            Args:
                param: A parameter.
            """
            return param

        tool_obj = _create_tool_class(test_function)

        assert isinstance(tool_obj, BaseTool)

    def test_create_tool_class_to_json_schema(self):
        """Test that tool can be converted to JSON schema."""

        def schema_test_function(param: str):
            """Schema test function.

            Args:
                param: A parameter.
            """
            return param

        tool_obj = _create_tool_class(schema_test_function)

        schema = tool_obj.to_json_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "schema_test_function"
        assert "Schema test function" in schema["function"]["description"]
        assert "param" in schema["function"]["parameters"]["properties"]
