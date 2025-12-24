#!/usr/bin/env python3
"""Test tool configuration system."""

import pytest
from victor.tools.base import ToolRegistry, BaseTool, ToolResult
from typing import Dict, Any


class DummyTool(BaseTool):
    """Dummy tool for testing."""

    @property
    def name(self) -> str:
        return "dummy_tool"

    @property
    def description(self) -> str:
        return "A dummy tool for testing"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, context: Dict[str, Any], **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="Dummy output")


def test_tool_registration_with_enabled():
    """Test tool registration with enabled parameter."""
    registry = ToolRegistry()

    # Register enabled tool
    tool1 = DummyTool()
    registry.register(tool1, enabled=True)
    assert registry.is_tool_enabled("dummy_tool")

    # Register disabled tool
    registry.unregister("dummy_tool")
    registry.register(tool1, enabled=False)
    assert not registry.is_tool_enabled("dummy_tool")


def test_enable_disable_methods():
    """Test enable/disable methods."""
    registry = ToolRegistry()
    tool = DummyTool()
    registry.register(tool)

    # Tool should be enabled by default
    assert registry.is_tool_enabled("dummy_tool")

    # Disable tool
    assert registry.disable_tool("dummy_tool")
    assert not registry.is_tool_enabled("dummy_tool")

    # Enable tool
    assert registry.enable_tool("dummy_tool")
    assert registry.is_tool_enabled("dummy_tool")

    # Try to enable non-existent tool
    assert not registry.enable_tool("non_existent")


def test_set_tool_states():
    """Test setting multiple tool states."""
    registry = ToolRegistry()

    # Register multiple tools
    for i in range(3):

        class TestTool(BaseTool):
            def __init__(self, num):
                self.num = num

            @property
            def name(self) -> str:
                return f"tool_{self.num}"

            @property
            def description(self) -> str:
                return f"Tool {self.num}"

            @property
            def parameters(self) -> Dict[str, Any]:
                return {"type": "object", "properties": {}, "required": []}

            async def execute(self, context: Dict[str, Any], **kwargs: Any) -> ToolResult:
                return ToolResult(success=True, output=f"Tool {self.num} output")

        registry.register(TestTool(i))

    # Set states
    registry.set_tool_states(
        {
            "tool_0": False,
            "tool_1": True,
            "tool_2": False,
        }
    )

    assert not registry.is_tool_enabled("tool_0")
    assert registry.is_tool_enabled("tool_1")
    assert not registry.is_tool_enabled("tool_2")


def test_list_tools_filtering():
    """Test list_tools with only_enabled parameter."""
    registry = ToolRegistry()

    # Register 3 tools, 2 enabled, 1 disabled
    for i in range(3):

        class TestTool(BaseTool):
            def __init__(self, num):
                self.num = num

            @property
            def name(self) -> str:
                return f"tool_{self.num}"

            @property
            def description(self) -> str:
                return f"Tool {self.num}"

            @property
            def parameters(self) -> Dict[str, Any]:
                return {"type": "object", "properties": {}, "required": []}

            async def execute(self, context: Dict[str, Any], **kwargs: Any) -> ToolResult:
                return ToolResult(success=True, output=f"Tool {self.num} output")

        registry.register(TestTool(i), enabled=(i != 1))  # tool_1 is disabled

    # List only enabled tools
    enabled_tools = registry.list_tools(only_enabled=True)
    assert len(enabled_tools) == 2
    assert all(tool.name != "tool_1" for tool in enabled_tools)

    # List all tools
    all_tools = registry.list_tools(only_enabled=False)
    assert len(all_tools) == 3


@pytest.mark.asyncio
async def test_execute_disabled_tool():
    """Test that executing a disabled tool returns an error."""
    registry = ToolRegistry()

    tool = DummyTool()
    registry.register(tool, enabled=False)

    # ToolRegistry.execute signature: (name, _exec_ctx, **kwargs)
    result = await registry.execute("dummy_tool", {})  # _exec_ctx is positional
    assert not result.success
    assert "disabled" in result.error.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
