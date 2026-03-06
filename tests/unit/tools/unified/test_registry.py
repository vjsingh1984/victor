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

"""Tests for UnifiedToolRegistry."""

import pytest

from victor.framework.tools import ToolCategory
from victor.tools.base import BaseTool, CostTier, ToolResult
from victor.tools.unified import (
    HookPhase,
    SelectionStrategy,
    ToolMetadata,
    ToolMetrics,
    UnifiedToolRegistry,
)


class DummyTool(BaseTool):
    """Dummy tool for testing."""

    @property
    def name(self) -> str:
        return "dummy_tool"

    @property
    def description(self) -> str:
        return "A dummy tool for testing"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, _exec_ctx, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="dummy output")


class AnotherDummyTool(BaseTool):
    """Another dummy tool for testing."""

    @property
    def name(self) -> str:
        return "another_tool"

    @property
    def description(self) -> str:
        return "Another dummy tool"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, _exec_ctx, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="output")


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry singleton before each test."""
    UnifiedToolRegistry.reset_instance()
    yield
    UnifiedToolRegistry.reset_instance()


class TestUnifiedToolRegistry:
    """Tests for UnifiedToolRegistry core functionality."""

    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """Test that registry follows singleton pattern."""
        registry1 = UnifiedToolRegistry.get_instance()
        registry2 = UnifiedToolRegistry.get_instance()

        assert registry1 is registry2
        assert id(registry1) == id(registry2)

    @pytest.mark.asyncio
    async def test_direct_instantiation_raises_error(self):
        """Test that direct instantiation raises RuntimeError."""
        with pytest.raises(RuntimeError, match="singleton"):
            UnifiedToolRegistry()

    @pytest.mark.asyncio
    async def test_register_tool(self):
        """Test registering a tool."""
        registry = UnifiedToolRegistry.get_instance()
        tool = DummyTool()

        await registry.register(tool)

        # Tool should be accessible
        retrieved = registry.get("dummy_tool")
        assert retrieved is not None
        assert retrieved.name == "dummy_tool"

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test listing tools."""
        registry = UnifiedToolRegistry.get_instance()

        await registry.register(DummyTool())
        await registry.register(AnotherDummyTool())

        # List all tools
        all_tools = registry.list_tools(enabled_only=False)
        assert set(all_tools) == {"dummy_tool", "another_tool"}

        # List enabled tools
        enabled_tools = registry.list_tools(enabled_only=True)
        assert set(enabled_tools) == {"dummy_tool", "another_tool"}

    @pytest.mark.asyncio
    async def test_enable_disable_tool(self):
        """Test enabling and disabling tools."""
        registry = UnifiedToolRegistry.get_instance()
        await registry.register(DummyTool())

        # Disable tool
        result = await registry.disable("dummy_tool")
        assert result is True

        # Should not appear in enabled list
        enabled_tools = registry.list_tools(enabled_only=True)
        assert "dummy_tool" not in enabled_tools

        # Re-enable tool
        result = await registry.enable("dummy_tool")
        assert result is True

        # Should appear in enabled list
        enabled_tools = registry.list_tools(enabled_only=True)
        assert "dummy_tool" in enabled_tools

    @pytest.mark.asyncio
    async def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = UnifiedToolRegistry.get_instance()
        await registry.register(DummyTool())

        # Unregister
        result = await registry.unregister("dummy_tool")
        assert result is True

        # Tool should be gone
        assert registry.get("dummy_tool") is None

        # Unregistering again should return False
        result = await registry.unregister("dummy_tool")
        assert result is False

    @pytest.mark.asyncio
    async def test_tool_metadata(self):
        """Test tool metadata."""
        registry = UnifiedToolRegistry.get_instance()
        await registry.register(
            DummyTool(),
            category=ToolCategory.FILESYSTEM,
            tier=CostTier.LOW,
        )

        metadata = registry.get_metadata("dummy_tool")

        assert metadata.name == "dummy_tool"
        assert metadata.category == ToolCategory.FILESYSTEM
        assert metadata.tier == CostTier.LOW
        assert metadata.enabled is True

    @pytest.mark.asyncio
    async def test_get_categories(self):
        """Test getting tools by category."""
        registry = UnifiedToolRegistry.get_instance()

        await registry.register(
            DummyTool(),
            category=ToolCategory.FILESYSTEM,
        )
        await registry.register(
            AnotherDummyTool(),
            category=ToolCategory.GIT,
        )

        categories = registry.get_categories()

        assert ToolCategory.FILESYSTEM in categories
        assert ToolCategory.GIT in categories
        assert "dummy_tool" in categories[ToolCategory.FILESYSTEM]
        assert "another_tool" in categories[ToolCategory.GIT]

    @pytest.mark.asyncio
    async def test_tool_aliases(self):
        """Test tool aliases."""
        registry = UnifiedToolRegistry.get_instance()
        await registry.register(DummyTool())

        # Add alias
        registry.add_alias("dummy_tool", "dummy")

        # Resolve alias
        canonical = registry.resolve_alias("dummy")
        assert canonical == "dummy_tool"

        # Get tool via alias
        tool = registry.get("dummy")
        assert tool is not None
        assert tool.name == "dummy_tool"

    @pytest.mark.asyncio
    async def test_deprecate_tool(self):
        """Test deprecating a tool."""
        registry = UnifiedToolRegistry.get_instance()
        await registry.register(DummyTool())

        # Deprecate
        await registry.deprecate(
            "dummy_tool",
            replacement="another_tool",
            message="Use another_tool instead",
        )

        metadata = registry.get_metadata("dummy_tool")
        assert metadata.deprecated is True
        assert metadata.replacement == "another_tool"
        assert metadata.deprecation_message == "Use another_tool instead"

    @pytest.mark.asyncio
    async def test_get_schemas(self):
        """Test getting JSON schemas."""
        registry = UnifiedToolRegistry.get_instance()
        await registry.register(DummyTool())

        schemas = registry.get_schemas(enabled_only=True)

        assert len(schemas) == 1
        assert schemas[0]["name"] == "dummy_tool"
        assert schemas[0]["description"] == "A dummy tool for testing"
        assert schemas[0]["parameters"]["type"] == "object"

    @pytest.mark.asyncio
    async def test_record_execution_metrics(self):
        """Test recording tool execution metrics."""
        registry = UnifiedToolRegistry.get_instance()
        await registry.register(DummyTool())

        # Record successful execution
        registry.record_execution("dummy_tool", success=True, duration_ms=100.0)
        registry.record_execution("dummy_tool", success=True, duration_ms=200.0)
        registry.record_execution("dummy_tool", success=False, duration_ms=50.0)

        metrics = registry.get_metrics("dummy_tool")

        assert metrics.call_count == 3
        assert metrics.success_count == 2
        assert metrics.error_count == 1
        assert metrics.last_used is not None
        # EMA calculation: after 100, 200, 50 -> ~108.33 (exponential moving average)
        # EMA formula: alpha = 2/(n+1), EMA = alpha*new + (1-alpha)*old_ema
        # After 100: avg=100
        # After 200: avg=0.666*200 + 0.333*100 = 166.67
        # After 50: avg=0.5*50 + 0.5*166.67 = 108.33
        assert 108 < metrics.avg_duration_ms < 109

    def test_singleton_reset(self):
        """Test resetting the singleton."""
        registry1 = UnifiedToolRegistry.get_instance()
        UnifiedToolRegistry.reset_instance()

        registry2 = UnifiedToolRegistry.get_instance()

        # Should be different instances after reset
        assert registry1 is not registry2


@pytest.mark.asyncio
class TestRegistryAdapters:
    """Tests for backward compatibility adapters."""

    async def test_shared_registry_adapter(self):
        """Test SharedToolRegistryAdapter."""
        from victor.tools.unified.adapters import SharedToolRegistryAdapter

        adapter = SharedToolRegistryAdapter()
        await adapter._unified.register(DummyTool())

        # Get tool classes
        tools = await adapter.get_tool_classes()
        assert "dummy_tool" in tools

        # Create instance
        instance = adapter.create_tool_instance("dummy_tool")
        assert instance is not None

    async def test_tool_registry_adapter(self):
        """Test ToolRegistryAdapter."""
        from victor.tools.unified.adapters import ToolRegistryAdapter

        adapter = ToolRegistryAdapter()

        # Register tool directly via unified registry (adapter.register is async-only)
        await adapter._unified.register(DummyTool())

        # List tools
        tools = adapter.list_tools()
        assert len(tools) > 0

        # Get tool
        tool = adapter.get("dummy_tool")
        assert tool is not None

        # Get schemas
        schemas = adapter.get_tool_schemas()
        assert len(schemas) > 0
