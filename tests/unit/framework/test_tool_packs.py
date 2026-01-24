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

"""Tests for tool packs framework."""

import pytest

from victor.framework.tool_packs import (
    BASE_FILE_OPS,
    BASE_GIT,
    BASE_WEB,
    CODING_PACK,
    DEVOPS_PACK,
    ToolPack,
    ToolPackRegistry,
    create_custom_pack,
    get_tool_pack_registry,
    register_default_packs,
    resolve_tool_pack,
)


class TestToolPack:
    """Tests for ToolPack dataclass."""

    def test_init_minimal(self):
        """Test ToolPack initialization with minimal fields."""
        pack = ToolPack(name="test")
        assert pack.name == "test"
        assert pack.tools == []
        assert pack.description == ""
        assert pack.extends is None
        assert pack.excludes == []
        assert pack.metadata == {}

    def test_init_full(self):
        """Test ToolPack initialization with all fields."""
        pack = ToolPack(
            name="test",
            tools=["tool1", "tool2"],
            description="Test pack",
            extends="base",
            excludes=["tool3"],
            metadata={"key": "value"},
        )
        assert pack.name == "test"
        assert pack.tools == ["tool1", "tool2"]
        assert pack.description == "Test pack"
        assert pack.extends == "base"
        assert pack.excludes == ["tool3"]
        assert pack.metadata == {"key": "value"}

    def test_cannot_extend_self(self):
        """Test that pack cannot extend itself."""
        with pytest.raises(ValueError, match="cannot extend itself"):
            ToolPack(name="test", extends="test")

    def test_requires_name(self):
        """Test that pack must have a name."""
        with pytest.raises(ValueError, match="must have a name"):
            ToolPack(name="")


class TestToolPackRegistry:
    """Tests for ToolPackRegistry."""

    def test_register_and_get(self):
        """Test registering and retrieving a pack."""
        registry = ToolPackRegistry()
        pack = ToolPack(name="test", tools=["tool1", "tool2"])

        registry.register(pack)
        retrieved = registry.get("test")

        assert retrieved is pack
        assert retrieved.name == "test"

    def test_register_duplicate_raises(self):
        """Test that registering duplicate pack raises error."""
        registry = ToolPackRegistry()
        pack1 = ToolPack(name="test", tools=["tool1"])
        pack2 = ToolPack(name="test", tools=["tool2"])

        registry.register(pack1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(pack2)

    def test_get_nonexistent_returns_none(self):
        """Test that getting nonexistent pack returns None."""
        registry = ToolPackRegistry()
        assert registry.get("nonexistent") is None

    def test_resolve_simple_pack(self):
        """Test resolving a pack with no inheritance."""
        registry = ToolPackRegistry()
        pack = ToolPack(name="test", tools=["tool1", "tool2"])
        registry.register(pack)

        tools = registry.resolve("test")
        assert tools == ["tool1", "tool2"]

    def test_resolve_inherited_pack(self):
        """Test resolving a pack that inherits from base."""
        registry = ToolPackRegistry()

        base = ToolPack(name="base", tools=["tool1", "tool2"])
        extended = ToolPack(name="extended", extends="base", tools=["tool3"])

        registry.register(base)
        registry.register(extended)

        tools = registry.resolve("extended")
        # Should have tools from both base and extended
        assert "tool1" in tools
        assert "tool2" in tools
        assert "tool3" in tools
        assert len(tools) == 3

    def test_resolve_with_excludes(self):
        """Test that excluded tools are removed from inherited tools."""
        registry = ToolPackRegistry()

        base = ToolPack(name="base", tools=["tool1", "tool2", "tool3"])
        extended = ToolPack(name="extended", extends="base", tools=["tool4"], excludes=["tool2"])

        registry.register(base)
        registry.register(extended)

        tools = registry.resolve("extended")
        # Should have tool1, tool3 from base (tool2 excluded)
        # And tool4 from extended
        assert "tool1" in tools
        assert "tool2" not in tools  # Excluded by extended pack
        assert "tool3" in tools
        assert "tool4" in tools

    def test_resolve_nonexistent_raises(self):
        """Test that resolving nonexistent pack raises error."""
        registry = ToolPackRegistry()

        with pytest.raises(ValueError, match="not found"):
            registry.resolve("nonexistent")

    def test_resolve_circular_dependency_raises(self):
        """Test that circular dependency is detected."""
        registry = ToolPackRegistry()

        pack1 = ToolPack(name="pack1", extends="pack2")
        pack2 = ToolPack(name="pack2", extends="pack1")

        registry.register(pack1)
        registry.register(pack2)

        with pytest.raises(ValueError, match="Circular dependency"):
            registry.resolve("pack1")

    def test_resolve_with_metadata(self):
        """Test resolving with metadata returns dict."""
        registry = ToolPackRegistry()

        pack = ToolPack(
            name="test",
            tools=["tool1"],
            metadata={"category": "test", "priority": 1},
        )
        registry.register(pack)

        result = registry.resolve("test", include_metadata=True)

        assert isinstance(result, dict)
        assert result["tools"] == ["tool1"]
        assert result["metadata"]["category"] == "test"
        assert result["count"] == 1

    def test_list_packs(self):
        """Test listing all registered packs."""
        registry = ToolPackRegistry()

        registry.register(ToolPack(name="pack1"))
        registry.register(ToolPack(name="pack2"))

        packs = registry.list_packs()
        assert set(packs) == {"pack1", "pack2"}

    def test_get_dependency_graph(self):
        """Test getting dependency graph."""
        registry = ToolPackRegistry()

        base = ToolPack(name="base", tools=["tool1"])
        extended1 = ToolPack(name="extended1", extends="base")
        extended2 = ToolPack(name="extended2", extends="base")

        registry.register(base)
        registry.register(extended1)
        registry.register(extended2)

        graph = registry.get_dependency_graph()

        assert "base" in graph
        assert set(graph["base"]) == {"extended1", "extended2"}


class TestDefaultPacks:
    """Tests for default tool packs."""

    def test_base_file_ops_pack(self):
        """Test base file ops pack has expected tools."""
        assert BASE_FILE_OPS.name == "base_file_ops"
        assert "read" in BASE_FILE_OPS.tools
        assert "write" in BASE_FILE_OPS.tools
        assert "edit" in BASE_FILE_OPS.tools
        assert "search" in BASE_FILE_OPS.tools

    def test_base_web_pack(self):
        """Test base web pack has expected tools."""
        assert BASE_WEB.name == "base_web"
        assert "web_search" in BASE_WEB.tools
        assert "fetch_url" in BASE_WEB.tools

    def test_coding_pack_extends_base(self):
        """Test coding pack extends base_file_ops."""
        assert CODING_PACK.extends == "base_file_ops"
        # The pack.tools only contains its own tools
        assert "semantic_search" in CODING_PACK.tools
        assert "lint" in CODING_PACK.tools
        # Inherited tools are added by registry.resolve()
        tools = resolve_tool_pack("coding")
        assert "read" in tools  # Inherited via resolve

    def test_devops_pack_extends_base(self):
        """Test devops pack extends base_file_ops."""
        assert DEVOPS_PACK.extends == "base_file_ops"
        # The pack.tools only contains its own tools
        assert "docker" in DEVOPS_PACK.tools
        assert "kubernetes" in DEVOPS_PACK.tools
        # Inherited tools are added by registry.resolve()
        tools = resolve_tool_pack("devops")
        assert "read" in tools  # Inherited via resolve


class TestResolveToolPack:
    """Tests for resolve_tool_pack convenience function."""

    def test_resolve_base_pack(self):
        """Test resolving base pack."""
        tools = resolve_tool_pack("base_file_ops")
        assert "read" in tools
        assert "write" in tools

    def test_resolve_extended_pack(self):
        """Test resolving pack that extends base."""
        tools = resolve_tool_pack("coding")
        # Should have base tools + coding-specific tools
        assert "read" in tools
        assert "semantic_search" in tools
        assert "lint" in tools

    def test_resolve_preserves_order(self):
        """Test that tool order is preserved during resolution."""
        registry = ToolPackRegistry()
        registry.register(BASE_FILE_OPS)

        tools = registry.resolve("base_file_ops")
        # Check order matches definition (no reordering)
        assert tools == BASE_FILE_OPS.tools

        # Test with inheritance - base tools should come first
        base = ToolPack(name="base", tools=["tool1", "tool2"])
        extended = ToolPack(name="extended", extends="base", tools=["tool3"])

        registry.register(base)
        registry.register(extended)

        resolved = registry.resolve("extended")
        # Base tools should come before extended tools
        assert resolved.index("tool1") < resolved.index("tool3")
        assert resolved.index("tool2") < resolved.index("tool3")


class TestCreateCustomPack:
    """Tests for create_custom_pack convenience function."""

    def test_create_custom_pack(self):
        """Test creating a custom pack."""
        custom = create_custom_pack(
            name="custom",
            extends="base_file_ops",
            additional_tools=["custom1", "custom2"],
            excludes=["grep"],
        )

        assert custom.name == "custom"
        assert custom.extends == "base_file_ops"
        assert custom.tools == ["custom1", "custom2"]
        assert custom.excludes == ["grep"]

    def test_resolve_custom_pack(self):
        """Test resolving custom pack works correctly."""
        registry = ToolPackRegistry()

        # Register base pack
        registry.register(BASE_FILE_OPS)

        # Create and register custom pack
        custom = create_custom_pack(
            name="custom",
            extends="base_file_ops",
            additional_tools=["custom1"],
            excludes=["grep"],
        )
        registry.register(custom)

        # Resolve
        tools = registry.resolve("custom")

        assert "read" in tools  # From base
        assert "write" in tools  # From base
        assert "custom1" in tools  # From custom
        assert "grep" not in tools  # Excluded


class TestRegisterDefaultPacks:
    """Tests for register_default_packs."""

    def test_register_default_packs(self):
        """Test that default packs are registered."""
        # Get fresh registry (not the global one)
        registry = ToolPackRegistry()
        register_default_packs(registry)

        # Should have all default packs
        packs = registry.list_packs()
        assert "base_file_ops" in packs
        assert "base_web" in packs
        assert "base_git" in packs
        assert "coding" in packs
        assert "devops" in packs
        assert "rag" in packs

    def test_resolve_packs_after_registration(self):
        """Test that packs can be resolved after registration."""
        registry = ToolPackRegistry()
        register_default_packs(registry)

        # Should be able to resolve all packs
        coding_tools = registry.resolve("coding")
        devops_tools = registry.resolve("devops")

        assert len(coding_tools) > 0
        assert len(devops_tools) > 0
        assert "read" in coding_tools
        assert "docker" in devops_tools
