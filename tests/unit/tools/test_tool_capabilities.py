# Test file for tool capabilities - simplified version

import pytest
from victor.tools.capabilities.system import (
    ToolCapability,
    CapabilityDefinition,
    CapabilityRegistry,
    CapabilitySelector,
)
from victor.tools.capabilities.definitions import BUILTIN_CAPABILITIES


class TestToolCapabilityEnum:
    """Test ToolCapability enum has required capabilities."""
    
    def test_file_read_exists(self):
        assert hasattr(ToolCapability, 'FILE_READ')
        assert ToolCapability.FILE_READ.value == "file_read"
    
    def test_file_write_exists(self):
        assert hasattr(ToolCapability, 'FILE_WRITE')
        assert ToolCapability.FILE_WRITE.value == "file_write"
    
    def test_has_at_least_20_capabilities(self):
        capabilities = [cap.value for cap in ToolCapability]
        assert len(set(capabilities)) >= 20


class TestCapabilityDefinition:
    """Test CapabilityDefinition dataclass."""
    
    def test_create_definition(self):
        definition = CapabilityDefinition(
            name=ToolCapability.FILE_READ,
            description="Read files",
            tools=["read", "ls"],
            dependencies=[],
            conflicts=[],
        )
        assert definition.name == ToolCapability.FILE_READ
        assert definition.tools == ["read", "ls"]


class TestCapabilityRegistry:
    """Test CapabilityRegistry."""
    
    def test_register_and_get_tools(self):
        registry = CapabilityRegistry()
        definition = CapabilityDefinition(
            name=ToolCapability.FILE_READ,
            description="Read files",
            tools=["read"],
            dependencies=[],
            conflicts=[],
        )
        registry.register_capability(definition)
        tools = registry.get_tools_for_capability(ToolCapability.FILE_READ)
        assert "read" in tools
    
    def test_duplicate_raises_error(self):
        registry = CapabilityRegistry()
        definition = CapabilityDefinition(
            name=ToolCapability.FILE_READ,
            description="Read files",
            tools=["read"],
            dependencies=[],
            conflicts=[],
        )
        registry.register_capability(definition)
        with pytest.raises(ValueError):
            registry.register_capability(definition)
    
    def test_resolve_dependencies(self):
        registry = CapabilityRegistry()
        
        registry.register_capability(CapabilityDefinition(
            name=ToolCapability.FILE_READ,
            description="Read files",
            tools=["read"],
            dependencies=[],
            conflicts=[],
        ))
        
        registry.register_capability(CapabilityDefinition(
            name=ToolCapability.FILE_WRITE,
            description="Write files",
            tools=["write"],
            dependencies=[ToolCapability.FILE_READ],
            conflicts=[],
        ))
        
        resolved = registry.resolve_dependencies([ToolCapability.FILE_WRITE])
        assert ToolCapability.FILE_WRITE in resolved
        assert ToolCapability.FILE_READ in resolved


class TestCapabilitySelector:
    """Test CapabilitySelector."""
    
    def test_select_tools_simple(self):
        registry = CapabilityRegistry()
        registry.register_capability(CapabilityDefinition(
            name=ToolCapability.FILE_READ,
            description="Read files",
            tools=["read", "ls"],
            dependencies=[],
            conflicts=[],
        ))
        
        selector = CapabilitySelector(registry)
        tools = selector.select_tools(
            required_capabilities=[ToolCapability.FILE_READ],
            excluded_tools=None,
        )
        assert "read" in tools
        assert "ls" in tools
    
    def test_select_tools_with_exclusions(self):
        registry = CapabilityRegistry()
        registry.register_capability(CapabilityDefinition(
            name=ToolCapability.FILE_READ,
            description="Read files",
            tools=["read", "ls"],
            dependencies=[],
            conflicts=[],
        ))
        
        selector = CapabilitySelector(registry)
        tools = selector.select_tools(
            required_capabilities=[ToolCapability.FILE_READ],
            excluded_tools={"ls"},
        )
        assert "read" in tools
        assert "ls" not in tools


class TestBuiltinCapabilities:
    """Test built-in capability definitions."""
    
    def test_builtin_is_list(self):
        assert isinstance(BUILTIN_CAPABILITIES, list)
    
    def test_builtin_not_empty(self):
        assert len(BUILTIN_CAPABILITIES) > 0
    
    def test_all_are_definitions(self):
        for cap in BUILTIN_CAPABILITIES:
            assert isinstance(cap, CapabilityDefinition)
    
    def test_file_read_exists(self):
        names = [cap.name for cap in BUILTIN_CAPABILITIES]
        assert ToolCapability.FILE_READ in names
    
    def test_file_write_depends_on_read(self):
        file_write = None
        for cap in BUILTIN_CAPABILITIES:
            if cap.name == ToolCapability.FILE_WRITE:
                file_write = cap
                break
        assert file_write is not None
        assert ToolCapability.FILE_READ in file_write.dependencies
