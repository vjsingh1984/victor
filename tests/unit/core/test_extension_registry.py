#!/usr/bin/env python3
"""Tests for Dynamic Extension Registry (OCP-Compliant).

Tests the dynamic extension system that enables Open/Closed Principle
compliance by allowing unlimited extension types without modifying core code.

Test Coverage:
- IExtension protocol compliance
- IExtensionRegistry protocol compliance
- ExtensionRegistry implementation
- Dynamic registration and discovery
- Type-safe extension management
"""

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from victor.core.verticals.protocols import (
    IExtension,
    IExtensionRegistry,
    ExtensionMetadata,
    StandardExtensionTypes,
)


# =============================================================================
# Test Extension Protocol
# =============================================================================


class TestIExtensionProtocol:
    """Tests for IExtension protocol compliance."""

    def test_protocol_defines_required_properties(self):
        """Test that protocol defines required properties."""
        # Protocol should define these properties/methods
        assert hasattr(IExtension, "extension_type")
        assert hasattr(IExtension, "name")
        assert hasattr(IExtension, "validate")
        assert hasattr(IExtension, "get_metadata")

    def test_can_create_mock_from_protocol(self):
        """Test that we can create a mock from the protocol."""
        mock_ext = MagicMock(spec=IExtension)

        # Should have all protocol methods/properties
        assert hasattr(mock_ext, "extension_type")
        assert hasattr(mock_ext, "name")
        assert hasattr(mock_ext, "validate")
        assert hasattr(mock_ext, "get_metadata")


class TestExtensionMetadata:
    """Tests for ExtensionMetadata dataclass."""

    def test_create_default_metadata(self):
        """Test creating metadata with defaults."""
        metadata = ExtensionMetadata()

        assert metadata.version == "0.5.0"
        assert metadata.description == ""
        assert metadata.author == ""
        assert metadata.dependencies == []
        assert metadata.tags == set()
        assert metadata.priority == 50

    def test_create_custom_metadata(self):
        """Test creating custom metadata."""
        metadata = ExtensionMetadata(
            version="2.0.0",
            description="Test extension",
            author="Test Author",
            dependencies=["ext1", "ext2"],
            tags={"test", "example"},
            priority=10,
        )

        assert metadata.version == "2.0.0"
        assert metadata.description == "Test extension"
        assert metadata.author == "Test Author"
        assert "ext1" in metadata.dependencies
        assert "test" in metadata.tags
        assert metadata.priority == 10


# =============================================================================
# Test Extension Implementations
# =============================================================================


@dataclass
class MockExtension(IExtension):
    """Mock extension for testing."""

    _extension_type: str
    _name: str
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[ExtensionMetadata] = None

    @property
    def extension_type(self) -> str:
        return self._extension_type

    @property
    def name(self) -> str:
        return self._name

    def validate(self) -> bool:
        return bool(self.config) and bool(self._name)

    def get_metadata(self) -> Dict[str, Any]:
        if self.metadata:
            return {
                "version": self.metadata.version,
                "description": self.metadata.description,
                "author": self.metadata.author,
            }
        return {"version": "0.5.0"}


class TestExtensionImplementations:
    """Tests for concrete extension implementations."""

    def test_mock_extension_implements_protocol(self):
        """Test that MockExtension implements IExtension."""
        ext = MockExtension(
            _extension_type="test",
            _name="test_ext",
            config={"key": "value"},
        )

        # Should implement protocol
        assert isinstance(ext, IExtension)

    def test_extension_type_property(self):
        """Test extension_type property."""
        ext = MockExtension(
            _extension_type="test_type",
            _name="test_ext",
            config={},
        )

        assert ext.extension_type == "test_type"

    def test_extension_name_property(self):
        """Test name property."""
        ext = MockExtension(
            _extension_type="test",
            _name="my_extension",
            config={},
        )

        assert ext.name == "my_extension"

    def test_extension_validate(self):
        """Test validate method."""
        # Valid extension
        ext_valid = MockExtension(
            _extension_type="test",
            _name="valid",
            config={"data": "value"},
        )
        assert ext_valid.validate() is True

        # Invalid extension (empty config)
        ext_invalid = MockExtension(
            _extension_type="test",
            _name="invalid",
            config={},
        )
        assert ext_invalid.validate() is False  # Empty dict is falsy

        # Invalid extension (empty name)
        ext_invalid_name = MockExtension(
            _extension_type="test",
            _name="",
            config={},
        )
        assert ext_invalid_name.validate() is False

    def test_extension_get_metadata(self):
        """Test get_metadata method."""
        metadata = ExtensionMetadata(
            version="2.0.0",
            description="Test",
        )
        ext = MockExtension(
            _extension_type="test",
            _name="test_ext",
            config={},
            metadata=metadata,
        )

        result = ext.get_metadata()
        assert result["version"] == "2.0.0"
        assert result["description"] == "Test"


# =============================================================================
# Test ExtensionRegistry Protocol
# =============================================================================


class TestIExtensionRegistryProtocol:
    """Tests for IExtensionRegistry protocol compliance."""

    def test_protocol_defines_required_methods(self):
        """Test that protocol defines required methods."""
        required_methods = [
            "register_extension",
            "unregister_extension",
            "get_extension",
            "get_extensions_by_type",
            "list_extension_types",
            "list_extensions",
            "has_extension",
            "count_extensions",
        ]

        for method in required_methods:
            assert hasattr(IExtensionRegistry, method)

    def test_can_create_mock_from_protocol(self):
        """Test that we can create a mock from the protocol."""
        mock_registry = MagicMock(spec=IExtensionRegistry)

        # Should have all protocol methods
        assert hasattr(mock_registry, "register_extension")
        assert hasattr(mock_registry, "unregister_extension")
        assert hasattr(mock_registry, "get_extension")
        assert hasattr(mock_registry, "get_extensions_by_type")
        assert hasattr(mock_registry, "list_extension_types")
        assert hasattr(mock_registry, "list_extensions")
        assert hasattr(mock_registry, "has_extension")
        assert hasattr(mock_registry, "count_extensions")


# =============================================================================
# Test ExtensionRegistry Implementation
# =============================================================================


class TestExtensionRegistry:
    """Tests for ExtensionRegistry implementation.

    Note: These tests verify the registry maintains OCP compliance
    by supporting unlimited extension types without core modifications.
    """

    @pytest.fixture
    def registry(self):
        """Create an ExtensionRegistry instance for testing."""
        from victor.core.verticals.extension_registry import ExtensionRegistry

        return ExtensionRegistry()

    def test_registry_implements_protocol(self, registry):
        """Test that registry implements IExtensionRegistry."""
        assert isinstance(registry, IExtensionRegistry)

    # -------------------------------------------------------------------------
    # Test register_extension
    # -------------------------------------------------------------------------

    def test_register_extension(self, registry):
        """Test registering a valid extension."""
        ext = MockExtension(
            _extension_type="test",
            _name="test_ext",
            config={"key": "value"},
        )

        registry.register_extension(ext)

        assert registry.has_extension("test", "test_ext")
        assert registry.count_extensions("test") == 1

    def test_register_extension_duplicate_raises_error(self, registry):
        """Test that registering duplicate extensions raises error."""
        ext = MockExtension(
            _extension_type="test",
            _name="test_ext",
            config={},
        )

        registry.register_extension(ext)

        # Should raise ValueError on duplicate
        with pytest.raises(ValueError):
            registry.register_extension(ext)

    def test_register_multiple_types(self, registry):
        """Test registering extensions of different types."""
        ext1 = MockExtension(_extension_type="type1", _name="ext1", config={})
        ext2 = MockExtension(_extension_type="type2", _name="ext2", config={})
        ext3 = MockExtension(_extension_type="type1", _name="ext3", config={})

        registry.register_extension(ext1)
        registry.register_extension(ext2)
        registry.register_extension(ext3)

        assert registry.count_extensions("type1") == 2
        assert registry.count_extensions("type2") == 1
        assert registry.count_extensions() == 3

    # -------------------------------------------------------------------------
    # Test unregister_extension
    # -------------------------------------------------------------------------

    def test_unregister_extension(self, registry):
        """Test unregistering an extension."""
        ext = MockExtension(_extension_type="test", _name="test_ext", config={})
        registry.register_extension(ext)

        result = registry.unregister_extension("test", "test_ext")

        assert result is True
        assert not registry.has_extension("test", "test_ext")

    def test_unregister_extension_not_found(self, registry):
        """Test unregistering non-existent extension returns False."""
        result = registry.unregister_extension("test", "nonexistent")
        assert result is False

    # -------------------------------------------------------------------------
    # Test get_extension
    # -------------------------------------------------------------------------

    def test_get_extension_found(self, registry):
        """Test getting a registered extension."""
        ext = MockExtension(_extension_type="test", _name="test_ext", config={})
        registry.register_extension(ext)

        retrieved = registry.get_extension("test", "test_ext")

        assert retrieved is ext
        assert retrieved.name == "test_ext"

    def test_get_extension_not_found(self, registry):
        """Test getting non-existent extension returns None."""
        retrieved = registry.get_extension("test", "nonexistent")
        assert retrieved is None

    # -------------------------------------------------------------------------
    # Test get_extensions_by_type
    # -------------------------------------------------------------------------

    def test_get_extensions_by_type(self, registry):
        """Test getting all extensions of a specific type."""
        ext1 = MockExtension(_extension_type="tools", _name="tool1", config={})
        ext2 = MockExtension(_extension_type="tools", _name="tool2", config={})
        ext3 = MockExtension(_extension_type="middleware", _name="mid1", config={})

        registry.register_extension(ext1)
        registry.register_extension(ext2)
        registry.register_extension(ext3)

        tools = registry.get_extensions_by_type("tools")
        middleware = registry.get_extensions_by_type("middleware")

        assert len(tools) == 2
        assert len(middleware) == 1
        assert ext1 in tools
        assert ext2 in tools
        assert ext3 in middleware

    def test_get_extensions_by_type_empty(self, registry):
        """Test getting extensions for non-existent type."""
        result = registry.get_extensions_by_type("nonexistent")
        assert result == []

    # -------------------------------------------------------------------------
    # Test list_extension_types
    # -------------------------------------------------------------------------

    def test_list_extension_types(self, registry):
        """Test listing all registered extension types."""
        ext1 = MockExtension(_extension_type="type1", _name="ext1", config={})
        ext2 = MockExtension(_extension_type="type2", _name="ext2", config={})
        ext3 = MockExtension(_extension_type="type1", _name="ext3", config={})

        registry.register_extension(ext1)
        registry.register_extension(ext2)
        registry.register_extension(ext3)

        types = registry.list_extension_types()

        assert "type1" in types
        assert "type2" in types
        assert len(types) == 2

    def test_list_extension_types_empty(self, registry):
        """Test listing types when no extensions registered."""
        types = registry.list_extension_types()
        assert types == []

    # -------------------------------------------------------------------------
    # Test list_extensions
    # -------------------------------------------------------------------------

    def test_list_extensions_all(self, registry):
        """Test listing all extensions."""
        ext1 = MockExtension(_extension_type="type1", _name="ext1", config={})
        ext2 = MockExtension(_extension_type="type2", _name="ext2", config={})

        registry.register_extension(ext1)
        registry.register_extension(ext2)

        all_exts = registry.list_extensions()
        type1_exts = registry.list_extensions("type1")

        assert len(all_exts) == 2
        assert len(type1_exts) == 1
        assert "ext1" in all_exts
        assert "ext2" in all_exts

    # -------------------------------------------------------------------------
    # Test has_extension
    # -------------------------------------------------------------------------

    def test_has_extension_true(self, registry):
        """Test has_extension returns True when extension exists."""
        ext = MockExtension(_extension_type="test", _name="test_ext", config={})
        registry.register_extension(ext)

        assert registry.has_extension("test", "test_ext") is True

    def test_has_extension_false(self, registry):
        """Test has_extension returns False when extension doesn't exist."""
        assert registry.has_extension("test", "nonexistent") is False

    # -------------------------------------------------------------------------
    # Test count_extensions
    # -------------------------------------------------------------------------

    def test_count_extensions_all(self, registry):
        """Test counting all extensions."""
        ext1 = MockExtension(_extension_type="type1", _name="ext1", config={})
        ext2 = MockExtension(_extension_type="type2", _name="ext2", config={})
        ext3 = MockExtension(_extension_type="type1", _name="ext3", config={})

        registry.register_extension(ext1)
        registry.register_extension(ext2)
        registry.register_extension(ext3)

        assert registry.count_extensions() == 3
        assert registry.count_extensions("type1") == 2
        assert registry.count_extensions("type2") == 1

    def test_count_extensions_by_type(self, registry):
        """Test counting extensions by specific type."""
        ext1 = MockExtension(_extension_type="tools", _name="tool1", config={})
        ext2 = MockExtension(_extension_type="tools", _name="tool2", config={})

        registry.register_extension(ext1)
        registry.register_extension(ext2)

        assert registry.count_extensions("tools") == 2
        assert registry.count_extensions("middleware") == 0


# =============================================================================
# Test OCP Compliance
# =============================================================================


class TestOCPCompliance:
    """Tests for Open/Closed Principle compliance.

    Verifies that the registry supports unlimited extension types
    without modifying core code.
    """

    @pytest.fixture
    def registry(self):
        """Create an ExtensionRegistry instance."""
        from victor.core.verticals.extension_registry import ExtensionRegistry

        return ExtensionRegistry()

    def test_custom_extension_type_without_core_modifications(self, registry):
        """Test that custom extension types work without core changes.

        This is the key OCP compliance test - we should be able to
        define a completely new extension type without modifying
        ExtensionRegistry or any core code.
        """

        # Define a custom extension type NOT in StandardExtensionTypes
        @dataclass
        class AnalyticsExtension(IExtension):
            _extension_type: str
            _name: str
            api_key: str = ""

            @property
            def extension_type(self) -> str:
                return self._extension_type

            @property
            def name(self) -> str:
                return self._name

            def validate(self) -> bool:
                return bool(self.api_key)

            def get_metadata(self) -> Dict[str, Any]:
                return {"version": "1.0"}

        # Register it without any core modifications
        ext = AnalyticsExtension(
            _extension_type="analytics", _name="google_analytics", api_key="key123"
        )
        registry.register_extension(ext)

        # Should work seamlessly
        assert registry.has_extension("analytics", "google_analytics")
        retrieved = registry.get_extension("analytics", "google_analytics")
        assert isinstance(retrieved, AnalyticsExtension)

    def test_multiple_custom_types(self, registry):
        """Test multiple custom extension types."""

        @dataclass
        class CacheExtension(IExtension):
            _extension_type: str
            _name: str
            size_mb: int = 0

            @property
            def extension_type(self) -> str:
                return self._extension_type

            @property
            def name(self) -> str:
                return self._name

            def validate(self) -> bool:
                return self.size_mb > 0

            def get_metadata(self) -> Dict[str, Any]:
                return {}

        @dataclass
        class MonitoringExtension(IExtension):
            _extension_type: str
            _name: str
            endpoint: str = ""

            @property
            def extension_type(self) -> str:
                return self._extension_type

            @property
            def name(self) -> str:
                return self._name

            def validate(self) -> bool:
                return self.endpoint.startswith("http")

            def get_metadata(self) -> Dict[str, Any]:
                return {}

        cache_ext = CacheExtension(_extension_type="cache", _name="redis", size_mb=100)
        monitor_ext = MonitoringExtension(
            _extension_type="monitoring", _name="prometheus", endpoint="http://localhost:9090"
        )

        registry.register_extension(cache_ext)
        registry.register_extension(monitor_ext)

        types = registry.list_extension_types()
        assert "cache" in types
        assert "monitoring" in types

    def test_standard_extension_types(self, registry):
        """Test that standard extension types work."""
        # Standard types should be usable
        assert StandardExtensionTypes.TOOLS in StandardExtensionTypes
        assert StandardExtensionTypes.MIDDLEWARE in StandardExtensionTypes
        assert StandardExtensionTypes.WORKFLOWS in StandardExtensionTypes

        # Create extensions with standard types
        tool_ext = MockExtension(_extension_type="tools", _name="tool1", config={})
        middleware_ext = MockExtension(_extension_type="middleware", _name="mid1", config={})

        registry.register_extension(tool_ext)
        registry.register_extension(middleware_ext)

        assert registry.count_extensions() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
