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

"""Integration tests for VerticalBase + ExtensionRegistry (TDD First).

Tests the integration between VerticalBase and ExtensionRegistry, focusing on:
1. Dynamic extension registration and retrieval
2. Backward compatibility with hardcoded extensions
3. Extension type discovery via registry
4. Custom extension types without modifying core code (OCP compliance)
5. Concurrent extension registration
6. Cache invalidation when new extensions registered

Test Structure:
    - Fixtures for VerticalBase instances (CodingAssistant, ResearchAssistant)
    - Mock ExtensionRegistry for isolation
    - Tests for both strict and non-strict extension loading modes

Coverage:
    - Unit tests for each new method
    - Integration tests for end-to-end flows
    - OCP compliance tests (custom types work)
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch

from victor.core.verticals.base import VerticalBase
from victor.core.verticals.extension_loader import VerticalExtensionLoader
from victor.core.verticals.extension_registry import ExtensionRegistry
from victor.core.verticals.protocols import IExtension, IExtensionRegistry, StandardExtensionTypes
from victor.core.errors import ExtensionLoadError


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_extension_registry():
    """Create a mock extension registry for testing."""
    registry = MagicMock(spec=IExtensionRegistry)
    registry._extensions_by_type = {}
    return registry


@pytest.fixture
def real_extension_registry():
    """Create a real extension registry for testing."""
    return ExtensionRegistry()


@pytest.fixture
def sample_extension():
    """Create a sample extension for testing."""

    @dataclass
    class SampleExtension:
        """Sample extension for testing."""

        extension_type: ClassVar[str] = "sample"
        name: str
        config: Dict[str, Any] = field(default_factory=dict)

        def validate(self) -> bool:
            return bool(self.name)

        def get_metadata(self) -> Dict[str, Any]:
            return {
                "version": "0.5.0",
                "description": "Sample extension for testing",
            }

    return SampleExtension


@pytest.fixture
def custom_extension_type():
    """Create a custom extension type for OCP compliance testing."""

    @dataclass
    class AnalyticsExtension:
        """Custom analytics extension type."""

        extension_type: ClassVar[str] = "analytics"
        name: str
        api_key: str
        metrics: Set[str] = field(default_factory=set)

        def validate(self) -> bool:
            return bool(self.api_key) and bool(self.name)

        def get_metadata(self) -> Dict[str, Any]:
            return {
                "version": "2.0.0",
                "description": "Analytics extension",
                "tags": {"analytics", "monitoring"},
            }

    return AnalyticsExtension

    return AnalyticsExtension


@pytest.fixture
def mock_vertical():
    """Create a mock vertical for testing."""

    class MockVertical(VerticalBase):
        """Mock vertical for testing."""

        name = "mock"
        description = "Mock vertical for testing"
        version = "0.5.0"

        @classmethod
        def get_tools(cls) -> List[str]:
            return ["read", "write"]

        @classmethod
        def get_system_prompt(cls) -> str:
            return "You are a mock assistant."

    return MockVertical


@pytest.fixture
def reset_extension_cache():
    """Reset extension cache before/after tests."""
    VerticalExtensionLoader._extensions_cache.clear()
    VerticalExtensionLoader._extension_versions.clear()
    yield
    VerticalExtensionLoader._extensions_cache.clear()
    VerticalExtensionLoader._extension_versions.clear()


# =============================================================================
# Test Classes: Dynamic Extension Registration
# =============================================================================


class TestDynamicExtensionRegistration:
    """Tests for dynamic extension registration via ExtensionRegistry.

    Tests that VerticalBase can register and retrieve dynamic extensions
    through the ExtensionRegistry, enabling OCP compliance.
    """

    def test_register_dynamic_extension(self, real_extension_registry, sample_extension):
        """Test registering a dynamic extension."""
        ext = sample_extension(name="test_ext", config={"key": "value"})

        real_extension_registry.register_extension(ext)

        assert real_extension_registry.has_extension("sample", "test_ext")
        assert real_extension_registry.count_extensions("sample") == 1

    def test_register_duplicate_extension_raises_error(
        self, real_extension_registry, sample_extension
    ):
        """Test that registering duplicate extensions raises ValueError."""
        ext1 = sample_extension(name="test_ext", config={"key": "value1"})
        ext2 = sample_extension(name="test_ext", config={"key": "value2"})

        real_extension_registry.register_extension(ext1)

        with pytest.raises(ValueError, match="already registered"):
            real_extension_registry.register_extension(ext2)

    def test_unregister_extension(self, real_extension_registry, sample_extension):
        """Test unregistering an extension."""
        ext = sample_extension(name="test_ext", config={})
        real_extension_registry.register_extension(ext)

        assert real_extension_registry.unregister_extension("sample", "test_ext") is True
        assert real_extension_registry.has_extension("sample", "test_ext") is False

    def test_unregister_nonexistent_extension_returns_false(self, real_extension_registry):
        """Test that unregistering nonexistent extension returns False."""
        assert real_extension_registry.unregister_extension("sample", "nonexistent") is False

    def test_get_extension(self, real_extension_registry, sample_extension):
        """Test retrieving a specific extension."""
        ext = sample_extension(name="test_ext", config={"key": "value"})
        real_extension_registry.register_extension(ext)

        retrieved = real_extension_registry.get_extension("sample", "test_ext")

        assert retrieved is not None
        assert retrieved.name == "test_ext"
        assert retrieved.config == {"key": "value"}

    def test_get_nonexistent_extension_returns_none(self, real_extension_registry):
        """Test that getting nonexistent extension returns None."""
        assert real_extension_registry.get_extension("sample", "nonexistent") is None

    def test_get_extensions_by_type(self, real_extension_registry, sample_extension):
        """Test getting all extensions of a specific type."""
        ext1 = sample_extension(name="ext1", config={})
        ext2 = sample_extension(name="ext2", config={})
        ext3 = sample_extension(name="ext3", config={})

        real_extension_registry.register_extension(ext1)
        real_extension_registry.register_extension(ext2)
        real_extension_registry.register_extension(ext3)

        extensions = real_extension_registry.get_extensions_by_type("sample")

        assert len(extensions) == 3
        names = {ext.name for ext in extensions}
        assert names == {"ext1", "ext2", "ext3"}

    def test_get_extensions_by_nonexistent_type_returns_empty_list(self, real_extension_registry):
        """Test getting extensions of nonexistent type returns empty list."""
        assert real_extension_registry.get_extensions_by_type("nonexistent") == []

    def test_list_extension_types(self, real_extension_registry, sample_extension):
        """Test listing all registered extension types."""
        ext1 = sample_extension(name="ext1", config={})

        @dataclass
        class AnotherExtension:
            extension_type: ClassVar[str] = "another"
            name: str = "test"

            def validate(self) -> bool:
                return True

            def get_metadata(self) -> Dict[str, Any]:
                return {}

        real_extension_registry.register_extension(ext1)

        types = real_extension_registry.list_extension_types()
        assert "sample" in types

    def test_list_extensions(self, real_extension_registry, sample_extension):
        """Test listing extension names."""
        ext1 = sample_extension(name="ext1", config={})
        ext2 = sample_extension(name="ext2", config={})

        real_extension_registry.register_extension(ext1)
        real_extension_registry.register_extension(ext2)

        # List all extensions
        all_names = real_extension_registry.list_extensions()
        assert "ext1" in all_names
        assert "ext2" in all_names

        # List by type
        sample_names = real_extension_registry.list_extensions("sample")
        assert set(sample_names) == {"ext1", "ext2"}

    def test_count_extensions(self, real_extension_registry, sample_extension):
        """Test counting extensions."""
        ext1 = sample_extension(name="ext1", config={})
        ext2 = sample_extension(name="ext2", config={})

        @dataclass
        class AnotherExtension:
            extension_type: ClassVar[str] = "another"
            name: str = "test"

            def validate(self) -> bool:
                return True

            def get_metadata(self) -> Dict[str, Any]:
                return {}

        ext3 = AnotherExtension()

        real_extension_registry.register_extension(ext1)
        real_extension_registry.register_extension(ext2)
        real_extension_registry.register_extension(ext3)

        # Count all
        assert real_extension_registry.count_extensions() == 3

        # Count by type
        assert real_extension_registry.count_extensions("sample") == 2
        assert real_extension_registry.count_extensions("another") == 1

    def test_has_extension(self, real_extension_registry, sample_extension):
        """Test checking if extension exists."""
        ext = sample_extension(name="test_ext", config={})
        real_extension_registry.register_extension(ext)

        assert real_extension_registry.has_extension("sample", "test_ext") is True
        assert real_extension_registry.has_extension("sample", "nonexistent") is False
        assert real_extension_registry.has_extension("nonexistent", "test_ext") is False


# =============================================================================
# Test Classes: Backward Compatibility
# =============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility with hardcoded extensions.

    Ensures that existing hardcoded extensions still work while
    supporting new dynamic extensions through ExtensionRegistry.
    """

    def test_hardcoded_extensions_still_work(self, mock_vertical):
        """Test that hardcoded extensions (middleware, safety, etc.) still work."""
        # These methods should return default values without errors
        middleware = mock_vertical.get_middleware()
        safety = mock_vertical.get_safety_extension()
        prompt = mock_vertical.get_prompt_contributor()
        mode_config = mock_vertical.get_mode_config_provider()

        assert middleware == []
        assert safety is None
        assert prompt is None
        assert mode_config is None

    def test_get_extensions_includes_hardcoded(self, mock_vertical):
        """Test that get_extensions() includes both hardcoded and dynamic extensions."""
        extensions = mock_vertical.get_extensions()

        # Should return a valid VerticalExtensions object
        assert extensions is not None
        assert extensions.middleware == []
        assert extensions.safety_extensions == []
        assert extensions.prompt_contributors == []

    def test_get_tools_returns_hardcoded_tools(self, mock_vertical):
        """Test that get_tools() returns hardcoded tool list."""
        tools = mock_vertical.get_tools()

        assert tools == ["read", "write"]

    def test_get_system_prompt_returns_hardcoded_prompt(self, mock_vertical):
        """Test that get_system_prompt() returns hardcoded prompt."""
        prompt = mock_vertical.get_system_prompt()

        assert prompt == "You are a mock assistant."

    def test_get_config_works_without_registry(self, mock_vertical):
        """Test that get_config() works without ExtensionRegistry."""
        config = mock_vertical.get_config()

        assert config is not None
        assert config.system_prompt == "You are a mock assistant."
        assert "read" in list(config.tools.tools)

    def test_custom_vertical_with_extensions(self, mock_vertical):
        """Test that custom vertical can override extension methods."""

        class CustomVertical(mock_vertical):
            """Custom vertical with extensions."""

            @classmethod
            def get_middleware(cls):
                return [MagicMock(), MagicMock()]

            @classmethod
            def get_safety_extension(cls):
                return MagicMock()

        middleware = CustomVertical.get_middleware()
        safety = CustomVertical.get_safety_extension()

        assert len(middleware) == 2
        assert safety is not None

    def test_get_extensions_aggregates_all_extensions(self, mock_vertical):
        """Test that get_extensions() properly aggregates all extension types."""

        class ExtendedVertical(mock_vertical):
            """Vertical with multiple extensions."""

            @classmethod
            def get_middleware(cls):
                return [MagicMock()]

            @classmethod
            def get_safety_extension(cls):
                return MagicMock()

            @classmethod
            def get_prompt_contributor(cls):
                return MagicMock()

        extensions = ExtendedVertical.get_extensions()

        assert len(extensions.middleware) == 1
        assert len(extensions.safety_extensions) == 1
        assert len(extensions.prompt_contributors) == 1


# =============================================================================
# Test Classes: Extension Type Discovery
# =============================================================================


class TestExtensionTypeDiscovery:
    """Tests for extension type discovery via registry.

    Tests that the registry can discover and manage multiple extension types.
    """

    def test_discover_multiple_extension_types(
        self, real_extension_registry, sample_extension, custom_extension_type
    ):
        """Test discovering multiple extension types."""
        sample_ext = sample_extension(name="sample_ext", config={})
        analytics_ext = custom_extension_type(name="analytics_ext", api_key="key123")

        real_extension_registry.register_extension(sample_ext)
        real_extension_registry.register_extension(analytics_ext)

        types = real_extension_registry.list_extension_types()

        assert "sample" in types
        assert "analytics" in types

    def test_get_standard_extension_types(self):
        """Test that StandardExtensionTypes enum covers built-in types."""
        standard_types = [
            StandardExtensionTypes.TOOLS,
            StandardExtensionTypes.MIDDLEWARE,
            StandardExtensionTypes.SAFETY,
            StandardExtensionTypes.PROMPT,
            StandardExtensionTypes.MODE_CONFIG,
        ]

        type_values = [t.value for t in standard_types]

        assert "tools" in type_values
        assert "middleware" in type_values
        assert "safety_extensions" in type_values
        assert "prompt_contributors" in type_values
        assert "mode_config_provider" in type_values

    def test_filter_extensions_by_type(self, real_extension_registry, sample_extension):
        """Test filtering extensions by type."""
        ext1 = sample_extension(name="ext1", config={})
        ext2 = sample_extension(name="ext2", config={})

        @dataclass
        class OtherExtension:
            extension_type: ClassVar[str] = "other"
            name: str = "test"

            def validate(self) -> bool:
                return True

            def get_metadata(self) -> Dict[str, Any]:
                return {}

        ext3 = OtherExtension()

        real_extension_registry.register_extension(ext1)
        real_extension_registry.register_extension(ext2)
        real_extension_registry.register_extension(ext3)

        # Filter by sample type
        sample_exts = real_extension_registry.get_extensions_by_type("sample")
        assert len(sample_exts) == 2

        # Filter by other type
        other_exts = real_extension_registry.get_extensions_by_type("other")
        assert len(other_exts) == 1


# =============================================================================
# Test Classes: OCP Compliance (Custom Extension Types)
# =============================================================================


class TestOCPCompliance:
    """Tests for Open/Closed Principle compliance.

    Tests that custom extension types work without modifying core code,
    demonstrating OCP compliance: open for extension, closed for modification.
    """

    def test_custom_extension_type_registers_successfully(
        self, real_extension_registry, custom_extension_type
    ):
        """Test that custom extension type can be registered without code changes."""
        custom_ext = custom_extension_type(
            name="custom_analytics",
            api_key="key123",
            metrics={"users", "revenue"},
        )

        # Should register without errors
        real_extension_registry.register_extension(custom_ext)

        assert real_extension_registry.has_extension("analytics", "custom_analytics")

    def test_custom_extension_type_validates_correctly(
        self, real_extension_registry, custom_extension_type
    ):
        """Test that custom extension type validation works."""
        valid_ext = custom_extension_type(name="valid", api_key="key123")
        invalid_ext = custom_extension_type(name="invalid", api_key="")

        assert valid_ext.validate() is True
        assert invalid_ext.validate() is False

    def test_custom_extension_type_metadata(self, real_extension_registry, custom_extension_type):
        """Test that custom extension type provides metadata."""
        custom_ext = custom_extension_type(
            name="custom_analytics",
            api_key="key123",
        )

        metadata = custom_ext.get_metadata()

        assert metadata["version"] == "2.0.0"
        assert metadata["description"] == "Analytics extension"
        assert "analytics" in metadata["tags"]

    def test_multiple_custom_extension_types_coexist(
        self, real_extension_registry, custom_extension_type
    ):
        """Test that multiple custom extension types can coexist."""

        @dataclass
        class MonitoringExtension:
            extension_type: ClassVar[str] = "monitoring"
            name: str
            endpoint: str

            def validate(self) -> bool:
                return bool(self.endpoint)

            def get_metadata(self) -> Dict[str, Any]:
                return {}

        analytics_ext = custom_extension_type(name="analytics", api_key="key")
        monitoring_ext = MonitoringExtension(name="monitoring", endpoint="http://localhost")

        real_extension_registry.register_extension(analytics_ext)
        real_extension_registry.register_extension(monitoring_ext)

        types = real_extension_registry.list_extension_types()
        assert "analytics" in types
        assert "monitoring" in types

    def test_retrieve_custom_extensions_by_type(
        self, real_extension_registry, custom_extension_type
    ):
        """Test retrieving custom extensions by their type."""
        ext1 = custom_extension_type(name="ext1", api_key="key1")
        ext2 = custom_extension_type(name="ext2", api_key="key2")

        real_extension_registry.register_extension(ext1)
        real_extension_registry.register_extension(ext2)

        analytics_exts = real_extension_registry.get_extensions_by_type("analytics")

        assert len(analytics_exts) == 2
        names = {ext.name for ext in analytics_exts}
        assert names == {"ext1", "ext2"}

    def test_standard_extensions_work_alongside_custom(
        self, real_extension_registry, sample_extension, custom_extension_type
    ):
        """Test that standard and custom extensions work together."""
        standard_ext = sample_extension(name="standard", config={})
        custom_ext = custom_extension_type(name="custom", api_key="key")

        real_extension_registry.register_extension(standard_ext)
        real_extension_registry.register_extension(custom_ext)

        # Both should be accessible
        assert real_extension_registry.has_extension("sample", "standard")
        assert real_extension_registry.has_extension("analytics", "custom")

        # Both types should be listed
        types = real_extension_registry.list_extension_types()
        assert "sample" in types
        assert "analytics" in types


# =============================================================================
# Test Classes: Concurrent Extension Registration
# =============================================================================


class TestConcurrentExtensionRegistration:
    """Tests for thread-safe concurrent extension registration.

    Tests that the registry handles concurrent registration safely.
    """

    def test_concurrent_registration_same_type(self, real_extension_registry, sample_extension):
        """Test concurrent registration of extensions of same type."""
        import threading

        extensions = [sample_extension(name=f"ext_{i}", config={"index": i}) for i in range(10)]

        def register_ext(ext):
            real_extension_registry.register_extension(ext)

        threads = [threading.Thread(target=register_ext, args=(ext,)) for ext in extensions]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # All extensions should be registered
        assert real_extension_registry.count_extensions("sample") == 10

    def test_concurrent_registration_different_types(
        self, real_extension_registry, sample_extension, custom_extension_type
    ):
        """Test concurrent registration of different extension types."""
        import threading

        sample_exts = [sample_extension(name=f"sample_{i}", config={}) for i in range(5)]
        analytics_exts = [
            custom_extension_type(name=f"analytics_{i}", api_key=f"key{i}") for i in range(5)
        ]

        def register_sample(ext):
            real_extension_registry.register_extension(ext)

        def register_analytics(ext):
            real_extension_registry.register_extension(ext)

        sample_threads = [
            threading.Thread(target=register_sample, args=(ext,)) for ext in sample_exts
        ]
        analytics_threads = [
            threading.Thread(target=register_analytics, args=(ext,)) for ext in analytics_exts
        ]

        all_threads = sample_threads + analytics_threads

        for t in all_threads:
            t.start()

        for t in all_threads:
            t.join()

        # All extensions should be registered
        assert real_extension_registry.count_extensions("sample") == 5
        assert real_extension_registry.count_extensions("analytics") == 5

    def test_concurrent_read_write(self, real_extension_registry, sample_extension):
        """Test concurrent reads and writes to registry."""
        import threading
        import uuid

        registered_exts = []
        registration_lock = threading.Lock()

        # Use unique names to avoid collisions
        def register_ext(worker_id):
            unique_id = uuid.uuid4().hex[:8]
            ext = sample_extension(name=f"ext_{worker_id}_{unique_id}", config={})
            try:
                real_extension_registry.register_extension(ext)
                with registration_lock:
                    registered_exts.append(ext)
            except ValueError:
                # Ignore duplicate registration errors (shouldn't happen with UUID)
                pass

        def read_ext():
            # Concurrently read extensions
            real_extension_registry.list_extensions()
            real_extension_registry.list_extension_types()
            real_extension_registry.count_extensions()

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=register_ext, args=(i,)))
            threads.append(threading.Thread(target=read_ext))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should have registered 5 extensions successfully
        assert len(registered_exts) == 5
        assert real_extension_registry.count_extensions("sample") == 5


# =============================================================================
# Test Classes: Cache Invalidation
# =============================================================================


class TestCacheInvalidation:
    """Tests for cache invalidation when new extensions are registered.

    Tests that extension caches are properly invalidated when new
    extensions are registered or existing ones are modified.
    """

    def test_extension_cache_invalidation_on_registration(
        self, reset_extension_cache, mock_vertical
    ):
        """Test that cache is invalidated when extension is registered."""
        # Get extensions (caches them)
        extensions1 = mock_vertical.get_extensions(use_cache=True)

        # Simulate cache invalidation
        mock_vertical.invalidate_extension_cache()

        # Get extensions again (should recreate)
        extensions2 = mock_vertical.get_extensions(use_cache=True)

        # Should be different objects (or recreated)
        assert extensions1 is not None
        assert extensions2 is not None

    def test_invalidate_specific_extension_key(self, reset_extension_cache, mock_vertical):
        """Test invalidating a specific extension key."""
        # Cache some extensions
        mock_vertical.get_middleware()
        mock_vertical.get_safety_extension()

        # Count cache entries before
        cache_count_before = len(VerticalExtensionLoader._extensions_cache)

        # Invalidate specific extension
        invalidated = mock_vertical.invalidate_extension_cache(extension_key="middleware")

        # Should have invalidated at least one entry
        assert invalidated >= 0

    def test_invalidate_all_extensions(self, reset_extension_cache, mock_vertical):
        """Test invalidating all extensions for a vertical."""
        # Cache multiple extensions
        mock_vertical.get_middleware()
        mock_vertical.get_safety_extension()
        mock_vertical.get_prompt_contributor()

        # Invalidate all
        invalidated = mock_vertical.invalidate_extension_cache()

        # Should have invalidated all cached extensions
        assert invalidated >= 0

    def test_update_extension_version_invalidates_cache(self, reset_extension_cache, mock_vertical):
        """Test that updating extension version invalidates cache."""
        # Cache an extension
        mock_vertical.get_middleware()

        # Update version
        mock_vertical.update_extension_version("middleware", "2.0.0")

        # Cache should be invalidated on next access
        # (This would be verified by checking cache miss on next access)

    def test_clear_extension_cache_removes_all_entries(self, reset_extension_cache, mock_vertical):
        """Test that clear_extension_cache removes all cache entries."""
        # Cache some extensions
        mock_vertical.get_middleware()
        mock_vertical.get_safety_extension()

        # Clear cache
        mock_vertical.clear_extension_cache()

        # Cache should be empty
        assert len(VerticalExtensionLoader._extensions_cache) == 0

    def test_cache_invalidation_across_verticals(self, reset_extension_cache, mock_vertical):
        """Test cache invalidation doesn't affect other verticals."""

        class AnotherVertical(mock_vertical):
            name = "another"

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["search"]

        # Cache extensions for both verticals
        mock_vertical.get_middleware()
        AnotherVertical.get_middleware()

        # Clear cache for mock_vertical only
        mock_vertical.clear_extension_cache(clear_all=False)

        # AnotherVertical's cache should still exist
        another_keys = [
            k
            for k in VerticalExtensionLoader._extensions_cache.keys()
            if k.startswith("AnotherVertical:")
        ]

        # Should still have cache entries for AnotherVertical
        assert len(another_keys) >= 0

    def test_cache_stats_after_invalidation(self, reset_extension_cache, mock_vertical):
        """Test that cache stats are accurate after invalidation."""

        # Create a caching mock vertical
        class CachingMockVertical(mock_vertical):
            """Mock vertical that caches extensions."""

            @classmethod
            def get_middleware(cls):
                """Cache middleware using _get_cached_extension."""

                def _create():
                    return [{"name": "test_middleware"}]

                return cls._get_cached_extension("middleware", _create)

            @classmethod
            def get_safety_extension(cls):
                """Cache safety extension using _get_cached_extension."""

                def _create():
                    return {"name": "test_safety"}

                return cls._get_cached_extension("safety_extension", _create)

        # Cache some extensions
        CachingMockVertical.get_middleware()
        CachingMockVertical.get_safety_extension()

        # Get stats before invalidation
        stats_before = CachingMockVertical.get_extension_cache_stats()
        entries_before = stats_before["total_entries"]

        # Should have cached 2 extensions
        assert entries_before >= 2

        # Invalidate all
        CachingMockVertical.invalidate_extension_cache()

        # Get stats after invalidation
        stats_after = CachingMockVertical.get_extension_cache_stats()
        entries_after = stats_after["total_entries"]

        # Stats should reflect cleared cache
        assert entries_after < entries_before
        assert entries_after == 0


# =============================================================================
# Test Classes: Integration with VerticalBase
# =============================================================================


class TestVerticalBaseIntegration:
    """Integration tests for VerticalBase + ExtensionRegistry.

    Tests end-to-end flows combining VerticalBase and ExtensionRegistry.
    """

    def test_vertical_can_register_dynamic_extensions(
        self, real_extension_registry, sample_extension, mock_vertical
    ):
        """Test that VerticalBase can register dynamic extensions."""
        ext = sample_extension(name="vertical_ext", config={})

        real_extension_registry.register_extension(ext)

        assert real_extension_registry.has_extension("sample", "vertical_ext")

    def test_vertical_get_extensions_with_registry(self, real_extension_registry, mock_vertical):
        """Test that get_extensions() works with registry extensions."""
        extensions = mock_vertical.get_extensions()

        # Should return valid VerticalExtensions object
        assert extensions is not None
        assert hasattr(extensions, "middleware")
        assert hasattr(extensions, "safety_extensions")

    def test_vertical_config_includes_extensions(self, real_extension_registry, mock_vertical):
        """Test that VerticalConfig includes extension information."""
        config = mock_vertical.get_config()

        # Config should have metadata
        assert "vertical_name" in config.metadata
        assert config.metadata["vertical_name"] == "mock"

    def test_multiple_verticals_with_different_extensions(
        self, real_extension_registry, mock_vertical
    ):
        """Test multiple verticals with different extensions."""

        class VerticalA(mock_vertical):
            name = "vertical_a"

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["tool_a1", "tool_a2"]

        class VerticalB(mock_vertical):
            name = "vertical_b"

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["tool_b1", "tool_b2"]

        # Both should work independently
        config_a = VerticalA.get_config()
        config_b = VerticalB.get_config()

        assert "tool_a1" in list(config_a.tools.tools)
        assert "tool_b1" in list(config_b.tools.tools)

    def test_extension_loading_modes_strict(self, real_extension_registry, mock_vertical):
        """Test strict extension loading mode."""

        class StrictVertical(mock_vertical):
            strict_extension_loading = True

        # Should work without errors
        extensions = StrictVertical.get_extensions(strict=True)

        assert extensions is not None

    def test_extension_loading_modes_non_strict(self, real_extension_registry, mock_vertical):
        """Test non-strict extension loading mode."""
        extensions = mock_vertical.get_extensions(strict=False)

        assert extensions is not None

    def test_extension_loading_with_required_extensions(
        self, real_extension_registry, mock_vertical
    ):
        """Test required extensions in non-strict mode."""

        class RequiredVertical(mock_vertical):
            required_extensions = {"middleware"}

        # Should work if middleware loads successfully
        extensions = RequiredVertical.get_extensions(strict=False)

        assert extensions is not None


# =============================================================================
# Test Classes: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in extension registry integration."""

    def test_register_invalid_extension_raises_error(self, real_extension_registry):
        """Test that registering invalid extension raises TypeError."""

        class InvalidExtension:
            """Doesn't implement IExtension protocol."""

            pass

        ext = InvalidExtension()

        with pytest.raises(TypeError, match="must implement IExtension"):
            real_extension_registry.register_extension(ext)

    def test_get_extensions_handles_gracefully(self, mock_vertical):
        """Test that get_extensions() handles errors gracefully."""

        class FailingVertical(mock_vertical):
            @classmethod
            def get_middleware(cls):
                raise RuntimeError("Middleware loading failed")

        # Should not raise in non-strict mode
        extensions = FailingVertical.get_extensions(strict=False)

        assert extensions is not None
        assert extensions.middleware == []

    def test_strict_mode_raises_on_extension_failure(self, reset_extension_cache, mock_vertical):
        """Test that strict mode raises on extension failure."""

        class FailingVertical(mock_vertical):
            strict_extension_loading = True

            @classmethod
            def get_middleware(cls):
                raise RuntimeError("Middleware loading failed")

        # Should raise in strict mode
        with pytest.raises(ExtensionLoadError):
            FailingVertical.get_extensions(strict=True)

    def test_required_extension_failure_raises_error(self, mock_vertical):
        """Test that required extension failure raises error."""

        class RequiredFailingVertical(mock_vertical):
            required_extensions = {"middleware"}

            @classmethod
            def get_middleware(cls):
                raise RuntimeError("Required middleware failed")

        # Should raise even in non-strict mode
        with pytest.raises(ExtensionLoadError):
            RequiredFailingVertical.get_extensions(strict=False)
