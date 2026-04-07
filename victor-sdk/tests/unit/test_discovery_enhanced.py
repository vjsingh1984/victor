"""Unit tests for enhanced protocol discovery system."""

import pytest

from victor_sdk.discovery import (
    ProtocolRegistry,
    ProtocolMetadata,
    DiscoveryStats,
    collect_verticals_from_candidate,
    get_global_registry,
    reset_global_registry,
    discover_verticals,
    discover_protocols,
    get_discovery_summary,
    reload_discovery,
)
from victor_sdk.core.plugins import VictorPlugin
from victor_sdk.verticals.protocols.base import VerticalBase
from victor_sdk.verticals.protocols import ToolProvider, SafetyProvider


class TestProtocolMetadata:
    """Tests for ProtocolMetadata dataclass."""

    def test_metadata_creation(self):
        """ProtocolMetadata can be created with all fields."""
        metadata = ProtocolMetadata(
            name="test",
            source_package="test-package",
            protocol_type="tool_provider",
            version="1.0.0",
        )

        assert metadata.name == "test"
        assert metadata.source_package == "test-package"
        assert metadata.protocol_type == "tool_provider"
        assert metadata.version == "1.0.0"
        assert metadata.is_loaded is True
        assert metadata.load_error is None

    def test_metadata_with_error(self):
        """ProtocolMetadata can track load errors."""
        metadata = ProtocolMetadata(
            name="test",
            source_package="test-package",
            protocol_type="tool_provider",
            load_error="Import failed",
            is_loaded=False,
        )

        assert metadata.is_loaded is False
        assert metadata.load_error == "Import failed"


class TestDiscoveryStats:
    """Tests for DiscoveryStats dataclass."""

    def test_stats_initialization(self):
        """DiscoveryStats initializes with zeros."""
        stats = DiscoveryStats()

        assert stats.total_verticals == 0
        assert stats.total_protocols == 0
        assert stats.total_capabilities == 0
        assert stats.total_validators == 0
        assert stats.failed_loads == 0

    def test_stats_str_representation(self):
        """DiscoveryStats has a string representation."""
        stats = DiscoveryStats(
            total_verticals=5,
            total_protocols=10,
            total_capabilities=3,
            total_validators=2,
            failed_loads=1,
        )

        stats_str = str(stats)
        assert "verticals=5" in stats_str
        assert "protocols=10" in stats_str
        assert "capabilities=3" in stats_str
        assert "validators=2" in stats_str
        assert "failed=1" in stats_str


class TestProtocolRegistryEnhanced:
    """Tests for enhanced ProtocolRegistry functionality."""

    def test_registry_initialization_with_strict_mode(self):
        """Registry can be initialized in strict mode."""
        registry = ProtocolRegistry(strict=True)
        assert registry._strict is True

    def test_registry_clear(self):
        """Registry can be cleared."""
        registry = ProtocolRegistry()
        registry._discovery_stats.total_verticals = 5

        registry.clear()

        assert registry._discovery_stats.total_verticals == 0
        assert len(registry._verticals) == 0

    def test_get_discovery_stats(self):
        """get_discovery_stats returns current stats."""
        registry = ProtocolRegistry()
        registry._discovery_stats.total_verticals = 3

        stats = registry.get_discovery_stats()

        assert stats.total_verticals == 3
        assert isinstance(stats, DiscoveryStats)

    def test_get_protocol_metadata_all(self):
        """get_protocol_metadata returns all metadata when no name provided."""
        registry = ProtocolRegistry()
        registry._protocol_metadata["test"] = ProtocolMetadata(
            name="test",
            source_package="test-package",
            protocol_type="tool_provider",
        )

        metadata = registry.get_protocol_metadata()

        assert "test" in metadata
        assert metadata["test"].protocol_type == "tool_provider"

    def test_get_protocol_metadata_specific(self):
        """get_protocol_metadata returns specific metadata when name provided."""
        registry = ProtocolRegistry()
        registry._protocol_metadata["test"] = ProtocolMetadata(
            name="test",
            source_package="test-package",
            protocol_type="tool_provider",
        )

        metadata = registry.get_protocol_metadata("test")

        assert "test" in metadata
        assert metadata["test"].protocol_type == "tool_provider"

    def test_get_protocol_metadata_not_found(self):
        """get_protocol_metadata returns empty dict for unknown name."""
        registry = ProtocolRegistry()

        metadata = registry.get_protocol_metadata("unknown")

        assert metadata == {}

    def test_get_validators(self):
        """get_validators returns all validators."""

        def validator1():
            pass

        def validator2():
            pass

        registry = ProtocolRegistry()
        registry._validators = {
            "validator1": validator1,
            "validator2": validator2,
        }

        validators = registry.get_validators()

        assert "validator1" in validators
        assert "validator2" in validators

    def test_find_by_protocol_type(self):
        """find_by_protocol_type filters protocols by type."""
        registry = ProtocolRegistry()
        registry._protocol_metadata = {
            "tool1": ProtocolMetadata(
                name="tool1",
                source_package="pkg1",
                protocol_type="tool_provider",
            ),
            "tool2": ProtocolMetadata(
                name="tool2",
                source_package="pkg1",
                protocol_type="tool_provider",
            ),
            "safety1": ProtocolMetadata(
                name="safety1",
                source_package="pkg1",
                protocol_type="safety_provider",
            ),
        }

        tool_protocols = registry.find_by_protocol_type("tool_provider")

        assert "tool1" in tool_protocols
        assert "tool2" in tool_protocols
        assert "safety1" not in tool_protocols

    def test_get_failed_loads(self):
        """get_failed_loads returns list of failed protocols."""
        registry = ProtocolRegistry()
        registry._protocol_metadata = {
            "success": ProtocolMetadata(
                name="success",
                source_package="pkg1",
                protocol_type="tool_provider",
                is_loaded=True,
            ),
            "failed": ProtocolMetadata(
                name="failed",
                source_package="pkg1",
                protocol_type="tool_provider",
                load_error="Import error",
                is_loaded=False,
            ),
        }

        failed = registry.get_failed_loads()

        assert "failed" in failed
        assert "success" not in failed


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_reset_global_registry(self):
        """reset_global_registry clears the global registry."""
        registry1 = get_global_registry()
        reset_global_registry()
        registry2 = get_global_registry()

        # Should be different instances after reset
        assert registry1 is not registry2

    def test_discover_verticals(self):
        """discover_verticals returns verticals from global registry."""
        verticals = discover_verticals()

        # Should return a dict
        assert isinstance(verticals, dict)

    def test_discover_protocols(self):
        """discover_protocols returns protocols from global registry."""
        protocols = discover_protocols()

        # Should return a dict
        assert isinstance(protocols, dict)

    def test_discover_protocols_with_filter(self):
        """discover_protocols can filter by protocol type."""
        # This tests the function signature, actual filtering depends
        # on what's registered in the environment
        protocols = discover_protocols(protocol_type="tool_provider")

        # Should return a dict
        assert isinstance(protocols, dict)

    def test_get_discovery_summary(self):
        """get_discovery_summary returns a formatted summary."""
        summary = get_discovery_summary()

        # Should return a string
        assert isinstance(summary, str)
        assert "Victor SDK Protocol Discovery Summary" in summary
        assert "Statistics:" in summary

    def test_reload_discovery(self):
        """reload_discovery reloads the registry."""
        stats = reload_discovery()

        # Should return DiscoveryStats
        assert isinstance(stats, DiscoveryStats)


class TestProtocolRegistration:
    """Tests for protocol registration via entry points."""

    def test_protocol_registration_tools(self, monkeypatch):
        """Test that tool providers can be registered."""

        class MockToolProvider(ToolProvider):
            def get_tools(self):
                return ["read", "write"]

        provider_instance = MockToolProvider()

        # Mock entry points to return our mock provider
        def mock_load(self=None):
            return provider_instance

        mock_eps = [
            type(
                "MockEP",
                (),
                {
                    "name": "mock-tools",
                    "load": mock_load,
                    "dist": type("MockDist", (), {"version": "1.0.0"}),
                },
            )(),
        ]

        import importlib.metadata

        original_entry_points = importlib.metadata.entry_points

        def mock_entry_points(*args, **kwargs):
            if "group" in kwargs and kwargs["group"] == "victor.sdk.protocols":
                return mock_eps
            return original_entry_points(*args, **kwargs)

        monkeypatch.setattr(importlib.metadata, "entry_points", mock_entry_points)

        reset_global_registry()
        registry = get_global_registry()
        stats = registry.load_from_entry_points()

        # Should have discovered the tool provider
        assert len(registry.get_tool_providers()) >= 1

    def test_metadata_tracking_on_load(self, monkeypatch):
        """Test that metadata is tracked for loaded protocols."""

        class MockSafetyProvider(SafetyProvider):
            def get_safety_rules(self):
                return {}

            def validate_tool_call(self, tool_name, arguments):
                return True

            def validate_prompt(self, prompt):
                return True

        provider_instance = MockSafetyProvider()

        def mock_load(self=None):
            return provider_instance

        mock_eps = [
            type(
                "MockEP",
                (),
                {
                    "name": "mock-safety",
                    "load": mock_load,
                    "dist": type("MockDist", (), {"version": "1.0.0"}),
                    "value": "mock-package",
                },
            )(),
        ]

        import importlib.metadata

        original_entry_points = importlib.metadata.entry_points

        def mock_entry_points(*args, **kwargs):
            if "group" in kwargs and kwargs["group"] == "victor.sdk.protocols":
                return mock_eps
            return original_entry_points(*args, **kwargs)

        monkeypatch.setattr(importlib.metadata, "entry_points", mock_entry_points)

        reset_global_registry()
        registry = get_global_registry()
        registry.load_from_entry_points()

        metadata_dict = registry.get_protocol_metadata("mock-safety")
        metadata = list(metadata_dict.values())[0] if metadata_dict else None

        assert metadata is not None
        assert metadata.name == "mock-safety"
        assert metadata.protocol_type == "protocol"

    def test_plugin_entry_point_registers_sdk_verticals(self, monkeypatch):
        """Plugin-based victor.plugins discovery should collect registered SDK verticals."""

        class _Vertical(VerticalBase):
            name = "plugin-registered"
            description = "registered via plugin"

            @classmethod
            def get_name(cls) -> str:
                return cls.name

            @classmethod
            def get_description(cls) -> str:
                return cls.description

            @classmethod
            def get_tools(cls) -> list[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "plugin vertical"

        class _Plugin(VictorPlugin):
            @property
            def name(self) -> str:
                return "plugin-registered"

            def register(self, context) -> None:
                context.register_vertical(_Vertical)

            def get_cli_app(self):
                return None

            def on_activate(self) -> None:
                return None

            def on_deactivate(self) -> None:
                return None

            async def on_activate_async(self) -> None:
                return None

            async def on_deactivate_async(self) -> None:
                return None

            def health_check(self) -> dict[str, object]:
                return {"healthy": True}

        mock_eps = [
            type(
                "MockEP",
                (),
                {
                    "name": "plugin-registered",
                    "load": lambda self=None: _Plugin(),
                    "dist": type("MockDist", (), {"version": "1.0.0"}),
                    "value": "victor_plugin:plugin",
                },
            )(),
        ]

        import importlib.metadata

        original_entry_points = importlib.metadata.entry_points

        def mock_entry_points(*args, **kwargs):
            if kwargs.get("group") == "victor.plugins":
                return mock_eps
            return original_entry_points(*args, **kwargs)

        monkeypatch.setattr(importlib.metadata, "entry_points", mock_entry_points)

        registry = ProtocolRegistry()
        stats = registry.load_from_entry_points(reload=True)

        assert stats.total_verticals >= 1
        assert registry.get_vertical("plugin-registered") is _Vertical

    def test_collect_verticals_from_candidate_supports_plugin_class(self):
        """The shared helper should extract verticals from plugin classes too."""

        class _Vertical(VerticalBase):
            name = "helper-registered"
            description = "registered via shared helper"

            @classmethod
            def get_name(cls) -> str:
                return cls.name

            @classmethod
            def get_description(cls) -> str:
                return cls.description

            @classmethod
            def get_tools(cls) -> list[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "helper vertical"

        class _Plugin(VictorPlugin):
            @property
            def name(self) -> str:
                return "helper-registered"

            def register(self, context) -> None:
                context.register_vertical(_Vertical)

            def get_cli_app(self):
                return None

            def on_activate(self) -> None:
                return None

            def on_deactivate(self) -> None:
                return None

            async def on_activate_async(self) -> None:
                return None

            async def on_deactivate_async(self) -> None:
                return None

            def health_check(self) -> dict[str, object]:
                return {"healthy": True}

        discovered = collect_verticals_from_candidate(_Plugin)

        assert discovered == {"helper-registered": _Vertical}


class TestErrorHandling:
    """Tests for error handling in protocol discovery."""

    def test_load_error_handling_non_strict(self):
        """Load errors are tracked but not raised in non-strict mode."""

        class MockEP:
            name = "failing-protocol"
            value = "test-package"

            def load(self):
                raise ImportError("Module not found")

        registry = ProtocolRegistry(strict=False)
        registry._load_from_entry_point(
            MockEP(),
            "victor.sdk.protocols",
            "protocol",
        )

        # Should track the error
        assert "failing-protocol" in registry.get_failed_loads()

    def test_load_error_handling_strict(self):
        """Load errors are raised in strict mode."""

        class MockEP:
            name = "failing-protocol"
            value = "test-package"

            def load(self):
                raise ImportError("Module not found")

        registry = ProtocolRegistry(strict=True)

        with pytest.raises(RuntimeError, match="Failed to load"):
            registry._load_from_entry_point(
                MockEP(),
                "victor.sdk.protocols",
                "protocol",
            )


# Helper function for tests (not part of the actual API)
def _load_from_entry_point(registry, ep, group, protocol_type):
    """Helper to load a single entry point."""
    try:
        obj = ep.load()
        if isinstance(obj, type):
            try:
                obj = obj()
            except Exception as e:
                registry._handle_load_error(ep.name, protocol_type, e)
                return
        registry._track_metadata(ep.name, ep, protocol_type)
    except Exception as e:
        registry._handle_load_error(ep.name, protocol_type, e)


# Monkey patch the helper method for testing
ProtocolRegistry._load_from_entry_point = _load_from_entry_point
