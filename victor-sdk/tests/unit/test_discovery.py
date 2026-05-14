"""Unit tests for protocol discovery."""

from unittest.mock import MagicMock, patch

from victor_sdk.discovery import (
    ProtocolRegistry,
    get_global_registry,
    reset_global_registry,
)


class TestProtocolRegistry:
    """Tests for ProtocolRegistry."""

    def test_initialization(self):
        """ProtocolRegistry initializes empty."""
        reset_global_registry()
        registry = ProtocolRegistry()

        assert registry.get_tool_providers() == []
        assert registry.get_safety_providers() == []
        assert registry.get_verticals() == {}

    def test_get_empty_capability_provider(self):
        """get_capability_provider() returns None for unknown capability."""
        registry = ProtocolRegistry()
        assert registry.get_capability_provider("unknown") is None

    def test_get_empty_validator(self):
        """get_validator() returns None for unknown validator."""
        registry = ProtocolRegistry()
        assert registry.get_validator("unknown") is None

    def test_list_empty_verticals(self):
        """list_vertical_names() returns empty list initially."""
        registry = ProtocolRegistry()
        assert registry.list_vertical_names() == []

    def test_list_empty_capabilities(self):
        """list_capability_names() returns empty list initially."""
        registry = ProtocolRegistry()
        assert registry.list_capability_names() == []

    def test_list_empty_validators(self):
        """list_validator_names() returns empty list initially."""
        registry = ProtocolRegistry()
        assert registry.list_validator_names() == []

    @patch("victor_sdk.discovery.importlib.metadata.entry_points")
    def test_loads_preferred_extension_protocol_group(self, mock_entry_points):
        """Preferred victor.extension.protocols entries load like legacy SDK entries."""

        class ToolProviderImpl:
            def get_tools(self):
                return ["read"]

        ep = MagicMock()
        ep.group = "victor.extension.protocols"
        ep.name = "example-tools"
        ep.value = "example:ToolProviderImpl"
        ep.load.return_value = ToolProviderImpl
        mock_entry_points.return_value = [ep]

        registry = ProtocolRegistry()
        stats = registry.load_from_entry_points()

        assert stats.total_protocols == 1
        assert registry.get_tool_providers()[0].get_tools() == ["read"]

    @patch("victor_sdk.discovery.importlib.metadata.entry_points")
    def test_loads_preferred_extension_capability_group(self, mock_entry_points):
        """Preferred victor.extension.capabilities entries load capability providers."""

        provider = object()
        ep = MagicMock()
        ep.group = "victor.extension.capabilities"
        ep.name = "example-capability"
        ep.value = "example:provider"
        ep.load.return_value = provider
        mock_entry_points.return_value = [ep]

        registry = ProtocolRegistry()
        stats = registry.load_from_entry_points()

        assert stats.total_capabilities == 1
        assert registry.get_capability_provider("example-capability") is provider

    @patch("victor_sdk.discovery.importlib.metadata.entry_points")
    def test_loads_preferred_extension_validator_group(self, mock_entry_points):
        """Preferred victor.extension.validators entries load validators."""

        def validator(_package):
            return None

        ep = MagicMock()
        ep.group = "victor.extension.validators"
        ep.name = "example-validator"
        ep.value = "example:validator"
        ep.load.return_value = validator
        mock_entry_points.return_value = [ep]

        registry = ProtocolRegistry()
        stats = registry.load_from_entry_points()

        assert stats.total_validators == 1
        assert registry.get_validator("example-validator") is validator


class TestGlobalRegistry:
    """Tests for global registry instance."""

    def test_get_global_registry_singleton(self):
        """get_global_registry() returns same instance."""
        reset_global_registry()
        registry1 = get_global_registry()
        registry2 = get_global_registry()
        assert registry1 is registry2

    def test_reset_global_registry(self):
        """reset_global_registry() clears the global instance."""
        registry1 = get_global_registry()
        reset_global_registry()
        registry2 = get_global_registry()
        assert registry1 is not registry2
