"""Unit tests for protocol discovery."""


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
