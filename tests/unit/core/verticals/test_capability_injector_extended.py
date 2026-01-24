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

"""Tests for extended CapabilityInjector with multi-capability support.

Tests for the SOLID-compliant capability injection system following:
- SRP: Each class has single responsibility
- OCP: Open for extension via registration
- LSP: All providers implement ICapabilityProvider protocol
- ISP: Narrow, focused protocols
- DIP: Depend on abstractions, not concretions

Run with: pytest tests/unit/core/verticals/test_capability_injector_extended.py -v
"""

import warnings

import pytest

from victor.core.verticals.capability_provider import (
    ICapabilityProvider,
    IConfigurableCapability,
    BaseCapabilityProvider,
    CapabilityProviderRegistry,
    get_capability_registry,
    reset_global_registry,
)
from victor.core.verticals.capability_injector import (
    CapabilityInjector,
    get_capability_injector,
    create_capability_injector,
    _FileOperationsProvider,
)


# =============================================================================
# Custom Capability Providers for Testing
# =============================================================================


class _MockCapability:
    """Mock capability for testing."""

    def __init__(self):
        self.initialized = True


class _MockConfigurableCapability:
    """Mock configurable capability for testing."""

    def __init__(self):
        self.config = {}

    def configure(self, **kwargs):
        """Configure the capability."""
        self.config.update(kwargs)


class _MockCapabilityProvider(BaseCapabilityProvider):
    """Mock capability provider for testing."""

    def __init__(self, container=None):
        super().__init__("mock_capability", container)
        self.create_count = 0

    def _create_instance(self):
        self.create_count += 1
        return _MockCapability()


class _MockConfigurableProvider(BaseCapabilityProvider):
    """Mock configurable provider for testing."""

    def __init__(self, container=None):
        super().__init__("mock_configurable", container)

    def _create_instance(self):
        return _MockConfigurableCapability()


# =============================================================================
# Capability Provider Protocol Tests
# =============================================================================


class TestICapabilityProvider:
    """Tests for ICapabilityProvider protocol (ISP + LSP compliance)."""

    def test_provider_implements_protocol(self):
        """Test that BaseCapabilityProvider implements ICapabilityProvider."""
        provider = _MockCapabilityProvider()

        # Verify protocol compliance
        assert isinstance(provider, ICapabilityProvider)
        assert hasattr(provider, "name")
        assert hasattr(provider, "get_instance")
        assert hasattr(provider, "reset")

    def test_provider_name_property(self):
        """Test provider name property."""
        provider = _MockCapabilityProvider()
        assert provider.name == "mock_capability"

    def test_get_instance_lazy_initialization(self):
        """Test that instance is created lazily."""
        provider = _MockCapabilityProvider()

        # Before getting instance, create_count is 0
        assert provider.create_count == 0

        # Get instance
        instance = provider.get_instance()

        # Should have been created once
        assert provider.create_count == 1
        assert isinstance(instance, _MockCapability)
        assert instance.initialized is True

    def test_get_instance_singleton(self):
        """Test that get_instance returns singleton."""
        provider = _MockCapabilityProvider()

        instance1 = provider.get_instance()
        instance2 = provider.get_instance()

        # Same instance
        assert instance1 is instance2

        # Only created once
        assert provider.create_count == 1

    def test_reset_clears_instance(self):
        """Test that reset clears cached instance."""
        provider = _MockCapabilityProvider()

        # Get instance
        instance1 = provider.get_instance()
        assert provider.create_count == 1

        # Reset
        provider.reset()

        # Get instance again
        instance2 = provider.get_instance()

        # New instance created
        assert provider.create_count == 2
        assert instance1 is not instance2


class TestIConfigurableCapability:
    """Tests for IConfigurableCapability protocol (ISP compliance)."""

    def test_configurable_capability_protocol(self):
        """Test that IConfigurableCapability protocol exists."""
        capability = _MockConfigurableCapability()

        # Verify protocol compliance
        assert isinstance(capability, IConfigurableCapability)
        assert hasattr(capability, "configure")

    def test_configure_method(self):
        """Test configure method works."""
        capability = _MockConfigurableCapability()

        capability.configure(option1="value1", option2="value2")

        assert capability.config == {"option1": "value1", "option2": "value2"}


# =============================================================================
# Capability Registry Tests
# =============================================================================


class TestCapabilityProviderRegistry:
    """Tests for CapabilityProviderRegistry (OCP + DIP compliance)."""

    def test_init_empty_registry(self):
        """Test registry starts empty."""
        registry = CapabilityProviderRegistry()

        assert registry.list_providers() == []
        assert registry.has_provider("test") is False

    def test_register_provider(self):
        """Test registering a provider."""
        registry = CapabilityProviderRegistry()
        provider = _MockCapabilityProvider()

        registry.register(provider)

        assert registry.has_provider("mock_capability") is True
        assert "mock_capability" in registry.list_providers()

    def test_register_duplicate_raises(self):
        """Test registering duplicate provider raises ValueError."""
        registry = CapabilityProviderRegistry()
        provider1 = _MockCapabilityProvider()
        provider2 = _MockCapabilityProvider()

        registry.register(provider1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(provider2)

    def test_get_provider(self):
        """Test getting registered provider."""
        registry = CapabilityProviderRegistry()
        provider = _MockCapabilityProvider()

        registry.register(provider)
        retrieved = registry.get_provider("mock_capability")

        assert retrieved is provider

    def test_get_provider_not_found(self):
        """Test getting non-existent provider returns None."""
        registry = CapabilityProviderRegistry()

        provider = registry.get_provider("nonexistent")

        assert provider is None

    def test_unregister_provider(self):
        """Test unregistering a provider."""
        registry = CapabilityProviderRegistry()
        provider = _MockCapabilityProvider()

        registry.register(provider)
        assert registry.has_provider("mock_capability") is True

        result = registry.unregister("mock_capability")

        assert result is True
        assert registry.has_provider("mock_capability") is False

    def test_unregister_not_found(self):
        """Test unregistering non-existent provider returns False."""
        registry = CapabilityProviderRegistry()

        result = registry.unregister("nonexistent")

        assert result is False

    def test_clear(self):
        """Test clearing all providers."""
        registry = CapabilityProviderRegistry()
        registry.register(_MockCapabilityProvider())
        registry.register(_MockConfigurableProvider())

        assert len(registry.list_providers()) == 2

        registry.clear()

        assert len(registry.list_providers()) == 0

    def test_reset_all(self):
        """Test resetting all provider instances."""
        registry = CapabilityProviderRegistry()
        provider = _MockCapabilityProvider()

        registry.register(provider)

        # Get instance (triggers creation)
        provider.get_instance()
        assert provider.create_count == 1

        # Reset all
        registry.reset_all()

        # Get instance again (should recreate)
        provider.get_instance()
        assert provider.create_count == 2


# =============================================================================
# Capability Injector Tests
# =============================================================================


class TestCapabilityInjector:
    """Tests for CapabilityInjector (SOLID compliance)."""

    def test_init_auto_registers_builtins(self):
        """Test that built-in capabilities are auto-registered."""
        # Reset global state first
        reset_global_registry()
        CapabilityInjector.reset_global()

        injector = CapabilityInjector()

        assert injector.has_capability("file_operations")

    def test_get_capability_by_name(self):
        """Test getting capability by name (new API)."""
        reset_global_registry()
        CapabilityInjector.reset_global()

        injector = CapabilityInjector()

        capability = injector.get_capability("file_operations")

        assert capability is not None

    def test_get_capability_default(self):
        """Test getting non-existent capability returns default."""
        reset_global_registry()
        CapabilityInjector.reset_global()

        injector = CapabilityInjector()

        result = injector.get_capability("nonexistent", default="default_value")

        assert result == "default_value"

    def test_register_custom_provider(self):
        """Test registering custom capability provider (OCP compliance)."""
        reset_global_registry()
        CapabilityInjector.reset_global()

        injector = CapabilityInjector()
        provider = _MockCapabilityProvider()

        injector.register_provider(provider)

        assert injector.has_capability("mock_capability")

        capability = injector.get_capability("mock_capability")
        assert isinstance(capability, _MockCapability)

    def test_list_capabilities(self):
        """Test listing all registered capabilities."""
        reset_global_registry()
        CapabilityInjector.reset_global()

        injector = CapabilityInjector()
        injector.register_provider(_MockCapabilityProvider())
        injector.register_provider(_MockConfigurableProvider())

        capabilities = injector.list_capabilities()

        assert "file_operations" in capabilities
        assert "mock_capability" in capabilities
        assert "mock_configurable" in capabilities

    def test_reset(self):
        """Test resetting all capability instances."""
        reset_global_registry()
        CapabilityInjector.reset_global()

        injector = CapabilityInjector()
        provider = _MockCapabilityProvider()
        injector.register_provider(provider)

        # Get instance (creates it)
        capability1 = injector.get_capability("mock_capability")
        assert provider.create_count == 1

        # Reset
        injector.reset()

        # Get instance again (should recreate)
        capability2 = injector.get_capability("mock_capability")
        assert provider.create_count == 2
        assert capability1 is not capability2

    def test_backward_compatible_get_file_operations(self):
        """Test backward compatible get_file_operations_capability method."""
        reset_global_registry()
        CapabilityInjector.reset_global()

        injector = CapabilityInjector()

        # Old API should still work
        capability = injector.get_file_operations_capability()

        assert capability is not None


# =============================================================================
# Global Registry Tests
# =============================================================================


class TestGlobalRegistry:
    """Tests for global capability registry (Singleton pattern)."""

    def test_global_registry_singleton(self):
        """Test that get_capability_registry returns singleton."""
        reset_global_registry()

        registry1 = get_capability_registry()
        registry2 = get_capability_registry()

        assert registry1 is registry2

    def test_reset_global_registry(self):
        """Test resetting global registry."""
        registry = get_capability_registry()
        registry.register(_MockCapabilityProvider())

        assert len(registry.list_providers()) > 0

        # Reset
        reset_global_registry()

        # Get new registry (should be empty)
        new_registry = get_capability_registry()
        assert len(new_registry.list_providers()) == 0


# =============================================================================
# Global Injector Tests
# =============================================================================


class TestGlobalInjector:
    """Tests for global capability injector (Singleton pattern)."""

    def test_global_injector_singleton(self):
        """Test that get_capability_injector returns singleton."""
        CapabilityInjector.reset_global()

        injector1 = get_capability_injector()
        injector2 = get_capability_injector()

        assert injector1 is injector2

    def test_global_injector_reset(self):
        """Test resetting global injector."""
        CapabilityInjector.reset_global()

        injector1 = get_capability_injector()
        CapabilityInjector.reset_global()

        injector2 = get_capability_injector()

        assert injector1 is not injector2


# =============================================================================
# SOLID Principles Verification Tests
# =============================================================================


class TestSOLIDCompliance:
    """Tests to verify SOLID principles compliance."""

    def test_srp_separate_responsibilities(self):
        """Test SRP: Registry and Injector have separate responsibilities."""
        # Registry manages registration
        registry = CapabilityProviderRegistry()
        provider = _MockCapabilityProvider()
        registry.register(provider)

        assert registry.has_provider("mock_capability")

        # Injector manages injection
        injector = CapabilityInjector()
        assert injector.get_capability("file_operations") is not None

        # Each has its own responsibility
        assert hasattr(registry, "register")
        assert hasattr(registry, "get_provider")
        assert hasattr(injector, "get_capability")
        assert hasattr(injector, "register_provider")

    def test_ocp_open_for_extension(self):
        """Test OCP: Can extend without modifying existing code."""
        reset_global_registry()
        CapabilityInjector.reset_global()

        injector = CapabilityInjector()

        # Define new capability type (no modifications to injector)
        class NewCapabilityProvider(BaseCapabilityProvider):
            def __init__(self, container=None):
                super().__init__("new_capability", container)

            def _create_instance(self):
                return _MockCapability()

        # Register new capability
        injector.register_provider(NewCapabilityProvider())

        # Use new capability without modifying injector code
        capability = injector.get_capability("new_capability")
        assert isinstance(capability, _MockCapability)

    def test_lsp_substitutable_providers(self):
        """Test LSP: All providers are substitutable via protocol."""
        reset_global_registry()
        CapabilityInjector.reset_global()

        # Create injector without auto-registration to avoid conflicts
        injector = CapabilityInjector(auto_register_builtins=False)

        # All providers implement ICapabilityProvider
        providers = [
            _MockCapabilityProvider(),
            _MockConfigurableProvider(),
        ]

        for provider in providers:
            assert isinstance(provider, ICapabilityProvider)
            injector.register_provider(provider)
            capability = injector.get_capability(provider.name)
            assert capability is not None

        # Also verify file_operations provider implements protocol
        file_ops_provider = _FileOperationsProvider()
        assert isinstance(file_ops_provider, ICapabilityProvider)

    def test_isp_narrow_protocols(self):
        """Test ISP: Protocols are narrow and focused."""
        # ICapabilityProvider has only 3 methods/properties
        provider = _MockCapabilityProvider()
        protocol_methods = [m for m in dir(ICapabilityProvider) if not m.startswith("_")]

        # Should be a small, focused set
        assert len(protocol_methods) <= 5

        # Check protocol has the essential methods
        assert "get_instance" in protocol_methods
        assert "reset" in protocol_methods

        # Provider implements the protocol
        assert hasattr(provider, "name")
        assert hasattr(provider, "get_instance")
        assert hasattr(provider, "reset")

    def test_dip_depend_on_abstractions(self):
        """Test DIP: Code depends on protocol abstractions."""
        reset_global_registry()

        # Injector works with ICapabilityProvider protocol
        injector = CapabilityInjector()

        # Can register any provider implementing the protocol
        provider = _MockCapabilityProvider()
        injector.register_provider(provider)

        # Retrieval works with protocol abstraction
        capability = injector.get_capability("mock_capability")
        assert capability is not None

        # No direct dependency on concrete provider class


# =============================================================================
# Integration Tests
# =============================================================================


class TestCapabilityInjectionIntegration:
    """Integration tests for capability injection system."""

    def test_end_to_end_capability_lifecycle(self):
        """Test complete capability lifecycle from registration to usage."""
        reset_global_registry()
        CapabilityInjector.reset_global()

        # 1. Create injector
        injector = CapabilityInjector()

        # 2. Register custom capability
        provider = _MockCapabilityProvider()
        injector.register_provider(provider)

        # 3. Check availability
        assert injector.has_capability("mock_capability")

        # 4. Get capability
        capability1 = injector.get_capability("mock_capability")
        assert isinstance(capability1, _MockCapability)

        # 5. Get again (should be same instance)
        capability2 = injector.get_capability("mock_capability")
        assert capability1 is capability2

        # 6. Reset
        injector.reset()

        # 7. Get after reset (should be new instance)
        capability3 = injector.get_capability("mock_capability")
        assert capability1 is not capability3

    def test_multiple_injectors_share_registry(self):
        """Test that multiple injectors share the global registry."""
        reset_global_registry()
        CapabilityInjector.reset_global()

        injector1 = CapabilityInjector()
        injector2 = CapabilityInjector()

        # Register with one injector
        provider = _MockCapabilityProvider()
        injector1.register_provider(provider)

        # Both should see the capability
        assert injector1.has_capability("mock_capability")
        assert injector2.has_capability("mock_capability")

        # Both can get the capability
        capability1 = injector1.get_capability("mock_capability")
        capability2 = injector2.get_capability("mock_capability")

        # Same singleton instance
        assert capability1 is capability2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
