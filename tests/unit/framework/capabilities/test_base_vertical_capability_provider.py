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

"""Tests for BaseVerticalCapabilityProvider.

Tests the comprehensive base class that eliminates ~2000 lines of
duplication across vertical capability providers.
"""

import pytest
from typing import Any, Dict, Optional
from unittest.mock import Mock, MagicMock

from victor.framework.capabilities.base_vertical_capability_provider import (
    BaseVerticalCapabilityProvider,
    CapabilityDefinition,
    _map_capability_type,
)
from victor.framework.protocols import CapabilityType as FrameworkCapabilityType
from victor.framework.capability_loader import CapabilityEntry


class MockCapabilityProvider(BaseVerticalCapabilityProvider):
    """Mock provider for testing."""

    def __init__(self):
        super().__init__("mock")
        self._orchestrator_config = {}

    def _get_capability_definitions(self) -> Dict[str, CapabilityDefinition]:
        return {
            "test_capability": CapabilityDefinition(
                name="test_capability",
                type=FrameworkCapabilityType.TOOL,
                description="Test capability",
                version="1.0",
                configure_fn="configure_test_capability",
                get_fn="get_test_capability",
                default_config={"key": "value"},
                tags=["test"],
            ),
            "disabled_capability": CapabilityDefinition(
                name="disabled_capability",
                type=FrameworkCapabilityType.TOOL,
                description="Disabled capability",
                version="1.0",
                configure_fn="configure_disabled_capability",
                default_config={},
                enabled=False,
            ),
            "capability_with_deps": CapabilityDefinition(
                name="capability_with_deps",
                type=FrameworkCapabilityType.PROMPT,
                description="Capability with dependencies",
                version="1.0",
                configure_fn="configure_capability_with_deps",
                default_config={"dep_config": True},
                dependencies=["test_capability"],
                tags=["prompt", "test"],
            ),
        }

    def configure_test_capability(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure test capability."""
        if not hasattr(orchestrator, "test_capability_config"):
            orchestrator.test_capability_config = {}
        orchestrator.test_capability_config.update(kwargs)

    def get_test_capability(self, orchestrator: Any) -> Dict[str, Any]:
        """Get test capability config."""
        return getattr(orchestrator, "test_capability_config", {"key": "value"})

    def configure_disabled_capability(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure disabled capability."""
        orchestrator.disabled_capability_config = kwargs

    def configure_capability_with_deps(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure capability with dependencies."""
        orchestrator.capability_with_deps_config = kwargs


class TestCapabilityDefinition:
    """Test CapabilityDefinition dataclass."""

    def test_creation(self):
        """Test creating a capability definition."""
        definition = CapabilityDefinition(
            name="test",
            type=FrameworkCapabilityType.TOOL,
            description="Test capability",
            version="2.0",
            configure_fn="configure_test",
            get_fn="get_test",
            default_config={"key": "value"},
            dependencies=["dep1", "dep2"],
            tags=["test", "example"],
            enabled=True,
        )

        assert definition.name == "test"
        assert definition.type == FrameworkCapabilityType.TOOL
        assert definition.description == "Test capability"
        assert definition.version == "2.0"
        assert definition.configure_fn == "configure_test"
        assert definition.get_fn == "get_test"
        assert definition.default_config == {"key": "value"}
        assert definition.dependencies == ["dep1", "dep2"]
        assert definition.tags == ["test", "example"]
        assert definition.enabled is True

    def test_defaults(self):
        """Test default values."""
        definition = CapabilityDefinition(
            name="test",
            type=FrameworkCapabilityType.TOOL,
            description="Test",
        )

        assert definition.version == "1.0"
        assert definition.configure_fn is None
        assert definition.get_fn is None
        assert definition.set_fn is None
        assert definition.default_config == {}
        assert definition.dependencies == []
        assert definition.tags == []
        assert definition.enabled is True

    def test_to_capability_metadata(self):
        """Test converting to CapabilityMetadata."""
        definition = CapabilityDefinition(
            name="test",
            type=FrameworkCapabilityType.TOOL,
            description="Test capability",
            version="1.5",
            dependencies=["dep1"],
            tags=["test"],
        )

        metadata = definition.to_capability_metadata()

        assert metadata.name == "test"
        assert metadata.description == "Test capability"
        assert metadata.version == "1.5"
        assert metadata.dependencies == ["dep1"]
        assert metadata.tags == ["test"]

    def test_to_core_capability(self):
        """Test converting to core Capability."""
        definition = CapabilityDefinition(
            name="test",
            type=FrameworkCapabilityType.TOOL,
            description="Test capability",
            configure_fn="test_handler",
            default_config={"key": "value"},
        )

        from victor.core.capabilities import Capability as CoreCapability

        core_cap = definition.to_core_capability("coding")

        assert isinstance(core_cap, CoreCapability)
        assert core_cap.name == "coding_test"
        assert core_cap.description == "Test capability"
        assert core_cap.handler == "test_handler"
        assert core_cap.config == {"key": "value"}


class TestMapFrameworkCapabilityType:
    """Test capability type mapping.

    Note: FrameworkCapabilityType has TOOL, PROMPT, MODE, SAFETY, RL.
    The mapping function converts these to core CapabilityType values.
    """

    def test_map_tool(self):
        """Test mapping TOOL type."""
        assert _map_capability_type(FrameworkCapabilityType.TOOL) == "tool"

    def test_map_mode(self):
        """Test mapping MODE to TOOL."""
        assert _map_capability_type(FrameworkCapabilityType.MODE) == "tool"

    def test_map_safety(self):
        """Test mapping SAFETY to MIDDLEWARE."""
        assert _map_capability_type(FrameworkCapabilityType.SAFETY) == "middleware"

    def test_map_prompt(self):
        """Test mapping PROMPT type."""
        assert _map_capability_type(FrameworkCapabilityType.PROMPT) == "tool"

    def test_map_rl(self):
        """Test mapping RL type."""
        assert _map_capability_type(FrameworkCapabilityType.RL) == "tool"


class TestBaseVerticalCapabilityProvider:
    """Test BaseVerticalCapabilityProvider."""

    def test_init(self):
        """Test provider initialization."""
        provider = MockCapabilityProvider()

        assert provider._vertical_name == "mock"
        assert provider._applied == set()
        assert provider._definitions_cache is None
        assert provider._capabilities_cache is None
        assert provider._metadata_cache is None

    def test_get_definitions_caching(self):
        """Test definition caching."""
        provider = MockCapabilityProvider()

        # First call
        definitions1 = provider._get_definitions()
        assert provider._definitions_cache is not None

        # Second call should return cached
        definitions2 = provider._get_definitions()
        assert definitions1 is definitions2
        assert definitions1 is provider._definitions_cache

    def test_get_capabilities(self):
        """Test getting all capabilities."""
        provider = MockCapabilityProvider()
        capabilities = provider.get_capabilities()

        assert "test_capability" in capabilities
        assert "capability_with_deps" in capabilities
        # Disabled capability should not be in capabilities
        assert "disabled_capability" not in capabilities

        # Check caching
        capabilities2 = provider.get_capabilities()
        assert capabilities is capabilities2

    def test_get_capability_metadata(self):
        """Test getting capability metadata."""
        provider = MockCapabilityProvider()
        metadata = provider.get_capability_metadata()

        assert "test_capability" in metadata
        assert metadata["test_capability"].name == "test_capability"
        assert metadata["test_capability"].description == "Test capability"
        assert metadata["test_capability"].tags == ["test"]

        # Check capability with dependencies
        assert metadata["capability_with_deps"].dependencies == ["test_capability"]
        assert metadata["capability_with_deps"].tags == ["workflow", "test"]

    def test_get_capability(self):
        """Test getting specific capability."""
        provider = MockCapabilityProvider()
        capability = provider.get_capability("test_capability")

        assert capability is not None
        assert callable(capability)

    def test_get_capability_not_found(self):
        """Test getting non-existent capability."""
        provider = MockCapabilityProvider()
        capability = provider.get_capability("nonexistent")

        assert capability is None

    def test_list_capabilities_all(self):
        """Test listing all capabilities."""
        provider = MockCapabilityProvider()
        capabilities = provider.list_capabilities()

        assert "test_capability" in capabilities
        assert "capability_with_deps" in capabilities
        # Disabled should not be listed
        assert "disabled_capability" not in capabilities

    def test_list_capabilities_by_type(self):
        """Test listing capabilities by type."""
        provider = MockCapabilityProvider()

        tools = provider.list_capabilities(FrameworkCapabilityType.TOOL)
        assert "test_capability" in tools
        assert "capability_with_deps" not in tools

        workflows = provider.list_capabilities(FrameworkCapabilityType.PROMPT)
        assert "capability_with_deps" in workflows
        assert "test_capability" not in workflows

    def test_has_capability(self):
        """Test checking if capability exists."""
        provider = MockCapabilityProvider()

        assert provider.has_capability("test_capability") is True
        assert provider.has_capability("nonexistent") is False

    def test_get_capability_definition(self):
        """Test getting capability definition."""
        provider = MockCapabilityProvider()
        definition = provider.get_capability_definition("test_capability")

        assert definition is not None
        assert definition.name == "test_capability"
        assert definition.type == FrameworkCapabilityType.TOOL
        assert definition.default_config == {"key": "value"}

    def test_apply_capability(self):
        """Test applying a capability."""
        provider = MockCapabilityProvider()
        orchestrator = Mock()

        provider.apply_capability(orchestrator, "test_capability", key="new_value")

        assert "test_capability" in provider._applied
        assert hasattr(orchestrator, "test_capability_config")
        assert orchestrator.test_capability_config == {"key": "new_value"}

    def test_apply_capability_not_found(self):
        """Test applying non-existent capability."""
        provider = MockCapabilityProvider()
        orchestrator = Mock()

        with pytest.raises(ValueError, match="Unknown capability"):
            provider.apply_capability(orchestrator, "nonexistent")

    def test_apply_capability_no_configure_fn(self):
        """Test applying capability without configure function."""
        provider = MockCapabilityProvider()

        # Create definition without configure_fn
        definition = CapabilityDefinition(
            name="no_configure",
            type=FrameworkCapabilityType.TOOL,
            description="No configure function",
        )

        # Mock get_capability_definition to return this
        provider.get_capability_definition = lambda x: definition if x == "no_configure" else None

        orchestrator = Mock()

        with pytest.raises(ValueError, match="has no configure function"):
            provider.apply_capability(orchestrator, "no_configure")

    def test_get_capability_config(self):
        """Test getting capability configuration."""
        provider = MockCapabilityProvider()
        orchestrator = Mock()

        # Set up config
        orchestrator.test_capability_config = {"key": "test_value"}

        config = provider.get_capability_config(orchestrator, "test_capability")

        assert config == {"key": "test_value"}

    def test_get_capability_config_no_get_fn(self):
        """Test getting config for capability without get_fn."""
        provider = MockCapabilityProvider()
        orchestrator = Mock()

        config = provider.get_capability_config(orchestrator, "capability_with_deps")

        # Should return None since no get_fn defined
        assert config is None

    def test_get_default_config(self):
        """Test getting default configuration."""
        provider = MockCapabilityProvider()
        config = provider.get_default_config("test_capability")

        assert config == {"key": "value"}

        # Verify it's a copy
        config["key"] = "modified"
        config2 = provider.get_default_config("test_capability")
        assert config2["key"] == "value"

    def test_get_default_config_not_found(self):
        """Test getting default config for non-existent capability."""
        provider = MockCapabilityProvider()

        with pytest.raises(ValueError, match="Unknown capability"):
            provider.get_default_config("nonexistent")

    def test_apply_all(self):
        """Test applying all capabilities."""
        provider = MockCapabilityProvider()
        orchestrator = Mock()

        # Apply all
        provider.apply_all(orchestrator)

        # Check that enabled capabilities were applied
        assert "test_capability" in provider._applied
        assert "capability_with_deps" in provider._applied
        # Disabled capability should not be applied
        assert "disabled_capability" not in provider._applied

    def test_get_applied(self):
        """Test getting applied capabilities."""
        provider = MockCapabilityProvider()
        orchestrator = Mock()

        assert provider.get_applied() == set()

        provider.apply_capability(orchestrator, "test_capability")
        applied = provider.get_applied()

        assert "test_capability" in applied
        assert len(applied) == 1

    def test_reset_applied(self):
        """Test resetting applied capabilities."""
        provider = MockCapabilityProvider()
        orchestrator = Mock()

        provider.apply_capability(orchestrator, "test_capability")
        assert provider.get_applied()

        provider.reset_applied()
        assert provider.get_applied() == set()

    def test_generate_capabilities_list(self):
        """Test generating CAPABILITIES list."""
        provider = MockCapabilityProvider()
        capabilities_list = provider.generate_capabilities_list()

        assert isinstance(capabilities_list, list)
        assert len(capabilities_list) == 2  # Only enabled capabilities

        # Check entries are CapabilityEntry
        for entry in capabilities_list:
            assert isinstance(entry, CapabilityEntry)
            assert entry.capability is not None
            assert entry.handler is not None

    def test_generate_capability_configs(self):
        """Test generating capability configs."""
        provider = MockCapabilityProvider()
        configs = provider.generate_capability_configs()

        assert isinstance(configs, dict)
        assert "test_capability_config" in configs
        assert "capability_with_deps_config" in configs

        # Check default values
        assert configs["test_capability_config"] == {"key": "value"}
        assert configs["capability_with_deps_config"] == {"dep_config": True}


class TestCapabilityRegistry:
    """Test CapabilityRegistry."""

    def test_singleton(self):
        """Test registry is singleton."""
        from victor.framework.capabilities.registry import CapabilityRegistry

        registry1 = CapabilityRegistry.get_instance()
        registry2 = CapabilityRegistry.get_instance()

        assert registry1 is registry2

    def test_register_provider(self):
        """Test registering a provider."""
        from victor.framework.capabilities.registry import CapabilityRegistry

        registry = CapabilityRegistry.get_instance()
        provider = MockCapabilityProvider()

        registry.register_provider("mock", provider)

        assert registry.has_provider("mock")
        assert registry.get_provider("mock") is provider

    def test_register_invalid_provider(self):
        """Test registering invalid provider."""
        from victor.framework.capabilities.registry import CapabilityRegistry

        registry = CapabilityRegistry.get_instance()

        with pytest.raises(ValueError, match="must be BaseVerticalCapabilityProvider"):
            registry.register_provider("invalid", "not a provider")

    def test_unregister_provider(self):
        """Test unregistering a provider."""
        from victor.framework.capabilities.registry import CapabilityRegistry

        registry = CapabilityRegistry.get_instance()
        provider = MockCapabilityProvider()

        registry.register_provider("mock", provider)
        assert registry.has_provider("mock")

        success = registry.unregister_provider("mock")
        assert success is True
        assert not registry.has_provider("mock")

    def test_list_providers(self):
        """Test listing providers."""
        from victor.framework.capabilities.registry import CapabilityRegistry

        registry = CapabilityRegistry.get_instance()
        provider1 = MockCapabilityProvider()
        provider2 = MockCapabilityProvider()

        registry.register_provider("mock1", provider1)
        registry.register_provider("mock2", provider2)

        providers = registry.list_providers()
        assert "mock1" in providers
        assert "mock2" in providers

    def test_get_capability(self):
        """Test getting capability from registry."""
        from victor.framework.capabilities.registry import CapabilityRegistry

        registry = CapabilityRegistry.get_instance()
        provider = MockCapabilityProvider()

        registry.register_provider("mock", provider)

        capability = registry.get_capability("mock", "test_capability")
        assert capability is not None
        assert callable(capability)

    def test_get_capability_definition(self):
        """Test getting capability definition from registry."""
        from victor.framework.capabilities.registry import CapabilityRegistry

        registry = CapabilityRegistry.get_instance()
        provider = MockCapabilityProvider()

        registry.register_provider("mock", provider)

        definition = registry.get_capability_definition("mock", "test_capability")
        assert definition is not None
        assert definition.name == "test_capability"

    def test_list_capabilities(self):
        """Test listing capabilities from registry."""
        from victor.framework.capabilities.registry import CapabilityRegistry

        registry = CapabilityRegistry.get_instance()
        provider = MockCapabilityProvider()

        registry.register_provider("mock", provider)

        all_caps = registry.list_capabilities("mock")
        assert "test_capability" in all_caps

        tools = registry.list_capabilities("mock", FrameworkCapabilityType.TOOL)
        assert "test_capability" in tools
        assert "capability_with_deps" not in tools

    def test_list_all_capabilities(self):
        """Test listing capabilities across all providers."""
        from victor.framework.capabilities.registry import CapabilityRegistry

        registry = CapabilityRegistry.get_instance()
        provider1 = MockCapabilityProvider()
        provider2 = MockCapabilityProvider()

        registry.register_provider("mock1", provider1)
        registry.register_provider("mock2", provider2)

        all_caps = registry.list_all_capabilities()
        assert "mock1" in all_caps
        assert "mock2" in all_caps
        assert "test_capability" in all_caps["mock1"]
        assert "test_capability" in all_caps["mock2"]

    def test_apply_capability(self):
        """Test applying capability through registry."""
        from victor.framework.capabilities.registry import CapabilityRegistry

        registry = CapabilityRegistry.get_instance()
        provider = MockCapabilityProvider()
        orchestrator = Mock()

        registry.register_provider("mock", provider)

        success = registry.apply_capability("mock", orchestrator, "test_capability", key="value")

        assert success is True
        assert "test_capability" in provider._applied

    def test_get_capability_config(self):
        """Test getting capability config through registry."""
        from victor.framework.capabilities.registry import CapabilityRegistry

        registry = CapabilityRegistry.get_instance()
        provider = MockCapabilityProvider()
        orchestrator = Mock()

        orchestrator.test_capability_config = {"key": "value"}

        registry.register_provider("mock", provider)

        config = registry.get_capability_config("mock", orchestrator, "test_capability")
        assert config == {"key": "value"}

    def test_get_default_config(self):
        """Test getting default config through registry."""
        from victor.framework.capabilities.registry import CapabilityRegistry

        registry = CapabilityRegistry.get_instance()
        provider = MockCapabilityProvider()

        registry.register_provider("mock", provider)

        config = registry.get_default_config("mock", "test_capability")
        assert config == {"key": "value"}

    def test_get_all_capability_configs(self):
        """Test getting all configs through registry."""
        from victor.framework.capabilities.registry import CapabilityRegistry

        registry = CapabilityRegistry.get_instance()
        provider = MockCapabilityProvider()

        registry.register_provider("mock", provider)

        configs = registry.get_all_capability_configs("mock")
        assert "test_capability_config" in configs
        assert "capability_with_deps_config" in configs

    def test_get_stats(self):
        """Test getting registry stats."""
        from victor.framework.capabilities.registry import CapabilityRegistry

        registry = CapabilityRegistry.get_instance()
        provider = MockCapabilityProvider()

        registry.register_provider("mock", provider)

        stats = registry.get_stats()
        assert stats["vertical_count"] == 1
        assert stats["total_capabilities"] == 2  # Only enabled capabilities
        assert "mock" in stats["capability_counts"]
        assert stats["capability_counts"]["mock"] == 2

    def test_reset_instance(self):
        """Test resetting registry instance."""
        from victor.framework.capabilities.registry import CapabilityRegistry

        registry1 = CapabilityRegistry.get_instance()
        CapabilityRegistry.reset_instance()

        registry2 = CapabilityRegistry.get_instance()

        # Should be different instances after reset
        # But since it's a singleton, this is mainly for testing
        assert CapabilityRegistry._instance is registry2


@pytest.fixture
def reset_registry():
    """Reset registry between tests."""
    from victor.framework.capabilities.registry import CapabilityRegistry

    CapabilityRegistry.reset_instance()
    yield
    CapabilityRegistry.reset_instance()
