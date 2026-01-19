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

"""Tests for DynamicCapabilityRegistry."""

import pytest

from victor.agent.capabilities.base import CapabilityBase, CapabilitySpec
from victor.agent.capabilities.registry import DynamicCapabilityRegistry


class TestDynamicCapabilityRegistry:
    """Test DynamicCapabilityRegistry class."""

    def test_registry_initialization(self):
        """Test that registry initializes successfully."""
        registry = DynamicCapabilityRegistry()

        # Should have loaded built-in capabilities
        assert len(registry.list_capabilities()) > 0

    def test_get_method_for_builtin_capability(self):
        """Test getting method name for built-in capability."""
        registry = DynamicCapabilityRegistry()

        method = registry.get_method_for_capability("enabled_tools")
        assert method == "set_enabled_tools"

    def test_get_method_for_unknown_capability_fallback(self):
        """Test fallback for unknown capability."""
        registry = DynamicCapabilityRegistry()

        method = registry.get_method_for_capability("unknown_capability")
        assert method == "set_unknown_capability"

    def test_has_capability_for_builtin(self):
        """Test has_capability for built-in capability."""
        registry = DynamicCapabilityRegistry()

        assert registry.has_capability("enabled_tools")
        assert registry.has_capability("tool_dependencies")

    def test_has_capability_for_unknown(self):
        """Test has_capability for unknown capability."""
        registry = DynamicCapabilityRegistry()

        assert not registry.has_capability("unknown_capability")

    def test_get_capability(self):
        """Test getting a capability spec."""
        registry = DynamicCapabilityRegistry()

        spec = registry.get_capability("enabled_tools")
        assert spec is not None
        assert spec.name == "enabled_tools"
        assert spec.method_name == "set_enabled_tools"

    def test_get_capability_unknown(self):
        """Test getting unknown capability returns None."""
        registry = DynamicCapabilityRegistry()

        spec = registry.get_capability("unknown_capability")
        assert spec is None

    def test_list_capabilities(self):
        """Test listing all capabilities."""
        registry = DynamicCapabilityRegistry()

        caps = registry.list_capabilities()

        assert isinstance(caps, dict)
        assert len(caps) > 0
        assert "enabled_tools" in caps

    def test_register_capability_at_runtime(self):
        """Test registering a capability at runtime."""
        registry = DynamicCapabilityRegistry()

        # Create a custom spec
        spec = CapabilitySpec(
            name="custom_capability",
            method_name="set_custom_capability",
            version="1.0",
            description="Custom capability",
        )

        registry.register_capability(spec)

        # Verify it was registered
        assert registry.has_capability("custom_capability")
        method = registry.get_method_for_capability("custom_capability")
        assert method == "set_custom_capability"

    def test_register_capability_override(self):
        """Test that registering same name overrides existing."""
        registry = DynamicCapabilityRegistry()

        # Register original
        spec1 = CapabilitySpec(
            name="override_test",
            method_name="method_v1",
            version="1.0",
        )
        registry.register_capability(spec1)

        # Register override
        spec2 = CapabilitySpec(
            name="override_test",
            method_name="method_v2",
            version="2.0",
        )
        registry.register_capability(spec2)

        # Should have overridden
        method = registry.get_method_for_capability("override_test")
        assert method == "method_v2"

    def test_unregister_capability(self):
        """Test unregistering a capability."""
        registry = DynamicCapabilityRegistry()

        # Register a capability
        spec = CapabilitySpec(
            name="temp_capability",
            method_name="set_temp",
        )
        registry.register_capability(spec)

        # Verify it exists
        assert registry.has_capability("temp_capability")

        # Unregister it
        result = registry.unregister_capability("temp_capability")

        assert result is True
        assert not registry.has_capability("temp_capability")

    def test_unregister_unknown_capability(self):
        """Test unregistering unknown capability returns False."""
        registry = DynamicCapabilityRegistry()

        result = registry.unregister_capability("unknown_capability")
        assert result is False

    def test_get_registry_stats(self):
        """Test getting registry statistics."""
        registry = DynamicCapabilityRegistry()

        stats = registry.get_registry_stats()

        assert "total_capabilities" in stats
        assert "entry_points_loaded" in stats
        assert isinstance(stats["total_capabilities"], int)
        assert isinstance(stats["entry_points_loaded"], bool)
        assert stats["total_capabilities"] > 0


class TestDynamicCapabilityRegistryWithEntryPoints:
    """Test registry entry point integration."""

    def test_entry_points_loaded_flag(self):
        """Test that entry_points_loaded flag is set correctly."""
        registry = DynamicCapabilityRegistry()

        # We can't test actual entry point loading in unit tests,
        # but we can verify the flag exists and is a bool
        stats = registry.get_registry_stats()
        assert isinstance(stats["entry_points_loaded"], bool)

    def test_builtin_capabilities_always_available(self):
        """Test that built-in capabilities are always available."""
        registry = DynamicCapabilityRegistry()

        # These should always be available from built-ins
        built_in_capabilities = [
            "enabled_tools",
            "tool_dependencies",
            "tool_sequences",
            "tiered_tool_config",
            "vertical_middleware",
            "vertical_safety_patterns",
            "vertical_context",
            "rl_hooks",
            "team_specs",
            "mode_configs",
            "default_budget",
            "custom_prompt",
            "prompt_section",
            "task_type_hints",
            "safety_patterns",
            "enrichment_strategy",
        ]

        for cap_name in built_in_capabilities:
            assert registry.has_capability(
                cap_name
            ), f"Built-in capability '{cap_name}' should be available"


class TestCustomCapabilityClass:
    """Test custom capability class registration."""

    def test_custom_capability_class(self):
        """Test creating and using a custom capability class."""

        class MyCustomCapability(CapabilityBase):
            @classmethod
            def get_spec(cls) -> CapabilitySpec:
                return CapabilitySpec(
                    name="my_custom",
                    method_name="set_my_custom",
                    version="1.5",
                    description="My custom capability",
                )

        registry = DynamicCapabilityRegistry()
        spec = MyCustomCapability.get_spec()

        # Register the spec
        registry.register_capability(spec)

        # Verify it works
        assert registry.has_capability("my_custom")
        method = registry.get_method_for_capability("my_custom")
        assert method == "set_my_custom"

        # Verify spec details
        retrieved_spec = registry.get_capability("my_custom")
        assert retrieved_spec.version == "1.5"
        assert retrieved_spec.description == "My custom capability"
