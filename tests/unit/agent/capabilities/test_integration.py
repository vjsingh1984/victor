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

"""Integration tests for capability system."""


from victor.agent.capability_registry import get_capability_registry, get_method_for_capability
from victor.agent.capabilities.base import CapabilitySpec


class TestCapabilityRegistryIntegration:
    """Test integration with capability_registry module."""

    def test_get_capability_registry(self):
        """Test getting global capability registry."""
        registry = get_capability_registry()

        assert registry is not None
        # Should be a singleton
        registry2 = get_capability_registry()
        assert registry is registry2

    def test_get_method_for_capability_integration(self):
        """Test get_method_for_capability function."""
        method = get_method_for_capability("enabled_tools")

        assert method == "set_enabled_tools"

    def test_get_method_for_unknown_capability(self):
        """Test get_method_for_capability with unknown capability."""
        method = get_method_for_capability("unknown_capability")

        # Should fall back to "set_unknown_capability"
        assert method == "set_unknown_capability"

    def test_backward_compatibility_with_legacy_dict(self):
        """Test that legacy CAPABILITY_METHOD_MAPPINGS still works."""
        from victor.agent.capability_registry import CAPABILITY_METHOD_MAPPINGS

        # Legacy dict should still exist for backward compatibility
        assert isinstance(CAPABILITY_METHOD_MAPPINGS, dict)
        assert "enabled_tools" in CAPABILITY_METHOD_MAPPINGS
        assert CAPABILITY_METHOD_MAPPINGS["enabled_tools"] == "set_enabled_tools"

    def test_registry_has_all_legacy_capabilities(self):
        """Test that registry has all capabilities from legacy dict."""
        from victor.agent.capability_registry import CAPABILITY_METHOD_MAPPINGS

        registry = get_capability_registry()

        # All legacy capabilities should be available in registry
        for cap_name in CAPABILITY_METHOD_MAPPINGS.keys():
            if cap_name != "enrichment_service":  # Skip attribute access
                assert registry.has_capability(
                    cap_name
                ), f"Registry should have capability '{cap_name}'"

    def test_dynamic_capability_registration_affects_global_registry(self):
        """Test that registering capability affects global registry."""
        registry = get_capability_registry()

        # Register a custom capability
        spec = CapabilitySpec(
            name="integration_test_capability",
            method_name="set_integration_test",
            version="1.0",
        )
        registry.register_capability(spec)

        # Should be available via get_method_for_capability
        method = get_method_for_capability("integration_test_capability")
        assert method == "set_integration_test"

        # Clean up
        registry.unregister_capability("integration_test_capability")


class TestCapabilityMethodMappings:
    """Test backward compatibility with legacy method mappings."""

    def test_all_capability_mappings_match_registry(self):
        """Test that all legacy mappings match registry values."""
        from victor.agent.capability_registry import CAPABILITY_METHOD_MAPPINGS

        registry = get_capability_registry()

        for cap_name, expected_method in CAPABILITY_METHOD_MAPPINGS.items():
            if cap_name == "enrichment_service":
                # Skip attribute access (not a method)
                continue
            actual_method = registry.get_method_for_capability(cap_name)
            assert (
                actual_method == expected_method
            ), f"Method mismatch for '{cap_name}': expected '{expected_method}', got '{actual_method}'"


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_registry_survives_invalid_entry_point(self):
        """Test that registry handles invalid entry points gracefully."""
        # This test verifies the registry doesn't crash if entry points fail
        # In real scenarios, we can't test actual entry points without mocking,
        # but we can verify the registry initializes successfully
        registry = get_capability_registry()

        # Should have loaded built-ins even if entry points failed
        assert len(registry.list_capabilities()) > 0

    def test_method_resolution_with_mixed_sources(self):
        """Test method resolution with mixed capability sources."""
        registry = get_capability_registry()

        # Built-in capability
        method1 = registry.get_method_for_capability("enabled_tools")
        assert method1 == "set_enabled_tools"

        # Runtime-registered capability
        spec = CapabilitySpec(
            name="runtime_capability",
            method_name="set_runtime",
        )
        registry.register_capability(spec)

        method2 = registry.get_method_for_capability("runtime_capability")
        assert method2 == "set_runtime"

        # Unknown capability (fallback)
        method3 = registry.get_method_for_capability("unknown_capability")
        assert method3 == "set_unknown_capability"

        # Clean up
        registry.unregister_capability("runtime_capability")
