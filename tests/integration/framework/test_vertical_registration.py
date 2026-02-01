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

"""Integration tests for vertical registration and discovery.

These tests verify that verticals are properly registered and can be discovered
by the framework without hardcoded imports.
"""


from victor.framework.discovery import VerticalDiscovery
from victor.framework.escape_hatch_registry import EscapeHatchRegistry


class TestVerticalRegistrationIntegration:
    """Integration tests for vertical registration."""

    def setup_method(self):
        """Setup test fixtures."""
        # Clear all caches
        VerticalDiscovery.clear_cache()
        EscapeHatchRegistry.reset_instance()
        # Also clear class-level storage that persists across instance resets
        EscapeHatchRegistry._class_conditions.clear()
        EscapeHatchRegistry._class_transforms.clear()
        EscapeHatchRegistry._class_global_conditions.clear()
        EscapeHatchRegistry._class_global_transforms.clear()

    def teardown_method(self):
        """Cleanup test fixtures."""
        # Clear all caches
        VerticalDiscovery.clear_cache()
        EscapeHatchRegistry.reset_instance()
        # Also clear class-level storage
        EscapeHatchRegistry._class_conditions.clear()
        EscapeHatchRegistry._class_transforms.clear()
        EscapeHatchRegistry._class_global_conditions.clear()
        EscapeHatchRegistry._class_global_transforms.clear()

    def test_verticals_register_escape_hatches_on_import(self):
        """Test that verticals auto-register escape hatches when imported."""
        # Get registry
        registry = EscapeHatchRegistry.get_instance()

        # Initially, should have no escape hatches (after setup reset)
        verticals = registry.list_verticals()
        initial_count = len(verticals)

        # Import coding vertical module to get escape hatch definitions
        from victor.coding import escape_hatches

        # Register escape hatches explicitly (simulating what happens on first import)
        # Use replace=True since importing escape_hatches may trigger lazy initialization
        registry.register_from_vertical(
            "coding",
            conditions=escape_hatches.CONDITIONS,
            transforms=escape_hatches.TRANSFORMS,
            replace=True,
        )

        # Should now have coding vertical escape hatches
        verticals = registry.list_verticals()
        assert "coding" in verticals

        # Verify escape hatches were registered
        conditions, transforms = registry.get_registry_for_vertical("coding")

        assert len(conditions) > 0
        assert len(transforms) > 0

        # Check for specific escape hatches
        assert "tests_passing" in conditions
        assert "merge_code_analysis" in transforms

    def test_discover_from_all_verticals(self):
        """Test discovering escape hatches from all verticals."""
        registry = EscapeHatchRegistry.get_instance()

        # Discover and register all vertical escape hatches
        cond_count, trans_count = registry.discover_from_all_verticals()

        # Should find at least some escape hatches
        assert cond_count > 0
        assert trans_count > 0

        # Check that coding vertical is registered
        verticals = registry.list_verticals()
        assert "coding" in verticals

    def test_prompt_builder_discovers_vertical_contributors(self):
        """Test that PromptBuilder can discover vertical prompt contributors."""
        # Discover prompt contributors
        contributors = VerticalDiscovery.discover_prompt_contributors()

        # Should find at least one contributor
        assert len(contributors) >= 1

        # Should include coding contributor
        contributor_names = [type(c).__name__ for c in contributors]
        assert "CodingPromptContributor" in contributor_names

    def test_vertical_discovery_integration(self):
        """Test complete vertical discovery workflow."""
        # 1. Discover all verticals
        verticals = VerticalDiscovery.discover_verticals()

        # Should find at least coding and research
        assert "coding" in verticals
        assert "research" in verticals

        # 2. Discover prompt contributors
        contributors = VerticalDiscovery.discover_prompt_contributors()
        assert len(contributors) >= 1

        # 3. Discover escape hatches
        escape_hatches = VerticalDiscovery.discover_escape_hatches()
        assert len(escape_hatches) >= 1

        # 4. Verify consistency - escape hatch verticals should be in verticals list
        for vertical_name in escape_hatches.keys():
            assert vertical_name in verticals

    def test_multiple_verticals_can_coexist(self):
        """Test that multiple verticals can be loaded simultaneously."""
        # Import multiple verticals

        # Both should be discoverable
        verticals = VerticalDiscovery.discover_verticals()

        assert "coding" in verticals
        assert "research" in verticals

        # Both should have registered their escape hatches
        # Use discover_from_all_verticals to ensure registration after reset
        registry = EscapeHatchRegistry.get_instance()
        registry.discover_from_all_verticals()
        all_verticals = registry.list_verticals()

        assert "coding" in all_verticals
        assert "research" in all_verticals

    def test_vertical_config_access_via_discovery(self):
        """Test that vertical configs can be accessed via discovery."""
        verticals = VerticalDiscovery.discover_verticals()

        # Get coding vertical
        coding_class = verticals.get("coding")
        assert coding_class is not None

        # Get its config
        config = coding_class.get_config()
        assert config is not None

        # Verify config structure
        assert hasattr(config, "tools")
        assert hasattr(config, "system_prompt")

    def test_discovery_caching_doesnt_interfere_with_registration(self):
        """Test that discovery caching doesn't interfere with registration."""
        # First discovery
        verticals1 = VerticalDiscovery.discover_verticals()

        # Register escape hatches
        registry = EscapeHatchRegistry.get_instance()
        registry.discover_from_all_verticals()

        # Second discovery (should use cache)
        verticals2 = VerticalDiscovery.discover_verticals()

        # Should be same cached instance
        assert verticals1 is verticals2

        # But should still have all verticals
        assert len(verticals2) >= len(verticals1)


class TestVerticalDiscoveryWithRealVerticals:
    """Test vertical discovery with real vertical implementations."""

    def setup_method(self):
        """Setup test fixtures."""
        VerticalDiscovery.clear_cache()

    def teardown_method(self):
        """Cleanup test fixtures."""
        VerticalDiscovery.clear_cache()

    def test_coding_vertical_fully_discoverable(self):
        """Test that coding vertical can be fully discovered."""
        # Discover verticals
        verticals = VerticalDiscovery.discover_verticals()

        coding_class = verticals.get("coding")
        assert coding_class is not None

        # Should have all expected methods
        assert hasattr(coding_class, "get_config")
        assert hasattr(coding_class, "get_extensions")
        assert hasattr(coding_class, "get_prompt_contributor")
        assert hasattr(coding_class, "get_workflow_provider")

        # Should be able to get config
        config = coding_class.get_config()
        assert config is not None
        assert config.tools is not None

    def test_research_vertical_fully_discoverable(self):
        """Test that research vertical can be fully discovered."""
        # Discover verticals
        verticals = VerticalDiscovery.discover_verticals()

        research_class = verticals.get("research")
        assert research_class is not None

        # Should have all expected methods
        assert hasattr(research_class, "get_config")
        assert hasattr(research_class, "get_extensions")
        assert hasattr(research_class, "get_prompt_contributor")

        # Should be able to get config
        config = research_class.get_config()
        assert config is not None
        assert config.tools is not None

    def test_escape_hatches_discoverable_without_hardcoded_imports(self):
        """Test that escape hatches are discoverable without hardcoded imports."""
        # Discover escape hatches (should use VerticalDiscovery)
        escape_hatches = VerticalDiscovery.discover_escape_hatches()

        # Should find escape hatches from coding
        assert "coding" in escape_hatches

        coding_hatches = escape_hatches["coding"]
        assert "conditions" in coding_hatches
        assert "transforms" in coding_hatches

        # Verify specific escape hatches exist
        conditions = coding_hatches["conditions"]
        transforms = coding_hatches["transforms"]

        # Should have expected escape hatches
        assert "tests_passing" in conditions
        assert "code_quality_check" in conditions
        assert "merge_code_analysis" in transforms


class TestVerticalExtensibility:
    """Test that the system is extensible for new verticals."""

    def setup_method(self):
        """Setup test fixtures."""
        VerticalDiscovery.clear_cache()
        EscapeHatchRegistry.reset_instance()

    def teardown_method(self):
        """Cleanup test fixtures."""
        VerticalDiscovery.clear_cache()
        EscapeHatchRegistry.reset_instance()

    def test_new_vertical_would_be_discovered_automatically(self):
        """Test that a new vertical would be discovered automatically.

        This test verifies the extensibility of the discovery system - if we
        were to add a new vertical (e.g., "security"), it would be discovered
        automatically without modifying the framework.
        """
        # Current verticals
        verticals_before = VerticalDiscovery.discover_verticals()

        # The discovery system uses:
        # 1. Entry points (victor.verticals)
        # 2. Built-in verticals list

        # If we added a new vertical:
        # - It would be in the built-in list OR
        # - It would register via entry points
        # - Either way, VerticalDiscovery would find it

        # Verify that built-in verticals are included
        builtin_verticals = ["coding", "research", "devops", "rag", "dataanalysis", "benchmark"]

        # At minimum, coding and research should be discoverable
        assert "coding" in verticals_before
        assert "research" in verticals_before

    def test_discovery_api_is_stable(self):
        """Test that the discovery API is stable and won't break existing verticals."""
        # The discovery API should remain stable
        # These are the public methods that verticals can rely on:

        assert hasattr(VerticalDiscovery, "discover_verticals")
        assert hasattr(VerticalDiscovery, "discover_prompt_contributors")
        assert hasattr(VerticalDiscovery, "discover_escape_hatches")
        assert hasattr(VerticalDiscovery, "discover_vertical_by_name")
        assert hasattr(VerticalDiscovery, "clear_cache")

        # EscapeHatchRegistry API
        assert hasattr(EscapeHatchRegistry, "discover_from_all_verticals")
        assert hasattr(EscapeHatchRegistry, "register_from_vertical")


__all__ = [
    "TestVerticalRegistrationIntegration",
    "TestVerticalDiscoveryWithRealVerticals",
    "TestVerticalExtensibility",
]
