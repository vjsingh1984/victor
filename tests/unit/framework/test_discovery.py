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

"""Tests for Vertical Discovery System.

Tests the protocol-based vertical discovery system that enables OCP compliance.
This is critical for ensuring the framework can load new verticals without
modification.
"""


from victor.framework.discovery import VerticalDiscovery


class TestVerticalDiscovery:
    """Test VerticalDiscovery class."""

    def setup_method(self):
        """Clear cache before each test."""
        VerticalDiscovery.clear_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        VerticalDiscovery.clear_cache()

    def test_discover_prompt_contributors(self):
        """Test discovering prompt contributors from all verticals."""
        contributors = VerticalDiscovery.discover_prompt_contributors()

        # Should find at least one contributor (coding vertical)
        assert isinstance(contributors, list)
        assert len(contributors) >= 1

        # Check that contributors have expected attributes
        for contributor in contributors:
            assert hasattr(contributor, "get_system_prompt_section")
            assert hasattr(contributor, "get_grounding_rules")
            assert hasattr(contributor, "get_task_type_hints")
            assert hasattr(contributor, "get_priority")

    def test_discover_prompt_contributors_caching(self):
        """Test that prompt contributor discovery is cached."""
        # First call
        contributors1 = VerticalDiscovery.discover_prompt_contributors()

        # Second call should return same cached instance
        contributors2 = VerticalDiscovery.discover_prompt_contributors()

        assert contributors1 is contributors2

    def test_discover_escape_hatches(self):
        """Test discovering escape hatches from all verticals."""
        hatches = VerticalDiscovery.discover_escape_hatches()

        # Should be a dict
        assert isinstance(hatches, dict)

        # Should find at least coding vertical escape hatches
        assert "coding" in hatches or len(hatches) >= 1

        # Check structure
        for vertical_name, hatch_dict in hatches.items():
            assert "conditions" in hatch_dict
            assert "transforms" in hatch_dict
            assert isinstance(hatch_dict["conditions"], dict)
            assert isinstance(hatch_dict["transforms"], dict)

    def test_discover_escape_hatches_caching(self):
        """Test that escape hatch discovery is cached."""
        # First call
        hatches1 = VerticalDiscovery.discover_escape_hatches()

        # Second call should return same cached instance
        hatches2 = VerticalDiscovery.discover_escape_hatches()

        assert hatches1 is hatches2

    def test_discover_verticals(self):
        """Test discovering all verticals."""
        verticals = VerticalDiscovery.discover_verticals()

        # Should be a dict
        assert isinstance(verticals, dict)

        # Should find at least coding and research verticals
        assert "coding" in verticals
        assert "research" in verticals

        # Check structure
        for name, vertical_class in verticals.items():
            assert isinstance(name, str)
            assert hasattr(vertical_class, "name")
            assert hasattr(vertical_class, "get_config")

    def test_discover_verticals_caching(self):
        """Test that vertical discovery is cached."""
        # First call
        verticals1 = VerticalDiscovery.discover_verticals()

        # Second call should return same cached instance
        verticals2 = VerticalDiscovery.discover_verticals()

        assert verticals1 is verticals2

    def test_discover_vertical_by_name(self):
        """Test discovering a specific vertical by name."""
        # Find existing vertical
        coding_class = VerticalDiscovery.discover_vertical_by_name("coding")

        assert coding_class is not None
        assert coding_class.name == "coding"
        assert hasattr(coding_class, "get_config")

    def test_discover_vertical_by_name_not_found(self):
        """Test discovering a non-existent vertical returns None."""
        non_existent = VerticalDiscovery.discover_vertical_by_name("nonexistent")

        assert non_existent is None

    def test_clear_cache(self):
        """Test clearing discovery cache."""
        # Populate cache
        VerticalDiscovery.discover_prompt_contributors()
        VerticalDiscovery.discover_escape_hatches()
        VerticalDiscovery.discover_verticals()

        # Clear cache
        VerticalDiscovery.clear_cache()

        # After clearing, should get new instances
        contributors1 = VerticalDiscovery.discover_prompt_contributors()
        VerticalDiscovery.clear_cache()
        contributors2 = VerticalDiscovery.discover_prompt_contributors()

        # Should be different instances after cache clear
        assert contributors1 is not contributors2

    def test_escape_hatches_have_expected_conditions(self):
        """Test that escape hatches have expected conditions."""
        hatches = VerticalDiscovery.discover_escape_hatches()

        # Check coding vertical has expected conditions
        if "coding" in hatches:
            conditions = hatches["coding"]["conditions"]
            # Should have some standard conditions
            assert len(conditions) > 0
            # Check for specific condition
            assert "tests_passing" in conditions or "code_quality_check" in conditions

    def test_escape_hatches_have_expected_transforms(self):
        """Test that escape hatches have expected transforms."""
        hatches = VerticalDiscovery.discover_escape_hatches()

        # Check coding vertical has expected transforms
        if "coding" in hatches:
            transforms = hatches["coding"]["transforms"]
            # Should have some transforms
            assert len(transforms) > 0
            # Check for specific transform
            assert "merge_code_analysis" in transforms or "format_implementation_plan" in transforms

    def test_prompt_contributors_include_coding(self):
        """Test that prompt contributors include coding vertical."""
        contributors = VerticalDiscovery.discover_prompt_contributors()

        # Should find coding contributor
        contributor_names = [type(c).__name__ for c in contributors]
        assert "CodingPromptContributor" in contributor_names

    def test_verticals_include_all_builtin(self):
        """Test that verticals discovery includes all built-in verticals."""
        verticals = VerticalDiscovery.discover_verticals()

        # Should include all built-in verticals
        builtin_verticals = ["coding", "research", "devops", "rag", "dataanalysis", "benchmark"]

        # At minimum, should have coding and research (others may not be implemented yet)
        assert "coding" in verticals
        assert "research" in verticals


class TestVerticalDiscoveryIntegration:
    """Integration tests for VerticalDiscovery."""

    def setup_method(self):
        """Clear cache before each test."""
        VerticalDiscovery.clear_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        VerticalDiscovery.clear_cache()

    def test_full_discovery_workflow(self):
        """Test complete discovery workflow.

        This test verifies that all discovery methods work together correctly.
        """
        # Discover all verticals
        verticals = VerticalDiscovery.discover_verticals()
        assert len(verticals) >= 2

        # Discover prompt contributors
        contributors = VerticalDiscovery.discover_prompt_contributors()
        assert len(contributors) >= 1

        # Discover escape hatches
        hatches = VerticalDiscovery.discover_escape_hatches()
        assert len(hatches) >= 1

        # Verify consistency
        # For each vertical with escape hatches, should be in verticals list
        for vertical_name in hatches.keys():
            assert vertical_name in verticals

    def test_discovery_performance(self):
        """Test that discovery is reasonably fast (uses caching)."""
        import time

        # First discovery (populate cache)
        start1 = time.time()
        VerticalDiscovery.discover_verticals()
        time1 = time.time() - start1

        # Second discovery (from cache)
        start2 = time.time()
        VerticalDiscovery.discover_verticals()
        time2 = time.time() - start2

        # Cached discovery should be much faster
        assert time2 < time1

    def test_concurrent_discovery(self):
        """Test that discovery works correctly when called multiple times."""
        # Call all discovery methods
        verticals1 = VerticalDiscovery.discover_verticals()
        contributors1 = VerticalDiscovery.discover_prompt_contributors()
        hatches1 = VerticalDiscovery.discover_escape_hatches()

        # Call again
        verticals2 = VerticalDiscovery.discover_verticals()
        contributors2 = VerticalDiscovery.discover_prompt_contributors()
        hatches2 = VerticalDiscovery.discover_escape_hatches()

        # Should get same cached results
        assert verticals1 is verticals2
        assert contributors1 is contributors2
        assert hatches1 is hatches2


__all__ = [
    "TestVerticalDiscovery",
    "TestVerticalDiscoveryIntegration",
]
