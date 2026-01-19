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

"""Tests for BaseVerticalModeProvider framework class."""

import pytest

from victor.core.mode_config import ModeConfigRegistry, ModeDefinition
from victor.framework.modes import BaseVerticalModeProvider


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the ModeConfigRegistry before each test."""
    ModeConfigRegistry.reset_instance()
    yield
    ModeConfigRegistry.reset_instance()


class TestBaseVerticalModeProvider:
    """Test BaseVerticalModeProvider functionality."""

    def test_auto_register_coding_modes(self):
        """Test that coding modes are auto-registered from VerticalModeDefaults."""
        provider = BaseVerticalModeProvider(vertical="coding")

        # Verify registration
        registry = ModeConfigRegistry.get_instance()
        assert "coding" in registry.list_verticals()

        # Check that coding-specific modes are available
        architect_mode = provider.get_mode("architect")
        assert architect_mode is not None
        assert architect_mode.name == "architect"
        assert architect_mode.tool_budget == 40

        refactor_mode = provider.get_mode("refactor")
        assert refactor_mode is not None
        assert refactor_mode.tool_budget == 25

    def test_auto_register_devops_modes(self):
        """Test that DevOps modes are auto-registered from VerticalModeDefaults."""
        provider = BaseVerticalModeProvider(vertical="devops")

        # Verify registration
        registry = ModeConfigRegistry.get_instance()
        assert "devops" in registry.list_verticals()

        # Check that DevOps-specific modes are available
        migration_mode = provider.get_mode("migration")
        assert migration_mode is not None
        assert migration_mode.tool_budget == 35

    def test_auto_register_research_modes(self):
        """Test that research modes are auto-registered from VerticalModeDefaults."""
        provider = BaseVerticalModeProvider(vertical="research")

        # Verify registration
        registry = ModeConfigRegistry.get_instance()
        assert "research" in registry.list_verticals()

        # Check that research-specific modes are available
        deep_mode = provider.get_mode("deep")
        assert deep_mode is not None
        assert deep_mode.tool_budget == 25

        academic_mode = provider.get_mode("academic")
        assert academic_mode is not None
        assert academic_mode.tool_budget == 40

    def test_auto_register_dataanalysis_modes(self):
        """Test that dataanalysis modes are auto-registered from VerticalModeDefaults."""
        provider = BaseVerticalModeProvider(vertical="dataanalysis")

        # Verify registration
        registry = ModeConfigRegistry.get_instance()
        assert "dataanalysis" in registry.list_verticals()

        # Check that dataanalysis-specific modes are available
        insights_mode = provider.get_mode("insights")
        assert insights_mode is not None
        assert insights_mode.tool_budget == 30

    def test_idempotent_registration(self):
        """Test that multiple instantiations don't cause duplicate registration."""
        # Create multiple providers for the same vertical
        provider1 = BaseVerticalModeProvider(vertical="coding")
        provider2 = BaseVerticalModeProvider(vertical="coding")

        # Both should work without errors
        assert provider1.get_mode("architect") is not None
        assert provider2.get_mode("architect") is not None

        # Registry should only have one entry
        registry = ModeConfigRegistry.get_instance()
        verticals = registry.list_verticals()
        assert verticals.count("coding") == 1

    def test_list_modes_includes_defaults(self):
        """Test that list_modes includes both default and vertical-specific modes."""
        provider = BaseVerticalModeProvider(vertical="coding")
        modes = provider.list_modes()

        # Check for default modes
        assert "quick" in modes
        assert "standard" in modes
        assert "comprehensive" in modes

        # Check for coding-specific modes
        assert "architect" in modes
        assert "refactor" in modes
        assert "debug" in modes
        assert "test" in modes

    def test_get_mode_for_complexity_coding(self):
        """Test complexity mapping for coding vertical."""
        provider = BaseVerticalModeProvider(vertical="coding")

        assert provider.get_mode_for_complexity("trivial") == "fast"
        assert provider.get_mode_for_complexity("simple") == "fast"
        assert provider.get_mode_for_complexity("moderate") == "default"
        assert provider.get_mode_for_complexity("complex") == "thorough"
        assert provider.get_mode_for_complexity("highly_complex") == "architect"

    def test_get_mode_for_complexity_devops(self):
        """Test complexity mapping for DevOps vertical."""
        provider = BaseVerticalModeProvider(vertical="devops")

        assert provider.get_mode_for_complexity("trivial") == "quick"
        assert provider.get_mode_for_complexity("simple") == "quick"
        assert provider.get_mode_for_complexity("moderate") == "standard"
        assert provider.get_mode_for_complexity("complex") == "comprehensive"
        assert provider.get_mode_for_complexity("highly_complex") == "migration"

    def test_get_mode_for_complexity_research(self):
        """Test complexity mapping for research vertical."""
        provider = BaseVerticalModeProvider(vertical="research")

        assert provider.get_mode_for_complexity("trivial") == "quick"
        assert provider.get_mode_for_complexity("simple") == "quick"
        assert provider.get_mode_for_complexity("moderate") == "standard"
        assert provider.get_mode_for_complexity("complex") == "deep"
        assert provider.get_mode_for_complexity("highly_complex") == "academic"

    def test_get_mode_for_complexity_dataanalysis(self):
        """Test complexity mapping for dataanalysis vertical."""
        provider = BaseVerticalModeProvider(vertical="dataanalysis")

        assert provider.get_mode_for_complexity("trivial") == "quick"
        assert provider.get_mode_for_complexity("simple") == "quick"
        assert provider.get_mode_for_complexity("moderate") == "standard"
        assert provider.get_mode_for_complexity("complex") == "insights"
        assert provider.get_mode_for_complexity("highly_complex") == "insights"

    def test_get_tool_budget_for_task(self):
        """Test getting tool budget for specific task types."""
        provider = BaseVerticalModeProvider(vertical="coding")

        # Test coding-specific task budgets
        assert provider.get_tool_budget_for_task("code_generation") == 3
        assert provider.get_tool_budget_for_task("refactor") == 15
        assert provider.get_tool_budget_for_task("debug") == 12
        assert provider.get_tool_budget_for_task("test") == 10

    def test_get_tool_budget_for_task_devops(self):
        """Test getting tool budget for DevOps task types."""
        provider = BaseVerticalModeProvider(vertical="devops")

        # Test DevOps-specific task budgets
        assert provider.get_tool_budget_for_task("deploy") == 8
        assert provider.get_tool_budget_for_task("migration") == 25

    def test_get_default_mode(self):
        """Test that default mode is set correctly for each vertical."""
        coding_provider = BaseVerticalModeProvider(vertical="coding")
        assert coding_provider.get_default_mode() == "default"

        devops_provider = BaseVerticalModeProvider(vertical="devops")
        assert devops_provider.get_default_mode() == "standard"

        benchmark_provider = BaseVerticalModeProvider(vertical="benchmark")
        assert benchmark_provider.get_default_mode() == "fast"

    def test_get_default_tool_budget(self):
        """Test that default tool budget is set correctly."""
        coding_provider = BaseVerticalModeProvider(vertical="coding")
        assert coding_provider.get_default_tool_budget() == 10

        devops_provider = BaseVerticalModeProvider(vertical="devops")
        assert devops_provider.get_default_tool_budget() == 15

    def test_get_mode_configs(self):
        """Test getting all mode configurations."""
        provider = BaseVerticalModeProvider(vertical="coding")
        configs = provider.get_mode_configs()

        # Should have both default and vertical-specific modes
        assert "quick" in configs
        assert "standard" in configs
        assert "architect" in configs
        assert "refactor" in configs

        # Check that configs are ModeConfig objects
        quick_config = configs["quick"]
        assert quick_config.tool_budget == 5
        assert quick_config.max_iterations == 10

    def test_unknown_vertical(self):
        """Test behavior with unknown vertical name."""
        # Should not raise error, but log warning
        provider = BaseVerticalModeProvider(vertical="unknown_vertical")

        # Should still work with default modes
        modes = provider.list_modes()
        assert "quick" in modes
        assert "standard" in modes

    def test_auto_register_disabled(self):
        """Test creating provider without auto-registration."""
        provider = BaseVerticalModeProvider(vertical="coding", auto_register=False)

        # Vertical should not be registered
        registry = ModeConfigRegistry.get_instance()
        assert "coding" not in registry.list_verticals()

    def test_custom_complexity_mapping_override(self):
        """Test overriding complexity mapping in subclass."""

        class CustomProvider(BaseVerticalModeProvider):
            def __init__(self):
                super().__init__(vertical="coding")

            def get_mode_for_complexity(self, complexity: str) -> str:
                # Custom mapping that always returns "quick"
                return "quick"

        provider = CustomProvider()
        assert provider.get_mode_for_complexity("highly_complex") == "quick"
        assert provider.get_mode_for_complexity("complex") == "quick"

    def test_backward_compatibility_with_registry(self):
        """Test that provider works correctly with ModeConfigRegistry."""
        provider = BaseVerticalModeProvider(vertical="coding")

        registry = ModeConfigRegistry.get_instance()

        # Get mode through provider
        provider_mode = provider.get_mode("architect")

        # Get mode through registry
        registry_mode = registry.get_mode("coding", "architect")

        # Should be the same
        assert provider_mode.name == registry_mode.name
        assert provider_mode.tool_budget == registry_mode.tool_budget

    def test_multiple_verticals_independence(self):
        """Test that multiple verticals maintain independent configurations."""
        coding = BaseVerticalModeProvider(vertical="coding")
        devops = BaseVerticalModeProvider(vertical="devops")
        research = BaseVerticalModeProvider(vertical="research")

        # Each should have its specific modes
        assert coding.get_mode("architect") is not None
        assert devops.get_mode("architect") is None  # DevOps doesn't have architect
        assert research.get_mode("academic") is not None

        # Each should have different default modes
        assert coding.get_default_mode() == "default"
        assert devops.get_default_mode() == "standard"
        assert research.get_default_mode() == "standard"


class TestVerticalModeProvidersIntegration:
    """Integration tests for actual vertical mode providers."""

    def test_coding_mode_config_provider(self):
        """Test CodingModeConfigProvider uses BaseVerticalModeProvider."""
        from victor.coding.mode_config import CodingModeConfigProvider

        provider = CodingModeConfigProvider()

        # Should inherit all functionality
        assert provider.get_mode("architect") is not None
        assert provider.get_mode_for_complexity("highly_complex") == "architect"
        assert provider.get_tool_budget_for_task("refactor") == 15

    def test_devops_mode_config_provider(self):
        """Test DevOpsModeConfigProvider uses BaseVerticalModeProvider."""
        from victor.devops.mode_config import DevOpsModeConfigProvider

        provider = DevOpsModeConfigProvider()

        # Should inherit all functionality
        assert provider.get_mode("migration") is not None
        assert provider.get_mode_for_complexity("highly_complex") == "migration"
        assert provider.get_tool_budget_for_task("deploy") == 8

    def test_research_mode_config_provider(self):
        """Test ResearchModeConfigProvider uses BaseVerticalModeProvider."""
        from victor.research.mode_config import ResearchModeConfigProvider

        provider = ResearchModeConfigProvider()

        # Should inherit all functionality
        assert provider.get_mode("academic") is not None
        assert provider.get_mode_for_complexity("highly_complex") == "academic"
        assert provider.get_tool_budget_for_task("literature_review") == 40

    def test_dataanalysis_mode_config_provider(self):
        """Test DataAnalysisModeConfigProvider uses BaseVerticalModeProvider."""
        from victor.dataanalysis.mode_config import DataAnalysisModeConfigProvider

        provider = DataAnalysisModeConfigProvider()

        # Should inherit all functionality
        assert provider.get_mode("insights") is not None
        assert provider.get_mode_for_complexity("complex") == "insights"
        assert provider.get_tool_budget_for_task("analyze") == 15

    def test_coding_convenience_functions(self):
        """Test coding convenience functions work correctly."""
        from victor.coding.mode_config import get_mode_config, get_tool_budget

        # Test get_mode_config
        mode = get_mode_config("architect")
        assert mode is not None
        assert mode.tool_budget == 40

        # Test get_tool_budget with mode
        assert get_tool_budget(mode_name="architect") == 40

        # Test get_tool_budget with task type
        assert get_tool_budget(task_type="refactor") == 15
