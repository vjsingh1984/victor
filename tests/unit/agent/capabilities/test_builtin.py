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

"""Tests for built-in capability implementations."""


from victor.agent.capabilities.builtin import (
    EnabledToolsCapability,
    ToolDependenciesCapability,
    ToolSequencesCapability,
    TieredToolConfigCapability,
    VerticalMiddlewareCapability,
    VerticalSafetyPatternsCapability,
    VerticalContextCapability,
    RlHooksCapability,
    TeamSpecsCapability,
    ModeConfigsCapability,
    DefaultBudgetCapability,
    CustomPromptCapability,
    PromptSectionCapability,
    TaskTypeHintsCapability,
    SafetyPatternsCapability,
    EnrichmentStrategyCapability,
)


class TestBuiltinCapabilities:
    """Test all built-in capability implementations."""

    def test_enabled_tools_capability(self):
        """Test EnabledToolsCapability."""
        spec = EnabledToolsCapability.get_spec()

        assert spec.name == "enabled_tools"
        assert spec.method_name == "set_enabled_tools"
        assert spec.version == "1.0"
        assert spec.description != ""

    def test_tool_dependencies_capability(self):
        """Test ToolDependenciesCapability."""
        spec = ToolDependenciesCapability.get_spec()

        assert spec.name == "tool_dependencies"
        assert spec.method_name == "set_tool_dependencies"
        assert spec.version == "1.0"

    def test_tool_sequences_capability(self):
        """Test ToolSequencesCapability."""
        spec = ToolSequencesCapability.get_spec()

        assert spec.name == "tool_sequences"
        assert spec.method_name == "set_tool_sequences"
        assert spec.version == "1.0"

    def test_tiered_tool_config_capability(self):
        """Test TieredToolConfigCapability."""
        spec = TieredToolConfigCapability.get_spec()

        assert spec.name == "tiered_tool_config"
        assert spec.method_name == "set_tiered_tool_config"
        assert spec.version == "1.0"

    def test_vertical_middleware_capability(self):
        """Test VerticalMiddlewareCapability."""
        spec = VerticalMiddlewareCapability.get_spec()

        assert spec.name == "vertical_middleware"
        assert spec.method_name == "apply_vertical_middleware"
        assert spec.version == "1.0"

    def test_vertical_safety_patterns_capability(self):
        """Test VerticalSafetyPatternsCapability."""
        spec = VerticalSafetyPatternsCapability.get_spec()

        assert spec.name == "vertical_safety_patterns"
        assert spec.method_name == "apply_vertical_safety_patterns"
        assert spec.version == "1.0"

    def test_vertical_context_capability(self):
        """Test VerticalContextCapability."""
        spec = VerticalContextCapability.get_spec()

        assert spec.name == "vertical_context"
        assert spec.method_name == "set_vertical_context"
        assert spec.version == "1.0"

    def test_rl_hooks_capability(self):
        """Test RlHooksCapability."""
        spec = RlHooksCapability.get_spec()

        assert spec.name == "rl_hooks"
        assert spec.method_name == "set_rl_hooks"
        assert spec.version == "1.0"

    def test_team_specs_capability(self):
        """Test TeamSpecsCapability."""
        spec = TeamSpecsCapability.get_spec()

        assert spec.name == "team_specs"
        assert spec.method_name == "set_team_specs"
        assert spec.version == "1.0"

    def test_mode_configs_capability(self):
        """Test ModeConfigsCapability."""
        spec = ModeConfigsCapability.get_spec()

        assert spec.name == "mode_configs"
        assert spec.method_name == "set_mode_configs"
        assert spec.version == "1.0"

    def test_default_budget_capability(self):
        """Test DefaultBudgetCapability."""
        spec = DefaultBudgetCapability.get_spec()

        assert spec.name == "default_budget"
        assert spec.method_name == "set_default_budget"
        assert spec.version == "1.0"

    def test_custom_prompt_capability(self):
        """Test CustomPromptCapability."""
        spec = CustomPromptCapability.get_spec()

        assert spec.name == "custom_prompt"
        assert spec.method_name == "set_custom_prompt"
        assert spec.version == "1.0"

    def test_prompt_section_capability(self):
        """Test PromptSectionCapability."""
        spec = PromptSectionCapability.get_spec()

        assert spec.name == "prompt_section"
        assert spec.method_name == "add_prompt_section"
        assert spec.version == "1.0"

    def test_task_type_hints_capability(self):
        """Test TaskTypeHintsCapability."""
        spec = TaskTypeHintsCapability.get_spec()

        assert spec.name == "task_type_hints"
        assert spec.method_name == "set_task_type_hints"
        assert spec.version == "1.0"

    def test_safety_patterns_capability(self):
        """Test SafetyPatternsCapability."""
        spec = SafetyPatternsCapability.get_spec()

        assert spec.name == "safety_patterns"
        assert spec.method_name == "add_safety_patterns"
        assert spec.version == "1.0"

    def test_enrichment_strategy_capability(self):
        """Test EnrichmentStrategyCapability."""
        spec = EnrichmentStrategyCapability.get_spec()

        assert spec.name == "enrichment_strategy"
        assert spec.method_name == "set_enrichment_strategy"
        assert spec.version == "1.0"


class TestBuiltinCapabilityNames:
    """Test that all built-in capabilities have valid names."""

    def test_all_capabilities_have_names(self):
        """Test that all capabilities have non-empty names."""
        capabilities = [
            EnabledToolsCapability,
            ToolDependenciesCapability,
            ToolSequencesCapability,
            TieredToolConfigCapability,
            VerticalMiddlewareCapability,
            VerticalSafetyPatternsCapability,
            VerticalContextCapability,
            RlHooksCapability,
            TeamSpecsCapability,
            ModeConfigsCapability,
            DefaultBudgetCapability,
            CustomPromptCapability,
            PromptSectionCapability,
            TaskTypeHintsCapability,
            SafetyPatternsCapability,
            EnrichmentStrategyCapability,
        ]

        for cap_class in capabilities:
            spec = cap_class.get_spec()
            assert spec.name, f"{cap_class.__name__} should have a name"
            assert spec.method_name, f"{cap_class.__name__} should have a method_name"

    def test_all_capabilities_have_valid_versions(self):
        """Test that all capabilities have valid version formats."""
        capabilities = [
            EnabledToolsCapability,
            ToolDependenciesCapability,
            ToolSequencesCapability,
            TieredToolConfigCapability,
            VerticalMiddlewareCapability,
            VerticalSafetyPatternsCapability,
            VerticalContextCapability,
            RlHooksCapability,
            TeamSpecsCapability,
            ModeConfigsCapability,
            DefaultBudgetCapability,
            CustomPromptCapability,
            PromptSectionCapability,
            TaskTypeHintsCapability,
            SafetyPatternsCapability,
            EnrichmentStrategyCapability,
        ]

        for cap_class in capabilities:
            spec = cap_class.get_spec()
            # Valid version is MAJOR.MINOR
            parts = spec.version.split(".")
            assert len(parts) == 2, f"{cap_class.__name__} version should be MAJOR.MINOR"
            major, minor = int(parts[0]), int(parts[1])
            assert major >= 0, f"{cap_class.__name__} major version should be >= 0"
            assert minor >= 0, f"{cap_class.__name__} minor version should be >= 0"
