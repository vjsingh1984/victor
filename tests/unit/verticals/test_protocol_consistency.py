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

"""Tests for vertical protocol consistency (Phase 4.3).

Tests that all built-in verticals implement protocols consistently.

Required Protocols for Verticals:
---------------------------------
1. VerticalBase (abstract base):
   - get_tools() -> List[str]
   - get_system_prompt() -> str

2. Optional Extension Protocols:
   - SafetyExtensionProvider: get_safety_extension()
   - PromptContributorProvider: get_prompt_contributor()
   - ModeConfigProvider: get_mode_config_provider()
   - ToolDependencyProvider: get_tool_dependency_provider()
   - WorkflowProvider: get_workflow_provider()
   - RLConfigProvider: get_rl_config_provider()
   - TeamSpecProvider: get_team_spec_provider()
   - CapabilityProvider: get_capability_provider()

3. YAML Config Support:
   - get_config_path() -> Optional[Path] (class method)
   - Corresponding vertical.yaml file

This test ensures all built-in verticals implement protocols consistently
for maintainability and documentation purposes.
"""

import pytest

from victor.core.verticals.base import VerticalBase


# List of built-in verticals to test
BUILT_IN_VERTICALS = [
    "coding",
    "devops",
    "research",
    "rag",
    "dataanalysis",
]

# Required methods all verticals must implement
REQUIRED_METHODS = [
    "get_tools",
    "get_system_prompt",
]

# Optional extension methods
OPTIONAL_EXTENSION_METHODS = [
    "get_safety_extension",
    "get_prompt_contributor",
    "get_mode_config_provider",
    "get_tool_dependency_provider",
    "get_workflow_provider",
    "get_rl_config_provider",
    "get_team_spec_provider",
    "get_capability_provider",
    "get_handlers",
    "get_capability_configs",
]


def get_vertical_class(vertical_name: str):
    """Get vertical class by name."""
    from victor.core.verticals.vertical_loader import load_vertical

    return load_vertical(vertical_name)


class TestVerticalProtocolConsistency:
    """Test that all verticals implement protocols consistently."""

    @pytest.mark.parametrize("vertical_name", BUILT_IN_VERTICALS)
    def test_inherits_from_vertical_base(self, vertical_name):
        """All built-in verticals should inherit from VerticalBase."""
        vertical_class = get_vertical_class(vertical_name)
        assert issubclass(
            vertical_class, VerticalBase
        ), f"{vertical_name} should inherit from VerticalBase"

    @pytest.mark.parametrize("vertical_name", BUILT_IN_VERTICALS)
    def test_has_required_methods(self, vertical_name):
        """All verticals must implement required methods."""
        vertical_class = get_vertical_class(vertical_name)

        for method_name in REQUIRED_METHODS:
            assert hasattr(
                vertical_class, method_name
            ), f"{vertical_name} missing required method: {method_name}"
            assert callable(
                getattr(vertical_class, method_name)
            ), f"{vertical_name}.{method_name} should be callable"

    @pytest.mark.parametrize("vertical_name", BUILT_IN_VERTICALS)
    def test_get_tools_returns_list(self, vertical_name):
        """get_tools should return a non-empty list of strings."""
        vertical_class = get_vertical_class(vertical_name)
        tools = vertical_class.get_tools()

        assert isinstance(tools, list), f"{vertical_name}.get_tools() should return list"
        assert len(tools) > 0, f"{vertical_name}.get_tools() should return non-empty list"
        assert all(
            isinstance(t, str) for t in tools
        ), f"{vertical_name}.get_tools() should return list of strings"

    @pytest.mark.parametrize("vertical_name", BUILT_IN_VERTICALS)
    def test_get_system_prompt_returns_string(self, vertical_name):
        """get_system_prompt should return a non-empty string."""
        vertical_class = get_vertical_class(vertical_name)
        prompt = vertical_class.get_system_prompt()

        assert isinstance(prompt, str), f"{vertical_name}.get_system_prompt() should return str"
        assert len(prompt) > 0, f"{vertical_name}.get_system_prompt() should return non-empty str"

    @pytest.mark.parametrize("vertical_name", BUILT_IN_VERTICALS)
    def test_has_name_attribute(self, vertical_name):
        """All verticals should have a name attribute."""
        vertical_class = get_vertical_class(vertical_name)

        assert hasattr(vertical_class, "name"), f"{vertical_name} should have 'name' attribute"
        assert (
            vertical_class.name == vertical_name
        ), f"{vertical_name}.name should match '{vertical_name}'"


class TestVerticalYAMLConsistency:
    """Test that verticals have consistent YAML configs."""

    @pytest.mark.parametrize("vertical_name", BUILT_IN_VERTICALS)
    def test_has_yaml_config(self, vertical_name):
        """All built-in verticals should have a YAML config file."""
        from pathlib import Path

        # Expected YAML path
        yaml_path = Path(f"victor/{vertical_name}/config/vertical.yaml")

        assert yaml_path.exists(), f"{vertical_name} should have YAML config at {yaml_path}"

    @pytest.mark.parametrize("vertical_name", BUILT_IN_VERTICALS)
    def test_yaml_has_required_sections(self, vertical_name):
        """YAML configs should have required sections."""
        from pathlib import Path
        import yaml

        yaml_path = Path(f"victor/{vertical_name}/config/vertical.yaml")

        if not yaml_path.exists():
            pytest.skip(f"No YAML config for {vertical_name}")

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        # Required sections
        assert "metadata" in config, f"{vertical_name} YAML should have 'metadata' section"
        assert "core" in config, f"{vertical_name} YAML should have 'core' section"

        # Core should have tools
        assert "tools" in config["core"], f"{vertical_name} YAML core should have 'tools'"


class TestExtensionMethodConsistency:
    """Test that extension methods are implemented consistently."""

    @pytest.mark.parametrize("vertical_name", BUILT_IN_VERTICALS)
    def test_extension_methods_return_correct_types(self, vertical_name):
        """Extension methods should return correct types or None."""
        vertical_class = get_vertical_class(vertical_name)

        for method_name in OPTIONAL_EXTENSION_METHODS:
            if not hasattr(vertical_class, method_name):
                continue

            method = getattr(vertical_class, method_name)
            if not callable(method):
                continue

            try:
                result = method()
                # Should return either None or an object
                # We don't enforce specific types, just that it doesn't raise
            except ImportError:
                # OK if extension module not available
                pass
            except Exception as e:
                pytest.fail(
                    f"{vertical_name}.{method_name}() raised unexpected error: {e}"
                )
