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

"""Integration tests for YAML-based vertical configuration.

Tests the complete flow from YAML file to VerticalConfig, ensuring:
- YAML config is loaded when available
- Falls back to programmatic methods when YAML is missing
- Config caching works correctly
- Backward compatibility is maintained
"""

import pytest
from pathlib import Path
from typing import Dict, Any

from victor.core.verticals.base import VerticalBase, VerticalConfig
from victor.framework.tools import ToolSet


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_yaml_vertical(tmp_path):
    """Create a temporary vertical with YAML configuration."""
    # Create directory structure
    vertical_dir = tmp_path / "test_yaml_vertical"
    config_dir = vertical_dir / "config"
    config_dir.mkdir(parents=True)

    # Create YAML config
    yaml_content = """
metadata:
  name: test_yaml
  description: Test vertical with YAML config
  version: "0.5.0"

core:
  tools:
    list:
      - read
      - write
      - edit
  system_prompt:
    source: inline
    text: "You are a test assistant for YAML configuration."

  stages:
    INITIAL:
      name: INITIAL
      description: Understanding the request
      tools: [read]
      keywords: [what, how]
      next_stages: [EXECUTING]
    EXECUTING:
      name: EXECUTING
      description: Making changes
      tools: [write, edit]
      keywords: [change, modify]
      next_stages: []

provider:
  hints:
    preferred: [anthropic]
  parameters:
    temperature: 0.7

evaluation:
  criteria:
    - "Test criterion 1"
    - "Test criterion 2"

tiered_tools:
  mandatory: [read]
  vertical_core: [write, edit]
"""
    (config_dir / "vertical.yaml").write_text(yaml_content)

    # Create Python vertical class
    import sys

    # Add temp_path to sys.path so we can import the module
    sys.path.insert(0, str(tmp_path))

    # Create minimal Python module
    init_file = vertical_dir / "__init__.py"
    init_content = """
from victor.core.verticals.base import VerticalBase
from typing import List, Dict

class TestYAMLVertical(VerticalBase):
    name = "test_yaml"
    description = "This should be overridden by YAML"
    version = "0.0.1"  # This should be overridden by YAML

    @classmethod
    def get_tools(cls) -> List[str]:
        # These should be overridden by YAML
        return ["legacy_tool"]

    @classmethod
    def get_system_prompt(cls) -> str:
        # This should be overridden by YAML
        return "Legacy prompt"
"""
    init_file.write_text(init_content)

    # Import the class
    from test_yaml_vertical import TestYAMLVertical

    yield TestYAMLVertical

    # Cleanup
    sys.path.remove(str(tmp_path))


@pytest.fixture
def temp_no_yaml_vertical(tmp_path):
    """Create a temporary vertical without YAML configuration."""
    vertical_dir = tmp_path / "test_no_yaml_vertical"
    vertical_dir.mkdir(parents=True)

    # Create Python vertical class
    import sys

    sys.path.insert(0, str(tmp_path))

    init_file = vertical_dir / "__init__.py"
    init_content = """
from victor.core.verticals.base import VerticalBase
from typing import List

class TestNoYAMLVertical(VerticalBase):
    name = "test_no_yaml"
    description = "Test vertical without YAML"
    version = "0.5.0"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read", "write"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Programmatic prompt"
"""
    init_file.write_text(init_content)

    from test_no_yaml_vertical import TestNoYAMLVertical

    yield TestNoYAMLVertical

    # Cleanup
    sys.path.remove(str(tmp_path))


# ============================================================================
# Tests
# ============================================================================


class TestYAMLConfigLoading:
    """Tests for YAML configuration loading."""

    def test_loads_yaml_config_when_available(self, temp_yaml_vertical):
        """Test that YAML config is loaded when vertical.yaml exists."""
        config = temp_yaml_vertical.get_config(use_cache=False)

        # Verify YAML values are loaded
        assert config.metadata["name"] == "test_yaml"
        assert config.metadata["description"] == "Test vertical with YAML config"
        assert config.metadata["version"] == "0.5.0"

        # Verify tools from YAML
        assert "read" in config.tools.tools
        assert "write" in config.tools.tools
        assert "edit" in config.tools.tools
        assert "legacy_tool" not in config.tools.tools

        # Verify system prompt from YAML
        assert "YAML configuration" in config.system_prompt
        assert "Legacy" not in config.system_prompt

    def test_yaml_config_overrides_programmatic_methods(self, temp_yaml_vertical):
        """Test that YAML config takes precedence over programmatic methods."""
        config = temp_yaml_vertical.get_config(use_cache=False)

        # YAML should override the class attributes
        assert config.metadata["description"] == "Test vertical with YAML config"
        assert config.metadata["description"] != temp_yaml_vertical.description

        assert config.metadata["version"] == "0.5.0"
        assert config.metadata["version"] != temp_yaml_vertical.version

    def test_yaml_config_loads_stages(self, temp_yaml_vertical):
        """Test that stages are loaded from YAML."""
        config = temp_yaml_vertical.get_config(use_cache=False)

        assert "INITIAL" in config.stages
        assert "EXECUTING" in config.stages

        initial_stage = config.stages["INITIAL"]
        assert initial_stage.name == "INITIAL"
        assert initial_stage.description == "Understanding the request"
        assert "read" in initial_stage.tools
        assert "what" in initial_stage.keywords
        assert "EXECUTING" in initial_stage.next_stages

    def test_yaml_config_loads_provider_hints(self, temp_yaml_vertical):
        """Test that provider hints are loaded from YAML."""
        config = temp_yaml_vertical.get_config(use_cache=False)

        assert "preferred" in config.provider_hints
        assert "anthropic" in config.provider_hints["preferred"]

    def test_yaml_config_loads_evaluation_criteria(self, temp_yaml_vertical):
        """Test that evaluation criteria are loaded from YAML."""
        config = temp_yaml_vertical.get_config(use_cache=False)

        assert len(config.evaluation_criteria) == 2
        assert "Test criterion 1" in config.evaluation_criteria

    def test_yaml_config_loads_extended_metadata(self, temp_yaml_vertical):
        """Test that tiered_tools are loaded into extended metadata."""
        config = temp_yaml_vertical.get_config(use_cache=False)

        assert "tiered_tools" in config.metadata
        tiered_tools = config.metadata["tiered_tools"]
        assert "mandatory" in tiered_tools
        assert "read" in tiered_tools["mandatory"]


class TestYAMLConfigFallback:
    """Tests for fallback to programmatic methods."""

    def test_falls_back_to_programmatic_without_yaml(self, temp_no_yaml_vertical):
        """Test that programmatic methods are used when YAML is missing."""
        config = temp_no_yaml_vertical.get_config(use_cache=False)

        # Verify programmatic values are used
        assert config.metadata["vertical_name"] == "test_no_yaml"
        assert config.metadata["description"] == temp_no_yaml_vertical.description
        assert config.metadata["vertical_version"] == temp_no_yaml_vertical.version

        # Verify tools from get_tools()
        assert "read" in config.tools.tools
        assert "write" in config.tools.tools

        # Verify system prompt from get_system_prompt()
        assert config.system_prompt == "Programmatic prompt"

    def test_use_yaml_false_skips_yaml(self, temp_yaml_vertical):
        """Test that use_yaml=False skips YAML loading."""
        config = temp_yaml_vertical.get_config(use_cache=False, use_yaml=False)

        # Should use programmatic methods, not YAML
        assert "legacy_tool" in config.tools.tools
        assert "Legacy prompt" in config.system_prompt


class TestYAMLConfigCaching:
    """Tests for config caching with YAML."""

    def test_yaml_config_is_cached(self, temp_yaml_vertical):
        """Test that YAML-loaded config is cached."""
        # Load config twice
        config1 = temp_yaml_vertical.get_config(use_cache=False)
        config2 = temp_yaml_vertical.get_config(use_cache=True)

        # Should return same cached instance
        assert config1 is config2

    def test_clear_cache_reloads_yaml(self, temp_yaml_vertical):
        """Test that clearing cache reloads YAML config."""
        # Load config
        config1 = temp_yaml_vertical.get_config(use_cache=False)

        # Clear cache
        temp_yaml_vertical.clear_config_cache()

        # Load again
        config2 = temp_yaml_vertical.get_config(use_cache=False)

        # Should be different instances (but same content)
        assert config1 is not config2
        assert config1.metadata["name"] == config2.metadata["name"]


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing verticals."""

    def test_existing_verticals_still_work(self):
        """Test that existing verticals without YAML still work."""
        # Use a built-in vertical
        from victor.coding import CodingAssistant

        config = CodingAssistant.get_config()

        # Should have loaded from programmatic methods (or YAML if it exists)
        assert config.tools is not None
        assert len(config.tools.tools) > 0
        assert config.system_prompt is not None
        assert len(config.system_prompt) > 0

    def test_get_tools_and_get_system_prompt_still_work(self, temp_no_yaml_vertical):
        """Test that individual get_* methods still work."""
        # These methods should still be callable
        tools = temp_no_yaml_vertical.get_tools()
        assert isinstance(tools, list)

        prompt = temp_no_yaml_vertical.get_system_prompt()
        assert isinstance(prompt, str)


# ============================================================================
# Tests for CodingAssistant YAML Config
# ============================================================================


class TestCodingAssistantYAML:
    """Tests for CodingAssistant's YAML configuration."""

    def test_coding_assistant_loads_yaml(self):
        """Test that CodingAssistant loads from YAML if it exists."""
        from victor.coding import CodingAssistant

        # Clear cache to force reload
        CodingAssistant.clear_config_cache()

        # Load config
        config = CodingAssistant.get_config()

        # Verify basic structure
        assert config is not None
        assert isinstance(config, VerticalConfig)
        assert config.tools is not None
        assert config.system_prompt is not None

        # If YAML exists, verify it has expected content
        yaml_path = CodingAssistant._get_yaml_config_path()
        if yaml_path and yaml_path.exists():
            # Loaded from YAML
            assert (
                "Victor" in config.system_prompt
                or "software development" in config.system_prompt.lower()
            )
        else:
            # Loaded from programmatic methods
            assert len(config.tools.tools) > 0

    def test_coding_assistant_has_stages(self):
        """Test that CodingAssistant has stages defined."""
        from victor.coding import CodingAssistant

        config = CodingAssistant.get_config()

        # Should have stages either from YAML or programmatic methods
        assert len(config.stages) > 0

        # Check for common stages
        stage_names = list(config.stages.keys())
        assert any(
            stage in stage_names for stage in ["INITIAL", "PLANNING", "EXECUTING", "VERIFICATION"]
        )
