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

"""Tests for VerticalConfigLoader.

Tests the YAML-based vertical configuration loading system that replaces
the 15+ get_* methods with a single YAML configuration file.

Phase 2, Work Stream 2.1: Declarative Vertical Configuration
"""

import tempfile
from pathlib import Path

import pytest

from victor.core.verticals.config_loader import VerticalConfigLoader, VerticalYAMLConfig


class TestVerticalConfigLoader:
    """Tests for VerticalConfigLoader."""

    @pytest.fixture
    def sample_yaml_config(self):
        """Create a sample vertical YAML configuration."""
        return """
name: coding
version: 2.0.0
description: "Software development assistant"

# Tool configuration
tools:
  - read
  - write
  - edit
  - grep
  - code_search
  - test
  - git

# System prompt configuration
system_prompt:
  source: file
  path: prompts/coding_system_prompt.txt

# Stage definitions
stages:
  INITIAL:
    description: "Understanding the coding task"
    keywords: [what, how, explain]
    tools: [read, overview]
    next_stages: [PLANNING, READING]

  PLANNING:
    description: "Planning implementation approach"
    keywords: [plan, design, approach]
    tools: [read, grep]
    next_stages: [READING]

  READING:
    description: "Reading and analyzing code"
    keywords: [read, analyze, understand]
    tools: [read, grep, code_search]
    next_stages: [ANALYZING]

# Middleware configuration
middleware:
  - name: code_validation
    class: victor.coding.middleware.CodeValidationMiddleware
    priority: 100

  - name: cache
    class: victor.framework.middleware.CacheMiddleware
    config:
      ttl_seconds: 300
      cacheable_tools: [read, ls, grep]

# Safety extension
safety:
  class: victor.coding.safety.CodingSafetyExtension

# Workflow configuration
workflows:
  source: directory
  path: workflows/

# Provider hints
provider_hints:
  preferred_providers: [anthropic, openai]
  requires_tools: true
  requires_streaming: true
"""

    @pytest.fixture
    def config_file(self, sample_yaml_config):
        """Create a temporary YAML config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(sample_yaml_config)
            config_path = Path(f.name)

        yield config_path

        # Cleanup
        config_path.unlink()

    @pytest.fixture
    def loader(self):
        """Create a VerticalConfigLoader instance."""
        return VerticalConfigLoader()

    def test_load_vertical_config(self, loader, config_file):
        """VerticalConfigLoader should load YAML configuration."""
        config = loader.load_vertical_config("coding", config_file)

        assert config is not None
        assert config.name == "coding"
        assert config.version == "2.0.0"
        assert config.description == "Software development assistant"

    def test_load_tools_from_yaml(self, loader, config_file):
        """VerticalConfigLoader should load tools list."""
        config = loader.load_vertical_config("coding", config_file)

        assert config.tools == [
            "read", "write", "edit", "grep",
            "code_search", "test", "git"
        ]

    def test_load_system_prompt_from_yaml(self, loader, config_file):
        """VerticalConfigLoader should load system prompt config."""
        config = loader.load_vertical_config("coding", config_file)

        assert config.system_prompt_config == {
            "source": "file",
            "path": "prompts/coding_system_prompt.txt"
        }

    def test_load_stages_from_yaml(self, loader, config_file):
        """VerticalConfigLoader should load stage definitions."""
        config = loader.load_vertical_config("coding", config_file)

        assert "INITIAL" in config.stages
        assert config.stages["INITIAL"]["description"] == "Understanding the coding task"
        assert config.stages["INITIAL"]["keywords"] == ["what", "how", "explain"]
        assert config.stages["INITIAL"]["tools"] == ["read", "overview"]

    def test_load_middleware_from_yaml(self, loader, config_file):
        """VerticalConfigLoader should load middleware config."""
        config = loader.load_vertical_config("coding", config_file)

        assert len(config.middleware) == 2
        assert config.middleware[0]["name"] == "code_validation"
        assert config.middleware[0]["class"] == "victor.coding.middleware.CodeValidationMiddleware"

        # Check middleware with config
        cache_middleware = [m for m in config.middleware if m["name"] == "cache"][0]
        assert cache_middleware["config"]["ttl_seconds"] == 300
        assert cache_middleware["config"]["cacheable_tools"] == ["read", "ls", "grep"]

    def test_load_safety_extension_from_yaml(self, loader, config_file):
        """VerticalConfigLoader should load safety extension config."""
        config = loader.load_vertical_config("coding", config_file)

        assert config.safety_extension == {
            "class": "victor.coding.safety.CodingSafetyExtension"
        }

    def test_load_workflows_from_yaml(self, loader, config_file):
        """VerticalConfigLoader should load workflow config."""
        config = loader.load_vertical_config("coding", config_file)

        assert config.workflows == {
            "source": "directory",
            "path": "workflows/"
        }

    def test_load_provider_hints_from_yaml(self, loader, config_file):
        """VerticalConfigLoader should load provider hints."""
        config = loader.load_vertical_config("coding", config_file)

        assert config.provider_hints["preferred_providers"] == ["anthropic", "openai"]
        assert config.provider_hints["requires_tools"] is True

    def test_escape_hatch_for_dynamic_tools(self, loader, tmp_path):
        """VerticalConfigLoader should support escape hatch for dynamic tool selection."""
        # Create a vertical class with escape hatch
        yaml_config = """
name: custom
version: 1.0.0
description: "Custom vertical with escape hatch"

tools:
  - read
  - write
"""
        config_file = tmp_path / "custom.yaml"
        config_file.write_text(yaml_config)

        class CustomVertical:
            name = "custom"

            @classmethod
            def escape_hatch_tools(cls, yaml_tools):
                """Add additional tools dynamically."""
                # Add environment-specific tools
                import os
                if os.getenv("ENABLE_ADVANCED_TOOLS"):
                    return yaml_tools + ["advanced_tool"]
                return yaml_tools

        config = loader.load_vertical_config(
            "custom",
            config_file,
            escape_hatch_class=CustomVertical
        )

        # Should have base tools
        assert "read" in config.tools
        assert "write" in config.tools

    def test_invalid_yaml_raises_error(self, loader, tmp_path):
        """VerticalConfigLoader should raise error for invalid YAML."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [unclosed")

        with pytest.raises(ValueError, match="Failed to parse YAML"):
            loader.load_vertical_config("test", config_file)

    def test_missing_required_field_raises_error(self, loader, tmp_path):
        """VerticalConfigLoader should raise error for missing required fields."""
        yaml_config = """
name: test
# Missing version and description
"""
        config_file = tmp_path / "incomplete.yaml"
        config_file.write_text(yaml_config)

        with pytest.raises(ValueError, match="Missing required field"):
            loader.load_vertical_config("test", config_file)


class TestVerticalYAMLConfig:
    """Tests for VerticalYAMLConfig dataclass."""

    def test_create_yaml_config(self):
        """VerticalYAMLConfig should create from parsed YAML data."""
        config = VerticalYAMLConfig(
            name="coding",
            version="2.0.0",
            description="Software development",
            tools=["read", "write"],
            system_prompt_config={"source": "file", "path": "prompt.txt"},
            stages={},
            middleware=[],
            safety_extension=None,
            workflows={"source": "directory"},
            provider_hints={}
        )

        assert config.name == "coding"
        assert config.tools == ["read", "write"]

    def test_yaml_config_to_dict(self):
        """VerticalYAMLConfig should convert to dictionary."""
        config = VerticalYAMLConfig(
            name="coding",
            version="2.0.0",
            description="Software development",
            tools=["read", "write"],
            system_prompt_config={},
            stages={},
            middleware=[],
            safety_extension=None,
            workflows={},
            provider_hints={}
        )

        config_dict = config.to_dict()

        assert config_dict["name"] == "coding"
        assert config_dict["tools"] == ["read", "write"]


__all__ = [
    "TestVerticalConfigLoader",
    "TestVerticalYAMLConfig",
]
