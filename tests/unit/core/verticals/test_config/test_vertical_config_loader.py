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

"""Tests for VerticalConfigLoader implementation.

Tests the YAML-based vertical configuration loader that replaces
multiple get_* methods with declarative YAML configuration.
"""

import pytest

from victor.core.verticals.config.vertical_config_loader import VerticalConfigLoader
from victor.core.verticals.base import VerticalConfig, StageDefinition


class TestVerticalConfigLoaderBasic:
    """Tests for basic YAML loading functionality."""

    def test_load_from_yaml_file(self, loader, temp_yaml_config):
        """Test loading configuration from YAML file."""
        config = loader.load_from_yaml(temp_yaml_config)

        assert config is not None
        assert isinstance(config, VerticalConfig)

    def test_load_minimal_config(self, loader, minimal_yaml):
        """Test loading minimal YAML config with only required fields."""
        config = loader.load_from_yaml(minimal_yaml)

        assert config.tools is not None
        assert len(config.tools.tools) > 0
        assert config.system_prompt is not None
        assert len(config.system_prompt) > 0

    def test_load_full_config(self, loader, full_yaml):
        """Test loading full YAML config with all optional fields."""
        config = loader.load_from_yaml(full_yaml)

        # Core fields
        assert config.tools is not None
        assert config.system_prompt is not None

        # Optional fields
        assert config.stages is not None
        assert len(config.stages) > 0
        assert config.provider_hints is not None
        assert config.evaluation_criteria is not None

    def test_missing_yaml_file(self, loader):
        """Test handling of missing YAML file."""
        result = loader.load_from_yaml("/nonexistent/path.yaml")

        assert result is None

    def test_invalid_yaml_syntax(self, loader, invalid_yaml):
        """Test handling of invalid YAML syntax."""
        strict_loader = VerticalConfigLoader(strict_validation=True)
        with pytest.raises(Exception):  # Could be YAMLError or ValueError
            strict_loader.load_from_yaml(invalid_yaml)


class TestVerticalConfigLoaderTools:
    """Tests for tools configuration loading."""

    def test_load_tools_list(self, loader, tools_config):
        """Test loading tools as a simple list."""
        config = loader.load_from_yaml(tools_config)

        assert config.tools is not None
        assert "read" in config.tools.tools
        assert "write" in config.tools.tools
        assert "edit" in config.tools.tools

    def test_load_tools_with_exclusions(self, loader, tools_with_exclusions):
        """Test loading tools with exclusions."""
        config = loader.load_from_yaml(tools_with_exclusions)

        # Should have base tools minus exclusions
        assert "read" in config.tools.tools
        assert "dangerous_tool" not in config.tools.tools


class TestVerticalConfigLoaderSystemPrompt:
    """Tests for system prompt configuration loading."""

    def test_load_inline_prompt(self, loader, inline_prompt_config):
        """Test loading inline system prompt."""
        config = loader.load_from_yaml(inline_prompt_config)

        assert "You are an expert" in config.system_prompt
        assert len(config.system_prompt) > 0

    def test_load_prompt_from_file(self, loader, file_prompt_config, temp_prompt_file):
        """Test loading system prompt from file."""
        config = loader.load_from_yaml(file_prompt_config)

        assert "This is from a file" in config.system_prompt


class TestVerticalConfigLoaderStages:
    """Tests for stages configuration loading."""

    def test_load_stages(self, loader, stages_config):
        """Test loading stage definitions."""
        config = loader.load_from_yaml(stages_config)

        assert config.stages is not None
        assert "INITIAL" in config.stages
        assert "PLANNING" in config.stages

    def test_stage_definition_structure(self, loader, stages_config):
        """Test that stage definitions have correct structure."""
        config = loader.load_from_yaml(stages_config)

        initial_stage = config.stages["INITIAL"]
        assert isinstance(initial_stage, StageDefinition)
        assert initial_stage.name == "INITIAL"
        assert initial_stage.description is not None
        assert len(initial_stage.tools) > 0
        assert len(initial_stage.keywords) > 0
        assert len(initial_stage.next_stages) > 0


class TestVerticalConfigLoaderProvider:
    """Tests for provider configuration loading."""

    def test_load_provider_hints(self, loader, provider_config):
        """Test loading provider hints."""
        config = loader.load_from_yaml(provider_config)

        assert config.provider_hints is not None
        assert "preferred" in config.provider_hints

    def test_load_provider_parameters(self, loader, provider_config):
        """Test loading provider parameters."""
        config = loader.load_from_yaml(provider_config)

        assert config.provider_hints is not None
        # Check for temperature in nested config
        assert any("temperature" in str(v) for v in config.provider_hints.values())


class TestVerticalConfigLoaderTieredTools:
    """Tests for tiered tools configuration loading."""

    def test_load_tiered_tools(self, loader, tiered_config):
        """Test loading tiered tools configuration."""
        config = loader.load_from_yaml(tiered_config)

        # Should be stored in metadata
        assert config.metadata is not None
        assert "tiered_tools" in config.metadata

    def test_tiered_tools_structure(self, loader, tiered_config):
        """Test that tiered tools have correct structure."""
        config = loader.load_from_yaml(tiered_config)

        tiered = config.metadata["tiered_tools"]
        assert "mandatory" in tiered
        assert "vertical_core" in tiered


class TestVerticalConfigLoaderExtensions:
    """Tests for extensions configuration loading."""

    def test_load_middleware_config(self, loader, extensions_config):
        """Test loading middleware configuration."""
        config = loader.load_from_yaml(extensions_config)

        # Should be stored in metadata
        assert config.metadata is not None
        assert "extensions" in config.metadata
        assert "middleware" in config.metadata["extensions"]

    def test_load_safety_config(self, loader, extensions_config):
        """Test loading safety patterns configuration."""
        config = loader.load_from_yaml(extensions_config)

        extensions = config.metadata["extensions"]
        assert "safety" in extensions


class TestVerticalConfigLoaderEvaluation:
    """Tests for evaluation configuration loading."""

    def test_load_evaluation_criteria(self, loader, evaluation_config):
        """Test loading evaluation criteria."""
        config = loader.load_from_yaml(evaluation_config)

        assert config.evaluation_criteria is not None
        assert len(config.evaluation_criteria) > 0

    def test_load_evaluation_metrics(self, loader, evaluation_config):
        """Test loading evaluation metrics."""
        config = loader.load_from_yaml(evaluation_config)

        assert config.metadata is not None
        assert "evaluation_metrics" in config.metadata


class TestVerticalConfigLoaderValidation:
    """Tests for YAML validation."""

    def test_validate_required_fields(self, loader, minimal_yaml):
        """Test that required fields are validated."""
        # Should not raise if required fields present
        config = loader.load_from_yaml(minimal_yaml)
        assert config is not None

    def test_validate_missing_tools(self, loader, invalid_no_tools):
        """Test validation fails when tools missing."""
        strict_loader = VerticalConfigLoader(strict_validation=True)
        with pytest.raises(ValueError):  # Missing required field
            strict_loader.load_from_yaml(invalid_no_tools)

    def test_validate_missing_prompt(self, loader, invalid_no_prompt):
        """Test validation fails when system prompt missing."""
        strict_loader = VerticalConfigLoader(strict_validation=True)
        with pytest.raises(ValueError):  # Missing required field
            strict_loader.load_from_yaml(invalid_no_prompt)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def loader():
    """Provide VerticalConfigLoader instance."""
    return VerticalConfigLoader()


@pytest.fixture
def temp_yaml_config(tmp_path):
    """Create a temporary YAML config file."""
    config_file = tmp_path / "vertical.yaml"
    yaml_content = """
metadata:
  name: test_vertical
  description: Test vertical for unit tests

core:
  tools:
    list:
      - read
      - write
  system_prompt:
    source: inline
    text: "You are a test assistant"

core:
  tools:
    list:
      - read
      - write
  system_prompt:
    source: inline
    text: "You are a test assistant"
"""
    config_file.write_text(yaml_content)
    return config_file


@pytest.fixture
def minimal_yaml(tmp_path):
    """Create minimal YAML with only required fields."""
    config_file = tmp_path / "minimal.yaml"
    yaml_content = """
metadata:
  name: minimal
  description: Minimal vertical

core:
  tools:
    list: [read, write]
  system_prompt:
    source: inline
    text: "Minimal prompt"
"""
    config_file.write_text(yaml_content)
    return config_file


@pytest.fixture
def full_yaml(tmp_path):
    """Create full YAML with all fields."""
    config_file = tmp_path / "full.yaml"
    yaml_content = """
metadata:
  name: full
  description: Full vertical
  version: "0.5.0"

core:
  tools:
    list: [read, write, edit]
  system_prompt:
    source: inline
    text: "Full prompt"
  stages:
    INITIAL:
      name: INITIAL
      description: Initial stage
      tools: [read]
      keywords: [what, how]
      next_stages: [PLANNING]
    PLANNING:
      name: PLANNING
      description: Planning stage
      tools: [read, grep]
      keywords: [plan, design]
      next_stages: []

provider:
  hints:
    preferred: [anthropic]
    min_context_window: 100000

evaluation:
  criteria:
    - "Criterion 1"
    - "Criterion 2"
"""
    config_file.write_text(yaml_content)
    return config_file


@pytest.fixture
def invalid_yaml(tmp_path):
    """Create invalid YAML file."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid: yaml: content: [unclosed")
    return config_file


@pytest.fixture
def tools_config(tmp_path):
    """Create YAML with tools configuration."""
    config_file = tmp_path / "tools.yaml"
    yaml_content = """
metadata:
  name: tools_test
  description: Tools test

core:
  tools:
    list: [read, write, edit]
  system_prompt:
    source: inline
    text: "Test"
"""
    config_file.write_text(yaml_content)
    return config_file


@pytest.fixture
def tools_with_exclusions(tmp_path):
    """Create YAML with tool exclusions."""
    config_file = tmp_path / "exclusions.yaml"
    yaml_content = """
metadata:
  name: exclusions_test
  description: Exclusions test

core:
  tools:
    list: [read, write, dangerous_tool]
    exclude: [dangerous_tool]
  system_prompt:
    source: inline
    text: "Test"
"""
    config_file.write_text(yaml_content)
    return config_file


@pytest.fixture
def inline_prompt_config(tmp_path):
    """Create YAML with inline prompt."""
    config_file = tmp_path / "inline_prompt.yaml"
    yaml_content = """
metadata:
  name: inline_prompt
  description: Inline prompt test

core:
  tools:
    list: [read]
  system_prompt:
    source: inline
    text: "You are an expert assistant for software development."
"""
    config_file.write_text(yaml_content)
    return config_file


@pytest.fixture
def file_prompt_config(tmp_path):
    """Create YAML with file-based prompt."""
    config_file = tmp_path / "file_prompt.yaml"
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("This is from a file")

    yaml_content = f"""
metadata:
  name: file_prompt
  description: File prompt test

core:
  tools:
    list: [read]
  system_prompt:
    source: file
    file_path: {prompt_file}
"""
    config_file.write_text(yaml_content)
    return config_file


@pytest.fixture
def temp_prompt_file(tmp_path):
    """Helper fixture for file_prompt_config."""
    return None  # Already created in file_prompt_config


@pytest.fixture
def stages_config(tmp_path):
    """Create YAML with stages configuration."""
    config_file = tmp_path / "stages.yaml"
    yaml_content = """
metadata:
  name: stages_test
  description: Stages test

core:
  tools:
    list: [read, write]
  system_prompt:
    source: inline
    text: "Test"
  stages:
    INITIAL:
      name: INITIAL
      description: Understanding the request
      tools: [read, ls, grep]
      keywords: [what, how, explain]
      next_stages: [PLANNING]
    PLANNING:
      name: PLANNING
      description: Creating implementation plan
      tools: [read, grep, code_search]
      keywords: [plan, design, approach]
      next_stages: []
"""
    config_file.write_text(yaml_content)
    return config_file


@pytest.fixture
def provider_config(tmp_path):
    """Create YAML with provider configuration."""
    config_file = tmp_path / "provider.yaml"
    yaml_content = """
metadata:
  name: provider_test
  description: Provider test

core:
  tools:
    list: [read]
  system_prompt:
    source: inline
    text: "Test"

provider:
  hints:
    preferred: [anthropic, openai]
  parameters:
    temperature: 0.7
    max_tokens: 4096
"""
    config_file.write_text(yaml_content)
    return config_file


@pytest.fixture
def tiered_config(tmp_path):
    """Create YAML with tiered tools configuration."""
    config_file = tmp_path / "tiered.yaml"
    yaml_content = """
metadata:
  name: tiered_test
  description: Tiered tools test

core:
  tools:
    list: [read]
  system_prompt:
    source: inline
    text: "Test"

tiered_tools:
  mandatory: [read, ls, grep]
  vertical_core: [code_search, edit]
  semantic_pool: [symbol, refs]
"""
    config_file.write_text(yaml_content)
    return config_file


@pytest.fixture
def extensions_config(tmp_path):
    """Create YAML with extensions configuration."""
    config_file = tmp_path / "extensions.yaml"
    yaml_content = """
metadata:
  name: extensions_test
  description: Extensions test

core:
  tools:
    list: [read]
  system_prompt:
    source: inline
    text: "Test"

extensions:
  middleware:
    enabled: true
    list:
      - name: TestMiddleware
        module: victor.test.middleware
        priority: HIGH
  safety:
    enabled: true
    patterns:
      - name: TestPattern
        category: test
        risk_level: MEDIUM
"""
    config_file.write_text(yaml_content)
    return config_file


@pytest.fixture
def evaluation_config(tmp_path):
    """Create YAML with evaluation configuration."""
    config_file = tmp_path / "evaluation.yaml"
    yaml_content = """
metadata:
  name: evaluation_test
  description: Evaluation test

core:
  tools:
    list: [read]
  system_prompt:
    source: inline
    text: "Test"

evaluation:
  criteria:
    - "Test criterion 1"
    - "Test criterion 2"
  metrics:
    - name: success_rate
      threshold: 0.9
      description: Success rate threshold
"""
    config_file.write_text(yaml_content)
    return config_file


@pytest.fixture
def invalid_no_tools(tmp_path):
    """Create invalid YAML missing tools."""
    config_file = tmp_path / "no_tools.yaml"
    yaml_content = """
metadata:
  name: no_tools
  description: Missing tools

core:
  system_prompt:
    source: inline
    text: "Test"
"""
    config_file.write_text(yaml_content)
    return config_file


@pytest.fixture
def invalid_no_prompt(tmp_path):
    """Create invalid YAML missing system prompt."""
    config_file = tmp_path / "no_prompt.yaml"
    yaml_content = """
metadata:
  name: no_prompt
  description: Missing prompt

core:
  tools:
    list: [read]
"""
    config_file.write_text(yaml_content)
    return config_file
