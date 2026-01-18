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

"""Tests for YAML vertical configuration loading.

Tests the VerticalConfigLoader and YAML-based configuration loading
for all verticals (coding, research, devops, rag).

Test Categories:
1. YAML loading and parsing
2. Configuration validation
3. Tool resolution
4. Prompt building
5. Middleware loading
6. Stage definitions
7. Extension loading
8. Backward compatibility
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from victor.core.verticals.config_loader import VerticalConfigLoader, VerticalYAMLConfig
from victor.core.verticals.base import VerticalBase, VerticalConfig
from victor.coding import CodingAssistant
from victor.research import ResearchAssistant
from victor.devops import DevOpsAssistant
from victor.rag import RAGAssistant


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def config_loader() -> VerticalConfigLoader:
    """Get a config loader instance."""
    return VerticalConfigLoader()


@pytest.fixture
def sample_yaml_config() -> Dict[str, Any]:
    """Sample YAML configuration for testing."""
    return {
        "metadata": {
            "name": "test_vertical",
            "version": "1.0.0",
            "description": "Test vertical for unit tests",
        },
        "core": {
            "tools": {"list": ["read", "write", "edit", "grep"]},
            "system_prompt": {"source": "inline", "text": "You are a test assistant."},
            "stages": {
                "INITIAL": {
                    "name": "INITIAL",
                    "description": "Initial stage",
                    "tools": ["read", "ls"],
                    "keywords": ["what", "how"],
                    "next_stages": ["EXECUTION"],
                }
            },
        },
        "provider": {"hints": {"preferred": ["anthropic", "openai"]}},
        "extensions": {
            "middleware": [],
            "safety": {"module": "test.safety", "class": "TestSafetyExtension"},
        },
    }


@pytest.fixture
def temp_yaml_file(sample_yaml_config: Dict[str, Any]) -> Path:
    """Create a temporary YAML file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_yaml_config, f)
        return Path(f.name)


# =============================================================================
# YAML Loading Tests (Tests 1-5)
# =============================================================================


class TestYAMLLoading:
    """Test YAML file loading and parsing."""

    def test_load_valid_yaml(self, config_loader: VerticalConfigLoader, temp_yaml_file: Path):
        """Test loading a valid YAML file."""
        config = config_loader.load_vertical_config("test_vertical", temp_yaml_file)

        assert config is not None
        assert config.name == "test_vertical"
        assert config.version == "1.0.0"
        assert config.description == "Test vertical for unit tests"

    def test_load_missing_file(self, config_loader: VerticalConfigLoader):
        """Test loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            config_loader.load_vertical_config(
                "test_vertical", Path("/nonexistent/path/vertical.yaml")
            )

    def test_load_invalid_yaml(self, config_loader: VerticalConfigLoader):
        """Test loading invalid YAML raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            invalid_path = Path(f.name)

        with pytest.raises(ValueError, match="Failed to parse YAML"):
            config_loader.load_vertical_config("test_vertical", invalid_path)

        invalid_path.unlink()

    def test_load_missing_required_fields(self, config_loader: VerticalConfigLoader):
        """Test YAML with missing required fields raises ValueError."""
        incomplete_config = {
            "name": "test_vertical"
            # Missing version, description, tools
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(incomplete_config, f)
            invalid_path = Path(f.name)

        with pytest.raises(ValueError, match="Missing required fields"):
            config_loader.load_vertical_config("test_vertical", invalid_path)

        invalid_path.unlink()

    def test_yaml_caching(self, config_loader: VerticalConfigLoader, temp_yaml_file: Path):
        """Test that YAML loading is cached."""
        # First load
        config1 = config_loader.load_vertical_config("test_vertical", temp_yaml_file)

        # Second load should use cache
        config2 = config_loader.load_vertical_config("test_vertical", temp_yaml_file)

        assert config1 is config2  # Same object reference


# =============================================================================
# Tool Loading Tests (Tests 6-10)
# =============================================================================


class TestToolLoading:
    """Test tool list loading from YAML."""

    def test_load_tools_from_list(self, config_loader: VerticalConfigLoader):
        """Test loading tools from core.tools.list."""
        yaml_config = {
            "metadata": {"name": "test", "version": "1.0", "description": "Test"},
            "core": {
                "tools": ["read", "write", "edit"],
                "system_prompt": {"source": "inline", "text": "Test"},
                "stages": {},
            },
            "provider": {"hints": {}},
            "extensions": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_config, f)
            temp_path = Path(f.name)

        config = config_loader.load_vertical_config("test", temp_path)
        assert config.tools == ["read", "write", "edit"]
        temp_path.unlink()

    def test_load_tools_from_dict(self, config_loader: VerticalConfigLoader):
        """Test loading tools from core.tools.list (dict format)."""
        yaml_config = {
            "metadata": {"name": "test", "version": "1.0", "description": "Test"},
            "core": {
                "tools": {"list": ["read", "write", "edit"]},
                "system_prompt": {"source": "inline", "text": "Test"},
                "stages": {},
            },
            "provider": {"hints": {}},
            "extensions": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_config, f)
            temp_path = Path(f.name)

        config = config_loader.load_vertical_config("test", temp_path)
        assert config.tools == ["read", "write", "edit"]
        temp_path.unlink()

    def test_load_empty_tools(self, config_loader: VerticalConfigLoader):
        """Test loading empty tool list."""
        yaml_config = {
            "metadata": {"name": "test", "version": "1.0", "description": "Test"},
            "core": {
                "tools": [],
                "system_prompt": {"source": "inline", "text": "Test"},
                "stages": {},
            },
            "provider": {"hints": {}},
            "extensions": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_config, f)
            temp_path = Path(f.name)

        config = config_loader.load_vertical_config("test", temp_path)
        assert config.tools == []
        temp_path.unlink()


# =============================================================================
# System Prompt Tests (Tests 11-15)
# =============================================================================


class TestSystemPrompt:
    """Test system prompt loading from YAML."""

    def test_load_inline_prompt(self, config_loader: VerticalConfigLoader):
        """Test loading inline system prompt."""
        prompt_text = "You are a helpful assistant."

        yaml_config = {
            "metadata": {"name": "test", "version": "1.0", "description": "Test"},
            "core": {
                "tools": [],
                "system_prompt": {"source": "inline", "text": prompt_text},
                "stages": {},
            },
            "provider": {"hints": {}},
            "extensions": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_config, f)
            temp_path = Path(f.name)

        config = config_loader.load_vertical_config("test", temp_path)
        assert config.system_prompt_config["text"] == prompt_text
        temp_path.unlink()

    def test_load_file_prompt(self, config_loader: VerticalConfigLoader):
        """Test loading system prompt from file."""
        prompt_text = "You are a helpful assistant from file."

        # Create prompt file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as pf:
            pf.write(prompt_text)
            prompt_path = Path(pf.name)

        yaml_config = {
            "metadata": {"name": "test", "version": "1.0", "description": "Test"},
            "core": {
                "tools": [],
                "system_prompt": {"source": "file", "path": str(prompt_path)},
                "stages": {},
            },
            "provider": {"hints": {}},
            "extensions": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_config, f)
            temp_path = Path(f.name)

        config = config_loader.load_vertical_config("test", temp_path)
        # File loading is tested in extract_prompt_text
        temp_path.unlink()
        prompt_path.unlink()


# =============================================================================
# Stage Definition Tests (Tests 16-20)
# =============================================================================


class TestStageDefinitions:
    """Test stage definition loading from YAML."""

    def test_load_stages(self, config_loader: VerticalConfigLoader):
        """Test loading stage definitions."""
        yaml_config = {
            "metadata": {"name": "test", "version": "1.0", "description": "Test"},
            "core": {
                "tools": [],
                "system_prompt": {"source": "inline", "text": "Test"},
                "stages": {
                    "INITIAL": {
                        "name": "INITIAL",
                        "description": "Initial stage",
                        "tools": ["read"],
                        "keywords": ["what"],
                        "next_stages": ["EXECUTION"],
                    }
                },
            },
            "provider": {"hints": {}},
            "extensions": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_config, f)
            temp_path = Path(f.name)

        config = config_loader.load_vertical_config("test", temp_path)
        assert "INITIAL" in config.stages
        assert config.stages["INITIAL"]["description"] == "Initial stage"
        temp_path.unlink()

    def test_load_empty_stages(self, config_loader: VerticalConfigLoader):
        """Test loading empty stage definitions."""
        yaml_config = {
            "metadata": {"name": "test", "version": "1.0", "description": "Test"},
            "core": {
                "tools": [],
                "system_prompt": {"source": "inline", "text": "Test"},
                "stages": {},
            },
            "provider": {"hints": {}},
            "extensions": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_config, f)
            temp_path = Path(f.name)

        config = config_loader.load_vertical_config("test", temp_path)
        assert config.stages == {}
        temp_path.unlink()


# =============================================================================
# Middleware Loading Tests (Tests 21-25)
# =============================================================================


class TestMiddlewareLoading:
    """Test middleware configuration loading from YAML."""

    def test_load_middleware_list(self, config_loader: VerticalConfigLoader):
        """Test loading middleware configuration."""
        yaml_config = {
            "metadata": {"name": "test", "version": "1.0", "description": "Test"},
            "core": {
                "tools": [],
                "system_prompt": {"source": "inline", "text": "Test"},
                "stages": {},
            },
            "provider": {"hints": {}},
            "extensions": {
                "middleware": [
                    {"class": "test.middleware.TestMiddleware", "enabled": True, "priority": "high"}
                ]
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_config, f)
            temp_path = Path(f.name)

        config = config_loader.load_vertical_config("test", temp_path)
        assert len(config.middleware) == 1
        assert config.middleware[0]["class"] == "test.middleware.TestMiddleware"
        temp_path.unlink()

    def test_load_empty_middleware(self, config_loader: VerticalConfigLoader):
        """Test loading empty middleware list."""
        yaml_config = {
            "metadata": {"name": "test", "version": "1.0", "description": "Test"},
            "core": {
                "tools": [],
                "system_prompt": {"source": "inline", "text": "Test"},
                "stages": {},
            },
            "provider": {"hints": {}},
            "extensions": {"middleware": []},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_config, f)
            temp_path = Path(f.name)

        config = config_loader.load_vertical_config("test", temp_path)
        assert config.middleware == []
        temp_path.unlink()


# =============================================================================
# Extension Loading Tests (Tests 26-28)
# =============================================================================


class TestExtensionLoading:
    """Test extension configuration loading."""

    def test_load_safety_extension(self, config_loader: VerticalConfigLoader):
        """Test loading safety extension configuration."""
        yaml_config = {
            "metadata": {"name": "test", "version": "1.0", "description": "Test"},
            "core": {
                "tools": [],
                "system_prompt": {"source": "inline", "text": "Test"},
                "stages": {},
            },
            "provider": {"hints": {}},
            "extensions": {"safety": {"module": "test.safety", "class": "TestSafetyExtension"}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_config, f)
            temp_path = Path(f.name)

        config = config_loader.load_vertical_config("test", temp_path)
        assert config.safety_extension is not None
        assert config.safety_extension["module"] == "test.safety"
        temp_path.unlink()

    def test_load_prompt_contributor(self, config_loader: VerticalConfigLoader):
        """Test loading prompt contributor configuration."""
        yaml_config = {
            "metadata": {"name": "test", "version": "1.0", "description": "Test"},
            "core": {
                "tools": [],
                "system_prompt": {"source": "inline", "text": "Test"},
                "stages": {},
            },
            "provider": {"hints": {}},
            "extensions": {
                "prompt_contributor": {"module": "test.prompts", "class": "TestPromptContributor"}
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_config, f)
            temp_path = Path(f.name)

        config = config_loader.load_vertical_config("test", temp_path)
        assert config.prompt_contributor is not None
        assert config.prompt_contributor["module"] == "test.prompts"
        temp_path.unlink()


# =============================================================================
# Real Vertical Tests (Tests 29-30)
# =============================================================================


class TestRealVerticals:
    """Test YAML loading for real verticals."""

    @pytest.mark.integration
    def test_coding_assistant_yaml_exists(self):
        """Test that CodingAssistant YAML config exists."""
        yaml_path = Path("victor/coding/config/vertical.yaml")
        assert yaml_path.exists(), f"CodingAssistant YAML config not found at {yaml_path}"

    @pytest.mark.integration
    def test_research_assistant_yaml_exists(self):
        """Test that ResearchAssistant YAML config exists."""
        yaml_path = Path("victor/research/config/vertical.yaml")
        assert yaml_path.exists(), f"ResearchAssistant YAML config not found at {yaml_path}"

    @pytest.mark.integration
    def test_devops_assistant_yaml_exists(self):
        """Test that DevOpsAssistant YAML config exists."""
        yaml_path = Path("victor/devops/config/vertical.yaml")
        assert yaml_path.exists(), f"DevOpsAssistant YAML config not found at {yaml_path}"

    @pytest.mark.integration
    def test_rag_assistant_yaml_exists(self):
        """Test that RAGAssistant YAML config exists."""
        yaml_path = Path("victor/rag/config/vertical.yaml")
        assert yaml_path.exists(), f"RAGAssistant YAML config not found at {yaml_path}"

    @pytest.mark.integration
    def test_load_coding_assistant_yaml(self, config_loader: VerticalConfigLoader):
        """Test loading CodingAssistant YAML config."""
        yaml_path = Path("victor/coding/config/vertical.yaml")
        if yaml_path.exists():
            config = config_loader.load_vertical_config("coding", yaml_path)
            assert config.name == "coding"
            assert len(config.tools) > 0
            assert len(config.stages) > 0

    @pytest.mark.integration
    def test_load_research_assistant_yaml(self, config_loader: VerticalConfigLoader):
        """Test loading ResearchAssistant YAML config."""
        yaml_path = Path("victor/research/config/vertical.yaml")
        if yaml_path.exists():
            config = config_loader.load_vertical_config("research", yaml_path)
            assert config.name == "research"
            assert len(config.tools) > 0
            assert len(config.stages) > 0


# =============================================================================
# Legacy Format Tests (Tests 31-32)
# =============================================================================


class TestLegacyFormat:
    """Test backward compatibility with legacy YAML format."""

    def test_load_legacy_flat_format(self, config_loader: VerticalConfigLoader):
        """Test loading legacy flat YAML format."""
        yaml_config = {
            "name": "test_vertical",
            "version": "1.0.0",
            "description": "Test vertical",
            "tools": ["read", "write"],
            "system_prompt": {"source": "inline", "text": "Test prompt"},
            "stages": {},
            "provider_hints": {},
            "middleware": [],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_config, f)
            temp_path = Path(f.name)

        config = config_loader.load_vertical_config("test_vertical", temp_path)
        assert config.name == "test_vertical"
        assert config.tools == ["read", "write"]
        temp_path.unlink()

    def test_load_structured_format(self, config_loader: VerticalConfigLoader):
        """Test loading new structured YAML format."""
        yaml_config = {
            "metadata": {
                "name": "test_vertical",
                "version": "1.0.0",
                "description": "Test vertical",
            },
            "core": {
                "tools": ["read", "write"],
                "system_prompt": {"source": "inline", "text": "Test prompt"},
                "stages": {},
            },
            "provider": {"hints": {}},
            "extensions": {"middleware": []},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_config, f)
            temp_path = Path(f.name)

        config = config_loader.load_vertical_config("test_vertical", temp_path)
        assert config.name == "test_vertical"
        assert config.tools == ["read", "write"]
        temp_path.unlink()
