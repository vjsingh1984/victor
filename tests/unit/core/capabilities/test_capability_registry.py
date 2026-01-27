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

"""Tests for capability registry system (Phase 5.1).

TDD tests for:
- CapabilityDefinition dataclass
- CapabilityRegistry singleton
- Entry point discovery
- YAML capability loading
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import yaml

from victor.core.capabilities.types import (
    CapabilityDefinition,
    CapabilityType,
    ConfigSchema,
)
from victor.core.capabilities.registry import CapabilityRegistry
from victor.core.capabilities.handler import CapabilityHandler


# =============================================================================
# Test CapabilityDefinition Dataclass
# =============================================================================


class TestCapabilityDefinition:
    """Tests for CapabilityDefinition dataclass."""

    def test_definition_has_required_fields(self):
        """Test that CapabilityDefinition has all required fields."""
        definition = CapabilityDefinition(
            name="test_capability",
            capability_type=CapabilityType.TOOL,
            description="A test capability",
        )

        assert definition.name == "test_capability"
        assert definition.capability_type == CapabilityType.TOOL
        assert definition.description == "A test capability"
        assert definition.config_schema == {}
        assert definition.default_config == {}
        assert definition.dependencies == []
        assert definition.tags == []
        assert definition.version == "0.5.0"

    def test_definition_validates_config_against_schema(self):
        """Test that config validation works against JSON schema."""
        schema = {
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                "enabled": {"type": "boolean"},
            },
            "required": ["threshold"],
        }

        definition = CapabilityDefinition(
            name="quality_check",
            capability_type=CapabilityType.MODE,
            description="Quality threshold check",
            config_schema=schema,
            default_config={"threshold": 0.8, "enabled": True},
        )

        # Valid config should pass
        assert definition.validate_config({"threshold": 0.5, "enabled": True})

        # Invalid config (missing required field) should fail
        assert not definition.validate_config({"enabled": True})

        # Invalid config (wrong type) should fail
        assert not definition.validate_config({"threshold": "high"})

    def test_definition_serializable_to_yaml(self):
        """Test that CapabilityDefinition can be serialized to YAML."""
        definition = CapabilityDefinition(
            name="git_safety",
            capability_type=CapabilityType.SAFETY,
            description="Git safety rules",
            default_config={"block_force_push": True},
            tags=["safety", "git"],
        )

        yaml_dict = definition.to_yaml_dict()

        assert yaml_dict["name"] == "git_safety"
        assert yaml_dict["capability_type"] == "safety"
        assert yaml_dict["description"] == "Git safety rules"
        assert yaml_dict["default_config"]["block_force_push"] is True
        assert "safety" in yaml_dict["tags"]

    def test_definition_loadable_from_yaml(self):
        """Test that CapabilityDefinition can be loaded from YAML dict."""
        yaml_dict = {
            "name": "code_style",
            "capability_type": "mode",
            "description": "Code style settings",
            "default_config": {"formatter": "black", "line_length": 100},
            "tags": ["style", "formatting"],
        }

        definition = CapabilityDefinition.from_yaml_dict(yaml_dict)

        assert definition.name == "code_style"
        assert definition.capability_type == CapabilityType.MODE
        assert definition.description == "Code style settings"
        assert definition.default_config["formatter"] == "black"
        assert definition.default_config["line_length"] == 100

    def test_definition_with_all_optional_fields(self):
        """Test CapabilityDefinition with all optional fields specified."""
        definition = CapabilityDefinition(
            name="full_capability",
            capability_type=CapabilityType.TOOL,
            description="A fully specified capability",
            config_schema={"type": "object"},
            default_config={"key": "value"},
            dependencies=["base_capability"],
            tags=["tag1", "tag2"],
            version="2.0",
            enabled=False,
        )

        assert definition.version == "2.0"
        assert definition.enabled is False
        assert "base_capability" in definition.dependencies


# =============================================================================
# Test CapabilityRegistry
# =============================================================================


class TestCapabilityRegistry:
    """Tests for CapabilityRegistry singleton."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        CapabilityRegistry.reset_instance()
        yield
        CapabilityRegistry.reset_instance()

    def test_registry_singleton(self):
        """Test that CapabilityRegistry is a singleton."""
        registry1 = CapabilityRegistry.get_instance()
        registry2 = CapabilityRegistry.get_instance()

        assert registry1 is registry2

    def test_register_capability(self):
        """Test registering a capability definition."""
        registry = CapabilityRegistry.get_instance()
        definition = CapabilityDefinition(
            name="test_cap",
            capability_type=CapabilityType.TOOL,
            description="Test capability",
        )

        registry.register(definition)

        assert registry.get("test_cap") is not None
        assert registry.get("test_cap").name == "test_cap"

    def test_register_duplicate_raises(self):
        """Test that registering duplicate capability raises error."""
        registry = CapabilityRegistry.get_instance()
        definition = CapabilityDefinition(
            name="duplicate_cap",
            capability_type=CapabilityType.TOOL,
            description="Duplicate capability",
        )

        registry.register(definition)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(definition)

    def test_register_duplicate_with_replace(self):
        """Test that duplicate registration works with replace=True."""
        registry = CapabilityRegistry.get_instance()
        definition1 = CapabilityDefinition(
            name="replace_cap",
            capability_type=CapabilityType.TOOL,
            description="Original",
        )
        definition2 = CapabilityDefinition(
            name="replace_cap",
            capability_type=CapabilityType.TOOL,
            description="Replaced",
        )

        registry.register(definition1)
        registry.register(definition2, replace=True)

        assert registry.get("replace_cap").description == "Replaced"

    def test_list_by_type(self):
        """Test listing capabilities by type."""
        registry = CapabilityRegistry.get_instance()

        registry.register(
            CapabilityDefinition(
                name="tool1", capability_type=CapabilityType.TOOL, description="Tool 1"
            )
        )
        registry.register(
            CapabilityDefinition(
                name="tool2", capability_type=CapabilityType.TOOL, description="Tool 2"
            )
        )
        registry.register(
            CapabilityDefinition(
                name="safety1",
                capability_type=CapabilityType.SAFETY,
                description="Safety 1",
            )
        )

        tools = registry.list_by_type(CapabilityType.TOOL)
        safety = registry.list_by_type(CapabilityType.SAFETY)

        assert len(tools) == 2
        assert "tool1" in tools
        assert "tool2" in tools
        assert len(safety) == 1
        assert "safety1" in safety

    def test_discover_from_entry_points(self):
        """Test discovering capabilities from entry points."""
        registry = CapabilityRegistry.get_instance()

        # Create mock entry point
        mock_definition = CapabilityDefinition(
            name="external_cap",
            capability_type=CapabilityType.TOOL,
            description="External capability",
        )

        mock_entry_point = MagicMock()
        mock_entry_point.name = "external_cap"
        mock_entry_point.load.return_value = mock_definition

        with patch("victor.core.capabilities.registry.entry_points") as mock_entry_points:
            mock_entry_points.return_value.select.return_value = [mock_entry_point]

            count = registry.discover_from_entry_points("victor.capabilities")

        assert count == 1
        assert registry.get("external_cap") is not None

    def test_load_from_yaml_file(self):
        """Test loading capabilities from YAML file."""
        registry = CapabilityRegistry.get_instance()

        yaml_content = """
capabilities:
  - name: yaml_cap1
    capability_type: tool
    description: YAML capability 1
    default_config:
      enabled: true
  - name: yaml_cap2
    capability_type: safety
    description: YAML capability 2
    tags:
      - safety
      - validation
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            count = registry.load_from_yaml(temp_path)

            assert count == 2
            assert registry.get("yaml_cap1") is not None
            assert registry.get("yaml_cap2") is not None
            assert registry.get("yaml_cap1").capability_type == CapabilityType.TOOL
            assert "safety" in registry.get("yaml_cap2").tags
        finally:
            temp_path.unlink()

    def test_list_all(self):
        """Test listing all capabilities."""
        registry = CapabilityRegistry.get_instance()

        registry.register(
            CapabilityDefinition(
                name="cap1", capability_type=CapabilityType.TOOL, description="Cap 1"
            )
        )
        registry.register(
            CapabilityDefinition(
                name="cap2", capability_type=CapabilityType.MODE, description="Cap 2"
            )
        )

        all_caps = registry.list_all()

        assert len(all_caps) == 2
        assert "cap1" in all_caps
        assert "cap2" in all_caps

    def test_unregister(self):
        """Test unregistering a capability."""
        registry = CapabilityRegistry.get_instance()
        definition = CapabilityDefinition(
            name="to_remove",
            capability_type=CapabilityType.TOOL,
            description="To be removed",
        )

        registry.register(definition)
        assert registry.get("to_remove") is not None

        registry.unregister("to_remove")
        assert registry.get("to_remove") is None


# =============================================================================
# Test CapabilityHandler
# =============================================================================


class TestCapabilityHandler:
    """Tests for CapabilityHandler auto-generation."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        CapabilityRegistry.reset_instance()
        yield
        CapabilityRegistry.reset_instance()

    def test_handler_generated_for_registered_capability(self):
        """Test that handlers are auto-generated for registered capabilities."""
        registry = CapabilityRegistry.get_instance()
        definition = CapabilityDefinition(
            name="handled_cap",
            capability_type=CapabilityType.MODE,
            description="Handled capability",
            default_config={"key": "default_value"},
        )

        registry.register(definition)
        handler = registry.get_handler("handled_cap")

        assert handler is not None
        assert isinstance(handler, CapabilityHandler)

    def test_handler_configure_validates_config(self):
        """Test that handler.configure validates config against schema."""
        registry = CapabilityRegistry.get_instance()
        definition = CapabilityDefinition(
            name="validated_cap",
            capability_type=CapabilityType.MODE,
            description="Validated capability",
            config_schema={
                "type": "object",
                "properties": {"value": {"type": "integer", "minimum": 0}},
                "required": ["value"],
            },
            default_config={"value": 10},
        )

        registry.register(definition)
        handler = registry.get_handler("validated_cap")

        # Create mock context
        mock_context = MagicMock()

        # Valid config should work
        handler.configure(mock_context, {"value": 5})
        mock_context.set_capability_config.assert_called()

        # Invalid config should raise
        with pytest.raises(ValueError, match="Invalid configuration"):
            handler.configure(mock_context, {"value": -1})

    def test_handler_get_returns_merged_config(self):
        """Test that handler.get returns merged config with defaults."""
        registry = CapabilityRegistry.get_instance()
        definition = CapabilityDefinition(
            name="merged_cap",
            capability_type=CapabilityType.MODE,
            description="Merged capability",
            default_config={"a": 1, "b": 2, "c": 3},
        )

        registry.register(definition)
        handler = registry.get_handler("merged_cap")

        # Create mock context that returns partial config
        mock_context = MagicMock()
        mock_context.get_capability_config.return_value = {"a": 10, "d": 4}

        config = handler.get_config(mock_context)

        # Should merge: stored values + defaults for missing keys
        assert config["a"] == 10  # Stored value
        assert config["b"] == 2  # Default
        assert config["c"] == 3  # Default
        assert config["d"] == 4  # Stored value (not in defaults)

    def test_handler_none_for_unregistered(self):
        """Test that get_handler returns None for unregistered capability."""
        registry = CapabilityRegistry.get_instance()

        handler = registry.get_handler("nonexistent_cap")

        assert handler is None


# =============================================================================
# Test YAML Schema Validation
# =============================================================================


class TestYAMLSchemaValidation:
    """Tests for YAML capability file schema validation."""

    def test_valid_yaml_schema(self):
        """Test that valid YAML schema passes validation."""
        yaml_content = {
            "version": "1.0",
            "capabilities": [
                {
                    "name": "valid_cap",
                    "capability_type": "tool",
                    "description": "Valid capability",
                }
            ],
        }

        # Should not raise
        from victor.core.capabilities.registry import validate_capability_yaml

        errors = validate_capability_yaml(yaml_content)
        assert len(errors) == 0

    def test_invalid_yaml_missing_required(self):
        """Test that YAML missing required fields fails validation."""
        yaml_content = {
            "capabilities": [
                {
                    "name": "invalid_cap",
                    # Missing capability_type and description
                }
            ],
        }

        from victor.core.capabilities.registry import validate_capability_yaml

        errors = validate_capability_yaml(yaml_content)
        assert len(errors) > 0
        assert any("capability_type" in str(e).lower() for e in errors)

    def test_invalid_capability_type(self):
        """Test that invalid capability_type fails validation."""
        yaml_content = {
            "capabilities": [
                {
                    "name": "invalid_type_cap",
                    "capability_type": "invalid_type",
                    "description": "Invalid type",
                }
            ],
        }

        from victor.core.capabilities.registry import validate_capability_yaml

        errors = validate_capability_yaml(yaml_content)
        assert len(errors) > 0


# =============================================================================
# Test Entry Point Discovery
# =============================================================================


class TestEntryPointDiscovery:
    """Tests for external capability discovery via entry points."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        CapabilityRegistry.reset_instance()
        yield
        CapabilityRegistry.reset_instance()

    def test_discover_capability_class(self):
        """Test discovering capability class from entry point."""
        registry = CapabilityRegistry.get_instance()

        # Mock a capability class that defines get_definition()
        class ExternalCapability:
            @classmethod
            def get_definition(cls) -> CapabilityDefinition:
                return CapabilityDefinition(
                    name="external_class_cap",
                    capability_type=CapabilityType.TOOL,
                    description="External class capability",
                )

        mock_entry_point = MagicMock()
        mock_entry_point.name = "external_class_cap"
        mock_entry_point.load.return_value = ExternalCapability

        with patch("victor.core.capabilities.registry.entry_points") as mock_entry_points:
            mock_entry_points.return_value.select.return_value = [mock_entry_point]

            count = registry.discover_from_entry_points("victor.capabilities")

        assert count == 1
        assert registry.get("external_class_cap") is not None

    def test_discover_handles_load_errors(self):
        """Test that discovery handles entry point load errors gracefully."""
        registry = CapabilityRegistry.get_instance()

        mock_entry_point = MagicMock()
        mock_entry_point.name = "broken_cap"
        mock_entry_point.load.side_effect = ImportError("Module not found")

        with patch("victor.core.capabilities.registry.entry_points") as mock_entry_points:
            mock_entry_points.return_value.select.return_value = [mock_entry_point]

            # Should not raise, should log warning
            count = registry.discover_from_entry_points("victor.capabilities")

        assert count == 0
        assert registry.get("broken_cap") is None
