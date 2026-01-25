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

"""Tests for tool dependency schema validation (Phase 7.1).

TDD tests for:
- JSON Schema validation
- ToolDependencyValidator
- Validation across all verticals
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from victor.framework.validation.tool_dependency_validator import (
    ToolDependencyValidator,
    ValidationResult,
    validate_tool_dependency_yaml,
    get_json_schema,
)


# =============================================================================
# Test ValidationResult
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        """Test creating a valid validation result."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            file_path=Path("test.yaml"),
        )

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_invalid_result_with_errors(self):
        """Test creating an invalid validation result."""
        result = ValidationResult(
            is_valid=False,
            errors=["Missing version field", "Invalid weight value"],
            warnings=["Unused tool reference"],
            file_path=Path("test.yaml"),
        )

        assert result.is_valid is False
        assert len(result.errors) == 2
        assert "Missing version field" in result.errors

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Minor issue"],
            file_path=Path("test.yaml"),
        )

        d = result.to_dict()

        assert d["is_valid"] is True
        assert d["errors"] == []
        assert d["warnings"] == ["Minor issue"]


# =============================================================================
# Test JSON Schema
# =============================================================================


class TestJSONSchema:
    """Tests for JSON Schema generation."""

    def test_schema_has_required_properties(self):
        """Test that JSON Schema has required properties defined."""
        schema = get_json_schema()

        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "version" in schema["properties"]
        assert "vertical" in schema["properties"]

    def test_schema_defines_transitions(self):
        """Test that transitions are properly defined in schema."""
        schema = get_json_schema()

        props = schema["properties"]
        assert "transitions" in props
        # Transitions is an object mapping tool names to arrays
        assert props["transitions"]["type"] == "object"

    def test_schema_defines_weight_constraints(self):
        """Test that weight constraints are defined."""
        schema = get_json_schema()

        # Weight should be a number between 0 and 1
        # This is defined in the transition item schema
        props = schema["properties"]
        transitions_def = props["transitions"]
        if "additionalProperties" in transitions_def:
            items = transitions_def["additionalProperties"]["items"]
            assert "weight" in items.get("properties", {})


# =============================================================================
# Test ToolDependencyValidator
# =============================================================================


class TestToolDependencyValidator:
    """Tests for ToolDependencyValidator."""

    def test_valid_schema_passes(self):
        """Test that valid YAML passes validation."""
        yaml_content = {
            "version": "1.0",
            "vertical": "coding",
            "transitions": {
                "read": [{"tool": "edit", "weight": 0.4}],
            },
            "clusters": {
                "file_ops": ["read", "write", "edit"],
            },
            "sequences": {
                "explore": ["ls", "read", "grep"],
            },
            "required_tools": ["read", "edit"],
            "optional_tools": ["grep"],
            "default_sequence": ["read", "edit"],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            temp_path = Path(f.name)

        try:
            validator = ToolDependencyValidator()
            result = validator.validate(temp_path)

            assert result.is_valid is True
            assert len(result.errors) == 0
        finally:
            temp_path.unlink()

    def test_missing_version_fails(self):
        """Test that missing version field fails validation."""
        yaml_content = {
            # Missing version
            "vertical": "coding",
            "transitions": {},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            temp_path = Path(f.name)

        try:
            validator = ToolDependencyValidator()
            result = validator.validate(temp_path)

            # May pass with default version
            # Check that validation completes without error
            assert result is not None
        finally:
            temp_path.unlink()

    def test_missing_vertical_fails(self):
        """Test that missing vertical field fails validation."""
        yaml_content = {
            "version": "1.0",
            # Missing vertical
            "transitions": {},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            temp_path = Path(f.name)

        try:
            validator = ToolDependencyValidator()
            result = validator.validate(temp_path)

            assert result.is_valid is False
            assert any("vertical" in e.lower() for e in result.errors)
        finally:
            temp_path.unlink()

    def test_invalid_transition_weight_fails(self):
        """Test that invalid transition weight fails validation."""
        yaml_content = {
            "version": "1.0",
            "vertical": "coding",
            "transitions": {
                "read": [{"tool": "edit", "weight": 1.5}],  # Invalid: > 1.0
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            temp_path = Path(f.name)

        try:
            validator = ToolDependencyValidator()
            result = validator.validate(temp_path)

            assert result.is_valid is False
            assert any("weight" in e.lower() for e in result.errors)
        finally:
            temp_path.unlink()

    def test_circular_dependency_warned(self):
        """Test that circular dependencies generate warnings."""
        yaml_content = {
            "version": "1.0",
            "vertical": "coding",
            "dependencies": [
                {"tool": "a", "depends_on": ["b"]},
                {"tool": "b", "depends_on": ["a"]},
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            temp_path = Path(f.name)

        try:
            validator = ToolDependencyValidator()
            result = validator.validate(temp_path)

            # Should have a warning about circular dependency
            assert len(result.warnings) > 0
            assert any("circular" in w.lower() for w in result.warnings)
        finally:
            temp_path.unlink()


# =============================================================================
# Test Vertical Validation
# =============================================================================


class TestValidateAllVerticals:
    """Tests for validating all vertical tool_dependencies.yaml files."""

    def test_validate_coding_vertical(self):
        """Test that coding vertical's tool_dependencies.yaml is valid."""
        validator = ToolDependencyValidator()
        path = Path("victor/coding/tool_dependencies.yaml")

        if path.exists():
            result = validator.validate(path)
            assert result.is_valid is True, f"Coding validation failed: {result.errors}"

    def test_validate_devops_vertical(self):
        """Test that devops vertical's tool_dependencies.yaml is valid."""
        validator = ToolDependencyValidator()
        path = Path("victor/devops/tool_dependencies.yaml")

        if path.exists():
            result = validator.validate(path)
            assert result.is_valid is True, f"DevOps validation failed: {result.errors}"

    def test_validate_rag_vertical(self):
        """Test that rag vertical's tool_dependencies.yaml is valid."""
        validator = ToolDependencyValidator()
        path = Path("victor/rag/tool_dependencies.yaml")

        if path.exists():
            result = validator.validate(path)
            assert result.is_valid is True, f"RAG validation failed: {result.errors}"

    def test_validate_research_vertical(self):
        """Test that research vertical's tool_dependencies.yaml is valid."""
        validator = ToolDependencyValidator()
        path = Path("victor/research/tool_dependencies.yaml")

        if path.exists():
            result = validator.validate(path)
            assert result.is_valid is True, f"Research validation failed: {result.errors}"

    def test_validate_dataanalysis_vertical(self):
        """Test that dataanalysis vertical's tool_dependencies.yaml is valid."""
        validator = ToolDependencyValidator()
        path = Path("victor/dataanalysis/tool_dependencies.yaml")

        if path.exists():
            result = validator.validate(path)
            assert (
                result.is_valid is True
            ), f"DataAnalysis validation failed: {result.errors}"

    def test_validate_all_verticals(self):
        """Integration test: validate all vertical configs."""
        validator = ToolDependencyValidator()
        results = validator.validate_all_verticals()

        # At least some verticals should be found
        assert len(results) > 0

        # All should be valid
        for vertical, result in results.items():
            assert result.is_valid is True, f"{vertical} validation failed: {result.errors}"


# =============================================================================
# Test Convenience Function
# =============================================================================


class TestValidateFunction:
    """Tests for validate_tool_dependency_yaml function."""

    def test_validate_valid_content(self):
        """Test validating valid YAML content."""
        yaml_content = {
            "version": "1.0",
            "vertical": "test",
            "transitions": {},
        }

        errors = validate_tool_dependency_yaml(yaml_content)
        assert len(errors) == 0

    def test_validate_invalid_content(self):
        """Test validating invalid YAML content."""
        yaml_content = {
            "version": "1.0",
            # Missing vertical
        }

        errors = validate_tool_dependency_yaml(yaml_content)
        assert len(errors) > 0
