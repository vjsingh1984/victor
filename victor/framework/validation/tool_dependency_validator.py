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

"""Tool dependency schema validator (Phase 7.1).

This module provides JSON Schema validation for tool_dependencies.yaml files,
complementing the existing Pydantic validation in victor.core.tool_dependency_schema.

Key Features:
- JSON Schema for external validation (IDE integration, CI/CD)
- ToolDependencyValidator class for programmatic validation
- Circular dependency detection
- Validation across all verticals

Example:
    from victor.framework.validation.tool_dependency_validator import (
        ToolDependencyValidator,
        get_json_schema,
    )

    # Validate a single file
    validator = ToolDependencyValidator()
    result = validator.validate(Path("tool_dependencies.yaml"))
    print(f"Valid: {result.is_valid}")

    # Validate all verticals
    results = validator.validate_all_verticals()
    for vertical, result in results.items():
        print(f"{vertical}: {'OK' if result.is_valid else result.errors}")

    # Get JSON Schema for external tools
    schema = get_json_schema()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


# JSON Schema for tool_dependencies.yaml
TOOL_DEPENDENCY_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Tool Dependency Configuration",
    "description": "Schema for vertical tool dependency YAML files",
    "type": "object",
    "required": ["vertical"],
    "properties": {
        "version": {
            "type": "string",
            "pattern": r"^\d+\.\d+$",
            "description": "Schema version for compatibility checking",
            "default": "1.0",
        },
        "vertical": {
            "type": "string",
            "description": "Name of the vertical (coding, devops, rag, etc.)",
            "minLength": 1,
        },
        "transitions": {
            "type": "object",
            "description": "Tool transition probability mappings",
            "additionalProperties": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["tool", "weight"],
                    "properties": {
                        "tool": {
                            "type": "string",
                            "description": "Target tool name for transition",
                            "minLength": 1,
                        },
                        "weight": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Transition probability weight (0.0 to 1.0)",
                        },
                    },
                },
            },
        },
        "clusters": {
            "type": "object",
            "description": "Groups of related tools",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
        },
        "sequences": {
            "type": "object",
            "description": "Named tool sequences for task types",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
        },
        "dependencies": {
            "type": "array",
            "description": "Tool dependency definitions",
            "items": {
                "type": "object",
                "required": ["tool"],
                "properties": {
                    "tool": {
                        "type": "string",
                        "description": "Tool name for this dependency",
                        "minLength": 1,
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tools that should be called before this one",
                    },
                    "enables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tools enabled after this one succeeds",
                    },
                    "weight": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Dependency strength weight (0.0 to 1.0)",
                        "default": 1.0,
                    },
                },
            },
        },
        "required_tools": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Essential tools for this vertical",
        },
        "optional_tools": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Tools that enhance but aren't required",
        },
        "default_sequence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Fallback sequence when task type unknown",
        },
        "metadata": {
            "type": "object",
            "description": "Optional additional metadata for extensibility",
            "additionalProperties": True,
        },
    },
}


def get_json_schema() -> dict[str, Any]:
    """Get the JSON Schema for tool_dependencies.yaml files.

    Returns:
        JSON Schema dictionary

    Example:
        import json
        schema = get_json_schema()
        with open("tool_dependencies.schema.json", "w") as f:
            json.dump(schema, f, indent=2)
    """
    return TOOL_DEPENDENCY_SCHEMA.copy()


@dataclass
class ValidationResult:
    """Result from validating a tool_dependencies.yaml file.

    Attributes:
        is_valid: Whether the file is valid
        errors: List of validation error messages
        warnings: List of validation warning messages
        file_path: Path to the validated file
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    file_path: Optional[Path] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with validation results
        """
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "file_path": str(self.file_path) if self.file_path else None,
        }


def validate_tool_dependency_yaml(data: dict[str, Any]) -> list[str]:
    """Validate tool dependency YAML data.

    This function provides basic validation without requiring jsonschema.
    For full JSON Schema validation, use ToolDependencyValidator.

    Args:
        data: YAML data dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: list[str] = []

    # Required field: vertical
    if "vertical" not in data:
        errors.append("Missing required field: 'vertical'")
    elif not isinstance(data["vertical"], str) or not data["vertical"].strip():
        errors.append("'vertical' must be a non-empty string")

    # Validate transitions
    if "transitions" in data:
        transitions = data["transitions"]
        if not isinstance(transitions, dict):
            errors.append("'transitions' must be an object")
        else:
            for source, targets in transitions.items():
                if not isinstance(targets, list):
                    errors.append(f"transitions.{source} must be an array")
                    continue
                for i, target in enumerate(targets):
                    if not isinstance(target, dict):
                        errors.append(f"transitions.{source}[{i}] must be an object")
                        continue
                    if "tool" not in target:
                        errors.append(f"transitions.{source}[{i}]: missing 'tool'")
                    if "weight" not in target:
                        errors.append(f"transitions.{source}[{i}]: missing 'weight'")
                    elif not isinstance(target["weight"], (int, float)):
                        errors.append(f"transitions.{source}[{i}]: 'weight' must be a number")
                    elif target["weight"] < 0 or target["weight"] > 1:
                        errors.append(
                            f"transitions.{source}[{i}]: 'weight' must be between 0 and 1"
                        )

    # Validate clusters
    if "clusters" in data:
        clusters = data["clusters"]
        if not isinstance(clusters, dict):
            errors.append("'clusters' must be an object")
        else:
            for name, tools in clusters.items():
                if not isinstance(tools, list):
                    errors.append(f"clusters.{name} must be an array")
                elif len(tools) == 0:
                    errors.append(f"clusters.{name} must have at least one tool")

    # Validate sequences
    if "sequences" in data:
        sequences = data["sequences"]
        if not isinstance(sequences, dict):
            errors.append("'sequences' must be an object")
        else:
            for name, tools in sequences.items():
                if not isinstance(tools, list):
                    errors.append(f"sequences.{name} must be an array")
                elif len(tools) == 0:
                    errors.append(f"sequences.{name} must have at least one tool")

    # Validate dependencies
    if "dependencies" in data:
        deps = data["dependencies"]
        if not isinstance(deps, list):
            errors.append("'dependencies' must be an array")
        else:
            for i, dep in enumerate(deps):
                if not isinstance(dep, dict):
                    errors.append(f"dependencies[{i}] must be an object")
                    continue
                if "tool" not in dep:
                    errors.append(f"dependencies[{i}]: missing 'tool'")
                if "weight" in dep:
                    weight = dep["weight"]
                    if not isinstance(weight, (int, float)):
                        errors.append(f"dependencies[{i}]: 'weight' must be a number")
                    elif weight < 0 or weight > 1:
                        errors.append(f"dependencies[{i}]: 'weight' must be between 0 and 1")

    return errors


def _detect_circular_dependencies(data: dict[str, Any]) -> list[str]:
    """Detect circular dependencies in tool dependency graph.

    Args:
        data: YAML data dictionary

    Returns:
        List of warning messages for circular dependencies
    """
    warnings: list[str] = []

    deps = data.get("dependencies", [])
    if not deps:
        return warnings

    # Build dependency graph
    graph: dict[str, set[str]] = {}
    for dep in deps:
        if not isinstance(dep, dict):
            continue
        tool = dep.get("tool")
        depends_on = dep.get("depends_on", [])
        if tool and isinstance(depends_on, list):
            graph[tool] = set(depends_on)

    # Detect cycles using DFS
    def has_cycle(node: str, visited: set[str], path: set[str]) -> bool:
        visited.add(node)
        path.add(node)

        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                if has_cycle(neighbor, visited, path):
                    return True
            elif neighbor in path:
                warnings.append(f"Circular dependency detected involving: {node} -> {neighbor}")
                return True

        path.remove(node)
        return False

    visited: set[str] = set()
    for node in graph:
        if node not in visited:
            has_cycle(node, visited, set())

    return warnings


class ToolDependencyValidator:
    """Validator for tool_dependencies.yaml files.

    Provides both programmatic validation and CLI support for validating
    tool dependency configurations across verticals.

    Example:
        validator = ToolDependencyValidator()

        # Validate single file
        result = validator.validate(Path("tool_dependencies.yaml"))
        if not result.is_valid:
            print(f"Errors: {result.errors}")

        # Validate all verticals
        results = validator.validate_all_verticals()
    """

    # Known vertical directories
    VERTICALS = ["coding", "devops", "rag", "research", "dataanalysis"]

    def __init__(self, use_json_schema: bool = False):
        """Initialize validator.

        Args:
            use_json_schema: If True, use jsonschema for validation (optional dep)
        """
        self._use_json_schema = use_json_schema

    def validate(self, path: Path) -> ValidationResult:
        """Validate a tool_dependencies.yaml file.

        Args:
            path: Path to YAML file

        Returns:
            ValidationResult with validation status
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check file exists
        if not path.exists():
            return ValidationResult(
                is_valid=False,
                errors=[f"File not found: {path}"],
                file_path=path,
            )

        # Load YAML
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid YAML: {e}"],
                file_path=path,
            )

        # Validate using JSON Schema if available and requested
        if self._use_json_schema:
            try:
                import jsonschema

                jsonschema.validate(instance=data, schema=TOOL_DEPENDENCY_SCHEMA)
            except ImportError:
                logger.debug("jsonschema not installed, using basic validation")
            except jsonschema.ValidationError as e:
                errors.append(f"Schema validation error: {e.message}")

        # Basic validation (always run)
        basic_errors = validate_tool_dependency_yaml(data)
        errors.extend(basic_errors)

        # Pydantic validation (leveraging existing infrastructure)
        try:
            from victor.core.tool_dependency_schema import ToolDependencySpec

            ToolDependencySpec.model_validate(data)
        except ImportError:
            logger.debug("Pydantic validation unavailable")
        except Exception as e:
            errors.append(f"Pydantic validation error: {str(e)}")

        # Detect circular dependencies
        circular_warnings = _detect_circular_dependencies(data)
        warnings.extend(circular_warnings)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            file_path=path,
        )

    def validate_all_verticals(
        self, base_path: Optional[Path] = None
    ) -> dict[str, ValidationResult]:
        """Validate tool_dependencies.yaml across all verticals.

        Args:
            base_path: Base path to victor directory (default: auto-detect)

        Returns:
            Dictionary mapping vertical names to ValidationResult
        """
        if base_path is None:
            # Auto-detect: assume we're in the project root
            base_path = Path("victor")
            if not base_path.exists():
                # Try relative to this file
                base_path = Path(__file__).parent.parent.parent.parent

        results: dict[str, ValidationResult] = {}

        for vertical in self.VERTICALS:
            vertical_path = base_path / vertical / "tool_dependencies.yaml"
            if vertical_path.exists():
                results[vertical] = self.validate(vertical_path)
            else:
                logger.debug(f"No tool_dependencies.yaml found for {vertical}")

        return results


__all__ = [
    "ToolDependencyValidator",
    "ValidationResult",
    "validate_tool_dependency_yaml",
    "get_json_schema",
    "TOOL_DEPENDENCY_SCHEMA",
]
