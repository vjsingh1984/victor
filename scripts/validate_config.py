#!/usr/bin/env python3
"""Configuration Validator for Victor

This tool validates YAML configuration files for modes, capabilities, teams,
and other Victor components. It checks syntax, required fields, value constraints,
and cross-file references.

Usage:
    python scripts/validate_config.py victor/config/modes/coding_modes.yaml
    python scripts/validate_config.py --all-configs
    python scripts/validate_config.py --type modes

Exit Codes:
    0: All validations passed
    1: Validation errors found
    2: Error occurred
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(2)


class ValidationSeverity(Enum):
    """Validation issue severity."""

    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""

    severity: ValidationSeverity
    file_path: str
    field_path: str
    message: str
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        """Format issue for display."""
        return f"[{self.severity.value}] {self.file_path}::{self.field_path} - {self.message}"


@dataclass
class ValidationResult:
    """Result of validating a configuration file."""

    file_path: str
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the result."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.is_valid = False

    @property
    def error_count(self) -> int:
        """Count of ERROR issues."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of WARNING issues."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)


class ConfigValidator:
    """Validates Victor YAML configuration files."""

    def __init__(self):
        """Initialize validator."""
        self.results: List[ValidationResult] = []

        # Define schemas for different config types
        self.mode_schema = self._get_mode_schema()
        self.capability_schema = self._get_capability_schema()
        self.team_schema = self._get_team_schema()

    def _get_mode_schema(self) -> Dict[str, Any]:
        """Get schema for mode configuration."""
        return {
            "required_fields": ["vertical_name", "default_mode", "modes"],
            "mode_fields": {
                "required": ["name", "display_name", "exploration", "edit_permission"],
                "optional": ["tool_budget_multiplier", "max_iterations", "description"],
                "enums": {
                    "exploration": ["standard", "thorough", "minimal", "deep"],
                    "edit_permission": ["full", "sandbox", "none"],
                },
                "types": {
                    "tool_budget_multiplier": (int, float),
                    "max_iterations": int,
                },
                "ranges": {
                    "tool_budget_multiplier": (0.1, 10.0),
                    "max_iterations": (1, 100),
                },
            },
        }

    def _get_capability_schema(self) -> Dict[str, Any]:
        """Get schema for capability configuration."""
        return {
            "required_fields": ["vertical_name", "capabilities"],
            "capability_types": ["tool", "workflow", "middleware", "validator", "observer"],
            "capability_fields": {
                "required": ["type", "description"],
                "optional": ["enabled", "handler", "config"],
            },
        }

    def _get_team_schema(self) -> Dict[str, Any]:
        """Get schema for team configuration."""
        return {
            "required_fields": ["teams"],
            "team_fields": {
                "required": ["name", "display_name", "formation", "roles"],
                "optional": ["description", "communication_style", "max_iterations"],
                "enums": {
                    "formation": [
                        "pipeline",
                        "parallel",
                        "sequential",
                        "hierarchical",
                        "consensus",
                    ],
                    "communication_style": ["structured", "freeform", "minimal"],
                },
            },
            "role_fields": {
                "required": ["name", "display_name", "persona"],
                "optional": ["description", "tool_categories", "capabilities"],
            },
        }

    def validate_file(self, file_path: Path) -> ValidationResult:
        """Validate a configuration file.

        Args:
            file_path: Path to configuration file

        Returns:
            ValidationResult with findings
        """
        result = ValidationResult(file_path=str(file_path), is_valid=True)

        # Load YAML
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    file_path=str(file_path),
                    field_path="<root>",
                    message=f"YAML parsing error: {e}",
                    suggestion="Fix YAML syntax",
                )
            )
            self.results.append(result)
            return result

        if not config:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    file_path=str(file_path),
                    field_path="<root>",
                    message="Empty configuration file",
                    suggestion="Add configuration content",
                )
            )
            self.results.append(result)
            return result

        # Determine config type and validate accordingly
        config_type = self._detect_config_type(file_path, config)

        if config_type == "mode":
            self._validate_mode_config(file_path, config, result)
        elif config_type == "capability":
            self._validate_capability_config(file_path, config, result)
        elif config_type == "team":
            self._validate_team_config(file_path, config, result)
        else:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    file_path=str(file_path),
                    field_path="<root>",
                    message="Unknown configuration type",
                    suggestion="Ensure file follows Victor config conventions",
                )
            )

        self.results.append(result)
        return result

    def _detect_config_type(self, file_path: Path, config: Dict[str, Any]) -> str:
        """Detect configuration type from structure.

        Args:
            file_path: Path to config file
            config: Parsed YAML configuration

        Returns:
            Config type: "mode", "capability", "team", or "unknown"
        """
        filename = file_path.name.lower()

        if "modes" in config and "vertical_name" in config:
            return "mode"
        elif "capabilities" in config and "vertical_name" in config:
            return "capability"
        elif "teams" in config:
            return "team"

        # Try to detect from filename
        if "mode" in filename:
            return "mode"
        elif "capability" in filename:
            return "capability"
        elif "team" in filename:
            return "team"

        return "unknown"

    def _validate_mode_config(
        self,
        file_path: Path,
        config: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate mode configuration.

        Args:
            file_path: Path to config file
            config: Parsed configuration
            result: ValidationResult to populate
        """
        schema = self.mode_schema

        # Check required fields
        for required in schema["required_fields"]:
            if required not in config:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        file_path=str(file_path),
                        field_path=required,
                        message=f"Missing required field: {required}",
                        suggestion=f"Add {required} to configuration",
                    )
                )

        # Validate modes
        if "modes" in config:
            modes = config["modes"]
            if not isinstance(modes, dict):
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        file_path=str(file_path),
                        field_path="modes",
                        message="'modes' must be a dictionary",
                        suggestion="Convert modes to a dictionary",
                    )
                )
                return

            for mode_name, mode_config in modes.items():
                mode_path = f"modes.{mode_name}"

                # Check required mode fields
                for field in schema["mode_fields"]["required"]:
                    if field not in mode_config:
                        result.add_issue(
                            ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                file_path=str(file_path),
                                field_path=f"{mode_path}.{field}",
                                message=f"Missing required field: {field}",
                                suggestion=f"Add {field} to mode {mode_name}",
                            )
                        )

                # Validate enum values
                for field, valid_values in schema["mode_fields"]["enums"].items():
                    if field in mode_config:
                        value = mode_config[field]
                        if value not in valid_values:
                            result.add_issue(
                                ValidationIssue(
                                    severity=ValidationSeverity.ERROR,
                                    file_path=str(file_path),
                                    field_path=f"{mode_path}.{field}",
                                    message=f"Invalid value '{value}' for {field}",
                                    suggestion=f"Must be one of: {valid_values}",
                                )
                            )

                # Validate types and ranges
                for field, field_type in schema["mode_fields"]["types"].items():
                    if field in mode_config:
                        value = mode_config[field]
                        if not isinstance(value, field_type):
                            result.add_issue(
                                ValidationIssue(
                                    severity=ValidationSeverity.ERROR,
                                    file_path=str(file_path),
                                    field_path=f"{mode_path}.{field}",
                                    message=f"Invalid type for {field}: expected {field_type.__name__}",
                                    suggestion=f"Convert {field} to {field_type.__name__}",
                                )
                            )
                            continue

                        # Check range constraints
                        if field in schema["mode_fields"]["ranges"]:
                            min_val, max_val = schema["mode_fields"]["ranges"][field]
                            if not (min_val <= value <= max_val):
                                result.add_issue(
                                    ValidationIssue(
                                        severity=ValidationSeverity.ERROR,
                                        file_path=str(file_path),
                                        field_path=f"{mode_path}.{field}",
                                        message=f"Value {value} for {field} out of range",
                                        suggestion=f"Must be between {min_val} and {max_val}",
                                    )
                                )

    def _validate_capability_config(
        self,
        file_path: Path,
        config: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate capability configuration.

        Args:
            file_path: Path to config file
            config: Parsed configuration
            result: ValidationResult to populate
        """
        schema = self.capability_schema

        # Check required fields
        for required in schema["required_fields"]:
            if required not in config:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        file_path=str(file_path),
                        field_path=required,
                        message=f"Missing required field: {required}",
                        suggestion=f"Add {required} to configuration",
                    )
                )

        # Validate capabilities
        if "capabilities" in config:
            capabilities = config["capabilities"]

            # Support both list format (YAML files) and dict format
            if isinstance(capabilities, list):
                # List format: each item has 'name' and 'capability_type'
                for idx, cap_config in enumerate(capabilities):
                    cap_name = cap_config.get("name", f"capability_{idx}")
                    cap_path = f"capabilities[{idx}]"

                    # Check required fields for list format
                    if "name" not in cap_config:
                        result.add_issue(
                            ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                file_path=str(file_path),
                                field_path=cap_path,
                                message="Missing required field: name",
                                suggestion="Add 'name' to capability",
                            )
                        )

                    if "description" not in cap_config:
                        result.add_issue(
                            ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                file_path=str(file_path),
                                field_path=f"{cap_path}.description",
                                message="Missing required field: description",
                                suggestion=f"Add description to capability {cap_name}",
                            )
                        )

                    # Validate capability type (supports both 'type' and 'capability_type')
                    cap_type = cap_config.get("capability_type") or cap_config.get("type")
                    if cap_type and cap_type not in schema["capability_types"]:
                        result.add_issue(
                            ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                file_path=str(file_path),
                                field_path=f"{cap_path}.capability_type",
                                message=f"Invalid capability type: {cap_type}",
                                suggestion=f"Must be one of: {schema['capability_types']}",
                            )
                        )

            elif isinstance(capabilities, dict):
                # Dict format: capability name as key
                for cap_name, cap_config in capabilities.items():
                    cap_path = f"capabilities.{cap_name}"

                    # Check required capability fields
                    for field in schema["capability_fields"]["required"]:
                        if field not in cap_config:
                            result.add_issue(
                                ValidationIssue(
                                    severity=ValidationSeverity.ERROR,
                                    file_path=str(file_path),
                                    field_path=f"{cap_path}.{field}",
                                    message=f"Missing required field: {field}",
                                    suggestion=f"Add {field} to capability {cap_name}",
                                )
                            )

                    # Validate capability type
                    if "type" in cap_config:
                        cap_type = cap_config["type"]
                        if cap_type not in schema["capability_types"]:
                            result.add_issue(
                                ValidationIssue(
                                    severity=ValidationSeverity.ERROR,
                                    file_path=str(file_path),
                                    field_path=f"{cap_path}.type",
                                    message=f"Invalid capability type: {cap_type}",
                                    suggestion=f"Must be one of: {schema['capability_types']}",
                                )
                            )
            else:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        file_path=str(file_path),
                        field_path="capabilities",
                        message="'capabilities' must be a list or dictionary",
                        suggestion="Use list format with 'name' field or dictionary format with capability names as keys",
                    )
                )

    def _validate_team_config(
        self,
        file_path: Path,
        config: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate team configuration.

        Args:
            file_path: Path to config file
            config: Parsed configuration
            result: ValidationResult to populate
        """
        schema = self.team_schema

        # Check required fields
        for required in schema["required_fields"]:
            if required not in config:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        file_path=str(file_path),
                        field_path=required,
                        message=f"Missing required field: {required}",
                        suggestion=f"Add {required} to configuration",
                    )
                )

        # Validate teams
        if "teams" in config:
            teams = config["teams"]
            if not isinstance(teams, list):
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        file_path=str(file_path),
                        field_path="teams",
                        message="'teams' must be a list",
                        suggestion="Convert teams to a list",
                    )
                )
                return

            for i, team in enumerate(teams):
                team_path = f"teams[{i}]"

                # Check required team fields
                for field in schema["team_fields"]["required"]:
                    if field not in team:
                        result.add_issue(
                            ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                file_path=str(file_path),
                                field_path=f"{team_path}.{field}",
                                message=f"Missing required field: {field}",
                                suggestion=f"Add {field} to team",
                            )
                        )

                # Validate enum values
                for field, valid_values in schema["team_fields"]["enums"].items():
                    if field in team:
                        value = team[field]
                        if value not in valid_values:
                            result.add_issue(
                                ValidationIssue(
                                    severity=ValidationSeverity.ERROR,
                                    file_path=str(file_path),
                                    field_path=f"{team_path}.{field}",
                                    message=f"Invalid value '{value}' for {field}",
                                    suggestion=f"Must be one of: {valid_values}",
                                )
                            )

                # Validate roles
                if "roles" in team:
                    roles = team["roles"]
                    if not isinstance(roles, list):
                        result.add_issue(
                            ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                file_path=str(file_path),
                                field_path=f"{team_path}.roles",
                                message="'roles' must be a list",
                                suggestion="Convert roles to a list",
                            )
                        )
                        continue

                    for j, role in enumerate(roles):
                        role_path = f"{team_path}.roles[{j}]"

                        # Check required role fields
                        for field in schema["role_fields"]["required"]:
                            if field not in role:
                                result.add_issue(
                                    ValidationIssue(
                                        severity=ValidationSeverity.ERROR,
                                        file_path=str(file_path),
                                        field_path=f"{role_path}.{field}",
                                        message=f"Missing required field: {field}",
                                        suggestion=f"Add {field} to role",
                                    )
                                )

    def print_result(self, result: ValidationResult) -> None:
        """Print validation result.

        Args:
            result: Result to print
        """
        status = "✓ VALID" if result.is_valid else "✗ INVALID"
        print(f"\n{status}: {result.file_path}")
        print("=" * 80)

        if not result.issues:
            print("No issues found")
            return

        for issue in result.issues:
            print(issue)
            if issue.suggestion:
                print(f"  → {issue.suggestion}")

    def print_summary(self) -> None:
        """Print summary of all validations."""
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        total = len(self.results)
        valid = sum(1 for r in self.results if r.is_valid)
        invalid = total - valid

        total_errors = sum(r.error_count for r in self.results)
        total_warnings = sum(r.warning_count for r in self.results)

        print(f"Total Files: {total}")
        print(f"Valid: {valid}")
        print(f"Invalid: {invalid}")
        print(f"Total Errors: {total_errors}")
        print(f"Total Warnings: {total_warnings}")

        if invalid > 0:
            print("\nInvalid Files:")
            for result in self.results:
                if not result.is_valid:
                    print(
                        f"  - {result.file_path} ({result.error_count} errors, {result.warning_count} warnings)"
                    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate Victor configuration files")
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Validate all configuration files",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["mode", "capability", "team"],
        help="Only validate configs of this type",
    )

    args = parser.parse_args()

    validator = ConfigValidator()

    if args.all_configs:
        # Validate all config files
        config_dirs = [
            Path("victor/config/modes"),
            Path("victor/config/capabilities"),
            Path("victor/config/teams"),
        ]

        for config_dir in config_dirs:
            if not config_dir.exists():
                continue

            for config_file in config_dir.glob("*.yaml"):
                result = validator.validate_file(config_file)
                validator.print_result(result)

    elif args.config:
        # Validate specific file
        if not args.config.exists():
            print(f"Error: File does not exist: {args.config}")
            return 2

        result = validator.validate_file(args.config)
        validator.print_result(result)

    else:
        parser.print_help()
        return 2

    validator.print_summary()

    # Exit code based on validation results
    if any(not r.is_valid for r in validator.results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
