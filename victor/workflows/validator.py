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

"""Workflow validator.

Validates workflow structure and semantics using the comprehensive linter.
This module provides a high-level validation API that integrates with the
linter for detailed validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from victor.workflows.linter import WorkflowLinter, LinterResult, lint_file, lint_dict
from victor.workflows.validation_rules import Severity, ValidationRule

logger = logging.getLogger(__name__)


class WorkflowValidator:
    """Validates workflow structure and semantics.

    This class provides a high-level API for workflow validation.
    It uses the WorkflowLinter internally for comprehensive validation.

    Responsibility (SRP):
    - Validate workflow structure (nodes, edges)
    - Check node type compatibility
    - Validate node properties (tool_budget, timeout)
    - Check for cycles and unreachable nodes
    - Apply best practices and security checks

    Non-responsibility:
    - Workflow loading (handled by YAMLWorkflowLoader)
    - Workflow compilation (handled by WorkflowCompiler)
    - Detailed rule implementations (handled by ValidationRule subclasses)

    Example:
        validator = WorkflowValidator()

        # Validate file
        result = validator.validate_file("workflow.yaml")
        if result.is_valid:
            print("Workflow is valid!")

        # Validate dict
        workflow = {"workflows": {...}}
        result = validator.validate_dict(workflow)

        # Validate with custom rules
        validator = WorkflowValidator(strict_mode=True)
        validator.add_rule(custom_rule)
        result = validator.validate_file("workflow.yaml")
    """

    def __init__(self, strict_mode: bool = False, rules: Optional[list[ValidationRule]] = None):
        """Initialize the validator.

        Args:
            strict_mode: Whether to enable strict validation (enables all rules)
            rules: Custom validation rules (uses DEFAULT_RULES if not provided)
        """
        self._strict_mode = strict_mode

        # Initialize linter with rules
        if rules:
            self.linter = WorkflowLinter(rules=rules)
        else:
            self.linter = WorkflowLinter()

        # In strict mode, enable INFO and SUGGESTION rules
        if strict_mode:
            for rule in self.linter.get_rules():
                if rule.severity in {Severity.INFO, Severity.SUGGESTION}:
                    rule.enabled = True

    def validate(self, workflow_def: dict[str, Any]) -> bool:
        """Validate a workflow definition (legacy API).

        Args:
            workflow_def: Workflow definition dict

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        result = self.validate_dict(workflow_def)

        if result and not result.is_valid:
            errors = [issue.message for issue in result.issues if issue.severity == Severity.ERROR]
            raise ValueError(f"Workflow validation failed: {'; '.join(errors)}")

        return True

    def validate_file(self, file_path: str | Path) -> LinterResult:
        """Validate a workflow file.

        Args:
            file_path: Path to YAML workflow file

        Returns:
            ValidationResult with detailed validation results
        """
        return self.linter.lint_file(file_path)

    def validate_dict(self, workflow: dict[str, Any]) -> LinterResult:
        """Validate a workflow dictionary.

        Args:
            workflow: Workflow definition dictionary

        Returns:
            ValidationResult with detailed validation results
        """
        return self.linter.lint_dict(workflow)

    def validate_directory(
        self, directory: str | Path, pattern: str = "*.yaml", recursive: bool = False
    ) -> LinterResult:
        """Validate all workflow files in a directory.

        Args:
            directory: Path to directory
            pattern: File pattern to match (default: *.yaml)
            recursive: Whether to search subdirectories

        Returns:
            ValidationResult with aggregated validation results
        """
        return self.linter.lint_directory(directory, pattern, recursive)

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule.

        Args:
            rule: ValidationRule to add
        """
        self.linter.add_rule(rule)

    def remove_rule(self, rule_id: str) -> None:
        """Remove a validation rule.

        Args:
            rule_id: ID of rule to remove
        """
        self.linter.remove_rule(rule_id)

    def enable_rule(self, rule_id: str) -> None:
        """Enable a validation rule.

        Args:
            rule_id: ID of rule to enable
        """
        self.linter.enable_rule(rule_id)

    def disable_rule(self, rule_id: str) -> None:
        """Disable a validation rule.

        Args:
            rule_id: ID of rule to disable
        """
        self.linter.disable_rule(rule_id)

    def set_rule_severity(self, rule_id: str, severity: Severity) -> None:
        """Change the severity of a rule.

        Args:
            rule_id: ID of rule to modify
            severity: New severity level
        """
        self.linter.set_rule_severity(rule_id, severity)

    def get_rules(self) -> list[ValidationRule]:
        """Get list of all validation rules.

        Returns:
            List of ValidationRule instances
        """
        return self.linter.get_rules()

    def get_enabled_rules(self) -> list[ValidationRule]:
        """Get list of enabled validation rules.

        Returns:
            List of enabled ValidationRule instances
        """
        return self.linter.get_enabled_rules()


__all__ = [
    "WorkflowValidator",
    "lint_file",
    "lint_dict",
]
