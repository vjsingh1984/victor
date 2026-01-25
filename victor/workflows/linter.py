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

"""Workflow linter for YAML workflow validation.

This module provides a comprehensive linter for YAML workflow definitions.
It checks for syntax errors, schema violations, connection issues, best practices,
security concerns, and complexity metrics.

Example:
    from victor.workflows.linter import WorkflowLinter

    linter = WorkflowLinter()
    result = linter.lint_file("workflow.yaml")

    if result.has_errors:
        print(f"Found {result.error_count} errors")
        for issue in result.issues:
            print(f"  {issue}")

    # Generate report
    report = result.generate_report()
    print(report)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, cast

import yaml

from victor.workflows.validation_rules import (
    DEFAULT_RULES,
    RuleCategory,
    Severity,
    ValidationIssue,
    ValidationRule,
)


@dataclass
class LinterResult:
    """Result of linting a workflow.

    Attributes:
        issues: List of validation issues found
        files_checked: Number of files checked
        workflow_count: Number of workflows validated
        duration_seconds: Time taken to lint
    """

    issues: List[ValidationIssue] = field(default_factory=list)
    files_checked: int = 0
    workflow_count: int = 0
    duration_seconds: float = 0.0

    @property
    def error_count(self) -> int:
        """Count of ERROR severity issues."""
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of WARNING severity issues."""
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    @property
    def info_count(self) -> int:
        """Count of INFO severity issues."""
        return sum(1 for i in self.issues if i.severity == Severity.INFO)

    @property
    def suggestion_count(self) -> int:
        """Count of SUGGESTION severity issues."""
        return sum(1 for i in self.issues if i.severity == Severity.SUGGESTION)

    @property
    def has_errors(self) -> bool:
        """Whether any errors were found."""
        return self.error_count > 0

    @property
    def has_warnings(self) -> bool:
        """Whether any warnings were found."""
        return self.warning_count > 0

    @property
    def is_valid(self) -> bool:
        """Whether the workflow is valid (no errors)."""
        return not self.has_errors

    def get_issues_by_severity(self, severity: Severity) -> List[ValidationIssue]:
        """Get issues filtered by severity."""
        return [i for i in self.issues if i.severity == severity]

    def get_issues_by_category(self, category: RuleCategory) -> List[ValidationIssue]:
        """Get issues filtered by category."""
        return [i for i in self.issues if i.category == category]

    def get_issues_by_location(self, location: str) -> List[ValidationIssue]:
        """Get issues filtered by location."""
        return [i for i in self.issues if i.location == location]

    def generate_report(
        self,
        format: str = "text",
        include_suggestions: bool = True,
        include_context: bool = False,
    ) -> str:
        """Generate a formatted report.

        Args:
            format: Report format ('text', 'json', 'markdown')
            include_suggestions: Whether to include suggestions
            include_context: Whether to include context

        Returns:
            Formatted report string
        """
        if format == "json":
            return self._generate_json_report(include_suggestions, include_context)
        elif format == "markdown":
            return self._generate_markdown_report(include_suggestions, include_context)
        else:
            return self._generate_text_report(include_suggestions, include_context)

    def _generate_text_report(self, include_suggestions: bool, include_context: bool) -> str:
        """Generate text report."""
        lines = []
        lines.append("=" * 60)
        lines.append("Workflow Linting Report")
        lines.append("=" * 60)
        lines.append(f"Files checked: {self.files_checked}")
        lines.append(f"Workflows validated: {self.workflow_count}")
        lines.append(f"Duration: {self.duration_seconds:.2f}s")
        lines.append("")

        # Summary
        lines.append("Summary:")
        lines.append(f"  Errors: {self.error_count}")
        lines.append(f"  Warnings: {self.warning_count}")
        lines.append(f"  Info: {self.info_count}")
        lines.append(f"  Suggestions: {self.suggestion_count}")
        lines.append("")

        if not self.issues:
            lines.append("✓ No issues found!")
            return "\n".join(lines)

        # Group by severity
        lines.append("Issues:")
        lines.append("")

        for severity in [Severity.ERROR, Severity.WARNING, Severity.INFO, Severity.SUGGESTION]:
            issues = self.get_issues_by_severity(severity)
            if not issues:
                continue

            icon = {
                Severity.ERROR: "✗",
                Severity.WARNING: "⚠",
                Severity.INFO: "ℹ",
                Severity.SUGGESTION: "💡",
            }[severity]
            lines.append(f"{icon} {severity.value.upper()} ({len(issues)}):")
            lines.append("")

            for issue in issues:
                lines.append(f"  [{issue.rule_id}] {issue.message}")
                lines.append(f"    Location: {issue.location}")

                if include_suggestions and issue.suggestion:
                    lines.append(f"    Suggestion: {issue.suggestion}")

                if include_context and issue.context:
                    lines.append(f"    Context: {json.dumps(issue.context, indent=6)}")

                lines.append("")

        return "\n".join(lines)

    def _generate_markdown_report(self, include_suggestions: bool, include_context: bool) -> str:
        """Generate Markdown report."""
        lines = []
        lines.append("# Workflow Linting Report")
        lines.append("")
        lines.append(f"- **Files checked:** {self.files_checked}")
        lines.append(f"- **Workflows validated:** {self.workflow_count}")
        lines.append(f"- **Duration:** {self.duration_seconds:.2f}s")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Errors:** {self.error_count}")
        lines.append(f"- **Warnings:** {self.warning_count}")
        lines.append(f"- **Info:** {self.info_count}")
        lines.append(f"- **Suggestions:** {self.suggestion_count}")
        lines.append("")

        if not self.issues:
            lines.append("✓ **No issues found!**")
            return "\n".join(lines)

        # Group by severity
        lines.append("## Issues")
        lines.append("")

        for severity in [Severity.ERROR, Severity.WARNING, Severity.INFO, Severity.SUGGESTION]:
            issues = self.get_issues_by_severity(severity)
            if not issues:
                continue

            icon = {
                Severity.ERROR: "❌",
                Severity.WARNING: "⚠️",
                Severity.INFO: "ℹ️",
                Severity.SUGGESTION: "💡",
            }[severity]
            lines.append(f"### {icon} {severity.value.upper()} ({len(issues)})")
            lines.append("")

            for issue in issues:
                lines.append(f"**[{issue.rule_id}]** {issue.message}")
                lines.append("")
                lines.append(f"- **Location:** `{issue.location}`")
                lines.append(f"- **Category:** {issue.category.value}")

                if include_suggestions and issue.suggestion:
                    lines.append(f"- **Suggestion:** {issue.suggestion}")

                if include_context and issue.context:
                    lines.append(
                        f"- **Context:** ```json\n{json.dumps(issue.context, indent=2)}\n```"
                    )

                lines.append("")

        return "\n".join(lines)

    def _generate_json_report(self, include_suggestions: bool, include_context: bool) -> str:
        """Generate JSON report."""
        report = {
            "summary": {
                "files_checked": self.files_checked,
                "workflow_count": self.workflow_count,
                "duration_seconds": self.duration_seconds,
                "error_count": self.error_count,
                "warning_count": self.warning_count,
                "info_count": self.info_count,
                "suggestion_count": self.suggestion_count,
            },
            "issues": [
                issue.to_dict()
                for issue in self.issues
                if include_suggestions or not issue.suggestion
            ],
        }

        return json.dumps(report, indent=2)


class WorkflowLinter:
    """Comprehensive workflow linter.

    The linter validates YAML workflow files against a set of rules.
    Rules can be enabled/disabled and customized.

    Example:
        linter = WorkflowLinter()
        result = linter.lint_file("workflow.yaml")

        if result.is_valid:
            print("Workflow is valid!")
        else:
            print(f"Found {result.error_count} errors")

    Customization:
        linter = WorkflowLinter()
        linter.disable_rule("goal_quality")
        linter.enable_rule("custom_rule")
        linter.set_rule_severity("tool_budget", Severity.INFO)
    """

    def __init__(self, rules: Optional[List[ValidationRule]] = None):
        """Initialize the linter.

        Args:
            rules: List of validation rules (uses DEFAULT_RULES if not provided)
        """
        self.rules: List[ValidationRule] = rules or list(DEFAULT_RULES)
        self._rule_map: Dict[str, ValidationRule] = {rule.rule_id: rule for rule in self.rules}

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule.

        Args:
            rule: ValidationRule instance to add
        """
        if rule.rule_id in self._rule_map:
            # Replace existing rule
            self.rules = [r for r in self.rules if r.rule_id != rule.rule_id]

        self.rules.append(rule)
        self._rule_map[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> None:
        """Remove a validation rule.

        Args:
            rule_id: ID of rule to remove
        """
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
        self._rule_map.pop(rule_id, None)

    def enable_rule(self, rule_id: str) -> None:
        """Enable a validation rule.

        Args:
            rule_id: ID of rule to enable
        """
        if rule_id in self._rule_map:
            self._rule_map[rule_id].enabled = True

    def disable_rule(self, rule_id: str) -> None:
        """Disable a validation rule.

        Args:
            rule_id: ID of rule to disable
        """
        if rule_id in self._rule_map:
            self._rule_map[rule_id].enabled = False

    def set_rule_severity(self, rule_id: str, severity: Severity) -> None:
        """Change the severity of a rule.

        Args:
            rule_id: ID of rule to modify
            severity: New severity level
        """
        if rule_id in self._rule_map:
            self._rule_map[rule_id].severity = severity

    def lint_file(self, file_path: str | Path) -> LinterResult:
        """Lint a single workflow file.

        Args:
            file_path: Path to YAML workflow file

        Returns:
            LinterResult with validation issues
        """
        import time

        start_time = time.time()

        # Load workflow
        workflow = self._load_workflow(file_path)

        # Run validation
        issues = self._validate_workflow(workflow)

        duration = time.time() - start_time

        return LinterResult(
            issues=issues,
            files_checked=1,
            workflow_count=len(workflow.get("workflows", {})),
            duration_seconds=duration,
        )

    def lint_directory(
        self, directory: str | Path, pattern: str = "*.yaml", recursive: bool = False
    ) -> LinterResult:
        """Lint all workflow files in a directory.

        Args:
            directory: Path to directory
            pattern: File pattern to match (default: *.yaml)
            recursive: Whether to search subdirectories

        Returns:
            LinterResult with aggregated validation issues
        """
        import time

        start_time = time.time()

        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        # Find YAML files
        glob_pattern = f"**/{pattern}" if recursive else pattern
        yaml_files = list(dir_path.glob(glob_pattern))

        all_issues = []
        workflow_count = 0

        for yaml_file in yaml_files:
            try:
                workflow = self._load_workflow(yaml_file)
                issues = self._validate_workflow(workflow)
                all_issues.extend(issues)
                workflow_count += len(workflow.get("workflows", {}))
            except Exception as e:
                # Add error for file that couldn't be loaded
                all_issues.append(
                    ValidationIssue(
                        rule_id="file_load_error",
                        severity=Severity.ERROR,
                        category=RuleCategory.SYNTAX,
                        message=f"Failed to load file: {e}",
                        location=str(yaml_file),
                    )
                )

        duration = time.time() - start_time

        return LinterResult(
            issues=all_issues,
            files_checked=len(yaml_files),
            workflow_count=workflow_count,
            duration_seconds=duration,
        )

    def lint_dict(self, workflow: Dict[str, Any]) -> LinterResult:
        """Lint a workflow dictionary.

        Args:
            workflow: Workflow definition dictionary

        Returns:
            LinterResult with validation issues
        """
        import time

        start_time = time.time()

        issues = self._validate_workflow(workflow)

        duration = time.time() - start_time

        return LinterResult(
            issues=issues,
            files_checked=0,
            workflow_count=len(workflow.get("workflows", {})),
            duration_seconds=duration,
        )

    def _load_workflow(self, file_path: str | Path) -> Dict[str, Any]:
        """Load workflow from YAML file.

        Args:
            file_path: Path to YAML file

        Returns:
            Workflow dictionary

        Raises:
            ValueError: If file cannot be loaded or parsed
        """
        path = Path(file_path)

        if not path.exists():
            raise ValueError(f"File not found: {file_path}")

        if path.suffix not in {".yaml", ".yml"}:
            raise ValueError(f"Not a YAML file: {file_path}")

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
                return cast(Dict[str, Any], data)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}")

    def _validate_workflow(self, workflow: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate workflow against all enabled rules.

        Args:
            workflow: Workflow dictionary

        Returns:
            List of validation issues
        """
        all_issues = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            try:
                issues = rule.check(workflow)
                all_issues.extend(issues)
            except Exception as e:
                # Rule failed unexpectedly - add error and continue
                all_issues.append(
                    ValidationIssue(
                        rule_id=rule.rule_id,
                        severity=Severity.ERROR,
                        category=RuleCategory.SCHEMA,
                        message=f"Rule execution failed: {e}",
                        location="<unknown>",
                        context={"exception": str(e)},
                    )
                )

        # Sort issues: by severity, then category, then location
        severity_order = {
            Severity.ERROR: 0,
            Severity.WARNING: 1,
            Severity.INFO: 2,
            Severity.SUGGESTION: 3,
        }
        all_issues.sort(
            key=lambda i: (severity_order.get(i.severity, 99), i.category.value, i.location)
        )

        return all_issues

    def get_rules(self) -> List[ValidationRule]:
        """Get list of all rules.

        Returns:
            List of validation rules
        """
        return list(self.rules)

    def get_enabled_rules(self) -> List[ValidationRule]:
        """Get list of enabled rules.

        Returns:
            List of enabled validation rules
        """
        return [r for r in self.rules if r.enabled]

    def get_rule(self, rule_id: str) -> Optional[ValidationRule]:
        """Get a specific rule by ID.

        Args:
            rule_id: ID of rule to get

        Returns:
            ValidationRule if found, None otherwise
        """
        return self._rule_map.get(rule_id)


# Convenience functions
def lint_file(file_path: str | Path) -> LinterResult:
    """Lint a workflow file with default rules.

    Args:
        file_path: Path to YAML workflow file

    Returns:
        LinterResult with validation issues

    Example:
        result = lint_file("workflow.yaml")
        if result.is_valid:
            print("Valid!")
    """
    linter = WorkflowLinter()
    return linter.lint_file(file_path)


def lint_dict(workflow: Dict[str, Any]) -> LinterResult:
    """Lint a workflow dictionary with default rules.

    Args:
        workflow: Workflow definition dictionary

    Returns:
        LinterResult with validation issues

    Example:
        workflow = {"workflows": {...}}
        result = lint_dict(workflow)
    """
    linter = WorkflowLinter()
    return linter.lint_dict(workflow)


__all__ = [
    "WorkflowLinter",
    "LinterResult",
    "lint_file",
    "lint_dict",
]
