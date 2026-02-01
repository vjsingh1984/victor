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

"""Tests for error_reporter module.

Tests the error reporting and aggregation functionality:
- ErrorReport class
- ErrorReporter class
- Multiple report formats (human, LLM, JSON, Markdown, compact)
- Error aggregation by category, severity, and node
- Error prioritization
"""

import json

from victor.workflows.generation.types import (
    ErrorSeverity,
    ErrorCategory,
    WorkflowValidationError,
    WorkflowGenerationValidationResult,
)
from victor.workflows.generation.error_reporter import ErrorReporter, ErrorReport


class TestErrorReport:
    """Tests for ErrorReport class."""

    def test_error_report_initialization(self):
        """Test ErrorReport initialization with all parameters."""
        # Create validation result with errors
        schema_error = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.CRITICAL,
            message="Missing required field",
            location="nodes[0]",
            suggestion="Add 'type' field",
        )

        structure_error = WorkflowValidationError(
            category=ErrorCategory.STRUCTURE,
            severity=ErrorSeverity.ERROR,
            message="Invalid edge reference",
            location="edges[0]",
            suggestion="Fix edge target",
        )

        result = WorkflowGenerationValidationResult(
            is_valid=False,
            schema_errors=[schema_error],
            structure_errors=[structure_error],
        )

        errors_by_category = {
            "schema": [schema_error],
            "structure": [structure_error],
            "semantic": [],
            "security": [],
        }

        errors_by_severity = {
            "critical": [schema_error],
            "error": [structure_error],
            "warning": [],
            "info": [],
        }

        errors_by_node = {
            "0": [schema_error],
            "workflow": [structure_error],
        }

        report = ErrorReport(
            result=result,
            errors_by_category=errors_by_category,
            errors_by_severity=errors_by_severity,
            errors_by_node=errors_by_node,
        )

        assert report.total_errors == 2
        assert report.critical_count == 1
        assert report.error_count == 1
        assert report.warning_count == 0
        assert report.info_count == 0
        assert report.result == result
        assert report.errors_by_category == errors_by_category
        assert report.errors_by_severity == errors_by_severity
        assert report.errors_by_node == errors_by_node

    def test_error_report_empty_errors(self):
        """Test ErrorReport with no errors."""
        result = WorkflowGenerationValidationResult(is_valid=True)

        errors_by_category = {
            "schema": [],
            "structure": [],
            "semantic": [],
            "security": [],
        }

        errors_by_severity = {
            "critical": [],
            "error": [],
            "warning": [],
            "info": [],
        }

        errors_by_node = {}

        report = ErrorReport(
            result=result,
            errors_by_category=errors_by_category,
            errors_by_severity=errors_by_severity,
            errors_by_node=errors_by_node,
        )

        assert report.total_errors == 0
        assert report.critical_count == 0
        assert report.error_count == 0
        assert report.warning_count == 0
        assert report.info_count == 0


class TestErrorReporterAggregateErrors:
    """Tests for ErrorReporter.aggregate_errors method."""

    def test_aggregate_errors_all_categories(self):
        """Test error aggregation across all categories."""
        reporter = ErrorReporter()

        schema_error = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.CRITICAL,
            message="Schema error",
            location="nodes[0]",
        )

        structure_error = WorkflowValidationError(
            category=ErrorCategory.STRUCTURE,
            severity=ErrorSeverity.ERROR,
            message="Structure error",
            location="edges[0]",
        )

        semantic_error = WorkflowValidationError(
            category=ErrorCategory.SEMANTIC,
            severity=ErrorSeverity.WARNING,
            message="Semantic error",
            location="nodes[1]",
        )

        security_error = WorkflowValidationError(
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.ERROR,
            message="Security error",
            location="nodes[2]",
        )

        result = WorkflowGenerationValidationResult(
            is_valid=False,
            schema_errors=[schema_error],
            structure_errors=[structure_error],
            semantic_errors=[semantic_error],
            security_errors=[security_error],
        )

        report = reporter.aggregate_errors(result)

        # Check category grouping
        assert len(report.errors_by_category["schema"]) == 1
        assert len(report.errors_by_category["structure"]) == 1
        assert len(report.errors_by_category["semantic"]) == 1
        assert len(report.errors_by_category["security"]) == 1

        # Check severity grouping
        assert len(report.errors_by_severity["critical"]) == 1
        assert len(report.errors_by_severity["error"]) == 2
        assert len(report.errors_by_severity["warning"]) == 1
        assert len(report.errors_by_severity["info"]) == 0

        # Check counts
        assert report.total_errors == 4
        assert report.critical_count == 1
        assert report.error_count == 2
        assert report.warning_count == 1

    def test_aggregate_errors_by_node(self):
        """Test error aggregation by node."""
        reporter = ErrorReporter()

        node1_error = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.ERROR,
            message="Node 1 error",
            location="nodes[node1]",
        )

        node2_error = WorkflowValidationError(
            category=ErrorCategory.SEMANTIC,
            severity=ErrorSeverity.WARNING,
            message="Node 2 error",
            location="nodes[node2]",
        )

        workflow_error = WorkflowValidationError(
            category=ErrorCategory.STRUCTURE,
            severity=ErrorSeverity.ERROR,
            message="Workflow error",
            location="entry_point",
        )

        result = WorkflowGenerationValidationResult(
            is_valid=False,
            schema_errors=[node1_error],
            semantic_errors=[node2_error],
            structure_errors=[workflow_error],
        )

        report = reporter.aggregate_errors(result)

        # Check node grouping
        assert "node1" in report.errors_by_node
        assert "node2" in report.errors_by_node
        assert "workflow" in report.errors_by_node

        assert len(report.errors_by_node["node1"]) == 1
        assert len(report.errors_by_node["node2"]) == 1
        assert len(report.errors_by_node["workflow"]) == 1


class TestErrorReporterHumanReport:
    """Tests for ErrorReporter.human_report method."""

    def test_human_report_valid_no_errors(self):
        """Test human report for valid workflow with no errors."""
        reporter = ErrorReporter()
        result = WorkflowGenerationValidationResult(is_valid=True, workflow_name="test_workflow")

        report = reporter.human_report(result)

        assert "VALID" in report
        assert "test_workflow" in report
        assert "All checks passed" in report
        assert "✓" in report

    def test_human_report_valid_with_warnings(self):
        """Test human report for valid workflow with warnings."""
        reporter = ErrorReporter()

        warning = WorkflowValidationError(
            category=ErrorCategory.SEMANTIC,
            severity=ErrorSeverity.WARNING,
            message="This is a warning",
            location="nodes[0]",
            suggestion="Consider fixing this",
        )

        result = WorkflowGenerationValidationResult(is_valid=True, semantic_errors=[warning])

        report = reporter.human_report(result)

        assert "VALID" in report
        assert "1 warnings" in report
        assert "⚠️" in report

    def test_human_report_invalid_with_errors(self):
        """Test human report for invalid workflow."""
        reporter = ErrorReporter()

        critical_error = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.CRITICAL,
            message="Critical error",
            location="nodes[0]",
            suggestion="Fix this",
        )

        error = WorkflowValidationError(
            category=ErrorCategory.STRUCTURE,
            severity=ErrorSeverity.ERROR,
            message="Regular error",
            location="edges[0]",
            suggestion="Fix that",
        )

        warning = WorkflowValidationError(
            category=ErrorCategory.SEMANTIC,
            severity=ErrorSeverity.WARNING,
            message="Warning message",
            location="nodes[1]",
        )

        result = WorkflowGenerationValidationResult(
            is_valid=False,
            schema_errors=[critical_error],
            structure_errors=[error],
            semantic_errors=[warning],
        )

        report = reporter.human_report(result)

        assert "INVALID" in report
        assert "CRITICAL ERRORS" in report
        assert "ERRORS" in report
        assert "WARNINGS" in report
        assert "🚨" in report
        assert "❌" in report
        assert "⚠️" in report
        assert "Total Issues: 3" in report

    def test_human_report_with_suggestions(self):
        """Test human report includes suggestions when enabled."""
        reporter = ErrorReporter()

        error = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.ERROR,
            message="Error message",
            location="nodes[0]",
            suggestion="Fix suggestion",
        )

        result = WorkflowGenerationValidationResult(is_valid=False, schema_errors=[error])

        report = reporter.human_report(result, show_suggestions=True)

        assert "💡" in report
        assert "Fix suggestion" in report

    def test_human_report_without_suggestions(self):
        """Test human report excludes suggestions when disabled."""
        reporter = ErrorReporter()

        error = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.ERROR,
            message="Error message",
            location="nodes[0]",
            suggestion="Fix suggestion",
        )

        result = WorkflowGenerationValidationResult(is_valid=False, schema_errors=[error])

        report = reporter.human_report(result, show_suggestions=False)

        assert "💡" not in report
        assert "Fix suggestion" not in report

    def test_human_report_with_context(self):
        """Test human report includes context when enabled."""
        reporter = ErrorReporter()

        error = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.ERROR,
            message="Error message",
            location="nodes[0]",
            context={"field": "type", "value": "invalid"},
        )

        result = WorkflowGenerationValidationResult(is_valid=False, schema_errors=[error])

        report = reporter.human_report(result, show_context=True)

        assert "Context:" in report
        assert "field" in report

    def test_human_report_without_workflow_name(self):
        """Test human report without workflow name."""
        reporter = ErrorReporter()

        result = WorkflowGenerationValidationResult(is_valid=True)

        report = reporter.human_report(result)

        assert "VALID" in report
        # Should not show workflow line
        assert "Workflow:" not in report


class TestErrorReporterLLMReport:
    """Tests for ErrorReporter.llm_report method."""

    def test_llm_report_valid(self):
        """Test LLM report for valid workflow."""
        reporter = ErrorReporter()
        result = WorkflowGenerationValidationResult(is_valid=True)

        report = reporter.llm_report(result)

        assert "Validation passed" in report
        assert "No changes required" in report

    def test_llm_report_valid_with_warnings(self):
        """Test LLM report for valid workflow with warnings."""
        reporter = ErrorReporter()

        warning = WorkflowValidationError(
            category=ErrorCategory.SEMANTIC,
            severity=ErrorSeverity.WARNING,
            message="Warning message",
            location="nodes[0]",
        )

        result = WorkflowGenerationValidationResult(is_valid=True, semantic_errors=[warning])

        report = reporter.llm_report(result)

        assert "Validation passed" in report
        assert "1 warnings" in report

    def test_llm_report_invalid(self):
        """Test LLM report for invalid workflow."""
        reporter = ErrorReporter()

        schema_error = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.CRITICAL,
            message="Schema error",
            location="nodes[0]",
            suggestion="Fix schema",
        )

        structure_error = WorkflowValidationError(
            category=ErrorCategory.STRUCTURE,
            severity=ErrorSeverity.ERROR,
            message="Structure error",
            location="edges[0]",
        )

        result = WorkflowGenerationValidationResult(
            is_valid=False,
            schema_errors=[schema_error],
            structure_errors=[structure_error],
        )

        report = reporter.llm_report(result)

        assert "VALIDATION FAILED" in report
        assert "SCHEMA ERRORS" in report
        assert "STRUCTURE ERRORS" in report
        assert "Location:" in report
        assert "Error:" in report

    def test_llm_report_with_fixes(self):
        """Test LLM report includes fixes when enabled."""
        reporter = ErrorReporter()

        error = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.ERROR,
            message="Error message",
            location="nodes[0]",
            suggestion="Fix this",
        )

        result = WorkflowGenerationValidationResult(is_valid=False, schema_errors=[error])

        report = reporter.llm_report(result, include_fixes=True)

        assert "Fix:" in report
        assert "Fix this" in report

    def test_llm_report_without_fixes(self):
        """Test LLM report excludes fixes when disabled."""
        reporter = ErrorReporter()

        error = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.ERROR,
            message="Error message",
            location="nodes[0]",
            suggestion="Fix this",
        )

        result = WorkflowGenerationValidationResult(is_valid=False, schema_errors=[error])

        report = reporter.llm_report(result, include_fixes=False)

        assert "Fix:" not in report
        assert "Fix this" not in report

    def test_llm_report_with_context(self):
        """Test LLM report includes context when enabled."""
        reporter = ErrorReporter()

        error = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.ERROR,
            message="Error message",
            location="nodes[0]",
            context={"node_id": "test"},
        )

        result = WorkflowGenerationValidationResult(is_valid=False, schema_errors=[error])

        report = reporter.llm_report(result, include_context=True)

        assert "Context:" in report

    def test_llm_report_max_errors_limit(self):
        """Test LLM report respects max_errors_per_category limit."""
        reporter = ErrorReporter()

        errors = [
            WorkflowValidationError(
                category=ErrorCategory.SCHEMA,
                severity=ErrorSeverity.ERROR,
                message=f"Error {i}",
                location=f"nodes[{i}]",
            )
            for i in range(25)
        ]

        result = WorkflowGenerationValidationResult(is_valid=False, schema_errors=errors)

        report = reporter.llm_report(result, max_errors_per_category=20)

        # 25 total - 20 shown = 5 more
        assert "5 more schema errors" in report

    def test_llm_report_all_error_types(self):
        """Test LLM report includes all error categories."""
        reporter = ErrorReporter()

        schema_error = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.ERROR,
            message="Schema error",
            location="nodes[0]",
        )

        structure_error = WorkflowValidationError(
            category=ErrorCategory.STRUCTURE,
            severity=ErrorSeverity.ERROR,
            message="Structure error",
            location="edges[0]",
        )

        semantic_error = WorkflowValidationError(
            category=ErrorCategory.SEMANTIC,
            severity=ErrorSeverity.ERROR,
            message="Semantic error",
            location="nodes[1]",
        )

        security_error = WorkflowValidationError(
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.ERROR,
            message="Security error",
            location="nodes[2]",
        )

        result = WorkflowGenerationValidationResult(
            is_valid=False,
            schema_errors=[schema_error],
            structure_errors=[structure_error],
            semantic_errors=[semantic_error],
            security_errors=[security_error],
        )

        report = reporter.llm_report(result)

        assert "SCHEMA ERRORS" in report
        assert "STRUCTURE ERRORS" in report
        assert "SEMANTIC ERRORS" in report
        assert "SECURITY ERRORS" in report
        assert "SUMMARY" in report
        assert "CRITICAL: You must fix" in report


class TestErrorReporterJSONReport:
    """Tests for ErrorReporter.json_report method."""

    def test_json_report_valid(self):
        """Test JSON report for valid workflow."""
        reporter = ErrorReporter()
        result = WorkflowGenerationValidationResult(is_valid=True, workflow_name="test_workflow")

        report = reporter.json_report(result)

        # Should be valid JSON
        data = json.loads(report)

        assert data["is_valid"] is True
        assert data["workflow_name"] == "test_workflow"
        assert "summary" in data
        assert "error_counts" in data
        assert "category_counts" in data

    def test_json_report_invalid(self):
        """Test JSON report for invalid workflow."""
        reporter = ErrorReporter()

        error = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.ERROR,
            message="Error message",
            location="nodes[0]",
            suggestion="Fix this",
        )

        result = WorkflowGenerationValidationResult(
            is_valid=False,
            schema_errors=[error],
            workflow_name="test_workflow",
        )

        report = reporter.json_report(result)

        # Should be valid JSON
        data = json.loads(report)

        assert data["is_valid"] is False
        assert len(data["errors"]["schema"]) == 1
        assert data["errors"]["schema"][0]["message"] == "Error message"
        assert data["errors"]["schema"][0]["location"] == "nodes[0]"
        assert data["errors"]["schema"][0]["suggestion"] == "Fix this"

    def test_json_report_custom_indent(self):
        """Test JSON report with custom indentation."""
        reporter = ErrorReporter()
        result = WorkflowGenerationValidationResult(is_valid=True)

        report_indent_4 = reporter.json_report(result, indent=4)
        report_indent_0 = reporter.json_report(result, indent=0)

        # indent=4 should have more spaces
        assert "    " in report_indent_4
        assert "    " not in report_indent_0


class TestErrorReporterMarkdownReport:
    """Tests for ErrorReporter.markdown_report method."""

    def test_markdown_report_valid(self):
        """Test markdown report for valid workflow."""
        reporter = ErrorReporter()
        result = WorkflowGenerationValidationResult(is_valid=True, workflow_name="test_workflow")

        report = reporter.markdown_report(result)

        assert "# Workflow Validation Report:" in report
        assert "**VALID**" in report
        assert "**Workflow:** test_workflow" in report
        assert "## Summary" in report

    def test_markdown_report_valid_with_warnings(self):
        """Test markdown report for valid workflow with warnings."""
        reporter = ErrorReporter()

        warning = WorkflowValidationError(
            category=ErrorCategory.SEMANTIC,
            severity=ErrorSeverity.WARNING,
            message="Warning message",
            location="nodes[0]",
            suggestion="Fix this",
        )

        result = WorkflowGenerationValidationResult(is_valid=True, semantic_errors=[warning])

        report = reporter.markdown_report(result, include_suggestions=True)

        assert "## Warnings" in report
        assert "Warning message" in report
        assert "*Suggestion*" in report
        assert "Fix this" in report

    def test_markdown_report_invalid(self):
        """Test markdown report for invalid workflow."""
        reporter = ErrorReporter()

        critical_error = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.CRITICAL,
            message="Critical error",
            location="nodes[0]",
            suggestion="Fix critical",
        )

        error = WorkflowValidationError(
            category=ErrorCategory.STRUCTURE,
            severity=ErrorSeverity.ERROR,
            message="Regular error",
            location="edges[0]",
            suggestion="Fix error",
        )

        warning = WorkflowValidationError(
            category=ErrorCategory.SEMANTIC,
            severity=ErrorSeverity.WARNING,
            message="Warning",
            location="nodes[1]",
        )

        result = WorkflowGenerationValidationResult(
            is_valid=False,
            schema_errors=[critical_error],
            structure_errors=[error],
            semantic_errors=[warning],
        )

        report = reporter.markdown_report(result, include_suggestions=True)

        assert "**INVALID**" in report
        assert "## Critical Errors" in report
        assert "## Errors" in report
        assert "## Warnings" in report
        assert "### nodes[0]" in report
        assert "**Suggestion:**" in report

    def test_markdown_report_without_suggestions(self):
        """Test markdown report excludes suggestions when disabled."""
        reporter = ErrorReporter()

        error = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.ERROR,
            message="Error message",
            location="nodes[0]",
            suggestion="Fix this",
        )

        result = WorkflowGenerationValidationResult(is_valid=False, schema_errors=[error])

        report = reporter.markdown_report(result, include_suggestions=False)

        assert "**Suggestion:**" not in report
        assert "Fix this" not in report


class TestErrorReporterCompactReport:
    """Tests for ErrorReporter.compact_report method."""

    def test_compact_report_valid(self):
        """Test compact report for valid workflow."""
        reporter = ErrorReporter()
        result = WorkflowGenerationValidationResult(is_valid=True)

        report = reporter.compact_report(result)

        assert report == "✓ Valid"

    def test_compact_report_invalid(self):
        """Test compact report for invalid workflow."""
        reporter = ErrorReporter()

        critical_error = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.CRITICAL,
            message="Critical",
            location="nodes[0]",
        )

        error = WorkflowValidationError(
            category=ErrorCategory.STRUCTURE,
            severity=ErrorSeverity.ERROR,
            message="Error",
            location="edges[0]",
        )

        warning = WorkflowValidationError(
            category=ErrorCategory.SEMANTIC,
            severity=ErrorSeverity.WARNING,
            message="Warning",
            location="nodes[1]",
        )

        result = WorkflowGenerationValidationResult(
            is_valid=False,
            schema_errors=[critical_error],
            structure_errors=[error],
            semantic_errors=[warning],
        )

        report = reporter.compact_report(result)

        assert "✗" in report
        assert "1 critical" in report
        assert "1 errors" in report
        assert "1 warnings" in report


class TestErrorReporterNodeReport:
    """Tests for ErrorReporter.node_report method."""

    def test_node_report_no_errors(self):
        """Test node report for node with no errors."""
        reporter = ErrorReporter()
        result = WorkflowGenerationValidationResult(is_valid=True)

        report = reporter.node_report(result, "node1")

        assert "No errors" in report
        assert "node1" in report

    def test_node_report_with_errors(self):
        """Test node report for node with errors."""
        reporter = ErrorReporter()

        critical_error = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.CRITICAL,
            message="Critical error",
            location="nodes[node1]",
            suggestion="Fix critical",
        )

        warning = WorkflowValidationError(
            category=ErrorCategory.SEMANTIC,
            severity=ErrorSeverity.WARNING,
            message="Warning",
            location="nodes[node1]",
            suggestion="Fix warning",
        )

        result = WorkflowGenerationValidationResult(
            is_valid=False,
            schema_errors=[critical_error],
            semantic_errors=[warning],
        )

        report = reporter.node_report(result, "node1")

        assert "node1" in report
        assert "Errors:" in report
        assert "CRITICAL" in report
        assert "WARNING" in report
        assert "🚨" in report
        assert "⚠️" in report
        assert "💡" in report


class TestErrorReporterCategoryReport:
    """Tests for ErrorReporter.category_report method."""

    def test_category_report_no_errors(self):
        """Test category report for category with no errors."""
        reporter = ErrorReporter()
        result = WorkflowGenerationValidationResult(is_valid=True)

        report = reporter.category_report(result, ErrorCategory.SCHEMA)

        assert "Schema validation: No errors" in report

    def test_category_report_with_errors(self):
        """Test category report for category with errors."""
        reporter = ErrorReporter()

        error1 = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.ERROR,
            message="Error 1",
            location="nodes[0]",
            suggestion="Fix 1",
        )

        error2 = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.WARNING,
            message="Error 2",
            location="nodes[1]",
            suggestion="Fix 2",
        )

        result = WorkflowGenerationValidationResult(is_valid=False, schema_errors=[error1, error2])

        report = reporter.category_report(result, ErrorCategory.SCHEMA)

        assert "Schema Errors (2):" in report
        assert "nodes[0]" in report
        assert "Error 1" in report
        assert "Fix 1" in report
        assert "💡" in report

    def test_category_report_all_categories(self):
        """Test category report for all error categories."""
        reporter = ErrorReporter()

        schema_error = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.ERROR,
            message="Schema",
            location="nodes[0]",
        )

        structure_error = WorkflowValidationError(
            category=ErrorCategory.STRUCTURE,
            severity=ErrorSeverity.ERROR,
            message="Structure",
            location="edges[0]",
        )

        semantic_error = WorkflowValidationError(
            category=ErrorCategory.SEMANTIC,
            severity=ErrorSeverity.ERROR,
            message="Semantic",
            location="nodes[1]",
        )

        security_error = WorkflowValidationError(
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.ERROR,
            message="Security",
            location="nodes[2]",
        )

        result = WorkflowGenerationValidationResult(
            is_valid=False,
            schema_errors=[schema_error],
            structure_errors=[structure_error],
            semantic_errors=[semantic_error],
            security_errors=[security_error],
        )

        # Test each category
        schema_report = reporter.category_report(result, ErrorCategory.SCHEMA)
        assert "Schema Errors (1):" in schema_report

        structure_report = reporter.category_report(result, ErrorCategory.STRUCTURE)
        assert "Structure Errors (1):" in structure_report

        semantic_report = reporter.category_report(result, ErrorCategory.SEMANTIC)
        assert "Semantic Errors (1):" in semantic_report

        security_report = reporter.category_report(result, ErrorCategory.SECURITY)
        assert "Security Errors (1):" in security_report


class TestErrorReporterPrioritizeErrors:
    """Tests for ErrorReporter.prioritize_errors method."""

    def test_prioritize_errors_by_severity(self):
        """Test error prioritization by severity."""
        reporter = ErrorReporter()

        critical = WorkflowValidationError(
            category=ErrorCategory.SEMANTIC,
            severity=ErrorSeverity.CRITICAL,
            message="Critical",
            location="nodes[0]",
        )

        error = WorkflowValidationError(
            category=ErrorCategory.SEMANTIC,
            severity=ErrorSeverity.ERROR,
            message="Error",
            location="nodes[1]",
        )

        warning = WorkflowValidationError(
            category=ErrorCategory.SEMANTIC,
            severity=ErrorSeverity.WARNING,
            message="Warning",
            location="nodes[2]",
        )

        info = WorkflowValidationError(
            category=ErrorCategory.SEMANTIC,
            severity=ErrorSeverity.INFO,
            message="Info",
            location="nodes[3]",
        )

        errors = [info, warning, error, critical]  # Intentionally out of order
        prioritized = reporter.prioritize_errors(errors)

        assert prioritized[0] == critical
        assert prioritized[1] == error
        assert prioritized[2] == warning
        assert prioritized[3] == info

    def test_prioritize_errors_by_category(self):
        """Test error prioritization by category within same severity."""
        reporter = ErrorReporter()

        schema = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.ERROR,
            message="Schema",
            location="a",
        )

        structure = WorkflowValidationError(
            category=ErrorCategory.STRUCTURE,
            severity=ErrorSeverity.ERROR,
            message="Structure",
            location="b",
        )

        semantic = WorkflowValidationError(
            category=ErrorCategory.SEMANTIC,
            severity=ErrorSeverity.ERROR,
            message="Semantic",
            location="c",
        )

        security = WorkflowValidationError(
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.ERROR,
            message="Security",
            location="d",
        )

        errors = [security, semantic, structure, schema]  # Intentionally out of order
        prioritized = reporter.prioritize_errors(errors)

        # Priority: schema (1) > structure (2) > semantic (3) > security (4)
        assert prioritized[0] == schema
        assert prioritized[1] == structure
        assert prioritized[2] == semantic
        assert prioritized[3] == security

    def test_prioritize_errors_by_location(self):
        """Test error prioritization by location (alphabetical) as tertiary sort."""
        reporter = ErrorReporter()

        error_z = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.ERROR,
            message="Error Z",
            location="z",
        )

        error_a = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.ERROR,
            message="Error A",
            location="a",
        )

        error_m = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.ERROR,
            message="Error M",
            location="m",
        )

        errors = [error_z, error_a, error_m]
        prioritized = reporter.prioritize_errors(errors)

        assert prioritized[0] == error_a
        assert prioritized[1] == error_m
        assert prioritized[2] == error_z

    def test_prioritize_errors_complex_scenario(self):
        """Test error prioritization with complex mix of errors."""
        reporter = ErrorReporter()

        # Create mix of errors with different severity, category, and location
        errors = [
            WorkflowValidationError(
                category=ErrorCategory.SECURITY,
                severity=ErrorSeverity.WARNING,
                message="Security warning",
                location="z",
            ),
            WorkflowValidationError(
                category=ErrorCategory.SCHEMA,
                severity=ErrorSeverity.ERROR,
                message="Schema error",
                location="a",
            ),
            WorkflowValidationError(
                category=ErrorCategory.STRUCTURE,
                severity=ErrorSeverity.CRITICAL,
                message="Structure critical",
                location="m",
            ),
            WorkflowValidationError(
                category=ErrorCategory.SEMANTIC,
                severity=ErrorSeverity.ERROR,
                message="Semantic error",
                location="b",
            ),
            WorkflowValidationError(
                category=ErrorCategory.SCHEMA,
                severity=ErrorSeverity.CRITICAL,
                message="Schema critical",
                location="c",
            ),
        ]

        prioritized = reporter.prioritize_errors(errors)

        # First should be critical errors
        assert prioritized[0].severity == ErrorSeverity.CRITICAL
        assert prioritized[1].severity == ErrorSeverity.CRITICAL

        # Then error severity
        assert prioritized[2].severity == ErrorSeverity.ERROR
        assert prioritized[3].severity == ErrorSeverity.ERROR

        # Then warning
        assert prioritized[4].severity == ErrorSeverity.WARNING

        # Within critical: schema (priority 1) before structure (priority 2)
        if prioritized[0].category == ErrorCategory.SCHEMA:
            assert prioritized[1].category == ErrorCategory.STRUCTURE
        else:
            assert prioritized[1].category == ErrorCategory.SCHEMA

    def test_prioritize_empty_list(self):
        """Test prioritization with empty list."""
        reporter = ErrorReporter()
        prioritized = reporter.prioritize_errors([])
        assert len(prioritized) == 0


class TestErrorReporterIntegration:
    """Integration tests for ErrorReporter with various scenarios."""

    def test_comprehensive_error_report(self):
        """Test comprehensive error report with all error types."""
        reporter = ErrorReporter()

        # Create various errors across all categories and severities
        errors = [
            WorkflowValidationError(
                category=ErrorCategory.SCHEMA,
                severity=ErrorSeverity.CRITICAL,
                message="Missing required field",
                location="nodes[0]",
                suggestion="Add 'type' field",
                context={"field": "type"},
            ),
            WorkflowValidationError(
                category=ErrorCategory.STRUCTURE,
                severity=ErrorSeverity.ERROR,
                message="Invalid edge reference",
                location="edges[0]",
                suggestion="Fix edge target",
            ),
            WorkflowValidationError(
                category=ErrorCategory.SEMANTIC,
                severity=ErrorSeverity.WARNING,
                message="Node goal unclear",
                location="nodes[1]",
                suggestion="Clarify goal",
            ),
            WorkflowValidationError(
                category=ErrorCategory.SECURITY,
                severity=ErrorSeverity.ERROR,
                message="Unsafe code execution",
                location="nodes[2]",
                suggestion="Add validation",
            ),
        ]

        result = WorkflowGenerationValidationResult(
            is_valid=False,
            schema_errors=[errors[0]],
            structure_errors=[errors[1]],
            semantic_errors=[errors[2]],
            security_errors=[errors[3]],
            workflow_name="comprehensive_test",
        )

        # Test all report formats
        human = reporter.human_report(result)
        llm = reporter.llm_report(result)
        json_str = reporter.json_report(result)
        markdown = reporter.markdown_report(result)
        compact = reporter.compact_report(result)

        # Verify human report
        assert "comprehensive_test" in human
        assert "CRITICAL ERRORS" in human
        assert "ERRORS" in human
        assert "WARNINGS" in human

        # Verify LLM report
        assert "SCHEMA ERRORS" in llm
        assert "STRUCTURE ERRORS" in llm
        assert "SEMANTIC ERRORS" in llm
        assert "SECURITY ERRORS" in llm

        # Verify JSON report
        json_data = json.loads(json_str)
        assert json_data["is_valid"] is False
        assert json_data["workflow_name"] == "comprehensive_test"
        assert len(json_data["errors"]["schema"]) == 1

        # Verify markdown report
        assert "# Workflow Validation Report:" in markdown
        assert "**INVALID**" in markdown

        # Verify compact report
        assert "✗" in compact
        assert "1 critical" in compact

    def test_edge_case_all_info_severity(self):
        """Test report with all info-level errors."""
        reporter = ErrorReporter()

        info_errors = [
            WorkflowValidationError(
                category=ErrorCategory.SEMANTIC,
                severity=ErrorSeverity.INFO,
                message=f"Info {i}",
                location=f"nodes[{i}]",
            )
            for i in range(3)
        ]

        # Info-level errors don't cause is_valid to be False
        # but we want to test that info messages show up when show_context=True
        # Let's create a result that's invalid with info messages included
        result = WorkflowGenerationValidationResult(
            is_valid=False,
            semantic_errors=info_errors,
        )

        human = reporter.human_report(result, show_context=True)

        # Info messages only show with show_context=True
        assert "INFO (3)" in human
        assert "ℹ️" in human

    def test_edge_case_large_error_count(self):
        """Test report with large number of errors."""
        reporter = ErrorReporter()

        # Create 100 errors
        errors = [
            WorkflowValidationError(
                category=ErrorCategory.SCHEMA,
                severity=ErrorSeverity.ERROR,
                message=f"Error {i}",
                location=f"nodes[{i}]",
            )
            for i in range(100)
        ]

        result = WorkflowGenerationValidationResult(is_valid=False, schema_errors=errors)

        compact = reporter.compact_report(result)

        assert "100 errors" in compact

        # Test LLM report with limit
        llm = reporter.llm_report(result, max_errors_per_category=20)
        assert "80 more schema errors" in llm
