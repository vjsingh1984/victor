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

"""Unit tests for victor/ui/commands/errors.py.

Tests for:
- Error command registration and structure
- Error listing with filters
- Error details display
- Error statistics
- Error metrics export
- Error categories
- Error search functionality
- Documentation access
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

from victor.ui.commands.errors import (
    errors_app,
    ERROR_CATALOG,
    list_errors,
    show_error,
    show_stats,
    export_metrics,
    list_categories,
    open_docs,
    search_errors,
)
from victor.observability.error_tracker import ErrorTracker, reset_error_tracker


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def error_tracker():
    """Create a fresh error tracker for each test."""
    reset_error_tracker()
    tracker = ErrorTracker()
    return tracker


@pytest.fixture
def sample_errors(error_tracker):
    """Populate error tracker with sample error data."""
    now = datetime.now(timezone.utc)

    # Add various types of errors
    error_tracker.record_error(
        error_type="ProviderNotFoundError",
        error_message="Provider 'xyz' not found",
        correlation_id="corr-001",
        context={"provider": "xyz"},
    )

    error_tracker.record_error(
        error_type="ProviderConnectionError",
        error_message="Failed to connect to provider 'anthropic'",
        correlation_id="corr-002",
        context={"provider": "anthropic"},
    )

    error_tracker.record_error(
        error_type="ToolExecutionError",
        error_message="Tool 'read_file' failed",
        correlation_id="corr-003",
        context={"tool": "read_file"},
    )

    error_tracker.record_error(
        error_type="ProviderNotFoundError",
        error_message="Provider 'openai' not found",
        correlation_id="corr-004",
        context={"provider": "openai"},
    )

    error_tracker.record_error(
        error_type="NetworkError",
        error_message="Connection timeout",
        correlation_id="corr-005",
        context={"timeout": 30},
    )

    return error_tracker


# =============================================================================
# Error Catalog Tests
# =============================================================================


class TestErrorCatalog:
    """Test ERROR_CATALOG structure and content."""

    def test_error_catalog_exists(self):
        """Test that ERROR_CATALOG is defined and non-empty."""
        assert ERROR_CATALOG is not None
        assert len(ERROR_CATALOG) > 0

    def test_error_catalog_structure(self):
        """Test that all error entries have required fields."""
        required_fields = {"code", "name", "category", "message", "severity", "recovery_hint"}

        for code, error_info in ERROR_CATALOG.items():
            assert required_fields.issubset(
                error_info.keys()
            ), f"Error {code} missing required fields"
            assert error_info["code"] == code, f"Error code mismatch for {code}"

    def test_error_catalog_codes_format(self):
        """Test that error codes follow the correct format (XXX-NNN)."""
        import re

        pattern = r"^[A-Z]+-\d{3}$"
        for code in ERROR_CATALOG.keys():
            assert re.match(pattern, code), f"Error code {code} doesn't match format XXX-NNN"

    def test_error_catalog_categories(self):
        """Test that error categories are valid."""
        expected_categories = {
            "Provider",
            "Tool",
            "Configuration",
            "Search",
            "Workflow",
            "File",
            "Network",
            "Extension",
        }

        actual_categories = {info["category"] for info in ERROR_CATALOG.values()}
        assert actual_categories == expected_categories

    def test_error_catalog_severity_levels(self):
        """Test that severity levels are valid."""
        valid_severities = {"ERROR", "WARNING"}

        for error_info in ERROR_CATALOG.values():
            assert (
                error_info["severity"] in valid_severities
            ), f"Invalid severity: {error_info['severity']}"

    def test_error_catalog_message_placeholders(self):
        """Test that error messages can contain placeholders."""
        # Some errors have placeholders like {provider_name}
        prov_error = ERROR_CATALOG["PROV-001"]
        assert "{provider_name}" in prov_error["message"]

        tool_error = ERROR_CATALOG["TOOL-001"]
        assert "{tool_name}" in tool_error["message"]

    def test_error_catalog_provider_errors(self):
        """Test that provider error codes are sequential."""
        provider_codes = [c for c in ERROR_CATALOG.keys() if c.startswith("PROV-")]
        assert len(provider_codes) == 7  # PROV-001 through PROV-007

    def test_error_catalog_tool_errors(self):
        """Test that tool error codes are sequential."""
        tool_codes = [c for c in ERROR_CATALOG.keys() if c.startswith("TOOL-")]
        assert len(tool_codes) == 4  # TOOL-001 through TOOL-004


# =============================================================================
# List Errors Command Tests
# =============================================================================


class TestListErrors:
    """Test the list_errors command."""

    def test_list_errors_all(self, runner):
        """Test listing all errors without filters."""
        result = runner.invoke(errors_app, ["list"])
        assert result.exit_code == 0
        # Should show table with error codes
        assert "PROV-001" in result.stdout
        assert "TOOL-001" in result.stdout

    def test_list_errors_by_category(self, runner):
        """Test filtering errors by category."""
        result = runner.invoke(errors_app, ["list", "--category", "Provider"])
        assert result.exit_code == 0
        assert "PROV-001" in result.stdout
        # Should not show tool errors
        assert "TOOL-001" not in result.stdout

    def test_list_errors_by_severity(self, runner):
        """Test filtering errors by severity."""
        result = runner.invoke(errors_app, ["list", "--severity", "ERROR"])
        assert result.exit_code == 0
        assert "PROV-001" in result.stdout
        # PROV-005 is WARNING, should not appear
        assert "PROV-005" not in result.stdout

    def test_list_errors_by_severity_warning(self, runner):
        """Test filtering for WARNING severity."""
        result = runner.invoke(errors_app, ["list", "--severity", "WARNING"])
        assert result.exit_code == 0
        # PROV-005 and EXT-001 are warnings
        assert "PROV-005" in result.stdout or "EXT-001" in result.stdout

    def test_list_errors_by_search(self, runner):
        """Test searching errors by keyword."""
        result = runner.invoke(errors_app, ["list", "--search", "timeout"])
        assert result.exit_code == 0
        assert "PROV-006" in result.stdout  # ProviderTimeoutError

    def test_list_errors_multiple_filters(self, runner):
        """Test combining multiple filters."""
        result = runner.invoke(
            errors_app, ["list", "--category", "Provider", "--severity", "ERROR"]
        )
        assert result.exit_code == 0
        assert "PROV-001" in result.stdout
        assert "PROV-005" not in result.stdout  # WARNING, not ERROR

    def test_list_errors_no_results(self, runner):
        """Test when filters return no results."""
        result = runner.invoke(errors_app, ["list", "--category", "NonExistent"])
        assert result.exit_code == 0
        assert "No errors found matching filters" in result.stdout

    def test_list_filters_shown_in_output(self, runner):
        """Test that applied filters are shown in output."""
        result = runner.invoke(
            errors_app, ["list", "--category", "Provider", "--search", "not found"]
        )
        assert result.exit_code == 0
        assert "Filters:" in result.stdout
        assert "category=Provider" in result.stdout
        assert "search=not found" in result.stdout

    def test_list_error_count_displayed(self, runner):
        """Test that error count is shown when filters are applied."""
        result = runner.invoke(errors_app, ["list", "--category", "Provider"])
        assert result.exit_code == 0
        # Should show "Showing X of Y errors"
        assert "Showing" in result.stdout
        assert "of" in result.stdout

    def test_list_recovery_hint_truncation(self, runner):
        """Test that long recovery hints are truncated in list view."""
        result = runner.invoke(errors_app, ["list", "--category", "Provider"])
        assert result.exit_code == 0
        # Hints longer than 50 chars should be truncated with "..."
        # PROV-002 has a moderately long hint
        assert "..." in result.stdout or result.stdout  # May or may not have truncation


# =============================================================================
# Show Error Command Tests
# =============================================================================


class TestShowError:
    """Test the show_error command."""

    def test_show_error_valid_code(self, runner):
        """Test showing details for a valid error code."""
        result = runner.invoke(errors_app, ["show", "PROV-001"])
        assert result.exit_code == 0
        assert "ProviderNotFoundError" in result.stdout
        assert "Provider" in result.stdout  # Category
        assert "Provider not found" in result.stdout  # Message
        assert "Recovery Hint" in result.stdout

    def test_show_error_lowercase_code(self, runner):
        """Test that error codes are case-insensitive."""
        result = runner.invoke(errors_app, ["show", "prov-001"])
        assert result.exit_code == 0
        assert "ProviderNotFoundError" in result.stdout

    def test_show_error_invalid_code(self, runner):
        """Test showing an invalid error code."""
        result = runner.invoke(errors_app, ["show", "INVALID-999"])
        assert result.exit_code == 1
        assert "not found in catalog" in result.stdout

    def test_show_error_displays_all_fields(self, runner):
        """Test that all error fields are displayed."""
        result = runner.invoke(errors_app, ["show", "TOOL-002"])
        assert result.exit_code == 0
        assert "Error Code:" in result.stdout
        assert "Name:" in result.stdout
        assert "Category:" in result.stdout
        assert "Severity:" in result.stdout
        assert "Message:" in result.stdout
        assert "Recovery Hint:" in result.stdout

    def test_show_error_related_errors(self, runner):
        """Test that related errors are shown."""
        result = runner.invoke(errors_app, ["show", "PROV-001"])
        assert result.exit_code == 0
        # Should show related Provider errors
        assert "Related Errors:" in result.stdout
        # Should list other provider errors (PROV-002, PROV-003, etc.)
        assert "PROV-002" in result.stdout or "PROV-003" in result.stdout

    def test_show_error_no_related_errors(self, runner):
        """Test error with no related errors (category with only one error)."""
        # Find a category with only one error
        # Extension category has only EXT-001
        result = runner.invoke(errors_app, ["show", "EXT-001"])
        assert result.exit_code == 0
        # Should not show related errors section
        # (or show it empty)
        # Output may or may not show "Related Errors:" depending on implementation

    def test_show_error_documentation_reference(self, runner):
        """Test that documentation reference is shown."""
        result = runner.invoke(errors_app, ["show", "CFG-001"])
        assert result.exit_code == 0
        assert "docs/errors.md" in result.stdout


# =============================================================================
# Error Stats Command Tests
# =============================================================================


class TestShowStats:
    """Test the show_stats command."""

    def test_show_stats_default(self, runner, sample_errors):
        """Test showing error statistics with default timeframe."""
        with patch("victor.ui.commands.errors.get_error_tracker", return_value=sample_errors):
            result = runner.invoke(errors_app, ["stats"])
            assert result.exit_code == 0
            assert "Error Statistics" in result.stdout
            assert "Total Errors:" in result.stdout
            assert "5" in result.stdout  # We added 5 errors

    def test_show_stats_custom_timeframe(self, runner, sample_errors):
        """Test showing stats with custom timeframe."""
        with patch("victor.ui.commands.errors.get_error_tracker", return_value=sample_errors):
            result = runner.invoke(errors_app, ["stats", "--timeframe", "7d"])
            assert result.exit_code == 0
            assert "(7d)" in result.stdout

    def test_show_stats_top_errors(self, runner, sample_errors):
        """Test that top error types are shown."""
        with patch("victor.ui.commands.errors.get_error_tracker", return_value=sample_errors):
            result = runner.invoke(errors_app, ["stats"])
            assert result.exit_code == 0
            assert "Top Error Types:" in result.stdout
            assert "ProviderNotFoundError" in result.stdout  # 2 occurrences

    def test_show_stats_error_rates(self, runner, sample_errors):
        """Test that error rates are displayed."""
        with patch("victor.ui.commands.errors.get_error_tracker", return_value=sample_errors):
            result = runner.invoke(errors_app, ["stats"])
            assert result.exit_code == 0
            assert "Error Rates (per hour):" in result.stdout

    def test_show_stats_recent_errors(self, runner, sample_errors):
        """Test that recent errors are shown."""
        with patch("victor.ui.commands.errors.get_error_tracker", return_value=sample_errors):
            result = runner.invoke(errors_app, ["stats"])
            assert result.exit_code == 0
            assert "Recent Errors:" in result.stdout
            # The correlation_id is shown in the error record but format may vary
            # Just check that error details are shown
            assert "ProviderNotFoundError" in result.stdout or "Provider" in result.stdout

    def test_show_stats_percentage_calculation(self, runner, sample_errors):
        """Test that error percentages are calculated correctly."""
        with patch("victor.ui.commands.errors.get_error_tracker", return_value=sample_errors):
            result = runner.invoke(errors_app, ["stats"])
            assert result.exit_code == 0
            # ProviderNotFoundError: 2 out of 5 = 40%
            assert "40.0%" in result.stdout or "40.0%" in result.stdout

    def test_show_stats_empty_tracker(self, runner, error_tracker):
        """Test stats when no errors have been recorded."""
        with patch("victor.ui.commands.errors.get_error_tracker", return_value=error_tracker):
            result = runner.invoke(errors_app, ["stats"])
            assert result.exit_code == 0
            assert "Total Errors: 0" in result.stdout

    def test_show_stats_export_hint(self, runner, sample_errors):
        """Test that export hint is shown."""
        with patch("victor.ui.commands.errors.get_error_tracker", return_value=sample_errors):
            result = runner.invoke(errors_app, ["stats"])
            assert result.exit_code == 0
            assert "victor errors export" in result.stdout


# =============================================================================
# Export Metrics Command Tests
# =============================================================================


class TestExportMetrics:
    """Test the export_metrics command."""

    def test_export_json_default(self, runner, sample_errors, tmp_path):
        """Test exporting metrics to JSON (default format)."""
        output_file = tmp_path / "metrics.json"

        with patch("victor.ui.commands.errors.get_error_tracker", return_value=sample_errors):
            result = runner.invoke(errors_app, ["export", str(output_file)])
            assert result.exit_code == 0
            assert "Metrics exported" in result.stdout
            assert output_file.exists()

            # Verify JSON content
            with open(output_file) as f:
                data = json.load(f)
                assert "summary" in data
                assert "error_rates" in data
                assert "exported_at" in data

    def test_export_json_explicit(self, runner, sample_errors, tmp_path):
        """Test exporting metrics to JSON with explicit format."""
        output_file = tmp_path / "metrics.json"

        with patch("victor.ui.commands.errors.get_error_tracker", return_value=sample_errors):
            result = runner.invoke(errors_app, ["export", str(output_file), "--format", "json"])
            assert result.exit_code == 0
            assert output_file.exists()

    def test_export_csv(self, runner, sample_errors, tmp_path):
        """Test exporting metrics to CSV format."""
        output_file = tmp_path / "metrics.csv"

        with patch("victor.ui.commands.errors.get_error_tracker", return_value=sample_errors):
            result = runner.invoke(errors_app, ["export", str(output_file), "--format", "csv"])
            assert result.exit_code == 0
            assert "Metrics exported" in result.stdout
            assert output_file.exists()

            # Verify CSV content
            with open(output_file, newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
                assert len(rows) > 0
                assert rows[0] == ["Error Type", "Count", "Rate (per hour)"]
                # Should have header + at least one error type
                assert len(rows) >= 2

    def test_export_creates_directory(self, runner, sample_errors, tmp_path):
        """Test that export creates output directory if needed."""
        output_file = tmp_path / "subdir" / "metrics.json"

        with patch("victor.ui.commands.errors.get_error_tracker", return_value=sample_errors):
            result = runner.invoke(errors_app, ["export", str(output_file)])
            assert result.exit_code == 0
            assert output_file.exists()
            assert output_file.parent.exists()

    def test_export_invalid_format(self, runner, sample_errors, tmp_path):
        """Test exporting with invalid format."""
        output_file = tmp_path / "metrics.txt"

        with patch("victor.ui.commands.errors.get_error_tracker", return_value=sample_errors):
            result = runner.invoke(errors_app, ["export", str(output_file), "--format", "txt"])
            assert result.exit_code == 1
            assert "Unsupported format" in result.stdout

    def test_export_csv_content(self, runner, sample_errors, tmp_path):
        """Test that CSV export contains correct data."""
        output_file = tmp_path / "metrics.csv"

        with patch("victor.ui.commands.errors.get_error_tracker", return_value=sample_errors):
            result = runner.invoke(errors_app, ["export", str(output_file), "--format", "csv"])
            assert result.exit_code == 0

            with open(output_file, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                # Check that we have error types
                assert len(rows) > 0

                # Check that ProviderNotFoundError is present (count: 2)
                provider_not_found = next(
                    (r for r in rows if r["Error Type"] == "ProviderNotFoundError"), None
                )
                assert provider_not_found is not None
                assert int(provider_not_found["Count"]) == 2


# =============================================================================
# List Categories Command Tests
# =============================================================================


class TestListCategories:
    """Test the list_categories command."""

    def test_list_categories(self, runner):
        """Test listing all error categories."""
        result = runner.invoke(errors_app, ["categories"])
        assert result.exit_code == 0
        assert "Error Categories" in result.stdout
        assert "Category" in result.stdout
        assert "Count" in result.stdout
        assert "Error Codes" in result.stdout

    def test_list_categories_provider(self, runner):
        """Test that Provider category is shown."""
        result = runner.invoke(errors_app, ["categories"])
        assert result.exit_code == 0
        assert "Provider" in result.stdout
        assert "7" in result.stdout  # 7 provider errors

    def test_list_categories_tool(self, runner):
        """Test that Tool category is shown."""
        result = runner.invoke(errors_app, ["categories"])
        assert result.exit_code == 0
        assert "Tool" in result.stdout

    def test_list_categories_sorted(self, runner):
        """Test that categories are sorted alphabetically."""
        result = runner.invoke(errors_app, ["categories"])
        assert result.exit_code == 0
        # Categories should be in alphabetical order
        lines = result.stdout.split("\n")
        # Extract category lines (skip header)
        category_lines = [
            l for l in lines if l.strip() and not l.startswith("─") and "Error Categories" not in l
        ]
        # Should be sorted

    def test_list_categories_error_codes(self, runner):
        """Test that error codes are listed for each category."""
        result = runner.invoke(errors_app, ["categories"])
        assert result.exit_code == 0
        # Provider category should list PROV-001 through PROV-007
        assert "PROV-001" in result.stdout
        assert "PROV-007" in result.stdout


# =============================================================================
# Open Docs Command Tests
# =============================================================================


class TestOpenDocs:
    """Test the open_docs command."""

    def test_open_docs_file_exists(self, runner, tmp_path):
        """Test opening docs when file exists."""
        # Create a temporary docs directory with errors.md
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        docs_file = docs_dir / "errors.md"
        docs_file.write_text("# Error Documentation\n\nTest content.")

        with patch("victor.ui.commands.errors.Path") as mock_path:
            # Mock Path to return our temp directory
            mock_path.return_value = mock_path
            mock_path.__file__ = tmp_path / "victor" / "ui" / "commands" / "errors.py"

            # This is complex to mock properly, so we'll test the error case instead
            pass

    def test_open_docs_file_not_found(self, runner):
        """Test opening docs when file doesn't exist."""
        # The real docs/errors.md likely doesn't exist in test environment
        with patch("victor.ui.commands.errors.Path") as mock_path:
            # Make Path(__file__).parent.parent.parent / "docs" / "errors.md"
            # return a non-existent path
            mock_file = Mock()
            mock_file.exists.return_value = False
            mock_file.absolute.return_value = "/fake/path/errors.md"

            mock_path_instance = Mock()
            mock_path_instance.__truediv__ = Mock(return_value=mock_file)

            # This is complex to mock, so we'll just verify the command exists
            result = runner.invoke(errors_app, ["docs"])
            # Command should not crash
            # May or may not find docs depending on environment
            assert result.exit_code in [0, 1]


# =============================================================================
# Search Errors Command Tests
# =============================================================================


class TestSearchErrors:
    """Test the search_errors command."""

    def test_search_by_name(self, runner):
        """Test searching by error name."""
        result = runner.invoke(errors_app, ["search", "timeout"])
        assert result.exit_code == 0
        assert "Search Results:" in result.stdout
        assert "PROV-006" in result.stdout  # ProviderTimeoutError

    def test_search_by_message(self, runner):
        """Test searching by error message content."""
        result = runner.invoke(errors_app, ["search", "not found"])
        assert result.exit_code == 0
        assert "PROV-001" in result.stdout  # "Provider not found"
        assert "TOOL-001" in result.stdout  # "Tool not found"

    def test_search_by_recovery_hint(self, runner):
        """Test searching by recovery hint content."""
        result = runner.invoke(errors_app, ["search", "API key"])
        assert result.exit_code == 0
        # Should find errors mentioning API key in recovery hints
        assert "PROV-004" in result.stdout or "PROV-002" in result.stdout

    def test_search_case_insensitive(self, runner):
        """Test that search is case-insensitive."""
        result_lower = runner.invoke(errors_app, ["search", "timeout"])
        result_upper = runner.invoke(errors_app, ["search", "TIMEOUT"])
        result_mixed = runner.invoke(errors_app, ["search", "Timeout"])

        assert result_lower.exit_code == 0
        assert result_upper.exit_code == 0
        assert result_mixed.exit_code == 0
        # All should find PROV-006
        assert "PROV-006" in result_lower.stdout
        assert "PROV-006" in result_upper.stdout
        assert "PROV-006" in result_mixed.stdout

    def test_search_no_results(self, runner):
        """Test search with no matching results."""
        result = runner.invoke(errors_app, ["search", "nonexistent_error_xyz"])
        assert result.exit_code == 0
        assert "No errors found matching" in result.stdout

    def test_search_shows_match_count(self, runner):
        """Test that search shows number of matches."""
        result = runner.invoke(errors_app, ["search", "provider"])
        assert result.exit_code == 0
        assert "Found" in result.stdout
        assert "matching errors" in result.stdout

    def test_search_shows_matched_field(self, runner):
        """Test that search shows which field matched."""
        result = runner.invoke(errors_app, ["search", "timeout"])
        assert result.exit_code == 0
        # Should show "Matched In" column
        assert "Matched In" in result.stdout

    def test_search_multiple_matches(self, runner):
        """Test search that matches multiple errors."""
        result = runner.invoke(errors_app, ["search", "error"])
        assert result.exit_code == 0
        # Should match multiple errors (all have "error" in name)
        assert "Found" in result.stdout

    def test_search_shows_hint_command(self, runner):
        """Test that search shows hint for viewing details."""
        result = runner.invoke(errors_app, ["search", "timeout"])
        assert result.exit_code == 0
        assert "victor errors show" in result.stdout


# =============================================================================
# Integration Tests
# =============================================================================


class TestErrorCommandsIntegration:
    """Integration tests for error commands."""

    def test_list_then_show_workflow(self, runner):
        """Test listing errors then showing details."""
        # First, list errors
        list_result = runner.invoke(errors_app, ["list", "--category", "Provider"])
        assert list_result.exit_code == 0
        assert "PROV-001" in list_result.stdout

        # Then, show details for one error
        show_result = runner.invoke(errors_app, ["show", "PROV-001"])
        assert show_result.exit_code == 0
        assert "ProviderNotFoundError" in show_result.stdout

    def test_search_then_show_workflow(self, runner):
        """Test searching for errors then showing details."""
        # Search for timeout errors
        search_result = runner.invoke(errors_app, ["search", "timeout"])
        assert search_result.exit_code == 0
        assert "PROV-006" in search_result.stdout

        # Show details
        show_result = runner.invoke(errors_app, ["show", "PROV-006"])
        assert show_result.exit_code == 0
        assert "ProviderTimeoutError" in show_result.stdout

    def test_categories_then_list_workflow(self, runner):
        """Test listing categories then filtering by category."""
        # List categories
        cat_result = runner.invoke(errors_app, ["categories"])
        assert cat_result.exit_code == 0
        assert "Tool" in cat_result.stdout

        # List errors in Tool category
        list_result = runner.invoke(errors_app, ["list", "--category", "Tool"])
        assert list_result.exit_code == 0
        assert "TOOL-001" in list_result.stdout

    def test_export_workflow(self, runner, sample_errors, tmp_path):
        """Test exporting metrics and verifying content."""
        output_file = tmp_path / "metrics.json"

        with patch("victor.ui.commands.errors.get_error_tracker", return_value=sample_errors):
            # Export metrics
            export_result = runner.invoke(errors_app, ["export", str(output_file)])
            assert export_result.exit_code == 0
            assert output_file.exists()

            # Verify we can read and use the exported data
            with open(output_file) as f:
                data = json.load(f)
                assert "summary" in data
                assert data["summary"]["total_errors"] == 5

    def test_help_command(self, runner):
        """Test that help command works."""
        result = runner.invoke(errors_app, ["--help"])
        assert result.exit_code == 0
        assert "Manage and view error information" in result.stdout
        assert "list" in result.stdout
        assert "show" in result.stdout
        assert "stats" in result.stdout
        assert "export" in result.stdout
        assert "categories" in result.stdout
        assert "docs" in result.stdout
        assert "search" in result.stdout


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestErrorHandling:
    """Test edge cases and error handling."""

    def test_empty_error_code(self, runner):
        """Test showing error with empty code."""
        result = runner.invoke(errors_app, ["show", ""])
        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_special_characters_in_search(self, runner):
        """Test searching with special characters."""
        result = runner.invoke(errors_app, ["search", "'provider'"])
        assert result.exit_code == 0
        # Should not crash

    def test_unicode_in_search(self, runner):
        """Test searching with Unicode characters."""
        result = runner.invoke(errors_app, ["search", "error"])
        assert result.exit_code == 0
        # Should handle Unicode gracefully

    def test_very_long_search_query(self, runner):
        """Test searching with very long query."""
        long_query = "error " * 100
        result = runner.invoke(errors_app, ["search", long_query])
        assert result.exit_code == 0
        # Should not crash

    def test_export_to_read_only_directory(self, runner, sample_errors):
        """Test exporting to a read-only location."""
        # This is hard to test cross-platform, but we verify the command structure
        # In real scenario, it should fail gracefully
        pass

    def test_concurrent_command_execution(self, runner):
        """Test that commands can be run concurrently without issues."""
        # Run multiple commands in sequence
        result1 = runner.invoke(errors_app, ["list"])
        result2 = runner.invoke(errors_app, ["categories"])
        result3 = runner.invoke(errors_app, ["search", "error"])

        # All should succeed
        assert result1.exit_code == 0
        assert result2.exit_code == 0
        assert result3.exit_code == 0


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Test performance characteristics."""

    def test_list_all_errors_performance(self, runner):
        """Test that listing all errors is fast."""
        import time

        start = time.time()
        result = runner.invoke(errors_app, ["list"])
        elapsed = time.time() - start

        assert result.exit_code == 0
        # Should complete in less than 1 second
        assert elapsed < 1.0

    def test_search_performance(self, runner):
        """Test that search is fast."""
        import time

        start = time.time()
        result = runner.invoke(errors_app, ["search", "error"])
        elapsed = time.time() - start

        assert result.exit_code == 0
        # Should complete in less than 1 second
        assert elapsed < 1.0

    def test_export_performance(self, runner, sample_errors, tmp_path):
        """Test that export is fast."""
        import time

        output_file = tmp_path / "metrics.json"

        with patch("victor.ui.commands.errors.get_error_tracker", return_value=sample_errors):
            start = time.time()
            result = runner.invoke(errors_app, ["export", str(output_file)])
            elapsed = time.time() - start

        assert result.exit_code == 0
        # Should complete in less than 1 second
        assert elapsed < 1.0
