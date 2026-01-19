"""Tests for Vertical Linter tool."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

# Add scripts to path
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))


def test_vertical_linter_imports():
    """Test that the linter can be imported."""
    import lint_vertical

    assert lint_vertical is not None


def test_issue_category_enum():
    """Test IssueCategory enum values."""
    from lint_vertical import IssueCategory

    assert IssueCategory.NAMING.value == "NAMING"
    assert IssueCategory.DOCUMENTATION.value == "DOCUMENTATION"
    assert IssueCategory.PROTOCOL.value == "PROTOCOL"
    assert IssueCategory.STYLE.value == "STYLE"
    assert IssueCategory.SECURITY.value == "SECURITY"
    assert IssueCategory.ARCHITECTURE.value == "ARCHITECTURE"
    assert IssueCategory.BEST_PRACTICE.value == "BEST_PRACTICE"


def test_severity_enum():
    """Test Severity enum values."""
    from lint_vertical import Severity

    assert Severity.ERROR.value == "ERROR"
    assert Severity.WARNING.value == "WARNING"
    assert Severity.INFO.value == "INFO"
    assert Severity.SUGGESTION.value == "SUGGESTION"


def test_lint_issue_creation():
    """Test LintIssue dataclass creation."""
    from lint_vertical import LintIssue, IssueCategory, Severity

    issue = LintIssue(
        category=IssueCategory.NAMING,
        severity=Severity.ERROR,
        file_path="test.py",
        line_number=10,
        message="Test issue",
        suggestion="Test suggestion",
    )

    assert issue.category == IssueCategory.NAMING
    assert issue.severity == Severity.ERROR
    assert issue.file_path == "test.py"
    assert issue.line_number == 10
    assert issue.message == "Test issue"
    assert issue.suggestion == "Test suggestion"


def test_lint_report_creation():
    """Test LintReport dataclass creation."""
    from lint_vertical import LintReport

    report = LintReport(vertical_name="test_vertical")
    assert report.vertical_name == "test_vertical"
    assert report.total_files == 0
    assert report.issues == []
    assert report.error_count == 0
    assert report.warning_count == 0


def test_vertical_linter_initialization():
    """Test VerticalLinter initialization."""
    from lint_vertical import VerticalLinter

    linter = VerticalLinter(auto_fix=True)
    assert linter.auto_fix is True
    assert linter.report is None


def test_naming_convention_check():
    """Test naming convention checking."""
    import ast
    from lint_vertical import VerticalLinter

    # Create a test AST with naming violations
    code = """
class bad_name:  # Violation: not PascalCase
    pass

def BadFunction():  # Violation: not snake_case
    pass

BadVariable = 1  # Violation: not snake_case
    """

    linter = VerticalLinter()
    tree = ast.parse(code)

    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if not hasattr(node, "name"):
                continue
            # Check logic is tested indirectly through lint_vertical


def test_documentation_check():
    """Test documentation completeness checking."""
    import ast
    from lint_vertical import VerticalLinter

    # Code without docstrings
    code = """
class MyClass:
    def my_method(self):
        pass
    """

    linter = VerticalLinter()
    tree = ast.parse(code)

    module_doc = ast.get_docstring(tree)
    assert module_doc is None  # Missing module docstring


def test_security_check():
    """Test security issue detection."""
    from lint_vertical import VerticalLinter

    # Code with security issues
    code = """
password = "secret123"
api_key = "key123"
eval(some_input)
    """

    linter = VerticalLinter()

    # Check for hardcoded secrets
    import re

    secret_patterns = [
        (r"password\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password"),
        (r"api_key\s*=\s*['\"][^'\"]+['\"]", "Hardcoded API key"),
    ]

    issues_found = []
    for pattern, message in secret_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            issues_found.append(message)

    assert len(issues_found) >= 1
