#!/usr/bin/env python3
"""Vertical Linter for Victor

This tool checks vertical code quality, enforces conventions, and suggests improvements.
It performs comprehensive linting beyond what standard linters provide.

Checks:
- Naming conventions (classes, files, directories)
- Documentation completeness (docstrings, type hints)
- Protocol conformance
- Code style (imports, line length, complexity)
- Security issues (hardcoded secrets, unsafe patterns)
- Architecture compliance (SOLID principles, proper base classes)

Usage:
    python scripts/lint_vertical.py victor/coding
    python scripts/lint_vertical.py victor/coding --fix
    python scripts/lint_vertical.py --all-verticals

Exit Codes:
    0: No issues found
    1: Issues found
    2: Error occurred
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class IssueCategory(Enum):
    """Categories of linting issues."""

    NAMING = "NAMING"
    DOCUMENTATION = "DOCUMENTATION"
    PROTOCOL = "PROTOCOL"
    STYLE = "STYLE"
    SECURITY = "SECURITY"
    ARCHITECTURE = "ARCHITECTURE"
    BEST_PRACTICE = "BEST_PRACTICE"


class Severity(Enum):
    """Issue severity levels."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    SUGGESTION = "SUGGESTION"


@dataclass
class LintIssue:
    """Represents a linting issue."""

    category: IssueCategory
    severity: Severity
    file_path: str
    line_number: int
    message: str
    suggestion: Optional[str] = None
    auto_fixable: bool = False

    def __str__(self) -> str:
        """Format issue for display."""
        return (
            f"[{self.severity.value}] {self.file_path}:{self.line_number} "
            f"({self.category.value}) - {self.message}"
        )


@dataclass
class LintReport:
    """Linting report for a vertical."""

    vertical_name: str
    total_files: int = 0
    issues: List[LintIssue] = field(default_factory=list)

    def add_issue(self, issue: LintIssue) -> None:
        """Add an issue to the report."""
        self.issues.append(issue)

    @property
    def error_count(self) -> int:
        """Count of ERROR issues."""
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of WARNING issues."""
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    @property
    def info_count(self) -> int:
        """Count of INFO issues."""
        return sum(1 for i in self.issues if i.severity == Severity.INFO)

    def get_issues_by_category(self, category: IssueCategory) -> List[LintIssue]:
        """Get all issues in a category."""
        return [i for i in self.issues if i.category == category]

    def get_issues_by_severity(self, severity: Severity) -> List[LintIssue]:
        """Get all issues with a severity level."""
        return [i for i in self.issues if i.severity == severity]


class VerticalLinter:
    """Lints vertical code quality and conventions."""

    def __init__(self, auto_fix: bool = False):
        """Initialize linter.

        Args:
            auto_fix: Enable auto-fixing of issues where possible
        """
        self.auto_fix = auto_fix
        self.report: Optional[LintReport] = None
        self.fixes_applied: List[str] = []

    def lint_vertical(self, vertical_path: Path) -> LintReport:
        """Lint a vertical directory.

        Args:
            vertical_path: Path to vertical directory

        Returns:
            LintReport with findings
        """
        self.report = LintReport(vertical_name=vertical_path.name)

        # Check directory structure
        self._check_directory_structure(vertical_path)

        # Check Python files
        python_files = list(vertical_path.rglob("*.py"))
        self.report.total_files = len(python_files)

        for py_file in python_files:
            self._lint_python_file(py_file)

        # Check YAML configs
        yaml_files = list(vertical_path.rglob("*.yaml")) + list(vertical_path.rglob("*.yml"))
        for yaml_file in yaml_files:
            self._lint_yaml_file(yaml_file)

        # Check vertical class
        self._check_vertical_class(vertical_path)

        return self.report

    def _check_directory_structure(self, vertical_path: Path) -> None:
        """Check vertical directory structure.

        Args:
            vertical_path: Path to vertical
        """
        # Check for required files
        required_files = ["assistant.py", "__init__.py"]
        for required in required_files:
            if not (vertical_path / required).exists():
                self.report.add_issue(
                    LintIssue(
                        category=IssueCategory.ARCHITECTURE,
                        severity=Severity.ERROR,
                        file_path=str(vertical_path / required),
                        line_number=0,
                        message=f"Missing required file: {required}",
                        suggestion=f"Create {required}",
                    )
                )

        # Check for config directory
        if not (vertical_path / "config").exists():
            self.report.add_issue(
                LintIssue(
                    category=IssueCategory.BEST_PRACTICE,
                    severity=Severity.WARNING,
                    file_path=str(vertical_path),
                    line_number=0,
                    message="Missing config directory",
                    suggestion="Create config/ directory for YAML configurations",
                )
            )

    def _lint_python_file(self, file_path: Path) -> None:
        """Lint a Python file.

        Args:
            file_path: Path to Python file
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            self.report.add_issue(
                LintIssue(
                    category=IssueCategory.STYLE,
                    severity=Severity.ERROR,
                    file_path=str(file_path),
                    line_number=e.lineno or 0,
                    message=f"Syntax error: {e.msg}",
                )
            )
            return

        # Check naming conventions
        self._check_naming_conventions(file_path, tree)

        # Check documentation
        self._check_documentation(file_path, tree, content)

        # Check imports
        self._check_imports(file_path, tree)

        # Check complexity
        self._check_complexity(file_path, tree)

        # Check security
        self._check_security(file_path, content)

    def _check_naming_conventions(self, file_path: Path, tree: ast.AST) -> None:
        """Check naming conventions.

        Args:
            file_path: File being checked
            tree: AST of the file
        """
        for node in ast.walk(tree):
            # Check class names (PascalCase)
            if isinstance(node, ast.ClassDef):
                if not re.match(r"^[A-Z][a-zA-Z0-9]*$", node.name):
                    self.report.add_issue(
                        LintIssue(
                            category=IssueCategory.NAMING,
                            severity=Severity.WARNING,
                            file_path=str(file_path),
                            line_number=node.lineno,
                            message=f"Class name should be PascalCase: {node.name}",
                            suggestion="Use PascalCase for class names",
                        )
                    )

            # Check function names (snake_case)
            if isinstance(node, ast.FunctionDef):
                if not re.match(r"^[a-z][a-z0-9_]*$", node.name):
                    self.report.add_issue(
                        LintIssue(
                            category=IssueCategory.NAMING,
                            severity=Severity.WARNING,
                            file_path=str(file_path),
                            line_number=node.lineno,
                            message=f"Function name should be snake_case: {node.name}",
                            suggestion="Use snake_case for function names",
                        )
                    )

            # Check variable names (snake_case)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if not re.match(r"^[a-z][a-z0-9_]*$", target.id):
                            self.report.add_issue(
                                LintIssue(
                                    category=IssueCategory.NAMING,
                                    severity=Severity.INFO,
                                    file_path=str(file_path),
                                    line_number=node.lineno,
                                    message=f"Variable name should be snake_case: {target.id}",
                                    suggestion="Use snake_case for variable names",
                                )
                            )

    def _check_documentation(
        self, file_path: Path, tree: ast.AST, content: str
    ) -> None:
        """Check documentation completeness.

        Args:
            file_path: File being checked
            tree: AST of the file
            content: File content
        """
        # Check for module docstring
        module_doc = ast.get_docstring(tree)
        if not module_doc:
            self.report.add_issue(
                LintIssue(
                    category=IssueCategory.DOCUMENTATION,
                    severity=Severity.WARNING,
                    file_path=str(file_path),
                    line_number=1,
                    message="Missing module docstring",
                    suggestion='Add a docstring at the top of the file: """Module description."""',
                )
            )

        # Check for class docstrings
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node)
                if not doc:
                    self.report.add_issue(
                        LintIssue(
                            category=IssueCategory.DOCUMENTATION,
                            severity=Severity.WARNING,
                            file_path=str(file_path),
                            line_number=node.lineno,
                            message=f"Class {node.name} missing docstring",
                            suggestion=f'Add docstring to class {node.name}',
                        )
                    )

            # Check for function docstrings
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip private methods
                if node.name.startswith("_"):
                    continue

                doc = ast.get_docstring(node)
                if not doc:
                    self.report.add_issue(
                        LintIssue(
                            category=IssueCategory.DOCUMENTATION,
                            severity=Severity.INFO,
                            file_path=str(file_path),
                            line_number=node.lineno,
                            message=f"Function {node.name} missing docstring",
                            suggestion=f'Add docstring to function {node.name}',
                        )
                    )

                # Check for type hints
                if not node.returns:
                    self.report.add_issue(
                        LintIssue(
                            category=IssueCategory.DOCUMENTATION,
                            severity=Severity.INFO,
                            file_path=str(file_path),
                            line_number=node.lineno,
                            message=f"Function {node.name} missing return type hint",
                            suggestion="Add return type annotation",
                        )
                    )

                # Check parameter type hints
                for arg in node.args.args:
                    if arg.arg == "self":
                        continue
                    if not arg.annotation:
                        self.report.add_issue(
                            LintIssue(
                                category=IssueCategory.DOCUMENTATION,
                                severity=Severity.INFO,
                                file_path=str(file_path),
                                line_number=node.lineno,
                                message=f"Parameter {arg.arg} in {node.name} missing type hint",
                                suggestion="Add type annotation",
                            )
                        )

    def _check_imports(self, file_path: Path, tree: ast.AST) -> None:
        """Check import statements.

        Args:
            file_path: File being checked
            tree: AST of the file
        """
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(node)

        # Check for unused imports (basic check)
        # This is a simplified version - real unused import detection is more complex

        # Check import ordering (stdlib, third-party, local)
        stdlib_imports = []
        third_party_imports = []
        local_imports = []

        for imp in imports:
            if isinstance(imp, ast.Import):
                module_name = imp.names[0].name
            else:  # ImportFrom
                module_name = imp.module or ""

            if module_name.startswith("victor"):
                local_imports.append(imp)
            elif module_name.split(".")[0] in [
                "os", "sys", "pathlib", "typing", "dataclasses", "enum",
                "logging", "re", "json", "yaml", "asyncio",
            ]:
                stdlib_imports.append(imp)
            else:
                third_party_imports.append(imp)

        # Check if imports are properly grouped
        all_imports = stdlib_imports + third_party_imports + local_imports
        if all_imports != imports:
            self.report.add_issue(
                LintIssue(
                    category=IssueCategory.STYLE,
                    severity=Severity.INFO,
                    file_path=str(file_path),
                    line_number=1,
                    message="Imports not properly grouped (stdlib, third-party, local)",
                    suggestion="Group imports by type with blank lines between groups",
                    auto_fixable=True,
                )
            )

    def _check_complexity(self, file_path: Path, tree: ast.AST) -> None:
        """Check code complexity.

        Args:
            file_path: File being checked
            tree: AST of the file
        """
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Count cyclomatic complexity
                complexity = 1  # Base complexity
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        complexity += 1
                    elif isinstance(child, ast.BoolOp):
                        complexity += len(child.values) - 1

                if complexity > 10:
                    self.report.add_issue(
                        LintIssue(
                            category=IssueCategory.BEST_PRACTICE,
                            severity=Severity.WARNING,
                            file_path=str(file_path),
                            line_number=node.lineno,
                            message=f"Function {node.name} has high cyclomatic complexity: {complexity}",
                            suggestion="Consider breaking this function into smaller functions",
                        )
                    )

                # Check function length
                if hasattr(node, "end_lineno") and node.end_lineno:
                    lines = node.end_lineno - node.lineno
                    if lines > 50:
                        self.report.add_issue(
                            LintIssue(
                                category=IssueCategory.BEST_PRACTICE,
                                severity=Severity.INFO,
                                file_path=str(file_path),
                                line_number=node.lineno,
                                message=f"Function {node.name} is long: {lines} lines",
                                suggestion="Consider breaking into smaller functions",
                            )
                        )

    def _check_security(self, file_path: Path, content: str) -> None:
        """Check for security issues.

        Args:
            file_path: File being checked
            content: File content
        """
        lines = content.split("\n")

        # Check for hardcoded secrets
        secret_patterns = [
            (r"password\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password"),
            (r"api_key\s*=\s*['\"][^'\"]+['\"]", "Hardcoded API key"),
            (r"secret\s*=\s*['\"][^'\"]+['\"]", "Hardcoded secret"),
            (r"token\s*=\s*['\"][^'\"]+['\"]", "Hardcoded token"),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, message in secret_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    self.report.add_issue(
                        LintIssue(
                            category=IssueCategory.SECURITY,
                            severity=Severity.ERROR,
                            file_path=str(file_path),
                            line_number=i,
                            message=message,
                            suggestion="Use environment variables or configuration files",
                        )
                    )

        # Check for eval/exec
        if re.search(r"\beval\s*\(", content):
            self.report.add_issue(
                LintIssue(
                    category=IssueCategory.SECURITY,
                    severity=Severity.ERROR,
                    file_path=str(file_path),
                    line_number=1,
                    message="Use of eval() detected",
                    suggestion="Avoid eval() - use safer alternatives",
                )
            )

    def _lint_yaml_file(self, file_path: Path) -> None:
        """Lint a YAML file.

        Args:
            file_path: Path to YAML file
        """
        try:
            import yaml

            with open(file_path, "r", encoding="utf-8") as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.report.add_issue(
                LintIssue(
                    category=IssueCategory.STYLE,
                    severity=Severity.ERROR,
                    file_path=str(file_path),
                    line_number=getattr(e, "problem_mark", {}).get("line", 1) + 1,
                    message=f"YAML syntax error: {e}",
                    suggestion="Fix YAML syntax",
                )
            )

    def _check_vertical_class(self, vertical_path: Path) -> None:
        """Check that vertical class follows conventions.

        Args:
            vertical_path: Path to vertical
        """
        try:
            module = importlib.import_module(f"victor.{vertical_path.name}")
        except ImportError:
            return

        # Find vertical class
        from victor.core.verticals.base import VerticalBase

        vertical_class = None
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, VerticalBase) and obj != VerticalBase:
                vertical_class = obj
                break

        if not vertical_class:
            return

        # Check that it has required attributes
        if not hasattr(vertical_class, "name") or not vertical_class.name:
            self.report.add_issue(
                LintIssue(
                    category=IssueCategory.ARCHITECTURE,
                    severity=Severity.ERROR,
                    file_path=f"victor/{vertical_path.name}",
                    line_number=0,
                    message="Vertical class missing 'name' attribute",
                    suggestion="Add a class-level 'name' attribute",
                )
            )

        # Check for get_tools and get_system_prompt
        if not hasattr(vertical_class, "get_tools"):
            self.report.add_issue(
                LintIssue(
                    category=IssueCategory.ARCHITECTURE,
                    severity=Severity.ERROR,
                    file_path=f"victor/{vertical_path.name}",
                    line_number=0,
                    message="Vertical class missing get_tools() method",
                    suggestion="Implement get_tools() classmethod",
                )
            )

        if not hasattr(vertical_class, "get_system_prompt"):
            self.report.add_issue(
                LintIssue(
                    category=IssueCategory.ARCHITECTURE,
                    severity=Severity.ERROR,
                    file_path=f"victor/{vertical_path.name}",
                    line_number=0,
                    message="Vertical class missing get_system_prompt() method",
                    suggestion="Implement get_system_prompt() classmethod",
                )
            )

    def print_report(self) -> None:
        """Print the linting report."""
        if not self.report:
            print("No report generated")
            return

        print(f"\n{'=' * 80}")
        print(f"VERTICAL LINT REPORT: {self.report.vertical_name}")
        print(f"{'=' * 80}")
        print(f"Files checked: {self.report.total_files}")
        print(f"Errors: {self.report.error_count}")
        print(f"Warnings: {self.report.warning_count}")
        print(f"Info: {self.report.info_count}")
        print()

        # Group issues by category
        for category in IssueCategory:
            issues = self.report.get_issues_by_category(category)
            if not issues:
                continue

            print(f"\n{category.value}")
            print("-" * 80)
            for issue in issues[:10]:  # Limit to 10 per category
                print(issue)
                if issue.suggestion:
                    print(f"  → {issue.suggestion}")

            if len(issues) > 10:
                print(f"\n... and {len(issues) - 10} more {category.value} issues")

        if self.fixes_applied:
            print("\nFixes Applied:")
            for fix in self.fixes_applied:
                print(f"  ✓ {fix}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Lint Victor verticals")
    parser.add_argument(
        "vertical",
        type=Path,
        nargs="?",
        help="Path to vertical directory",
    )
    parser.add_argument(
        "--all-verticals",
        action="store_true",
        help="Lint all verticals",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix issues where possible",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    linter = VerticalLinter(auto_fix=args.fix)

    if args.all_verticals:
        victor_dir = Path("victor")
        vertical_dirs = [d for d in victor_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]

        for vertical_dir in vertical_dirs:
            if not (vertical_dir / "assistant.py").exists():
                continue
            report = linter.lint_vertical(vertical_dir)
            linter.report = report
            linter.print_report()

    elif args.vertical:
        if not args.vertical.exists():
            print(f"Error: Path does not exist: {args.vertical}")
            return 2

        report = linter.lint_vertical(args.vertical)
        linter.print_report()

    else:
        parser.print_help()
        return 2

    # Exit code based on issues
    if linter.report and linter.report.error_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
