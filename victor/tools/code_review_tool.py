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

"""Code review tool for automated code quality analysis.

Features:
- Code quality metrics (complexity, maintainability)
- Security vulnerability detection
- Best practices checking
- Performance issue detection
- Documentation coverage
- Test coverage analysis
"""

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import logging

from victor.tools.decorators import tool

logger = logging.getLogger(__name__)


# Module-level configuration
_max_complexity: int = 10


def set_code_review_config(max_complexity: int = 10) -> None:
    """Set code review configuration.

    Args:
        max_complexity: Maximum allowed cyclomatic complexity (default: 10).
    """
    global _max_complexity
    _max_complexity = max_complexity


# Security patterns to detect
SECURITY_PATTERNS = {
    "hardcoded_password": r"password\s*=\s*['\"][\w]+['\"]",
    "hardcoded_key": r"(api_key|secret_key|private_key)\s*=\s*['\"][\w]+['\"]",
    "sql_injection": r"(execute|cursor\.execute)\(['\"].*%s.*['\"]",
    "command_injection": r"(os\.system|subprocess\.call|eval|exec)\(",
    "insecure_random": r"random\.(random|randint|choice)",
    "weak_crypto": r"(md5|sha1)\(",
}

# Code smell patterns
CODE_SMELLS = {
    "print_debug": r"print\s*\(",
    "commented_code": r"^\s*#.*[{};]",
    "long_line": r".{121,}",  # Lines over 120 chars
    "multiple_returns": r"return\s+",
    "bare_except": r"except\s*:",
    "global_variable": r"^(global|GLOBAL)\s+\w+",
}


# Helper functions

def _check_security(content: str, file_path: Path) -> List[Dict[str, Any]]:
    """Check for security issues."""
    issues = []
    lines = content.split("\n")

    for line_num, line in enumerate(lines, 1):
        for issue_type, pattern in SECURITY_PATTERNS.items():
            if re.search(pattern, line, re.IGNORECASE):
                severity = _get_security_severity(issue_type)
                issues.append(
                    {
                        "type": "security",
                        "severity": severity,
                        "issue": issue_type.replace("_", " ").title(),
                        "file": str(file_path),
                        "line": line_num,
                        "code": line.strip(),
                        "recommendation": _get_security_recommendation(issue_type),
                    }
                )

    return issues


def _check_code_smells(content: str, file_path: Path) -> List[Dict[str, Any]]:
    """Check for code smells."""
    issues = []
    lines = content.split("\n")

    for line_num, line in enumerate(lines, 1):
        for smell_type, pattern in CODE_SMELLS.items():
            if re.search(pattern, line):
                issues.append(
                    {
                        "type": "code_smell",
                        "severity": "low",
                        "issue": smell_type.replace("_", " ").title(),
                        "file": str(file_path),
                        "line": line_num,
                        "code": line.strip()[:80],
                        "recommendation": _get_smell_recommendation(smell_type),
                    }
                )

    return issues


def _check_complexity(content: str, file_path: Path) -> List[Dict[str, Any]]:
    """Check cyclomatic complexity."""
    issues = []

    try:
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = _calculate_complexity(node)

                if complexity > _max_complexity:
                    issues.append(
                        {
                            "type": "complexity",
                            "severity": "medium" if complexity <= 15 else "high",
                            "issue": "High Complexity",
                            "file": str(file_path),
                            "line": node.lineno,
                            "code": f"Function: {node.name}",
                            "metric": complexity,
                            "recommendation": f"Refactor to reduce complexity (current: {complexity}, max: {_max_complexity})",
                        }
                    )

    except SyntaxError:
        logger.warning("Syntax error in %s, skipping complexity check", file_path)

    return issues


def _check_documentation(content: str, file_path: Path) -> List[Dict[str, Any]]:
    """Check documentation coverage."""
    issues = []

    try:
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    issues.append(
                        {
                            "type": "documentation",
                            "severity": "low",
                            "issue": "Missing Docstring",
                            "file": str(file_path),
                            "line": node.lineno,
                            "code": f"{node.__class__.__name__}: {node.name}",
                            "recommendation": "Add docstring to document purpose and usage",
                        }
                    )

    except SyntaxError:
        pass

    return issues


def _calculate_complexity(node: ast.AST) -> int:
    """Calculate cyclomatic complexity of a function."""
    complexity = 1  # Base complexity

    for child in ast.walk(node):
        # Each decision point adds 1
        if isinstance(
            child,
            (
                ast.If,
                ast.While,
                ast.For,
                ast.ExceptHandler,
                ast.With,
                ast.Assert,
                ast.BoolOp,
            ),
        ):
            complexity += 1

    return complexity


def _get_security_severity(issue_type: str) -> str:
    """Get severity level for security issue."""
    critical = ["sql_injection", "command_injection"]
    high = ["hardcoded_password", "hardcoded_key"]
    medium = ["weak_crypto", "insecure_random"]

    if issue_type in critical:
        return "critical"
    elif issue_type in high:
        return "high"
    elif issue_type in medium:
        return "medium"
    return "low"


def _get_security_recommendation(issue_type: str) -> str:
    """Get recommendation for security issue."""
    recommendations = {
        "hardcoded_password": "Use environment variables or secure vaults",
        "hardcoded_key": "Store keys in environment variables or secure key management",
        "sql_injection": "Use parameterized queries instead of string formatting",
        "command_injection": "Validate and sanitize inputs, use subprocess with list arguments",
        "insecure_random": "Use secrets module for cryptographic randomness",
        "weak_crypto": "Use SHA-256 or stronger algorithms",
    }
    return recommendations.get(issue_type, "Review and fix security issue")


def _get_smell_recommendation(smell_type: str) -> str:
    """Get recommendation for code smell."""
    recommendations = {
        "print_debug": "Use logging module instead of print statements",
        "commented_code": "Remove commented code, use version control",
        "long_line": "Split long lines for better readability (PEP 8: max 79 chars)",
        "multiple_returns": "Consider single return point or refactor",
        "bare_except": "Catch specific exceptions instead of bare except",
        "global_variable": "Avoid global variables, use function parameters",
    }
    return recommendations.get(smell_type, "Review and improve code quality")


def _build_report(
    file_path: Path,
    issues: List[Dict[str, Any]],
    include_metrics: bool,
    content: str,
) -> str:
    """Build code review report."""
    report = [f"Code Review Report: {file_path}"]
    report.append("=" * 70)
    report.append("")

    if not issues:
        report.append("✓ No issues found!")
        return "\n".join(report)

    # Group by severity
    critical = [i for i in issues if i.get("severity") == "critical"]
    high = [i for i in issues if i.get("severity") == "high"]
    medium = [i for i in issues if i.get("severity") == "medium"]
    low = [i for i in issues if i.get("severity") == "low"]

    report.append(f"Total Issues: {len(issues)}")
    report.append(f"  Critical: {len(critical)}")
    report.append(f"  High: {len(high)}")
    report.append(f"  Medium: {len(medium)}")
    report.append(f"  Low: {len(low)}")
    report.append("")

    # Detail issues by severity
    for severity, severity_issues in [
        ("CRITICAL", critical),
        ("HIGH", high),
        ("MEDIUM", medium),
        ("LOW", low),
    ]:
        if severity_issues:
            report.append(f"{severity} Issues:")
            report.append("-" * 70)
            for issue in severity_issues:
                report.append(f"  Line {issue['line']}: {issue['issue']}")
                report.append(f"  Code: {issue['code']}")
                report.append(f"  Recommendation: {issue['recommendation']}")
                report.append("")

    if include_metrics:
        lines = content.split("\n")
        report.append("Metrics:")
        report.append(f"  Lines of code: {len(lines)}")
        report.append(f"  Issue density: {len(issues) / max(len(lines), 1):.2f} issues/line")

    return "\n".join(report)


def _build_summary_report(
    dir_path: Path,
    file_count: int,
    issues: List[Dict[str, Any]],
    include_metrics: bool,
) -> str:
    """Build summary report for directory review."""
    report = [f"Code Review Summary: {dir_path}"]
    report.append("=" * 70)
    report.append("")
    report.append(f"Files Reviewed: {file_count}")
    report.append(f"Total Issues: {len(issues)}")
    report.append("")

    # Group by severity
    critical = [i for i in issues if i.get("severity") == "critical"]
    high = [i for i in issues if i.get("severity") == "high"]
    medium = [i for i in issues if i.get("severity") == "medium"]
    low = [i for i in issues if i.get("severity") == "low"]

    report.append("Issue Breakdown:")
    report.append(f"  Critical: {len(critical)}")
    report.append(f"  High: {len(high)}")
    report.append(f"  Medium: {len(medium)}")
    report.append(f"  Low: {len(low)}")
    report.append("")

    # Group by type
    by_type: Dict[str, int] = {}
    for issue in issues:
        issue_type = issue.get("type", "unknown")
        by_type[issue_type] = by_type.get(issue_type, 0) + 1

    report.append("Issues by Type:")
    for issue_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
        report.append(f"  {issue_type}: {count}")
    report.append("")

    # Top files with issues
    by_file: Dict[str, int] = {}
    for issue in issues:
        file_name = issue.get("file", "unknown")
        by_file[file_name] = by_file.get(file_name, 0) + 1

    if by_file:
        report.append("Top Files with Issues:")
        for file_name, count in sorted(by_file.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]:
            report.append(f"  {Path(file_name).name}: {count} issues")

    return "\n".join(report)


def _build_security_report(path: Path, issues: List[Dict[str, Any]]) -> str:
    """Build security scan report."""
    report = [f"Security Scan Report: {path}"]
    report.append("=" * 70)
    report.append("")

    if not issues:
        report.append("✓ No security issues found!")
        return "\n".join(report)

    report.append(f"Total Security Issues: {len(issues)}")
    report.append("")

    # Group by severity
    critical = [i for i in issues if i.get("severity") == "critical"]
    high = [i for i in issues if i.get("severity") == "high"]
    medium = [i for i in issues if i.get("severity") == "medium"]

    for severity, severity_issues in [
        ("CRITICAL", critical),
        ("HIGH", high),
        ("MEDIUM", medium),
    ]:
        if severity_issues:
            report.append(f"{severity} Security Issues:")
            report.append("-" * 70)
            for issue in severity_issues:
                report.append(f"  {issue['issue']} - {issue['file']}:{issue['line']}")
                report.append(f"  Code: {issue['code']}")
                report.append(f"  Fix: {issue['recommendation']}")
                report.append("")

    return "\n".join(report)


def _build_complexity_report(path: Path, complexity_data: List[Dict[str, Any]]) -> str:
    """Build complexity analysis report."""
    report = [f"Complexity Analysis Report: {path}"]
    report.append("=" * 70)
    report.append("")

    if not complexity_data:
        report.append("✓ No complexity issues found!")
        return "\n".join(report)

    report.append(f"Functions with High Complexity: {len(complexity_data)}")
    report.append("")

    # Sort by complexity
    sorted_data = sorted(
        complexity_data, key=lambda x: x.get("metric", 0), reverse=True
    )

    for item in sorted_data[:20]:  # Top 20
        report.append(
            f"  {item['code']} - Complexity: {item.get('metric', 'N/A')}"
        )
        report.append(f"    File: {item['file']}:{item['line']}")
        report.append(f"    Recommendation: {item['recommendation']}")
        report.append("")

    return "\n".join(report)


def _build_best_practices_report(path: Path, issues: List[Dict[str, Any]]) -> str:
    """Build best practices report."""
    report = [f"Best Practices Report: {path}"]
    report.append("=" * 70)
    report.append("")

    if not issues:
        report.append("✓ All best practices followed!")
        return "\n".join(report)

    report.append(f"Total Best Practice Issues: {len(issues)}")
    report.append("")

    # Group by issue type
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for issue in issues:
        issue_type = issue.get("issue", "unknown")
        if issue_type not in by_type:
            by_type[issue_type] = []
        by_type[issue_type].append(issue)

    for issue_type, type_issues in sorted(
        by_type.items(), key=lambda x: len(x[1]), reverse=True
    ):
        report.append(f"{issue_type}: {len(type_issues)} occurrences")
        report.append(f"  Recommendation: {type_issues[0]['recommendation']}")
        report.append("")

    return "\n".join(report)


# Tool functions

@tool
async def code_review_file(
    path: str,
    include_metrics: bool = False,
) -> Dict[str, Any]:
    """
    Review a single file.

    Performs comprehensive code review including security checks,
    code smells, complexity analysis, and documentation coverage.

    Args:
        path: File path to review.
        include_metrics: Include detailed metrics (default: False).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - issues_count: Number of issues found
        - issues: List of all issues with details
        - formatted_report: Human-readable review report
        - error: Error message if failed
    """
    if not path:
        return {"success": False, "error": "Missing required parameter: path"}

    file_path = Path(path)
    if not file_path.exists():
        return {"success": False, "error": f"File not found: {path}"}

    if not file_path.is_file():
        return {"success": False, "error": f"Path is not a file: {path}"}

    # Read file content
    try:
        content = file_path.read_text()
    except Exception as e:
        return {"success": False, "error": f"Failed to read file: {e}"}

    # Perform review
    issues = []

    # Security issues
    security_issues = _check_security(content, file_path)
    issues.extend(security_issues)

    # Code smells
    smells = _check_code_smells(content, file_path)
    issues.extend(smells)

    # Complexity (for Python files)
    if file_path.suffix == ".py":
        complexity_issues = _check_complexity(content, file_path)
        issues.extend(complexity_issues)

        # Documentation
        doc_issues = _check_documentation(content, file_path)
        issues.extend(doc_issues)

    # Build report
    report = _build_report(file_path, issues, include_metrics, content)

    return {
        "success": True,
        "issues_count": len(issues),
        "issues": issues,
        "formatted_report": report
    }


@tool
async def code_review_directory(
    path: str,
    file_pattern: str = "*.py",
    include_metrics: bool = False,
) -> Dict[str, Any]:
    """
    Review all files in directory.

    Scans all matching files in the directory and provides
    a comprehensive summary of all issues found.

    Args:
        path: Directory path to review.
        file_pattern: File pattern for matching (default: *.py).
        include_metrics: Include detailed metrics (default: False).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - files_reviewed: Number of files reviewed
        - total_issues: Total number of issues found
        - issues: List of all issues with details
        - formatted_report: Human-readable summary report
        - error: Error message if failed
    """
    if not path:
        return {"success": False, "error": "Missing required parameter: path"}

    dir_path = Path(path)
    if not dir_path.exists():
        return {"success": False, "error": f"Directory not found: {path}"}

    # Find all matching files
    files = list(dir_path.rglob(file_pattern))

    if not files:
        return {
            "success": True,
            "files_reviewed": 0,
            "total_issues": 0,
            "issues": [],
            "message": f"No files matching pattern '{file_pattern}' found in {path}"
        }

    # Review each file
    all_issues = []
    file_count = 0

    for file_path in files:
        if file_path.is_file():
            try:
                content = file_path.read_text()
                file_issues = []

                # Security
                file_issues.extend(_check_security(content, file_path))

                # Code smells
                file_issues.extend(_check_code_smells(content, file_path))

                # Python-specific
                if file_path.suffix == ".py":
                    file_issues.extend(_check_complexity(content, file_path))
                    file_issues.extend(_check_documentation(content, file_path))

                all_issues.extend(file_issues)
                file_count += 1

            except Exception as e:
                logger.warning("Failed to review %s: %s", file_path, e)

    # Build summary report
    report = _build_summary_report(dir_path, file_count, all_issues, include_metrics)

    return {
        "success": True,
        "files_reviewed": file_count,
        "total_issues": len(all_issues),
        "issues": all_issues,
        "formatted_report": report
    }


@tool
async def code_review_security(
    path: str,
    severity: str = "low",
) -> Dict[str, Any]:
    """
    Perform security-focused scan.

    Scans for security vulnerabilities including hardcoded secrets,
    SQL injection risks, command injection, and weak crypto.

    Args:
        path: File or directory path to scan.
        severity: Minimum severity to report (low, medium, high, critical)
                 (default: low).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - security_issues: Number of security issues found
        - issues: List of security issues with details
        - formatted_report: Human-readable security report
        - error: Error message if failed
    """
    if not path:
        return {"success": False, "error": "Missing required parameter: path"}

    path_obj = Path(path)
    if not path_obj.exists():
        return {"success": False, "error": f"Path not found: {path}"}

    # Collect all security issues
    security_issues = []

    if path_obj.is_file():
        content = path_obj.read_text()
        security_issues = _check_security(content, path_obj)
    else:
        # Scan directory
        for file_path in path_obj.rglob("*.py"):
            if file_path.is_file():
                try:
                    content = file_path.read_text()
                    issues = _check_security(content, file_path)
                    security_issues.extend(issues)
                except Exception as e:
                    logger.warning("Failed to scan %s: %s", file_path, e)

    # Filter by severity
    severity_levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    min_severity = severity_levels.get(severity, 0)

    filtered_issues = [
        issue
        for issue in security_issues
        if severity_levels.get(issue.get("severity", "low"), 0) >= min_severity
    ]

    # Build security report
    report = _build_security_report(path_obj, filtered_issues)

    return {
        "success": True,
        "security_issues": len(filtered_issues),
        "issues": filtered_issues,
        "formatted_report": report
    }


@tool
async def code_review_complexity(path: str) -> Dict[str, Any]:
    """
    Analyze code complexity.

    Calculates cyclomatic complexity for all functions and identifies
    functions that exceed the complexity threshold.

    Args:
        path: File or directory path to analyze.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - high_complexity_count: Number of high complexity functions
        - complexity_data: List of complexity issues with details
        - formatted_report: Human-readable complexity report
        - error: Error message if failed
    """
    if not path:
        return {"success": False, "error": "Missing required parameter: path"}

    path_obj = Path(path)
    if not path_obj.exists():
        return {"success": False, "error": f"Path not found: {path}"}

    complexity_data = []

    if path_obj.is_file() and path_obj.suffix == ".py":
        content = path_obj.read_text()
        issues = _check_complexity(content, path_obj)
        complexity_data.extend(issues)
    else:
        for file_path in path_obj.rglob("*.py"):
            if file_path.is_file():
                try:
                    content = file_path.read_text()
                    issues = _check_complexity(content, file_path)
                    complexity_data.extend(issues)
                except Exception as e:
                    logger.warning("Failed to analyze %s: %s", file_path, e)

    # Build complexity report
    report = _build_complexity_report(path_obj, complexity_data)

    return {
        "success": True,
        "high_complexity_count": len(complexity_data),
        "complexity_data": complexity_data,
        "formatted_report": report
    }


@tool
async def code_review_best_practices(path: str) -> Dict[str, Any]:
    """
    Check coding best practices.

    Checks for common code smells including print debugging,
    long lines, commented code, and bare except clauses.

    Args:
        path: File or directory path to check.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - issues_count: Number of best practice issues found
        - issues: List of issues with details
        - formatted_report: Human-readable best practices report
        - error: Error message if failed
    """
    if not path:
        return {"success": False, "error": "Missing required parameter: path"}

    path_obj = Path(path)
    if not path_obj.exists():
        return {"success": False, "error": f"Path not found: {path}"}

    best_practice_issues = []

    if path_obj.is_file():
        content = path_obj.read_text()
        issues = _check_code_smells(content, path_obj)
        best_practice_issues.extend(issues)
    else:
        for file_path in path_obj.rglob("*.py"):
            if file_path.is_file():
                try:
                    content = file_path.read_text()
                    issues = _check_code_smells(content, file_path)
                    best_practice_issues.extend(issues)
                except Exception as e:
                    logger.warning("Failed to check %s: %s", file_path, e)

    # Build best practices report
    report = _build_best_practices_report(path_obj, best_practice_issues)

    return {
        "success": True,
        "issues_count": len(best_practice_issues),
        "issues": best_practice_issues,
        "formatted_report": report
    }


# Keep class for backward compatibility
class CodeReviewTool:
    """Deprecated: Use individual code_review_* functions instead."""

    def __init__(self, max_complexity: int = 10):
        """Initialize - deprecated."""
        import warnings
        warnings.warn(
            "CodeReviewTool class is deprecated. Use code_review_* functions instead.",
            DeprecationWarning,
            stacklevel=2
        )
        set_code_review_config(max_complexity=max_complexity)
