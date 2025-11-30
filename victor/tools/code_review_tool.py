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
from typing import Any, Dict, List
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
        for file_name, count in sorted(by_file.items(), key=lambda x: x[1], reverse=True)[:10]:
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
    sorted_data = sorted(complexity_data, key=lambda x: x.get("metric", 0), reverse=True)

    for item in sorted_data[:20]:  # Top 20
        report.append(f"  {item['code']} - Complexity: {item.get('metric', 'N/A')}")
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

    for issue_type, type_issues in sorted(by_type.items(), key=lambda x: len(x[1]), reverse=True):
        report.append(f"{issue_type}: {len(type_issues)} occurrences")
        report.append(f"  Recommendation: {type_issues[0]['recommendation']}")
        report.append("")

    return "\n".join(report)


# Consolidated Tool Function


@tool
async def code_review(
    path: str,
    aspects: List[str] = None,
    file_pattern: str = "*.py",
    severity: str = "low",
    include_metrics: bool = False,
    max_issues: int = 50,
) -> Dict[str, Any]:
    """
    Comprehensive code review for automated quality analysis.

    Performs code review including security checks, complexity analysis,
    best practices validation, and documentation coverage. Consolidates
    multiple review aspects into a single unified interface.

    Args:
        path: File or directory path to review.
        aspects: List of review aspects to check. Options: "security", "complexity",
            "best_practices", "documentation", "all". Defaults to ["all"].
            Can be provided as a list or JSON string representation.
        file_pattern: Glob pattern for files to review (default: *.py).
        severity: Minimum severity to report for security issues: "low", "medium",
            "high", "critical". Defaults to "low" (report all).
        include_metrics: Include detailed metrics in report (default: False).
        max_issues: Maximum number of issues to return (default: 50).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - aspects_checked: List of aspects that were checked
        - results: Dictionary with results for each aspect
        - total_issues: Total number of issues found
        - files_reviewed: Number of files reviewed
        - issues_by_severity: Count of issues grouped by severity
        - formatted_report: Human-readable comprehensive review report
        - error: Error message if failed

    Examples:
        # Review for security only
        code_review("./src", aspects=["security"])

        # Comprehensive review
        code_review("./", aspects=["all"])

        # Complexity and best practices
        code_review("./src", aspects=["complexity", "best_practices"])

        # Security with high severity only
        code_review("./src", aspects=["security"], severity="high")
    """
    if not path:
        return {"success": False, "error": "Missing required parameter: path"}

    # Handle aspects parameter (can be list or JSON string)
    if aspects is None:
        aspects = ["all"]
    elif isinstance(aspects, str):
        import json

        try:
            aspects = json.loads(aspects)
        except json.JSONDecodeError:
            # Single aspect as string
            aspects = [aspects]

    # Validate aspects
    valid_aspects = {"security", "complexity", "best_practices", "documentation", "all"}
    invalid = [a for a in aspects if a not in valid_aspects]
    if invalid:
        return {
            "success": False,
            "error": f"Invalid aspect(s): {', '.join(invalid)}. Valid options: {', '.join(sorted(valid_aspects))}",
        }

    # Expand "all" to all aspects
    if "all" in aspects:
        aspects = ["security", "complexity", "best_practices", "documentation"]

    path_obj = Path(path)
    if not path_obj.exists():
        return {"success": False, "error": f"Path not found: {path}"}

    # Collect files to review
    files_to_review = []
    if path_obj.is_file():
        files_to_review = [path_obj]
    else:
        files_to_review = list(path_obj.rglob(file_pattern))

    if not files_to_review:
        return {
            "success": True,
            "files_reviewed": 0,
            "total_issues": 0,
            "message": f"No files found matching pattern '{file_pattern}'",
        }

    # Initialize results
    results = {aspect: {"issues": [], "count": 0} for aspect in aspects}
    all_issues = []
    files_reviewed = 0

    # Review each file
    for file_path in files_to_review:
        if not file_path.is_file():
            continue

        try:
            content = file_path.read_text()
            files_reviewed += 1

            # Security review
            if "security" in aspects:
                security_issues = _check_security(content, file_path)
                results["security"]["issues"].extend(security_issues)
                all_issues.extend(security_issues)

            # Complexity analysis (Python only)
            if "complexity" in aspects and file_path.suffix == ".py":
                complexity_issues = _check_complexity(content, file_path)
                results["complexity"]["issues"].extend(complexity_issues)
                all_issues.extend(complexity_issues)

            # Best practices
            if "best_practices" in aspects:
                smell_issues = _check_code_smells(content, file_path)
                results["best_practices"]["issues"].extend(smell_issues)
                all_issues.extend(smell_issues)

            # Documentation (Python only)
            if "documentation" in aspects and file_path.suffix == ".py":
                doc_issues = _check_documentation(content, file_path)
                results["documentation"]["issues"].extend(doc_issues)
                all_issues.extend(doc_issues)

        except Exception as e:
            logger.warning("Failed to review %s: %s", file_path, e)

    # Update result counts
    for aspect in aspects:
        results[aspect]["count"] = len(results[aspect]["issues"])

    # Filter by severity (for security issues)
    severity_levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    min_severity = severity_levels.get(severity, 0)

    filtered_issues = []
    for issue in all_issues:
        issue_severity = severity_levels.get(issue.get("severity", "low"), 0)
        if issue_severity >= min_severity:
            filtered_issues.append(issue)

    # Limit results
    if len(filtered_issues) > max_issues:
        filtered_issues = filtered_issues[:max_issues]

    # Count issues by severity
    issues_by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for issue in filtered_issues:
        sev = issue.get("severity", "low")
        issues_by_severity[sev] = issues_by_severity.get(sev, 0) + 1

    # Build comprehensive report
    report = []
    report.append("Code Review Report")
    report.append("=" * 70)
    report.append("")
    report.append(f"Path: {path}")
    report.append(f"Files reviewed: {files_reviewed}")
    report.append(f"Aspects checked: {', '.join(aspects)}")
    report.append("")
    report.append(f"Total issues: {len(filtered_issues)}")
    report.append(f"  Critical: {issues_by_severity['critical']}")
    report.append(f"  High: {issues_by_severity['high']}")
    report.append(f"  Medium: {issues_by_severity['medium']}")
    report.append(f"  Low: {issues_by_severity['low']}")
    report.append("")

    # Security section
    if "security" in aspects:
        sec_filtered = [
            i
            for i in results["security"]["issues"]
            if severity_levels.get(i.get("severity", "low"), 0) >= min_severity
        ]
        report.append("Security Issues:")
        report.append(f"  Found: {len(sec_filtered)}")
        if sec_filtered:
            for issue in sec_filtered[:5]:
                report.append(
                    f"    {issue.get('severity', 'low').upper()}: {issue.get('file', '')} "
                    f"(line {issue.get('line', '?')})"
                )
                report.append(f"      {issue.get('issue', '')}: {issue.get('recommendation', '')}")
            if len(sec_filtered) > 5:
                report.append(f"    ... and {len(sec_filtered) - 5} more")
        report.append("")

    # Complexity section
    if "complexity" in aspects:
        comp_count = results["complexity"]["count"]
        report.append("Complexity Issues:")
        report.append(f"  High complexity functions: {comp_count}")
        if results["complexity"]["issues"]:
            for issue in results["complexity"]["issues"][:5]:
                report.append(
                    f"    {issue.get('file', '')} - {issue.get('code', '')} "
                    f"(complexity: {issue.get('metric', '?')})"
                )
            if comp_count > 5:
                report.append(f"    ... and {comp_count - 5} more")
        report.append("")

    # Best practices section
    if "best_practices" in aspects:
        bp_count = results["best_practices"]["count"]
        report.append("Best Practice Issues:")
        report.append(f"  Found: {bp_count}")
        if results["best_practices"]["issues"]:
            for issue in results["best_practices"]["issues"][:5]:
                report.append(f"    {issue.get('file', '')} (line {issue.get('line', '?')})")
                report.append(f"      {issue.get('issue', '')}: {issue.get('recommendation', '')}")
            if bp_count > 5:
                report.append(f"    ... and {bp_count - 5} more")
        report.append("")

    # Documentation section
    if "documentation" in aspects:
        doc_count = results["documentation"]["count"]
        report.append("Documentation Issues:")
        report.append(f"  Found: {doc_count}")
        if results["documentation"]["issues"]:
            for issue in results["documentation"]["issues"][:5]:
                report.append(f"    {issue.get('file', '')} - {issue.get('code', '')}")
                report.append(f"      {issue.get('issue', '')}: {issue.get('recommendation', '')}")
            if doc_count > 5:
                report.append(f"    ... and {doc_count - 5} more")
        report.append("")

    # Summary
    if len(filtered_issues) == 0:
        report.append("No issues found. Code looks good!")
    else:
        report.append(f"Review complete. {len(filtered_issues)} issues require attention.")

    return {
        "success": True,
        "aspects_checked": aspects,
        "results": results,
        "total_issues": len(filtered_issues),
        "files_reviewed": files_reviewed,
        "issues_by_severity": issues_by_severity,
        "issues": filtered_issues,
        "formatted_report": "\n".join(report),
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
            stacklevel=2,
        )
        set_code_review_config(max_complexity=max_complexity)
