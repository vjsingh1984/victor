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
- Multi-language support (Python, JavaScript, TypeScript, Java, Go, Rust, C, C++, etc.)
- Code quality metrics (complexity, maintainability)
- Security vulnerability detection (language-specific patterns)
- Best practices checking (code smells, anti-patterns)
- Documentation coverage
- Tree-sitter AST analysis for accurate complexity calculation

Supported languages: Python, JavaScript, TypeScript, Java, Go, Rust, C, C++, C#,
Ruby, PHP, Kotlin, Swift, Scala, Bash, SQL, Lua, Elixir, Haskell, R
"""

import logging
from pathlib import Path
from typing import Any, Optional

from victor.tools.base import AccessMode, CostTier, DangerLevel, Priority
from victor.tools.decorators import tool
from victor.tools.language_analyzer import (
    AnalysisIssue,
    detect_language,
    get_analyzer_for_file,
    supported_extensions,
    EXTENSION_TO_LANGUAGE,
    LANGUAGE_GLOB_PATTERNS,
)

logger = logging.getLogger(__name__)

# Constants
_DEFAULT_MAX_COMPLEXITY: int = 10


def _issue_to_dict(issue: AnalysisIssue) -> dict[str, Any]:
    """Convert AnalysisIssue to dictionary format."""
    return {
        "type": issue.type,
        "severity": issue.severity,
        "issue": issue.issue,
        "file": issue.file,
        "line": issue.line,
        "code": issue.code,
        "recommendation": issue.recommendation,
        "metric": issue.metric,
    }


def _analyze_file(
    file_path: Path,
    aspects: list[str],
    max_complexity: int,
) -> list[dict[str, Any]]:
    """Analyze a single file using the language analyzer.

    Args:
        file_path: Path to the file
        aspects: List of aspects to check
        max_complexity: Maximum complexity threshold

    Returns:
        List of issues found
    """
    issues = []

    # Get language-specific analyzer
    analyzer = get_analyzer_for_file(file_path, max_complexity=max_complexity)
    if analyzer is None:
        # Unsupported file type - skip silently
        return []

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return []

    # Run analysis
    result = analyzer.analyze(content, file_path, aspects)

    if not result.success:
        logger.warning(f"Analysis failed for {file_path}: {result.error}")
        return []

    # Convert issues to dict format
    for issue in result.issues:
        issues.append(_issue_to_dict(issue))

    return issues


def _get_glob_patterns_for_languages(languages: Optional[list[str]] = None) -> list[str]:
    """Get glob patterns for specified languages or all supported languages.

    Args:
        languages: List of language names, or None for all

    Returns:
        List of glob patterns
    """
    if languages is None:
        # All supported extensions
        return [f"*{ext}" for ext in supported_extensions()]

    patterns = []
    for lang in languages:
        lang = lang.lower()
        if lang in LANGUAGE_GLOB_PATTERNS:
            patterns.append(LANGUAGE_GLOB_PATTERNS[lang])
        else:
            # Try to find extensions for this language
            for ext, lang_name in EXTENSION_TO_LANGUAGE.items():
                if lang_name == lang:
                    patterns.append(f"*{ext}")
    return patterns if patterns else ["*.py"]  # Default to Python


def _build_report(
    path: Path,
    files_reviewed: int,
    aspects: list[str],
    results: dict[str, dict[str, Any]],
    filtered_issues: list[dict[str, Any]],
    issues_by_severity: dict[str, int],
    languages_found: set[Any],
) -> str:
    """Build comprehensive code review report."""
    report = ["Code Review Report"]
    report.append("=" * 70)
    report.append("")
    report.append(f"Path: {path}")
    report.append(f"Files reviewed: {files_reviewed}")
    report.append(
        f"Languages found: {', '.join(sorted(languages_found)) if languages_found else 'None'}"
    )
    report.append(f"Aspects checked: {', '.join(aspects)}")
    report.append("")
    report.append(f"Total issues: {len(filtered_issues)}")
    report.append(f"  Critical: {issues_by_severity.get('critical', 0)}")
    report.append(f"  High: {issues_by_severity.get('high', 0)}")
    report.append(f"  Medium: {issues_by_severity.get('medium', 0)}")
    report.append(f"  Low: {issues_by_severity.get('low', 0)}")
    report.append("")

    # Security section
    if "security" in aspects:
        sec_issues = results.get("security", {}).get("issues", [])
        report.append("Security Issues:")
        report.append(f"  Found: {len(sec_issues)}")
        if sec_issues:
            for issue in sec_issues[:5]:
                report.append(
                    f"    {issue.get('severity', 'low').upper()}: {issue.get('file', '')} "
                    f"(line {issue.get('line', '?')})"
                )
                report.append(f"      {issue.get('issue', '')}: {issue.get('recommendation', '')}")
            if len(sec_issues) > 5:
                report.append(f"    ... and {len(sec_issues) - 5} more")
        report.append("")

    # Complexity section
    if "complexity" in aspects:
        comp_issues = results.get("complexity", {}).get("issues", [])
        report.append("Complexity Issues:")
        report.append(f"  High complexity functions: {len(comp_issues)}")
        if comp_issues:
            for issue in comp_issues[:5]:
                report.append(
                    f"    {issue.get('file', '')} - {issue.get('code', '')} "
                    f"(complexity: {issue.get('metric', '?')})"
                )
            if len(comp_issues) > 5:
                report.append(f"    ... and {len(comp_issues) - 5} more")
        report.append("")

    # Best practices section
    if "best_practices" in aspects:
        bp_issues = results.get("best_practices", {}).get("issues", [])
        report.append("Best Practice Issues:")
        report.append(f"  Found: {len(bp_issues)}")
        if bp_issues:
            for issue in bp_issues[:5]:
                report.append(f"    {issue.get('file', '')} (line {issue.get('line', '?')})")
                report.append(f"      {issue.get('issue', '')}: {issue.get('recommendation', '')}")
            if len(bp_issues) > 5:
                report.append(f"    ... and {len(bp_issues) - 5} more")
        report.append("")

    # Documentation section
    if "documentation" in aspects:
        doc_issues = results.get("documentation", {}).get("issues", [])
        report.append("Documentation Issues:")
        report.append(f"  Found: {len(doc_issues)}")
        if doc_issues:
            for issue in doc_issues[:5]:
                report.append(f"    {issue.get('file', '')} - {issue.get('code', '')}")
                report.append(f"      {issue.get('issue', '')}: {issue.get('recommendation', '')}")
            if len(doc_issues) > 5:
                report.append(f"    ... and {len(doc_issues) - 5} more")
        report.append("")

    # Summary
    if len(filtered_issues) == 0:
        report.append("No issues found. Code looks good!")
    else:
        report.append(f"Review complete. {len(filtered_issues)} issues require attention.")

    return "\n".join(report)


# Consolidated Tool Function


@tool(
    cost_tier=CostTier.LOW,
    category="analysis",
    keywords=[
        "review",
        "code",
        "quality",
        "security",
        "complexity",
        "best practices",
        "documentation",
    ],
    mandatory_keywords=["review code", "code review", "analyze code"],  # Force inclusion
    task_types=["analysis", "review"],  # Classification-aware selection
    stages=["analysis", "verification"],  # Conversation stages where relevant
    priority=Priority.MEDIUM,  # Task-specific analysis tool
    access_mode=AccessMode.READONLY,  # Only reads files
    danger_level=DangerLevel.SAFE,  # No side effects
)
async def code_review(
    path: str,
    aspects: Optional[list[str]] = None,
    file_pattern: Optional[str] = None,
    languages: Optional[list[str]] = None,
    severity: str = "low",
    include_metrics: bool = False,
    max_issues: int = 50,
) -> dict[str, Any]:
    """
    Comprehensive code review for automated quality analysis.

    Performs code review including security checks, complexity analysis,
    best practices validation, and documentation coverage. Consolidates
    multiple review aspects into a single unified interface.

    Supports 20+ programming languages: Python, JavaScript, TypeScript,
    Java, Go, Rust, C, C++, C#, Ruby, PHP, Kotlin, Swift, Scala, Bash,
    SQL, Lua, Elixir, Haskell, R.

    Args:
        path: File or directory path to review.
        aspects: List of review aspects to check. Options: "security", "complexity",
            "best_practices", "documentation", "all". Defaults to ["all"].
            Can be provided as a list or JSON string representation.
        file_pattern: Glob pattern for files to review (e.g., "*.py", "*.{js,ts}").
            If not provided, auto-detects based on 'languages' parameter or reviews
            all supported file types.
        languages: List of languages to review (e.g., ["python", "javascript"]).
            If not provided, reviews all supported languages.
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
        - languages_found: List of languages detected in reviewed files
        - issues_by_severity: Count of issues grouped by severity
        - formatted_report: Human-readable comprehensive review report
        - error: Error message if failed

    Examples:
        # Review for security only
        code_review("./src", aspects=["security"])

        # Comprehensive review of all files
        code_review("./", aspects=["all"])

        # Review only JavaScript/TypeScript files
        code_review("./src", languages=["javascript", "typescript"])

        # Review Python and Go with high severity only
        code_review("./src", languages=["python", "go"], severity="high")

        # Review specific file pattern
        code_review("./src", file_pattern="*.{py,js,ts}")
    """
    if not path:
        return {"success": False, "error": "Missing required parameter: path"}

    # Handle aspects parameter (can be list, JSON string, or single string)
    if aspects is None:
        aspects = ["all"]
    elif isinstance(aspects, str):  # type: ignore[unreachable]
        # Try to parse as JSON string, otherwise treat as single aspect
        import json  # type: ignore[unreachable]

        try:
            aspects = json.loads(aspects)
        except json.JSONDecodeError:
            # Not a JSON string, treat as single aspect
            aspects = [aspects]
    else:
        # Ensure aspects is a list
        if not isinstance(aspects, list):
            aspects = [aspects]  # type: ignore[unreachable]

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

    # Handle languages parameter (can be list or JSON string)
    if isinstance(languages, str):
        import json  # type: ignore[unreachable]

        try:
            languages = json.loads(languages)
        except json.JSONDecodeError:
            languages = [languages]

    path_obj = Path(path)
    if not path_obj.exists():
        return {"success": False, "error": f"Path not found: {path}"}

    # Determine file patterns
    if file_pattern:
        patterns = [file_pattern]
    else:
        patterns = _get_glob_patterns_for_languages(languages)

    # Collect files to review
    files_to_review = []
    if path_obj.is_file():
        files_to_review = [path_obj]
    else:
        # Collect files matching any pattern
        seen = set()
        for pattern in patterns:
            for f in path_obj.rglob(pattern):
                if f not in seen and f.is_file():
                    seen.add(f)
                    files_to_review.append(f)

    if not files_to_review:
        return {
            "success": True,
            "files_reviewed": 0,
            "total_issues": 0,
            "languages_found": [],
            "message": "No files found matching patterns",
        }

    # Initialize results
    results: dict[str, dict[str, Any]] = {aspect: {"issues": [], "count": 0} for aspect in aspects}
    all_issues = []
    files_reviewed = 0
    languages_found = set()

    # Review each file
    for file_path in files_to_review:
        # Detect language
        lang = detect_language(file_path)
        if lang is None:
            continue

        # Filter by language if specified
        if languages and lang not in [lang_item.lower() for lang_item in languages]:
            continue

        languages_found.add(lang)

        try:
            # Use language analyzer
            file_issues = _analyze_file(file_path, aspects, _DEFAULT_MAX_COMPLEXITY)
            files_reviewed += 1

            # Sort issues by aspect
            for issue in file_issues:
                issue_type = issue.get("type", "")
                all_issues.append(issue)

                if issue_type == "security" and "security" in aspects:
                    results["security"]["issues"].append(issue)
                elif issue_type == "complexity" and "complexity" in aspects:
                    results["complexity"]["issues"].append(issue)
                elif issue_type == "smell" and "best_practices" in aspects:
                    results["best_practices"]["issues"].append(issue)
                elif issue_type == "documentation" and "documentation" in aspects:
                    results["documentation"]["issues"].append(issue)

        except Exception as e:
            logger.warning(f"Failed to review {file_path}: {e}")

    # Update result counts
    for aspect in aspects:
        results[aspect]["count"] = len(results[aspect]["issues"])

    # Filter by severity
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
    formatted_report = _build_report(
        path_obj,
        files_reviewed,
        aspects,
        results,
        filtered_issues,
        issues_by_severity,
        languages_found,
    )

    return {
        "success": True,
        "aspects_checked": aspects,
        "results": results,
        "total_issues": len(filtered_issues),
        "files_reviewed": files_reviewed,
        "languages_found": list(languages_found),
        "issues_by_severity": issues_by_severity,
        "issues": filtered_issues,
        "formatted_report": formatted_report,
    }
