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

"""Code metrics and quality analysis tool.

Features:
- Cyclomatic complexity calculation
- Maintainability index
- Technical debt estimation
- Performance profiling
- Comprehensive code analysis
- Quality reports
"""

import ast
from pathlib import Path
from typing import Any, Dict
import logging

from victor.tools.decorators import tool

logger = logging.getLogger(__name__)


def _calculate_complexity_score(code: str) -> int:
    """Calculate cyclomatic complexity of code."""
    try:
        tree = ast.parse(code)
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity
    except:
        return 0


def _calculate_maintainability_index(code: str) -> float:
    """Calculate maintainability index (0-100)."""
    try:
        tree = ast.parse(code)
        lines = len([l for l in code.split('\n') if l.strip()])

        # Simplified maintainability calculation
        complexity = _calculate_complexity_score(code)

        # MI = max(0, (171 - 5.2 * ln(volume) - 0.23 * complexity - 16.2 * ln(lines)) * 100 / 171)
        # Simplified version
        mi = 100 - (complexity * 2) - (lines / 10)
        return max(0, min(100, mi))
    except:
        return 0.0


@tool
async def metrics_complexity(file: str, threshold: int = 10) -> Dict[str, Any]:
    """
    Calculate cyclomatic complexity of code.

    Analyzes code structure to determine complexity score.
    Higher scores indicate more complex, harder to maintain code.

    Args:
        file: Path to source file.
        threshold: Complexity threshold for warnings (default: 10).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - complexity: Complexity score
        - threshold: Threshold value
        - status: 'ok' or 'warning' based on threshold
        - message: Interpretation message
        - error: Error message if failed
    """
    if not file:
        return {"success": False, "error": "Missing required parameter: file"}

    file_path = Path(file)
    if not file_path.exists():
        return {"success": False, "error": f"File not found: {file}"}

    try:
        code = file_path.read_text()
        complexity = _calculate_complexity_score(code)

        status = "ok" if complexity <= threshold else "warning"
        message = f"Complexity: {complexity} ({'OK' if status == 'ok' else 'High - consider refactoring'})"

        return {
            "success": True,
            "complexity": complexity,
            "threshold": threshold,
            "status": status,
            "message": message
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to analyze: {str(e)}"}


@tool
async def metrics_maintainability(file: str) -> Dict[str, Any]:
    """
    Calculate maintainability index.

    Measures how maintainable the code is on a scale of 0-100.
    Higher scores indicate more maintainable code.

    Args:
        file: Path to source file.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - maintainability_index: Score 0-100
        - rating: 'excellent', 'good', 'fair', or 'poor'
        - message: Interpretation message
        - error: Error message if failed
    """
    if not file:
        return {"success": False, "error": "Missing required parameter: file"}

    file_path = Path(file)
    if not file_path.exists():
        return {"success": False, "error": f"File not found: {file}"}

    try:
        code = file_path.read_text()
        mi = _calculate_maintainability_index(code)

        if mi >= 80:
            rating = "excellent"
        elif mi >= 60:
            rating = "good"
        elif mi >= 40:
            rating = "fair"
        else:
            rating = "poor"

        return {
            "success": True,
            "maintainability_index": round(mi, 2),
            "rating": rating,
            "message": f"Maintainability: {mi:.2f}/100 ({rating})"
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to analyze: {str(e)}"}


@tool
async def metrics_debt(file: str) -> Dict[str, Any]:
    """
    Estimate technical debt.

    Estimates technical debt based on code quality indicators.

    Args:
        file: Path to source file.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - debt_hours: Estimated hours to address issues
        - debt_level: 'low', 'medium', or 'high'
        - issues: List of identified issues
        - error: Error message if failed
    """
    if not file:
        return {"success": False, "error": "Missing required parameter: file"}

    file_path = Path(file)
    if not file_path.exists():
        return {"success": False, "error": f"File not found: {file}"}

    try:
        code = file_path.read_text()
        complexity = _calculate_complexity_score(code)
        mi = _calculate_maintainability_index(code)

        issues = []
        debt_hours = 0

        if complexity > 20:
            issues.append("High complexity detected")
            debt_hours += 4
        elif complexity > 10:
            issues.append("Moderate complexity")
            debt_hours += 2

        if mi < 40:
            issues.append("Low maintainability")
            debt_hours += 8
        elif mi < 60:
            issues.append("Below average maintainability")
            debt_hours += 4

        debt_level = "low" if debt_hours < 4 else ("medium" if debt_hours < 12 else "high")

        return {
            "success": True,
            "debt_hours": debt_hours,
            "debt_level": debt_level,
            "issues": issues if issues else ["No major issues detected"],
            "message": f"Technical debt: {debt_hours} hours ({debt_level})"
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to analyze: {str(e)}"}


@tool
async def metrics_profile(file: str) -> Dict[str, Any]:
    """
    Profile code performance.

    Basic performance analysis of code structure.

    Args:
        file: Path to source file.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - lines: Total lines of code
        - functions: Number of functions
        - classes: Number of classes
        - message: Summary message
        - error: Error message if failed
    """
    if not file:
        return {"success": False, "error": "Missing required parameter: file"}

    file_path = Path(file)
    if not file_path.exists():
        return {"success": False, "error": f"File not found: {file}"}

    try:
        code = file_path.read_text()
        tree = ast.parse(code)

        functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        lines = len([l for l in code.split('\n') if l.strip()])

        return {
            "success": True,
            "lines": lines,
            "functions": functions,
            "classes": classes,
            "message": f"Code profile: {lines} lines, {functions} functions, {classes} classes"
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to profile: {str(e)}"}


@tool
async def metrics_analyze(file: str) -> Dict[str, Any]:
    """
    Comprehensive code analysis.

    Performs complete analysis including complexity, maintainability,
    and technical debt estimation.

    Args:
        file: Path to source file.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - complexity: Complexity analysis
        - maintainability: Maintainability analysis
        - debt: Technical debt analysis
        - profile: Code profile
        - formatted_report: Human-readable report
        - error: Error message if failed
    """
    if not file:
        return {"success": False, "error": "Missing required parameter: file"}

    # Run all analyses
    complexity_result = await metrics_complexity(file=file)
    maintainability_result = await metrics_maintainability(file=file)
    debt_result = await metrics_debt(file=file)
    profile_result = await metrics_profile(file=file)

    if not all(r.get("success") for r in [complexity_result, maintainability_result, debt_result, profile_result]):
        return {"success": False, "error": "Analysis failed"}

    # Build report
    report = []
    report.append(f"Code Analysis Report: {file}")
    report.append("=" * 70)
    report.append("")
    report.append(f"Complexity: {complexity_result['complexity']} ({complexity_result['status']})")
    report.append(f"Maintainability: {maintainability_result['maintainability_index']}/100 ({maintainability_result['rating']})")
    report.append(f"Technical Debt: {debt_result['debt_hours']} hours ({debt_result['debt_level']})")
    report.append("")
    report.append(f"Profile: {profile_result['lines']} lines, {profile_result['functions']} functions, {profile_result['classes']} classes")

    if debt_result['issues']:
        report.append("")
        report.append("Issues:")
        for issue in debt_result['issues']:
            report.append(f"  • {issue}")

    return {
        "success": True,
        "complexity": complexity_result,
        "maintainability": maintainability_result,
        "debt": debt_result,
        "profile": profile_result,
        "formatted_report": "\n".join(report)
    }


@tool
async def metrics_report(file: str) -> Dict[str, Any]:
    """
    Generate quality report.

    Creates a detailed quality report with recommendations.

    Args:
        file: Path to source file.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - report: Comprehensive quality report
        - recommendations: List of recommendations
        - error: Error message if failed
    """
    # Use analyze to get data
    analysis = await metrics_analyze(file=file)

    if not analysis["success"]:
        return analysis

    # Generate recommendations
    recommendations = []

    complexity = analysis["complexity"]["complexity"]
    mi = analysis["maintainability"]["maintainability_index"]

    if complexity > 10:
        recommendations.append("Consider breaking down complex functions")
    if mi < 60:
        recommendations.append("Improve code maintainability through refactoring")
    if analysis["debt"]["debt_hours"] > 8:
        recommendations.append("Significant technical debt - prioritize refactoring")

    if not recommendations:
        recommendations.append("Code quality is good - maintain current standards")

    return {
        "success": True,
        "report": analysis["formatted_report"],
        "recommendations": recommendations,
        "formatted_report": analysis["formatted_report"] + "\n\nRecommendations:\n" + "\n".join(f"  • {r}" for r in recommendations)
    }


# Keep class for backward compatibility
class MetricsTool:
    """Deprecated: Use individual metrics_* functions instead."""

    def __init__(self):
        """Initialize - deprecated."""
        import warnings
        warnings.warn(
            "MetricsTool class is deprecated. Use metrics_* functions instead.",
            DeprecationWarning,
            stacklevel=2
        )
