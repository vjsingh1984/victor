"""
Core review engine that orchestrates code analysis using Victor AI.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from victor.coding.ast import Parser
from victor.coding.review import CodeReviewer


@dataclass
class Issue:
    """Represents a code issue found during review."""

    file: str
    line: int
    column: int
    severity: str
    category: str
    message: str
    suggestion: Optional[str] = None
    confidence: float = 1.0


@dataclass
class ReviewResult:
    """Results from a code review."""

    files_analyzed: int
    total_issues: int
    issues_by_severity: Dict[str, int]
    issues_by_type: Dict[str, int]
    issues: List[Issue]


class ReviewEngine:
    """Main review engine using Victor AI capabilities."""

    def __init__(self, orchestrator, config):
        """Initialize review engine.

        Args:
            orchestrator: Victor AI agent orchestrator
            config: Review configuration
        """
        self.orchestrator = orchestrator
        self.config = config
        self.parser = Parser()
        self.code_reviewer = CodeReviewer(orchestrator)

    async def review(self, target_path: Path, **options) -> Dict[str, Any]:
        """Perform code review on target path.

        Args:
            target_path: Path to file or directory
            **options: Review options (recursive, severity, checks, etc.)

        Returns:
            Review results dictionary
        """
        # Collect files to review
        files = self._collect_files(target_path, **options)

        if not files:
            return {
                "files_analyzed": 0,
                "total_issues": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "issues_by_type": {},
                "issues": [],
                "top_issues": [],
            }

        # Run checks on each file
        all_issues = []
        for file_path in files:
            issues = await self._review_file(file_path, **options)
            all_issues.extend(issues)

        # Aggregate results
        return self._aggregate_results(files, all_issues)

    def _collect_files(self, target_path: Path, **options) -> List[Path]:
        """Collect files to review."""
        files = []

        if target_path.is_file():
            return [target_path]

        if not options.get("recursive", False):
            # Only top-level files
            for ext in [".py", ".js", ".ts", ".go", ".rs", ".java"]:
                files.extend(target_path.glob(f"*{ext}"))
        else:
            # Recursive search
            for ext in [".py", ".js", ".ts", ".go", ".rs", ".java"]:
                files.extend(target_path.rglob(f"*{ext}"))

        # Filter by ignore patterns
        ignore_patterns = options.get("ignore_patterns", self.config.ignore_patterns)
        if ignore_patterns:
            files = self._filter_ignored(files, ignore_patterns)

        return files

    def _filter_ignored(self, files: List[Path], patterns: List[str]) -> List[Path]:
        """Filter files by ignore patterns."""
        import fnmatch

        filtered = []
        for file_path in files:
            ignored = False
            for pattern in patterns:
                if fnmatch.fnmatch(str(file_path), pattern):
                    ignored = True
                    break
            if not ignored:
                filtered.append(file_path)

        return filtered

    async def _review_file(self, file_path: Path, **options) -> List[Issue]:
        """Review a single file."""
        issues = []

        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Determine language
            language = self._get_language(file_path)

            # Parse AST
            ast = self.parser.parse(content, language=language)

            # Run enabled checks
            enabled_checks = options.get("checks", self.config.enabled_checks)

            if "security" in enabled_checks:
                security_issues = await self._security_check(file_path, content, ast)
                issues.extend(security_issues)

            if "style" in enabled_checks:
                style_issues = await self._style_check(file_path, content)
                issues.extend(style_issues)

            if "complexity" in enabled_checks:
                complexity_issues = await self._complexity_check(file_path, ast)
                issues.extend(complexity_issues)

            if "quality" in enabled_checks:
                quality_issues = await self._quality_check(file_path, content, ast)
                issues.extend(quality_issues)

        except Exception as e:
            # Add error as an issue
            issues.append(
                Issue(
                    file=str(file_path),
                    line=0,
                    column=0,
                    severity="low",
                    category="error",
                    message=f"Failed to analyze file: {str(e)}",
                    confidence=1.0,
                )
            )

        return issues

    async def _security_check(self, file_path: Path, content: str, ast) -> List[Issue]:
        """Perform security vulnerability scan."""
        issues = []

        # Use Victor AI's code reviewer for security analysis
        security_result = await self.code_reviewer.security_scan(
            file_path=str(file_path), content=content
        )

        for finding in security_result.findings:
            issues.append(
                Issue(
                    file=str(file_path),
                    line=finding.line,
                    column=finding.column,
                    severity=finding.severity,
                    category="security",
                    message=finding.message,
                    suggestion=finding.suggestion,
                    confidence=finding.confidence,
                )
            )

        return issues

    async def _style_check(self, file_path: Path, content: str) -> List[Issue]:
        """Perform style checking."""
        issues = []
        lines = content.split("\n")

        max_line_length = self.config.max_line_length

        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > max_line_length:
                issues.append(
                    Issue(
                        file=str(file_path),
                        line=i,
                        column=max_line_length,
                        severity="low",
                        category="style",
                        message=f"Line too long ({len(line)} > {max_line_length} characters)",
                        suggestion=f"Break line into multiple lines",
                        confidence=1.0,
                    )
                )

            # Check for trailing whitespace
            if line.rstrip() != line:
                issues.append(
                    Issue(
                        file=str(file_path),
                        line=i,
                        column=len(line),
                        severity="low",
                        category="style",
                        message="Trailing whitespace",
                        suggestion="Remove trailing whitespace",
                        confidence=1.0,
                    )
                )

        return issues

    async def _complexity_check(self, file_path: Path, ast) -> List[Issue]:
        """Perform complexity analysis."""
        issues = []

        # Use Victor AI's complexity analyzer
        complexity_result = await self.code_reviewer.complexity_analysis(
            file_path=str(file_path), ast=ast
        )

        max_complexity = self.config.max_complexity

        for func_name, complexity in complexity_result.functions.items():
            if complexity > max_complexity:
                # Find function line number from AST
                line = complexity_result.function_lines.get(func_name, 0)

                issues.append(
                    Issue(
                        file=str(file_path),
                        line=line,
                        column=0,
                        severity="medium",
                        category="complexity",
                        message=f"Function '{func_name}' has complexity {complexity} (threshold: {max_complexity})",
                        suggestion="Consider refactoring into smaller functions",
                        confidence=1.0,
                    )
                )

        return issues

    async def _quality_check(self, file_path: Path, content: str, ast) -> List[Issue]:
        """Perform quality analysis."""
        issues = []

        # Use Victor AI for quality assessment
        quality_result = await self.code_reviewer.quality_check(
            file_path=str(file_path), content=content, ast=ast
        )

        for issue in quality_result.issues:
            issues.append(
                Issue(
                    file=str(file_path),
                    line=issue.line,
                    column=issue.column,
                    severity=issue.severity,
                    category="quality",
                    message=issue.message,
                    suggestion=issue.suggestion,
                    confidence=issue.confidence,
                )
            )

        return issues

    def _get_language(self, file_path: Path) -> str:
        """Determine programming language from file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
        }
        return ext_map.get(file_path.suffix, "python")

    def _aggregate_results(self, files: List[Path], issues: List[Issue]) -> Dict[str, Any]:
        """Aggregate review results."""
        # Count by severity
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for issue in issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

        # Count by type
        type_counts = {}
        for issue in issues:
            type_counts[issue.category] = type_counts.get(issue.category, 0) + 1

        # Sort issues by severity and confidence
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_issues = sorted(
            issues, key=lambda i: (severity_order.get(i.severity, 4), -i.confidence)
        )

        # Convert issues to dicts for output
        top_issues = [
            {
                "file": issue.file,
                "line": issue.line,
                "column": issue.column,
                "severity": issue.severity,
                "category": issue.category,
                "message": issue.message,
                "suggestion": issue.suggestion,
                "confidence": issue.confidence,
            }
            for issue in sorted_issues[:50]  # Top 50 issues
        ]

        return {
            "files_analyzed": len(files),
            "total_issues": len(issues),
            "critical": severity_counts.get("critical", 0),
            "high": severity_counts.get("high", 0),
            "medium": severity_counts.get("medium", 0),
            "low": severity_counts.get("low", 0),
            "issues_by_type": type_counts,
            "issues": top_issues,
            "top_issues": top_issues,
        }
