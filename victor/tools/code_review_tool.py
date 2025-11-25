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

from victor.tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class CodeReviewTool(BaseTool):
    """Tool for automated code review and quality analysis."""

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

    def __init__(self, max_complexity: int = 10):
        """Initialize code review tool.

        Args:
            max_complexity: Maximum allowed cyclomatic complexity
        """
        super().__init__()
        self.max_complexity = max_complexity

    @property
    def name(self) -> str:
        """Get tool name."""
        return "code_review"

    @property
    def description(self) -> str:
        """Get tool description."""
        return """Automated code review and quality analysis.

Perform comprehensive code review including:
- Code quality metrics
- Security vulnerability detection
- Best practices checking
- Performance issue detection
- Documentation coverage
- Complexity analysis

Operations:
- review_file: Review a single file
- review_directory: Review all files in directory
- security_scan: Focus on security issues
- complexity: Analyze complexity metrics
- best_practices: Check coding standards

Example workflows:
1. Review single file:
   code_review(operation="review_file", path="api/routes.py")

2. Security scan:
   code_review(operation="security_scan", path="src/")

3. Complexity analysis:
   code_review(operation="complexity", path="core/")

4. Full review:
   code_review(operation="review_directory", path="src/", include_metrics=True)
"""

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameters."""
        return self.convert_parameters_to_schema(
        [
            ToolParameter(
                name="operation",
                type="string",
                description="Operation: review_file, review_directory, security_scan, complexity, best_practices",
                required=True,
            ),
            ToolParameter(
                name="path",
                type="string",
                description="File or directory path to review",
                required=True,
            ),
            ToolParameter(
                name="include_metrics",
                type="boolean",
                description="Include detailed metrics (default: false)",
                required=False,
            ),
            ToolParameter(
                name="severity",
                type="string",
                description="Minimum severity: low, medium, high, critical (default: low)",
                required=False,
            ),
            ToolParameter(
                name="file_pattern",
                type="string",
                description="File pattern for directory review (default: *.py)",
                required=False,
            ),
        ]
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute code review operation.

        Args:
            operation: Operation to perform
            **kwargs: Operation-specific parameters

        Returns:
            Tool result with review findings
        """
        operation = kwargs.get("operation")

        if not operation:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: operation",
            )

        try:
            if operation == "review_file":
                return await self._review_file(kwargs)
            elif operation == "review_directory":
                return await self._review_directory(kwargs)
            elif operation == "security_scan":
                return await self._security_scan(kwargs)
            elif operation == "complexity":
                return await self._complexity_analysis(kwargs)
            elif operation == "best_practices":
                return await self._best_practices(kwargs)
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown operation: {operation}",
                )

        except Exception as e:
            logger.exception("Code review failed")
            return ToolResult(
                success=False, output="", error=f"Code review error: {str(e)}"
            )

    async def _review_file(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Review a single file."""
        path = kwargs.get("path")
        include_metrics = kwargs.get("include_metrics", False)

        if not path:
            return ToolResult(
                success=False, output="", error="Missing required parameter: path"
            )

        file_path = Path(path)
        if not file_path.exists():
            return ToolResult(
                success=False, output="", error=f"File not found: {path}"
            )

        if not file_path.is_file():
            return ToolResult(
                success=False, output="", error=f"Path is not a file: {path}"
            )

        # Read file content
        try:
            content = file_path.read_text()
        except Exception as e:
            return ToolResult(
                success=False, output="", error=f"Failed to read file: {e}"
            )

        # Perform review
        issues = []

        # Security issues
        security_issues = self._check_security(content, file_path)
        issues.extend(security_issues)

        # Code smells
        smells = self._check_code_smells(content, file_path)
        issues.extend(smells)

        # Complexity (for Python files)
        if file_path.suffix == ".py":
            complexity_issues = self._check_complexity(content, file_path)
            issues.extend(complexity_issues)

            # Documentation
            doc_issues = self._check_documentation(content, file_path)
            issues.extend(doc_issues)

        # Build report
        report = self._build_report(file_path, issues, include_metrics, content)

        return ToolResult(
            success=True,
            output=report,
            error="",
        )

    async def _review_directory(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Review all files in directory."""
        path = kwargs.get("path")
        pattern = kwargs.get("file_pattern", "*.py")
        include_metrics = kwargs.get("include_metrics", False)

        if not path:
            return ToolResult(
                success=False, output="", error="Missing required parameter: path"
            )

        dir_path = Path(path)
        if not dir_path.exists():
            return ToolResult(
                success=False, output="", error=f"Directory not found: {path}"
            )

        # Find all matching files
        files = list(dir_path.rglob(pattern))

        if not files:
            return ToolResult(
                success=True,
                output=f"No files matching pattern '{pattern}' found in {path}",
                error="",
            )

        # Review each file
        all_issues = []
        file_count = 0

        for file_path in files:
            if file_path.is_file():
                try:
                    content = file_path.read_text()
                    file_issues = []

                    # Security
                    file_issues.extend(self._check_security(content, file_path))

                    # Code smells
                    file_issues.extend(self._check_code_smells(content, file_path))

                    # Python-specific
                    if file_path.suffix == ".py":
                        file_issues.extend(self._check_complexity(content, file_path))
                        file_issues.extend(self._check_documentation(content, file_path))

                    all_issues.extend(file_issues)
                    file_count += 1

                except Exception as e:
                    logger.warning("Failed to review %s: %s", file_path, e)

        # Build summary report
        report = self._build_summary_report(
            dir_path, file_count, all_issues, include_metrics
        )

        return ToolResult(
            success=True,
            output=report,
            error="",
        )

    async def _security_scan(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Perform security-focused scan."""
        path = kwargs.get("path")
        severity = kwargs.get("severity", "low")

        if not path:
            return ToolResult(
                success=False, output="", error="Missing required parameter: path"
            )

        path_obj = Path(path)
        if not path_obj.exists():
            return ToolResult(
                success=False, output="", error=f"Path not found: {path}"
            )

        # Collect all security issues
        security_issues = []

        if path_obj.is_file():
            content = path_obj.read_text()
            security_issues = self._check_security(content, path_obj)
        else:
            # Scan directory
            for file_path in path_obj.rglob("*.py"):
                if file_path.is_file():
                    try:
                        content = file_path.read_text()
                        issues = self._check_security(content, file_path)
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
        report = self._build_security_report(path_obj, filtered_issues)

        return ToolResult(
            success=True,
            output=report,
            error="",
        )

    async def _complexity_analysis(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Analyze code complexity."""
        path = kwargs.get("path")

        if not path:
            return ToolResult(
                success=False, output="", error="Missing required parameter: path"
            )

        path_obj = Path(path)
        if not path_obj.exists():
            return ToolResult(
                success=False, output="", error=f"Path not found: {path}"
            )

        complexity_data = []

        if path_obj.is_file() and path_obj.suffix == ".py":
            content = path_obj.read_text()
            issues = self._check_complexity(content, path_obj)
            complexity_data.extend(issues)
        else:
            for file_path in path_obj.rglob("*.py"):
                if file_path.is_file():
                    try:
                        content = file_path.read_text()
                        issues = self._check_complexity(content, file_path)
                        complexity_data.extend(issues)
                    except Exception as e:
                        logger.warning("Failed to analyze %s: %s", file_path, e)

        # Build complexity report
        report = self._build_complexity_report(path_obj, complexity_data)

        return ToolResult(
            success=True,
            output=report,
            error="",
        )

    async def _best_practices(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Check coding best practices."""
        path = kwargs.get("path")

        if not path:
            return ToolResult(
                success=False, output="", error="Missing required parameter: path"
            )

        path_obj = Path(path)
        if not path_obj.exists():
            return ToolResult(
                success=False, output="", error=f"Path not found: {path}"
            )

        best_practice_issues = []

        if path_obj.is_file():
            content = path_obj.read_text()
            issues = self._check_code_smells(content, path_obj)
            best_practice_issues.extend(issues)
        else:
            for file_path in path_obj.rglob("*.py"):
                if file_path.is_file():
                    try:
                        content = file_path.read_text()
                        issues = self._check_code_smells(content, file_path)
                        best_practice_issues.extend(issues)
                    except Exception as e:
                        logger.warning("Failed to check %s: %s", file_path, e)

        # Build best practices report
        report = self._build_best_practices_report(path_obj, best_practice_issues)

        return ToolResult(
            success=True,
            output=report,
            error="",
        )

    def _check_security(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Check for security issues."""
        issues = []
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            for issue_type, pattern in self.SECURITY_PATTERNS.items():
                if re.search(pattern, line, re.IGNORECASE):
                    severity = self._get_security_severity(issue_type)
                    issues.append(
                        {
                            "type": "security",
                            "severity": severity,
                            "issue": issue_type.replace("_", " ").title(),
                            "file": str(file_path),
                            "line": line_num,
                            "code": line.strip(),
                            "recommendation": self._get_security_recommendation(
                                issue_type
                            ),
                        }
                    )

        return issues

    def _check_code_smells(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Check for code smells."""
        issues = []
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            for smell_type, pattern in self.CODE_SMELLS.items():
                if re.search(pattern, line):
                    issues.append(
                        {
                            "type": "code_smell",
                            "severity": "low",
                            "issue": smell_type.replace("_", " ").title(),
                            "file": str(file_path),
                            "line": line_num,
                            "code": line.strip()[:80],
                            "recommendation": self._get_smell_recommendation(smell_type),
                        }
                    )

        return issues

    def _check_complexity(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Check cyclomatic complexity."""
        issues = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity = self._calculate_complexity(node)

                    if complexity > self.max_complexity:
                        issues.append(
                            {
                                "type": "complexity",
                                "severity": "medium" if complexity <= 15 else "high",
                                "issue": "High Complexity",
                                "file": str(file_path),
                                "line": node.lineno,
                                "code": f"Function: {node.name}",
                                "metric": complexity,
                                "recommendation": f"Refactor to reduce complexity (current: {complexity}, max: {self.max_complexity})",
                            }
                        )

        except SyntaxError:
            logger.warning("Syntax error in %s, skipping complexity check", file_path)

        return issues

    def _check_documentation(
        self, content: str, file_path: Path
    ) -> List[Dict[str, Any]]:
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

    def _calculate_complexity(self, node: ast.AST) -> int:
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

    def _get_security_severity(self, issue_type: str) -> str:
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

    def _get_security_recommendation(self, issue_type: str) -> str:
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

    def _get_smell_recommendation(self, smell_type: str) -> str:
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
        self,
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
        self,
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

    def _build_security_report(
        self, path: Path, issues: List[Dict[str, Any]]
    ) -> str:
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

    def _build_complexity_report(
        self, path: Path, complexity_data: List[Dict[str, Any]]
    ) -> str:
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

    def _build_best_practices_report(
        self, path: Path, issues: List[Dict[str, Any]]
    ) -> str:
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
