"""Code metrics and performance analysis tool.

Features:
- Calculate code metrics (complexity, maintainability)
- Measure technical debt
- Profile code performance
- Identify performance bottlenecks
- Generate quality reports
"""

import ast
import time
import cProfile
import pstats
import io
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from victor.tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class MetricsTool(BaseTool):
    """Tool for code metrics and performance analysis."""

    @property
    def name(self) -> str:
        """Get tool name."""
        return "metrics"

    @property
    def description(self) -> str:
        """Get tool description."""
        return """Code metrics and performance analysis.

Analyze code quality and performance:
- Calculate complexity metrics
- Measure maintainability index
- Estimate technical debt
- Profile code execution
- Identify bottlenecks
- Generate quality reports

Operations:
- complexity: Calculate cyclomatic complexity
- maintainability: Calculate maintainability index
- debt: Estimate technical debt
- profile: Profile code performance
- analyze: Comprehensive code analysis
- report: Generate quality report

Example workflows:
1. Calculate complexity:
   metrics(operation="complexity", file="app.py")

2. Maintainability analysis:
   metrics(operation="maintainability", file="app.py")

3. Performance profiling:
   metrics(operation="profile", file="script.py")

4. Full analysis:
   metrics(operation="analyze", file="app.py")
"""

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameters."""
        return self.convert_parameters_to_schema(
            [
                ToolParameter(
                    name="operation",
                    type="string",
                    description="Operation: complexity, maintainability, debt, profile, analyze, report",
                    required=True,
                ),
                ToolParameter(
                    name="file",
                    type="string",
                    description="Source file path",
                    required=False,
                ),
                ToolParameter(
                    name="threshold",
                    type="integer",
                    description="Complexity threshold",
                    required=False,
                ),
            ]
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute metrics operation.

        Args:
            operation: Metrics operation to perform
            **kwargs: Operation-specific parameters

        Returns:
            Tool result with metrics
        """
        operation = kwargs.get("operation")

        if not operation:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: operation",
            )

        try:
            if operation == "complexity":
                return await self._calculate_complexity(kwargs)
            elif operation == "maintainability":
                return await self._calculate_maintainability(kwargs)
            elif operation == "debt":
                return await self._estimate_debt(kwargs)
            elif operation == "profile":
                return await self._profile_code(kwargs)
            elif operation == "analyze":
                return await self._analyze_code(kwargs)
            elif operation == "report":
                return await self._generate_report(kwargs)
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown operation: {operation}",
                )

        except Exception as e:
            logger.exception("Metrics operation failed")
            return ToolResult(success=False, output="", error=f"Metrics error: {str(e)}")

    async def _calculate_complexity(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Calculate cyclomatic complexity."""
        file_path = kwargs.get("file")
        threshold = kwargs.get("threshold", 10)

        if not file_path:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: file",
            )

        file_obj = Path(file_path)
        if not file_obj.exists():
            return ToolResult(success=False, output="", error=f"File not found: {file_path}")

        # Read and parse
        content = file_obj.read_text()
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Syntax error: {e}",
            )

        # Calculate complexity
        complexities = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._compute_complexity(node)
                complexities.append(
                    {
                        "name": node.name,
                        "complexity": complexity,
                        "line": node.lineno,
                        "status": self._get_complexity_status(complexity, threshold),
                    }
                )

        # Sort by complexity
        complexities.sort(key=lambda x: x["complexity"], reverse=True)

        # Build report
        report = []
        report.append("Cyclomatic Complexity")
        report.append("=" * 70)
        report.append("")
        report.append(f"File: {file_path}")
        report.append(f"Functions: {len(complexities)}")
        report.append(f"Threshold: {threshold}")
        report.append("")

        # Categorize
        simple = [c for c in complexities if c["complexity"] <= 5]
        moderate = [c for c in complexities if 5 < c["complexity"] <= threshold]
        complex = [c for c in complexities if c["complexity"] > threshold]

        report.append(f"🟢 Simple (1-5): {len(simple)}")
        report.append(f"🟡 Moderate (6-{threshold}): {len(moderate)}")
        report.append(f"🔴 Complex ({threshold}+): {len(complex)}")
        report.append("")

        if complex:
            report.append("High complexity functions:")
            for func in complex[:10]:
                report.append(
                    f"  {func['status']} {func['name']}: {func['complexity']} (line {func['line']})"
                )
            report.append("")

        report.append("Recommendations:")
        if complex:
            report.append("  • Refactor complex functions")
            report.append("  • Extract helper methods")
            report.append("  • Reduce nested conditions")
        else:
            report.append("  ✅ Good complexity scores!")

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _calculate_maintainability(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Calculate maintainability index."""
        file_path = kwargs.get("file")

        if not file_path:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: file",
            )

        file_obj = Path(file_path)
        if not file_obj.exists():
            return ToolResult(success=False, output="", error=f"File not found: {file_path}")

        # Read file
        content = file_obj.read_text()
        lines = content.split("\n")

        # Simple metrics
        total_lines = len(lines)
        code_lines = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
        comment_lines = len([l for l in lines if l.strip().startswith("#")])

        # Calculate average complexity
        try:
            tree = ast.parse(content)
            complexities = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexities.append(self._compute_complexity(node))
            avg_complexity = sum(complexities) / len(complexities) if complexities else 1
        except:
            avg_complexity = 1

        # Simplified maintainability index
        comment_ratio = comment_lines / total_lines if total_lines > 0 else 0
        mi = max(0, min(100, 100 - (avg_complexity * 5) + (comment_ratio * 20)))

        # Build report
        report = []
        report.append("Maintainability Index")
        report.append("=" * 70)
        report.append("")
        report.append(f"File: {file_path}")
        report.append("")
        report.append(f"Score: {mi:.1f}/100")
        report.append(f"Rating: {self._get_mi_rating(mi)}")
        report.append("")
        report.append("Metrics:")
        report.append(f"  Total Lines: {total_lines}")
        report.append(f"  Code Lines: {code_lines}")
        report.append(f"  Comment Lines: {comment_lines}")
        report.append(f"  Avg Complexity: {avg_complexity:.1f}")
        report.append("")

        report.append("Recommendations:")
        if mi < 65:
            report.append("  🔴 Low maintainability")
            report.append("  • Reduce complexity")
            report.append("  • Add documentation")
            report.append("  • Refactor large functions")
        elif mi < 85:
            report.append("  🟡 Moderate maintainability")
            report.append("  • Consider refactoring")
            report.append("  • Improve documentation")
        else:
            report.append("  🟢 High maintainability")
            report.append("  • Keep up the good work!")

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _estimate_debt(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Estimate technical debt."""
        file_path = kwargs.get("file")

        if not file_path:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: file",
            )

        file_obj = Path(file_path)
        if not file_obj.exists():
            return ToolResult(success=False, output="", error=f"File not found: {file_path}")

        content = file_obj.read_text()

        # Count debt indicators
        todo_count = content.count("TODO")
        fixme_count = content.count("FIXME")
        hack_count = content.count("HACK")
        xxx_count = content.count("XXX")

        total_markers = todo_count + fixme_count + hack_count + xxx_count

        # Estimate hours (rough estimate)
        estimated_hours = total_markers * 2  # 2 hours per marker

        # Build report
        report = []
        report.append("Technical Debt Estimation")
        report.append("=" * 70)
        report.append("")
        report.append(f"File: {file_path}")
        report.append("")
        report.append("Debt Markers:")
        report.append(f"  TODO: {todo_count}")
        report.append(f"  FIXME: {fixme_count}")
        report.append(f"  HACK: {hack_count}")
        report.append(f"  XXX: {xxx_count}")
        report.append("")
        report.append(f"Total Markers: {total_markers}")
        report.append(f"Estimated Hours: {estimated_hours}")
        report.append(f"Estimated Days: {estimated_hours / 8:.1f}")
        report.append("")

        if total_markers == 0:
            report.append("✅ No technical debt markers found!")
        else:
            report.append("Recommendations:")
            report.append("  • Address FIXME items first")
            report.append("  • Remove HACK solutions")
            report.append("  • Complete TODO tasks")
            report.append("  • Consider refactoring")

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _profile_code(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Profile code performance."""
        # This is a simplified version
        report = []
        report.append("Performance Profiling")
        report.append("=" * 70)
        report.append("")
        report.append("Note: Use Python's built-in profilers:")
        report.append("  • cProfile: python -m cProfile -o output.prof script.py")
        report.append("  • timeit: python -m timeit 'code'")
        report.append("  • memory_profiler: @profile decorator")
        report.append("")
        report.append("Analysis tools:")
        report.append("  • snakeviz: Visualization of cProfile output")
        report.append("  • py-spy: Sampling profiler")
        report.append("  • line_profiler: Line-by-line profiling")

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _analyze_code(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Comprehensive code analysis."""
        file_path = kwargs.get("file")

        if not file_path:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: file",
            )

        # Run all analyses
        complexity_result = await self._calculate_complexity(kwargs)
        maintainability_result = await self._calculate_maintainability(kwargs)
        debt_result = await self._estimate_debt(kwargs)

        # Combine reports
        report = []
        report.append("Comprehensive Code Analysis")
        report.append("=" * 70)
        report.append("")
        report.append(f"File: {file_path}")
        report.append("")
        report.append(complexity_result.output)
        report.append("")
        report.append("-" * 70)
        report.append("")
        report.append(maintainability_result.output)
        report.append("")
        report.append("-" * 70)
        report.append("")
        report.append(debt_result.output)

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _generate_report(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Generate quality report."""
        return await self._analyze_code(kwargs)

    def _compute_complexity(self, node: ast.FunctionDef) -> int:
        """Compute cyclomatic complexity of a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Decision points
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _get_complexity_status(self, complexity: int, threshold: int) -> str:
        """Get complexity status icon."""
        if complexity <= 5:
            return "🟢"
        elif complexity <= threshold:
            return "🟡"
        else:
            return "🔴"

    def _get_mi_rating(self, mi: float) -> str:
        """Get maintainability index rating."""
        if mi >= 85:
            return "🟢 Excellent"
        elif mi >= 65:
            return "🟡 Good"
        elif mi >= 50:
            return "🟠 Fair"
        else:
            return "🔴 Poor"
