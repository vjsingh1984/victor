"""Batch processing tool for multi-file operations.

Features:
- Process multiple files in parallel
- Pattern-based file selection
- Bulk operations (search, replace, analyze)
- Progress tracking
- Error handling and reporting
- Dry-run mode
"""

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from victor.tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class BatchProcessorTool(BaseTool):
    """Tool for batch processing multiple files."""

    def __init__(self, max_workers: int = 4):
        """Initialize batch processor.

        Args:
            max_workers: Maximum parallel workers
        """
        super().__init__()
        self.max_workers = max_workers

    @property
    def name(self) -> str:
        """Get tool name."""
        return "batch_process"

    @property
    def description(self) -> str:
        """Get tool description."""
        return """Batch processing for multi-file operations.

Process multiple files efficiently with parallel execution:
- Search across multiple files
- Bulk find and replace
- Batch analysis and transformations
- Progress tracking and reporting

Operations:
- search: Search pattern across files
- replace: Find and replace in multiple files
- analyze: Analyze multiple files
- transform: Apply transformations to files
- list: List files matching pattern

Example workflows:
1. Search across files:
   batch_process(operation="search", pattern="TODO", path="src/", file_pattern="*.py")

2. Bulk replace:
   batch_process(operation="replace", find="old_api", replace="new_api", path="src/")

3. Analyze files:
   batch_process(operation="analyze", path="src/", file_pattern="*.py")

4. Dry run (preview changes):
   batch_process(operation="replace", find="foo", replace="bar", dry_run=True)
"""

    @property
    def parameters(self) -> List[ToolParameter]:
        """Get tool parameters."""
        return [
            ToolParameter(
                name="operation",
                type="string",
                description="Operation: search, replace, analyze, transform, list",
                required=True,
            ),
            ToolParameter(
                name="path",
                type="string",
                description="Directory path to process",
                required=True,
            ),
            ToolParameter(
                name="file_pattern",
                type="string",
                description="File pattern (glob) to match (default: *.*)",
                required=False,
            ),
            ToolParameter(
                name="pattern",
                type="string",
                description="Search pattern (for search operation)",
                required=False,
            ),
            ToolParameter(
                name="find",
                type="string",
                description="Text to find (for replace operation)",
                required=False,
            ),
            ToolParameter(
                name="replace",
                type="string",
                description="Replacement text (for replace operation)",
                required=False,
            ),
            ToolParameter(
                name="regex",
                type="boolean",
                description="Use regex for pattern matching (default: false)",
                required=False,
            ),
            ToolParameter(
                name="dry_run",
                type="boolean",
                description="Preview changes without applying (default: false)",
                required=False,
            ),
            ToolParameter(
                name="max_files",
                type="integer",
                description="Maximum files to process (default: 1000)",
                required=False,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute batch operation.

        Args:
            operation: Operation to perform
            **kwargs: Operation-specific parameters

        Returns:
            Tool result with batch processing results
        """
        operation = kwargs.get("operation")

        if not operation:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: operation",
            )

        try:
            if operation == "search":
                return await self._batch_search(kwargs)
            elif operation == "replace":
                return await self._batch_replace(kwargs)
            elif operation == "analyze":
                return await self._batch_analyze(kwargs)
            elif operation == "transform":
                return await self._batch_transform(kwargs)
            elif operation == "list":
                return await self._list_files(kwargs)
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown operation: {operation}",
                )

        except Exception as e:
            logger.exception("Batch processing failed")
            return ToolResult(
                success=False, output="", error=f"Batch processing error: {str(e)}"
            )

    async def _batch_search(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Search pattern across multiple files."""
        path = kwargs.get("path")
        pattern = kwargs.get("pattern")
        file_pattern = kwargs.get("file_pattern", "*.*")
        use_regex = kwargs.get("regex", False)
        max_files = kwargs.get("max_files", 1000)

        if not path or not pattern:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameters: path, pattern",
            )

        path_obj = Path(path)
        if not path_obj.exists():
            return ToolResult(
                success=False, output="", error=f"Path not found: {path}"
            )

        # Find matching files
        files = list(path_obj.rglob(file_pattern))[:max_files]

        if not files:
            return ToolResult(
                success=True,
                output=f"No files matching pattern '{file_pattern}' found",
                error="",
            )

        # Search in parallel
        results = await self._parallel_search(files, pattern, use_regex)

        # Build report
        report = self._build_search_report(path_obj, pattern, results)

        return ToolResult(
            success=True,
            output=report,
            error="",
        )

    async def _batch_replace(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Find and replace across multiple files."""
        path = kwargs.get("path")
        find_text = kwargs.get("find")
        replace_text = kwargs.get("replace")
        file_pattern = kwargs.get("file_pattern", "*.py")
        use_regex = kwargs.get("regex", False)
        dry_run = kwargs.get("dry_run", False)
        max_files = kwargs.get("max_files", 1000)

        if not path or not find_text:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameters: path, find",
            )

        path_obj = Path(path)
        if not path_obj.exists():
            return ToolResult(
                success=False, output="", error=f"Path not found: {path}"
            )

        # Find matching files
        files = list(path_obj.rglob(file_pattern))[:max_files]

        if not files:
            return ToolResult(
                success=True,
                output=f"No files matching pattern '{file_pattern}' found",
                error="",
            )

        # Replace in parallel
        results = await self._parallel_replace(
            files, find_text, replace_text or "", use_regex, dry_run
        )

        # Build report
        report = self._build_replace_report(
            path_obj, find_text, replace_text, results, dry_run
        )

        return ToolResult(
            success=True,
            output=report,
            error="",
        )

    async def _batch_analyze(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Analyze multiple files."""
        path = kwargs.get("path")
        file_pattern = kwargs.get("file_pattern", "*.py")
        max_files = kwargs.get("max_files", 1000)

        if not path:
            return ToolResult(
                success=False, output="", error="Missing required parameter: path"
            )

        path_obj = Path(path)
        if not path_obj.exists():
            return ToolResult(
                success=False, output="", error=f"Path not found: {path}"
            )

        # Find matching files
        files = list(path_obj.rglob(file_pattern))[:max_files]

        if not files:
            return ToolResult(
                success=True,
                output=f"No files matching pattern '{file_pattern}' found",
                error="",
            )

        # Analyze in parallel
        results = await self._parallel_analyze(files)

        # Build report
        report = self._build_analyze_report(path_obj, results)

        return ToolResult(
            success=True,
            output=report,
            error="",
        )

    async def _batch_transform(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Apply transformations to multiple files."""
        # Placeholder for future implementation
        return ToolResult(
            success=False,
            output="",
            error="Transform operation not yet implemented",
        )

    async def _list_files(self, kwargs: Dict[str, Any]) -> ToolResult:
        """List files matching pattern."""
        path = kwargs.get("path")
        file_pattern = kwargs.get("file_pattern", "*.*")
        max_files = kwargs.get("max_files", 1000)

        if not path:
            return ToolResult(
                success=False, output="", error="Missing required parameter: path"
            )

        path_obj = Path(path)
        if not path_obj.exists():
            return ToolResult(
                success=False, output="", error=f"Path not found: {path}"
            )

        # Find matching files
        files = list(path_obj.rglob(file_pattern))[:max_files]

        report = []
        report.append(f"Files matching '{file_pattern}' in {path}:")
        report.append("=" * 70)
        report.append("")
        report.append(f"Total: {len(files)} files")
        report.append("")

        for file_path in sorted(files)[:100]:  # Show first 100
            rel_path = file_path.relative_to(path_obj)
            size = file_path.stat().st_size
            report.append(f"  {rel_path} ({size} bytes)")

        if len(files) > 100:
            report.append("")
            report.append(f"... and {len(files) - 100} more files")

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _parallel_search(
        self, files: List[Path], pattern: str, use_regex: bool
    ) -> List[Dict[str, Any]]:
        """Search files in parallel."""
        results = []

        def search_file(file_path: Path) -> Optional[Dict[str, Any]]:
            try:
                content = file_path.read_text()
                matches = []

                if use_regex:
                    for line_num, line in enumerate(content.split("\n"), 1):
                        if re.search(pattern, line):
                            matches.append(
                                {"line": line_num, "text": line.strip()[:100]}
                            )
                else:
                    for line_num, line in enumerate(content.split("\n"), 1):
                        if pattern in line:
                            matches.append(
                                {"line": line_num, "text": line.strip()[:100]}
                            )

                if matches:
                    return {"file": str(file_path), "matches": matches}

            except Exception as e:
                logger.warning("Failed to search %s: %s", file_path, e)

            return None

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(search_file, f): f for f in files}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        return results

    async def _parallel_replace(
        self,
        files: List[Path],
        find_text: str,
        replace_text: str,
        use_regex: bool,
        dry_run: bool,
    ) -> List[Dict[str, Any]]:
        """Replace in files in parallel."""
        results = []

        def replace_in_file(file_path: Path) -> Optional[Dict[str, Any]]:
            try:
                content = file_path.read_text()
                original = content

                if use_regex:
                    new_content = re.sub(find_text, replace_text, content)
                else:
                    new_content = content.replace(find_text, replace_text)

                if new_content != original:
                    count = content.count(find_text) if not use_regex else len(
                        re.findall(find_text, content)
                    )

                    if not dry_run:
                        file_path.write_text(new_content)

                    return {
                        "file": str(file_path),
                        "replacements": count,
                        "modified": not dry_run,
                    }

            except Exception as e:
                logger.warning("Failed to process %s: %s", file_path, e)
                return {"file": str(file_path), "error": str(e)}

            return None

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(replace_in_file, f): f for f in files}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        return results

    async def _parallel_analyze(self, files: List[Path]) -> List[Dict[str, Any]]:
        """Analyze files in parallel."""
        results = []

        def analyze_file(file_path: Path) -> Dict[str, Any]:
            try:
                content = file_path.read_text()
                lines = content.split("\n")

                return {
                    "file": str(file_path),
                    "lines": len(lines),
                    "size": file_path.stat().st_size,
                    "extension": file_path.suffix,
                }

            except Exception as e:
                logger.warning("Failed to analyze %s: %s", file_path, e)
                return {
                    "file": str(file_path),
                    "error": str(e),
                }

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(analyze_file, f): f for f in files}

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        return results

    def _build_search_report(
        self, path: Path, pattern: str, results: List[Dict[str, Any]]
    ) -> str:
        """Build search results report."""
        report = []
        report.append(f"Batch Search Results: '{pattern}' in {path}")
        report.append("=" * 70)
        report.append("")

        total_matches = sum(len(r["matches"]) for r in results)
        report.append(f"Found in {len(results)} files ({total_matches} matches)")
        report.append("")

        for result in results[:20]:  # Show first 20 files
            rel_path = Path(result["file"]).relative_to(path) if path.is_dir() else Path(result["file"]).name
            report.append(f"📄 {rel_path}")
            for match in result["matches"][:5]:  # Show first 5 matches per file
                report.append(f"  Line {match['line']}: {match['text']}")
            if len(result["matches"]) > 5:
                report.append(f"  ... and {len(result['matches']) - 5} more matches")
            report.append("")

        if len(results) > 20:
            report.append(f"... and {len(results) - 20} more files")

        return "\n".join(report)

    def _build_replace_report(
        self,
        path: Path,
        find_text: str,
        replace_text: Optional[str],
        results: List[Dict[str, Any]],
        dry_run: bool,
    ) -> str:
        """Build replace results report."""
        report = []
        mode = "DRY RUN" if dry_run else "EXECUTED"
        report.append(f"Batch Replace Results ({mode}): '{find_text}' → '{replace_text}' in {path}")
        report.append("=" * 70)
        report.append("")

        total_replacements = sum(r.get("replacements", 0) for r in results)
        report.append(f"Modified {len(results)} files ({total_replacements} replacements)")
        report.append("")

        for result in results[:30]:  # Show first 30
            rel_path = Path(result["file"]).relative_to(path) if path.is_dir() else Path(result["file"]).name
            if "error" in result:
                report.append(f"❌ {rel_path}: {result['error']}")
            else:
                status = "Preview" if dry_run else "Modified"
                report.append(
                    f"✓ {rel_path}: {result.get('replacements', 0)} replacements ({status})"
                )

        if dry_run:
            report.append("")
            report.append("⚠️  This was a DRY RUN - no files were modified")
            report.append("   Run without dry_run=True to apply changes")

        return "\n".join(report)

    def _build_analyze_report(
        self, path: Path, results: List[Dict[str, Any]]
    ) -> str:
        """Build analysis report."""
        report = []
        report.append(f"Batch Analysis Report: {path}")
        report.append("=" * 70)
        report.append("")

        # Calculate stats
        total_lines = sum(r.get("lines", 0) for r in results)
        total_size = sum(r.get("size", 0) for r in results)

        # Group by extension
        by_ext: Dict[str, Dict[str, int]] = {}
        for result in results:
            ext = result.get("extension", "no extension")
            if ext not in by_ext:
                by_ext[ext] = {"count": 0, "lines": 0, "size": 0}

            by_ext[ext]["count"] += 1
            by_ext[ext]["lines"] += result.get("lines", 0)
            by_ext[ext]["size"] += result.get("size", 0)

        report.append(f"Files analyzed: {len(results)}")
        report.append(f"Total lines: {total_lines:,}")
        report.append(f"Total size: {total_size / 1024:.2f} KB")
        report.append("")

        report.append("By file type:")
        for ext, stats in sorted(by_ext.items(), key=lambda x: x[1]["count"], reverse=True):
            report.append(
                f"  {ext}: {stats['count']} files, {stats['lines']:,} lines, {stats['size'] / 1024:.2f} KB"
            )

        return "\n".join(report)
