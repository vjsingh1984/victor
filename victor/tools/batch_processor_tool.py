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
from typing import Any, Dict, List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from victor.tools.decorators import tool

logger = logging.getLogger(__name__)

# Module-level state
_max_workers: int = 4


def set_batch_processor_config(max_workers: int = 4) -> None:
    """Configure batch processor settings.

    Args:
        max_workers: Maximum parallel workers for batch operations.
    """
    global _max_workers
    _max_workers = max_workers


# Helper functions for parallel processing

async def _parallel_search(
    files: List[Path], pattern: str, use_regex: bool
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
    with ThreadPoolExecutor(max_workers=_max_workers) as executor:
        futures = {executor.submit(search_file, f): f for f in files}

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    return results


async def _parallel_replace(
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
    with ThreadPoolExecutor(max_workers=_max_workers) as executor:
        futures = {executor.submit(replace_in_file, f): f for f in files}

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    return results


async def _parallel_analyze(files: List[Path]) -> List[Dict[str, Any]]:
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
    with ThreadPoolExecutor(max_workers=_max_workers) as executor:
        futures = {executor.submit(analyze_file, f): f for f in files}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    return results


def _build_search_report(
    path: Path, pattern: str, results: List[Dict[str, Any]]
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
    path: Path, results: List[Dict[str, Any]]
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


# Tool functions

@tool
async def batch_search(
    path: str,
    pattern: str,
    file_pattern: str = "*.*",
    regex: bool = False,
    max_files: int = 1000,
) -> Dict[str, Any]:
    """
    Search for a pattern across multiple files in parallel.

    Efficiently searches through files matching a glob pattern to find
    lines containing the specified search pattern. Supports both literal
    text and regex pattern matching.

    Args:
        path: Directory path to search in.
        pattern: Search pattern to find in files.
        file_pattern: Glob pattern to match files (default: *.*).
        regex: Use regex for pattern matching (default: False).
        max_files: Maximum number of files to process (default: 1000).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - results: List of files with matches (file path, line numbers, text)
        - total_files: Number of files searched
        - total_matches: Total number of matches found
        - formatted_report: Human-readable search results
        - error: Error message if failed
    """
    if not path or not pattern:
        return {
            "success": False,
            "error": "Missing required parameters: path, pattern"
        }

    path_obj = Path(path)
    if not path_obj.exists():
        return {
            "success": False,
            "error": f"Path not found: {path}"
        }

    # Find matching files
    files = list(path_obj.rglob(file_pattern))[:max_files]

    if not files:
        return {
            "success": True,
            "results": [],
            "total_files": 0,
            "total_matches": 0,
            "message": f"No files matching pattern '{file_pattern}' found"
        }

    # Search in parallel
    results = await _parallel_search(files, pattern, regex)
    total_matches = sum(len(r["matches"]) for r in results)

    # Build report
    report = _build_search_report(path_obj, pattern, results)

    return {
        "success": True,
        "results": results,
        "total_files": len(results),
        "total_matches": total_matches,
        "formatted_report": report
    }


@tool
async def batch_replace(
    path: str,
    find: str,
    replace: str = "",
    file_pattern: str = "*.py",
    regex: bool = False,
    dry_run: bool = False,
    max_files: int = 1000,
) -> Dict[str, Any]:
    """
    Find and replace text across multiple files in parallel.

    Performs bulk find-and-replace operations across files. Supports
    dry-run mode to preview changes before applying them.

    Args:
        path: Directory path to process.
        find: Text to find in files.
        replace: Replacement text (default: empty string).
        file_pattern: Glob pattern to match files (default: *.py).
        regex: Use regex for find/replace (default: False).
        dry_run: Preview changes without modifying files (default: False).
        max_files: Maximum number of files to process (default: 1000).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - results: List of modified files with replacement counts
        - total_files: Number of files modified
        - total_replacements: Total number of replacements made
        - dry_run: Whether this was a dry run
        - formatted_report: Human-readable replacement results
        - error: Error message if failed
    """
    if not path or not find:
        return {
            "success": False,
            "error": "Missing required parameters: path, find"
        }

    path_obj = Path(path)
    if not path_obj.exists():
        return {
            "success": False,
            "error": f"Path not found: {path}"
        }

    # Find matching files
    files = list(path_obj.rglob(file_pattern))[:max_files]

    if not files:
        return {
            "success": True,
            "results": [],
            "total_files": 0,
            "total_replacements": 0,
            "message": f"No files matching pattern '{file_pattern}' found"
        }

    # Replace in parallel
    results = await _parallel_replace(files, find, replace, regex, dry_run)
    total_replacements = sum(r.get("replacements", 0) for r in results)

    # Build report
    report = _build_replace_report(path_obj, find, replace, results, dry_run)

    return {
        "success": True,
        "results": results,
        "total_files": len(results),
        "total_replacements": total_replacements,
        "dry_run": dry_run,
        "formatted_report": report
    }


@tool
async def batch_analyze(
    path: str,
    file_pattern: str = "*.py",
    max_files: int = 1000,
) -> Dict[str, Any]:
    """
    Analyze multiple files and generate statistics.

    Analyzes files to gather statistics like line counts, file sizes,
    and grouping by file type. Useful for understanding codebase structure.

    Args:
        path: Directory path to analyze.
        file_pattern: Glob pattern to match files (default: *.py).
        max_files: Maximum number of files to process (default: 1000).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - results: List of analyzed files with stats
        - total_files: Number of files analyzed
        - total_lines: Total line count across all files
        - total_size: Total size in bytes
        - by_extension: Statistics grouped by file extension
        - formatted_report: Human-readable analysis report
        - error: Error message if failed
    """
    if not path:
        return {
            "success": False,
            "error": "Missing required parameter: path"
        }

    path_obj = Path(path)
    if not path_obj.exists():
        return {
            "success": False,
            "error": f"Path not found: {path}"
        }

    # Find matching files
    files = list(path_obj.rglob(file_pattern))[:max_files]

    if not files:
        return {
            "success": True,
            "results": [],
            "total_files": 0,
            "message": f"No files matching pattern '{file_pattern}' found"
        }

    # Analyze in parallel
    results = await _parallel_analyze(files)

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

    # Build report
    report = _build_analyze_report(path_obj, results)

    return {
        "success": True,
        "results": results,
        "total_files": len(results),
        "total_lines": total_lines,
        "total_size": total_size,
        "by_extension": by_ext,
        "formatted_report": report
    }


@tool
async def batch_list_files(
    path: str,
    file_pattern: str = "*.*",
    max_files: int = 1000,
) -> Dict[str, Any]:
    """
    List files matching a pattern.

    Lists all files matching the specified glob pattern with their sizes.
    Useful for exploring directory contents.

    Args:
        path: Directory path to list files from.
        file_pattern: Glob pattern to match files (default: *.*).
        max_files: Maximum number of files to list (default: 1000).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - files: List of file paths with sizes
        - total_files: Number of files found
        - formatted_report: Human-readable file listing
        - error: Error message if failed
    """
    if not path:
        return {
            "success": False,
            "error": "Missing required parameter: path"
        }

    path_obj = Path(path)
    if not path_obj.exists():
        return {
            "success": False,
            "error": f"Path not found: {path}"
        }

    # Find matching files
    files = list(path_obj.rglob(file_pattern))[:max_files]

    # Prepare file info
    file_info = []
    for file_path in sorted(files):
        rel_path = file_path.relative_to(path_obj)
        size = file_path.stat().st_size
        file_info.append({
            "path": str(rel_path),
            "size": size
        })

    # Build report
    report = []
    report.append(f"Files matching '{file_pattern}' in {path}:")
    report.append("=" * 70)
    report.append("")
    report.append(f"Total: {len(files)} files")
    report.append("")

    for info in file_info[:100]:  # Show first 100
        report.append(f"  {info['path']} ({info['size']} bytes)")

    if len(files) > 100:
        report.append("")
        report.append(f"... and {len(files) - 100} more files")

    return {
        "success": True,
        "files": file_info,
        "total_files": len(files),
        "formatted_report": "\n".join(report)
    }


@tool
async def batch_transform(
    path: str,
    file_pattern: str = "*.py",
    max_files: int = 1000,
) -> Dict[str, Any]:
    """
    Apply transformations to multiple files.

    Placeholder for future batch transformation functionality.
    This operation is not yet implemented.

    Args:
        path: Directory path to process.
        file_pattern: Glob pattern to match files (default: *.py).
        max_files: Maximum number of files to process (default: 1000).

    Returns:
        Dictionary containing error about unimplemented operation.
    """
    return {
        "success": False,
        "error": "Transform operation not yet implemented"
    }


# Keep class for backward compatibility
class BatchProcessorTool:
    """Deprecated: Use individual batch_* functions instead."""

    def __init__(self, max_workers: int = 4):
        """Initialize - deprecated."""
        import warnings
        warnings.warn(
            "BatchProcessorTool class is deprecated. Use batch_* functions instead.",
            DeprecationWarning,
            stacklevel=2
        )
        set_batch_processor_config(max_workers)
