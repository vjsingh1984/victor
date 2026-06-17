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
# See the the License for the specific language governing permissions and
# limitations under the License.

"""Smart CI/CD tool with caching and batch operations.

This tool provides intelligent CI/CD operations that avoid redundant API calls
and prevent spin detection from deduplication blocking.

Key features:
- Batch view multiple CI runs in one API call
- Cached CI/CD queries (5-minute TTL)
- Smart log aggregation to temporary files
- Optimized GitHub CLI interactions
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool
from victor.tools.cicd_optimizer import (
    CICDCommandOptimizer,
    get_cicd_cache,
    optimize_cicd_query,
)

logger = logging.getLogger(__name__)


@tool(
    name="cicd_batch_view_runs",
    category="cicd",
    danger_level=DangerLevel.SAFE,
    access_mode=AccessMode.READONLY,
    priority=Priority.HIGH,
)
def cicd_batch_view_runs(
    run_ids: List[str],
    repo_path: Optional[str] = None
) -> Dict[str, Any]:
    """View multiple CI/CD runs efficiently.

    Instead of calling 'gh run view' multiple times (which triggers deduplication),
    this tool batches the requests into a single 'gh run list' call and filters locally.

    Args:
        run_ids: List of run IDs to view (e.g., ["24652792", "24652793"])
        repo_path: Path to repository (optional, uses current dir if not specified)

    Returns:
        Dict mapping run_id -> run_info with status, conclusion, name, etc.

    Example:
        >>> cicd_batch_view_runs(["24652792", "24652793"], "/path/to/repo")
        {
            "24652792": {"status": "completed", "conclusion": "success", ...},
            "24652793": {"status": "completed", "conclusion": "failure", ...}
        }
    """
    if not run_ids:
        return {"error": "No run IDs provided"}

    logger.info(f"Batch viewing {len(run_ids)} CI runs")

    results = CICDCommandOptimizer.batch_gh_run_view(run_ids, repo_path)

    if not results:
        return {
            "error": "Failed to fetch run information",
            "run_ids_requested": run_ids
        }

    # Format results for LLM consumption
    summary = {
        "total_runs": len(results),
        "runs": {}
    }

    for run_id, run_info in results.items():
        summary["runs"][run_id] = {
            "id": run_id,
            "status": run_info.get("status", "unknown"),
            "conclusion": run_info.get("conclusion", "unknown"),
            "name": run_info.get("name", ""),
            "display_title": run_info.get("displayTitle", ""),
            "event": run_info.get("event", ""),
            "branch": run_info.get("headBranch", ""),
            "created_at": run_info.get("createdAt", ""),
            "updated_at": run_info.get("updatedAt", ""),
        }

    return summary


@tool(
    name="cicd_list_runs",
    category="cicd",
    danger_level=DangerLevel.SAFE,
    access_mode=AccessMode.READONLY,
    priority=Priority.HIGH,
)
def cicd_list_runs(
    limit: int = 20,
    repo_path: Optional[str] = None,
    status_filter: Optional[str] = None
) -> Dict[str, Any]:
    """List recent CI/CD runs with caching.

    Args:
        limit: Maximum number of runs to return (default: 20)
        repo_path: Path to repository (optional)
        status_filter: Optional filter by status (e.g., "failure", "success")

    Returns:
        List of recent runs with metadata

    Example:
        >>> cicd_list_runs(limit=10, status_filter="failure")
        {
            "total": 10,
            "runs": [
                {"id": "24652792", "status": "completed", "conclusion": "failure", ...},
                ...
            ]
        }
    """
    cmd = f"gh run list --limit {limit} --json databaseId,status,conclusion,name,createdAt,updatedAt,event,headBranch,displayTitle"

    if status_filter:
        # gh CLI doesn't support filtering in list command, filter locally
        pass

    from_cache, stdout, stderr = optimize_cicd_query(cmd, cwd=repo_path)

    if not stdout:
        return {
            "error": "Failed to list runs",
            "stderr": stderr
        }

    try:
        runs = json.loads(stdout)

        # Apply status filter if specified
        if status_filter:
            runs = [r for r in runs if r.get("conclusion") == status_filter]

        return {
            "total": len(runs),
            "from_cache": from_cache,
            "runs": runs
        }
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse GitHub response: {e}",
            "raw_output": stdout[:500]
        }


@tool(
    name="cicd_analyze_logs",
    "Prevents multiple API calls for the same log data.",
    category="cicd",
    danger_level=DangerLevel.SAFE,
    access_mode=AccessMode.READONLY,
    priority=Priority.MEDIUM,
)
def cicd_analyze_logs(
    run_ids: List[str],
    search_pattern: Optional[str] = None,
    repo_path: Optional[str] = None
) -> Dict[str, Any]:
    """Aggregate and analyze logs from multiple CI/CD runs.

    Instead of multiple 'gh run view' calls with log fetching, this tool
    fetches all logs once and performs analysis locally.

    Args:
        run_ids: List of run IDs to analyze logs from
        search_pattern: Optional pattern to search for in logs (grep-style)
        repo_path: Path to repository (optional)

    Returns:
        Aggregated log analysis results

    Example:
        >>> cicd_analyze_logs(["24652792", "24652793"], search_pattern="error")
        {
            "total_runs": 2,
            "pattern_matches": 15,
            "log_file": "/tmp/cicd_logs_abc123.txt",
            "summary": "..."
        }
    """
    if not run_ids:
        return {"error": "No run IDs provided"}

    logger.info(f"Analyzing logs for {len(run_ids)} runs")

    # Fetch logs for each run
    log_sources = []
    for run_id in run_ids:
        # Check cache first
        cmd = f"gh run view {run_id} --log"
        from_cache, stdout, stderr = optimize_cicd_query(cmd, cwd=repo_path)

        if stdout:
            # Write to temp file
            fd, temp_path = tempfile.mkstemp(suffix=f"_run_{run_id}.log")
            with open(temp_path, 'w') as f:
                f.write(f"=== Run {run_id} ===\n")
                f.write(stdout)
            log_sources.append(temp_path)

    if not log_sources:
        return {
            "error": "No logs found for specified runs",
            "run_ids": run_ids
        }

    # Aggregate logs
    aggregated_file = CICDCommandOptimizer.aggregate_logs_to_file(log_sources)

    # Search if pattern provided
    search_results = []
    if search_pattern:
        try:
            result = subprocess.run(
                ["grep", "-i", "-c", search_pattern, str(aggregated_file)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                # Get matching lines
                result2 = subprocess.run(
                    ["grep", "-i", search_pattern, str(aggregated_file)],
                    capture_output=True,
                    text=True
                )
                search_results = result2.stdout.split('\n')[:50]  # Limit to 50 matches
        except Exception as e:
            logger.warning(f"Search failed: {e}")

    return {
        "total_runs": len(run_ids),
        "logs_analyzed": len(log_sources),
        "aggregated_log_file": str(aggregated_file),
        "search_pattern": search_pattern,
        "matches_found": len(search_results),
        "sample_matches": search_results[:10],  # Return first 10 matches
        "summary": f"Analyzed {len(log_sources)} run logs. "
                   f"Search pattern '{search_pattern}' found {len(search_results)} matches."
    }


@tool(
    name="cicd_cache_stats",
    "Shows cache hit rate and number of cached entries.",
    category="cicd",
    danger_level=DangerLevel.SAFE,
    access_mode=AccessMode.READONLY,
    priority=Priority.LOW,
)
def cicd_cache_stats() -> Dict[str, Any]:
    """Get CI/CD query cache statistics.

    Returns:
        Cache statistics including hit rate and entry count

    Example:
        >>> cicd_cache_stats()
        {
            "cache_entries": 15,
            "ttl_minutes": 5,
            "description": "CI/CD query cache for reducing redundant API calls"
        }
    """
    cache = get_cicd_cache()

    return {
        "cache_entries": len(cache._cache),
        "ttl_minutes": int(cache._ttl.total_seconds() / 60),
        "description": "CI/CD query cache for reducing redundant API calls",
        "benefits": [
            "Reduces GitHub API calls",
            "Prevents spin detection from deduplication",
            "Faster response times for cached queries",
            "Lower rate limiting risk"
        ]
    }


@tool(
    name="cicd_clear_cache",
    "Use this to force refresh of cached CI/CD data.",
    category="cicd",
    danger_level=DangerLevel.SAFE,
    access_mode=AccessMode.READONLY,
    priority=Priority.LOW,
)
def cicd_clear_cache() -> Dict[str, Any]:
    """Clear the CI/CD query cache.

    Returns:
        Confirmation message

    Example:
        >>> cicd_clear_cache()
        {"message": "Cache cleared", "entries_removed": 15}
    """
    cache = get_cicd_cache()
    count = len(cache._cache)
    cache.clear()

    return {
        "message": "CI/CD cache cleared",
        "entries_removed": count
    }


@tool(
    name="cicd_run_diagnosis",
    "Analyzes failed runs, checks logs, and identifies issues efficiently.",
    category="cicd",
    danger_level=DangerLevel.SAFE,
    access_mode=AccessMode.READONLY,
    priority=Priority.HIGH,
)
def cicd_run_diagnosis(
    run_id: str,
    repo_path: Optional[str] = None
) -> Dict[str, Any]:
    """Perform comprehensive diagnosis of a CI/CD run.

    Args:
        run_id: Run ID to diagnose
        repo_path: Path to repository (optional)

    Returns:
        Comprehensive diagnosis information

    Example:
        >>> cicd_run_diagnosis("24652792")
        {
            "run_id": "24652792",
            "status": "failure",
            "conclusion": "failure",
            "jobs": [...],
            "logs_summary": "...",
            "common_errors": ["error1", "error2"],
            "recommendations": ["fix1", "fix2"]
        }
    """
    logger.info(f"Diagnosing CI/CD run {run_id}")

    # Get run info (cached)
    cmd = f"gh run view {run_id} --json status,conclusion,name,jobs,databaseId"
    from_cache, stdout, stderr = optimize_cicd_query(cmd, cwd=repo_path)

    if not stdout:
        return {
            "error": f"Failed to get run info for {run_id}",
            "stderr": stderr
        }

    try:
        run_info = json.loads(stdout)

        # Get logs (cached)
        log_cmd = f"gh run view {run_id} --log-failed"
        from_cache_logs, log_stdout, log_stderr = optimize_cicd_query(log_cmd, cwd=repo_path)

        # Analyze logs for common errors
        common_errors = []
        if log_stdout:
            # Look for common error patterns
            error_patterns = [
                "Error:",
                "ERROR:",
                "failed",
                "Failure:",
                "Exception",
                "Traceback",
                "Caused by:",
            ]

            for pattern in error_patterns:
                if pattern in log_stdout:
                    # Count occurrences
                    count = log_stdout.count(pattern)
                    common_errors.append(f"{pattern}: {count} occurrences")

        return {
            "run_id": run_id,
            "database_id": run_info.get("databaseId", ""),
            "status": run_info.get("status", ""),
            "conclusion": run_info.get("conclusion", ""),
            "name": run_info.get("name", ""),
            "jobs": run_info.get("jobs", [])[:10],  # Limit to 10 jobs
            "from_cache": from_cache,
            "logs_from_cache": from_cache_logs,
            "common_errors": common_errors[:5],  # Top 5 errors
            "recommendations": _generate_recommendations(run_info, common_errors)
        }
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse run info: {e}",
            "raw_output": stdout[:500]
        }


def _generate_recommendations(
    run_info: Dict[str, Any],
    common_errors: List[str]
) -> List[str]:
    """Generate recommendations based on run analysis.

    Args:
        run_info: Run information
        common_errors: List of common errors found

    Returns:
        List of recommendations
    """
    recommendations = []

    conclusion = run_info.get("conclusion", "")
    status = run_info.get("status", "")

    if conclusion == "failure":
        recommendations.append("Review the failed jobs and logs above")
        recommendations.append("Check for recent changes that might have broken the build")

    if status == "queued":
        recommendations.append("Run may be waiting for a runner - check GitHub Actions status")

    if "test" in str(run_info.get("name", "")).lower():
        recommendations.append("Check test failures locally before pushing")

    if any("Error:" in e or "Exception" in e for e in common_errors):
        recommendations.append("Look for exceptions in the logs and fix the root cause")

    return recommendations
