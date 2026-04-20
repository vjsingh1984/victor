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

"""CI/CD command optimizer with caching and batch operations.

This module provides intelligent optimization for CI/CD tool execution to address:
1. Multiple redundant API calls (e.g., repeated gh run view commands)
2. No caching of CI/CD query results
3. Spin detection triggered by deduplication blocking legitimate queries

Features:
- Result caching for CI/CD commands (5-minute TTL)
- Batch operations for viewing multiple runs/workflows
- Smart deduplication that allows different arguments
- Temporary file aggregation for log analysis
"""

import asyncio
import hashlib
import json
import logging
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class CICDCommandCache:
    """Cache for CI/CD command results with TTL."""

    def __init__(self, ttl_minutes: int = 5):
        """Initialize cache.

        Args:
            ttl_minutes: Time-to-live for cache entries in minutes
        """
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._ttl = timedelta(minutes=ttl_minutes)

    def _generate_key(self, command: str, cwd: Optional[str]) -> str:
        """Generate cache key from command and working directory.

        Args:
            command: The command string
            cwd: Current working directory

        Returns:
            Cache key hash
        """
        key_data = f"{command}:{cwd or ''}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, command: str, cwd: Optional[str] = None) -> Optional[Any]:
        """Get cached result if available and not expired.

        Args:
            command: The command string
            cwd: Current working directory

        Returns:
            Cached result or None if not found/expired
        """
        key = self._generate_key(command, cwd)
        if key in self._cache:
            result, timestamp = self._cache[key]
            if datetime.now() - timestamp < self._ttl:
                logger.debug(f"Cache HIT: {command[:50]}...")
                return result
            else:
                # Expired, remove
                del self._cache[key]
        return None

    def set(self, command: str, result: Any, cwd: Optional[str] = None) -> None:
        """Cache a command result.

        Args:
            command: The command string
            result: The result to cache
            cwd: Current working directory
        """
        key = self._generate_key(command, cwd)
        self._cache[key] = (result, datetime.now())
        logger.debug(f"Cached: {command[:50]}...")

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        logger.debug("Cache cleared")

    def cleanup_expired(self) -> int:
        """Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if now - timestamp >= self._ttl
        ]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)


# Global cache instance
_cicd_cache = CICDCommandCache(ttl_minutes=5)


def get_cicd_cache() -> CICDCommandCache:
    """Get the global CI/CD cache instance.

    Returns:
        Global cache instance
    """
    return _cicd_cache


class CICDCommandOptimizer:
    """Optimizer for CI/CD command execution.

    Provides caching, batching, and smart deduplication for CI/CD operations.
    """

    # Commands that benefit from caching
    CACHEABLE_PATTERNS = [
        "gh run view",
        "gh run list",
        "gh workflow list",
        "gh workflow view",
        "gh api /repos/*/actions/runs",
        "git log",
        "git show",
    ]

    # Commands that can be batched
    BATCHABLE_PATTERNS = [
        ("gh run view", "gh run list"),  # view multiple runs
        ("gh workflow view", "gh workflow list"),  # view multiple workflows
    ]

    @classmethod
    def is_cacheable(cls, command: str) -> bool:
        """Check if a command should be cached.

        Args:
            command: The command to check

        Returns:
            True if command is cacheable
        """
        cmd_lower = command.lower()
        return any(pattern in cmd_lower for pattern in cls.CACHEABLE_PATTERNS)

    @classmethod
    def should_batch(cls, commands: List[str]) -> bool:
        """Check if multiple commands should be batched.

        Args:
            commands: List of commands to check

        Returns:
            True if commands should be batched
        """
        if len(commands) < 2:
            return False

        # Check if commands are similar (e.g., multiple 'gh run view' with different IDs)
        first_cmd_base = commands[0].split()[0:3]  # e.g., ['gh', 'run', 'view']
        for cmd in commands[1:]:
            cmd_base = cmd.split()[0:3]
            if cmd_base != first_cmd_base:
                return False
        return True

    @classmethod
    async def execute_cached(
        cls,
        command: str,
        cwd: Optional[str] = None,
        force_refresh: bool = False
    ) -> Tuple[bool, str, str]:
        """Execute command with caching.

        Args:
            command: Command to execute
            cwd: Working directory
            force_refresh: Skip cache and force execution

        Returns:
            Tuple of (from_cache, stdout, stderr)
        """
        if not force_refresh and cls.is_cacheable(command):
            cached = get_cicd_cache().get(command, cwd)
            if cached is not None:
                return True, cached[0], cached[1]

        # Execute command
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')

            # Cache if cacheable
            if cls.is_cacheable(command) and process.returncode == 0:
                get_cicd_cache().set(command, (stdout_str, stderr_str), cwd)

            return False, stdout_str, stderr_str

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return False, "", str(e)

    @classmethod
    def batch_gh_run_view(
        cls,
        run_ids: List[str],
        repo_path: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Batch multiple 'gh run view' commands into efficient queries.

        Instead of:
            gh run view 123
            gh run view 456
            gh run view 789

        Uses:
            gh run list --limit 50  (gets all runs in one call)
            Then filters locally

        Args:
            run_ids: List of run IDs to view
            repo_path: Path to repo

        Returns:
            Dict mapping run_id -> run_info
        """
        if not run_ids:
            return {}

        logger.info(f"Batching {len(run_ids)} gh run view calls")

        # First, try to get all runs in one call
        try:
            cmd = "gh run list --limit 50 --json databaseId,status,conclusion,name,createdAt,updatedAt,event,headBranch,displayTitle"
            from_cache, stdout, stderr = asyncio.run(
                cls.execute_cached(cmd, cwd=repo_path)
            )

            if stdout:
                runs = json.loads(stdout)
                runs_by_id = {
                    str(run["databaseId"]): run
                    for run in runs
                    if str(run["databaseId"]) in run_ids
                }

                # Cache individual run views for faster subsequent access
                for run_id, run_info in runs_by_id.items():
                    cache_key = f"gh run view {run_id}"
                    get_cicd_cache().set(
                        cache_key,
                        (json.dumps(run_info, indent=2), ""),
                        repo_path
                    )

                logger.info(f"Batch fetched {len(runs_by_id)}/{len(run_ids)} runs")
                return runs_by_id

        except Exception as e:
            logger.warning(f"Batch fetch failed, falling back to individual calls: {e}")

        # Fallback: individual calls (still cached)
        results = {}
        for run_id in run_ids:
            cmd = f"gh run view {run_id} --json status,conclusion,name,createdAt,updatedAt,event,headBranch,displayTitle"
            from_cache, stdout, stderr = asyncio.run(
                cls.execute_cached(cmd, cwd=repo_path)
            )
            if stdout:
                try:
                    results[run_id] = json.loads(stdout)
                except json.JSONDecodeError:
                    results[run_id] = {"raw": stdout}

        return results

    @classmethod
    def aggregate_logs_to_file(
        cls,
        log_sources: List[str],
        output_file: Optional[Path] = None
    ) -> Path:
        """Aggregate multiple log sources into a temporary file for analysis.

        Instead of multiple grep/filter calls on log data, fetch once and analyze locally.

        Args:
            log_sources: List of log file paths or commands that produce logs
            output_file: Optional output file path

        Returns:
            Path to aggregated log file
        """
        if output_file is None:
            fd, output_file = tempfile.mkstemp(suffix="_logs.txt", prefix="cicd_")
            output_file = Path(output_file)

        logger.info(f"Aggregating {len(log_sources)} log sources to {output_file}")

        with open(output_file, 'w') as outfile:
            for i, source in enumerate(log_sources):
                outfile.write(f"\n{'='*80}\n")
                outfile.write(f"LOG SOURCE {i+1}: {source}\n")
                outfile.write(f"{'='*80}\n\n")

                # Check if source is a file path or command
                if Path(source).exists():
                    # It's a file
                    try:
                        with open(source, 'r') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"Error reading file: {e}\n")
                else:
                    # It's a command, execute it
                    try:
                        result = subprocess.run(
                            source,
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        outfile.write(result.stdout)
                        if result.stderr:
                            outfile.write(f"\nSTDERR:\n{result.stderr}\n")
                    except Exception as e:
                        outfile.write(f"Error executing command: {e}\n")

                outfile.write("\n\n")

        logger.info(f"Aggregated {len(log_sources)} log sources")
        return output_file


def optimize_cicd_query(
    command: str,
    cwd: Optional[str] = None
) -> Tuple[bool, str, str]:
    """Optimize a CI/CD query command.

    This function checks if a command can be served from cache,
    batched with other commands, or optimized in any way.

    Args:
        command: The command to optimize
        cwd: Working directory

    Returns:
        Tuple of (from_cache, stdout, stderr)
    """
    optimizer = CICDCommandOptimizer()

    # Check if it's a batchable gh run view pattern
    if "gh run view" in command and "gh run list" not in command:
        # This is a single view, check cache first
        return asyncio.run(
            optimizer.execute_cached(command, cwd=cwd)
        )

    # Check if it's a gh run list (cacheable)
    if "gh run list" in command:
        return asyncio.run(
            optimizer.execute_cached(command, cwd=cwd)
        )

    # Not optimizable, execute normally
    return asyncio.run(
        optimizer.execute_cached(command, cwd=cwd, force_refresh=False)
    )


# Export for use in other modules
__all__ = [
    "CICDCommandCache",
    "CICDCommandOptimizer",
    "get_cicd_cache",
    "optimize_cicd_query",
]
