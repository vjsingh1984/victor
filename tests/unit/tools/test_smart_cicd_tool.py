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

"""Tests for smart CI/CD tool with caching and batch operations."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from victor.tools.smart_cicd_tool import (
    cicd_batch_view_runs,
    cicd_list_runs,
    cicd_analyze_logs,
    cicd_cache_stats,
    cicd_clear_cache,
    cicd_run_diagnosis,
)
from victor.tools.cicd_optimizer import (
    CICDCommandCache,
    CICDCommandOptimizer,
    get_cicd_cache,
)


class TestCICDCommandCache:
    """Test CI/CD command cache."""

    def test_cache_initialization(self):
        """Test cache initializes correctly."""
        cache = CICDCommandCache(ttl_minutes=5)
        assert cache._ttl.total_seconds() == 300
        assert len(cache._cache) == 0

    def test_cache_set_and_get(self):
        """Test cache set and get operations."""
        cache = CICDCommandCache(ttl_minutes=5)

        # Set value
        cache.set("test command", ("stdout", "stderr"))

        # Get value
        result = cache.get("test command")
        assert result == ("stdout", "stderr")

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = CICDCommandCache(ttl_minutes=5)
        result = cache.get("nonexistent command")
        assert result is None

    def test_cache_key_generation(self):
        """Test cache keys are generated correctly."""
        cache = CICDCommandCache(ttl_minutes=5)

        key1 = cache._generate_key("test command", "/path/to/repo")
        key2 = cache._generate_key("test command", "/path/to/repo")
        key3 = cache._generate_key("test command", "/different/path")

        # Same command and path should generate same key
        assert key1 == key2

        # Different path should generate different key
        assert key1 != key3

    def test_cache_clear(self):
        """Test cache clear operation."""
        cache = CICDCommandCache(ttl_minutes=5)

        cache.set("cmd1", ("out1", "err1"))
        cache.set("cmd2", ("out2", "err2"))
        assert len(cache._cache) == 2

        cache.clear()
        assert len(cache._cache) == 0

    def test_global_cache_instance(self):
        """Test global cache instance is accessible."""
        cache = get_cicd_cache()
        assert isinstance(cache, CICDCommandCache)


class TestCICDCommandOptimizer:
    """Test CI/CD command optimizer."""

    def test_is_cacheable(self):
        """Test cacheable command detection."""
        assert CICDCommandOptimizer.is_cacheable("gh run view 123")
        assert CICDCommandOptimizer.is_cacheable("gh run list")
        assert CICDCommandOptimizer.is_cacheable("gh workflow list")
        assert CICDCommandOptimizer.is_cacheable("git log --oneline -10")
        assert not CICDCommandOptimizer.is_cacheable("echo hello")

    def test_should_batch_similar_commands(self):
        """Test batching detection for similar commands."""
        commands = [
            "gh run view 123",
            "gh run view 456",
            "gh run view 789",
        ]
        assert CICDCommandOptimizer.should_batch(commands)

    def test_should_not_batch_different_commands(self):
        """Test batching detection for different commands."""
        commands = [
            "gh run view 123",
            "gh workflow list",
            "git log",
        ]
        assert not CICDCommandOptimizer.should_batch(commands)

    def test_should_not_batch_single_command(self):
        """Test single command should not be batched."""
        commands = ["gh run view 123"]
        assert not CICDCommandOptimizer.should_batch(commands)


class TestCICDBatchViewRuns:
    """Test batch CI/CD run viewing."""

    @patch('victor.tools.cicd_optimizer.subprocess.run')
    @patch('victor.tools.cicd_optimizer.asyncio.create_subprocess_shell')
    async def test_batch_view_runs_empty_list(self, mock_subprocess, mock_run):
        """Test batch view with empty run ID list."""
        result = cicd_batch_view_runs([])
        assert "error" in result
        assert "No run IDs provided" in result["error"]

    @patch('victor.tools.cicd_optimizer.asyncio.create_subprocess_shell')
    async def test_batch_view_runs_success(self, mock_subprocess):
        """Test successful batch view of runs."""
        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (
            b'[{"databaseId": "123", "status": "completed", "conclusion": "success"}]',
            b''
        )
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        result = cicd_batch_view_runs(["123"])

        assert "runs" in result
        assert "123" in result["runs"]
        assert result["runs"]["123"]["status"] == "completed"


class TestCICDListRuns:
    """Test CI/CD run listing."""

    @patch('victor.tools.cicd_optimizer.optimize_cicd_query')
    def test_list_runs_success(self, mock_optimize):
        """Test successful run listing."""
        mock_optimize.return_value = (
            False,
            '[{"databaseId": "123", "status": "completed"}]',
            ''
        )

        result = cicd_list_runs(limit=10)

        assert "total" in result
        assert result["total"] == 1
        assert "runs" in result

    @patch('victor.tools.cicd_optimizer.optimize_cicd_query')
    def test_list_runs_with_filter(self, mock_optimize):
        """Test run listing with status filter."""
        mock_optimize.return_value = (
            False,
            '[{"databaseId": "123", "conclusion": "failure"}]',
            ''
        )

        result = cicd_list_runs(limit=10, status_filter="failure")

        assert "total" in result
        assert result["total"] == 1

    @patch('victor.tools.cicd_optimizer.optimize_cicd_query')
    def test_list_runs_api_error(self, mock_optimize):
        """Test run listing with API error."""
        mock_optimize.return_value = (False, "", "API rate limit exceeded")

        result = cicd_list_runs()

        assert "error" in result
        assert "Failed to list runs" in result["error"]


class TestCICDAnalyzeLogs:
    """Test CI/CD log analysis."""

    @patch('victor.tools.cicd_optimizer.CICDCommandOptimizer.aggregate_logs_to_file')
    @patch('victor.tools.cicd_optimizer.optimize_cicd_query')
    def test_analyze_logs_empty_list(self, mock_optimize, mock_aggregate):
        """Test log analysis with empty run ID list."""
        result = cicd_analyze_logs([])
        assert "error" in result
        assert "No run IDs provided" in result["error"]

    @patch('victor.tools.cicd_optimizer.CICDCommandOptimizer.aggregate_logs_to_file')
    @patch('victor.tools.cicd_optimizer.optimize_cicd_query')
    def test_analyze_logs_success(self, mock_optimize, mock_aggregate):
        """Test successful log analysis."""
        mock_optimize.return_value = (False, "Log line 1\nLog line 2\n", "")
        mock_aggregate.return_value = "/tmp/aggregated_logs.txt"

        result = cicd_analyze_logs(["123", "456"])

        assert "total_runs" in result
        assert result["total_runs"] == 2
        assert "aggregated_log_file" in result


class TestCICDCacheStats:
    """Test CI/CD cache statistics."""

    def test_cache_stats(self):
        """Test getting cache statistics."""
        # Add some entries to cache
        cache = get_cicd_cache()
        cache.set("cmd1", ("out1", "err1"))
        cache.set("cmd2", ("out2", "err2"))

        result = cicd_cache_stats()

        assert "cache_entries" in result
        assert result["cache_entries"] == 2
        assert "ttl_minutes" in result
        assert result["ttl_minutes"] == 5


class TestCICDClearCache:
    """Test CI/CD cache clearing."""

    def test_clear_cache(self):
        """Test clearing cache."""
        # Add some entries to cache
        cache = get_cicd_cache()
        cache.set("cmd1", ("out1", "err1"))
        cache.set("cmd2", ("out2", "err2"))

        result = cicd_clear_cache()

        assert "message" in result
        assert "entries_removed" in result
        assert result["entries_removed"] == 2
        assert len(cache._cache) == 0


class TestCICDRunDiagnosis:
    """Test CI/CD run diagnosis."""

    @patch('victor.tools.cicd_optimizer.optimize_cicd_query')
    def test_run_diagnosis_success(self, mock_optimize):
        """Test successful run diagnosis."""
        mock_optimize.return_value = (
            False,
            json.dumps({
                "databaseId": "123",
                "status": "completed",
                "conclusion": "failure",
                "name": "Test",
                "jobs": []
            }),
            ""
        )

        result = cicd_run_diagnosis("123")

        assert "run_id" in result
        assert result["run_id"] == "123"
        assert "status" in result
        assert result["status"] == "completed"
        assert "recommendations" in result

    @patch('victor.tools.cicd_optimizer.optimize_cicd_query')
    def test_run_diagnosis_api_error(self, mock_optimize):
        """Test run diagnosis with API error."""
        mock_optimize.return_value = (False, "", "Run not found")

        result = cicd_run_diagnosis("999")

        assert "error" in result
        assert "Failed to get run info" in result["error"]


class TestCICDIntegration:
    """Integration tests for CI/CD optimization."""

    @patch('victor.tools.cicd_optimizer.asyncio.create_subprocess_shell')
    async def test_cache_hit_avoids_api_call(self, mock_subprocess):
        """Test that cache hit avoids redundant API call."""
        from victor.tools.smart_cicd_tool import cicd_list_runs

        # Mock first call
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (
            b'[{"databaseId": "123"}]',
            b''
        )
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        # First call - should hit API
        result1 = cicd_list_runs(limit=10)
        assert mock_subprocess.call_count == 1

        # Second call - should hit cache
        result2 = cicd_list_runs(limit=10)
        assert mock_subprocess.call_count == 1  # No additional call

    def test_batch_view_reduces_calls(self):
        """Test that batch viewing reduces number of API calls."""
        from victor.tools.smart_cicd_tool import cicd_batch_view_runs

        with patch('victor.tools.cicd_optimizer.asyncio.create_subprocess_shell') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (
                b'[{"databaseId": "123"}, {"databaseId": "456"}, {"databaseId": "789"}]',
                b''
            )
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            # Batch view 3 runs
            result = cicd_batch_view_runs(["123", "456", "789"])

            # Should only make 1 API call instead of 3
            assert mock_subprocess.call_count == 1
            assert result["total_runs"] == 3
