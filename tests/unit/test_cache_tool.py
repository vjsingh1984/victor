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

"""Tests for cache_tool module."""

import pytest
from unittest.mock import patch, MagicMock

from victor.tools.cache_tool import cache_stats, cache_clear, cache_info, set_cache_manager


class TestCacheStats:
    """Tests for cache_stats function."""

    @pytest.mark.asyncio
    async def test_cache_stats_no_manager(self):
        """Test cache_stats when no manager is set."""
        with patch("victor.tools.cache_tool._cache_manager", None):
            result = await cache_stats()
            assert result["success"] is False
            assert "not initialized" in result["error"]

    @pytest.mark.asyncio
    async def test_cache_stats_with_manager(self):
        """Test getting cache statistics."""
        mock_manager = MagicMock()
        mock_manager.get_stats.return_value = {
            "memory_hit_rate": 0.75,
            "disk_hit_rate": 0.5,
            "memory_hits": 100,
            "memory_misses": 25,
            "disk_hits": 50,
            "disk_misses": 50,
            "sets": 200,
        }
        with patch("victor.tools.cache_tool._cache_manager", mock_manager):
            result = await cache_stats()
            assert result["success"] is True
            assert "stats" in result
            assert "formatted_report" in result


class TestCacheClear:
    """Tests for cache_clear function."""

    @pytest.mark.asyncio
    async def test_cache_clear_no_manager(self):
        """Test cache_clear when no manager is set."""
        with patch("victor.tools.cache_tool._cache_manager", None):
            result = await cache_clear()
            assert result["success"] is False
            assert "not initialized" in result["error"]

    @pytest.mark.asyncio
    async def test_cache_clear_all(self):
        """Test clearing all cache."""
        mock_manager = MagicMock()
        mock_manager.clear.return_value = 10
        with patch("victor.tools.cache_tool._cache_manager", mock_manager):
            result = await cache_clear()
            assert result["success"] is True
            mock_manager.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_clear_namespace(self):
        """Test clearing specific namespace."""
        mock_manager = MagicMock()
        mock_manager.clear.return_value = 5
        with patch("victor.tools.cache_tool._cache_manager", mock_manager):
            result = await cache_clear(namespace="test")
            assert result["success"] is True


class TestCacheInfo:
    """Tests for cache_info function."""

    @pytest.mark.asyncio
    async def test_cache_info_no_manager(self):
        """Test cache_info when no manager is set."""
        with patch("victor.tools.cache_tool._cache_manager", None):
            result = await cache_info()
            assert result["success"] is False
            assert "not initialized" in result["error"]

    @pytest.mark.asyncio
    async def test_cache_info_with_manager(self):
        """Test getting cache info."""
        mock_manager = MagicMock()
        mock_manager.config.enable_memory = True
        mock_manager.config.enable_disk = True
        mock_manager.config.memory_max_size = 1000
        mock_manager.config.memory_ttl = 300
        mock_manager.config.disk_max_size = 1000000000
        mock_manager.config.disk_ttl = 86400
        mock_manager.config.disk_path = "/tmp/cache"
        with patch("victor.tools.cache_tool._cache_manager", mock_manager):
            result = await cache_info()
            assert result["success"] is True


class TestSetCacheManager:
    """Tests for set_cache_manager function."""

    def test_set_cache_manager(self):
        """Test setting the cache manager."""
        mock_manager = MagicMock()
        set_cache_manager(mock_manager)
        # The function sets the global, we can't directly test it
        # but we can verify no exception was raised
