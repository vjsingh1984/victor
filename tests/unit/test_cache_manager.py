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

"""Tests for cache/manager module."""

import tempfile
from pathlib import Path

from victor.cache.manager import CacheManager
from victor.cache.config import CacheConfig


class TestCacheManager:
    """Tests for CacheManager class."""

    def test_cache_manager_default_init(self):
        """Test CacheManager initialization with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(disk_path=Path(tmpdir) / "cache")
            manager = CacheManager(config=config)
            assert manager.config is not None

    def test_cache_manager_memory_only(self):
        """Test CacheManager with memory cache only."""
        config = CacheConfig(enable_memory=True, enable_disk=False)
        manager = CacheManager(config=config)
        assert manager._memory_cache is not None
        assert manager._disk_cache is None

    def test_cache_manager_disk_only(self):
        """Test CacheManager with disk cache only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(
                enable_memory=False,
                enable_disk=True,
                disk_path=Path(tmpdir) / "cache",
            )
            manager = CacheManager(config=config)
            assert manager._memory_cache is None
            assert manager._disk_cache is not None

    def test_cache_manager_get_stats(self):
        """Test getting cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(disk_path=Path(tmpdir) / "cache")
            manager = CacheManager(config=config)
            stats = manager.get_stats()
            assert "memory_hits" in stats
            assert "disk_hits" in stats
            assert "sets" in stats

    def test_cache_manager_set_and_get(self):
        """Test setting and getting values from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(disk_path=Path(tmpdir) / "cache")
            manager = CacheManager(config=config)

            # Set a value
            manager.set("test_key", {"data": "value"})

            # Get the value
            result = manager.get("test_key")
            assert result == {"data": "value"}

    def test_cache_manager_get_missing(self):
        """Test getting a missing key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(disk_path=Path(tmpdir) / "cache")
            manager = CacheManager(config=config)

            result = manager.get("nonexistent_key")
            assert result is None

    def test_cache_manager_clear(self):
        """Test clearing cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(disk_path=Path(tmpdir) / "cache")
            manager = CacheManager(config=config)

            # Set a value
            manager.set("key1", "value1")

            # Clear cache
            manager.clear()

            # Value should be gone
            result = manager.get("key1")
            assert result is None


class TestCacheConfig:
    """Tests for CacheConfig class."""

    def test_cache_config_defaults(self):
        """Test CacheConfig default values."""
        config = CacheConfig()
        assert config.enable_memory is True
        assert config.enable_disk is True
        assert config.memory_max_size > 0
        assert config.memory_ttl > 0

    def test_cache_config_custom(self):
        """Test CacheConfig with custom values."""
        config = CacheConfig(
            enable_memory=False,
            enable_disk=True,
            memory_max_size=500,
            memory_ttl=600,
        )
        assert config.enable_memory is False
        assert config.memory_max_size == 500
        assert config.memory_ttl == 600
