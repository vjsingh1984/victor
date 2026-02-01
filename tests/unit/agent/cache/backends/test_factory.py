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

"""Tests for cache backend factory.

Tests the CacheBackendFactory for creating backends from configuration.
"""

from unittest.mock import patch

import pytest

from victor.agent.cache.backends.factory import CacheBackendFactory
from victor.agent.cache.backends.memory import MemoryCacheBackend
from victor.protocols import ICacheBackend
from victor.agent.cache.backends.sqlite import SQLiteCacheBackend


# =============================================================================
# Memory Backend Creation Tests
# =============================================================================


class TestMemoryBackendCreation:
    """Tests for creating in-memory cache backends."""

    def test_create_memory_backend_default_options(self):
        """Test creating memory backend with default options."""
        config = {"type": "memory"}

        backend = CacheBackendFactory.create_backend(config)

        assert isinstance(backend, MemoryCacheBackend)
        assert isinstance(backend, ICacheBackend)

    def test_create_memory_backend_with_options(self):
        """Test creating memory backend with custom options."""
        config = {
            "type": "memory",
            "options": {
                "default_ttl_seconds": 1800,
                "enable_stats": False,
            },
        }

        backend = CacheBackendFactory.create_backend(config)

        assert isinstance(backend, MemoryCacheBackend)

    def test_create_memory_backend_with_empty_options(self):
        """Test creating memory backend with empty options dict."""
        config = {"type": "memory", "options": {}}

        backend = CacheBackendFactory.create_backend(config)

        assert isinstance(backend, MemoryCacheBackend)


# =============================================================================
# Redis Backend Creation Tests
# =============================================================================


class TestRedisBackendCreation:
    """Tests for creating Redis cache backends."""

    @patch("victor.agent.cache.backends.redis.aioredis", None)  # Mock as if installed
    def test_create_redis_backend_with_required_options(self):
        """Test creating Redis backend with required options."""
        config = {
            "type": "redis",
            "options": {
                "redis_url": "redis://localhost:6379/0",
            },
        }

        # This will fail ImportError check, but we can test the factory logic
        with pytest.raises(ImportError):
            backend = CacheBackendFactory.create_backend(config)

    @patch("victor.agent.cache.backends.redis.aioredis", None)
    def test_create_redis_backend_missing_required_option(self):
        """Test that missing redis_url raises ValueError."""
        config = {
            "type": "redis",
            "options": {
                "key_prefix": "myapp",
            },
        }

        with pytest.raises(ValueError, match="Missing required options.*redis_url"):
            CacheBackendFactory.create_backend(config)

    @patch("victor.agent.cache.backends.redis.aioredis", None)
    def test_create_redis_backend_with_all_options(self):
        """Test creating Redis backend with all options."""
        config = {
            "type": "redis",
            "options": {
                "redis_url": "redis://localhost:6379/0",
                "key_prefix": "myapp",
                "default_ttl_seconds": 1800,
                "connection_pool_size": 20,
            },
        }

        with pytest.raises(ImportError):
            CacheBackendFactory.create_backend(config)


# =============================================================================
# SQLite Backend Creation Tests
# =============================================================================


class TestSQLiteBackendCreation:
    """Tests for creating SQLite cache backends."""

    def test_create_sqlite_backend_default_options(self):
        """Test creating SQLite backend with default options."""
        config = {"type": "sqlite"}

        backend = CacheBackendFactory.create_backend(config)

        assert isinstance(backend, SQLiteCacheBackend)
        assert isinstance(backend, ICacheBackend)

    def test_create_sqlite_backend_with_file_path(self):
        """Test creating SQLite backend with file path."""
        config = {
            "type": "sqlite",
            "options": {
                "db_path": "/tmp/cache.db",
            },
        }

        backend = CacheBackendFactory.create_backend(config)

        assert isinstance(backend, SQLiteCacheBackend)

    def test_create_sqlite_backend_with_all_options(self):
        """Test creating SQLite backend with all options."""
        config = {
            "type": "sqlite",
            "options": {
                "db_path": "/tmp/cache.db",
                "default_ttl_seconds": 1800,
                "cleanup_interval_seconds": 600,
                "enable_wal": False,
            },
        }

        backend = CacheBackendFactory.create_backend(config)

        assert isinstance(backend, SQLiteCacheBackend)


# =============================================================================
# Default Backend Tests
# =============================================================================


class TestDefaultBackend:
    """Tests for creating default backend."""

    def test_create_default_backend(self):
        """Test that default backend is MemoryCacheBackend."""
        backend = CacheBackendFactory.create_default_backend()

        assert isinstance(backend, MemoryCacheBackend)
        assert isinstance(backend, ICacheBackend)


# =============================================================================
# Configuration Validation Tests
# =============================================================================


class TestConfigurationValidation:
    """Tests for configuration validation."""

    def test_validate_valid_memory_config(self):
        """Test validating valid memory backend config."""
        config = {"type": "memory"}

        is_valid, errors = CacheBackendFactory.validate_config(config)

        assert is_valid
        assert len(errors) == 0

    def test_validate_valid_sqlite_config(self):
        """Test validating valid SQLite backend config."""
        config = {"type": "sqlite", "options": {"db_path": "/tmp/cache.db"}}

        is_valid, errors = CacheBackendFactory.validate_config(config)

        assert is_valid
        assert len(errors) == 0

    def test_validate_valid_redis_config(self):
        """Test validating valid Redis backend config."""
        config = {
            "type": "redis",
            "options": {"redis_url": "redis://localhost:6379/0"},
        }

        is_valid, errors = CacheBackendFactory.validate_config(config)

        assert is_valid
        assert len(errors) == 0

    def test_validate_missing_type(self):
        """Test that missing type field is invalid."""
        config = {"options": {}}

        is_valid, errors = CacheBackendFactory.validate_config(config)

        assert not is_valid
        assert any("type" in error for error in errors)

    def test_validate_unknown_backend_type(self):
        """Test that unknown backend type is invalid."""
        config = {"type": "memcached"}

        is_valid, errors = CacheBackendFactory.validate_config(config)

        assert not is_valid
        assert any("Unknown backend type" in error for error in errors)

    def test_validate_redis_missing_required_option(self):
        """Test that Redis config without redis_url is invalid."""
        config = {"type": "redis", "options": {"key_prefix": "myapp"}}

        is_valid, errors = CacheBackendFactory.validate_config(config)

        assert not is_valid
        assert any("Missing required options" in error for error in errors)
        assert any("redis_url" in error for error in errors)

    def test_validate_config_not_dict(self):
        """Test that non-dict config is invalid."""
        config = "memory"

        is_valid, errors = CacheBackendFactory.validate_config(config)

        assert not is_valid
        assert any("must be a dict" in error for error in errors)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_create_backend_with_invalid_type(self):
        """Test that invalid backend type raises ValueError."""
        config = {"type": "invalid_type"}

        with pytest.raises(ValueError, match="Unknown backend type"):
            CacheBackendFactory.create_backend(config)

    def test_create_backend_without_type(self):
        """Test that config without type raises ValueError."""
        config = {"options": {}}

        with pytest.raises(ValueError, match="must specify 'type' field"):
            CacheBackendFactory.create_backend(config)

    def test_create_backend_config_not_dict(self):
        """Test that non-dict config raises ValueError."""
        config = "memory"

        with pytest.raises(ValueError, match="must be a dict"):
            CacheBackendFactory.create_backend(config)

    def test_create_redis_backend_without_url(self):
        """Test that Redis config without URL raises ValueError."""
        config = {"type": "redis", "options": {}}

        with pytest.raises(ValueError, match="Missing required options.*redis_url"):
            CacheBackendFactory.create_backend(config)


# =============================================================================
# Integration Tests
# =============================================================================


class TestFactoryIntegration:
    """Integration tests for factory with backend operations."""

    @pytest.mark.asyncio
    async def test_memory_backend_integration(self):
        """Test that created memory backend works correctly."""
        config = {"type": "memory"}
        backend = CacheBackendFactory.create_backend(config)

        # Memory backend doesn't require connect()
        await backend.set("key1", "value1", "test_namespace")
        value = await backend.get("key1", "test_namespace")

        assert value == "value1"

    @pytest.mark.asyncio
    async def test_sqlite_backend_integration(self):
        """Test that created SQLite backend works correctly."""
        config = {"type": "sqlite", "options": {"db_path": ":memory:"}}
        backend = CacheBackendFactory.create_backend(config)

        await backend.connect()
        await backend.set("key1", "value1", "test_namespace")
        value = await backend.get("key1", "test_namespace")
        await backend.disconnect()

        assert value == "value1"
