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

"""Cache backend factory for creating backends from configuration.

This module provides CacheBackendFactory, a factory for creating cache
backends from configuration dictionaries or environment variables.

Supported Backend Types:
    - memory: In-memory caching (default, no dependencies)
    - redis: Distributed caching (requires redis>=4.0)
    - sqlite: Persistent caching (built-in, no dependencies)

Configuration Format:
    The factory accepts a configuration dictionary with the following structure:

    {
        "type": "memory" | "redis" | "sqlite",
        "options": {
            # Backend-specific options
        }
    }

    Memory Backend Options:
        - default_ttl_seconds: Default TTL (default: 3600)
        - enable_stats: Enable statistics (default: True)

    Redis Backend Options:
        - redis_url: Redis connection URL (required)
            Examples: "redis://localhost:6379/0", "rediss://localhost:6379/0"
        - key_prefix: Key prefix (default: "victor")
        - default_ttl_seconds: Default TTL (default: 3600)
        - connection_pool_size: Connection pool size (default: 10)

    SQLite Backend Options:
        - db_path: Database file path (default: ":memory:")
        - default_ttl_seconds: Default TTL (default: 3600)
        - cleanup_interval_seconds: Cleanup interval (default: 300)
        - enable_wal: Enable WAL mode (default: True)

Example:
    # Memory backend (default)
    backend = CacheBackendFactory.create_backend({"type": "memory"})

    # Redis backend
    backend = CacheBackendFactory.create_backend({
        "type": "redis",
        "options": {
            "redis_url": "redis://localhost:6379/0",
            "key_prefix": "myapp",
            "default_ttl_seconds": 1800,
        }
    })

    # SQLite backend
    backend = CacheBackendFactory.create_backend({
        "type": "sqlite",
        "options": {
            "db_path": "/var/cache/victor/cache.db",
            "default_ttl_seconds": 3600,
        }
    })

    # Use backend
    await backend.connect()
    await backend.set("key", "value", "namespace")
    value = await backend.get("key", "namespace")
    await backend.disconnect()
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from victor.agent.cache.backends.memory import MemoryCacheBackend
from victor.agent.cache.backends.protocol import ICacheBackend
from victor.agent.cache.backends.redis import RedisCacheBackend
from victor.agent.cache.backends.sqlite import SQLiteCacheBackend


logger = logging.getLogger(__name__)


class CacheBackendFactory:
    """Factory for creating cache backends from configuration.

    This factory provides a unified interface for creating different
    cache backends based on configuration. It validates configuration
    and handles missing dependencies gracefully.

    Example:
        # Create backend from config
        config = {
            "type": "redis",
            "options": {
                "redis_url": "redis://localhost:6379/0",
                "default_ttl_seconds": 1800,
            }
        }
        backend = CacheBackendFactory.create_backend(config)

        # Use backend
        await backend.connect()
        await backend.set("key", "value", "namespace")
        await backend.disconnect()
    """

    # Supported backend types
    BACKEND_MEMORY = "memory"
    BACKEND_REDIS = "redis"
    BACKEND_SQLITE = "sqlite"

    # Required options for each backend type
    REQUIRED_OPTIONS: Dict[str, set] = {
        BACKEND_REDIS: {"redis_url"},
        BACKEND_MEMORY: set(),
        BACKEND_SQLITE: set(),
    }

    @staticmethod
    def create_backend(config: Dict[str, Any]) -> ICacheBackend:
        """Create a cache backend from configuration.

        Args:
            config: Configuration dictionary with format:
                {
                    "type": "memory" | "redis" | "sqlite",
                    "options": {...}  # Optional, backend-specific options
                }

        Returns:
            Configured cache backend instance

        Raises:
            ValueError: If configuration is invalid
            ImportError: If required dependencies are missing

        Example:
            config = {
                "type": "redis",
                "options": {
                    "redis_url": "redis://localhost:6379/0",
                    "key_prefix": "myapp",
                }
            }
            backend = CacheBackendFactory.create_backend(config)
        """
        # Validate config
        if not isinstance(config, dict):
            raise ValueError(f"Config must be a dict, got {type(config).__name__}")

        if "type" not in config:
            raise ValueError("Config must specify 'type' field")

        backend_type = config["type"]
        options = config.get("options", {})

        # Create backend based on type
        if backend_type == CacheBackendFactory.BACKEND_MEMORY:
            return CacheBackendFactory._create_memory_backend(options)

        elif backend_type == CacheBackendFactory.BACKEND_REDIS:
            return CacheBackendFactory._create_redis_backend(options)

        elif backend_type == CacheBackendFactory.BACKEND_SQLITE:
            return CacheBackendFactory._create_sqlite_backend(options)

        else:
            supported = ", ".join(
                [
                    CacheBackendFactory.BACKEND_MEMORY,
                    CacheBackendFactory.BACKEND_REDIS,
                    CacheBackendFactory.BACKEND_SQLITE,
                ]
            )
            raise ValueError(
                f"Unknown backend type: {backend_type}. " f"Supported types: {supported}"
            )

    @staticmethod
    def _create_memory_backend(options: Dict[str, Any]) -> MemoryCacheBackend:
        """Create in-memory cache backend.

        Args:
            options: Backend options
                - default_ttl_seconds: Default TTL (default: 3600)
                - enable_stats: Enable statistics (default: True)

        Returns:
            Configured MemoryCacheBackend instance
        """
        default_ttl = options.get("default_ttl_seconds", 3600)
        enable_stats = options.get("enable_stats", True)

        return MemoryCacheBackend(
            default_ttl_seconds=default_ttl,
            enable_stats=enable_stats,
        )

    @staticmethod
    def _create_redis_backend(options: Dict[str, Any]) -> RedisCacheBackend:
        """Create Redis cache backend.

        Args:
            options: Backend options
                - redis_url: Redis connection URL (required)
                - key_prefix: Key prefix (default: "victor")
                - default_ttl_seconds: Default TTL (default: 3600)
                - connection_pool_size: Connection pool size (default: 10)

        Returns:
            Configured RedisCacheBackend instance

        Raises:
            ValueError: If required options are missing
        """
        # Check required options
        required = CacheBackendFactory.REQUIRED_OPTIONS[CacheBackendFactory.BACKEND_REDIS]
        missing = required - set(options.keys())
        if missing:
            raise ValueError(f"Missing required options for Redis backend: {', '.join(missing)}")

        redis_url = options["redis_url"]
        key_prefix = options.get("key_prefix", "victor")
        default_ttl = options.get("default_ttl_seconds", 3600)
        pool_size = options.get("connection_pool_size", 10)

        return RedisCacheBackend(
            redis_url=redis_url,
            key_prefix=key_prefix,
            default_ttl_seconds=default_ttl,
            connection_pool_size=pool_size,
        )

    @staticmethod
    def _create_sqlite_backend(options: Dict[str, Any]) -> SQLiteCacheBackend:
        """Create SQLite cache backend.

        Args:
            options: Backend options
                - db_path: Database file path (default: ":memory:")
                - default_ttl_seconds: Default TTL (default: 3600)
                - cleanup_interval_seconds: Cleanup interval (default: 300)
                - enable_wal: Enable WAL mode (default: True)

        Returns:
            Configured SQLiteCacheBackend instance
        """
        db_path = options.get("db_path", ":memory:")
        default_ttl = options.get("default_ttl_seconds", 3600)
        cleanup_interval = options.get("cleanup_interval_seconds", 300)
        enable_wal = options.get("enable_wal", True)

        return SQLiteCacheBackend(
            db_path=db_path,
            default_ttl_seconds=default_ttl,
            cleanup_interval_seconds=cleanup_interval,
            enable_wal=enable_wal,
        )

    @staticmethod
    def create_default_backend() -> ICacheBackend:
        """Create a default cache backend (in-memory).

        Returns:
            MemoryCacheBackend instance with default settings

        Example:
            backend = CacheBackendFactory.create_default_backend()
            await backend.connect()
        """
        return MemoryCacheBackend()

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate cache backend configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, error_messages)

        Example:
            config = {"type": "redis", "options": {"redis_url": "redis://localhost"}}
            is_valid, errors = CacheBackendFactory.validate_config(config)
            if not is_valid:
                print(f"Invalid config: {errors}")
        """
        errors = []

        # Check config type
        if not isinstance(config, dict):
            errors.append(f"Config must be a dict, got {type(config).__name__}")
            return False, errors

        # Check backend type
        if "type" not in config:
            errors.append("Config must specify 'type' field")
            return False, errors

        backend_type = config["type"]

        # Check if backend type is supported
        valid_types = [
            CacheBackendFactory.BACKEND_MEMORY,
            CacheBackendFactory.BACKEND_REDIS,
            CacheBackendFactory.BACKEND_SQLITE,
        ]
        if backend_type not in valid_types:
            errors.append(
                f"Unknown backend type: {backend_type}. " f"Supported: {', '.join(valid_types)}"
            )
            return False, errors

        # Check required options
        options = config.get("options", {})
        required = CacheBackendFactory.REQUIRED_OPTIONS.get(backend_type, set())
        missing = required - set(options.keys())

        if missing:
            errors.append(
                f"Missing required options for {backend_type} backend: {', '.join(missing)}"
            )

        return len(errors) == 0, errors


__all__ = [
    "CacheBackendFactory",
]
