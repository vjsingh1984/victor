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

"""Cache backend implementations for Victor AI.

This package provides pluggable cache backends implementing ICacheBackend:
- MemoryCacheBackend: Fast in-memory caching (default)
- RedisCacheBackend: Distributed caching with pub/sub
- SQLiteCacheBackend: Persistent caching across restarts
- CacheBackendFactory: Factory for creating backends from config

Usage:
    # Use factory to create backend
    from victor.agent.cache.backends import CacheBackendFactory

    config = {
        "type": "redis",
        "options": {
            "redis_url": "redis://localhost:6379/0",
            "default_ttl_seconds": 1800,
        }
    }
    backend = CacheBackendFactory.create_backend(config)

    await backend.connect()
    await backend.set("key", "value", "namespace")
    value = await backend.get("key", "namespace")
    await backend.disconnect()
"""

from victor.agent.cache.backends.factory import CacheBackendFactory
from victor.agent.cache.backends.memory import MemoryCacheBackend
from victor.agent.cache.backends.protocol import ICacheBackend
from victor.agent.cache.backends.redis import RedisCacheBackend
from victor.agent.cache.backends.sqlite import SQLiteCacheBackend

__all__ = [
    "ICacheBackend",
    "MemoryCacheBackend",
    "RedisCacheBackend",
    "SQLiteCacheBackend",
    "CacheBackendFactory",
]
