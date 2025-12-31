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

"""Caching system for Victor.

This module has moved to victor.storage.cache.
Import from victor.storage.cache instead for new code.

This module provides backward-compatible re-exports.
"""

# Re-export from new location for backward compatibility
from victor.storage.cache.config import CacheConfig
from victor.storage.cache.tiered_cache import TieredCache

# New unified cache manager
from victor.storage.cache.manager import (
    CacheManager,
    CacheNamespace,
    CacheStats,
    get_cache_manager,
    set_cache_manager,
    reset_cache_manager,
    get_tools_cache,
    get_embeddings_cache,
    get_responses_cache,
    get_code_search_cache,
)

__all__ = [
    # Configuration
    "CacheConfig",
    # Low-level cache
    "TieredCache",
    # Unified cache manager
    "CacheManager",
    "CacheNamespace",
    "CacheStats",
    "get_cache_manager",
    "set_cache_manager",
    "reset_cache_manager",
    # Convenience accessors
    "get_tools_cache",
    "get_embeddings_cache",
    "get_responses_cache",
    "get_code_search_cache",
]
