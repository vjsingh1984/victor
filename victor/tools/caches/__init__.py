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

"""Tool selection caching infrastructure.

This module provides high-performance caching for tool selection operations
to reduce latency by 30-50%.

Components:
    - CacheKeyGenerator: Generates cache keys from queries, context, config
    - ToolSelectionCache: LRU cache with TTL for selections
    - CachedSelection: Wrapper for cached tool selections

Usage:
    from victor.tools.caches import (
        get_tool_selection_cache,
        get_cache_key_generator,
    )

    # Get cache instance
    cache = get_tool_selection_cache()

    # Get key generator
    key_gen = get_cache_key_generator()

    # Generate cache key
    key = key_gen.generate_query_key(
        query="read the file",
        tools_hash="abc123...",
        config_hash="def456..."
    )

    # Check cache
    cached = cache.get_query(key)
    if cached:
        tools = cached.value
    else:
        # Perform selection
        tools = select_tools(...)
        cache.put_query(key, tools)
"""

from victor.tools.caches.adaptive_cache import (
    AdaptiveMetrics,
    AdaptiveLRUCache,
)
from victor.tools.caches.adaptive_ttl import AdaptiveTTLCache
from victor.tools.caches.advanced_cache_manager import (
    AdvancedCacheManager,
    AdvancedCacheMetrics,
    get_advanced_cache,
    reset_advanced_cache,
)
from victor.tools.caches.cache_keys import (
    CacheKeyGenerator,
    calculate_tools_hash,
    generate_context_key,
    generate_query_key,
    get_cache_key_generator,
)
from victor.tools.caches.multi_level_cache import (
    CacheEntry,
    LevelMetrics,
    MultiLevelCache,
)
from victor.tools.caches.persistent_cache import (
    PersistentCacheEntry,
    PersistentSelectionCache,
    get_persistent_cache,
    reset_persistent_cache,
)
from victor.tools.caches.predictive_warmer import (
    PredictionResult,
    PredictiveCacheWarmer,
    QueryPattern,
)
from victor.tools.caches.selection_cache import (
    CachedSelection,
    CacheMetrics,
    ToolSelectionCache,
    get_tool_selection_cache,
    invalidate_tool_selection_cache,
)

__all__ = [
    # Cache key generation
    "CacheKeyGenerator",
    "get_cache_key_generator",
    "generate_query_key",
    "generate_context_key",
    "calculate_tools_hash",
    # Selection cache
    "ToolSelectionCache",
    "get_tool_selection_cache",
    "invalidate_tool_selection_cache",
    "CachedSelection",
    "CacheMetrics",
    # Advanced optimization
    "AdaptiveMetrics",
    "AdaptiveLRUCache",
    "AdaptiveTTLCache",
    "PredictionResult",
    "PredictiveCacheWarmer",
    "QueryPattern",
    "CacheEntry",
    "LevelMetrics",
    "MultiLevelCache",
    "PersistentCacheEntry",
    "PersistentSelectionCache",
    "get_persistent_cache",
    "reset_persistent_cache",
    # Advanced cache manager
    "AdvancedCacheManager",
    "AdvancedCacheMetrics",
    "get_advanced_cache",
    "reset_advanced_cache",
]
