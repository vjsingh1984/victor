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

"""Advanced caching system for Victor AI.

This package provides comprehensive caching capabilities:
- Multi-level cache (L1/L2 hierarchy)
- Cache warming strategies
- Semantic caching with vector similarity
- Intelligent invalidation system
- Analytics and monitoring

Modules:
    response_cache: Basic response caching with semantic similarity
    multi_level_cache: Two-tier cache system with L1/L2 hierarchy
    cache_warming: Proactive cache warming strategies
    semantic_cache: Enhanced semantic caching with vector search
    invalidation: Comprehensive cache invalidation system
    cache_analytics: Monitoring and analytics for cache systems
"""

from victor.core.cache.response_cache import (
    ResponseCache,
    CacheEntry,
    CacheStats,
    get_response_cache,
    reset_response_cache,
)

from victor.core.cache.multi_level_cache import (
    MultiLevelCache,
    CacheLevel,
    CacheLevelConfig,
    WritePolicy,
    EvictionPolicy,
)

from victor.core.cache.cache_warming import (
    CacheWarmer,
    AccessTracker,
    AccessPattern,
    WarmingStrategy,
    WarmingConfig,
)

from victor.core.cache.semantic_cache import (
    SemanticCache,
    SemanticCacheEntry,
)

from victor.core.cache.invalidation import (
    CacheInvalidator,
    CacheTagManager,
    InvalidationDependencyGraph,
    InvalidationStrategy,
    InvalidationConfig,
    TaggedEntry,
)

from victor.core.cache.cache_analytics import (
    CacheAnalytics,
    CacheMetrics,
    HotKey,
    Recommendation,
)

__all__ = [
    # Response cache (original)
    "ResponseCache",
    "CacheEntry",
    "CacheStats",
    "get_response_cache",
    "reset_response_cache",
    # Multi-level cache
    "MultiLevelCache",
    "CacheLevel",
    "CacheLevelConfig",
    "WritePolicy",
    "EvictionPolicy",
    # Cache warming
    "CacheWarmer",
    "AccessTracker",
    "AccessPattern",
    "WarmingStrategy",
    "WarmingConfig",
    # Semantic cache
    "SemanticCache",
    "SemanticCacheEntry",
    # Invalidation
    "CacheInvalidator",
    "CacheTagManager",
    "InvalidationDependencyGraph",
    "InvalidationStrategy",
    "InvalidationConfig",
    "TaggedEntry",
    # Analytics
    "CacheAnalytics",
    "CacheMetrics",
    "HotKey",
    "Recommendation",
]
