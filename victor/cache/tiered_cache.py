# Backwards compatibility redirect
# This module has been moved to victor.storage.cache.tiered_cache
from victor.storage.cache.tiered_cache import * # noqa: F403
from victor.storage.cache.tiered_cache import TieredCache

# Legacy alias for backward compatibility
CacheManager = TieredCache
