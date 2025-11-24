"""Caching system for Victor.

Tiered caching architecture:
- L1: Memory cache (fast, short-lived)
- L2: Disk cache (persistent, longer-lived)

Features:
- Automatic tiering
- TTL support
- Size limits
- Thread-safe
- No external dependencies
"""

from victor.cache.manager import CacheManager
from victor.cache.config import CacheConfig

__all__ = ["CacheManager", "CacheConfig"]
