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

"""Universal Registry System for Victor Framework.

This package provides unified registry infrastructure for managing
framework entities with configurable cache strategies and striped locks
for linear scalability.

Exports: UniversalRegistry[Any]: Type-safe generic registry with striped locks
    CacheStrategy: Cache invalidation strategies
    RegistryEntry: Registry entry with metadata
    StripedLockManager: Striped lock manager for concurrent access

Example:
    from victor.core.registries import UniversalRegistry, CacheStrategy

    # Get mode registry
    mode_registry = UniversalRegistry.get_registry("modes", CacheStrategy.LRU)

    # Register configuration
    mode_registry.register("build", config, namespace="coding")

    # Retrieve with automatic cache validation
    config = mode_registry.get("build", namespace="coding")
"""

from victor.core.registries.striped_locks import (
    LockMetrics,
    ReadWriteLock,
    StripedLockManager,
    StripedReadWriteLockManager,
)
from victor.core.registries.universal_registry import (
    CacheStrategy,
    RegistryEntry,
    UniversalRegistry,
    create_universal_registry,
)

__all__ = [
    "UniversalRegistry",
    "CacheStrategy",
    "RegistryEntry",
    "create_universal_registry",
    "StripedLockManager",
    "ReadWriteLock",
    "StripedReadWriteLockManager",
    "LockMetrics",
]
