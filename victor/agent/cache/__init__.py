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

"""Tool cache management with dependency-aware invalidation.

This package provides hierarchical caching for tool results with:
- Namespace-aware isolation (GLOBAL, SESSION, REQUEST, TOOL)
- Tool-to-tool dependency tracking
- Tool-to-file dependency tracking
- Cascading cache invalidation
- Pluggable cache backends via ICacheBackend protocol
"""

from victor.agent.cache.dependency_graph import ToolDependencyGraph
from victor.agent.cache.tool_cache_manager import (
    ToolCacheManager,
    CacheNamespace,
    CacheEntry,
)

__all__ = [
    "ToolCacheManager",
    "ToolDependencyGraph",
    "CacheNamespace",
    "CacheEntry",
]
