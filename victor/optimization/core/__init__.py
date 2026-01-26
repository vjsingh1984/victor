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

"""Hot path optimizations for frequently called code.

This is the canonical location for hot path optimization utilities including:
- Lazy imports for heavy dependencies
- Optimized JSON serialization (using orjson when available)
- Thread-safe memoization
- Performance monitoring utilities
- Retry decorators

Usage:
    from victor.optimization.core import (
        json_dumps,
        json_loads,
        LazyImport,
        timed,
        retry,
    )

    # Fast JSON
    data = json_dumps({"key": "value"})

    # Lazy imports
    numpy = LazyImport("numpy")

    # Performance timing
    @timed
    def my_function():
        pass

Performance Impact:
    - JSON: 3-5x faster with orjson
    - Lazy imports: 20-30% faster startup
"""

# Import from local module (canonical location)
from victor.optimization.core.hot_path import (
    LazyImport,
    lazy_import,
    json_dumps,
    json_loads,
    json_dump,
    json_load,
    ThreadSafeMemoized,
    cached_property,
    timed,
    retry,
    async_retry,
    PerformanceMonitor,
)

# Common lazy imports
from victor.optimization.core.hot_path import (
    numpy,
    pandas,
    httpx,
)

__all__ = [
    "LazyImport",
    "lazy_import",
    "json_dumps",
    "json_loads",
    "json_dump",
    "json_load",
    "ThreadSafeMemoized",
    "cached_property",
    "timed",
    "retry",
    "async_retry",
    "PerformanceMonitor",
    "numpy",
    "pandas",
    "httpx",
]
