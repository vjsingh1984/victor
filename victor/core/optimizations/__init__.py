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

.. deprecated:: 0.6.0
    This module is deprecated. Please migrate to ``victor.optimization.core``.
    This module will be removed in v1.0.0.

Migration Guide:
    Old (deprecated):
        from victor.core.optimizations import json_dumps, json_loads, LazyImport

    New (recommended):
        from victor.optimization.core import json_dumps, json_loads, LazyImport
        # Or use the unified import:
        from victor.optimization import json_dumps, json_loads, LazyImport
"""

import warnings

# Emit deprecation warning on import
warnings.warn(
    "victor.core.optimizations is deprecated and will be removed in v1.0.0. "
    "Use victor.optimization.core instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location
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
