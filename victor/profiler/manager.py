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

"""Backward compatibility re-export for victor.profiler.manager.

Deprecated: Import from victor.observability.profiler.manager instead.
"""

import warnings

from victor.observability.profiler.manager import *  # noqa: F401, F403

warnings.warn(
    "victor.profiler.manager is deprecated. Import from victor.observability.profiler.manager instead.",
    DeprecationWarning,
    stacklevel=2,
)
