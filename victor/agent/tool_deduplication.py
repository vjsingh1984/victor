# Copyright 2025 Vijaykumar Singh <singhv@gmail.com>
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

"""Tool call deduplication tracker to prevent redundant operations.

.. deprecated::
    This module has been renamed to :mod:`victor.agent.tool_call_tracker`
    for clarity. The old name was confusing because it sounded similar to
    ``victor.tools.deduplication`` which handles tool DEFINITION deduplication.

    This module now re-exports everything from ``tool_call_tracker`` for
    backward compatibility.

    Use ``from victor.agent.tool_call_tracker import ToolCallTracker`` instead.

Key distinction:
    - **victor/agent/tool_call_tracker.py** → Tracks runtime tool CALLS (prevents redundant operations)
    - **victor/tools/deduplication/** → Deduplicates tool DEFINITIONS (resolves naming conflicts)
"""

import warnings

# Re-export everything from the new module for backward compatibility
from victor.agent.tool_call_tracker import (
    ToolCallTracker as ToolDeduplicationTracker,
    TrackedToolCall,
    get_tool_call_tracker as get_deduplication_tracker,
    reset_tool_call_tracker as reset_deduplication_tracker,
    is_redundant_call,
    track_call,
)

__all__ = [
    "ToolDeduplicationTracker",
    "TrackedToolCall",
    "get_deduplication_tracker",
    "reset_deduplication_tracker",
    "is_redundant_call",
    "track_call",
]

# Emit deprecation warning when module is imported
warnings.warn(
    "victor.agent.tool_deduplication is deprecated. "
    "Use victor.agent.tool_call_tracker instead. "
    "This module will be removed in version 0.10.0.",
    DeprecationWarning,
    stacklevel=2,
)
