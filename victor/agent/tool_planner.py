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

"""DEPRECATED: ToolPlanner has moved to services.

.. deprecated::
    Import ToolPlanner from victor.agent.services.tool_planning_runtime instead.
    This module remains as a backward compatibility redirect.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "victor.agent.tool_planner is deprecated. "
    "Import ToolPlanner from victor.agent.services.tool_planning_runtime instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.agent.services.tool_planning_runtime import ToolPlanner

__all__ = ["ToolPlanner"]
