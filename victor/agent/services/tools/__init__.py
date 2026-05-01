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

"""Canonical tool helper services.

This package contains helper components used to decompose tool behavior, but
the canonical runtime-owned ``ToolService`` lives in
``victor.agent.services.tool_service`` and is exported from
``victor.agent.services``.
"""

from victor.agent.services.tools.tool_selector_service import (
    ToolSelectorService,
    ToolSelectorServiceConfig,
)
from victor.agent.services.tools.tool_executor_service import (
    ToolExecutorService,
    ToolExecutorServiceConfig,
)
from victor.agent.services.tools.tool_tracker_service import (
    ToolTrackerService,
    ToolTrackerServiceConfig,
)
from victor.agent.services.tools.tool_planner_service import (
    ToolPlannerService,
    ToolPlannerServiceConfig,
)
from victor.agent.services.tools.tool_result_processor import (
    ToolResultProcessor,
    ToolResultProcessorConfig,
)

__all__ = [
    "ToolSelectorService",
    "ToolSelectorServiceConfig",
    "ToolExecutorService",
    "ToolExecutorServiceConfig",
    "ToolTrackerService",
    "ToolTrackerServiceConfig",
    "ToolPlannerService",
    "ToolPlannerServiceConfig",
    "ToolResultProcessor",
    "ToolResultProcessorConfig",
]
