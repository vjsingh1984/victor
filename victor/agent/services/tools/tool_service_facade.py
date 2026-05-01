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

"""Removed duplicate tool service facade.

The canonical runtime-owned ToolService lives in
``victor.agent.services.tool_service`` and is exported from
``victor.agent.services``. This module used to provide a second,
incompatible ToolService surface and has been intentionally removed.
"""

raise ImportError(
    "victor.agent.services.tools.tool_service_facade has been removed. "
    "Use victor.agent.services.tool_service.ToolService and "
    "victor.agent.services.tool_service.ToolServiceConfig instead."
)
