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

"""Registry for tools with progressive parameters."""

from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProgressiveToolConfig:
    """Configuration for progressive tool parameters."""

    tool_name: str
    progressive_params: Dict[str, Any] = field(default_factory=dict)
    initial_values: Dict[str, Any] = field(default_factory=dict)
    max_values: Dict[str, Any] = field(default_factory=dict)


class ProgressiveToolsRegistry:
    """Registry for tools that support progressive parameter escalation.

    Removes hardcoded PROGRESSIVE_TOOLS from orchestrator.
    """

    _instance: Optional["ProgressiveToolsRegistry"] = None

    def __init__(self):
        self._tools: Dict[str, ProgressiveToolConfig] = {}

    @classmethod
    def get_instance(cls) -> "ProgressiveToolsRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def register(
        self,
        tool_name: str,
        progressive_params: Dict[str, Any],
        initial_values: Optional[Dict[str, Any]] = None,
        max_values: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._tools[tool_name] = ProgressiveToolConfig(
            tool_name=tool_name,
            progressive_params=progressive_params,
            initial_values=initial_values or {},
            max_values=max_values or {},
        )

    def is_progressive(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def get_config(self, tool_name: str) -> Optional[ProgressiveToolConfig]:
        return self._tools.get(tool_name)

    def list_progressive_tools(self) -> Set[str]:
        return set(self._tools.keys())


def get_progressive_registry() -> ProgressiveToolsRegistry:
    return ProgressiveToolsRegistry.get_instance()
