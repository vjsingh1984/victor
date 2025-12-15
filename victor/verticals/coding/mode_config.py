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

"""Coding-specific mode configurations.

This module defines operational modes for software development tasks,
including tool budgets, iteration limits, and temperature settings.
"""

from __future__ import annotations

from typing import Dict

from victor.verticals.protocols import ModeConfig, ModeConfigProviderProtocol


# Mode configurations for coding tasks
CODING_MODE_CONFIGS: Dict[str, ModeConfig] = {
    "fast": ModeConfig(
        name="fast",
        tool_budget=5,
        max_iterations=10,
        temperature=0.5,
        description="Quick code changes with minimal exploration",
    ),
    "default": ModeConfig(
        name="default",
        tool_budget=10,
        max_iterations=30,
        temperature=0.7,
        description="Balanced mode for typical coding tasks",
    ),
    "explore": ModeConfig(
        name="explore",
        tool_budget=20,
        max_iterations=60,
        temperature=0.7,
        description="Extended exploration for understanding codebases",
    ),
    "thorough": ModeConfig(
        name="thorough",
        tool_budget=30,
        max_iterations=80,
        temperature=0.8,
        description="Deep analysis and comprehensive changes",
    ),
    "architect": ModeConfig(
        name="architect",
        tool_budget=40,
        max_iterations=100,
        temperature=0.8,
        description="Architecture analysis and design tasks",
    ),
    "refactor": ModeConfig(
        name="refactor",
        tool_budget=25,
        max_iterations=60,
        temperature=0.6,
        description="Code refactoring with safety checks",
    ),
    "debug": ModeConfig(
        name="debug",
        tool_budget=15,
        max_iterations=40,
        temperature=0.5,
        description="Debugging and issue investigation",
    ),
    "test": ModeConfig(
        name="test",
        tool_budget=15,
        max_iterations=40,
        temperature=0.5,
        description="Test creation and execution",
    ),
}


# Default tool budgets by task type (used when no mode specified)
DEFAULT_TOOL_BUDGETS: Dict[str, int] = {
    "code_generation": 3,
    "create_simple": 2,
    "create": 5,
    "edit": 5,
    "search": 6,
    "action": 15,
    "analysis_deep": 25,
    "analyze": 12,
    "design": 25,
    "refactor": 15,
    "debug": 12,
    "test": 10,
    "general": 8,
}


class CodingModeConfigProvider(ModeConfigProviderProtocol):
    """Mode configuration provider for coding vertical.

    Provides coding-specific operational modes with appropriate
    tool budgets and iteration limits for different task types.
    """

    def __init__(
        self,
        additional_modes: Dict[str, ModeConfig] | None = None,
        default_mode: str = "default",
    ):
        """Initialize the provider.

        Args:
            additional_modes: Additional or override modes
            default_mode: Name of the default mode
        """
        self._modes = CODING_MODE_CONFIGS.copy()
        if additional_modes:
            self._modes.update(additional_modes)
        self._default_mode = default_mode

    def get_mode_configs(self) -> Dict[str, ModeConfig]:
        """Get all coding mode configurations.

        Returns:
            Dict mapping mode names to configurations
        """
        return self._modes.copy()

    def get_default_mode(self) -> str:
        """Get the default mode name.

        Returns:
            Name of the default mode
        """
        return self._default_mode

    def get_default_tool_budget(self) -> int:
        """Get default tool budget when no mode is specified.

        Returns:
            Default tool call budget
        """
        return 10

    def get_tool_budget_for_task(self, task_type: str) -> int:
        """Get recommended tool budget for a specific task type.

        Args:
            task_type: The detected task type

        Returns:
            Recommended tool budget
        """
        return DEFAULT_TOOL_BUDGETS.get(task_type.lower(), self.get_default_tool_budget())


def get_mode_config(mode_name: str) -> ModeConfig | None:
    """Get a specific mode configuration.

    Convenience function for direct mode lookup.

    Args:
        mode_name: Name of the mode

    Returns:
        ModeConfig or None if not found
    """
    return CODING_MODE_CONFIGS.get(mode_name.lower())


def get_tool_budget(mode_name: str | None = None, task_type: str | None = None) -> int:
    """Get tool budget based on mode or task type.

    Args:
        mode_name: Optional mode name
        task_type: Optional task type

    Returns:
        Recommended tool budget
    """
    if mode_name:
        config = get_mode_config(mode_name)
        if config:
            return config.tool_budget

    if task_type:
        return DEFAULT_TOOL_BUDGETS.get(task_type.lower(), 10)

    return 10


__all__ = [
    "CodingModeConfigProvider",
    "CODING_MODE_CONFIGS",
    "DEFAULT_TOOL_BUDGETS",
    "get_mode_config",
    "get_tool_budget",
]
