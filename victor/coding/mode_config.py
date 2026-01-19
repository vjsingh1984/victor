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

"""Coding-specific mode configuration.

REFACTORED: Now uses BaseVerticalModeProvider from framework, eliminating
~137 lines of duplicate code. All mode definitions are auto-registered from
VerticalModeDefaults in victor.core.mode_config.

Before (187 lines):
    - Duplicated mode definitions (_CODING_MODES dict)
    - Duplicated task budgets (_CODING_TASK_BUDGETS dict)
    - Manual registration function (_register_coding_modes)
    - Custom provider class with duplicate logic

After (50 lines):
    - Single provider class inheriting BaseVerticalModeProvider
    - All modes auto-registered from VerticalModeDefaults
    - Complexity mapping handled by framework ComplexityMapper
    - Backward compatibility maintained via convenience functions

SOLID Compliance:
    - SRP: BaseVerticalModeProvider handles registration, complexity mapping
    - DIP: Depends on abstractions (RegistryBasedModeConfigProvider)
    - OCP: Open for extension (can override get_mode_for_complexity)
"""

from __future__ import annotations

from victor.core.mode_config import ModeConfig
from victor.framework.modes import BaseVerticalModeProvider


# =============================================================================
# Provider (Uses Framework Base)
# =============================================================================


class CodingModeConfigProvider(BaseVerticalModeProvider):
    """Mode configuration provider for coding vertical.

    Leverages BaseVerticalModeProvider to auto-register all coding modes
    from VerticalModeDefaults in victor.core.mode_config.

    Complexity mapping:
        - trivial → fast
        - simple → fast
        - moderate → default
        - complex → thorough
        - highly_complex → architect

    Available modes (auto-registered from VerticalModeDefaults):
        - Default modes: quick, fast, standard, default, comprehensive, thorough,
          explore, plan, extended
        - Coding-specific: architect (40 budget), refactor (25 budget),
          debug (15 budget), test (15 budget)

    Example:
        provider = CodingModeConfigProvider()
        mode = provider.get_mode("architect")
        budget = provider.get_tool_budget_for_task("refactor")
        recommended_mode = provider.get_mode_for_complexity("complex")
    """

    def __init__(self) -> None:
        """Initialize coding mode provider.

        Auto-registers all coding modes from VerticalModeDefaults.
        """
        super().__init__(vertical="coding")


# =============================================================================
# Convenience Functions (Backward Compatibility)
# =============================================================================


def get_mode_config(mode_name: str) -> ModeConfig | None:
    """Get a specific mode configuration.

    Args:
        mode_name: Name of the mode

    Returns:
        ModeConfig or None if not found

    Example:
        mode = get_mode_config("architect")
        print(mode.tool_budget)  # 40
    """
    provider = CodingModeConfigProvider()
    mode = provider.get_mode(mode_name)
    return mode.to_mode_config() if mode else None


def get_tool_budget(mode_name: str | None = None, task_type: str | None = None) -> int:
    """Get tool budget based on mode or task type.

    Args:
        mode_name: Optional mode name
        task_type: Optional task type

    Returns:
        Recommended tool budget

    Example:
        budget = get_tool_budget(task_type="refactor")  # 15
        budget = get_tool_budget(mode_name="architect")  # 40
    """
    provider = CodingModeConfigProvider()
    if mode_name:
        mode = provider.get_mode(mode_name)
        return mode.tool_budget if mode else 10
    return provider.get_tool_budget_for_task(task_type) if task_type else 10


__all__ = [
    "CodingModeConfigProvider",
    "get_mode_config",
    "get_tool_budget",
]
