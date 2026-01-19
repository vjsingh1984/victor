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

"""Research mode configuration.

REFACTORED: Now uses BaseVerticalModeProvider from framework, eliminating
~104 lines of duplicate code. All mode definitions are auto-registered from
VerticalModeDefaults in victor.core.mode_config.

Before (154 lines):
    - Duplicated mode definitions (_RESEARCH_MODES dict)
    - Duplicated task budgets (_RESEARCH_TASK_BUDGETS dict)
    - Manual registration function (_register_research_modes)
    - Custom provider class with duplicate logic

After (50 lines):
    - Single provider class inheriting BaseVerticalModeProvider
    - All modes auto-registered from VerticalModeDefaults
    - Complexity mapping handled by framework ComplexityMapper
    - Backward compatibility maintained

SOLID Compliance:
    - SRP: BaseVerticalModeProvider handles registration, complexity mapping
    - DIP: Depends on abstractions (RegistryBasedModeConfigProvider)
    - OCP: Open for extension (can override get_mode_for_complexity)
"""

from __future__ import annotations

from victor.framework.modes import BaseVerticalModeProvider


# =============================================================================
# Provider (Uses Framework Base)
# =============================================================================


class ResearchModeConfigProvider(BaseVerticalModeProvider):
    """Mode configuration provider for research vertical.

    Leverages BaseVerticalModeProvider to auto-register all research modes
    from VerticalModeDefaults in victor.core.mode_config.

    Complexity mapping:
        - trivial → quick
        - simple → quick
        - moderate → standard
        - complex → deep
        - highly_complex → academic

    Available modes (auto-registered from VerticalModeDefaults):
        - Default modes: quick, fast, standard, default, comprehensive, thorough,
          explore, plan, extended
        - Research-specific: deep (25 budget), academic (40 budget)

    Example:
        provider = ResearchModeConfigProvider()
        mode = provider.get_mode("academic")
        budget = provider.get_tool_budget_for_task("literature_review")
        recommended_mode = provider.get_mode_for_complexity("complex")
    """

    def __init__(self) -> None:
        """Initialize research mode provider.

        Auto-registers all research modes from VerticalModeDefaults.
        """
        super().__init__(vertical="research")


__all__ = [
    "ResearchModeConfigProvider",
]
