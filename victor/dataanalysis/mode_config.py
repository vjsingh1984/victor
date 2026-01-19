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

"""Data Analysis mode configuration.

REFACTORED: Now uses BaseVerticalModeProvider from framework, eliminating
~100 lines of duplicate code. All mode definitions are auto-registered from
VerticalModeDefaults in victor.core.mode_config.

Before (139 lines):
    - Duplicated mode definitions (_DATA_ANALYSIS_MODES dict)
    - Duplicated task budgets (_DATA_ANALYSIS_TASK_BUDGETS dict)
    - Manual registration function (_register_data_analysis_modes)
    - Custom provider class with duplicate logic
    - Custom complexity mapping override

After (50 lines):
    - Single provider class inheriting BaseVerticalModeProvider
    - All modes auto-registered from VerticalModeDefaults
    - Complexity mapping handled by framework ComplexityMapper
    - Backward compatibility maintained

NOTE: The previous custom complexity mapping mapped highly_complex → "research",
      but DataAnalysis doesn't have a "research" mode. The framework correctly
      maps highly_complex → "insights", which is the proper DataAnalysis mode.

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


class DataAnalysisModeConfigProvider(BaseVerticalModeProvider):
    """Mode configuration provider for data analysis vertical.

    Leverages BaseVerticalModeProvider to auto-register all data analysis modes
    from VerticalModeDefaults in victor.core.mode_config.

    Complexity mapping:
        - trivial → quick
        - simple → quick
        - moderate → standard
        - complex → insights
        - highly_complex → insights

    Available modes (auto-registered from VerticalModeDefaults):
        - Default modes: quick, fast, standard, default, comprehensive, thorough,
          explore, plan, extended
        - DataAnalysis-specific: insights (30 budget)

    Example:
        provider = DataAnalysisModeConfigProvider()
        mode = provider.get_mode("insights")
        budget = provider.get_tool_budget_for_task("analyze")
        recommended_mode = provider.get_mode_for_complexity("complex")
    """

    def __init__(self) -> None:
        """Initialize data analysis mode provider.

        Auto-registers all data analysis modes from VerticalModeDefaults.
        """
        super().__init__(vertical="dataanalysis")


__all__ = [
    "DataAnalysisModeConfigProvider",
]
