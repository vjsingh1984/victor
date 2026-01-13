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

"""Research mode configurations using central registry.

This module now uses the consolidated framework defaults from
VerticalModeDefaults for complexity-to-mode mapping, eliminating
duplicate code while preserving research-specific mode configurations.

SOLID Refactoring: ComplexityMapper from framework handles complexity
mapping, eliminating ~15 LOC of duplicate logic.
"""

from __future__ import annotations

from typing import Dict

from victor.core.mode_config import (
    ModeConfig,
    ModeConfigRegistry,
    ModeDefinition,
    RegistryBasedModeConfigProvider,
)


# =============================================================================
# Research-Specific Modes (Registered with Central Registry)
# =============================================================================

_RESEARCH_MODES: Dict[str, ModeDefinition] = {
    "deep": ModeDefinition(
        name="deep",
        tool_budget=30,
        max_iterations=60,
        temperature=0.8,
        description="Comprehensive research with full verification cycle",
        exploration_multiplier=2.0,
        allowed_stages=[
            "INITIAL",
            "SEARCHING",
            "READING",
            "SYNTHESIZING",
            "WRITING",
            "VERIFICATION",
            "COMPLETION",
        ],
    ),
    "academic": ModeDefinition(
        name="academic",
        tool_budget=50,
        max_iterations=100,
        temperature=0.8,
        description="Thorough literature review with extensive citation",
        exploration_multiplier=3.0,
        allowed_stages=[
            "INITIAL",
            "SEARCHING",
            "READING",
            "SYNTHESIZING",
            "WRITING",
            "VERIFICATION",
            "COMPLETION",
        ],
    ),
}

# Research-specific task type budgets
_RESEARCH_TASK_BUDGETS: Dict[str, int] = {
    "simple_lookup": 3,
    "fact_check": 8,
    "comparison": 12,
    "trend_analysis": 20,
    "literature_review": 40,
    "comprehensive_report": 50,
}


# =============================================================================
# Register with Central Registry
# =============================================================================


def _register_research_modes() -> None:
    """Register research modes with the central registry.

    This function is idempotent - safe to call multiple times.
    Called by ResearchModeConfigProvider.__init__ when provider is instantiated
    during vertical integration. Module-level auto-registration removed to avoid
    load-order coupling.
    """
    registry = ModeConfigRegistry.get_instance()
    registry.register_vertical(
        name="research",
        modes=_RESEARCH_MODES,
        task_budgets=_RESEARCH_TASK_BUDGETS,
        default_mode="standard",
        default_budget=15,
    )


# NOTE: Import-time auto-registration removed (SOLID compliance)
# Registration happens when ResearchModeConfigProvider is instantiated during
# vertical integration. The provider's __init__ calls _register_research_modes()
# for idempotent registration.


# =============================================================================
# Provider (Protocol Compatibility)
# =============================================================================


class ResearchModeConfigProvider(RegistryBasedModeConfigProvider):
    """Mode configuration provider for research vertical.

    Uses the central ModeConfigRegistry with framework defaults for
    complexity mapping. ComplexityMapper from framework provides
    vertical-aware complexity-to-mode mapping (complex→deep, highly_complex→academic).

    SOLID Refactoring: No override needed for get_mode_for_complexity() -
    framework's ComplexityMapper provides identical mapping.
    """

    def __init__(self) -> None:
        """Initialize research mode provider."""
        # Ensure registration (idempotent - handles singleton reset)
        _register_research_modes()
        super().__init__(
            vertical="research",
            default_mode="standard",
            default_budget=15,
        )

    # NOTE: get_mode_for_complexity() now uses ComplexityMapper from framework
    # The framework provides identical mapping:
    #   trivial → quick, simple → quick, moderate → standard
    #   complex → deep, highly_complex → academic
    # No override needed - eliminated ~15 lines of duplicate code


__all__ = [
    "ResearchModeConfigProvider",
]
