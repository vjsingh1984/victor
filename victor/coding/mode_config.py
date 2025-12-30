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

"""Coding-specific mode configurations using central registry.

This module registers coding-specific operational modes with the central
ModeConfigRegistry and exports a registry-based provider for protocol
compatibility.
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
# Coding-Specific Modes (Registered with Central Registry)
# =============================================================================

# Vertical-specific modes that extend/override defaults
_CODING_MODES: Dict[str, ModeDefinition] = {
    "architect": ModeDefinition(
        name="architect",
        tool_budget=40,
        max_iterations=100,
        temperature=0.8,
        description="Architecture analysis and design tasks",
        exploration_multiplier=2.5,
    ),
    "refactor": ModeDefinition(
        name="refactor",
        tool_budget=25,
        max_iterations=60,
        temperature=0.6,
        description="Code refactoring with safety checks",
        exploration_multiplier=1.5,
    ),
    "debug": ModeDefinition(
        name="debug",
        tool_budget=15,
        max_iterations=40,
        temperature=0.5,
        description="Debugging and issue investigation",
        exploration_multiplier=1.2,
    ),
    "test": ModeDefinition(
        name="test",
        tool_budget=15,
        max_iterations=40,
        temperature=0.5,
        description="Test creation and execution",
        exploration_multiplier=1.0,
    ),
}

# Coding-specific task type budgets
_CODING_TASK_BUDGETS: Dict[str, int] = {
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


# =============================================================================
# Register with Central Registry
# =============================================================================


def _register_coding_modes() -> None:
    """Register coding modes with the central registry."""
    registry = ModeConfigRegistry.get_instance()
    registry.register_vertical(
        name="coding",
        modes=_CODING_MODES,
        task_budgets=_CODING_TASK_BUDGETS,
        default_mode="default",
        default_budget=10,
    )


# Register on module load
_register_coding_modes()


# =============================================================================
# Provider (Protocol Compatibility)
# =============================================================================


class CodingModeConfigProvider(RegistryBasedModeConfigProvider):
    """Mode configuration provider for coding vertical.

    Uses the central ModeConfigRegistry but provides coding-specific
    complexity mapping.
    """

    def __init__(self) -> None:
        """Initialize coding mode provider."""
        # Ensure registration (idempotent - handles singleton reset)
        _register_coding_modes()
        super().__init__(
            vertical="coding",
            default_mode="default",
            default_budget=10,
        )

    def get_mode_for_complexity(self, complexity: str) -> str:
        """Map complexity level to coding mode.

        Args:
            complexity: Complexity level

        Returns:
            Recommended mode name
        """
        mapping = {
            "trivial": "fast",
            "simple": "fast",
            "moderate": "default",
            "complex": "thorough",
            "highly_complex": "architect",
        }
        return mapping.get(complexity, "default")


def get_mode_config(mode_name: str) -> ModeConfig | None:
    """Get a specific mode configuration.

    Args:
        mode_name: Name of the mode

    Returns:
        ModeConfig or None if not found
    """
    registry = ModeConfigRegistry.get_instance()
    modes = registry.get_modes("coding")
    return modes.get(mode_name.lower())


def get_tool_budget(mode_name: str | None = None, task_type: str | None = None) -> int:
    """Get tool budget based on mode or task type.

    Args:
        mode_name: Optional mode name
        task_type: Optional task type

    Returns:
        Recommended tool budget
    """
    registry = ModeConfigRegistry.get_instance()
    return registry.get_tool_budget(
        vertical="coding",
        mode_name=mode_name,
        task_type=task_type,
    )


__all__ = [
    "CodingModeConfigProvider",
    "get_mode_config",
    "get_tool_budget",
]
