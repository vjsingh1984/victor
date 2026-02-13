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

"""DevOps mode configurations using central registry.

This module registers DevOps-specific operational modes with the central
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
# DevOps-Specific Modes (Registered with Central Registry)
# =============================================================================

_DEVOPS_MODES: Dict[str, ModeDefinition] = {
    "migration": ModeDefinition(
        name="migration",
        tool_budget=60,
        max_iterations=120,
        temperature=0.7,
        description="Large-scale infrastructure migrations",
        exploration_multiplier=2.5,
        allowed_stages=[
            "INITIAL",
            "ASSESSMENT",
            "PLANNING",
            "IMPLEMENTATION",
            "VALIDATION",
            "DEPLOYMENT",
            "MONITORING",
            "COMPLETION",
        ],
    ),
}

# DevOps-specific task type budgets
_DEVOPS_TASK_BUDGETS: Dict[str, int] = {
    "dockerfile_simple": 5,
    "dockerfile_complex": 10,
    "docker_compose": 12,
    "ci_cd_basic": 15,
    "ci_cd_advanced": 25,
    "kubernetes_manifest": 15,
    "kubernetes_helm": 25,
    "terraform_module": 20,
    "terraform_full": 40,
    "monitoring_setup": 20,
}


# =============================================================================
# Register with Central Registry
# =============================================================================


def _register_devops_modes() -> None:
    """Register DevOps modes with the central registry."""
    registry = ModeConfigRegistry.get_instance()
    registry.register_vertical(
        name="devops",
        modes=_DEVOPS_MODES,
        task_budgets=_DEVOPS_TASK_BUDGETS,
        default_mode="standard",
        default_budget=20,
    )


# NOTE: Import-time auto-registration removed (SOLID compliance)
# Registration happens when DevOpsModeConfigProvider is instantiated during
# vertical integration. The provider's __init__ calls _register_devops_modes()
# for idempotent registration.


# =============================================================================
# Provider (Protocol Compatibility)
# =============================================================================


class DevOpsModeConfigProvider(RegistryBasedModeConfigProvider):
    """Mode configuration provider for DevOps vertical.

    Uses the central ModeConfigRegistry but provides DevOps-specific
    complexity mapping.
    """

    def __init__(self) -> None:
        """Initialize DevOps mode provider."""
        # Ensure registration (idempotent - handles singleton reset)
        _register_devops_modes()
        super().__init__(
            vertical="devops",
            default_mode="standard",
            default_budget=20,
        )

    def get_mode_for_complexity(self, complexity: str) -> str:
        """Map complexity level to DevOps mode.

        Args:
            complexity: Complexity level

        Returns:
            Recommended mode name
        """
        mapping = {
            "trivial": "quick",
            "simple": "quick",
            "moderate": "standard",
            "complex": "comprehensive",
            "highly_complex": "migration",
        }
        return mapping.get(complexity, "standard")


__all__ = [
    "DevOpsModeConfigProvider",
]
