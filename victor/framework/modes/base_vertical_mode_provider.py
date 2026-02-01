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

"""Base vertical mode provider framework class.

This module provides BaseVerticalModeProvider, which eliminates ~400 lines of
duplication across verticals by:

1. Auto-registering modes from VerticalModeDefaults (centralized in core)
2. Providing vertical-aware complexity mapping via ComplexityMapper
3. Delegating to ModeConfigRegistry for all mode operations
4. Eliminating the need for verticals to define mode dictionaries

Verticals only need to:
- Inherit from BaseVerticalModeProvider
- Call super().__init__(vertical_name)
- Optionally override get_mode_for_complexity() if custom mapping needed

Example:
    # Before (187 lines in victor/coding/mode_config.py)
    _CODING_MODES = {
        "architect": ModeDefinition(...),
        "refactor": ModeDefinition(...),
        # ... 50+ lines
    }
    _CODING_TASK_BUDGETS = {
        "code_generation": 3,
        "refactor": 15,
        # ... 20+ lines
    }
    def _register_coding_modes():
        # ... 20+ lines of registration code
    class CodingModeConfigProvider(RegistryBasedModeConfigProvider):
        def __init__(self):
            _register_coding_modes()
            super().__init__("coding", ...)

    # After (10 lines!)
    from victor.framework.modes import BaseVerticalModeProvider

    class CodingModeProvider(BaseVerticalModeProvider):
        def __init__(self):
            super().__init__("coding")
            # That's it! All modes auto-registered from VerticalModeDefaults
"""

from __future__ import annotations

import logging
from typing import Optional

from victor.core.mode_config import (
    ModeConfigRegistry,
    ModeDefinition,
    RegistryBasedModeConfigProvider,
    VerticalModeDefaults,
)

logger = logging.getLogger(__name__)


class BaseVerticalModeProvider(RegistryBasedModeConfigProvider):
    """Base class for vertical mode configuration.

    Eliminates ~400 lines of duplication across verticals by leveraging the
    centralized VerticalModeDefaults from victor.core.mode_config.

    This class automatically:
    1. Registers vertical-specific modes from VerticalModeDefaults
    2. Sets up vertical-aware complexity mapping
    3. Delegates all mode operations to ModeConfigRegistry
    4. Provides fallback to default modes

    Verticals only need to inherit and call super().__init__(vertical_name).
    Custom complexity mapping can be provided by overriding
    get_mode_for_complexity().

    Example:
        from victor.framework.modes import BaseVerticalModeProvider

        class CodingModeProvider(BaseVerticalModeProvider):
            def __init__(self):
                super().__init__("coding")

        # Usage
        provider = CodingModeProvider()
        mode = provider.get_mode("architect")
        budget = provider.get_tool_budget_for_task("refactor")
        recommended_mode = provider.get_mode_for_complexity("complex")

    Architecture:
        - Uses VerticalModeDefaults for pre-configured mode definitions
        - Delegates to ModeConfigRegistry for mode lookup and storage
        - Uses ComplexityMapper for vertical-aware complexity mapping
        - Inherits from RegistryBasedModeConfigProvider for protocol compatibility
    """

    # Default mode and budget for each vertical
    _VERTICAL_DEFAULTS: dict[str, tuple[str, int]] = {
        "coding": ("default", 10),
        "devops": ("standard", 15),
        "research": ("standard", 12),
        "rag": ("standard", 12),
        "dataanalysis": ("standard", 15),
        "benchmark": ("fast", 10),
    }

    def __init__(
        self,
        vertical: str,
        auto_register: bool = True,
        use_defaults: bool = True,
    ):
        """Initialize the vertical mode provider.

        Args:
            vertical: Vertical name (e.g., "coding", "devops", "research")
            auto_register: If True (default), auto-register modes from VerticalModeDefaults
            use_defaults: If True (default), use predefined defaults for mode/budget
        """
        self._vertical = vertical.lower()

        # Auto-register modes from VerticalModeDefaults if enabled
        if auto_register:
            self._auto_register_modes()

        # Get defaults for this vertical
        if use_defaults and self._vertical in self._VERTICAL_DEFAULTS:
            default_mode, default_budget = self._VERTICAL_DEFAULTS[self._vertical]
        else:
            default_mode = "standard"
            default_budget = 10

        # Initialize parent with vertical and defaults
        super().__init__(
            vertical=self._vertical,
            default_mode=default_mode,
            default_budget=default_budget,
        )

        logger.debug(
            f"Initialized {self.__class__.__name__} for vertical '{self._vertical}' "
            f"with default_mode='{default_mode}', default_budget={default_budget}"
        )

    def _auto_register_modes(self) -> None:
        """Auto-register modes from VerticalModeDefaults.

        This method leverages the pre-configured mode definitions in
        VerticalModeDefaults, eliminating the need for each vertical to
        define its own mode dictionaries and registration functions.

        The registration is idempotent - safe to call multiple times.
        """
        registry = ModeConfigRegistry.get_instance()

        # Check if already registered (avoid duplicate registration)
        if self._vertical in registry.list_verticals():
            logger.debug(f"Vertical '{self._vertical}' already registered, skipping")
            return

        # Get modes and budgets from VerticalModeDefaults
        modes = self._get_modes_from_defaults()
        task_budgets = self._get_task_budgets_from_defaults()
        default_mode, default_budget = self._VERTICAL_DEFAULTS.get(self._vertical, ("standard", 10))

        # Register with central registry
        registry.register_vertical(
            name=self._vertical,
            modes=modes,
            task_budgets=task_budgets,
            default_mode=default_mode,
            default_budget=default_budget,
        )

        logger.info(
            f"Auto-registered {len(modes)} modes and {len(task_budgets)} "
            f"task budgets for vertical '{self._vertical}'"
        )

    def _get_modes_from_defaults(self) -> dict[str, ModeDefinition]:
        """Get mode definitions from VerticalModeDefaults.

        Returns:
            Dict of mode name to ModeDefinition for this vertical
        """
        # Map vertical names to getter methods
        getters = {
            "coding": VerticalModeDefaults.get_coding_modes,
            "devops": VerticalModeDefaults.get_devops_modes,
            "research": VerticalModeDefaults.get_research_modes,
            "rag": VerticalModeDefaults.get_rag_modes,
            "dataanalysis": VerticalModeDefaults.get_dataanalysis_modes,
            "benchmark": VerticalModeDefaults.get_benchmark_modes,
        }

        getter = getters.get(self._vertical)
        if getter:
            return getter()
        else:
            logger.warning(
                f"No predefined modes for vertical '{self._vertical}' in VerticalModeDefaults. "
                f"Override _get_modes_from_defaults() to provide custom modes."
            )
            return {}

    def _get_task_budgets_from_defaults(self) -> dict[str, int]:
        """Get task budgets from VerticalModeDefaults.

        Returns:
            Dict of task type to tool budget for this vertical
        """
        # Map vertical names to getter methods
        getters = {
            "coding": VerticalModeDefaults.get_coding_task_budgets,
            "devops": VerticalModeDefaults.get_devops_task_budgets,
            "research": VerticalModeDefaults.get_research_task_budgets,
            "rag": VerticalModeDefaults.get_rag_task_budgets,
            "dataanalysis": VerticalModeDefaults.get_dataanalysis_task_budgets,
            "benchmark": VerticalModeDefaults.get_benchmark_task_budgets,
        }

        getter = getters.get(self._vertical)
        if getter:
            return getter()
        else:
            logger.warning(
                f"No predefined task budgets for vertical '{self._vertical}' in VerticalModeDefaults. "
                f"Override _get_task_budgets_from_defaults() to provide custom budgets."
            )
            return {}

    def get_mode_for_complexity(self, complexity: str) -> str:
        """Map complexity level to mode name (vertical-aware).

        This method uses the ComplexityMapper which provides vertical-specific
        overrides for complexity-to-mode mapping.

        Verticals can override this method to provide custom complexity mapping
        if the default ComplexityMapper behavior doesn't meet their needs.

        Args:
            complexity: Complexity level (trivial, simple, moderate, complex, highly_complex)

        Returns:
            Recommended mode name (with vertical-specific overrides applied)

        Example:
            provider = CodingModeProvider()
            mode = provider.get_mode_for_complexity("highly_complex")  # Returns "architect"
        """
        # Use the parent's implementation which delegates to ComplexityMapper
        return super().get_mode_for_complexity(complexity)

    def list_modes(self) -> list[str]:
        """List all available modes for this vertical.

        Returns:
            List of mode names (including defaults and vertical-specific modes)

        Example:
            provider = CodingModeProvider()
            modes = provider.list_modes()
            # ["quick", "fast", "standard", "default", "comprehensive",
            #  "thorough", "explore", "plan", "extended", "architect", "refactor", ...]
        """
        registry = ModeConfigRegistry.get_instance()
        return registry.list_modes(self._vertical)

    def get_mode(self, mode_name: str) -> Optional[ModeDefinition]:
        """Get a specific mode definition.

        Args:
            mode_name: Name of the mode

        Returns:
            ModeDefinition or None if not found

        Example:
            provider = CodingModeProvider()
            architect_mode = provider.get_mode("architect")
            print(architect_mode.tool_budget)  # 40
        """
        registry = ModeConfigRegistry.get_instance()
        return registry.get_mode(self._vertical, mode_name)


__all__ = [
    "BaseVerticalModeProvider",
]
