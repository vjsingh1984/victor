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

"""Unified mode configuration registry for operational modes.

This module provides a central registry for operational modes across verticals,
eliminating ~180 LOC of duplicated mode configurations while providing:
- Default modes applicable to all verticals (quick, standard, comprehensive)
- Vertical-specific mode overrides and extensions
- Task-based tool budget recommendations
- Mode provider abstraction for backward compatibility

Example usage:
    from victor.core.mode_config import (
        ModeConfigRegistry,
        get_mode_config,
        get_tool_budget,
    )

    # Get registry instance
    registry = ModeConfigRegistry.get_instance()

    # Register vertical-specific modes
    registry.register_vertical(
        "coding",
        modes={"architect": ModeDefinition(tool_budget=40, ...)},
        task_budgets={"refactor": 15, "debug": 12},
    )

    # Get mode config (falls back to defaults)
    config = registry.get_mode("coding", "quick")

    # Get tool budget for task type
    budget = registry.get_tool_budget("coding", "refactor")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class ModeLevel(Enum):
    """Predefined mode levels with standard configurations."""

    QUICK = "quick"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    EXTENDED = "extended"


@dataclass
class ModeConfig:
    """Simplified mode configuration with core operational parameters.

    This is the primary dataclass for mode configuration, providing the essential
    parameters needed for agent operation control.

    Validation is performed at construction time via __post_init__.
    Invalid values will raise ValueError with descriptive messages.

    Attributes:
        tool_budget: Maximum number of tool calls allowed (1-500)
        max_iterations: Maximum conversation iterations (1-500)
        exploration_multiplier: Factor to multiply exploration iterations (0.1-10.0)
        allowed_tools: Optional set of allowed tool names (None means all tools allowed)
    """

    tool_budget: int
    max_iterations: int
    exploration_multiplier: float = 1.0
    allowed_tools: Optional[Union[List[str], Set[str]]] = None

    def __post_init__(self) -> None:
        """Validate configuration at construction time."""
        # Validate tool_budget (1-500)
        if not isinstance(self.tool_budget, int):
            self.tool_budget = int(self.tool_budget)  # type: ignore[unreachable]

        if not (1 <= self.tool_budget <= 500):
            raise ValueError(f"tool_budget must be between 1 and 500, got {self.tool_budget}")

        # Validate max_iterations (1-500)
        if not isinstance(self.max_iterations, int):
            self.max_iterations = int(self.max_iterations)  # type: ignore[unreachable]

        if not (1 <= self.max_iterations <= 500):
            raise ValueError(f"max_iterations must be between 1 and 500, got {self.max_iterations}")

        # Validate exploration_multiplier (0.1-10.0)
        if not isinstance(self.exploration_multiplier, (int, float)):
            self.exploration_multiplier = float(self.exploration_multiplier)  # type: ignore[unreachable]

        if not (0.1 <= self.exploration_multiplier <= 10.0):
            raise ValueError(
                f"exploration_multiplier must be between 0.1 and 10.0, "
                f"got {self.exploration_multiplier}"
            )

        # Convert allowed_tools list to set if needed
        if self.allowed_tools is not None and isinstance(self.allowed_tools, list):
            self.allowed_tools = set(self.allowed_tools)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModeConfig":
        """Create ModeConfig from dictionary with validation.

        Args:
            data: Dictionary with configuration values

        Returns:
            Validated ModeConfig instance

        Raises:
            ValueError: If configuration is invalid
        """
        return cls(
            tool_budget=data.get("tool_budget", 15),
            max_iterations=data.get("max_iterations", 10),
            exploration_multiplier=data.get("exploration_multiplier", 1.0),
            allowed_tools=data.get("allowed_tools"),
        )


@dataclass
class ModeDefinition:
    """Definition of an operational mode.

    Extended mode definition with additional metadata for vertical-specific
    configurations. Use ModeConfig for simpler use cases.

    Validation is performed at construction time via __post_init__.
    Invalid values will raise ValueError with descriptive messages.

    Attributes:
        name: Mode identifier (1-50 chars)
        tool_budget: Maximum tool calls allowed (1-500)
        max_iterations: Maximum conversation iterations (1-500)
        temperature: LLM temperature setting (0.0-2.0)
        description: Human-readable description (max 500 chars)
        exploration_multiplier: Factor to multiply exploration iterations (0.1-10.0)
        allowed_stages: Optional list of allowed workflow stages
        priority_tools: Optional list of tools to prioritize
        allowed_tools: Optional set of allowed tool names
        metadata: Additional mode-specific configuration
    """

    name: str
    tool_budget: int
    max_iterations: int
    temperature: float = 0.7
    description: str = ""
    exploration_multiplier: float = 1.0
    allowed_stages: List[str] = field(default_factory=list)
    priority_tools: List[str] = field(default_factory=list)
    allowed_tools: Optional[Union[List[str], Set[str]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration at construction time."""
        # Validate name (1-50 chars)
        if not isinstance(self.name, str):
            raise ValueError("name must be a string")
        if not (1 <= len(self.name) <= 50):
            raise ValueError(f"name must be between 1 and 50 characters, got {len(self.name)}")

        # Validate tool_budget (1-500)
        if not isinstance(self.tool_budget, int):
            self.tool_budget = int(self.tool_budget)  # type: ignore[unreachable]
        if not (1 <= self.tool_budget <= 500):
            raise ValueError(f"tool_budget must be between 1 and 500, got {self.tool_budget}")

        # Validate max_iterations (1-500)
        if not isinstance(self.max_iterations, int):
            self.max_iterations = int(self.max_iterations)  # type: ignore[unreachable]
        if not (1 <= self.max_iterations <= 500):
            raise ValueError(f"max_iterations must be between 1 and 500, got {self.max_iterations}")

        # Validate temperature (0.0-2.0)
        if not isinstance(self.temperature, (int, float)):
            self.temperature = float(self.temperature)  # type: ignore[unreachable]
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")

        # Validate description (max 500 chars)
        if not isinstance(self.description, str):
            self.description = str(self.description)  # type: ignore[unreachable]
        if len(self.description) > 500:
            logger.warning(f"description exceeds 500 chars ({len(self.description)}), truncating")
            self.description = self.description[:500]

        # Validate exploration_multiplier (0.1-10.0)
        if not isinstance(self.exploration_multiplier, (int, float)):
            self.exploration_multiplier = float(self.exploration_multiplier)  # type: ignore[unreachable]
        if not (0.1 <= self.exploration_multiplier <= 10.0):
            raise ValueError(
                f"exploration_multiplier must be between 0.1 and 10.0, "
                f"got {self.exploration_multiplier}"
            )

        # Convert allowed_tools list to set if needed
        if self.allowed_tools is not None and isinstance(self.allowed_tools, list):
            self.allowed_tools = set(self.allowed_tools)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModeDefinition":
        """Create ModeDefinition from dictionary with validation.

        Args:
            data: Dictionary with configuration values

        Returns:
            Validated ModeDefinition instance

        Raises:
            ValueError: If configuration is invalid
        """
        return cls(
            name=data.get("name", ""),
            tool_budget=data.get("tool_budget", 15),
            max_iterations=data.get("max_iterations", 10),
            temperature=data.get("temperature", 0.7),
            description=data.get("description", ""),
            exploration_multiplier=data.get("exploration_multiplier", 1.0),
            allowed_stages=data.get("allowed_stages", []),
            priority_tools=data.get("priority_tools", []),
            allowed_tools=data.get("allowed_tools"),
            metadata=data.get("metadata", {}),
        )

    def to_mode_config(self) -> ModeConfig:
        """Convert to simplified ModeConfig."""
        return ModeConfig(
            tool_budget=self.tool_budget,
            max_iterations=self.max_iterations,
            exploration_multiplier=self.exploration_multiplier,
            allowed_tools=self.allowed_tools,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "tool_budget": self.tool_budget,
            "max_iterations": self.max_iterations,
            "temperature": self.temperature,
            "description": self.description,
            "exploration_multiplier": self.exploration_multiplier,
            "allowed_stages": self.allowed_stages,
            "priority_tools": self.priority_tools,
            "allowed_tools": list(self.allowed_tools) if self.allowed_tools else None,
            "metadata": self.metadata,
        }


@dataclass
class VerticalModeConfig:
    """Mode configuration for a vertical.

    Attributes:
        vertical_name: Name of the vertical
        modes: Vertical-specific mode overrides
        task_budgets: Task type to tool budget mapping
        default_mode: Name of the default mode
        default_budget: Default tool budget when no mode specified
    """

    vertical_name: str
    modes: Dict[str, ModeDefinition] = field(default_factory=dict)
    task_budgets: Dict[str, int] = field(default_factory=dict)
    default_mode: str = "standard"
    default_budget: int = 10


# Default modes applicable to all verticals
DEFAULT_MODES: Dict[str, ModeDefinition] = {
    "quick": ModeDefinition(
        name="quick",
        tool_budget=10,
        max_iterations=10,
        temperature=0.5,
        description="Quick tasks with minimal exploration",
        exploration_multiplier=0.5,
    ),
    "fast": ModeDefinition(
        name="fast",
        tool_budget=10,
        max_iterations=10,
        temperature=0.5,
        description="Alias for quick mode",
        exploration_multiplier=0.5,
    ),
    "standard": ModeDefinition(
        name="standard",
        tool_budget=50,
        max_iterations=40,
        temperature=0.7,
        description="Balanced mode for typical tasks",
        exploration_multiplier=1.0,
    ),
    "default": ModeDefinition(
        name="default",
        tool_budget=25,
        max_iterations=30,
        temperature=0.7,
        description="Default operational mode",
        exploration_multiplier=1.0,
    ),
    "comprehensive": ModeDefinition(
        name="comprehensive",
        tool_budget=100,
        max_iterations=80,
        temperature=0.8,
        description="Thorough analysis and comprehensive changes",
        exploration_multiplier=2.0,
    ),
    "thorough": ModeDefinition(
        name="thorough",
        tool_budget=100,
        max_iterations=80,
        temperature=0.8,
        description="Alias for comprehensive mode",
        exploration_multiplier=2.0,
    ),
    "explore": ModeDefinition(
        name="explore",
        tool_budget=75,
        max_iterations=60,
        temperature=0.7,
        description="Extended exploration for understanding",
        exploration_multiplier=3.0,
    ),
    "plan": ModeDefinition(
        name="plan",
        tool_budget=50,
        max_iterations=50,
        temperature=0.6,
        description="Planning and analysis mode before implementation",
        exploration_multiplier=2.5,
    ),
    "extended": ModeDefinition(
        name="extended",
        tool_budget=100,
        max_iterations=100,
        temperature=0.8,
        description="Extended operations for large tasks",
        exploration_multiplier=2.5,
    ),
}

# Default task type budgets (fallback for all verticals)
# Significantly increased to support comprehensive exploration
DEFAULT_TASK_BUDGETS: Dict[str, int] = {
    # General tasks
    "general": 50,
    "simple": 10,
    "moderate": 25,
    "complex": 50,
    # Analysis tasks
    "analyze": 50,
    "analysis_deep": 100,
    "explore": 75,
    "search": 25,
    # Modification tasks
    "create": 15,
    "create_simple": 5,
    "edit": 15,
    "refactor": 50,
    # Specialized tasks
    "debug": 20,
    "test": 15,
    "design": 100,
}


class ModeConfigRegistry:
    """Central registry for operational mode configurations.

    Provides unified access to mode configurations across all verticals,
    with fallback to default modes and task-based budget recommendations.

    This is a singleton - use get_instance() to access.
    """

    _instance: Optional["ModeConfigRegistry"] = None

    def __init__(self) -> None:
        """Initialize the registry with default modes."""
        self._default_modes: Dict[str, ModeDefinition] = DEFAULT_MODES.copy()
        self._default_task_budgets: Dict[str, int] = DEFAULT_TASK_BUDGETS.copy()
        self._verticals: Dict[str, VerticalModeConfig] = {}
        self._mode_aliases: Dict[str, str] = {
            "fast": "quick",
            "thorough": "comprehensive",
        }

    @classmethod
    def get_instance(cls) -> "ModeConfigRegistry":
        """Get the singleton registry instance.

        Returns:
            The global ModeConfigRegistry instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    def register_vertical(
        self,
        name: str,
        modes: Optional[Dict[str, ModeDefinition]] = None,
        task_budgets: Optional[Dict[str, int]] = None,
        default_mode: str = "standard",
        default_budget: int = 10,
    ) -> None:
        """Register a vertical with its mode configurations.

        Args:
            name: Vertical name (e.g., "coding", "devops")
            modes: Vertical-specific mode definitions (overrides defaults)
            task_budgets: Task type to budget mapping
            default_mode: Default mode for this vertical
            default_budget: Default budget when no mode/task specified
        """
        self._verticals[name] = VerticalModeConfig(
            vertical_name=name,
            modes=modes or {},
            task_budgets=task_budgets or {},
            default_mode=default_mode,
            default_budget=default_budget,
        )

    def get_mode(self, vertical: Optional[str], mode_name: str) -> Optional[ModeDefinition]:
        """Get mode configuration, with vertical override support.

        Args:
            vertical: Vertical name (None for default only)
            mode_name: Mode name to look up

        Returns:
            ModeDefinition or None if not found
        """
        mode_name_lower = mode_name.lower()

        # Check vertical-specific mode first (before alias resolution)
        if vertical and vertical in self._verticals:
            vertical_config = self._verticals[vertical]
            if mode_name_lower in vertical_config.modes:
                return vertical_config.modes[mode_name_lower]

        # Check default modes (before alias resolution)
        if mode_name_lower in self._default_modes:
            return self._default_modes[mode_name_lower]

        # Resolve aliases and try again
        resolved_name = self._mode_aliases.get(mode_name_lower)
        if resolved_name:
            if vertical and vertical in self._verticals:
                vertical_config = self._verticals[vertical]
                if resolved_name in vertical_config.modes:
                    return vertical_config.modes[resolved_name]
            if resolved_name in self._default_modes:
                return self._default_modes[resolved_name]

        return None

    def get_mode_configs(self, vertical: Optional[str] = None) -> Dict[str, ModeDefinition]:
        """Get all mode configurations for a vertical.

        Args:
            vertical: Optional vertical name

        Returns:
            Dict of mode name to ModeDefinition
        """
        # Start with defaults
        modes = self._default_modes.copy()

        # Override with vertical-specific modes
        if vertical and vertical in self._verticals:
            modes.update(self._verticals[vertical].modes)

        return modes

    def get_tool_budget(
        self,
        vertical: Optional[str] = None,
        mode_name: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> int:
        """Get recommended tool budget.

        Priority order:
        1. Mode-specific budget (if mode_name provided)
        2. Vertical task-type budget (if vertical and task_type provided)
        3. Default task-type budget (if task_type provided)
        4. Vertical default budget
        5. Global default (10)

        Args:
            vertical: Optional vertical name
            mode_name: Optional mode name
            task_type: Optional task type

        Returns:
            Recommended tool budget
        """
        # 1. Try mode-specific budget
        if mode_name:
            mode = self.get_mode(vertical, mode_name)
            if mode:
                return mode.tool_budget

        # 2. Try vertical task-type budget
        if vertical and vertical in self._verticals and task_type:
            vertical_config = self._verticals[vertical]
            task_lower = task_type.lower()
            if task_lower in vertical_config.task_budgets:
                return vertical_config.task_budgets[task_lower]

        # 3. Try default task-type budget
        if task_type:
            task_lower = task_type.lower()
            if task_lower in self._default_task_budgets:
                return self._default_task_budgets[task_lower]

        # 4. Try vertical default
        if vertical and vertical in self._verticals:
            return self._verticals[vertical].default_budget

        # 5. Global default
        return 10

    def get_max_iterations(
        self,
        vertical: Optional[str] = None,
        mode_name: Optional[str] = None,
    ) -> int:
        """Get max iterations for a mode.

        Args:
            vertical: Optional vertical name
            mode_name: Optional mode name

        Returns:
            Max iterations (defaults to 30)
        """
        if mode_name:
            mode = self.get_mode(vertical, mode_name)
            if mode:
                return mode.max_iterations
        return 30

    def get_default_mode(self, vertical: Optional[str] = None) -> str:
        """Get the default mode name for a vertical.

        Args:
            vertical: Optional vertical name

        Returns:
            Default mode name
        """
        if vertical and vertical in self._verticals:
            return self._verticals[vertical].default_mode
        return "default"

    def list_modes(self, vertical: Optional[str] = None) -> List[str]:
        """List available mode names.

        Args:
            vertical: Optional vertical name

        Returns:
            List of mode names
        """
        modes = set(self._default_modes.keys())
        if vertical and vertical in self._verticals:
            modes.update(self._verticals[vertical].modes.keys())
        return sorted(modes)

    def list_verticals(self) -> List[str]:
        """List registered verticals.

        Returns:
            List of vertical names
        """
        return sorted(self._verticals.keys())

    def add_mode_alias(self, alias: str, target: str) -> None:
        """Add an alias for a mode name.

        Args:
            alias: Alias name
            target: Target mode name
        """
        self._mode_aliases[alias.lower()] = target.lower()

    def get_modes(self, vertical: Optional[str] = None) -> Dict[str, ModeConfig]:
        """Get all modes as simplified ModeConfig objects.

        Returns default modes when no vertical specified, or merged modes
        with vertical overrides when vertical is provided.

        Args:
            vertical: Optional vertical name for mode overrides

        Returns:
            Dict mapping mode name to ModeConfig
        """
        result: Dict[str, ModeConfig] = {}

        # Add defaults first
        for name, mode_def in self._default_modes.items():
            result[name] = mode_def.to_mode_config()

        # Override with vertical-specific modes
        if vertical and vertical in self._verticals:
            for name, mode_def in self._verticals[vertical].modes.items():
                result[name] = mode_def.to_mode_config()

        return result

    def register_modes(self, vertical: str, modes: Dict[str, ModeConfig]) -> None:
        """Register vertical-specific modes using simplified ModeConfig.

        This method allows registering modes using the simpler ModeConfig
        dataclass. The modes will be converted to ModeDefinition internally.

        Args:
            vertical: Vertical name
            modes: Dict mapping mode name to ModeConfig
        """
        mode_definitions: Dict[str, ModeDefinition] = {}
        for name, config in modes.items():
            mode_definitions[name] = ModeDefinition(
                name=name,
                tool_budget=config.tool_budget,
                max_iterations=config.max_iterations,
                exploration_multiplier=config.exploration_multiplier,
                allowed_tools=config.allowed_tools,
            )

        if vertical in self._verticals:
            self._verticals[vertical].modes.update(mode_definitions)
        else:
            self._verticals[vertical] = VerticalModeConfig(
                vertical_name=vertical,
                modes=mode_definitions,
            )


# =============================================================================
# Convenience Functions
# =============================================================================


def get_mode_config(mode_name: str, vertical: Optional[str] = None) -> Optional[ModeDefinition]:
    """Get mode configuration.

    Convenience function for direct mode lookup.

    Args:
        mode_name: Name of the mode
        vertical: Optional vertical name

    Returns:
        ModeDefinition or None if not found
    """
    registry = ModeConfigRegistry.get_instance()
    return registry.get_mode(vertical, mode_name)


def get_tool_budget(
    mode_name: Optional[str] = None,
    task_type: Optional[str] = None,
    vertical: Optional[str] = None,
) -> int:
    """Get tool budget based on mode or task type.

    Args:
        mode_name: Optional mode name
        task_type: Optional task type
        vertical: Optional vertical name

    Returns:
        Recommended tool budget
    """
    registry = ModeConfigRegistry.get_instance()
    return registry.get_tool_budget(vertical, mode_name, task_type)


def register_vertical_modes(
    vertical: str,
    modes: Dict[str, ModeDefinition],
    task_budgets: Optional[Dict[str, int]] = None,
) -> None:
    """Register modes for a vertical.

    Convenience function for vertical registration.

    Args:
        vertical: Vertical name
        modes: Mode definitions
        task_budgets: Optional task type budgets
    """
    registry = ModeConfigRegistry.get_instance()
    registry.register_vertical(vertical, modes, task_budgets)


# =============================================================================
# Complexity Mapper (Vertical-Aware Mode Selection)
# =============================================================================


class ComplexityMapper:
    """Maps complexity levels to mode names with vertical-specific overrides.

    This utility consolidates the duplicate get_mode_for_complexity() logic
    from all verticals into a single, configurable framework component.

    Each vertical can register custom mappings for specific complexity levels,
    with fallback to a standard base mapping.

    Example:
        mapper = ComplexityMapper("coding")
        mapper.register_override("highly_complex", "architect")
        mode = mapper.get_mode_for_complexity("highly_complex")  # Returns "architect"

        # For research vertical
        research_mapper = ComplexityMapper("research")
        research_mapper.register_override("complex", "deep")
        research_mapper.register_override("highly_complex", "academic")
    """

    # Standard complexity-to-mode mapping (used as fallback)
    _BASE_MAPPING: Dict[str, str] = {
        "trivial": "quick",
        "simple": "quick",
        "moderate": "standard",
        "complex": "comprehensive",
        "highly_complex": "extended",
    }

    def __init__(self, vertical: str) -> None:
        """Initialize complexity mapper for a vertical.

        Args:
            vertical: Vertical name (e.g., "coding", "devops", "research")
        """
        self._vertical = vertical
        self._overrides: Dict[str, str] = {}
        self._load_default_overrides()

    def _load_default_overrides(self) -> None:
        """Load default vertical-specific overrides.

        These are the common patterns observed across verticals.
        """
        # Coding vertical: fast for trivial/simple, thorough for complex, architect for highly complex
        if self._vertical == "coding":
            self._overrides = {
                "trivial": "fast",
                "simple": "fast",
                "moderate": "default",
                "complex": "thorough",
                "highly_complex": "architect",
            }

        # Benchmark vertical: fast mode for trivial/simple, thorough for complex
        elif self._vertical == "benchmark":
            self._overrides = {
                "trivial": "fast",
                "simple": "fast",
                "complex": "thorough",
                "highly_complex": "thorough",
            }

        # Research vertical: deep and academic modes
        elif self._vertical == "research":
            self._overrides = {
                "complex": "deep",
                "highly_complex": "academic",
            }

        # DevOps vertical: migration mode for highly complex tasks
        elif self._vertical == "devops":
            self._overrides = {"highly_complex": "migration"}

        # RAG vertical: index mode for highly complex tasks
        elif self._vertical == "rag":
            self._overrides = {"highly_complex": "index"}

        # DataAnalysis vertical: insights mode for complex tasks
        elif self._vertical == "dataanalysis":
            self._overrides = {"complex": "insights", "highly_complex": "insights"}

    def register_override(self, complexity: str, mode: str) -> None:
        """Register a vertical-specific override for a complexity level.

        Args:
            complexity: Complexity level (trivial, simple, moderate, complex, highly_complex)
            mode: Mode name to use for this complexity level
        """
        self._overrides[complexity] = mode

    def get_mode_for_complexity(self, complexity: str) -> str:
        """Get mode name for a complexity level.

        Args:
            complexity: Complexity level string

        Returns:
            Mode name (uses vertical-specific override if available,
                     otherwise falls back to base mapping)
        """
        # Try vertical-specific override first
        if complexity in self._overrides:
            return self._overrides[complexity]

        # Fall back to base mapping
        return self._BASE_MAPPING.get(complexity, "standard")

    @classmethod
    def get_base_mapping(cls) -> Dict[str, str]:
        """Get the standard base complexity-to-mode mapping.

        Returns:
            Dictionary mapping complexity levels to mode names
        """
        return cls._BASE_MAPPING.copy()


# =============================================================================
# Vertical Mode Registry (Pre-configured Defaults)
# =============================================================================


class VerticalModeDefaults:
    """Pre-configured mode defaults for all verticals.

    This class provides out-of-the-box configurations for all 6 verticals,
    eliminating the need for each vertical to register its own modes.

    Verticals can still override these defaults by calling register_vertical()
    with their own configurations if needed.
    """

    @staticmethod
    def get_coding_modes() -> Dict[str, ModeDefinition]:
        """Get coding-specific mode definitions.

        Returns:
            Dict of mode name to ModeDefinition
        """
        return {
            "architect": ModeDefinition(
                name="architect",
                tool_budget=100,
                max_iterations=100,
                temperature=0.8,
                description="Architecture analysis and design tasks",
                exploration_multiplier=2.5,
            ),
            "refactor": ModeDefinition(
                name="refactor",
                tool_budget=50,
                max_iterations=60,
                temperature=0.6,
                description="Code refactoring with safety checks",
                exploration_multiplier=1.5,
            ),
            "debug": ModeDefinition(
                name="debug",
                tool_budget=25,
                max_iterations=40,
                temperature=0.5,
                description="Debugging and issue investigation",
                exploration_multiplier=1.2,
            ),
            "test": ModeDefinition(
                name="test",
                tool_budget=20,
                max_iterations=40,
                temperature=0.5,
                description="Test creation and execution",
                exploration_multiplier=1.0,
            ),
        }

    @staticmethod
    def get_coding_task_budgets() -> Dict[str, int]:
        """Get coding-specific task budgets.

        Returns:
            Dict of task type to tool budget
        """
        return {
            "code_generation": 10,
            "create_simple": 5,
            "create": 15,
            "edit": 15,
            "search": 25,
            "action": 50,
            "analysis_deep": 100,
            "analyze": 50,
            "design": 100,
            "refactor": 50,
            "debug": 20,
            "test": 15,
            "general": 50,
        }

    @staticmethod
    def get_devops_modes() -> Dict[str, ModeDefinition]:
        """Get devops-specific mode definitions.

        Returns:
            Dict of mode name to ModeDefinition
        """
        return {
            "migration": ModeDefinition(
                name="migration",
                tool_budget=35,
                max_iterations=90,
                temperature=0.7,
                description="Complex infrastructure migrations",
                exploration_multiplier=2.5,
            ),
        }

    @staticmethod
    def get_devops_task_budgets() -> Dict[str, int]:
        """Get devops-specific task budgets.

        Returns:
            Dict of task type to tool budget
        """
        return {
            "deploy": 8,
            "infrastructure_create": 15,
            "migration": 25,
            "troubleshoot": 12,
        }

    @staticmethod
    def get_rag_modes() -> Dict[str, ModeDefinition]:
        """Get RAG-specific mode definitions.

        Returns:
            Dict of mode name to ModeDefinition
        """
        return {
            "index": ModeDefinition(
                name="index",
                tool_budget=50,
                max_iterations=70,
                temperature=0.6,
                description="Document ingestion and indexing",
                exploration_multiplier=1.5,
            ),
        }

    @staticmethod
    def get_rag_task_budgets() -> Dict[str, int]:
        """Get RAG-specific task budgets.

        Returns:
            Dict of task type to tool budget
        """
        return {
            "ingest": 10,
            "query": 5,
            "index_update": 15,
        }

    @staticmethod
    def get_dataanalysis_modes() -> Dict[str, ModeDefinition]:
        """Get dataanalysis-specific mode definitions.

        Returns:
            Dict of mode name to ModeDefinition
        """
        return {
            "insights": ModeDefinition(
                name="insights",
                tool_budget=75,
                max_iterations=80,
                temperature=0.7,
                description="Deep data analysis and insight generation",
                exploration_multiplier=2.0,
            ),
        }

    @staticmethod
    def get_dataanalysis_task_budgets() -> Dict[str, int]:
        """Get dataanalysis-specific task budgets.

        Returns:
            Dict of task type to tool budget
        """
        return {
            "analyze": 15,
            "visualize": 10,
            "transform": 12,
        }

    @staticmethod
    def get_research_modes() -> Dict[str, ModeDefinition]:
        """Get research-specific mode definitions.

        Returns:
            Dict of mode name to ModeDefinition
        """
        return {
            "deep": ModeDefinition(
                name="deep",
                tool_budget=75,
                max_iterations=70,
                temperature=0.7,
                description="Deep research with extensive analysis",
                exploration_multiplier=2.0,
            ),
            "academic": ModeDefinition(
                name="academic",
                tool_budget=100,
                max_iterations=100,
                temperature=0.8,
                description="Academic-level research with citation management",
                exploration_multiplier=3.0,
            ),
        }

    @staticmethod
    def get_research_task_budgets() -> Dict[str, int]:
        """Get research-specific task budgets.

        Returns:
            Dict of task type to tool budget
        """
        return {
            "quick_search": 5,
            "deep_research": 20,
            "synthesis": 15,
        }

    @staticmethod
    def get_benchmark_modes() -> Dict[str, ModeDefinition]:
        """Get benchmark-specific mode definitions.

        Returns:
            Dict of mode name to ModeDefinition
        """
        return {}

    @staticmethod
    def get_benchmark_task_budgets() -> Dict[str, int]:
        """Get benchmark-specific task budgets.

        Returns:
            Dict of task type to tool budget
        """
        return {
            "test_patch": 5,
            "verify_fix": 10,
            "run_benchmark": 15,
        }

    @classmethod
    def register_all_verticals(cls, registry: Optional[ModeConfigRegistry] = None) -> None:
        """Register all verticals with their default configurations.

        This is the main entry point for pre-configuring all verticals.
        Call this during framework initialization to set up all verticals.

        Args:
            registry: Optional registry instance (uses singleton if not provided)

        Example:
            from victor.core.mode_config import VerticalModeDefaults

            # Register all verticals with defaults
            VerticalModeDefaults.register_all_verticals()
        """
        if registry is None:
            registry = ModeConfigRegistry.get_instance()

        # Coding vertical
        registry.register_vertical(
            name="coding",
            modes=cls.get_coding_modes(),
            task_budgets=cls.get_coding_task_budgets(),
            default_mode="default",
            default_budget=10,
        )

        # DevOps vertical
        registry.register_vertical(
            name="devops",
            modes=cls.get_devops_modes(),
            task_budgets=cls.get_devops_task_budgets(),
            default_mode="standard",
            default_budget=15,
        )

        # RAG vertical
        registry.register_vertical(
            name="rag",
            modes=cls.get_rag_modes(),
            task_budgets=cls.get_rag_task_budgets(),
            default_mode="standard",
            default_budget=12,
        )

        # DataAnalysis vertical
        registry.register_vertical(
            name="dataanalysis",
            modes=cls.get_dataanalysis_modes(),
            task_budgets=cls.get_dataanalysis_task_budgets(),
            default_mode="standard",
            default_budget=15,
        )

        # Research vertical
        registry.register_vertical(
            name="research",
            modes=cls.get_research_modes(),
            task_budgets=cls.get_research_task_budgets(),
            default_mode="standard",
            default_budget=12,
        )

        # Benchmark vertical
        registry.register_vertical(
            name="benchmark",
            modes=cls.get_benchmark_modes(),
            task_budgets=cls.get_benchmark_task_budgets(),
            default_mode="fast",
            default_budget=10,
        )


# =============================================================================
# Registry-Based Provider (Protocol Adapter)
# =============================================================================


class RegistryBasedModeConfigProvider:
    """Mode config provider that delegates to the central registry.

    This class implements the ModeConfigProviderProtocol pattern but uses
    the central ModeConfigRegistry, eliminating duplicate code in verticals.

    Now enhanced with ComplexityMapper for vertical-aware complexity-to-mode mapping.

    Example:
        # In vertical __init__.py or mode_config.py
        from victor.core.mode_config import (
            ModeConfigRegistry,
            ModeDefinition,
            RegistryBasedModeConfigProvider,
        )

        # Register vertical-specific modes at import time
        registry = ModeConfigRegistry.get_instance()
        registry.register_vertical(
            "coding",
            modes={
                "architect": ModeDefinition(name="architect", tool_budget=40, ...),
                "refactor": ModeDefinition(name="refactor", tool_budget=25, ...),
            },
            task_budgets={"code_generation": 3, "refactor": 15},
        )

        # Export provider for protocol compatibility
        CodingModeConfigProvider = RegistryBasedModeConfigProvider("coding")
    """

    def __init__(
        self,
        vertical: str,
        default_mode: str = "default",
        default_budget: int = 10,
    ):
        """Initialize registry-based provider.

        Args:
            vertical: Vertical name for mode lookups
            default_mode: Default mode name
            default_budget: Default tool budget
        """
        self._vertical = vertical
        self._default_mode = default_mode
        self._default_budget = default_budget
        self._complexity_mapper = ComplexityMapper(vertical)

    def get_mode_configs(self) -> Dict[str, "ModeConfig"]:
        """Get mode configurations from the registry.

        Returns:
            Dict mapping mode names to ModeConfig
        """
        registry = ModeConfigRegistry.get_instance()
        return registry.get_modes(self._vertical)

    def get_default_mode(self) -> str:
        """Get the default mode name.

        Returns:
            Name of the default mode
        """
        registry = ModeConfigRegistry.get_instance()
        return registry.get_default_mode(self._vertical)

    def get_default_tool_budget(self, task_type: str | None = None) -> int:
        """Get default tool budget, optionally for a task type.

        Args:
            task_type: Optional task type for specialized budget

        Returns:
            Recommended tool budget
        """
        registry = ModeConfigRegistry.get_instance()
        return registry.get_tool_budget(
            vertical=self._vertical,
            task_type=task_type,
        )

    def get_default_max_iterations(self, task_type: str | None = None) -> int:
        """Get default max iterations.

        Args:
            task_type: Optional task type

        Returns:
            Max iterations (default 2x tool budget)
        """
        budget = self.get_default_tool_budget(task_type)
        return budget * 2

    def get_tool_budget_for_task(self, task_type: str) -> int:
        """Get recommended tool budget for a specific task type.

        Args:
            task_type: The detected task type

        Returns:
            Recommended tool budget
        """
        return self.get_default_tool_budget(task_type)

    def get_mode_for_complexity(self, complexity: str) -> str:
        """Map complexity level to mode name (vertical-aware).

        This method now uses the ComplexityMapper which provides vertical-specific
        overrides for complexity-to-mode mapping.

        Args:
            complexity: Complexity level (trivial, simple, moderate, complex, highly_complex)

        Returns:
            Recommended mode name (with vertical-specific overrides applied)
        """
        return self._complexity_mapper.get_mode_for_complexity(complexity)


__all__ = [
    # Enums
    "ModeLevel",
    # Dataclasses
    "ModeConfig",
    "ModeDefinition",
    "VerticalModeConfig",
    # Constants
    "DEFAULT_MODES",
    "DEFAULT_TASK_BUDGETS",
    # Registry
    "ModeConfigRegistry",
    # Complexity Mapper
    "ComplexityMapper",
    # Vertical Defaults
    "VerticalModeDefaults",
    # Provider
    "RegistryBasedModeConfigProvider",
    # Functions
    "get_mode_config",
    "get_tool_budget",
    "register_vertical_modes",
]
