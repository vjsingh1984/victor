"""Centralized format adapter for vertical configuration normalization.

This module addresses LSP (Liskov Substitution Principle) compliance by providing
a central adapter for normalizing vertical configuration formats. Instead of
scattered format conversions in individual verticals, all legacy format
adaptation happens here.

This ensures that:
1. All verticals can be safely swapped without breaking callers
2. Return types are consistent across all protocol implementations
3. Legacy formats are adapted centrally, not in each vertical

Usage:
    from victor.core.verticals.format_adapter import VerticalFormatAdapter

    # Normalize mode configs from any vertical
    raw_configs = mode_provider.get_mode_configs()
    normalized = VerticalFormatAdapter.normalize_mode_configs(raw_configs)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class ModeConfig:
    """Normalized mode configuration structure.

    This provides a consistent structure for mode configs regardless of
    how individual verticals define their configurations.
    """

    name: str
    tool_budget: int = 10
    max_iterations: int = 30
    temperature: float = 0.7
    description: str = ""
    allowed_stages: Optional[List[str]] = None
    exploration_multiplier: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "tool_budget": self.tool_budget,
            "max_iterations": self.max_iterations,
            "temperature": self.temperature,
            "description": self.description,
            "exploration_multiplier": self.exploration_multiplier,
        }
        if self.allowed_stages is not None:
            result["allowed_stages"] = self.allowed_stages
        return result


class VerticalFormatAdapter:
    """Central adapter for normalizing vertical configuration formats.

    This class provides static methods for converting between different
    configuration formats used by verticals, ensuring LSP compliance
    by providing consistent return types.

    All format conversion logic should be added here rather than in
    individual vertical implementations.
    """

    @staticmethod
    def normalize_mode_config(mode_config: Union["ModeConfig", Dict[str, Any], Any]) -> ModeConfig:
        """Convert any mode config format to standard ModeConfig.

        Handles:
        - ModeConfig instances (pass through)
        - Dict representations (convert to ModeConfig)
        - Dataclass instances with compatible attributes (extract values)

        Args:
            mode_config: Mode configuration in any supported format

        Returns:
            Normalized ModeConfig instance

        Raises:
            ValueError: If mode_config cannot be normalized
        """
        # Already a ModeConfig
        if isinstance(mode_config, ModeConfig):
            return mode_config

        # Dictionary representation
        if isinstance(mode_config, dict):
            return ModeConfig(
                name=mode_config.get("name", "unknown"),
                tool_budget=mode_config.get("tool_budget", 10),
                max_iterations=mode_config.get("max_iterations", 30),
                temperature=mode_config.get("temperature", 0.7),
                description=mode_config.get("description", ""),
                allowed_stages=mode_config.get("allowed_stages"),
                exploration_multiplier=mode_config.get("exploration_multiplier", 1.0),
            )

        # Dataclass or object with attributes
        if hasattr(mode_config, "tool_budget"):
            return ModeConfig(
                name=getattr(mode_config, "name", "unknown"),
                tool_budget=getattr(mode_config, "tool_budget", 10),
                max_iterations=getattr(mode_config, "max_iterations", 30),
                temperature=getattr(mode_config, "temperature", 0.7),
                description=getattr(mode_config, "description", ""),
                allowed_stages=getattr(mode_config, "allowed_stages", None),
                exploration_multiplier=getattr(mode_config, "exploration_multiplier", 1.0),
            )

        raise ValueError(f"Cannot normalize mode config of type {type(mode_config)}")

    @staticmethod
    def normalize_mode_configs(
        configs: Dict[str, Union["ModeConfig", Dict[str, Any], Any]],
    ) -> Dict[str, ModeConfig]:
        """Normalize all mode configs in a dictionary.

        Args:
            configs: Dictionary mapping mode names to configurations

        Returns:
            Dictionary with all values normalized to ModeConfig instances
        """
        return {
            name: VerticalFormatAdapter.normalize_mode_config(config)
            for name, config in configs.items()
        }

    @staticmethod
    def normalize_task_type_hint(hint: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Normalize a task type hint to consistent structure.

        Ensures all task type hints have required fields:
        - hint: str - The task hint text
        - tool_budget: int - Suggested tool budget
        - priority_tools: List[str] - Priority tools for this task type

        Args:
            hint: Task type hint in any supported format

        Returns:
            Normalized task type hint dictionary
        """
        if isinstance(hint, dict):
            return {
                "hint": hint.get("hint", ""),
                "tool_budget": hint.get("tool_budget", 10),
                "priority_tools": hint.get("priority_tools", []),
            }

        # Handle TaskTypeHint dataclass or similar
        if hasattr(hint, "hint"):
            return {
                "hint": getattr(hint, "hint", ""),
                "tool_budget": getattr(hint, "tool_budget", 10),
                "priority_tools": list(getattr(hint, "priority_tools", [])),
            }

        # Fallback - assume it's a string hint
        if isinstance(hint, str):
            return {
                "hint": hint,
                "tool_budget": 10,
                "priority_tools": [],
            }

        return {
            "hint": str(hint),
            "tool_budget": 10,
            "priority_tools": [],
        }

    @staticmethod
    def normalize_task_type_hints(hints: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Normalize all task type hints in a dictionary.

        Args:
            hints: Dictionary mapping task types to hints

        Returns:
            Dictionary with all values normalized
        """
        return {
            name: VerticalFormatAdapter.normalize_task_type_hint(hint)
            for name, hint in hints.items()
        }

    @staticmethod
    def normalize_safety_pattern(pattern: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Normalize a safety pattern to consistent structure.

        Args:
            pattern: Safety pattern in any supported format

        Returns:
            Normalized safety pattern dictionary
        """
        if isinstance(pattern, dict):
            return {
                "pattern": pattern.get("pattern", ""),
                "description": pattern.get("description", ""),
                "severity": pattern.get("severity", "warning"),
                "action": pattern.get("action", "warn"),
            }

        # Handle SafetyPattern dataclass or similar
        if hasattr(pattern, "pattern"):
            return {
                "pattern": getattr(pattern, "pattern", ""),
                "description": getattr(pattern, "description", ""),
                "severity": getattr(pattern, "severity", "warning"),
                "action": getattr(pattern, "action", "warn"),
            }

        # String pattern
        if isinstance(pattern, str):
            return {
                "pattern": pattern,
                "description": "",
                "severity": "warning",
                "action": "warn",
            }

        return {
            "pattern": str(pattern),
            "description": "",
            "severity": "warning",
            "action": "warn",
        }

    @staticmethod
    def normalize_safety_patterns(patterns: List[Any]) -> List[Dict[str, Any]]:
        """Normalize all safety patterns in a list.

        Args:
            patterns: List of safety patterns

        Returns:
            List with all patterns normalized
        """
        return [VerticalFormatAdapter.normalize_safety_pattern(p) for p in patterns]

    @staticmethod
    def normalize_stage_definition(stage: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Normalize a stage definition to consistent structure.

        Args:
            stage: Stage definition in any supported format

        Returns:
            Normalized stage definition dictionary
        """
        if isinstance(stage, dict):
            return {
                "name": stage.get("name", ""),
                "description": stage.get("description", ""),
                "tools": set(stage.get("tools", [])),
                "keywords": list(stage.get("keywords", [])),
                "next_stages": set(stage.get("next_stages", [])),
            }

        # Handle StageDefinition dataclass
        if hasattr(stage, "name"):
            return {
                "name": getattr(stage, "name", ""),
                "description": getattr(stage, "description", ""),
                "tools": set(getattr(stage, "tools", [])),
                "keywords": list(getattr(stage, "keywords", [])),
                "next_stages": set(getattr(stage, "next_stages", [])),
            }

        return {
            "name": str(stage),
            "description": "",
            "tools": set(),
            "keywords": [],
            "next_stages": set(),
        }

    @staticmethod
    def normalize_stage_definitions(stages: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Normalize all stage definitions in a dictionary.

        Args:
            stages: Dictionary mapping stage names to definitions

        Returns:
            Dictionary with all stages normalized
        """
        return {
            name: VerticalFormatAdapter.normalize_stage_definition(stage)
            for name, stage in stages.items()
        }
