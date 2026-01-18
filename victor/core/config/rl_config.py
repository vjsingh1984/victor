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

"""Data-Driven RL Configuration System.

This module provides a centralized, YAML-based configuration system for
reinforcement learning across all verticals, complementing the existing
BaseRLConfig pattern while adding data-driven flexibility.

Design Patterns:
    - Registry: Singleton RLConfigRegistry for config access
    - Factory: Generate defaults when YAML not found
    - Data-Driven: YAML files override code defaults
    - Compatibility: Works with existing BaseRLConfig subclasses

Use Cases:
    - RL configuration (learners, task mappings, quality thresholds)
    - Vertical-specific RL overrides
    - Provider patience configuration
    - Tool selection optimization

Example:
    from victor.core.config import RLConfigRegistry

    registry = RLConfigRegistry.get_instance()
    config = registry.get_rl_config("coding")
    print(config.task_type_mappings)  # {"debugging": ["read", "grep"], ...}
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from victor.framework.rl import LearnerType

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Reinforcement learning configuration for a vertical.

    Loaded from YAML files in victor/config/rl/. This provides
    a data-driven alternative to hard-coded BaseRLConfig subclasses,
    while maintaining compatibility with the existing framework.

    Attributes:
        vertical_name: Name of the vertical
        active_learners: List of learner types to activate
        task_type_mappings: Maps task types to recommended tools
        quality_thresholds: Quality thresholds by task type (0.0-1.0)
        default_patience: Continuation patience by provider
        exploration_bonus: Bonus for tool selection exploration
        extensions: Vertical-specific extensions (e.g., preferred providers)
    """

    vertical_name: str
    active_learners: List[LearnerType] = field(default_factory=list)
    task_type_mappings: Dict[str, List[str]] = field(default_factory=dict)
    quality_thresholds: Dict[str, float] = field(default_factory=dict)
    default_patience: Dict[str, int] = field(default_factory=dict)
    exploration_bonus: float = 0.15
    extensions: Dict[str, Any] = field(default_factory=dict)

    def get_tools_for_task(self, task_type: str) -> List[str]:
        """Get recommended tools for a task type.

        Args:
            task_type: Type of task (e.g., "debugging", "deployment")

        Returns:
            List of recommended tool names, or empty list if not found
        """
        return self.task_type_mappings.get(task_type.lower(), [])

    def get_quality_threshold(self, task_type: str) -> float:
        """Get quality threshold for a task type.

        Args:
            task_type: Type of task

        Returns:
            Quality threshold (0.0-1.0), or 0.80 as default
        """
        return self.quality_thresholds.get(task_type.lower(), 0.80)

    def get_patience(self, provider: str) -> int:
        """Get continuation patience for a provider.

        Args:
            provider: Provider name (e.g., "anthropic", "ollama")

        Returns:
            Number of retry attempts, or 4 as default
        """
        return self.default_patience.get(provider.lower(), 4)

    def is_learner_active(self, learner: LearnerType) -> bool:
        """Check if a learner is active.

        Args:
            learner: Learner type to check

        Returns:
            True if learner is in active_learners list
        """
        return learner in self.active_learners

    def get_extension(self, key: str, default: Any = None) -> Any:
        """Get a vertical-specific extension value.

        Args:
            key: Extension key (e.g., "preferred_providers_by_task")
            default: Default value if not found

        Returns:
            Extension value or default
        """
        return self.extensions.get(key, default)


class RLConfigRegistry:
    """Registry for RL configurations across all verticals.

    Singleton pattern with thread-safe lazy loading. Uses the
    Universal Registry for caching configurations.

    Usage:
        registry = RLConfigRegistry.get_instance()
        config = registry.get_rl_config("coding")
        tools = config.get_tools_for_task("debugging")
    """

    _instance: Optional["RLConfigRegistry"] = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls) -> "RLConfigRegistry":
        """Create or return singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the registry (only once)."""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config_dir = Path(__file__).parent.parent.parent / "config" / "rl"
                    self._cache: Dict[str, RLConfig] = {}
                    self._initialized = True
                    logger.debug(
                        f"RLConfigRegistry initialized with config dir: {self._config_dir}"
                    )

    @classmethod
    def get_instance(cls) -> "RLConfigRegistry":
        """Get the singleton registry instance."""
        return cls()

    def get_rl_config(self, vertical: str) -> RLConfig:
        """Get RL configuration for a vertical.

        Loads from YAML file with fallback to defaults.

        Args:
            vertical: Vertical name (e.g., "coding", "research")

        Returns:
            RLConfig instance with vertical configuration

        Raises:
            ValueError: If vertical is empty or invalid
        """
        if not vertical:
            raise ValueError("Vertical name cannot be empty")

        vertical = vertical.lower()

        # Check cache
        if vertical in self._cache:
            return self._cache[vertical]

        # Load from YAML
        config = self._load_from_yaml(vertical)
        if config is None:
            # Fallback to defaults
            config = self._get_default_config(vertical)

        self._cache[vertical] = config
        return config

    def _load_from_yaml(self, vertical: str) -> Optional[RLConfig]:
        """Load configuration from YAML file.

        Args:
            vertical: Vertical name

        Returns:
            RLConfig or None if file not found
        """
        yaml_file = self._config_dir / f"{vertical}_rl.yaml"

        if not yaml_file.exists():
            logger.debug(f"No RL config file found for {vertical}: {yaml_file}")
            return None

        try:
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)

            if not data or data.get("vertical_name") != vertical:
                logger.warning(f"Invalid RL config file for {vertical}")
                return None

            # Parse learner types
            active_learners = []
            for learner_name in data.get("active_learners", []):
                try:
                    active_learners.append(LearnerType(learner_name))
                except ValueError:
                    logger.warning(f"Unknown learner type: {learner_name}")

            # Extract extensions (non-standard fields)
            standard_fields = {
                "vertical_name",
                "active_learners",
                "task_type_mappings",
                "quality_thresholds",
                "default_patience",
                "exploration_bonus",
            }
            extensions = {k: v for k, v in data.items() if k not in standard_fields}

            config = RLConfig(
                vertical_name=vertical,
                active_learners=active_learners,
                task_type_mappings=data.get("task_type_mappings", {}),
                quality_thresholds=data.get("quality_thresholds", {}),
                default_patience=data.get("default_patience", {}),
                exploration_bonus=data.get("exploration_bonus", 0.15),
                extensions=extensions,
            )

            logger.debug(f"Loaded RL config for {vertical} from {yaml_file}")
            return config

        except Exception as e:
            logger.error(f"Failed to load RL config for {vertical}: {e}")
            return None

    def _get_default_config(self, vertical: str) -> RLConfig:
        """Get default configuration when YAML not found.

        Args:
            vertical: Vertical name

        Returns:
            RLConfig with sensible defaults
        """
        # Default learners for most verticals
        default_learners = [
            LearnerType.TOOL_SELECTOR,
            LearnerType.CONTINUATION_PATIENCE,
            LearnerType.GROUNDING_THRESHOLD,
        ]

        # Default patience map
        default_patience = {
            "anthropic": 4,
            "openai": 4,
            "google": 4,
            "deepseek": 5,
            "ollama": 6,
        }

        # Default task mappings (minimal)
        default_mappings = {
            "general": ["read", "write"],
        }

        # Default quality thresholds
        default_thresholds = {
            "general": 0.80,
        }

        return RLConfig(
            vertical_name=vertical,
            active_learners=default_learners,
            task_type_mappings=default_mappings,
            quality_thresholds=default_thresholds,
            default_patience=default_patience,
            exploration_bonus=0.15,
        )

    def list_verticals(self) -> List[str]:
        """List all available vertical configurations.

        Returns:
            List of vertical names with available configs
        """
        if not self._config_dir.exists():
            return []

        verticals = []
        for yaml_file in self._config_dir.glob("*_rl.yaml"):
            vertical = yaml_file.stem.replace("_rl", "")
            verticals.append(vertical)

        return sorted(verticals)

    def invalidate(self, vertical: Optional[str] = None) -> None:
        """Invalidate cached configuration(s).

        Args:
            vertical: Specific vertical to invalidate, or None for all
        """
        if vertical:
            self._cache.pop(vertical.lower(), None)
        else:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dict with cache size, available verticals, etc.
        """
        return {
            "cache_size": len(self._cache),
            "cached_verticals": list(self._cache.keys()),
            "available_verticals": self.list_verticals(),
            "config_dir": str(self._config_dir),
        }


__all__ = [
    "RLConfig",
    "RLConfigRegistry",
]
