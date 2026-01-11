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

"""Base RL configuration for vertical-agnostic RL defaults.

Provides common RL configuration that can be inherited by any vertical.
Reduces code duplication and ensures consistent RL behavior across verticals.

Design Philosophy:
- Framework provides BASE DEFAULTS (active learners, patience maps)
- Verticals provide SPECIFIC DATA (task mappings, quality thresholds)
- Eliminates duplicate code across vertical RL configs

Usage:
    from victor.framework.rl.config import BaseRLConfig
    from victor.framework.rl import LearnerType

    class CodingRLConfig(BaseRLConfig):
        task_type_mappings: Dict[str, List[str]] = field(
            default_factory=lambda: {"debug": ["read", "grep"]}
        )
        quality_thresholds: Dict[str, float] = field(
            default_factory=lambda: {"debug": 0.85}
        )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

# Import LearnerType from enum directly to avoid circular import
from victor.framework.rl import LearnerType


# Default patience map - shared across all verticals
# These are reasonable defaults for continuation learning
DEFAULT_PATIENCE_MAP: Dict[str, int] = {
    "anthropic": 4,
    "openai": 4,
    "google": 4,
    "deepseek": 5,  # More patient with DeepSeek
    "xai": 4,
    "moonshot": 4,
    "kimi": 4,
    "ollama": 6,  # More patient with local models
    "lmstudio": 6,
    "vllm": 6,
}

# Default active learners - shared across all verticals
DEFAULT_ACTIVE_LEARNERS: List[LearnerType] = [
    LearnerType.TOOL_SELECTOR,
    LearnerType.CONTINUATION_PATIENCE,
    LearnerType.GROUNDING_THRESHOLD,
]


@dataclass
class BaseRLConfig:
    """Base RL configuration with shared defaults.

    Provides common RL configuration that applies to all verticals:
    - Default active learners
    - Default patience by provider
    - Common methods for accessing config

    Verticals should inherit and extend with:
    - task_type_mappings: Vertical-specific tool recommendations
    - quality_thresholds: Vertical-specific quality thresholds
    - Any vertical-specific extras

    Attributes:
        active_learners: Learners to activate (inherited from defaults)
        task_type_mappings: Maps task types to recommended tools (vertical provides)
        quality_thresholds: Quality thresholds by task type (vertical provides)
        default_patience: Default continuation patience by provider (shared)
    """

    # Learners to activate - shared defaults
    active_learners: List[LearnerType] = field(
        default_factory=lambda: list(DEFAULT_ACTIVE_LEARNERS)
    )

    # Task type to tool mappings - must be provided by subclass
    task_type_mappings: Dict[str, List[str]] = field(default_factory=dict)

    # Quality thresholds by task type - must be provided by subclass
    quality_thresholds: Dict[str, float] = field(default_factory=dict)

    # Continuation patience by provider - shared defaults
    default_patience: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_PATIENCE_MAP))

    # Exploration bonus for tool selection (can be overridden)
    exploration_bonus: float = 0.15

    def get_tools_for_task(self, task_type: str) -> List[str]:
        """Get recommended tools for a task type.

        Args:
            task_type: Type of task (e.g., "debug", "refactoring")

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

    def get_rl_config(self) -> Dict[str, Any]:
        """Return RL configuration as dictionary (protocol compliance).

        Implements RLConfigProviderProtocol.get_rl_config() to enable
        integration with the vertical framework.

        Returns:
            Dict with RL configuration including:
            - active_learners: List of learner type values
            - task_type_mappings: Map task types to recommended tools
            - quality_thresholds: Task-specific quality thresholds
            - default_patience: Provider-specific patience settings
        """
        return {
            "active_learners": [learner.value for learner in self.active_learners],
            "task_type_mappings": self.task_type_mappings,
            "quality_thresholds": self.quality_thresholds,
            "default_patience": self.default_patience,
        }

    def __repr__(self) -> str:
        """String representation showing config size."""
        return (
            f"{self.__class__.__name__}("
            f"learners={len(self.active_learners)}, "
            f"task_types={len(self.task_type_mappings)})"
        )


__all__ = [
    "BaseRLConfig",
    "DEFAULT_PATIENCE_MAP",
    "DEFAULT_ACTIVE_LEARNERS",
]
