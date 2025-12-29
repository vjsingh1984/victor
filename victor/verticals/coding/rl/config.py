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

"""RL configuration for coding vertical.

Provides coding-specific configuration for the RL system including
learner activations, task type mappings, and quality thresholds.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set

from victor.framework.rl import LearnerType


@dataclass
class CodingRLConfig:
    """RL configuration for coding vertical.

    This configuration customizes RL behavior for software development
    tasks, including which learners are active, tool recommendations
    for different task types, and quality thresholds.

    Attributes:
        active_learners: Learners to activate for coding tasks
        task_type_mappings: Maps task types to recommended tools
        quality_thresholds: Quality thresholds by task type
        default_patience: Default continuation patience by provider
        exploration_bonus: Bonus weight for less-used tools

    Example:
        config = CodingRLConfig()
        config.get_tools_for_task("debugging")
        # Returns: ["read", "grep", "shell", "run_tests", "git_log"]
    """

    # Learners to activate
    active_learners: List[LearnerType] = field(
        default_factory=lambda: [
            LearnerType.TOOL_SELECTOR,
            LearnerType.CONTINUATION_PATIENCE,
            LearnerType.GROUNDING_THRESHOLD,
            LearnerType.MODE_TRANSITION,
            LearnerType.QUALITY_WEIGHTS,
        ]
    )

    # Task type to tool mappings for tool selection learning
    task_type_mappings: Dict[str, List[str]] = field(
        default_factory=lambda: {
            # Analysis tasks
            "refactoring": [
                "refactor",
                "rename_symbol",
                "extract_function",
                "edit_files",
                "read_file",
            ],
            "debugging": [
                "read_file",
                "grep",
                "bash",
                "run_tests",
                "git_log",
                "symbols",
                "references",
            ],
            "exploration": [
                "read_file",
                "grep",
                "code_search",
                "semantic_code_search",
                "project_overview",
                "symbols",
                "list_directory",
            ],
            # Implementation tasks
            "feature": [
                "read_file",
                "write_file",
                "edit_files",
                "bash",
                "git_status",
                "git_diff",
            ],
            "implementation": [
                "read_file",
                "write_file",
                "edit_files",
                "bash",
                "run_tests",
            ],
            "testing": [
                "run_tests",
                "test_file",
                "bash",
                "read_file",
                "write_file",
            ],
            # Documentation tasks
            "documentation": [
                "read_file",
                "write_file",
                "edit_files",
                "grep",
            ],
            # Review tasks
            "review": [
                "read_file",
                "grep",
                "git_diff",
                "git_log",
                "references",
            ],
        }
    )

    # Quality thresholds by task type (higher = stricter)
    quality_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "refactoring": 0.90,  # High bar for refactoring
            "debugging": 0.85,
            "feature": 0.80,
            "implementation": 0.80,
            "exploration": 0.70,  # Lower bar for exploration
            "testing": 0.85,
            "documentation": 0.75,
            "review": 0.80,
        }
    )

    # Continuation patience by provider (retries before giving up)
    default_patience: Dict[str, int] = field(
        default_factory=lambda: {
            "anthropic": 3,
            "openai": 3,
            "google": 3,
            "deepseek": 5,  # More patient with DeepSeek
            "ollama": 7,  # Most patient with local models
            "lmstudio": 7,
            "vllm": 7,
        }
    )

    # Exploration bonus for tool selection (encourage trying different tools)
    exploration_bonus: float = 0.15

    # Tools that should never be recommended together (conflicting)
    conflicting_tools: Dict[str, Set[str]] = field(
        default_factory=lambda: {
            "write_file": {"edit_files"},  # Use one or the other
            "edit_files": {"write_file"},
        }
    )

    def get_tools_for_task(self, task_type: str) -> List[str]:
        """Get recommended tools for a task type.

        Args:
            task_type: Type of task

        Returns:
            List of recommended tool names
        """
        return self.task_type_mappings.get(task_type.lower(), [])

    def get_quality_threshold(self, task_type: str) -> float:
        """Get quality threshold for a task type.

        Args:
            task_type: Type of task

        Returns:
            Quality threshold (0.0-1.0)
        """
        return self.quality_thresholds.get(task_type.lower(), 0.80)

    def get_patience(self, provider: str) -> int:
        """Get continuation patience for a provider.

        Args:
            provider: Provider name

        Returns:
            Number of retry attempts
        """
        return self.default_patience.get(provider.lower(), 3)

    def is_learner_active(self, learner: LearnerType) -> bool:
        """Check if a learner is active.

        Args:
            learner: Learner type to check

        Returns:
            True if learner is active
        """
        return learner in self.active_learners

    def __repr__(self) -> str:
        return (
            f"CodingRLConfig(learners={len(self.active_learners)}, "
            f"task_types={len(self.task_type_mappings)})"
        )


# Default singleton instance
_default_config: CodingRLConfig | None = None


def get_default_config() -> CodingRLConfig:
    """Get the default coding RL configuration.

    Returns:
        Default CodingRLConfig instance
    """
    global _default_config
    if _default_config is None:
        _default_config = CodingRLConfig()
    return _default_config


__all__ = [
    "CodingRLConfig",
    "get_default_config",
]
