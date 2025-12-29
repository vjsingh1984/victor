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

"""RL integration for DevOps vertical."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from victor.framework.rl import LearnerType


@dataclass
class DevOpsRLConfig:
    """RL configuration for DevOps vertical."""

    active_learners: List[LearnerType] = field(
        default_factory=lambda: [
            LearnerType.TOOL_SELECTOR,
            LearnerType.CONTINUATION_PATIENCE,
            LearnerType.GROUNDING_THRESHOLD,
        ]
    )

    task_type_mappings: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "deployment": ["bash", "docker", "git_status", "read_file", "edit_files"],
            "containerization": ["docker", "bash", "read_file", "write_file"],
            "monitoring": ["bash", "read_file", "write_file", "grep"],
            "configuration": ["read_file", "write_file", "edit_files", "grep"],
            "troubleshooting": ["bash", "read_file", "grep", "docker"],
        }
    )

    quality_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "deployment": 0.90,  # High bar for deployments
            "containerization": 0.85,
            "monitoring": 0.80,
            "configuration": 0.85,
            "troubleshooting": 0.80,
        }
    )

    default_patience: Dict[str, int] = field(
        default_factory=lambda: {
            "anthropic": 3,
            "openai": 3,
            "ollama": 5,
        }
    )

    def get_tools_for_task(self, task_type: str) -> List[str]:
        return self.task_type_mappings.get(task_type.lower(), [])

    def get_quality_threshold(self, task_type: str) -> float:
        return self.quality_thresholds.get(task_type.lower(), 0.85)

    def get_patience(self, provider: str) -> int:
        return self.default_patience.get(provider.lower(), 3)

    def is_learner_active(self, learner: LearnerType) -> bool:
        return learner in self.active_learners

    def __repr__(self) -> str:
        return (
            f"DevOpsRLConfig(learners={len(self.active_learners)}, "
            f"task_types={len(self.task_type_mappings)})"
        )


class DevOpsRLHooks:
    """RL recording hooks for DevOps middleware."""

    def __init__(self, config: Optional[DevOpsRLConfig] = None):
        self._config = config or DevOpsRLConfig()

    @property
    def config(self) -> DevOpsRLConfig:
        return self._config

    def get_tool_recommendation(
        self,
        task_type: str,
        available_tools: Optional[List[str]] = None,
    ) -> List[str]:
        config_tools = self._config.get_tools_for_task(task_type)
        if available_tools:
            return [t for t in config_tools if t in available_tools]
        return config_tools

    def get_patience_recommendation(self, provider: str, model: str) -> int:
        return self._config.get_patience(provider)

    def get_quality_threshold(self, task_type: str) -> float:
        return self._config.get_quality_threshold(task_type)

    def __repr__(self) -> str:
        return f"DevOpsRLHooks(config={self._config})"


_default_config: DevOpsRLConfig | None = None
_hooks_instance: DevOpsRLHooks | None = None


def get_default_config() -> DevOpsRLConfig:
    global _default_config
    if _default_config is None:
        _default_config = DevOpsRLConfig()
    return _default_config


def get_devops_rl_hooks() -> DevOpsRLHooks:
    global _hooks_instance
    if _hooks_instance is None:
        _hooks_instance = DevOpsRLHooks()
    return _hooks_instance


__all__ = [
    "DevOpsRLConfig",
    "DevOpsRLHooks",
    "get_default_config",
    "get_devops_rl_hooks",
]
