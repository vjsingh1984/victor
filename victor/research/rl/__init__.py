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

"""RL integration for Research vertical.

Uses canonical tool names from ToolNames to ensure consistent naming
across RL Q-values, workflow patterns, and vertical configurations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from victor.framework.rl import LearnerType
from victor.framework.tool_naming import ToolNames


@dataclass
class ResearchRLConfig:
    """RL configuration for Research vertical.

    Configures which RL learners to use and their parameters
    for research tasks.
    """

    active_learners: List[LearnerType] = field(
        default_factory=lambda: [
            LearnerType.TOOL_SELECTOR,
            LearnerType.CONTINUATION_PATIENCE,
            LearnerType.GROUNDING_THRESHOLD,
        ]
    )

    # Uses canonical ToolNames constants for consistency
    task_type_mappings: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "research": [
                ToolNames.WEB_SEARCH,
                ToolNames.WEB_FETCH,
                ToolNames.READ,
                ToolNames.WRITE,
            ],
            "fact_check": [ToolNames.WEB_SEARCH, ToolNames.WEB_FETCH, ToolNames.GREP],
            "literature": [ToolNames.WEB_SEARCH, ToolNames.WEB_FETCH, ToolNames.READ],
            "competitive": [ToolNames.WEB_SEARCH, ToolNames.WEB_FETCH, ToolNames.WRITE],
            "synthesis": [ToolNames.READ, ToolNames.WRITE, ToolNames.EDIT],
            "exploration": [
                ToolNames.WEB_SEARCH,
                ToolNames.READ,
                ToolNames.GREP,
                ToolNames.OVERVIEW,
            ],
        }
    )

    quality_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "research": 0.85,  # High bar for research accuracy
            "fact_check": 0.90,  # Very high bar for fact verification
            "literature": 0.85,
            "competitive": 0.80,
            "synthesis": 0.80,
            "exploration": 0.75,
        }
    )

    default_patience: Dict[str, int] = field(
        default_factory=lambda: {
            "anthropic": 4,  # Higher patience for research tasks
            "openai": 4,
            "google": 4,  # Google often used for research
            "ollama": 6,
        }
    )

    # Research-specific: prefer providers with web access
    preferred_providers_by_task: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "research": ["anthropic", "openai", "google"],
            "fact_check": ["anthropic", "openai"],
            "literature": ["anthropic", "openai", "google"],
        }
    )

    def get_tools_for_task(self, task_type: str) -> List[str]:
        """Get recommended tools for a task type."""
        return self.task_type_mappings.get(task_type.lower(), [])

    def get_quality_threshold(self, task_type: str) -> float:
        """Get quality threshold for a task type."""
        return self.quality_thresholds.get(task_type.lower(), 0.80)

    def get_patience(self, provider: str) -> int:
        """Get patience setting for a provider."""
        return self.default_patience.get(provider.lower(), 4)

    def get_preferred_providers(self, task_type: str) -> List[str]:
        """Get preferred providers for a task type."""
        return self.preferred_providers_by_task.get(
            task_type.lower(),
            ["anthropic", "openai", "google"],
        )

    def is_learner_active(self, learner: LearnerType) -> bool:
        """Check if a learner is active."""
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
        return (
            f"ResearchRLConfig(learners={len(self.active_learners)}, "
            f"task_types={len(self.task_type_mappings)})"
        )


class ResearchRLHooks:
    """RL recording hooks for Research middleware.

    Provides hooks for recording RL training data during research tasks.
    """

    def __init__(self, config: Optional[ResearchRLConfig] = None):
        self._config = config or ResearchRLConfig()

    @property
    def config(self) -> ResearchRLConfig:
        return self._config

    def get_tool_recommendation(
        self,
        task_type: str,
        available_tools: Optional[List[str]] = None,
    ) -> List[str]:
        """Get tool recommendations for a task type."""
        config_tools = self._config.get_tools_for_task(task_type)
        if available_tools:
            return [t for t in config_tools if t in available_tools]
        return config_tools

    def get_patience_recommendation(self, provider: str, model: str) -> int:
        """Get patience recommendation for provider/model."""
        return self._config.get_patience(provider)

    def get_quality_threshold(self, task_type: str) -> float:
        """Get quality threshold for task type."""
        return self._config.get_quality_threshold(task_type)

    def get_preferred_providers(self, task_type: str) -> List[str]:
        """Get preferred providers for task type."""
        return self._config.get_preferred_providers(task_type)

    def should_verify_sources(self, task_type: str) -> bool:
        """Check if source verification is recommended."""
        return task_type.lower() in {"fact_check", "research", "literature"}

    def get_min_sources(self, task_type: str) -> int:
        """Get minimum number of sources recommended."""
        minimums = {
            "fact_check": 3,  # Need multiple sources for verification
            "research": 2,
            "literature": 5,  # Academic requires more sources
            "competitive": 2,
        }
        return minimums.get(task_type.lower(), 2)

    def __repr__(self) -> str:
        return f"ResearchRLHooks(config={self._config})"


# Module-level singletons
_default_config: ResearchRLConfig | None = None
_hooks_instance: ResearchRLHooks | None = None


def get_default_config() -> ResearchRLConfig:
    """Get default research RL configuration."""
    global _default_config
    if _default_config is None:
        _default_config = ResearchRLConfig()
    return _default_config


def get_research_rl_hooks() -> ResearchRLHooks:
    """Get research RL hooks instance."""
    global _hooks_instance
    if _hooks_instance is None:
        _hooks_instance = ResearchRLHooks()
    return _hooks_instance


__all__ = [
    "ResearchRLConfig",
    "ResearchRLHooks",
    "get_default_config",
    "get_research_rl_hooks",
]
