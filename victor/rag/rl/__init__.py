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

"""RL integration for RAG vertical.

Uses canonical tool names from ToolNames to ensure consistent naming
across RL Q-values, workflow patterns, and vertical configurations.

This module provides:
- RAGRLConfig: Configuration for active learners, task type mappings, and quality thresholds
- RAGRLHooks: Hooks for tracking search quality and synthesis quality
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor.framework.rl import LearnerType
from victor.framework.tool_naming import ToolNames


@dataclass
class RAGRLConfig:
    """RL configuration for RAG vertical.

    Configures which learners are active and how they should behave
    for RAG-specific tasks like search, synthesis, and ingestion.
    """

    active_learners: List[LearnerType] = field(
        default_factory=lambda: [
            LearnerType.TOOL_SELECTOR,
            LearnerType.GROUNDING_THRESHOLD,
            LearnerType.QUALITY_WEIGHTS,
        ]
    )

    # Uses canonical ToolNames constants and RAG-specific tool names
    task_type_mappings: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "search": [
                "rag_search",
                "rag_query",
                ToolNames.READ,
            ],
            "ingest": [
                "rag_ingest",
                ToolNames.READ,
                ToolNames.LS,
                ToolNames.WEB_FETCH,
            ],
            "synthesis": [
                "rag_query",
                "rag_search",
            ],
            "management": [
                "rag_list",
                "rag_delete",
                "rag_stats",
            ],
            "exploration": [
                "rag_search",
                "rag_list",
                "rag_stats",
                ToolNames.READ,
            ],
        }
    )

    quality_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "search": 0.80,  # Search relevance threshold
            "synthesis": 0.85,  # Answer quality threshold (higher for factual)
            "ingest": 0.75,  # Ingestion success threshold
            "management": 0.70,  # Management operation threshold
            "exploration": 0.75,  # Exploration quality threshold
        }
    )

    default_patience: Dict[str, int] = field(
        default_factory=lambda: {
            "anthropic": 3,
            "openai": 3,
            "ollama": 5,
            "google": 3,
        }
    )

    def get_tools_for_task(self, task_type: str) -> List[str]:
        """Get recommended tools for a task type.

        Args:
            task_type: Type of task (search, ingest, synthesis, etc.)

        Returns:
            List of tool names for the task type
        """
        return self.task_type_mappings.get(task_type.lower(), [])

    def get_quality_threshold(self, task_type: str) -> float:
        """Get quality threshold for a task type.

        Args:
            task_type: Type of task

        Returns:
            Quality threshold (0.0-1.0), defaults to 0.80
        """
        return self.quality_thresholds.get(task_type.lower(), 0.80)

    def get_patience(self, provider: str) -> int:
        """Get continuation patience for a provider.

        Args:
            provider: LLM provider name

        Returns:
            Patience value (number of continuation attempts)
        """
        return self.default_patience.get(provider.lower(), 3)

    def is_learner_active(self, learner: LearnerType) -> bool:
        """Check if a learner is active.

        Args:
            learner: Learner type to check

        Returns:
            True if learner is in active_learners
        """
        return learner in self.active_learners

    def __repr__(self) -> str:
        return (
            f"RAGRLConfig(learners={len(self.active_learners)}, "
            f"task_types={len(self.task_type_mappings)})"
        )


class RAGRLHooks:
    """RL recording hooks for RAG middleware.

    Provides methods to get tool recommendations, patience settings,
    and quality thresholds based on task context.
    """

    def __init__(self, config: Optional[RAGRLConfig] = None):
        """Initialize with optional custom config.

        Args:
            config: Custom RAGRLConfig, or None to use default
        """
        self._config = config or RAGRLConfig()

    @property
    def config(self) -> RAGRLConfig:
        """Get the RL configuration."""
        return self._config

    def get_tool_recommendation(
        self,
        task_type: str,
        available_tools: Optional[List[str]] = None,
    ) -> List[str]:
        """Get tool recommendations for a task type.

        Args:
            task_type: Type of task
            available_tools: Optional filter for available tools

        Returns:
            List of recommended tool names
        """
        config_tools = self._config.get_tools_for_task(task_type)
        if available_tools:
            return [t for t in config_tools if t in available_tools]
        return config_tools

    def get_patience_recommendation(self, provider: str, model: str) -> int:
        """Get patience recommendation for provider/model.

        Args:
            provider: LLM provider name
            model: Model name (currently unused, for future model-specific tuning)

        Returns:
            Recommended patience value
        """
        return self._config.get_patience(provider)

    def get_quality_threshold(self, task_type: str) -> float:
        """Get quality threshold for task type.

        Args:
            task_type: Type of task

        Returns:
            Quality threshold (0.0-1.0)
        """
        return self._config.get_quality_threshold(task_type)

    def __repr__(self) -> str:
        return f"RAGRLHooks(config={self._config})"


# Module-level singletons
_default_config: RAGRLConfig | None = None
_hooks_instance: RAGRLHooks | None = None


def get_default_config() -> RAGRLConfig:
    """Get the default RAG RL configuration singleton.

    Returns:
        RAGRLConfig singleton instance
    """
    global _default_config
    if _default_config is None:
        _default_config = RAGRLConfig()
    return _default_config


def get_rag_rl_hooks() -> RAGRLHooks:
    """Get the RAG RL hooks singleton.

    Returns:
        RAGRLHooks singleton instance
    """
    global _hooks_instance
    if _hooks_instance is None:
        _hooks_instance = RAGRLHooks()
    return _hooks_instance


__all__ = [
    "RAGRLConfig",
    "RAGRLHooks",
    "get_default_config",
    "get_rag_rl_hooks",
]
