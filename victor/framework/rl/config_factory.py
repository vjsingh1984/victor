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

"""RL Config Factory for vertical-specific configurations (Phase 9.1).

This module consolidates RL configuration code that was duplicated across
5 verticals (~400 LOC saved). It provides:

1. RLConfigFactory: Factory for creating vertical-specific RL configs
2. GenericRLHooks: Consolidated hooks class replacing vertical-specific hooks
3. YAML-first configuration with code fallbacks

Design Philosophy:
- YAML provides data, code provides behavior
- Factory pattern for consistent config creation
- Single source of truth for RL configurations
- Backward compatible with existing vertical configs

Usage:
    from victor.framework.rl.config_factory import RLConfigFactory, GenericRLHooks

    # Get config for vertical
    config = RLConfigFactory.create("coding")

    # Create hooks with config
    hooks = GenericRLHooks(config)

    # Get recommendations
    tools = hooks.get_tool_recommendation("debugging")
    patience = hooks.get_patience_recommendation("anthropic", "claude-3-opus")
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from victor.framework.rl import LearnerType
from victor.framework.rl.config import (
    BaseRLConfig,
    DEFAULT_ACTIVE_LEARNERS,
    DEFAULT_PATIENCE_MAP,
)

logger = logging.getLogger(__name__)


# Default config directory
_CONFIG_DIR = Path(__file__).parent.parent.parent / "config" / "rl"


def _learner_type_from_str(name: str) -> Optional[LearnerType]:
    """Convert string to LearnerType enum."""
    try:
        return LearnerType(name)
    except ValueError:
        # Try uppercase conversion
        try:
            return LearnerType[name.upper()]
        except KeyError:
            logger.warning(f"Unknown learner type: {name}")
            return None


@dataclass
class VerticalRLConfig(BaseRLConfig):
    """Extended RL config with vertical-specific extensions.

    This dataclass extends BaseRLConfig with additional fields that
    some verticals may need (e.g., preferred_output_length for DataAnalysis,
    conflicting_tools for Coding).
    """

    # Coding-specific: tools that conflict
    conflicting_tools: dict[str, set] = field(default_factory=dict)  # type: ignore[type-arg]

    # DataAnalysis-specific: output length preferences
    preferred_output_length: dict[str, str] = field(default_factory=dict)

    # Research-specific: provider preferences
    preferred_providers_by_task: dict[str, list[str]] = field(default_factory=dict)

    def get_preferred_output_length(self, task_type: str) -> str:
        """Get preferred output length for task type (DataAnalysis)."""
        return self.preferred_output_length.get(task_type.lower(), "medium")

    def get_preferred_providers(self, task_type: str) -> list[str]:
        """Get preferred providers for task type (Research)."""
        return self.preferred_providers_by_task.get(
            task_type.lower(),
            ["anthropic", "openai", "google"],
        )


class RLConfigFactory:
    """Factory for creating vertical-specific RL configurations.

    This factory consolidates RL config creation that was previously
    duplicated across 5 vertical modules. It supports:

    1. Built-in configs for standard verticals (coding, devops, rag, etc.)
    2. YAML-based configs for customization
    3. Caching for performance

    Thread-safe singleton pattern with per-vertical caching.

    Example:
        # Create coding config
        config = RLConfigFactory.create("coding")

        # Create from custom YAML
        config = RLConfigFactory.create_from_yaml(Path("my_rl.yaml"), "custom")
    """

    _cache: dict[str, BaseRLConfig] = {}
    _lock = threading.Lock()
    _configs_loaded: bool = False
    _yaml_configs: dict[str, dict[str, Any]] = {}

    @classmethod
    def create(cls, vertical: str) -> BaseRLConfig:
        """Create or retrieve cached RL config for vertical.

        Args:
            vertical: Vertical name (coding, devops, rag, dataanalysis, research)

        Returns:
            BaseRLConfig or subclass configured for the vertical
        """
        vertical_lower = vertical.lower()

        with cls._lock:
            if vertical_lower in cls._cache:
                return cls._cache[vertical_lower]

            # Load YAML configs if not done
            if not cls._configs_loaded:
                cls._load_yaml_configs()

            # Create config
            config = cls._create_config(vertical_lower)
            cls._cache[vertical_lower] = config
            return config

    @classmethod
    def create_from_yaml(cls, yaml_path: Path, vertical: str) -> BaseRLConfig:
        """Create config from a specific YAML file.

        Args:
            yaml_path: Path to YAML config file
            vertical: Vertical name to load

        Returns:
            BaseRLConfig configured from YAML
        """
        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load YAML from {yaml_path}: {e}")
            return BaseRLConfig()

        verticals = data.get("verticals", {})
        vertical_data = verticals.get(vertical, {})

        return cls._config_from_dict(vertical_data)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the config cache."""
        with cls._lock:
            cls._cache.clear()
            cls._configs_loaded = False
            cls._yaml_configs.clear()

    @classmethod
    def _load_yaml_configs(cls) -> None:
        """Load all YAML configs from config directory."""
        try:
            # Try unified config first
            unified_path = _CONFIG_DIR / "unified_rl_config.yaml"
            if unified_path.exists():
                with open(unified_path, "r") as f:
                    data = yaml.safe_load(f) or {}
                    cls._yaml_configs = data.get("verticals", {})
                    logger.debug(f"Loaded unified RL config from {unified_path}")
            else:
                # Fall back to individual files
                for yaml_file in _CONFIG_DIR.glob("*_rl.yaml"):
                    try:
                        with open(yaml_file, "r") as f:
                            data = yaml.safe_load(f) or {}
                            vertical_name = data.get(
                                "vertical_name", yaml_file.stem.replace("_rl", "")
                            )
                            cls._yaml_configs[vertical_name] = data
                    except Exception as e:
                        logger.warning(f"Failed to load {yaml_file}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load YAML configs: {e}")

        cls._configs_loaded = True

    @classmethod
    def _create_config(cls, vertical: str) -> BaseRLConfig:
        """Create config for a specific vertical."""
        # Check YAML configs first
        if vertical in cls._yaml_configs:
            return cls._config_from_dict(cls._yaml_configs[vertical])

        # Fall back to built-in configs
        return cls._builtin_config(vertical)

    @classmethod
    def _config_from_dict(cls, data: dict[str, Any]) -> BaseRLConfig:
        """Create config from dictionary data."""
        if not data:
            return BaseRLConfig()

        # Parse active learners
        active_learners: list[LearnerType] = []
        for learner_str in data.get("active_learners", []):
            learner = _learner_type_from_str(learner_str)
            if learner:
                active_learners.append(learner)

        if not active_learners:
            active_learners = list(DEFAULT_ACTIVE_LEARNERS)

        # Parse task type mappings
        task_type_mappings = data.get("task_type_mappings", {})

        # Parse quality thresholds
        quality_thresholds = data.get("quality_thresholds", {})

        # Parse patience map
        default_patience = data.get("default_patience", dict(DEFAULT_PATIENCE_MAP))

        # Parse exploration bonus
        exploration_bonus = data.get("exploration_bonus", 0.15)

        # Check for extended fields
        conflicting_tools = {}
        for tool, conflicts in data.get("conflicting_tools", {}).items():
            conflicting_tools[tool] = set(conflicts)

        preferred_output_length = data.get("preferred_output_length", {})
        preferred_providers = data.get("preferred_providers_by_task", {})

        # Use extended config if any extensions present
        if conflicting_tools or preferred_output_length or preferred_providers:
            return VerticalRLConfig(
                active_learners=active_learners,
                task_type_mappings=task_type_mappings,
                quality_thresholds=quality_thresholds,
                default_patience=default_patience,
                exploration_bonus=exploration_bonus,
                conflicting_tools=conflicting_tools,
                preferred_output_length=preferred_output_length,
                preferred_providers_by_task=preferred_providers,
            )

        return BaseRLConfig(
            active_learners=active_learners,
            task_type_mappings=task_type_mappings,
            quality_thresholds=quality_thresholds,
            default_patience=default_patience,
            exploration_bonus=exploration_bonus,
        )

    @classmethod
    def _builtin_config(cls, vertical: str) -> BaseRLConfig:
        """Get built-in config for known verticals.

        These are fallbacks when YAML is not available.
        """
        if vertical == "coding":
            return cls._coding_config()
        elif vertical == "devops":
            return cls._devops_config()
        elif vertical == "rag":
            return cls._rag_config()
        elif vertical == "dataanalysis":
            return cls._dataanalysis_config()
        elif vertical == "research":
            return cls._research_config()
        else:
            # Return base config for unknown verticals
            return BaseRLConfig()

    @classmethod
    def _coding_config(cls) -> VerticalRLConfig:
        """Built-in coding vertical config."""
        return VerticalRLConfig(
            active_learners=[
                LearnerType.TOOL_SELECTOR,
                LearnerType.CONTINUATION_PATIENCE,
                LearnerType.GROUNDING_THRESHOLD,
                LearnerType.MODE_TRANSITION,
                LearnerType.QUALITY_WEIGHTS,
            ],
            task_type_mappings={
                "refactoring": ["rename", "extract", "edit", "read"],
                "debugging": ["read", "grep", "shell", "test", "git", "symbol", "refs"],
                "exploration": ["read", "grep", "code_search", "overview", "symbol", "ls"],
                "feature": ["read", "write", "edit", "shell", "git"],
                "implementation": ["read", "write", "edit", "shell", "test"],
                "testing": ["test", "shell", "read", "write"],
                "documentation": ["read", "write", "edit", "grep"],
                "review": ["read", "grep", "git", "refs"],
            },
            quality_thresholds={
                "refactoring": 0.90,
                "debugging": 0.85,
                "feature": 0.80,
                "implementation": 0.80,
                "exploration": 0.70,
                "testing": 0.85,
                "documentation": 0.75,
                "review": 0.80,
            },
            default_patience={
                "anthropic": 3,
                "openai": 3,
                "google": 3,
                "deepseek": 5,
                "ollama": 7,
                "lmstudio": 7,
                "vllm": 7,
            },
            conflicting_tools={
                "write": {"edit"},
                "edit": {"write"},
            },
        )

    @classmethod
    def _devops_config(cls) -> BaseRLConfig:
        """Built-in devops vertical config."""
        return BaseRLConfig(
            task_type_mappings={
                "deployment": ["shell", "docker", "git", "read", "edit"],
                "containerization": ["docker", "shell", "read", "write"],
                "monitoring": ["shell", "read", "write", "grep"],
                "configuration": ["read", "write", "edit", "grep"],
                "troubleshooting": ["shell", "read", "grep", "docker"],
            },
            quality_thresholds={
                "deployment": 0.90,
                "containerization": 0.85,
                "monitoring": 0.80,
                "configuration": 0.85,
                "troubleshooting": 0.80,
            },
        )

    @classmethod
    def _rag_config(cls) -> BaseRLConfig:
        """Built-in RAG vertical config."""
        return BaseRLConfig(
            active_learners=[
                LearnerType.TOOL_SELECTOR,
                LearnerType.GROUNDING_THRESHOLD,
                LearnerType.QUALITY_WEIGHTS,
            ],
            task_type_mappings={
                "search": ["rag_search", "rag_query", "read"],
                "ingest": ["rag_ingest", "read", "ls", "web_fetch"],
                "synthesis": ["rag_query", "rag_search"],
                "management": ["rag_list", "rag_delete", "rag_stats"],
                "exploration": ["rag_search", "rag_list", "rag_stats", "read"],
            },
            quality_thresholds={
                "search": 0.80,
                "synthesis": 0.85,
                "ingest": 0.75,
                "management": 0.70,
                "exploration": 0.75,
            },
            default_patience={
                "anthropic": 3,
                "openai": 3,
                "ollama": 5,
                "google": 3,
            },
        )

    @classmethod
    def _dataanalysis_config(cls) -> VerticalRLConfig:
        """Built-in data analysis vertical config."""
        return VerticalRLConfig(
            task_type_mappings={
                "eda": ["read", "shell", "write", "ls"],
                "cleaning": ["shell", "read", "write"],
                "visualization": ["shell", "write"],
                "statistics": ["shell", "read", "write"],
                "ml": ["shell", "read", "write"],
                "profiling": ["read", "shell", "ls"],
                "reporting": ["write", "read", "shell"],
            },
            quality_thresholds={
                "eda": 0.75,
                "cleaning": 0.85,
                "visualization": 0.80,
                "statistics": 0.90,
                "ml": 0.85,
                "profiling": 0.75,
                "reporting": 0.80,
            },
            preferred_output_length={
                "eda": "medium",
                "cleaning": "medium",
                "visualization": "long",
                "statistics": "medium",
                "ml": "long",
                "profiling": "short",
                "reporting": "long",
            },
        )

    @classmethod
    def _research_config(cls) -> VerticalRLConfig:
        """Built-in research vertical config."""
        return VerticalRLConfig(
            task_type_mappings={
                "research": ["web_search", "web_fetch", "read", "write"],
                "fact_check": ["web_search", "web_fetch", "grep"],
                "literature": ["web_search", "web_fetch", "read"],
                "competitive": ["web_search", "web_fetch", "write"],
                "synthesis": ["read", "write", "edit"],
                "exploration": ["web_search", "read", "grep", "overview"],
            },
            quality_thresholds={
                "research": 0.85,
                "fact_check": 0.90,
                "literature": 0.85,
                "competitive": 0.80,
                "synthesis": 0.80,
                "exploration": 0.75,
            },
            preferred_providers_by_task={
                "research": ["anthropic", "openai", "google"],
                "fact_check": ["anthropic", "openai"],
                "literature": ["anthropic", "openai", "google"],
            },
        )


class GenericRLHooks:
    """Generic RL hooks that work with any vertical config.

    This class consolidates the duplicate *RLHooks classes that were
    in each vertical module. It provides a unified interface for:

    - Tool recommendations
    - Patience recommendations
    - Quality thresholds

    The hooks delegate to the underlying BaseRLConfig or VerticalRLConfig.

    Example:
        config = RLConfigFactory.create("coding")
        hooks = GenericRLHooks(config)

        tools = hooks.get_tool_recommendation("debugging")
        patience = hooks.get_patience_recommendation("anthropic", "claude-3-opus")
    """

    def __init__(self, config: Optional[BaseRLConfig] = None):
        """Initialize hooks with optional config.

        Args:
            config: RL config to use. If None, uses BaseRLConfig defaults.
        """
        self._config = config or BaseRLConfig()

    @property
    def config(self) -> BaseRLConfig:
        """Get the underlying RL config."""
        return self._config

    def get_tool_recommendation(
        self,
        task_type: str,
        available_tools: Optional[list[str]] = None,
    ) -> list[str]:
        """Get tool recommendations for a task type.

        Args:
            task_type: Type of task (debugging, implementation, etc.)
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

    def should_include_code(self, task_type: str) -> bool:
        """Check if code should be included in output.

        Default implementation returns True. Override for vertical-specific behavior.
        """
        return True

    def __repr__(self) -> str:
        return f"GenericRLHooks(config={self._config})"


# Factory functions for backward compatibility
def get_rl_config_factory() -> type:
    """Get the RLConfigFactory class."""
    return RLConfigFactory


def create_rl_config(vertical: str) -> BaseRLConfig:
    """Create RL config for vertical (convenience function)."""
    return RLConfigFactory.create(vertical)


def create_rl_hooks(vertical: str) -> GenericRLHooks:
    """Create RL hooks for vertical (convenience function)."""
    config = RLConfigFactory.create(vertical)
    return GenericRLHooks(config)


__all__ = [
    "RLConfigFactory",
    "GenericRLHooks",
    "VerticalRLConfig",
    "get_rl_config_factory",
    "create_rl_config",
    "create_rl_hooks",
]
