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

"""Feature flag system for gradual rollout of new features.

This module provides a comprehensive feature flag system that supports:
- Environment variable configuration
- YAML configuration file loading
- Runtime flag enable/disable
- Thread-safe flag management

Feature flags enable gradual rollout of new architecture components
without breaking existing functionality.

Example:
    from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

    manager = get_feature_flag_manager()

    # Check if a feature is enabled
    if manager.is_enabled(FeatureFlag.USE_NEW_CHAT_SERVICE):
        # Use new chat service
        pass
    else:
        # Use legacy implementation
        pass

    # Enable a feature at runtime
    manager.enable(FeatureFlag.USE_NEW_CHAT_SERVICE)

Environment Variables:
    VICTOR_USE_NEW_CHAT_SERVICE=true
    VICTOR_USE_NEW_TOOL_SERVICE=false
    VICTOR_USE_COMPOSITION_OVER_INHERITANCE=true

YAML Configuration (~/.victor/features.yaml):
    features:
        use_new_chat_service: true
        use_new_tool_service: false
        use_composition_over_inheritance: true
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from victor.core.yaml_utils import safe_load as yaml_safe_load

logger = logging.getLogger(__name__)


class FeatureFlag(Enum):
    """Feature flags for gradual rollout of SOLID refactoring.

    Phase 1 - Foundation:
        USE_NEW_CHAT_SERVICE: Use extracted ChatService instead of orchestrator methods
        USE_NEW_TOOL_SERVICE: Use extracted ToolService instead of orchestrator methods
        USE_NEW_CONTEXT_SERVICE: Use extracted ContextService for context management
        USE_NEW_PROVIDER_SERVICE: Use extracted ProviderService for provider management
        USE_NEW_RECOVERY_SERVICE: Use extracted RecoveryService for error recovery
        USE_NEW_SESSION_SERVICE: Use extracted SessionService for session management

    Phase 4 - Vertical Composition:
        USE_COMPOSITION_OVER_INHERITANCE: Use composition-based verticals instead of inheritance

    Phase 5 - Tool Registration:
        USE_STRATEGY_BASED_TOOL_REGISTRATION: Use strategy pattern for tool registration

    Usage:
        Set via environment variable: VICTOR_USE_NEW_CHAT_SERVICE=true
        Set via YAML config: features.use_new_chat_service: true
        Enable at runtime: manager.enable(FeatureFlag.USE_NEW_CHAT_SERVICE)
    """

    # Phase 3 - Service Implementation
    USE_NEW_CHAT_SERVICE = "use_new_chat_service"
    USE_NEW_TOOL_SERVICE = "use_new_tool_service"
    USE_NEW_CONTEXT_SERVICE = "use_new_context_service"
    USE_NEW_PROVIDER_SERVICE = "use_new_provider_service"
    USE_NEW_RECOVERY_SERVICE = "use_new_recovery_service"
    USE_NEW_SESSION_SERVICE = "use_new_session_service"

    # Phase 4 - Vertical Composition
    USE_COMPOSITION_OVER_INHERITANCE = "use_composition_over_inheritance"

    # Phase 5 - Tool Registration
    USE_STRATEGY_BASED_TOOL_REGISTRATION = "use_strategy_based_tool_registration"

    # Phase 6 - Service Layer (Strangler Fig)
    USE_SERVICE_LAYER = "use_service_layer"

    # Phase 7 - LLM Decision Service
    USE_LLM_DECISION_SERVICE = "use_llm_decision_service"

    # Phase 8 - Edge Model for micro-decisions
    USE_EDGE_MODEL = "use_edge_model"

    # Phase 10 - Agentic Loop (single-turn execution with perception/evaluation)
    USE_AGENTIC_LOOP = "use_agentic_loop"

    # Optimization flags (default: False — opt-in only)
    USE_SEMANTIC_RESPONSE_CACHE = "use_semantic_response_cache"
    USE_CONTEXT_TEMPERATURE = "use_context_temperature"
    USE_CONFIDENCE_MONITOR = "use_confidence_monitor"

    # Phase 11 - Smart Model Routing (automatic local→cloud fallback)
    USE_SMART_ROUTING = "use_smart_routing"

    # Priority 4 - Learning from Execution (meta-learning, user feedback, explainability)
    USE_LEARNING_FROM_EXECUTION = "use_learning_from_execution"

    # Tool Broadcasting Optimization (context-aware, economy-first)
    TOOL_STRATEGY_V2 = "tool_strategy_v2"

    # Phase 13 - Agentic rollout controls (default: False — opt-in only)
    USE_AGENTIC_BENCH_GATES = "use_agentic_bench_gates"
    USE_CALIBRATED_COMPLETION = "use_calibrated_completion"
    USE_AGENTIC_RETRIEVAL_REPAIR = "use_agentic_retrieval_repair"
    USE_UTILITY_RETRIEVAL = "use_utility_retrieval"
    USE_PROMPT_COMPLETENESS_GUARD = "use_prompt_completeness_guard"
    USE_PROMPT_DICTIONARY_COMPRESSION = "use_prompt_dictionary_compression"
    USE_PRIME_MEMORY_EVOLUTION = "use_prime_memory_evolution"
    USE_EXTERNAL_AGENTIC_BENCHMARKS = "use_external_agentic_benchmarks"

    # Phase 9 - Prompt Optimization (controlled via settings.prompt_optimization)
    # USE_PROMPT_OPTIMIZER → settings.prompt_optimization.enabled
    # USE_GEPA_V2          → settings.prompt_optimization.gepa.enabled
    # USE_GEPA_TRACE_ENRICHMENT → settings.prompt_optimization.gepa.capture_reasoning

    # Phase 12 - Rich Formatting (unified formatter system with preview integration)
    # USE_RICH_FORMATTING → settings.rich_formatting_enabled (master switch)
    # This flag controls the entire Rich formatter system including:
    # - Tool output formatting with Rich markup
    # - Preview strategy integration
    # - Performance guards and error handling
    USE_RICH_FORMATTING = "use_rich_formatting"

    def get_env_var_name(self) -> str:
        """Get the environment variable name for this flag.

        Returns:
            Environment variable name (e.g., VICTOR_USE_NEW_CHAT_SERVICE)
        """
        return f"VICTOR_{self.value.upper()}"

    def get_yaml_key(self) -> str:
        """Get the YAML configuration key for this flag.

        Returns:
            YAML key (e.g., use_new_chat_service)
        """
        return self.value

    def is_opt_in_by_default(self) -> bool:
        """Whether the flag should stay disabled unless explicitly enabled."""
        return self in {
            FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE,
            FeatureFlag.USE_CONTEXT_TEMPERATURE,
            FeatureFlag.USE_CONFIDENCE_MONITOR,
            FeatureFlag.USE_LEARNING_FROM_EXECUTION,
            FeatureFlag.TOOL_STRATEGY_V2,
            FeatureFlag.USE_AGENTIC_BENCH_GATES,
            FeatureFlag.USE_CALIBRATED_COMPLETION,
            FeatureFlag.USE_AGENTIC_RETRIEVAL_REPAIR,
            FeatureFlag.USE_UTILITY_RETRIEVAL,
            FeatureFlag.USE_PROMPT_COMPLETENESS_GUARD,
            FeatureFlag.USE_PROMPT_DICTIONARY_COMPRESSION,
            FeatureFlag.USE_PRIME_MEMORY_EVOLUTION,
            FeatureFlag.USE_EXTERNAL_AGENTIC_BENCHMARKS,
        }

    def get_default_enabled(self, fallback: bool) -> bool:
        """Return the effective default state for this flag."""
        if self.is_opt_in_by_default():
            return False
        return fallback


@dataclass
class FeatureFlagConfig:
    """Configuration for feature flag loading.

    Attributes:
        config_path: Path to YAML configuration file
        env_prefix: Prefix for environment variables
        default_enabled: Default state for flags not explicitly configured
        strict_mode: If True, raises errors on invalid config; if False, logs warnings
    """

    config_path: Optional[Path] = None
    env_prefix: str = "VICTOR_"
    default_enabled: bool = True
    strict_mode: bool = False


class FeatureFlagManager:
    """Manage feature flags with environment variable and YAML config support.

    The manager loads flag values from multiple sources in order of precedence:
    1. Runtime enable() calls (highest priority)
    2. Environment variables
    3. YAML configuration file
    4. Default value (lowest priority)

    Thread-safe for concurrent access to flag state.

    Example:
        manager = FeatureFlagManager()

        # Check if a feature is enabled
        if manager.is_enabled(FeatureFlag.USE_NEW_CHAT_SERVICE):
            use_new_service()

        # Enable a feature at runtime
        manager.enable(FeatureFlag.USE_NEW_CHAT_SERVICE)

        # Get all enabled flags
        enabled_flags = manager.get_enabled_flags()
    """

    def __init__(self, config: Optional[FeatureFlagConfig] = None) -> None:
        """Initialize feature flag manager.

        Args:
            config: Optional configuration for flag loading behavior
        """
        self._config = config or FeatureFlagConfig()
        self._flags: Dict[FeatureFlag, bool] = {}
        self._runtime_overrides: Dict[FeatureFlag, bool] = {}
        self._lock = threading.RLock()
        self._load_from_config()

    def is_enabled(self, flag: FeatureFlag) -> bool:
        """Check if a feature flag is enabled.

        Checks in order of precedence:
        1. Runtime overrides (from enable() calls)
        2. Environment variables
        3. YAML configuration
        4. Default value

        Args:
            flag: Feature flag to check

        Returns:
            True if the flag is enabled, False otherwise
        """
        with self._lock:
            # Check runtime override first
            if flag in self._runtime_overrides:
                return self._runtime_overrides[flag]

            # Check environment variable
            env_var = flag.get_env_var_name()
            if env_var in os.environ:
                return os.environ[env_var].lower() in ("true", "1", "yes", "on")

            # Check loaded configuration
            if flag in self._flags:
                return self._flags[flag]

            # Return default
            return flag.get_default_enabled(self._config.default_enabled)

    def enable(self, flag: FeatureFlag) -> None:
        """Enable a feature flag at runtime.

        Runtime overrides take precedence over environment variables
        and configuration files.

        Args:
            flag: Feature flag to enable
        """
        with self._lock:
            self._runtime_overrides[flag] = True
            logger.debug(f"Enabled feature flag: {flag.value}")

    def disable(self, flag: FeatureFlag) -> None:
        """Disable a feature flag at runtime.

        Runtime overrides take precedence over environment variables
        and configuration files.

        Args:
            flag: Feature flag to disable
        """
        with self._lock:
            self._runtime_overrides[flag] = False
            logger.debug(f"Disabled feature flag: {flag.value}")

    def clear_runtime_override(self, flag: FeatureFlag) -> None:
        """Clear runtime override for a feature flag.

        After clearing, the flag value will be determined by
        environment variables or configuration files.

        Args:
            flag: Feature flag to clear override for
        """
        with self._lock:
            self._runtime_overrides.pop(flag, None)
            logger.debug(f"Cleared runtime override for: {flag.value}")

    def set(self, flag: FeatureFlag, enabled: bool) -> None:
        """Set a feature flag state at runtime.

        Convenience method that calls enable() or disable().

        Args:
            flag: Feature flag to set
            enabled: Whether to enable or disable the flag
        """
        if enabled:
            self.enable(flag)
        else:
            self.disable(flag)

    def get_enabled_flags(self) -> Dict[FeatureFlag, bool]:
        """Get all feature flags and their current states.

        Returns:
            Dictionary mapping feature flags to their enabled state
        """
        with self._lock:
            return {flag: self.is_enabled(flag) for flag in FeatureFlag}

    def reload_config(self) -> None:
        """Reload configuration from environment variables and YAML file.

        Clears runtime overrides and reloads flag values from configuration sources.
        """
        with self._lock:
            self._runtime_overrides.clear()
            self._flags.clear()
            self._load_from_config()
            logger.info("Reloaded feature flag configuration")

    def _load_from_config(self) -> None:
        """Load flag values from environment variables and YAML configuration."""
        # Load from YAML file if path is specified
        if self._config.config_path and self._config.config_path.exists():
            self._load_from_yaml()

        # Environment variables are checked directly in is_enabled()
        # No need to preload them

    def _load_from_yaml(self) -> None:
        """Load flag values from YAML configuration file.

        Expected YAML format:
            features:
                use_new_chat_service: true
                use_new_tool_service: false
                use_composition_over_inheritance: true
        """
        try:
            with open(self._config.config_path, "r") as f:
                data = yaml_safe_load(f)

            if not data or not isinstance(data, dict):
                logger.warning(f"Invalid feature flag config: {self._config.config_path}")
                return

            features = data.get("features", {})
            if not isinstance(features, dict):
                logger.warning("Invalid 'features' section in config")
                return

            # Map YAML keys to FeatureFlag enums
            yaml_key_to_flag = {flag.get_yaml_key(): flag for flag in FeatureFlag}

            for key, value in features.items():
                if key in yaml_key_to_flag:
                    flag = yaml_key_to_flag[key]
                    if isinstance(value, bool):
                        self._flags[flag] = value
                    elif self._config.strict_mode:
                        raise ValueError(
                            f"Invalid value for feature flag '{key}': "
                            f"expected bool, got {type(value).__name__}"
                        )
                    else:
                        logger.warning(
                            f"Invalid value for feature flag '{key}': "
                            f"expected bool, got {type(value).__name__}"
                        )

        except Exception as e:
            if self._config.strict_mode:
                raise
            logger.warning(f"Failed to load feature flags from YAML: {e}")


# =============================================================================
# Global Manager
# =============================================================================

_global_manager: Optional[FeatureFlagManager] = None
_global_lock = threading.Lock()


def get_feature_flag_manager(
    config: Optional[FeatureFlagConfig] = None,
    force_reload: bool = False,
) -> FeatureFlagManager:
    """Get the global feature flag manager.

    Creates a new manager if one doesn't exist or if force_reload is True.

    Args:
        config: Optional configuration for the manager
        force_reload: If True, recreate the manager even if one exists

    Returns:
        Global feature flag manager instance
    """
    global _global_manager

    with _global_lock:
        if _global_manager is None or force_reload:
            # Determine default config path
            if config is None or config.config_path is None:
                from victor.config.settings import GLOBAL_VICTOR_DIR

                default_config_path = GLOBAL_VICTOR_DIR / "features.yaml"
                if config is None:
                    config = FeatureFlagConfig(config_path=default_config_path)
                else:
                    # Update existing config with default path
                    config = FeatureFlagConfig(
                        config_path=default_config_path,
                        env_prefix=config.env_prefix,
                        default_enabled=config.default_enabled,
                        strict_mode=config.strict_mode,
                    )

            _global_manager = FeatureFlagManager(config)

        return _global_manager


def reset_feature_flag_manager() -> None:
    """Reset the global feature flag manager.

    Primarily useful for testing when you need to clear flag state.
    """
    global _global_manager
    with _global_lock:
        _global_manager = None


def is_feature_enabled(flag: FeatureFlag) -> bool:
    """Convenience function to check if a feature flag is enabled.

    Args:
        flag: Feature flag to check

    Returns:
        True if the flag is enabled, False otherwise
    """
    return get_feature_flag_manager().is_enabled(flag)


def enable_feature(flag: FeatureFlag) -> None:
    """Convenience function to enable a feature flag.

    Args:
        flag: Feature flag to enable
    """
    get_feature_flag_manager().enable(flag)


def disable_feature(flag: FeatureFlag) -> None:
    """Convenience function to disable a feature flag.

    Args:
        flag: Feature flag to disable
    """
    get_feature_flag_manager().disable(flag)
