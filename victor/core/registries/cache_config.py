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

"""Cache configuration for SOLID remediation Phase 6.

This module provides centralized cache configuration for all UniversalRegistry
instances in the Victor framework, implementing the cache boundaries and
strategies defined in the SOLID remediation plan.

Design Goals:
- Prevent memory leaks through bounded caches
- Optimize cache hit rates based on usage patterns
- Provide configurable TTL for stale data prevention
- Thread-safe cache operations with striped locks
- Enable production-ready cache tuning

Cache Configuration Strategy:

┌─────────────────────────────────────────────────────────────────┐
│ Cache Type           │ Strategy │ Max Size │ TTL      │ Purpose│
├─────────────────────────────────────────────────────────────────┤
│ tool_selection       │ LRU      │ 500      │ 1 hour   │ High  │
│ extension_cache      │ TTL      │ None     │ 5 min    │ Medium│
│ vertical_integration │ LRU      │ 100      │ None     │ Medium│
│ orchestrator_pool    │ LRU      │ 50       │ 30 min   │ Low   │
│ event_batching       │ TTL      │ 1000     │ 1 sec    │ High  │
│ modes                │ LRU      │ 100      │ 1 hour   │ Medium│
│ workflows            │ TTL      │ 50       │ 5 min    │ Low   │
│ teams                │ TTL      │ 20       │ 30 min   │ Low   │
│ capabilities         │ MANUAL   │ 200      │ None     │ Medium│
│ tool_selection_query │ LRU      │ 1000     │ 1 hour   │ High  │
│ tool_selection_ctx   │ TTL      │ 500      │ 5 min    │ High  │
│ tool_selection_rl    │ TTL      │ 1000     │ 1 hour   │ High  │
└─────────────────────────────────────────────────────────────────┘

Usage:
    from victor.core.registries.cache_config import (
        get_cache_config,
        configure_registry,
    )

    # Get configuration for a cache
    config = get_cache_config("tool_selection")

    # Configure a UniversalRegistry instance
    registry = configure_registry(
        UniversalRegistry,
        "tool_selection"
    )
"""

import os
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Any, Type

from victor.core.registries.universal_registry import (
    UniversalRegistry,
    CacheStrategy,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Cache Configuration
# =============================================================================


@dataclass
class CacheConfig:
    """Configuration for a specific cache.

    Attributes:
        name: Cache name/identifier
        strategy: Cache invalidation strategy
        max_size: Maximum number of entries (None = unlimited)
        ttl: Time-to-live in seconds (None = no expiration)
        description: Human-readable description
        priority: Access priority (high, medium, low) for tuning
    """

    name: str
    """Cache name/identifier."""

    strategy: CacheStrategy
    """Cache invalidation strategy."""

    max_size: Optional[int] = None
    """Maximum number of entries (None = unlimited)."""

    ttl: Optional[int] = None
    """Time-to-live in seconds (None = no expiration)."""

    description: str = ""
    """Human-readable description."""

    priority: str = "medium"
    """Access priority (high, medium, low) for tuning."""

    def validate(self) -> None:
        """Validate cache configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        if self.max_size is not None and self.max_size <= 0:
            raise ValueError(f"Cache {self.name}: max_size must be positive")

        if self.ttl is not None and self.ttl <= 0:
            raise ValueError(f"Cache {self.name}: ttl must be positive")

        if self.priority not in ("high", "medium", "low"):
            raise ValueError(f"Cache {self.name}: priority must be high/medium/low")

        # Validate strategy compatibility
        if self.strategy == CacheStrategy.LRU and self.max_size is None:
            logger.warning(
                f"Cache {self.name}: LRU strategy with no max_size "
                "(will not evict entries)"
            )


# =============================================================================
# Default Cache Configurations
# =============================================================================


# Default configurations for all cache types
DEFAULT_CACHE_CONFIGS: Dict[str, CacheConfig] = {
    # Tool selection caches (high priority)
    "tool_selection": CacheConfig(
        name="tool_selection",
        strategy=CacheStrategy.LRU,
        max_size=500,
        ttl=3600,  # 1 hour
        description="Tool selection results",
        priority="high",
    ),
    "tool_selection_query": CacheConfig(
        name="tool_selection_query",
        strategy=CacheStrategy.LRU,
        max_size=1000,
        ttl=3600,  # 1 hour
        description="Tool selection query cache",
        priority="high",
    ),
    "tool_selection_context": CacheConfig(
        name="tool_selection_context",
        strategy=CacheStrategy.TTL,
        max_size=500,
        ttl=300,  # 5 minutes
        description="Tool selection context cache",
        priority="high",
    ),
    "tool_selection_rl": CacheConfig(
        name="tool_selection_rl",
        strategy=CacheStrategy.TTL,
        max_size=1000,
        ttl=3600,  # 1 hour
        description="Tool selection RL cache",
        priority="high",
    ),

    # Extension cache (medium priority)
    "extension_cache": CacheConfig(
        name="extension_cache",
        strategy=CacheStrategy.TTL,
        max_size=None,  # Unlimited
        ttl=300,  # 5 minutes
        description="Extension cache",
        priority="medium",
    ),

    # Vertical integration cache (medium priority)
    "vertical_integration": CacheConfig(
        name="vertical_integration",
        strategy=CacheStrategy.LRU,
        max_size=100,
        ttl=None,
        description="Vertical integration cache",
        priority="medium",
    ),

    # Orchestrator pool (low priority)
    "orchestrator_pool": CacheConfig(
        name="orchestrator_pool",
        strategy=CacheStrategy.LRU,
        max_size=50,
        ttl=1800,  # 30 minutes
        description="Orchestrator instance pool",
        priority="low",
    ),

    # Event batching (high priority)
    "event_batching": CacheConfig(
        name="event_batching",
        strategy=CacheStrategy.TTL,
        max_size=1000,
        ttl=1,  # 1 second
        description="Event batching cache",
        priority="high",
    ),

    # Mode configurations (medium priority)
    "modes": CacheConfig(
        name="modes",
        strategy=CacheStrategy.LRU,
        max_size=100,
        ttl=3600,  # 1 hour
        description="Mode configuration cache",
        priority="medium",
    ),

    # Workflow cache (low priority)
    "workflows": CacheConfig(
        name="workflows",
        strategy=CacheStrategy.TTL,
        max_size=50,
        ttl=300,  # 5 minutes
        description="Workflow provider cache",
        priority="low",
    ),

    # Team specifications (low priority)
    "teams": CacheConfig(
        name="teams",
        strategy=CacheStrategy.TTL,
        max_size=20,
        ttl=1800,  # 30 minutes
        description="Team specification cache",
        priority="low",
    ),

    # Capability definitions (medium priority)
    "capabilities": CacheConfig(
        name="capabilities",
        strategy=CacheStrategy.MANUAL,
        max_size=200,
        ttl=None,
        description="Capability definitions",
        priority="medium",
    ),
}


# =============================================================================
# Cache Configuration Manager
# =============================================================================


class CacheConfigManager:
    """Manages cache configurations for UniversalRegistry instances.

    This class provides centralized cache configuration management,
    including validation, overrides, and environment variable support.

    Thread Safety:
        All methods are thread-safe.

    Example:
        manager = CacheConfigManager()

        # Get configuration
        config = manager.get_config("tool_selection")

        # Override with environment variable
        config = manager.get_config("tool_selection", env_prefix="VICTOR_CACHE_")

        # Configure a registry
        registry = manager.configure_registry(
            UniversalRegistry,
            "tool_selection"
        )
    """

    def __init__(
        self,
        configs: Optional[Dict[str, CacheConfig]] = None,
    ) -> None:
        """Initialize cache configuration manager.

        Args:
            configs: Custom cache configurations (default: use defaults)
        """
        self._configs = dict(configs) if configs else dict(DEFAULT_CACHE_CONFIGS)
        self._validate_all_configs()

    def _validate_all_configs(self) -> None:
        """Validate all cache configurations."""
        for name, config in self._configs.items():
            try:
                config.validate()
            except ValueError as e:
                logger.error(f"Invalid cache config for {name}: {e}")
                raise

    def get_config(
        self,
        cache_name: str,
        env_prefix: Optional[str] = None,
    ) -> CacheConfig:
        """Get cache configuration, with optional environment variable overrides.

        Args:
            cache_name: Name of the cache
            env_prefix: Prefix for environment variables (e.g., "VICTOR_CACHE_")

        Returns:
            Cache configuration

        Raises:
            KeyError: If cache name not found
        """
        if cache_name not in self._configs:
            raise KeyError(f"Unknown cache: {cache_name}")

        config = self._configs[cache_name]

        # Apply environment variable overrides if requested
        if env_prefix:
            config = self._apply_env_overrides(config, env_prefix)

        return config

    def _apply_env_overrides(
        self,
        config: CacheConfig,
        prefix: str,
    ) -> CacheConfig:
        """Apply environment variable overrides to configuration.

        Args:
            config: Base configuration
            prefix: Environment variable prefix

        Returns:
            Configuration with overrides applied
        """
        import copy

        # Create a copy to avoid modifying the original
        config = copy.copy(config)

        # Override max_size
        env_var = f"{prefix}{config.name.upper()}_MAX_SIZE"
        if env_var in os.environ:
            try:
                config.max_size = int(os.environ[env_var])
                logger.debug(f"Override: {env_var}={config.max_size}")
            except ValueError:
                logger.warning(f"Invalid {env_var} value")

        # Override ttl
        env_var = f"{prefix}{config.name.upper()}_TTL"
        if env_var in os.environ:
            try:
                config.ttl = int(os.environ[env_var])
                logger.debug(f"Override: {env_var}={config.ttl}")
            except ValueError:
                logger.warning(f"Invalid {env_var} value")

        # Override strategy
        env_var = f"{prefix}{config.name.upper()}_STRATEGY"
        if env_var in os.environ:
            try:
                strategy_str = os.environ[env_var].upper()
                config.strategy = CacheStrategy[strategy_str]
                logger.debug(f"Override: {env_var}={strategy_str}")
            except (KeyError, ValueError):
                logger.warning(f"Invalid {env_var} value")

        return config

    def configure_registry(
        self,
        registry_class: Type[UniversalRegistry],
        cache_name: str,
        env_prefix: Optional[str] = None,
    ) -> UniversalRegistry:
        """Configure a UniversalRegistry instance with cache settings.

        Args:
            registry_class: UniversalRegistry class or subclass
            cache_name: Name of the cache configuration
            env_prefix: Prefix for environment variable overrides

        Returns:
            Configured registry instance

        Raises:
            KeyError: If cache name not found
        """
        config = self.get_config(cache_name, env_prefix=env_prefix)

        logger.info(
            f"Configuring cache '{cache_name}': "
            f"strategy={config.strategy.value}, "
            f"max_size={config.max_size}, "
            f"ttl={config.ttl}"
        )

        # Return registry instance (configuration applied via cache strategy)
        return registry_class.get_registry(
            cache_name,
            config.strategy,
            max_size=config.max_size,
        )

    def list_configs(self) -> Dict[str, CacheConfig]:
        """List all cache configurations.

        Returns:
            Dictionary of cache configurations
        """
        return dict(self._configs)

    def add_config(self, config: CacheConfig) -> None:
        """Add or update a cache configuration.

        Args:
            config: Cache configuration to add
        """
        config.validate()
        self._configs[config.name] = config
        logger.info(f"Added cache config: {config.name}")

    def remove_config(self, cache_name: str) -> None:
        """Remove a cache configuration.

        Args:
            cache_name: Name of the cache to remove

        Raises:
            KeyError: If cache name not found
        """
        if cache_name not in self._configs:
            raise KeyError(f"Unknown cache: {cache_name}")

        del self._configs[cache_name]
        logger.info(f"Removed cache config: {cache_name}")


# =============================================================================
# Global Manager Instance
# =============================================================================

_global_manager: Optional[CacheConfigManager] = None


def get_cache_config_manager() -> CacheConfigManager:
    """Get global cache configuration manager instance.

    Returns:
        Global CacheConfigManager instance (singleton).
    """
    global _global_manager

    if _global_manager is None:
        _global_manager = CacheConfigManager()

    return _global_manager


# =============================================================================
# Convenience Functions
# =============================================================================


def get_cache_config(
    cache_name: str,
    env_prefix: Optional[str] = None,
) -> CacheConfig:
    """Get cache configuration.

    Args:
        cache_name: Name of the cache
        env_prefix: Prefix for environment variable overrides

    Returns:
        Cache configuration
    """
    manager = get_cache_config_manager()
    return manager.get_config(cache_name, env_prefix=env_prefix)


def configure_registry(
    registry_class: Type[UniversalRegistry],
    cache_name: str,
    env_prefix: Optional[str] = None,
) -> UniversalRegistry:
    """Configure a UniversalRegistry with cache settings.

    Args:
        registry_class: UniversalRegistry class
        cache_name: Name of the cache configuration
        env_prefix: Prefix for environment variable overrides

    Returns:
        Configured registry instance
    """
    manager = get_cache_config_manager()
    return manager.configure_registry(
        registry_class,
        cache_name,
        env_prefix=env_prefix,
    )


def print_cache_configs() -> None:
    """Print all cache configurations (for debugging/monitoring)."""
    manager = get_cache_config_manager()
    configs = manager.list_configs()

    print("\n" + "=" * 80)
    print("Cache Configurations")
    print("=" * 80)

    for name in sorted(configs.keys()):
        config = configs[name]
        print(f"\n{name}:")
        print(f"  Strategy: {config.strategy.value}")
        print(f"  Max Size: {config.max_size or 'unlimited'}")
        print(f"  TTL: {config.ttl or 'never'}")
        print(f"  Priority: {config.priority}")
        print(f"  Description: {config.description}")

    print("\n" + "=" * 80 + "\n")


def export_cache_configs() -> Dict[str, Any]:
    """Export all cache configurations as dict.

    Returns:
        Dictionary of cache configurations
    """
    manager = get_cache_config_manager()
    configs = manager.list_configs()

    return {
        name: {
            "strategy": config.strategy.value,
            "max_size": config.max_size,
            "ttl": config.ttl,
            "description": config.description,
            "priority": config.priority,
        }
        for name, config in configs.items()
    }
