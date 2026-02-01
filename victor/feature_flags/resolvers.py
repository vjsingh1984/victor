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

"""Flag resolution logic for feature flag system.

This module provides the flag resolution pipeline with priority-based lookup:
1. Environment variables (VICTOR_FEATURE_*_ENABLED)
2. Settings file (settings.feature_flags.*)
3. Runtime API (manager.set_flag())
4. Default value

Supports staged rollout, A/B testing variants, and integration with Settings.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional
from collections.abc import Callable

from victor.config.settings import Settings

logger = logging.getLogger(__name__)


class FlagResolver(ABC):
    """Abstract base for flag resolution strategies.

    Implementations define how to resolve feature flag values from different sources.

    Design Pattern: Strategy Pattern
    - Each resolver implements a specific resolution strategy
    - Resolvers can be chained for fallback behavior
    """

    @abstractmethod
    def resolve(self, flag_name: str, default: bool = False) -> Optional[bool]:
        """Resolve flag value.

        Args:
            flag_name: Name of the flag (e.g., "hierarchical_planning_enabled")
            default: Default value if flag not found

        Returns:
            Flag value or None if not resolved by this strategy
        """
        pass

    @abstractmethod
    def set(self, flag_name: str, value: bool) -> bool:
        """Set flag value.

        Args:
            flag_name: Name of the flag
            value: Value to set

        Returns:
            True if flag was set successfully
        """
        pass


class EnvironmentFlagResolver(FlagResolver):
    """Resolve flags from environment variables.

    Environment variable format: VICTOR_FEATURE_<FLAG_NAME>_ENABLED
    Example: VICTOR_FEATURE_HIERARCHICAL_PLANNING_ENABLED=true

    Priority: 1 (Highest)
    """

    def __init__(self) -> None:
        """Initialize environment resolver."""
        self._prefix = "VICTOR_FEATURE_"
        self._suffix = "_ENABLED"

    def resolve(self, flag_name: str, default: bool = False) -> Optional[bool]:
        """Resolve flag from environment variable.

        Args:
            flag_name: Name of the flag
            default: Default value (not used in this resolver)

        Returns:
            Flag value or None if not set
        """
        # Convert flag_name to ENV_VAR_NAME format
        # Example: hierarchical_planning_enabled -> HIERARCHICAL_PLANNING_ENABLED
        env_name = f"{self._prefix}{flag_name.upper()}{self._suffix}"

        env_value = os.getenv(env_name)
        if env_value is None:
            return None

        # Parse boolean from string
        return self._parse_bool(env_value)

    def set(self, flag_name: str, value: bool) -> bool:
        """Set flag via environment variable (not recommended - requires process restart).

        Args:
            flag_name: Name of the flag
            value: Value to set

        Returns:
            True (always succeeds, but requires restart)
        """
        env_name = f"{self._prefix}{flag_name.upper()}{self._suffix}"
        os.environ[env_name] = str(value)
        logger.warning(f"Set environment variable {env_name}={value} (requires restart)")
        return True

    @staticmethod
    def _parse_bool(value: str) -> bool:
        """Parse boolean from string.

        Args:
            value: String value

        Returns:
            Boolean value

        Accepts: true, yes, 1, on (case-insensitive)
        """
        return value.lower() in ("true", "yes", "1", "on", "enabled")


class SettingsFlagResolver(FlagResolver):
    """Resolve flags from Settings configuration.

    Settings format: settings.feature_flags.<flag_name>
    Example: settings.feature_flags = {"hierarchical_planning_enabled": True}

    Priority: 2
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize settings resolver.

        Args:
            settings: Settings instance (creates default if None)
        """
        self._settings = settings or Settings()

    def resolve(self, flag_name: str, default: bool = False) -> Optional[bool]:
        """Resolve flag from settings.

        Args:
            flag_name: Name of the flag
            default: Default value (not used in this resolver)

        Returns:
            Flag value or None if not in settings
        """
        # Access feature_flags from settings
        feature_flags = getattr(self._settings, "feature_flags", None)
        if not isinstance(feature_flags, dict):
            return None

        value = feature_flags.get(flag_name)
        if value is None:
            return None

        # Ensure boolean
        return bool(value)

    def set(self, flag_name: str, value: bool) -> bool:
        """Set flag in settings.

        Args:
            flag_name: Name of the flag
            value: Value to set

        Returns:
            True if set successfully

        Note: Changes to Settings don't persist unless saved to profiles.yaml
        """
        if not hasattr(self._settings, "feature_flags"):
            self._settings.feature_flags = {}  # type: ignore[attr-defined]

        feature_flags = self._settings.feature_flags  # type: ignore[attr-defined]
        if isinstance(feature_flags, dict):
            feature_flags[flag_name] = value
        logger.info(f"Set feature flag {flag_name}={value} in settings")
        return True


class RuntimeFlagResolver(FlagResolver):
    """Resolve flags from runtime storage (in-memory).

    Stores flags in process memory for hot-reload without restart.
    Changes are lost on process restart unless persisted elsewhere.

    Priority: 3
    """

    def __init__(self) -> None:
        """Initialize runtime resolver."""
        self._flags: dict[str, bool] = {}

    def resolve(self, flag_name: str, default: bool = False) -> Optional[bool]:
        """Resolve flag from runtime storage.

        Args:
            flag_name: Name of the flag
            default: Default value (not used in this resolver)

        Returns:
            Flag value or None if not in runtime storage
        """
        return self._flags.get(flag_name)

    def set(self, flag_name: str, value: bool) -> bool:
        """Set flag in runtime storage.

        Args:
            flag_name: Name of the flag
            value: Value to set

        Returns:
            True (always succeeds)
        """
        self._flags[flag_name] = value
        logger.info(f"Set feature flag {flag_name}={value} in runtime storage")
        return True

    def clear(self, flag_name: str) -> None:
        """Clear flag from runtime storage.

        Args:
            flag_name: Name of the flag to clear
        """
        self._flags.pop(flag_name, None)
        logger.info(f"Cleared feature flag {flag_name} from runtime storage")

    def get_all(self) -> dict[str, bool]:
        """Get all runtime flags.

        Returns:
            Dictionary of all flag names to values
        """
        return self._flags.copy()

    def reset(self) -> None:
        """Reset all runtime flags."""
        self._flags.clear()
        logger.info("Reset all runtime feature flags")


class ChainedFlagResolver(FlagResolver):
    """Chain multiple resolvers with fallback priority.

    Tries each resolver in order until one returns a non-None value.
    If all resolvers return None, returns the default value.

    Priority Order (default):
    1. EnvironmentFlagResolver
    2. SettingsFlagResolver
    3. RuntimeFlagResolver
    4. Default value

    Design Pattern: Chain of Responsibility
    """

    def __init__(self, resolvers: Optional[list[FlagResolver]] = None) -> None:
        """Initialize chained resolver.

        Args:
            resolvers: List of resolvers in priority order (default: env -> settings -> runtime)
        """
        if resolvers is None:
            # Default priority: env -> settings -> runtime
            resolvers = [
                EnvironmentFlagResolver(),
                SettingsFlagResolver(),
                RuntimeFlagResolver(),
            ]

        self._resolvers = resolvers

    def resolve(self, flag_name: str, default: bool = False) -> Optional[bool]:
        """Resolve flag using chain of resolvers.

        Args:
            flag_name: Name of the flag
            default: Default value if none of the resolvers find it

        Returns:
            Flag value from first resolver that finds it, or default
        """
        for resolver in self._resolvers:
            try:
                value = resolver.resolve(flag_name)
                if value is not None:
                    logger.debug(
                        f"Flag {flag_name} resolved to {value} by {resolver.__class__.__name__}"
                    )
                    return value
            except Exception as e:
                logger.warning(
                    f"Resolver {resolver.__class__.__name__} failed for {flag_name}: {e}"
                )
                continue

        logger.debug(f"Flag {flag_name} not resolved by any resolver, using default: {default}")
        return default

    def set(self, flag_name: str, value: bool) -> bool:
        """Set flag in all writable resolvers.

        Args:
            flag_name: Name of the flag
            value: Value to set

        Returns:
            True if at least one resolver succeeded
        """
        success = False
        for resolver in self._resolvers:
            try:
                if resolver.set(flag_name, value):
                    success = True
            except Exception as e:
                logger.warning(f"Failed to set {flag_name} in {resolver.__class__.__name__}: {e}")

        return success

    def add_resolver(self, resolver: FlagResolver, priority: int = -1) -> None:
        """Add resolver to chain.

        Args:
            resolver: Resolver instance to add
            priority: Position in chain (-1 = append, 0 = prepend)

        Example:
            chain.add_resolver(CustomResolver(), priority=0)  # Prepend
        """
        if priority == -1:
            self._resolvers.append(resolver)
        else:
            self._resolvers.insert(priority, resolver)


class StagedRolloutResolver(FlagResolver):
    """Resolver for staged percentage-based rollout.

    Enables flags for a percentage of users based on a stable identifier.
    Useful for canary deployments and A/B testing.

    Priority: Custom (typically wrapped in ChainedFlagResolver)

    Example:
        # Enable for 10% of users
        resolver = StagedRolloutResolver(rollout_percentage=10)
        enabled = resolver.resolve("new_feature", default=False)
    """

    def __init__(
        self,
        rollout_percentage: float = 0.0,
        user_id_provider: Optional[Callable[[], str]] = None,
    ) -> None:
        """Initialize staged rollout resolver.

        Args:
            rollout_percentage: Percentage of users to enable (0.0-100.0)
            user_id_provider: Callable that returns user ID (defaults to hostname)

        Example:
            # Roll out to 25% of users
            resolver = StagedRolloutResolver(rollout_percentage=25.0)
        """
        if not 0.0 <= rollout_percentage <= 100.0:
            raise ValueError(
                f"rollout_percentage must be between 0 and 100, got {rollout_percentage}"
            )

        self._rollout_percentage = rollout_percentage
        self._user_id_provider = user_id_provider or self._default_user_id

    @staticmethod
    def _default_user_id() -> str:
        """Get default user ID from hostname."""
        import socket

        return socket.gethostname()

    def resolve(self, flag_name: str, default: bool = False) -> Optional[bool]:
        """Resolve flag based on staged rollout percentage.

        Args:
            flag_name: Name of the flag
            default: Default value

        Returns:
            True if user is in rollout percentage
        """
        # Get stable user identifier
        user_id = self._user_id_provider()

        # Create deterministic hash from flag_name + user_id
        import hashlib

        combined = f"{flag_name}:{user_id}".encode()
        hash_value = int(hashlib.sha256(combined).hexdigest(), 16)

        # Map hash to 0-100 range
        rollout_value = (hash_value % 10000) / 100.0

        # Check if within rollout percentage
        enabled = rollout_value < self._rollout_percentage

        if enabled:
            logger.debug(
                f"User {user_id} in rollout for {flag_name} (percentage: {self._rollout_percentage}%)"
            )

        return enabled

    def set(self, flag_name: str, value: bool) -> bool:
        """Set rollout percentage (not supported - use constructor).

        Args:
            flag_name: Not used
            value: Not used

        Returns:
            False (not supported)
        """
        logger.warning(
            "StagedRolloutResolver does not support set() - create new instance with different percentage"
        )
        return False

    def set_rollout_percentage(self, percentage: float) -> None:
        """Update rollout percentage.

        Args:
            percentage: New rollout percentage (0.0-100.0)
        """
        if not 0.0 <= percentage <= 100.0:
            raise ValueError(f"rollout_percentage must be between 0 and 100, got {percentage}")

        self._rollout_percentage = percentage
        logger.info(f"Updated rollout percentage to {percentage}%")


class ABTestingResolver(FlagResolver):
    """Resolver for A/B testing variants.

    Assigns users to different variants (A, B, C, etc.) based on stable identifier.
    Useful for testing different implementations of the same feature.

    Example:
        resolver = ABTestingResolver(variants=["A", "B", "C"])
        variant = resolver.resolve_variant("experiment_1")
        if variant == "A":
            # Use control implementation
        elif variant == "B":
            # Use treatment 1
        elif variant == "C":
            # Use treatment 2
    """

    def __init__(
        self,
        variants: list[str],
        weights: Optional[list[float]] = None,
        user_id_provider: Optional[Callable[[], str]] = None,
    ) -> None:
        """Initialize A/B testing resolver.

        Args:
            variants: List of variant identifiers (e.g., ["A", "B", "C"])
            weights: Optional weights for each variant (defaults to uniform)
            user_id_provider: Callable that returns user ID

        Example:
            # 50/50 split
            resolver = ABTestingResolver(variants=["control", "treatment"])

            # 80/20 split
            resolver = ABTestingResolver(
                variants=["control", "treatment"],
                weights=[0.8, 0.2]
            )
        """
        if not variants:
            raise ValueError("variants must not be empty")

        self._variants = variants

        if weights is None:
            # Uniform distribution
            self._weights = [1.0 / len(variants)] * len(variants)
        else:
            if len(weights) != len(variants):
                raise ValueError("weights must have same length as variants")
            if abs(sum(weights) - 1.0) > 0.001:
                raise ValueError("weights must sum to 1.0")
            self._weights = weights

        self._user_id_provider = user_id_provider or StagedRolloutResolver._default_user_id

    def resolve_variant(self, experiment_name: str) -> str:
        """Resolve which variant a user is assigned to.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Variant identifier (e.g., "A", "B", "C")
        """
        # Get stable user identifier
        user_id = self._user_id_provider()

        # Create deterministic hash
        import hashlib

        combined = f"{experiment_name}:{user_id}".encode()
        hash_value = int(hashlib.sha256(combined).hexdigest(), 16)

        # Map hash to variant based on weights
        rollout_value = (hash_value % 10000) / 100.0

        cumulative = 0.0
        for variant, weight in zip(self._variants, self._weights, strict=False):
            cumulative += weight * 100.0
            if rollout_value < cumulative:
                logger.debug(f"User {user_id} assigned to variant {variant} for {experiment_name}")
                return variant

        # Fallback to last variant (shouldn't reach here due to float precision)
        return self._variants[-1]

    def resolve(self, flag_name: str, default: bool = False) -> Optional[bool]:
        """Resolve flag based on variant assignment.

        Args:
            flag_name: Name in format "experiment_name:variant" (e.g., "experiment_1:A")
            default: Default value

        Returns:
            True if user is assigned to specified variant

        Example:
            # Check if user is in variant B
            in_variant_b = resolver.resolve("experiment_1:B")
        """
        if ":" not in flag_name:
            return None

        experiment_name, expected_variant = flag_name.split(":", 1)

        if expected_variant not in self._variants:
            logger.warning(f"Variant {expected_variant} not in {self._variants}")
            return None

        actual_variant = self.resolve_variant(experiment_name)
        return actual_variant == expected_variant

    def set(self, flag_name: str, value: bool) -> bool:
        """Set variant (not supported - use constructor).

        Args:
            flag_name: Not used
            value: Not used

        Returns:
            False (not supported)
        """
        logger.warning("ABTestingResolver does not support set()")
        return False
