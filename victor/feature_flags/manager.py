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

"""Feature flag manager for gradual rollout of new capabilities.

This module provides the FeatureFlagManager class which manages feature flag
resolution, validation, runtime updates, and audit logging.

Key Features:
- Flag resolution with priority-based lookup (env -> settings -> runtime -> default)
- Runtime flag updates (hot-reload without restart)
- Flag validation and dependency checking
- Audit logging for all flag changes
- Integration with existing observability infrastructure
- Support for staged rollout and A/B testing

Example:
    from victor.feature_flags import get_feature_flag_manager

    manager = get_feature_flag_manager()

    # Check if feature is enabled
    if manager.is_enabled("hierarchical_planning_enabled"):
        # Use hierarchical planning
        pass

    # Set flag at runtime
    manager.set_flag("enhanced_memory_enabled", True)

    # Get all flags
    flags = manager.get_all_flags()
    for flag_name, flag_value in flags.items():
        print(f"{flag_name}: {flag_value}")
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from victor.feature_flags.flags import (
    FEATURE_FLAGS,
    get_flag_metadata,
    validate_flag_dependencies,
    get_all_flag_names,
    get_flags_by_category,
    get_stable_flags,
    get_experimental_flags,
)
from victor.feature_flags.resolvers import (
    ChainedFlagResolver,
    EnvironmentFlagResolver,
    SettingsFlagResolver,
    RuntimeFlagResolver,
    StagedRolloutResolver,
)

logger = logging.getLogger(__name__)


# Global singleton instance
_manager_instance: Optional[FeatureFlagManager] = None
_manager_lock = threading.Lock()


class FlagChangeAuditLog:
    """Audit log entry for flag changes."""

    def __init__(
        self,
        flag_name: str,
        old_value: Optional[bool],
        new_value: bool,
        source: str,
        timestamp: datetime,
    ) -> None:
        """Initialize audit log entry.

        Args:
            flag_name: Name of the flag
            old_value: Previous value (None if newly set)
            new_value: New value
            source: Source of change (env, settings, runtime, api)
            timestamp: When the change occurred
        """
        self.flag_name = flag_name
        self.old_value = old_value
        self.new_value = new_value
        self.source = source
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "flag_name": self.flag_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }


class FeatureFlagManager:
    """Manager for feature flags with resolution, validation, and audit logging.

    Provides centralized feature flag management with:
    - Multi-source resolution (environment, settings, runtime, default)
    - Dependency validation
    - Runtime updates (hot-reload)
    - Audit logging
    - Integration with observability

    Design Patterns:
    - Singleton: Single manager instance per process
    - Strategy: Pluggable resolvers for different sources
    - Observer: Callbacks for flag changes

    Example:
        manager = FeatureFlagManager()

        # Check if flag is enabled
        if manager.is_enabled("hierarchical_planning_enabled"):
            pass

        # Set flag at runtime
        manager.set_flag("enhanced_memory_enabled", True)

        # Register callback for flag changes
        manager.on_flag_changed("my_feature", lambda name, value: print(f"{name} = {value}"))
    """

    def __init__(
        self,
        resolver: Optional[ChainedFlagResolver] = None,
        enable_audit_logging: bool = True,
    ) -> None:
        """Initialize feature flag manager.

        Args:
            resolver: Custom flag resolver (default: env -> settings -> runtime)
            enable_audit_logging: Whether to log all flag changes
        """
        self._resolver = resolver or ChainedFlagResolver()
        self._enable_audit_logging = enable_audit_logging

        # Audit log (thread-safe)
        self._audit_log: List[FlagChangeAuditLog] = []
        self._audit_lock = threading.RLock()

        # Change callbacks
        self._callbacks: Dict[str, List[Callable[[str, bool], None]]] = {}
        self._callbacks_lock = threading.RLock()

        # Cache for resolved values
        self._cache: Dict[str, bool] = {}
        self._cache_lock = threading.RLock()

        logger.info("FeatureFlagManager initialized")

    def is_enabled(self, flag_name: str, use_cache: bool = True) -> bool:
        """Check if a feature flag is enabled.

        Args:
            flag_name: Name of the flag
            use_cache: Whether to use cached value (default: True)

        Returns:
            True if flag is enabled

        Raises:
            ValueError: If flag name is not recognized

        Example:
            if manager.is_enabled("hierarchical_planning_enabled"):
                use_hierarchical_planning()
        """
        # Validate flag name
        if flag_name not in FEATURE_FLAGS:
            raise ValueError(
                f"Unknown feature flag: {flag_name}. "
                f"Valid flags: {', '.join(get_all_flag_names())}"
            )

        # Check cache first
        if use_cache:
            with self._cache_lock:
                if flag_name in self._cache:
                    return self._cache[flag_name]

        # Get default value from flag definition
        flag_def = FEATURE_FLAGS[flag_name]
        default_value = flag_def.get("default", False)

        # Resolve flag value
        value = self._resolver.resolve(flag_name, default=default_value)

        # Ensure boolean
        if value is None:
            value = default_value

        # Validate dependencies
        if value:
            if not validate_flag_dependencies(flag_name, self.get_all_flags()):
                logger.warning(
                    f"Flag {flag_name} is enabled but dependencies are not satisfied. Disabling."
                )
                value = False

        # Cache the result
        if use_cache:
            with self._cache_lock:
                self._cache[flag_name] = value

        return bool(value)

    def set_flag(self, flag_name: str, value: bool, source: str = "api") -> bool:
        """Set a feature flag value at runtime.

        Args:
            flag_name: Name of the flag
            value: Value to set
            source: Source of change (for audit logging)

        Returns:
            True if flag was set successfully

        Raises:
            ValueError: If flag name is not recognized or dependencies not satisfied

        Example:
            manager.set_flag("enhanced_memory_enabled", True)
        """
        # Validate flag name
        if flag_name not in FEATURE_FLAGS:
            raise ValueError(
                f"Unknown feature flag: {flag_name}. "
                f"Valid flags: {', '.join(get_all_flag_names())}"
            )

        # Check dependencies before enabling
        if value and not validate_flag_dependencies(flag_name, self.get_all_flags()):
            missing_deps = [
                dep
                for dep in FEATURE_FLAGS[flag_name].get("dependencies", [])
                if not self.is_enabled(dep)
            ]
            raise ValueError(
                f"Cannot enable {flag_name}: dependencies not satisfied: {', '.join(missing_deps)}"
            )

        # Get old value for audit log
        old_value = None
        try:
            old_value = self._resolver.resolve(flag_name)
        except Exception:
            pass

        # Set flag via resolver
        if isinstance(self._resolver, ChainedFlagResolver):
            success = self._resolver.set(flag_name, value)
        else:
            success = self._resolver.set(flag_name, value)

        if not success:
            logger.warning(f"Failed to set flag {flag_name}={value}")
            return False

        # Update cache
        with self._cache_lock:
            self._cache[flag_name] = value

        # Audit log
        if self._enable_audit_logging:
            self._audit_log.append(
                FlagChangeAuditLog(
                    flag_name=flag_name,
                    old_value=old_value,
                    new_value=value,
                    source=source,
                    timestamp=datetime.now(timezone.utc),
                )
            )

        # Trigger callbacks
        self._trigger_callbacks(flag_name, value)

        logger.info(f"Feature flag {flag_name} set to {value} (source: {source})")
        return True

    def get_all_flags(self, include_disabled: bool = True) -> Dict[str, bool]:
        """Get all feature flag values.

        Args:
            include_disabled: Whether to include disabled flags (default: True)

        Returns:
            Dictionary of flag names to values

        Example:
            flags = manager.get_all_flags()
            for flag_name, flag_value in flags.items():
                print(f"{flag_name}: {flag_value}")
        """
        flags = {}

        for flag_name in get_all_flag_names():
            try:
                value = self.is_enabled(flag_name)
                if include_disabled or value:
                    flags[flag_name] = value
            except Exception as e:
                logger.warning(f"Failed to resolve flag {flag_name}: {e}")

        return flags

    def get_flag_metadata(self, flag_name: str) -> Dict[str, Any]:
        """Get metadata for a feature flag.

        Args:
            flag_name: Name of the flag

        Returns:
            Metadata dictionary

        Example:
            metadata = manager.get_flag_metadata("hierarchical_planning_enabled")
            print(metadata["description"])
        """
        return get_flag_metadata(flag_name)

    def get_flags_by_category(self, category: str) -> Dict[str, bool]:
        """Get all flags in a specific category.

        Args:
            category: Category name (planning, memory, skills, multimodal, personas, performance)

        Returns:
            Dictionary of flag names to values

        Example:
            planning_flags = manager.get_flags_by_category("planning")
        """
        flag_names = get_flags_by_category(category).keys()
        return {name: self.is_enabled(name) for name in flag_names}

    def get_stable_flags(self) -> Dict[str, bool]:
        """Get all flags marked as stable (safe for production).

        Returns:
            Dictionary of stable flag names to values

        Example:
            stable = manager.get_stable_flags()
        """
        flag_names = get_stable_flags().keys()
        return {name: self.is_enabled(name) for name in flag_names}

    def get_experimental_flags(self) -> Dict[str, bool]:
        """Get all flags marked as experimental (not stable).

        Returns:
            Dictionary of experimental flag names to values

        Example:
            experimental = manager.get_experimental_flags()
        """
        flag_names = get_experimental_flags().keys()
        return {name: self.is_enabled(name) for name in flag_names}

    def reset_flag(self, flag_name: str) -> bool:
        """Reset a flag to its default value.

        Args:
            flag_name: Name of the flag

        Returns:
            True if flag was reset successfully

        Example:
            manager.reset_flag("hierarchical_planning_enabled")
        """
        flag_def = FEATURE_FLAGS.get(flag_name)
        if not flag_def:
            return False

        default_value = flag_def.get("default", False)
        return self.set_flag(flag_name, default_value, source="reset")

    def clear_cache(self) -> None:
        """Clear the flag value cache.

        Forces next is_enabled() call to re-resolve from sources.

        Example:
            manager.clear_cache()
        """
        with self._cache_lock:
            self._cache.clear()
        logger.info("Feature flag cache cleared")

    def on_flag_changed(self, flag_name: str, callback: Callable[[str, bool], None]) -> None:
        """Register a callback for flag changes.

        Args:
            flag_name: Name of the flag (or "*" for all flags)
            callback: Function to call when flag changes (receives flag_name, value)

        Example:
            manager.on_flag_changed("my_feature", lambda name, value: print(f"{name} = {value}"))
        """
        with self._callbacks_lock:
            if flag_name not in self._callbacks:
                self._callbacks[flag_name] = []
            self._callbacks[flag_name].append(callback)

    def _trigger_callbacks(self, flag_name: str, value: bool) -> None:
        """Trigger callbacks for flag change.

        Args:
            flag_name: Name of the flag
            value: New value
        """
        with self._callbacks_lock:
            # Trigger specific flag callbacks
            if flag_name in self._callbacks:
                for callback in self._callbacks[flag_name]:
                    try:
                        callback(flag_name, value)
                    except Exception as e:
                        logger.error(f"Flag change callback failed for {flag_name}: {e}")

            # Trigger wildcard callbacks
            if "*" in self._callbacks:
                for callback in self._callbacks["*"]:
                    try:
                        callback(flag_name, value)
                    except Exception as e:
                        logger.error(f"Wildcard flag change callback failed for {flag_name}: {e}")

    def get_audit_log(
        self, flag_name: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit log entries for flag changes.

        Args:
            flag_name: Optional flag name to filter by (None = all flags)
            limit: Maximum number of entries to return

        Returns:
            List of audit log entries

        Example:
            log = manager.get_audit_log("hierarchical_planning_enabled")
            for entry in log:
                print(f"{entry['timestamp']}: {entry['flag_name']} = {entry['new_value']}")
        """
        with self._audit_lock:
            if flag_name:
                entries = [e for e in self._audit_log if e.flag_name == flag_name]
            else:
                entries = self._audit_log[:]

        # Sort by timestamp (newest first) and limit
        entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)[:limit]

        return [e.to_dict() for e in entries]

    def clear_audit_log(self) -> None:
        """Clear the audit log.

        Example:
            manager.clear_audit_log()
        """
        with self._audit_lock:
            self._audit_log.clear()
        logger.info("Feature flag audit log cleared")

    def export_state(self) -> Dict[str, Any]:
        """Export current state of all flags.

        Returns:
            Dictionary with flag values and metadata

        Example:
            state = manager.export_state()
            import json
            print(json.dumps(state, indent=2))
        """
        return {
            "flags": self.get_all_flags(),
            "metadata": {name: self.get_flag_metadata(name) for name in get_all_flag_names()},
            "audit_log_size": len(self._audit_log),
            "cache_size": len(self._cache),
        }

    def import_state(self, state: Dict[str, Any], merge: bool = True) -> None:
        """Import flag state from dictionary.

        Args:
            state: State dictionary from export_state()
            merge: Whether to merge with existing state (True) or replace (False)

        Example:
            state = manager.export_state()
            manager.import_state(state)
        """
        flags = state.get("flags", {})

        for flag_name, value in flags.items():
            try:
                if flag_name in FEATURE_FLAGS:
                    self.set_flag(flag_name, value, source="import")
            except Exception as e:
                logger.warning(f"Failed to import flag {flag_name}: {e}")

        logger.info(f"Imported {len(flags)} flags (merge={merge})")


def get_feature_flag_manager() -> FeatureFlagManager:
    """Get singleton feature flag manager instance.

    Returns:
        FeatureFlagManager instance

    Example:
        manager = get_feature_flag_manager()
        if manager.is_enabled("hierarchical_planning_enabled"):
            pass
    """
    global _manager_instance

    if _manager_instance is None:
        with _manager_lock:
            if _manager_instance is None:
                _manager_instance = FeatureFlagManager()
                logger.info("Created singleton FeatureFlagManager instance")

    return _manager_instance


def reset_feature_flag_manager() -> None:
    """Reset the singleton feature flag manager.

    Primarily useful for testing when you need a clean manager instance.

    Example:
        reset_feature_flag_manager()
        manager = get_feature_flag_manager()  # Fresh instance
    """
    global _manager_instance

    with _manager_lock:
        _manager_instance = None

    logger.info("Reset singleton FeatureFlagManager instance")
