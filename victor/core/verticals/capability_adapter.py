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

"""Capability adapter for DIP compliance.

This adapter bridges legacy direct orchestrator attribute writes with
new VerticalContext-based config storage, enabling gradual migration
with backward compatibility.

Design Philosophy:
- Two-phase migration: backward compatible → context only
- Deprecation warnings for legacy code paths
- Feature flag control (VICTOR_USE_CONTEXT_CONFIG)
- Zero breaking changes

OLD Pattern (avoid - tight coupling):
    # In victor/coding/capabilities.py
    def configure_code_style(orchestrator, **kwargs):
        if hasattr(orchestrator, "code_style"):  # DIP violation
            orchestrator.code_style = {...}  # Direct mutation

NEW Pattern (preferred - DIP compliant):
    # In victor/coding/capabilities.py
    def configure_code_style(orchestrator, context, **kwargs):
        adapter = get_capability_adapter(context)
        adapter.set_capability_config(
            orchestrator,
            "code_style",
            {...}
        )  # Decoupled via adapter

Migration Path:
    Phase 1 (Current): BACKWARD_COMPATIBLE mode
        - Writes to both context and orchestrator
        - Reads from context, falls back to orchestrator
        - Emits deprecation warnings

    Phase 2 (Future): CONTEXT_ONLY mode
        - Writes only to context
        - Reads only from context
        - No orchestrator attribute access

Usage:
    from victor.core.verticals.capability_adapter import get_capability_adapter

    # In vertical integration code
    adapter = get_capability_adapter(vertical_context)
    adapter.set_capability_config(
        orchestrator,
        "code_style",
        {"formatter": "black", "linter": "ruff"}
    )
"""

import os
import warnings
from enum import Enum
from typing import Any, Dict, Optional, TYPE_CHECKING

from victor.core.verticals.context import VerticalContext

if TYPE_CHECKING:
    pass


# =============================================================================
# Legacy Write Mode Enum
# =============================================================================


class LegacyWriteMode(Enum):
    """Mode for handling legacy orchestrator writes."""

    BACKWARD_COMPATIBLE = "backward_compatible"
    """Write to both context and orchestrator (default)."""

    CONTEXT_ONLY = "context_only"
    """Write only to context, ignore orchestrator."""


# =============================================================================
# Capability Adapter
# =============================================================================


class CapabilityAdapter:
    """Adapter for migrating from direct orchestrator writes to context-based storage.

    This adapter enables gradual migration from DIP-violating direct attribute
    writes to SOLID-compliant context-based storage.

    Attributes:
        context: VerticalContext for storing configs
        legacy_mode: How to handle legacy orchestrator writes
    """

    def __init__(
        self,
        context: VerticalContext,
        legacy_mode: Optional[LegacyWriteMode] = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            context: VerticalContext for config storage
            legacy_mode: How to handle legacy writes (auto-detected from env if None)
        """
        self.context = context

        if legacy_mode is None:
            # Auto-detect from feature flag
            use_context = os.getenv("VICTOR_USE_CONTEXT_CONFIG", "false").lower() == "true"
            legacy_mode = (
                LegacyWriteMode.CONTEXT_ONLY if use_context else LegacyWriteMode.BACKWARD_COMPATIBLE
            )

        self.legacy_mode = legacy_mode

    def set_capability_config(
        self,
        name_or_orchestrator: Any,
        config_or_name: Any = None,
        config: Optional[Any] = None,
    ) -> None:
        """Store a capability configuration.

        Supports two call signatures:
        1. set_capability_config(orchestrator, name, config) - legacy with orchestrator
        2. set_capability_config(name, config) - context-only (recommended)

        In BACKWARD_COMPATIBLE mode:
            - Stores in context (new way)
            - Also stores in orchestrator if provided (legacy way)
            - Emits deprecation warning

        In CONTEXT_ONLY mode:
            - Stores only in context
            - Ignores orchestrator parameter
            - No warning

        Args:
            name_or_orchestrator: Config name (str) or orchestrator (Any)
            config_or_name: Config value (if first arg is name) or name (if first arg is orchestrator)
            config: Config value (if using 3-arg signature)
        """
        # Detect call signature
        if isinstance(name_or_orchestrator, str):
            # 2-arg call: set_capability_config(name, config)
            name = name_or_orchestrator
            orchestrator = None
            cfg_value = config_or_name
        else:
            # 3-arg call: set_capability_config(orchestrator, name, config)
            orchestrator = name_or_orchestrator
            name = config_or_name
            cfg_value = config

        # Always store in context (new way)
        self.context.set_capability_config(name, cfg_value)

        # Legacy handling (only if orchestrator provided)
        if orchestrator and self.legacy_mode == LegacyWriteMode.BACKWARD_COMPATIBLE:
            # Emit deprecation warning
            warnings.warn(
                f"Direct orchestrator attribute '{name}' is deprecated. "
                f"Use context.set_capability_config('{name}', config) instead. "
                f"This will become an error in a future release.",
                DeprecationWarning,
                stacklevel=3,
            )

            # Handle nested config patterns (e.g., "safety_config.git")
            if "." in name:
                # Nested dict update pattern
                base_attr, nested_key = name.split(".", 1)
                if not hasattr(orchestrator, base_attr):
                    setattr(orchestrator, base_attr, {})

                base_dict = getattr(orchestrator, base_attr)
                if isinstance(base_dict, dict):
                    base_dict[nested_key] = cfg_value
                else:
                    # Fall back to direct attribute set
                    setattr(orchestrator, name, cfg_value)
            else:
                # Direct attribute set
                setattr(orchestrator, name, cfg_value)

    def get_capability_config(
        self,
        orchestrator_or_name: Any,
        name_or_default: Any = None,
        default: Optional[Any] = None,
    ) -> Any:
        """Retrieve a capability configuration.

        Supports two call signatures:
        1. get_capability_config(orchestrator, name, default=None) - legacy with orchestrator
        2. get_capability_config(name, default=None) - context-only (recommended)

        In BACKWARD_COMPATIBLE mode:
            - Checks context first (new way)
            - Falls back to orchestrator if provided (legacy way)
            - Emits deprecation warning if using orchestrator

        In CONTEXT_ONLY mode:
            - Checks only context
            - No orchestrator access

        Args:
            orchestrator_or_name: Orchestrator (Any) or config name (str)
            name_or_default: Config name (if first arg is orchestrator) or default value
            default: Default value (if using 3-arg signature)

        Returns:
            Configuration value or default
        """
        # Detect call signature
        if isinstance(orchestrator_or_name, str):
            # 2-arg call: get_capability_config(name, default)
            name = orchestrator_or_name
            orchestrator = None
            default_value = name_or_default if default is None else default
        else:
            # 3-arg call: get_capability_config(orchestrator, name, default)
            orchestrator = orchestrator_or_name
            name = name_or_default
            default_value = default if default is not None else None

        # Check context first (new way)
        value = self.context.get_capability_config(name)

        if value is not None:
            return value

        # Legacy fallback (only if orchestrator provided)
        if orchestrator and self.legacy_mode == LegacyWriteMode.BACKWARD_COMPATIBLE:
            # Emit deprecation warning
            warnings.warn(
                f"Accessing orchestrator attribute '{name}' is deprecated. "
                f"Use context.get_capability_config('{name}') instead. "
                f"This will become an error in a future release.",
                DeprecationWarning,
                stacklevel=3,
            )

            # Check orchestrator
            if hasattr(orchestrator, name):
                return getattr(orchestrator, name)

            # Handle nested patterns
            if "." in name:
                base_attr, nested_key = name.split(".", 1)
                if hasattr(orchestrator, base_attr):
                    base_dict = getattr(orchestrator, base_attr)
                    if isinstance(base_dict, dict):
                        return base_dict.get(nested_key, default_value)

        return default_value

    def apply_capability_configs(
        self,
        orchestrator: Any,
        configs: Dict[str, Any],
    ) -> None:
        """Apply multiple capability configurations at once.

        Args:
            orchestrator: Target orchestrator
            configs: Dict mapping config names to values
        """
        for name, config in configs.items():
            self.set_capability_config(orchestrator, name, config)

    def migrate_from_orchestrator(
        self,
        orchestrator: Any,
        config_names: Optional[list[str]] = None,
    ) -> int:
        """Migrate configs from orchestrator to context.

        Reads configs from orchestrator attributes and stores them
        in context, then optionally clears orchestrator attributes.

        Args:
            orchestrator: Source orchestrator
            config_names: List of config names to migrate (auto-detect if None)

        Returns:
            Number of configs migrated
        """
        # Known config names from verticals
        known_configs = [
            "safety_config",
            "code_style",
            "test_config",
            "lsp_config",
            "refactor_config",
            "rag_config",
            "source_verification_config",
        ]

        if config_names is None:
            config_names = known_configs

        migrated = 0

        for name in config_names:
            if hasattr(orchestrator, name):
                value = getattr(orchestrator, name)
                if value is not None:
                    self.context.set_capability_config(name, value)
                    migrated += 1

        return migrated

    def clear_orchestrator_configs(
        self,
        orchestrator: Any,
        config_names: list[str],
    ) -> None:
        """Clear migrated configs from orchestrator.

        After migration, this can be used to clean up orchestrator attributes.

        Args:
            orchestrator: Target orchestrator
            config_names: List of config names to clear
        """
        for name in config_names:
            if hasattr(orchestrator, name):
                delattr(orchestrator, name)


# =============================================================================
# Factory Functions
# =============================================================================


_adapter_cache: Dict[int, CapabilityAdapter] = {}


def get_capability_adapter(context: VerticalContext) -> CapabilityAdapter:
    """Get or create a CapabilityAdapter for a context.

    Args:
        context: VerticalContext instance

    Returns:
        Cached CapabilityAdapter instance
    """
    # Use id(context) as cache key
    cache_key = id(context)

    if cache_key not in _adapter_cache:
        _adapter_cache[cache_key] = CapabilityAdapter(context)

    return _adapter_cache[cache_key]


def clear_adapter_cache() -> None:
    """Clear the adapter cache (mainly for testing)."""
    global _adapter_cache
    _adapter_cache = {}


__all__ = [
    "CapabilityAdapter",
    "LegacyWriteMode",
    "get_capability_adapter",
    "clear_adapter_cache",
]
