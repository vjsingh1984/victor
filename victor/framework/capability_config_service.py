# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Framework-level storage for vertical capability configuration state."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class CapabilityConfigMergePolicy(str, Enum):
    """Merge behavior when writing capability configuration."""

    REPLACE = "replace"
    SHALLOW_MERGE = "shallow_merge"


class CapabilityConfigService:
    """Centralized runtime store for capability configuration."""

    def __init__(self) -> None:
        self._configs: Dict[str, Any] = {}

    def has_config(self, name: str) -> bool:
        """Return True if a config key exists."""
        return name in self._configs

    def get_config(self, name: str, default: Any = None) -> Any:
        """Get config value by name."""
        return self._configs.get(name, default)

    def set_config(
        self,
        name: str,
        config: Any,
        *,
        merge_policy: CapabilityConfigMergePolicy = CapabilityConfigMergePolicy.REPLACE,
    ) -> Any:
        """Set config by name using the provided merge policy."""
        if (
            merge_policy == CapabilityConfigMergePolicy.SHALLOW_MERGE
            and isinstance(self._configs.get(name), dict)
            and isinstance(config, dict)
        ):
            merged = dict(self._configs[name])
            merged.update(config)
            self._configs[name] = merged
            return merged

        self._configs[name] = config
        return config

    def apply_configs(
        self,
        configs: Dict[str, Any],
        *,
        merge_policy: CapabilityConfigMergePolicy = CapabilityConfigMergePolicy.REPLACE,
    ) -> None:
        """Apply multiple configs with a uniform merge policy."""
        for name, config in configs.items():
            self.set_config(name, config, merge_policy=merge_policy)

    def list_names(self) -> list[str]:
        """List known config names."""
        return sorted(self._configs.keys())

    def clear(self, name: Optional[str] = None) -> None:
        """Clear one config by name, or all configs if name is None."""
        if name is None:
            self._configs.clear()
            return
        self._configs.pop(name, None)

    def snapshot(self) -> Dict[str, Any]:
        """Return a shallow copy of all stored configs."""
        return dict(self._configs)


__all__ = [
    "CapabilityConfigMergePolicy",
    "CapabilityConfigService",
]

