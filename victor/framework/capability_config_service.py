# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Framework-level storage for vertical capability configuration state."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY = "__global__"


class CapabilityConfigMergePolicy(str, Enum):
    """Merge behavior when writing capability configuration."""

    REPLACE = "replace"
    SHALLOW_MERGE = "shallow_merge"


class CapabilityConfigService:
    """Centralized runtime store for capability configuration."""

    def __init__(self) -> None:
        self._configs_by_scope: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def normalize_scope_key(scope_key: Optional[str]) -> str:
        """Normalize an optional scope key to a stable non-empty string."""
        if scope_key is None:
            return DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY
        normalized = str(scope_key).strip()
        return normalized or DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY

    def _get_scope_bucket(self, scope_key: Optional[str], *, create: bool = False) -> Dict[str, Any]:
        """Get the config bucket for one scope."""
        normalized_scope_key = self.normalize_scope_key(scope_key)
        if create:
            return self._configs_by_scope.setdefault(normalized_scope_key, {})
        return self._configs_by_scope.get(normalized_scope_key, {})

    def has_config(self, name: str, *, scope_key: Optional[str] = None) -> bool:
        """Return True if a config key exists."""
        return name in self._get_scope_bucket(scope_key)

    def get_config(self, name: str, default: Any = None, *, scope_key: Optional[str] = None) -> Any:
        """Get config value by name."""
        return self._get_scope_bucket(scope_key).get(name, default)

    def set_config(
        self,
        name: str,
        config: Any,
        *,
        merge_policy: CapabilityConfigMergePolicy = CapabilityConfigMergePolicy.REPLACE,
        scope_key: Optional[str] = None,
    ) -> Any:
        """Set config by name using the provided merge policy."""
        bucket = self._get_scope_bucket(scope_key, create=True)
        if (
            merge_policy == CapabilityConfigMergePolicy.SHALLOW_MERGE
            and isinstance(bucket.get(name), dict)
            and isinstance(config, dict)
        ):
            merged = dict(bucket[name])
            merged.update(config)
            bucket[name] = merged
            return merged

        bucket[name] = config
        return config

    def apply_configs(
        self,
        configs: Dict[str, Any],
        *,
        merge_policy: CapabilityConfigMergePolicy = CapabilityConfigMergePolicy.REPLACE,
        scope_key: Optional[str] = None,
    ) -> None:
        """Apply multiple configs with a uniform merge policy."""
        for name, config in configs.items():
            self.set_config(name, config, merge_policy=merge_policy, scope_key=scope_key)

    def list_names(self, *, scope_key: Optional[str] = None) -> list[str]:
        """List known config names."""
        return sorted(self._get_scope_bucket(scope_key).keys())

    def clear(
        self,
        name: Optional[str] = None,
        *,
        scope_key: Optional[str] = None,
        clear_all_scopes: bool = False,
    ) -> None:
        """Clear configs by key/scope.

        Default behavior clears only the selected scope (global scope when omitted).
        Set ``clear_all_scopes=True`` to wipe every scope.
        """
        if clear_all_scopes:
            self._configs_by_scope.clear()
            return

        normalized_scope_key = self.normalize_scope_key(scope_key)
        if name is None:
            self._configs_by_scope.pop(normalized_scope_key, None)
            return
        bucket = self._configs_by_scope.get(normalized_scope_key)
        if bucket is None:
            return
        bucket.pop(name, None)
        if not bucket:
            self._configs_by_scope.pop(normalized_scope_key, None)

    def snapshot(self, *, scope_key: Optional[str] = None) -> Dict[str, Any]:
        """Return a shallow copy of stored configs for one scope."""
        return dict(self._get_scope_bucket(scope_key))

    def snapshot_all_scopes(self) -> Dict[str, Dict[str, Any]]:
        """Return a shallow copy of all scope buckets for diagnostics."""
        return {scope: dict(configs) for scope, configs in self._configs_by_scope.items()}


__all__ = [
    "CapabilityConfigMergePolicy",
    "CapabilityConfigService",
    "DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY",
]
