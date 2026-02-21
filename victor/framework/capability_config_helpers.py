# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared helpers for capability config service access and fallback behavior."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Optional

from victor.framework.capability_config_service import (
    CapabilityConfigMergePolicy,
    CapabilityConfigService,
    DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY,
)
from victor.framework.protocols import CapabilityConfigScopePortProtocol
from victor.framework.strict_mode import ensure_not_private_fallback


def resolve_capability_config_service(orchestrator: Any) -> Optional[CapabilityConfigService]:
    """Resolve framework CapabilityConfigService via orchestrator container port."""
    get_container = getattr(orchestrator, "get_service_container", None)
    container = (
        get_container() if callable(get_container) else getattr(orchestrator, "container", None)
    )
    if container is None or not hasattr(container, "get_optional"):
        return None

    try:
        service = container.get_optional(CapabilityConfigService)
    except Exception:
        return None

    return service if isinstance(service, CapabilityConfigService) else None


def resolve_capability_config_scope_key(orchestrator: Any) -> str:
    """Resolve capability config scope key using explicit orchestrator port first."""

    def _normalize(scope_key: Any) -> str:
        if scope_key is None:
            return DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY
        normalized = str(scope_key).strip()
        return normalized or DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY

    if isinstance(orchestrator, CapabilityConfigScopePortProtocol):
        try:
            return _normalize(orchestrator.get_capability_config_scope_key())
        except Exception:
            return DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY

    getter = getattr(orchestrator, "get_capability_config_scope_key", None)
    if callable(getter):
        try:
            return _normalize(getter())
        except Exception:
            return DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY

    for attr_name in ("capability_config_scope_key", "active_session_id", "session_id"):
        attr_value = getattr(orchestrator, attr_name, None)
        if attr_value:
            return _normalize(attr_value)

    return DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY


def load_capability_config(
    orchestrator: Any,
    name: str,
    defaults: Dict[str, Any],
    *,
    fallback_attr: Optional[str] = None,
    legacy_service_names: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Load capability config from service-first storage with legacy fallback."""
    default_copy = deepcopy(defaults)
    service = resolve_capability_config_service(orchestrator)
    scope_key = resolve_capability_config_scope_key(orchestrator)
    if service is not None:
        if legacy_service_names:
            if service.has_config(name, scope_key=scope_key):
                return service.get_config(name, default_copy, scope_key=scope_key)
            for legacy_name in legacy_service_names:
                if service.has_config(legacy_name, scope_key=scope_key):
                    return service.get_config(legacy_name, default_copy, scope_key=scope_key)
            return default_copy

        return service.get_config(name, default_copy, scope_key=scope_key)

    target_attr = fallback_attr or name
    ensure_not_private_fallback(target_attr, operation="read")
    return getattr(orchestrator, target_attr, default_copy)


def store_capability_config(
    orchestrator: Any,
    name: str,
    config: Dict[str, Any],
    *,
    fallback_attr: Optional[str] = None,
    require_existing_attr: bool = True,
    merge_policy: CapabilityConfigMergePolicy = CapabilityConfigMergePolicy.REPLACE,
) -> bool:
    """Store capability config in service-first storage with legacy fallback.

    Returns:
        True if written via CapabilityConfigService, False when fallback path is used.
    """
    service = resolve_capability_config_service(orchestrator)
    scope_key = resolve_capability_config_scope_key(orchestrator)
    if service is not None:
        service.set_config(name, config, merge_policy=merge_policy, scope_key=scope_key)
        return True

    target_attr = fallback_attr or name
    ensure_not_private_fallback(target_attr, operation="write")
    if not require_existing_attr or hasattr(orchestrator, target_attr):
        setattr(orchestrator, target_attr, config)
    return False


def update_capability_config_section(
    orchestrator: Any,
    *,
    root_name: str,
    section_name: str,
    section_config: Dict[str, Any],
    root_defaults: Dict[str, Any],
    fallback_attr: Optional[str] = None,
    require_existing_attr: bool = True,
) -> Dict[str, Any]:
    """Merge one section into a grouped capability config."""
    root_config = load_capability_config(
        orchestrator,
        root_name,
        root_defaults,
        fallback_attr=fallback_attr,
    )
    merged = dict(root_config) if isinstance(root_config, dict) else deepcopy(root_defaults)
    merged[section_name] = section_config
    store_capability_config(
        orchestrator,
        root_name,
        merged,
        fallback_attr=fallback_attr,
        require_existing_attr=require_existing_attr,
    )
    return merged


__all__ = [
    "load_capability_config",
    "resolve_capability_config_scope_key",
    "resolve_capability_config_service",
    "store_capability_config",
    "update_capability_config_section",
]
