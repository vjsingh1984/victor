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
)


def resolve_capability_config_service(orchestrator: Any) -> Optional[CapabilityConfigService]:
    """Resolve framework CapabilityConfigService via orchestrator container port."""
    get_container = getattr(orchestrator, "get_service_container", None)
    container = get_container() if callable(get_container) else getattr(orchestrator, "container", None)
    if container is None or not hasattr(container, "get_optional"):
        return None

    try:
        service = container.get_optional(CapabilityConfigService)
    except Exception:
        return None

    return service if isinstance(service, CapabilityConfigService) else None


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
    if service is not None:
        if legacy_service_names:
            if service.has_config(name):
                return service.get_config(name, default_copy)
            for legacy_name in legacy_service_names:
                if service.has_config(legacy_name):
                    return service.get_config(legacy_name, default_copy)
            return default_copy

        return service.get_config(name, default_copy)

    target_attr = fallback_attr or name
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
    if service is not None:
        service.set_config(name, config, merge_policy=merge_policy)
        return True

    target_attr = fallback_attr or name
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
    "resolve_capability_config_service",
    "store_capability_config",
    "update_capability_config_section",
]

