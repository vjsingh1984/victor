"""Small DI resolution helpers for compatibility-only optional services."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from victor.core.container import get_container


def resolve_optional_service(
    service_type: Any = None,
    *,
    legacy_key: Optional[str] = None,
    required_attrs: Iterable[str] = (),
) -> Optional[Any]:
    """Resolve an optional service from the global composition root.

    This helper keeps compatibility lookups out of runtime business modules while
    legacy callers finish migrating to constructor injection or ExecutionContext.
    """
    try:
        container = get_container()
    except Exception:
        return None

    get_optional = getattr(container, "get_optional", None)
    get = getattr(container, "get", None)

    candidates: list[Any] = []
    if legacy_key is not None and callable(get):
        try:
            candidates.append(get(legacy_key))
        except Exception:
            pass

    if service_type is not None and callable(get_optional):
        try:
            candidates.append(get_optional(service_type))
        except Exception:
            pass

    if service_type is not None and callable(get):
        try:
            candidates.append(get(service_type))
        except Exception:
            pass

    for service in candidates:
        if service is None:
            continue
        if all(hasattr(service, attr) for attr in required_attrs):
            return service
    return None
