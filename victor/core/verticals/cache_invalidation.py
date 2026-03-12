"""Host-owned cache invalidation helpers for vertical package changes."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

from victor.core.verticals.base import VerticalBase, VerticalRegistry
from victor.core.verticals.vertical_loader import get_vertical_loader

logger = logging.getLogger(__name__)


class VerticalRuntimeInvalidationReason(str, Enum):
    """Reasons for invalidating vertical runtime/discovery caches."""

    INSTALL = "install"
    UNINSTALL = "uninstall"
    UPGRADE = "upgrade"
    RELOAD = "reload"


@dataclass(frozen=True)
class VerticalRuntimeInvalidationResult:
    """Result of a vertical runtime cache invalidation pass."""

    reason: VerticalRuntimeInvalidationReason
    package_name: str | None = None
    config_cache_cleared: bool = False
    registry_reset: bool = False
    loader_refreshed: bool = False
    errors: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def successful(self) -> bool:
        """Return True when all invalidation hooks completed without error."""

        return not self.errors


def invalidate_vertical_runtime_state(
    reason: VerticalRuntimeInvalidationReason | str,
    *,
    package_name: str | None = None,
) -> VerticalRuntimeInvalidationResult:
    """Invalidate runtime/discovery caches after package topology changes.

    This is intentionally host-owned. It clears class-level vertical config
    caches, resets registry discovery state, and refreshes the shared
    `VerticalLoader` plugin caches so the current process can observe package
    changes without waiting for TTL expiry.
    """

    normalized_reason = VerticalRuntimeInvalidationReason(reason)
    errors: list[str] = []
    config_cache_cleared = False
    registry_reset = False
    loader_refreshed = False

    try:
        VerticalBase.clear_config_cache(clear_all=True)
        config_cache_cleared = True
    except Exception as exc:
        errors.append(f"config_cache:{exc}")
        logger.warning("Failed to clear vertical config cache: %s", exc)

    try:
        VerticalRegistry.reset_discovery()
        registry_reset = True
    except Exception as exc:
        errors.append(f"registry_reset:{exc}")
        logger.warning("Failed to reset vertical discovery state: %s", exc)

    try:
        get_vertical_loader().refresh_plugins()
        loader_refreshed = True
    except Exception as exc:
        errors.append(f"loader_refresh:{exc}")
        logger.warning("Failed to refresh vertical loader plugins: %s", exc)

    result = VerticalRuntimeInvalidationResult(
        reason=normalized_reason,
        package_name=package_name,
        config_cache_cleared=config_cache_cleared,
        registry_reset=registry_reset,
        loader_refreshed=loader_refreshed,
        errors=tuple(errors),
    )

    logger.info(
        "Vertical runtime invalidation complete",
        extra={
            "reason": result.reason.value,
            "package_name": result.package_name,
            "config_cache_cleared": result.config_cache_cleared,
            "registry_reset": result.registry_reset,
            "loader_refreshed": result.loader_refreshed,
            "errors": list(result.errors),
        },
    )
    return result
