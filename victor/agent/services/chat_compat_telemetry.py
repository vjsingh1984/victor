# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Telemetry helpers for deprecated chat compatibility surfaces.

These counters are intentionally process-local and dependency-light so
deprecated compatibility shims can report usage without re-coupling
themselves to evolving runtime services.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import DefaultDict, Dict, Tuple

_LOCK = threading.Lock()
_COUNTS: DefaultDict[Tuple[str, str, str], int] = defaultdict(int)


def record_deprecated_chat_shim_access(component: str, surface: str, route: str) -> None:
    """Record one deprecated chat compatibility access event."""
    key = (component, surface, route)
    with _LOCK:
        _COUNTS[key] += 1


def get_deprecated_chat_shim_telemetry() -> Dict[str, int]:
    """Return a stable flat snapshot of deprecated chat shim access counters."""
    with _LOCK:
        snapshot = dict(_COUNTS)

    telemetry: Dict[str, int] = {}
    total = 0
    for (component, surface, route), count in snapshot.items():
        telemetry[f"{component}.{surface}.{route}"] = count
        total += count
    telemetry["total"] = total
    return telemetry


def reset_deprecated_chat_shim_telemetry() -> None:
    """Reset deprecated chat shim telemetry counters."""
    with _LOCK:
        _COUNTS.clear()


__all__ = [
    "get_deprecated_chat_shim_telemetry",
    "record_deprecated_chat_shim_access",
    "reset_deprecated_chat_shim_telemetry",
]
