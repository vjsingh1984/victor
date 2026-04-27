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
from typing import Any, DefaultDict, Dict, Tuple

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


def get_deprecated_chat_shim_report() -> Dict[str, Any]:
    """Return a structured summary of deprecated chat compatibility usage."""
    telemetry = get_deprecated_chat_shim_telemetry()
    report: Dict[str, Any] = {
        "total": telemetry.get("total", 0),
        "deprecated_surface_count": 0,
        "components": {},
        "route_totals": {},
        "active_routes": [],
        "active_surfaces": [],
    }

    surface_totals: Dict[str, int] = {}

    for key, count in telemetry.items():
        if key == "total":
            continue

        component, surface, route = key.split(".", 2)
        component_entry = report["components"].setdefault(
            component,
            {"total": 0, "surfaces": {}},
        )
        surface_entry = component_entry["surfaces"].setdefault(
            surface,
            {"total": 0, "routes": {}},
        )

        component_entry["total"] += count
        surface_entry["total"] += count
        surface_entry["routes"][route] = count
        report["route_totals"][route] = report["route_totals"].get(route, 0) + count

        surface_key = f"{component}.{surface}"
        surface_totals[surface_key] = surface_totals.get(surface_key, 0) + count

    report["deprecated_surface_count"] = len(surface_totals)
    report["active_routes"] = [
        {"route": route, "count": count}
        for route, count in sorted(
            report["route_totals"].items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    report["active_surfaces"] = [
        {"surface": surface, "count": count}
        for surface, count in sorted(
            surface_totals.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    return report


def has_deprecated_chat_shim_usage() -> bool:
    """Return whether any deprecated chat compatibility surface was used."""
    return get_deprecated_chat_shim_telemetry().get("total", 0) > 0


def reset_deprecated_chat_shim_telemetry() -> None:
    """Reset deprecated chat shim telemetry counters."""
    with _LOCK:
        _COUNTS.clear()


__all__ = [
    "get_deprecated_chat_shim_report",
    "get_deprecated_chat_shim_telemetry",
    "has_deprecated_chat_shim_usage",
    "record_deprecated_chat_shim_access",
    "reset_deprecated_chat_shim_telemetry",
]
