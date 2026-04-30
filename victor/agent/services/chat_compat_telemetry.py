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

"""Telemetry for deprecated chat coordinator shims.

This module provides telemetry functions for tracking access to deprecated
chat coordinator shims during the migration to the new chat service architecture.

DEPRECATED: Chat shims have been removed. This module is retained for
observability but can be removed in a future cleanup.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# In-memory telemetry storage
_telemetry: Dict[str, int] = {"total": 0}


def record_deprecated_chat_shim_access(
    location: str,
    name: str,
    access_type: str,
) -> None:
    """Record access to a deprecated chat coordinator shim.

    Args:
        location: Where the access occurred (e.g., "coordinators_package")
        name: Name of the deprecated shim
        access_type: Type of access (e.g., "package_export", "attribute")
    """
    key = f"{location}.{name}.{access_type}"
    _telemetry[key] = _telemetry.get(key, 0) + 1
    _telemetry["total"] += 1
    logger.debug(f"Deprecated chat shim access: {location} accessed {name} via {access_type}")


def get_deprecated_chat_shim_telemetry() -> Dict[str, int]:
    """Get the raw telemetry data.

    Returns:
        Dict mapping access keys to counts, plus a "total" key.
    """
    return dict(_telemetry)


def reset_deprecated_chat_shim_telemetry() -> None:
    """Reset all telemetry data."""
    global _telemetry
    _telemetry = {"total": 0}


def has_deprecated_chat_shim_usage() -> bool:
    """Check if any deprecated chat shims have been accessed.

    Returns:
        True if any telemetry has been recorded.
    """
    return _telemetry.get("total", 0) > 0


def get_deprecated_chat_shim_report() -> Dict[str, Any]:
    """Get a structured report of deprecated chat shim usage.

    Returns:
        Structured report with grouped telemetry data.
    """
    total = _telemetry.get("total", 0)

    # Group by component, surface, and route
    components: Dict[str, Dict[str, Any]] = {}
    route_totals: Dict[str, int] = {}
    surfaces: Dict[str, int] = {}

    for key, count in _telemetry.items():
        if key == "total":
            continue

        parts = key.split(".")
        if len(parts) >= 3:
            component, surface, route = parts[0], parts[1], parts[2]

            # Track components
            if component not in components:
                components[component] = {"total": 0, "surfaces": {}}
            components[component]["total"] += count

            # Track surfaces within components
            if surface not in components[component]["surfaces"]:
                components[component]["surfaces"][surface] = {"total": 0, "routes": {}}
            components[component]["surfaces"][surface]["total"] += count

            # Track routes within surfaces
            if route not in components[component]["surfaces"][surface]["routes"]:
                components[component]["surfaces"][surface]["routes"][route] = 0
            components[component]["surfaces"][surface]["routes"][route] += count

            # Track route totals
            route_totals[route] = route_totals.get(route, 0) + count

            # Track surface names
            surface_key = f"{component}.{surface}"
            surfaces[surface_key] = surfaces.get(surface_key, 0) + count

    # Build active components list
    active_components = [
        {"component": k, "count": v["total"]}
        for k, v in sorted(components.items(), key=lambda x: -x[1]["total"])
    ]

    # Build active routes list (sorted by count descending, then route name ascending)
    active_routes = [
        {"route": k, "count": v}
        for k, v in sorted(route_totals.items(), key=lambda x: (-x[1], x[0]))
    ]

    # Build active surfaces list (sorted by count descending, then surface name ascending)
    active_surfaces = [
        {"surface": k, "count": v} for k, v in sorted(surfaces.items(), key=lambda x: (-x[1], x[0]))
    ]

    # Build removal candidates (surfaces with low usage)
    removal_candidates = [
        {
            "surface": surface_key,
            "count": count,
            "routes": components[component]["surfaces"][surface]["routes"],
        }
        for component, data in components.items()
        for surface, surface_data in data["surfaces"].items()
        for surface_key, count in [(f"{component}.{surface}", surface_data["total"])]
    ]
    removal_candidates.sort(key=lambda x: x["count"])

    return {
        "total": total,
        "deprecated_surface_count": len(surfaces),
        "components": components,
        "route_totals": route_totals,
        "active_components": active_components,
        "active_routes": active_routes,
        "active_surfaces": active_surfaces,
        "removal_candidates": removal_candidates,
    }


__all__ = [
    "record_deprecated_chat_shim_access",
    "get_deprecated_chat_shim_telemetry",
    "reset_deprecated_chat_shim_telemetry",
    "has_deprecated_chat_shim_usage",
    "get_deprecated_chat_shim_report",
]
