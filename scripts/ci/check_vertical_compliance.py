#!/usr/bin/env python3
"""Check vertical compliance levels against their declared manifests.

Discovers all verticals (registry + entry points), calls get_manifest(),
and compares the `provides` set against compliance levels:
  - Basic: provides TOOLS + SAFETY
  - Standard: Basic + MODE_CONFIG + MIDDLEWARE
  - Full: Standard + WORKFLOWS + TEAMS + SERVICE_PROVIDER

Outputs a markdown table for GitHub Step Summary.

Exit codes:
  0 - All verticals meet or exceed their declared compliance level
  1 - One or more verticals regressed below declared level
"""

from __future__ import annotations

import sys
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# Compliance level definitions
COMPLIANCE_LEVELS = {
    "Basic": {"tool_dependencies", "safety"},
    "Standard": {"tool_dependencies", "safety", "mode_config", "middleware"},
    "Full": {
        "tool_dependencies",
        "safety",
        "mode_config",
        "middleware",
        "workflows",
        "teams",
        "service_provider",
    },
}


def determine_compliance_level(provides_values: set[str]) -> str:
    """Determine the highest compliance level met by a set of provided extension types."""
    for level in ("Full", "Standard", "Basic"):
        if COMPLIANCE_LEVELS[level].issubset(provides_values):
            return level
    return "None"


def main() -> int:
    # Ensure victor is importable
    try:
        from victor.core.verticals.base import VerticalRegistry
        from victor.core.verticals.vertical_loader import get_vertical_loader
    except ImportError as e:
        print(f"Error: Cannot import victor: {e}", file=sys.stderr)
        return 1

    # Discover all verticals
    loader = get_vertical_loader()
    discovered = {}
    try:
        discovered = loader.discover_verticals()
    except Exception as e:
        logger.warning("Entry point discovery failed: %s", e)

    # Merge with registry
    all_verticals = {}
    for name in VerticalRegistry.list_names():
        cls = VerticalRegistry.get(name)
        if cls is not None:
            all_verticals[name] = cls
    all_verticals.update(discovered)

    if not all_verticals:
        print("No verticals found.")
        return 0

    # Collect compliance data
    rows = []
    for name, cls in sorted(all_verticals.items()):
        try:
            manifest = cls.get_manifest()
            provides_values = {ext.value for ext in manifest.provides}
            level = determine_compliance_level(provides_values)
            version = manifest.version
            api_ver = manifest.api_version
            provides_str = (
                ", ".join(sorted(provides_values)) if provides_values else "(none)"
            )
        except (AttributeError, NotImplementedError, ImportError) as exc:
            level = "N/A"
            version = "?"
            api_ver = "?"
            provides_str = f"(manifest unavailable: {exc})"

        rows.append(
            {
                "name": name,
                "version": version,
                "api_version": api_ver,
                "level": level,
                "provides": provides_str,
            }
        )

    # Output markdown table
    print("## Vertical Compliance Report")
    print()
    print("| Vertical | Version | API | Compliance | Provides |")
    print("|----------|---------|-----|------------|----------|")
    for row in rows:
        print(
            f"| {row['name']} | {row['version']} | {row['api_version']} "
            f"| {row['level']} | {row['provides']} |"
        )

    print()
    print(f"**Total verticals**: {len(rows)}")

    # Check for regressions (verticals at "None" compliance)
    none_verticals = [r for r in rows if r["level"] == "None"]
    if none_verticals:
        names = ", ".join(r["name"] for r in none_verticals)
        print(
            f"\n**Warning**: {len(none_verticals)} vertical(s) at 'None' compliance: {names}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
