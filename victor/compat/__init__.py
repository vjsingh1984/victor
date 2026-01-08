# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Compatibility registry for legacy imports.

All compatibility aliases and semantic rename registries have been removed.
This module remains as a placeholder for future migrations.
"""

from typing import Any, Dict

# Version for tracking compat changes
COMPAT_VERSION = "0.5.0"

# Registry of all deprecated aliases for runtime warnings
DEPRECATED_ALIASES: Dict[str, Dict[str, Any]] = {}

# Registry of semantic renames (not aliases - different behavior)
SEMANTIC_RENAMES: Dict[str, Dict[str, str]] = {}


# =============================================================================
# MODULE-LEVEL DUPLICATIONS (for future cleanup)
# =============================================================================
# These are full module duplications that should be consolidated in a future sprint.
# One module should be canonical, others should import from it.

MODULE_DUPLICATIONS = {}


def get_canonical_source(alias_path: str) -> str:
    """Get the canonical source for a deprecated alias.

    Args:
        alias_path: Full import path of the alias

    Returns:
        Canonical import path, or original if not deprecated
    """
    info = DEPRECATED_ALIASES.get(alias_path)
    if info:
        return info["canonical"]
    return alias_path


def list_all_aliases() -> Dict[str, str]:
    """List all deprecated aliases and their canonical sources."""
    return {path: info["canonical"] for path, info in DEPRECATED_ALIASES.items()}
