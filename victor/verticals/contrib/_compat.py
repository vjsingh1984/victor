"""Compatibility helpers for contrib vertical deprecation.

Provides utilities for contrib plugins to detect when their external
replacement package is installed and skip registration accordingly.
"""

from __future__ import annotations

import importlib.util
import logging

logger = logging.getLogger(__name__)


def external_package_installed(package_name: str) -> bool:
    """Check if an external vertical package is installed.

    When the external package is available, the contrib plugin should
    skip its own registration to avoid collisions. The external package
    registers via entry points and takes priority.

    Args:
        package_name: Python package name (e.g., "victor_coding").

    Returns:
        True if the external package is installed.
    """
    if importlib.util.find_spec(package_name) is not None:
        logger.debug(
            "Skipping contrib registration — external package '%s' is installed.",
            package_name,
        )
        return True
    return False
