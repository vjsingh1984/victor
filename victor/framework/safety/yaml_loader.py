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

"""YAML safety pattern loader utilities (Phase 6.2).

This module provides utilities for loading vertical safety pattern definitions
from YAML files and registering them with the SafetyPatternRegistry.

Example:
    from victor.framework.safety.yaml_loader import register_vertical_patterns

    # Register patterns from YAML file
    count = register_vertical_patterns("coding")
    print(f"Registered {count} coding patterns")

    # Or specify custom path
    count = register_vertical_patterns(
        "custom",
        yaml_path=Path("my_patterns.yaml")
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from victor.framework.safety.registry import SafetyPatternRegistry

logger = logging.getLogger(__name__)


def get_vertical_pattern_path(vertical_name: str) -> Path:
    """Get the default pattern YAML path for a vertical.

    Args:
        vertical_name: Name of the vertical (e.g., "coding", "devops")

    Returns:
        Path to the pattern YAML file
    """
    # Path relative to victor package root
    victor_root = Path(__file__).parent.parent.parent
    return victor_root / vertical_name / "config" / "patterns.yaml"


def register_vertical_patterns(
    vertical_name: str,
    yaml_path: Optional[Path] = None,
    registry: Optional[SafetyPatternRegistry] = None,
) -> int:
    """Register safety patterns from a vertical's YAML file with the registry.

    This function loads pattern definitions from YAML and registers them
    with the SafetyPatternRegistry singleton (or provided registry instance).

    Args:
        vertical_name: Name of the vertical (e.g., "coding", "devops")
        yaml_path: Optional path to YAML file (default: auto-detect)
        registry: Optional registry instance (default: singleton)

    Returns:
        Number of patterns registered

    Example:
        # Register coding patterns
        count = register_vertical_patterns("coding")

        # Use custom path
        count = register_vertical_patterns(
            "custom",
            yaml_path=Path("path/to/patterns.yaml")
        )
    """
    if yaml_path is None:
        yaml_path = get_vertical_pattern_path(vertical_name)

    if not yaml_path.exists():
        logger.debug(f"No pattern YAML found for {vertical_name} at {yaml_path}")
        return 0

    if registry is None:
        registry = SafetyPatternRegistry.get_instance()

    try:
        count = registry.load_from_yaml(yaml_path)
        logger.info(f"Registered {count} patterns for {vertical_name} vertical")
        return count
    except Exception as e:
        logger.error(f"Failed to load patterns for {vertical_name}: {e}")
        return 0


def register_all_vertical_patterns(
    registry: Optional[SafetyPatternRegistry] = None,
) -> int:
    """Register safety patterns from all known verticals.

    Args:
        registry: Optional registry instance (default: singleton)

    Returns:
        Total number of patterns registered
    """
    verticals = ["coding", "devops", "research", "rag", "dataanalysis"]
    total = 0

    for vertical in verticals:
        count = register_vertical_patterns(vertical, registry=registry)
        total += count

    logger.info(f"Registered {total} patterns from {len(verticals)} verticals")
    return total


def register_builtin_scanners(
    registry: Optional[SafetyPatternRegistry] = None,
) -> int:
    """Register built-in safety scanners with the registry.

    This registers the standard scanners (SecretScanner, CommandScanner, etc.)
    for use with the registry's scan_with_scanner method.

    Args:
        registry: Optional registry instance (default: singleton)

    Returns:
        Number of scanners registered
    """
    from victor.framework.safety.scanners import (
        SecretScanner,
        CommandScanner,
        FilePathScanner,
    )

    if registry is None:
        registry = SafetyPatternRegistry.get_instance()

    scanners = [
        ("secrets", SecretScanner()),
        ("commands", CommandScanner()),
        ("filepaths", FilePathScanner()),
    ]

    count = 0
    for name, scanner in scanners:
        try:
            registry.register_scanner(name, scanner)
            count += 1
            logger.debug(f"Registered scanner '{name}'")
        except Exception as e:
            logger.warning(f"Failed to register scanner '{name}': {e}")

    logger.info(f"Registered {count} built-in scanners")
    return count


__all__ = [
    "get_vertical_pattern_path",
    "register_vertical_patterns",
    "register_all_vertical_patterns",
    "register_builtin_scanners",
]
