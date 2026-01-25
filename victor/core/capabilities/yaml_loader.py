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

"""YAML capability loader utilities (Phase 5.2).

This module provides utilities for loading vertical capability definitions
from YAML files and registering them with the CapabilityRegistry.

Example:
    from victor.core.capabilities.yaml_loader import register_vertical_capabilities

    # Register capabilities from YAML file
    count = register_vertical_capabilities("coding")
    print(f"Registered {count} coding capabilities")

    # Or specify custom path
    count = register_vertical_capabilities(
        "custom",
        yaml_path=Path("my_capabilities.yaml")
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from victor.core.capabilities.registry import CapabilityRegistry

logger = logging.getLogger(__name__)


def get_vertical_capability_path(vertical_name: str) -> Path:
    """Get the default capability YAML path for a vertical.

    Args:
        vertical_name: Name of the vertical (e.g., "coding", "devops")

    Returns:
        Path to the capability YAML file
    """
    # Path relative to victor package root
    victor_root = Path(__file__).parent.parent.parent
    return victor_root / vertical_name / "config" / "capabilities.yaml"


def register_vertical_capabilities(
    vertical_name: str,
    yaml_path: Optional[Path] = None,
    registry: Optional[CapabilityRegistry] = None,
) -> int:
    """Register capabilities from a vertical's YAML file with the registry.

    This function loads capability definitions from YAML and registers them
    with the CapabilityRegistry singleton (or provided registry instance).

    Args:
        vertical_name: Name of the vertical (e.g., "coding", "devops")
        yaml_path: Optional path to YAML file (default: auto-detect)
        registry: Optional registry instance (default: singleton)

    Returns:
        Number of capabilities registered

    Example:
        # Register coding capabilities
        count = register_vertical_capabilities("coding")

        # Use custom path
        count = register_vertical_capabilities(
            "custom",
            yaml_path=Path("path/to/capabilities.yaml")
        )
    """
    if yaml_path is None:
        yaml_path = get_vertical_capability_path(vertical_name)

    if not yaml_path.exists():
        logger.debug(f"No capability YAML found for {vertical_name} at {yaml_path}")
        return 0

    if registry is None:
        registry = CapabilityRegistry.get_instance()

    try:
        count = registry.load_from_yaml(yaml_path)
        logger.info(f"Registered {count} capabilities for {vertical_name} vertical")
        return count
    except Exception as e:
        logger.error(f"Failed to load capabilities for {vertical_name}: {e}")
        return 0


def register_all_vertical_capabilities(
    registry: Optional[CapabilityRegistry] = None,
) -> int:
    """Register capabilities from all known verticals.

    Args:
        registry: Optional registry instance (default: singleton)

    Returns:
        Total number of capabilities registered
    """
    verticals = ["coding", "devops", "research", "rag", "dataanalysis"]
    total = 0

    for vertical in verticals:
        count = register_vertical_capabilities(vertical, registry=registry)
        total += count

    logger.info(f"Registered {total} capabilities from {len(verticals)} verticals")
    return total


__all__ = [
    "get_vertical_capability_path",
    "register_vertical_capabilities",
    "register_all_vertical_capabilities",
]
