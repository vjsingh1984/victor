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

"""Vertical Discovery System for Protocol-Based Framework Extension.

This module provides a protocol-based discovery system that enables the framework
to dynamically load vertical capabilities without hardcoded imports. This is
CRITICAL for Open/Closed Principle (OCP) compliance - the framework is now open
for extension (new verticals can be added) but closed for modification (framework
code doesn't need to change).

Key Benefits:
    - OCP Compliance: No hardcoded vertical imports in framework
    - Protocol-Based: Uses protocols for loose coupling
    - Auto-Discovery: Automatic registration of vertical capabilities
    - Extensible: New verticals automatically discovered
    - Type-Safe: Full type hints and protocol compliance

Architecture:
    VerticalDiscovery provides static methods to discover:
    - Prompt contributors from verticals implementing PromptContributorProtocol
    - Escape hatches from verticals with escape_hatches modules
    - All registered verticals from entry points

Usage:
    # Discover all prompt contributors
    from victor.framework.discovery import VerticalDiscovery

    contributors = VerticalDiscovery.discover_prompt_contributors()
    for contributor in contributors:
        print(f"Found: {contributor.__class__.__name__}")

    # Discover escape hatches
    hatches = VerticalDiscovery.discover_escape_hatches()
    print(f"Found {len(hatches)} verticals with escape hatches")

    # Get all verticals
    verticals = VerticalDiscovery.discover_verticals()
    for name, vertical_class in verticals.items():
        print(f"Vertical: {name}")
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from victor.core.verticals.base import VerticalBase

logger = logging.getLogger(__name__)


class VerticalDiscovery:
    """Discovers vertical capabilities using protocols and entry points.

    This class provides static methods for discovering vertical capabilities
    without hardcoded imports. All discovery is based on protocols and
    entry points, ensuring OCP compliance.

    Thread Safety:
        Discovery operations are read-only and thread-safe after initialization.
        The class uses lazy loading and caching for performance.

    Example:
        # Discover prompt contributors
        contributors = VerticalDiscovery.discover_prompt_contributors()

        # Discover escape hatches
        hatches = VerticalDiscovery.discover_escape_hatches()

        # Discover all verticals
        verticals = VerticalDiscovery.discover_verticals()
    """

    # Cache for discovered items
    _prompt_contributors_cache: Optional[List[Any]] = None
    _escape_hatches_cache: Optional[Dict[str, Dict[str, Any]]] = None
    _verticals_cache: Optional[Dict[str, Type[VerticalBase]]] = None

    @staticmethod
    def discover_prompt_contributors() -> List[Any]:
        """Discover all prompt contributors from registered verticals.

        Searches for verticals that implement PromptContributorProtocol
        and returns instances of their prompt contributors.

        Returns:
            List of prompt contributor instances

        Example:
            contributors = VerticalDiscovery.discover_prompt_contributors()
            for contributor in contributors:
                builder.add_from_contributor(contributor)
        """
        # Return cached if available
        if VerticalDiscovery._prompt_contributors_cache is not None:
            return VerticalDiscovery._prompt_contributors_cache

        contributors = []

        try:
            # Import protocol for type checking
            from victor.core.verticals.protocols import PromptContributorProtocol

            # Discover all verticals
            verticals = VerticalDiscovery.discover_verticals()

            # Check each vertical for prompt contributor
            for vertical_name, vertical_class in verticals.items():
                try:
                    # Check if vertical has get_prompt_contributor method
                    if hasattr(vertical_class, "get_prompt_contributor"):
                        contributor = vertical_class.get_prompt_contributor()
                        if contributor is not None:
                            # Verify it implements the protocol
                            if isinstance(contributor, PromptContributorProtocol):
                                contributors.append(contributor)
                                logger.debug(
                                    f"Discovered prompt contributor from '{vertical_name}'"
                                )
                            else:
                                logger.warning(
                                    f"Vertical '{vertical_name}' prompt contributor "
                                    "does not implement PromptContributorProtocol"
                                )
                except Exception as e:
                    logger.debug(f"Failed to load prompt contributor from '{vertical_name}': {e}")
                    continue

        except Exception as e:
            logger.error(f"Error during prompt contributor discovery: {e}")

        # Cache and return
        VerticalDiscovery._prompt_contributors_cache = contributors
        return contributors

    @staticmethod
    def discover_escape_hatches() -> Dict[str, Dict[str, Any]]:
        """Discover all escape hatches from registered verticals.

        Searches for escape_hatches.py modules in each vertical and loads
        their CONDITIONS and TRANSFORMS dictionaries.

        Returns:
            Dict mapping vertical names to their escape hatch dicts:
            {
                "coding": {"conditions": {...}, "transforms": {...}},
                "research": {"conditions": {...}, "transforms": {...}},
            }

        Example:
            hatches = VerticalDiscovery.discover_escape_hatches()
            for vertical_name, hatch_dict in hatches.items():
                conditions = hatch_dict["conditions"]
                transforms = hatch_dict["transforms"]
        """
        # Return cached if available
        if VerticalDiscovery._escape_hatches_cache is not None:
            return VerticalDiscovery._escape_hatches_cache

        escape_hatches: Dict[str, Dict[str, Any]] = {}

        try:
            # Discover all verticals
            verticals = VerticalDiscovery.discover_verticals()

            # Try to load escape hatches from each vertical
            for vertical_name in verticals.keys():
                try:
                    # Build module path
                    module_path = f"victor.{vertical_name}.escape_hatches"

                    # Import the module
                    module = importlib.import_module(module_path)

                    # Extract CONDITIONS and TRANSFORMS
                    conditions = getattr(module, "CONDITIONS", {})
                    transforms = getattr(module, "TRANSFORMS", {})

                    if conditions or transforms:
                        escape_hatches[vertical_name] = {
                            "conditions": conditions,
                            "transforms": transforms,
                        }
                        logger.debug(
                            f"Discovered {len(conditions)} conditions and "
                            f"{len(transforms)} transforms from '{vertical_name}'"
                        )

                except ImportError:
                    # No escape_hatches module for this vertical (that's OK)
                    logger.debug(f"No escape_hatches module for '{vertical_name}'")
                    continue
                except Exception as e:
                    logger.debug(f"Failed to load escape hatches from '{vertical_name}': {e}")
                    continue

        except Exception as e:
            logger.error(f"Error during escape hatch discovery: {e}")

        # Cache and return
        VerticalDiscovery._escape_hatches_cache = escape_hatches
        return escape_hatches

    @staticmethod
    def discover_verticals() -> Dict[str, Type[VerticalBase]]:
        """Discover all registered verticals.

        Loads verticals from:
        1. Entry points (victor.verticals)
        2. Built-in verticals (coding, research, devops, etc.)

        Returns:
            Dict mapping vertical names to their classes:
            {
                "coding": <class 'victor.coding.CodingAssistant'>,
                "research": <class 'victor.research.ResearchAssistant'>,
            }

        Example:
            verticals = VerticalDiscovery.discover_verticals()
            for name, vertical_class in verticals.items():
                print(f"Vertical: {name}")
                config = vertical_class.get_config()
        """
        # Return cached if available
        if VerticalDiscovery._verticals_cache is not None:
            return VerticalDiscovery._verticals_cache

        verticals: Dict[str, Type[VerticalBase]] = {}

        try:
            # Load from entry points first (external verticals)
            try:
                from importlib.metadata import entry_points

                eps = entry_points()
                if hasattr(eps, "select"):
                    # Python 3.10+
                    selected_eps = eps.select(group="victor.verticals")
                else:
                    # Python 3.9
                    selected_eps: Union[Any, List[Any]] = list(eps.get("victor.verticals", []))

                for ep in selected_eps:
                    try:
                        vertical_class = ep.load()
                        if hasattr(vertical_class, "name"):
                            verticals[vertical_class.name] = vertical_class
                            logger.debug(
                                f"Discovered vertical '{vertical_class.name}' from entry point"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to load vertical from entry point: {e}")
            except Exception as e:
                logger.debug(f"No entry points found: {e}")

            # Load built-in verticals (if not already loaded)
            builtin_verticals = [
                "coding",
                "research",
                "devops",
                "rag",
                "dataanalysis",
                "benchmark",
            ]

            for vertical_name in builtin_verticals:
                if vertical_name not in verticals:
                    try:
                        module_path = f"victor.{vertical_name}"
                        module = importlib.import_module(module_path)

                        # Look for vertical class (e.g., CodingAssistant)
                        vertical_class_name = (
                            f"{vertical_name.replace('_', ' ').title().replace(' ', '')}"
                            f"Assistant"
                        )

                        if hasattr(module, vertical_class_name):
                            vertical_class = getattr(module, vertical_class_name)
                            verticals[vertical_name] = vertical_class
                            logger.debug(f"Discovered built-in vertical '{vertical_name}'")
                        else:
                            logger.warning(
                                f"Built-in vertical '{vertical_name}' missing "
                                f"{vertical_class_name} class"
                            )
                    except ImportError as e:
                        logger.debug(f"Built-in vertical '{vertical_name}' not found: {e}")
                    except Exception as e:
                        logger.warning(f"Failed to load built-in vertical '{vertical_name}': {e}")

        except Exception as e:
            logger.error(f"Error during vertical discovery: {e}")

        # Cache and return
        VerticalDiscovery._verticals_cache = verticals
        return verticals

    @staticmethod
    def discover_vertical_by_name(vertical_name: str) -> Optional[Type[Any]]:
        """Discover a specific vertical by name.

        Args:
            vertical_name: Name of the vertical to discover

        Returns:
            Vertical class if found, None otherwise

        Example:
            VerticalClass = VerticalDiscovery.discover_vertical_by_name("coding")
            if VerticalClass:
                config = VerticalClass.get_config()
        """
        verticals = VerticalDiscovery.discover_verticals()
        return verticals.get(vertical_name)

    @staticmethod
    def clear_cache() -> None:
        """Clear all discovery caches.

        This is primarily useful for testing. After clearing, subsequent
        discovery calls will reload all verticals.

        Example:
            VerticalDiscovery.clear_cache()
            contributors = VerticalDiscovery.discover_prompt_contributors()
        """
        VerticalDiscovery._prompt_contributors_cache = None
        VerticalDiscovery._escape_hatches_cache = None
        VerticalDiscovery._verticals_cache = None
        logger.debug("Discovery cache cleared")


__all__ = [
    "VerticalDiscovery",
]
