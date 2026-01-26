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

"""Plugin discovery system for OCP compliance.

This module provides unified plugin discovery that replaces hardcoded vertical
registration with entry points and YAML fallback (TDD approach).

Design Philosophy:
- Entry points for extensible plugin discovery (Python standard)
- YAML fallback for air-gapped environments
- Built-in verticals always available
- LRU caching for performance (<10ms cached)
- Air-gapped mode detection (VICTOR_AIRGAPPED)

OLD Pattern (violates OCP - hardcoded registration):
    # In victor/core/verticals/__init__.py:178-186
    def _register_builtin_verticals():
        VerticalRegistry.register_lazy_import("coding", "victor.coding:CodingAssistant")
        VerticalRegistry.register_lazy_import("research", "victor.research:ResearchAssistant")
        # ... hardcoded list requiring code modification to add verticals

NEW Pattern (OCP compliant - plugin discovery):
    discovery = PluginDiscovery()
    result = discovery.discover_all()

    # Result includes built-in, entry point, and YAML discovered verticals
    # External plugins can register via pyproject.toml entry points
    # No code modification needed to add new verticals

Usage:
    from victor.core.verticals.plugin_discovery import get_plugin_discovery

    # Discover all verticals
    discovery = get_plugin_discovery()
    result = discovery.discover_all()

    # Register discovered verticals
    for name, vertical_class in result.verticals.items():
        VerticalRegistry.register(vertical_class)
"""

import os
import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Dict, Type, Optional, Any, TYPE_CHECKING, cast, List, Union

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint
else:
    try:
        from importlib.metadata import EntryPoint
    except ImportError:
        from importlib_metadata import EntryPoint

if TYPE_CHECKING:
    from importlib.metadata import entry_points
else:
    try:
        from importlib.metadata import entry_points
    except ImportError:
        from importlib_metadata import entry_points

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from victor.core.verticals.base import VerticalBase


logger = logging.getLogger(__name__)


# =============================================================================
# Plugin Source Enum
# =============================================================================


class PluginSource(Enum):
    """Source of plugin discovery."""

    BUILTIN = "builtin"
    """Built-in vertical (included in Victor core)."""

    ENTRY_POINT = "entry_point"
    """Discovered from Python entry points."""

    YAML = "yaml"
    """Discovered from YAML configuration file."""


# =============================================================================
# Discovery Result
# =============================================================================


@dataclass
class DiscoveryResult:
    """Result of plugin discovery operation.

    Attributes:
        verticals: Dict mapping vertical name to vertical class
        sources: Dict mapping vertical name to discovery source
        lazy_imports: Dict mapping vertical name to lazy import string ("module:Class")
    """

    verticals: Dict[str, Type["VerticalBase"]] = field(default_factory=dict)
    sources: Dict[str, PluginSource] = field(default_factory=dict)
    lazy_imports: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# Built-in Vertical Configuration
# =============================================================================


@dataclass
class BuiltinVerticalConfig:
    """Configuration for a built-in vertical."""

    name: str
    """Vertical name (used for registration)."""

    lazy_import: str
    """Lazy import string in format "module:Class"."""

    @property
    def module_path(self) -> str:
        """Extract module path from lazy import."""
        return self.lazy_import.split(":")[0]

    @property
    def class_name(self) -> str:
        """Extract class name from lazy import."""
        return self.lazy_import.split(":")[1]


# =============================================================================
# Plugin Discovery
# =============================================================================


class PluginDiscovery:
    """Unified plugin discovery system.

    Replaces hardcoded vertical registration with extensible plugin discovery.
    Supports three sources: built-in, entry points, and YAML fallback.

    Attributes:
        enable_entry_points: Whether to discover from entry points
        enable_yaml_fallback: Whether to use YAML fallback
        enable_cache: Whether to cache discovery results
        cache_size: Max size of LRU cache
    """

    # Entry point group for vertical discovery
    ENTRY_POINT_GROUP = "victor.verticals"

    # Built-in vertical configurations
    BUILTIN_VERTICALS = [
        BuiltinVerticalConfig("coding", "victor.coding:CodingAssistant"),
        BuiltinVerticalConfig("research", "victor.research:ResearchAssistant"),
        BuiltinVerticalConfig("devops", "victor.devops:DevOpsAssistant"),
        BuiltinVerticalConfig("dataanalysis", "victor.dataanalysis:DataAnalysisAssistant"),
        BuiltinVerticalConfig("rag", "victor.rag:RAGAssistant"),
        BuiltinVerticalConfig("benchmark", "victor.benchmark:BenchmarkVertical"),
    ]

    # YAML config path for air-gapped fallback
    YAML_CONFIG_PATH = Path("victor/config/verticals/external_verticals.yaml")

    def __init__(
        self,
        enable_entry_points: bool = True,
        enable_yaml_fallback: bool = True,
        enable_cache: bool = True,
        cache_size: int = 100,
    ) -> None:
        """Initialize PluginDiscovery.

        Args:
            enable_entry_points: Enable entry point discovery (auto-disabled in air-gapped mode)
            enable_yaml_fallback: Enable YAML fallback (auto-enabled in air-gapped mode)
            enable_cache: Enable LRU caching of discovery results
            cache_size: Max cache size for LRU cache
        """
        # Detect air-gapped mode
        airgapped = os.getenv("VICTOR_AIRGAPPED", "false").lower() == "true"

        # Auto-detect plugin discovery flag
        use_plugin_discovery = os.getenv("VICTOR_USE_PLUGIN_DISCOVERY", "true").lower() == "true"

        if not use_plugin_discovery:
            # Plugin discovery disabled entirely
            self.enable_entry_points = False
            self.enable_yaml_fallback = False
        elif airgapped:
            # Air-gapped mode: disable entry points, enable YAML fallback
            self.enable_entry_points = False
            self.enable_yaml_fallback = True
        else:
            # Normal mode: use provided settings
            self.enable_entry_points = enable_entry_points
            self.enable_yaml_fallback = enable_yaml_fallback

        self.enable_cache = enable_cache
        self.cache_size = cache_size

        self.logger = logger

        # Simple in-memory cache (not using lru_cache decorator for flexibility)
        self._cache: Dict[str, DiscoveryResult] = {}

    def discover_from_entry_points(self) -> DiscoveryResult:
        """Discover verticals from Python entry points.

        Returns:
            DiscoveryResult with discovered verticals

        Raises:
            ImportError: If entry point loading fails (logged, not raised)
        """
        result = DiscoveryResult()

        if not self.enable_entry_points:
            return result

        try:
            # Handle both old and new entry_points API
            # Python 3.10+: entry_points(group=...) returns iterable directly
            # Python <3.10: _entry_points().group(...) returns iterable
            from importlib.metadata import entry_points as _entry_points_internal

            eps_result = _entry_points_internal(group=self.ENTRY_POINT_GROUP)

            # Check if eps_result has .select() method (new API)
            if hasattr(eps_result, "select"):
                # New API: use .select() to get entry points
                eps = eps_result.select(group=self.ENTRY_POINT_GROUP)
            else:
                # Old API: eps_result is already the iterable
                eps = eps_result

            for ep in cast(List[Union[Any, "EntryPoint"]], eps):
                try:
                    # Load the vertical class
                    vertical_class = ep.load()

                    # Validate it's a VerticalBase subclass
                    if not self._validate_vertical(vertical_class):
                        self.logger.warning(
                            f"Entry point '{ep.name}' does not provide a valid VerticalBase subclass, skipping"
                        )
                        continue

                    # Register discovered vertical
                    result.verticals[ep.name] = vertical_class
                    result.sources[ep.name] = PluginSource.ENTRY_POINT

                    self.logger.debug(f"Discovered vertical '{ep.name}' from entry point")

                except ImportError as e:
                    self.logger.warning(f"Failed to load entry point '{ep.name}': {e}")
                    continue
                except Exception as e:
                    self.logger.warning(f"Error loading entry point '{ep.name}': {e}")
                    continue

        except Exception as e:
            self.logger.warning(f"Failed to discover entry points: {e}")

        return result

    def discover_from_yaml(self) -> DiscoveryResult:
        """Discover verticals from YAML configuration file.

        Returns:
            DiscoveryResult with discovered verticals

        Raises:
            yaml.YAMLError: If YAML parsing fails (logged, not raised)
        """
        result = DiscoveryResult()

        if not self.enable_yaml_fallback:
            return result

        # Check if YAML file exists
        if not self.YAML_CONFIG_PATH.exists():
            self.logger.debug(f"YAML config not found at {self.YAML_CONFIG_PATH}")
            return result

        # Check if yaml is available
        # Note: yaml is None when not installed, but MyPy sees it as always available
        # This check is kept for runtime safety even if MyPy thinks it's unreachable
        if yaml is None:
            self.logger.warning("YAML library not available, skipping YAML discovery")
            return result

        try:
            # Read YAML config
            yaml_content = self.YAML_CONFIG_PATH.read_text()
            config = yaml.safe_load(yaml_content)

            if not config or "verticals" not in config:
                self.logger.debug("Invalid YAML config: missing 'verticals' key")
                return result

            # Check if verticals is None (empty YAML section)
            verticals_list = config.get("verticals")
            if not verticals_list:
                self.logger.debug("No verticals configured in YAML")
                return result

            # Load each vertical from YAML
            for vertical_config in verticals_list:
                try:
                    name = vertical_config.get("name")
                    module_path = vertical_config.get("module")
                    class_name = vertical_config.get("class")

                    if not all([name, module_path, class_name]):
                        self.logger.warning(f"Invalid vertical config: {vertical_config}, skipping")
                        continue

                    # Import the module
                    module = import_module(module_path)

                    # Get the vertical class
                    vertical_class = getattr(module, class_name)

                    # Validate it's a VerticalBase subclass
                    if not self._validate_vertical(vertical_class):
                        self.logger.warning(
                            f"YAML vertical '{name}' does not provide a valid VerticalBase subclass, skipping"
                        )
                        continue

                    # Register discovered vertical
                    result.verticals[name] = vertical_class
                    result.sources[name] = PluginSource.YAML

                    self.logger.debug(f"Discovered vertical '{name}' from YAML")

                except ImportError as e:
                    self.logger.warning(f"Failed to import module for YAML vertical: {e}")
                    continue
                except AttributeError as e:
                    self.logger.warning(f"Class not found in module: {e}")
                    continue
                except Exception as e:
                    self.logger.warning(f"Error loading YAML vertical: {e}")
                    continue

        except yaml.YAMLError as e:
            self.logger.warning(f"Failed to parse YAML config: {e}")
        except Exception as e:
            self.logger.warning(f"Error reading YAML config: {e}")

        return result

    def discover_builtin_verticals(self) -> DiscoveryResult:
        """Discover all built-in verticals.

        Returns:
            DiscoveryResult with all built-in verticals (lazy imports)
        """
        result = DiscoveryResult()

        for builtin_config in self.BUILTIN_VERTICALS:
            # Store lazy import info
            result.lazy_imports[builtin_config.name] = builtin_config.lazy_import

            # For built-in, we don't actually import the class
            # Just store the lazy import string for later registration
            # Use a placeholder to satisfy type checking
            result.verticals[builtin_config.name] = None  # type: ignore[assignment]
            result.sources[builtin_config.name] = PluginSource.BUILTIN

        return result

    def discover_all(self) -> DiscoveryResult:
        """Discover verticals from all available sources.

        Merges results from built-in, entry points, and YAML sources.
        Entry points take precedence over built-in (for customization).

        Returns:
            DiscoveryResult with all discovered verticals

        Raises:
            Exception: Logged but not raised, returns partial result
        """
        # Check cache first
        if self.enable_cache:
            cache_key = "discover_all"
            if cache_key in self._cache:
                self.logger.debug("Returning cached discovery result")
                return self._cache[cache_key]

        result = DiscoveryResult()

        try:
            # Discover from all sources
            builtin_result = self.discover_builtin_verticals()
            entry_point_result = (
                self.discover_from_entry_points() if self.enable_entry_points else DiscoveryResult()
            )
            yaml_result = (
                self.discover_from_yaml() if self.enable_yaml_fallback else DiscoveryResult()
            )

            # Merge results (entry points override built-in, YAML overrides both)
            all_results = [builtin_result, entry_point_result, yaml_result]

            # Define precedence order (higher index = higher precedence)
            precedence_order = {
                PluginSource.BUILTIN: 0,
                PluginSource.ENTRY_POINT: 1,
                PluginSource.YAML: 2,
            }

            for partial_result in all_results:
                for name, vertical_class in partial_result.verticals.items():
                    source = partial_result.sources.get(name, PluginSource.BUILTIN)

                    # Check if we should add/override this vertical
                    if name not in result.verticals:
                        # Not present yet, add it
                        result.verticals[name] = vertical_class
                        result.sources[name] = source
                    else:
                        # Already present, check precedence
                        existing_source = result.sources.get(name, PluginSource.BUILTIN)
                        if precedence_order.get(source, 0) > precedence_order.get(
                            existing_source, 0
                        ):
                            # Higher precedence source, override
                            result.verticals[name] = vertical_class
                            result.sources[name] = source

                # Merge lazy imports
                for name, lazy_import in partial_result.lazy_imports.items():
                    if name not in result.lazy_imports:
                        result.lazy_imports[name] = lazy_import

            # Cache the result
            if self.enable_cache:
                self._cache["discover_all"] = result

        except Exception as e:
            self.logger.error(f"Error during plugin discovery: {e}")
            # Return partial result even if discovery fails

        return result

    def clear_cache(self) -> None:
        """Clear the discovery cache."""
        self._cache.clear()
        self.logger.debug("Discovery cache cleared")

    def _validate_vertical(self, vertical_class: Any) -> bool:
        """Validate that a class or instance is a valid VerticalBase.

        Args:
            vertical_class: Class or instance to validate

        Returns:
            True if valid VerticalBase subclass or instance, False otherwise
        """
        from victor.core.verticals.base import VerticalBase

        try:
            # Check if it's an instance (not a type)
            if not isinstance(vertical_class, type):
                # It's an instance, check if it's an instance of VerticalBase
                return isinstance(vertical_class, VerticalBase)

            # It's a class, check if it's a subclass of VerticalBase (but not VerticalBase itself)
            return issubclass(vertical_class, VerticalBase) and vertical_class is not VerticalBase
        except (TypeError, AttributeError):
            return False


# =============================================================================
# Factory Functions
# =============================================================================


_plugin_discovery_cache: Optional[PluginDiscovery] = None


def get_plugin_discovery(
    enable_entry_points: bool = True,
    enable_yaml_fallback: bool = True,
    enable_cache: bool = True,
    cache_size: int = 100,
) -> PluginDiscovery:
    """Get or create singleton PluginDiscovery instance.

    Args:
        enable_entry_points: Enable entry point discovery
        enable_yaml_fallback: Enable YAML fallback
        enable_cache: Enable caching
        cache_size: Cache size for LRU cache

    Returns:
        PluginDiscovery singleton instance
    """
    global _plugin_discovery_cache

    # Check if cached instance exists with compatible settings
    if _plugin_discovery_cache is None:
        _plugin_discovery_cache = PluginDiscovery(
            enable_entry_points=enable_entry_points,
            enable_yaml_fallback=enable_yaml_fallback,
            enable_cache=enable_cache,
            cache_size=cache_size,
        )
    else:
        # Check if environment changed
        airgapped = os.getenv("VICTOR_AIRGAPPED", "false").lower() == "true"
        use_plugin_discovery = os.getenv("VICTOR_USE_PLUGIN_DISCOVERY", "true").lower() == "true"

        # If environment changed, create new instance
        if airgapped and _plugin_discovery_cache.enable_entry_points:
            _plugin_discovery_cache = PluginDiscovery(
                enable_entry_points=False,
                enable_yaml_fallback=True,
                enable_cache=enable_cache,
                cache_size=cache_size,
            )
        elif not use_plugin_discovery and (
            _plugin_discovery_cache.enable_entry_points
            or _plugin_discovery_cache.enable_yaml_fallback
        ):
            _plugin_discovery_cache = PluginDiscovery(
                enable_entry_points=False,
                enable_yaml_fallback=False,
                enable_cache=enable_cache,
                cache_size=cache_size,
            )

    return _plugin_discovery_cache


def clear_plugin_discovery_cache() -> None:
    """Clear the singleton PluginDiscovery cache (mainly for testing)."""
    global _plugin_discovery_cache
    _plugin_discovery_cache = None


__all__ = [
    "PluginSource",
    "DiscoveryResult",
    "BuiltinVerticalConfig",
    "PluginDiscovery",
    "get_plugin_discovery",
    "clear_plugin_discovery_cache",
]
