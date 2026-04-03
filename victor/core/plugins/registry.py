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

"""Plugin registry for Victor.

Handles discovery and registration of Victor plugins via entry points.
"""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

from victor.core.aot_manifest import AOTManifestManager
from victor.core.container import ServiceContainer, get_container
from victor.core.plugins.protocol import VictorPlugin
from victor.framework.module_loader import get_entry_point_cache

if TYPE_CHECKING:
    from victor.core.plugins.context import HostPluginContext

logger = logging.getLogger(__name__)


def call_lifecycle_hook(plugin: VictorPlugin, hook: str, **kwargs: Any) -> Any:
    """Call an optional lifecycle hook on a plugin if it implements it.

    Uses hasattr-based dispatch for backward compatibility with plugins
    that don't implement newer lifecycle methods.

    Args:
        plugin: The plugin instance.
        hook: Method name (e.g., 'on_activate', 'on_deactivate', 'health_check').
        **kwargs: Arguments to pass to the hook.

    Returns:
        Hook return value, or None if not implemented.
    """
    method = getattr(plugin, hook, None)
    if method is not None and callable(method):
        try:
            return method(**kwargs)
        except Exception as e:
            logger.warning("Lifecycle hook '%s' failed on plugin '%s': %s", hook, plugin.name, e)
    return None


async def call_lifecycle_hook_async(plugin: VictorPlugin, hook: str, **kwargs: Any) -> Any:
    """Async-aware lifecycle hook dispatch.

    Prefers ``{hook}_async`` if the plugin implements it, otherwise
    falls back to the synchronous ``{hook}`` method.

    Args:
        plugin: The plugin instance.
        hook: Base method name (e.g., 'on_activate').
        **kwargs: Arguments to pass to the hook.

    Returns:
        Hook return value, or None if not implemented.
    """
    # Try async variant first
    async_method = getattr(plugin, f"{hook}_async", None)
    if async_method is not None and callable(async_method):
        try:
            return await async_method(**kwargs)
        except Exception as e:
            logger.warning(
                "Async lifecycle hook '%s_async' failed on plugin '%s': %s",
                hook,
                plugin.name,
                e,
            )
            return None

    # Fall back to sync hook
    return call_lifecycle_hook(plugin, hook, **kwargs)


class _LegacyVerticalPluginAdapter:
    """Wraps a VerticalBase subclass as a VictorPlugin for backward compatibility.

    This adapter allows verticals registered via the deprecated 'victor.verticals'
    entry point group to work within the unified plugin system.
    """

    def __init__(self, ep_name: str, vertical_cls: type) -> None:
        self._ep_name = ep_name
        self._vertical_cls = vertical_cls
        self._name = getattr(vertical_cls, "name", ep_name)

    @property
    def name(self) -> str:
        return self._name

    def register(self, context: Any) -> None:
        """Register the wrapped vertical."""
        if hasattr(context, "register_vertical"):
            context.register_vertical(self._vertical_cls)

    def get_cli_app(self) -> None:
        return None

    def on_activate(self) -> None:
        pass

    def on_deactivate(self) -> None:
        pass

    def health_check(self) -> Dict[str, Any]:
        return {"healthy": True, "adapter": "legacy_vertical"}


class _ExternalPluginAdapter:
    """Wraps an ExternalPluginManager RegisteredPlugin as a VictorPlugin.

    This adapter allows subprocess-based external plugins (discovered via
    plugin.json manifests) to appear in the unified PluginRegistry alongside
    entry-point-based Python plugins.
    """

    def __init__(self, registered_plugin: Any) -> None:
        self._plugin = registered_plugin

    @property
    def name(self) -> str:
        return self._plugin.plugin_id

    def register(self, context: Any) -> None:
        """External plugins register tools via their tool definitions."""
        if not hasattr(context, "register_tool"):
            return
        for tool_def in self._plugin.manifest.tools:
            logger.debug(
                "External plugin %s provides tool: %s",
                self.name,
                tool_def.name,
            )

    def get_cli_app(self) -> None:
        return None

    def on_activate(self) -> None:
        pass

    def on_deactivate(self) -> None:
        pass

    def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": self._plugin.enabled,
            "adapter": "external_plugin",
            "kind": self._plugin.kind.value,
            "version": self._plugin.version,
            "tools": len(self._plugin.manifest.tools),
        }


class PluginRegistry:
    """Registry for discovering and managing Victor plugins.

    Discovers plugins via the 'victor.plugins' entry point group
    and optionally from external manifest-based plugins.
    """

    _instance: Optional[PluginRegistry] = None
    _lock = threading.Lock()
    ENTRY_POINT_GROUP = "victor.plugins"
    LEGACY_ENTRY_POINT_GROUP = "victor.verticals"
    AOT_GROUPS = ["victor.plugins", "victor.verticals"]

    def __init__(self) -> None:
        """Initialize registry."""
        self._plugins: Dict[str, VictorPlugin] = {}
        self._context: Optional[HostPluginContext] = None
        self._discovered = False

    @classmethod
    def get_instance(cls) -> PluginRegistry:
        """Get the singleton instance of PluginRegistry."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = PluginRegistry()
        return cls._instance

    def discover(self, force: bool = False) -> List[VictorPlugin]:
        """Discover and load plugins from entry points.

        Uses a three-tier discovery strategy:
        1. AOT manifest (O(1) file read, fastest)
        2. EntryPointCache (memory/disk cache with env hash)
        3. Full entry_points() scan (slowest, updates caches)

        Args:
            force: Whether to force re-discovery.

        Returns:
            List of discovered plugins.
        """
        if self._discovered and not force:
            return list(self._plugins.values())

        # Try AOT manifest fast-path (skip when forcing refresh)
        aot_entries = None
        if not force:
            aot_entries = self._try_aot_fast_path()

        if aot_entries is not None:
            ep_entries = aot_entries.get(self.ENTRY_POINT_GROUP, {})
        else:
            cache = get_entry_point_cache()
            ep_entries = cache.get_entry_points(self.ENTRY_POINT_GROUP, force_refresh=force)

        for name, value in ep_entries.items():
            try:
                plugin_cls = self._load_plugin_from_value(name, value)
                logger.info(f"Loading plugin from entry point: {name}")

                # If it's a class, instantiate it
                if isinstance(plugin_cls, type):
                    plugin = plugin_cls()
                else:
                    plugin = plugin_cls

                if isinstance(plugin, VictorPlugin):
                    self._plugins[plugin.name] = plugin
                    logger.info(f"Discovered plugin: {plugin.name} (from {value})")
                else:
                    logger.warning(
                        f"Entry point {name} did not provide a valid VictorPlugin: {plugin_cls}"
                    )
            except Exception as e:
                logger.error(f"Failed to load plugin from entry point {name}: {e}")

        # Scan legacy 'victor.verticals' group for backward compatibility
        if aot_entries is not None:
            legacy_entries = aot_entries.get(self.LEGACY_ENTRY_POINT_GROUP, {})
        else:
            legacy_entries = cache.get_entry_points(
                self.LEGACY_ENTRY_POINT_GROUP, force_refresh=force
            )
        for name, value in legacy_entries.items():
            if name in self._plugins:
                # victor.plugins takes precedence on name collision
                continue
            try:
                obj = self._load_plugin_from_value(name, value)
                if isinstance(obj, VictorPlugin):
                    self._plugins[obj.name] = obj
                    logger.info("Discovered plugin: %s (from legacy victor.verticals)", obj.name)
                elif isinstance(obj, type):
                    # Wrap VerticalBase subclass as a plugin adapter
                    adapter = _LegacyVerticalPluginAdapter(name, obj)
                    self._plugins[adapter.name] = adapter
                    logger.info("Discovered legacy vertical: %s (wrapped as plugin)", name)
                    import warnings

                    warnings.warn(
                        f"Vertical '{name}' registered via victor.verticals entry point. "
                        "Migrate to victor.plugins for full plugin lifecycle support. "
                        "The victor.verticals group is deprecated and will be removed in v0.7.0.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                else:
                    # Try as an instance
                    if hasattr(obj, "name"):
                        adapter = _LegacyVerticalPluginAdapter(name, type(obj))
                        self._plugins[adapter.name] = adapter
                        logger.info(
                            "Discovered legacy vertical instance: %s (wrapped as plugin)", name
                        )
            except Exception as e:
                logger.error("Failed to load legacy entry point %s: %s", name, e)

        # Update AOT manifest after slow-path discovery
        if aot_entries is None:
            self._update_aot_manifest()

        # Discover external manifest-based plugins and bridge them in.
        self._discover_external_plugins()

        self._discovered = True
        return list(self._plugins.values())

    def _discover_external_plugins(self) -> None:
        """Discover external plugins via plugin.json manifests.

        Wraps each discovered external plugin as a VictorPlugin adapter
        so it appears alongside entry-point-based plugins.
        """
        try:
            from victor.core.plugins.external import ExternalPluginManager

            manager = ExternalPluginManager()
            plugins = manager.discover_plugins()
            for plugin in plugins:
                if plugin.plugin_id in self._plugins:
                    continue  # Entry-point plugins take precedence
                adapter = _ExternalPluginAdapter(plugin)
                self._plugins[adapter.name] = adapter
                logger.info(
                    "Discovered external plugin: %s (%s, %d tools)",
                    plugin.plugin_id,
                    plugin.kind.value,
                    len(plugin.manifest.tools),
                )
        except ImportError:
            logger.debug("External plugin support not available")
        except Exception as e:
            logger.debug("External plugin discovery failed: %s", e)

    def _try_aot_fast_path(self) -> Optional[Dict[str, Dict[str, str]]]:
        """Try to load entry points from AOT manifest.

        Returns:
            Dict mapping group names to their entry point dicts
            (name -> "module:attr"), or None if no valid manifest.
        """
        try:
            manager = AOTManifestManager()
            manifest = manager.load_manifest()
            if manifest is None:
                return None

            result: Dict[str, Dict[str, str]] = {}
            for group, entries in manifest.entries.items():
                group_dict: Dict[str, str] = {}
                for entry in entries:
                    value = f"{entry.module}:{entry.attr}" if entry.attr else entry.module
                    group_dict[entry.name] = value
                result[group] = group_dict

            logger.debug("AOT manifest hit: loaded %d groups", len(result))
            return result
        except Exception as e:
            logger.debug("AOT manifest fast-path failed: %s", e)
            return None

    def _update_aot_manifest(self) -> None:
        """Update AOT manifest after a slow-path scan."""
        try:
            manager = AOTManifestManager()
            manifest = manager.build_manifest(self.AOT_GROUPS)
            manager.save_manifest(manifest)
            logger.debug("Updated AOT manifest with %d groups", len(manifest.entries))
        except Exception as e:
            logger.debug("Failed to update AOT manifest: %s", e)

    def _load_plugin_from_value(self, name: str, value: str) -> Any:
        """Load a plugin object from an entry point value string.

        Args:
            name: Entry point name.
            value: Entry point value in "module:attr" or "module.attr" format.

        Returns:
            Loaded class or object.
        """
        if ":" in value:
            module_name, attr_name = value.split(":", 1)
        else:
            module_name, attr_name = value.rsplit(".", 1)

        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    def register_all(self, container: Optional[ServiceContainer] = None) -> None:
        """Register all discovered plugins into the container.

        Args:
            container: Service container to register into. Defaults to global.
        """
        from victor.core.plugins.context import HostPluginContext

        container = container or get_container()
        self._context = HostPluginContext(container)

        for plugin in self.discover():
            try:
                plugin.register(self._context)
                logger.debug(f"Registered plugin: {plugin.name}")
            except Exception as e:
                logger.error(f"Failed to register plugin {plugin.name}: {e}", exc_info=True)

    @property
    def context(self) -> Optional[HostPluginContext]:
        """Return the registration context containing plugin commands."""
        return self._context

    def get_plugin(self, name: str) -> Optional[VictorPlugin]:
        """Get a plugin by name.

        Args:
            name: Plugin identifier.

        Returns:
            VictorPlugin or None if not found.
        """
        return self._plugins.get(name)

    def check_plugin_health(self) -> Dict[str, Dict[str, Any]]:
        """Run health checks on all registered plugins.

        Returns:
            Dict mapping plugin names to their health status dicts.
            Plugins without health_check get {'healthy': True, 'reason': 'no health_check'}.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for name, plugin in self._plugins.items():
            status = call_lifecycle_hook(plugin, "health_check")
            if status is None:
                status = {"healthy": True, "reason": "no health_check"}
            results[name] = status
        return results

    def list_plugins(self) -> List[VictorPlugin]:
        """List all discovered plugins.

        Returns:
            List of plugins.
        """
        return list(self._plugins.values())

    @property
    def is_discovered(self) -> bool:
        """Whether discovery has been run."""
        return self._discovered

    def get_vertical_classes(self) -> Dict[str, type]:
        """Extract vertical classes from discovered plugins.

        Returns classes from `_LegacyVerticalPluginAdapter` instances,
        enabling VerticalLoader to delegate discovery to this registry
        instead of scanning entry points independently.

        Returns:
            Dict mapping vertical names to their classes.
        """
        result: Dict[str, type] = {}
        for name, plugin in self._plugins.items():
            if isinstance(plugin, _LegacyVerticalPluginAdapter):
                result[plugin.name] = plugin._vertical_cls
        return result

    def list_all_with_type(self) -> List[Dict[str, Any]]:
        """Return a unified view of all plugins with type classification.

        Each entry is a dict with keys: name, type, version, enabled.
        The ``type`` field classifies the plugin as:
        - ``"vertical"`` — VerticalBase wrapped via _LegacyVerticalPluginAdapter
        - ``"external"`` — Subprocess plugin wrapped via _ExternalPluginAdapter
        - ``"plugin"`` — Regular VictorPlugin (entry-point-based)

        Returns:
            Sorted list of plugin info dicts.
        """
        if not self._discovered:
            return []

        entries: List[Dict[str, Any]] = []
        for name, plugin in self._plugins.items():
            if isinstance(plugin, _LegacyVerticalPluginAdapter):
                plugin_type = "vertical"
                version = getattr(plugin._vertical_cls, "version", "0.0.0")
                enabled = True
            elif isinstance(plugin, _ExternalPluginAdapter):
                plugin_type = "external"
                version = getattr(plugin, "_plugin", None)
                version = getattr(version, "version", "0.0.0") if version else "0.0.0"
                enabled = getattr(getattr(plugin, "_plugin", None), "enabled", True)
            else:
                plugin_type = "plugin"
                version = getattr(plugin, "version", "0.0.0")
                enabled = True

            entries.append(
                {
                    "name": name,
                    "type": plugin_type,
                    "version": str(version),
                    "enabled": enabled,
                }
            )

        return sorted(entries, key=lambda e: e["name"])
