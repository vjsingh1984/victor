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

"""Tool plugin manager for dynamic tool loading.

This module provides the ToolPluginManager class for discovering,
loading, and managing tool plugins at runtime.

Plugin Discovery:
- Plugins are discovered from configured directories
- Each plugin directory should contain a plugin.py with a Plugin class
- Plugins can also be loaded from Python packages

Usage:
    manager = ToolPluginManager(
        plugin_dirs=[Path("~/.victor/plugins")],
        config={"my_plugin": {"api_key": "..."}}
    )

    # Load all discovered plugins
    manager.discover_and_load()

    # Get all tools from loaded plugins
    tools = manager.get_all_tools()

    # Register tools with a ToolRegistry
    manager.register_tools(registry)
"""

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from victor.tools.base import BaseTool, ToolRegistry
from victor.tools.plugin import PluginMetadata, ToolPlugin

logger = logging.getLogger(__name__)


class PluginLoadError(Exception):
    """Raised when a plugin fails to load."""

    pass


class ToolPluginRegistry:
    """Registry for discovering, loading, and managing tool plugins.

    The plugin registry handles:
    - Plugin discovery from directories
    - Dynamic loading/unloading of plugins
    - Plugin lifecycle management (init, cleanup)
    - Tool registration with ToolRegistry
    - Plugin configuration management

    Attributes:
        plugin_dirs: List of directories to search for plugins
        config: Plugin configuration dictionary
        loaded_plugins: Dictionary of loaded plugins by name

    Note: Previously named `ToolPluginManager`. Alias kept for backward compatibility.
    """

    def __init__(
        self,
        plugin_dirs: Optional[List[Path]] = None,
        config: Optional[Dict[str, Dict[str, Any]]] = None,
        auto_load: bool = False,
    ):
        """Initialize plugin registry.

        Args:
            plugin_dirs: Directories to search for plugins
            config: Plugin configuration (plugin_name -> config dict)
            auto_load: Automatically discover and load plugins on init
        """
        self.plugin_dirs: List[Path] = []
        if plugin_dirs:
            for dir_path in plugin_dirs:
                expanded = Path(dir_path).expanduser()
                if expanded.exists():
                    self.plugin_dirs.append(expanded)
                else:
                    logger.debug(f"Plugin directory does not exist: {expanded}")

        self.config = config or {}
        self.loaded_plugins: Dict[str, ToolPlugin] = {}
        self._disabled_plugins: Set[str] = set()

        if auto_load:
            self.discover_and_load()

    def discover_plugins(self) -> List[Path]:
        """Discover plugin directories.

        Searches plugin directories for valid plugins.
        A valid plugin directory contains a plugin.py file.

        Returns:
            List of paths to discovered plugin directories
        """
        discovered = []

        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.is_dir():
                continue

            # Check for plugins as subdirectories
            for item in plugin_dir.iterdir():
                if item.is_dir():
                    plugin_file = item / "plugin.py"
                    if plugin_file.exists():
                        discovered.append(item)
                        logger.debug(f"Discovered plugin: {item.name}")

            # Also check for plugin.py directly in plugin_dir
            direct_plugin = plugin_dir / "plugin.py"
            if direct_plugin.exists():
                discovered.append(plugin_dir)

        logger.info(f"Discovered {len(discovered)} plugins in {len(self.plugin_dirs)} directories")
        return discovered

    def load_plugin_from_path(self, plugin_path: Path) -> Optional[ToolPlugin]:
        """Load a plugin from a directory path.

        Args:
            plugin_path: Path to plugin directory

        Returns:
            Loaded ToolPlugin instance or None on failure
        """
        plugin_file = plugin_path / "plugin.py"
        if not plugin_file.exists():
            logger.warning(f"No plugin.py found in {plugin_path}")
            return None

        try:
            # Generate unique module name
            module_name = f"victor_plugin_{plugin_path.name}"

            # Load module from file
            spec = importlib.util.spec_from_file_location(module_name, plugin_file)
            if spec is None or spec.loader is None:
                raise PluginLoadError(f"Could not load spec for {plugin_file}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Find Plugin class
            if not hasattr(module, "Plugin"):
                raise PluginLoadError(f"No 'Plugin' class found in {plugin_file}")

            plugin_class = module.Plugin
            if not issubclass(plugin_class, ToolPlugin):
                raise PluginLoadError(f"Plugin class must extend ToolPlugin: {plugin_file}")

            # Get plugin-specific config
            plugin_config = self.config.get(plugin_class.name, {})

            # Instantiate plugin
            plugin = plugin_class(config=plugin_config)
            plugin.get_metadata().path = plugin_path

            logger.info(f"Loaded plugin: {plugin.name} v{plugin.version} from {plugin_path}")
            return plugin

        except Exception as e:
            logger.error(f"Failed to load plugin from {plugin_path}: {e}")
            return None

    def load_plugin_from_package(self, package_name: str) -> Optional[ToolPlugin]:
        """Load a plugin from an installed Python package.

        The package should have a top-level Plugin class extending ToolPlugin.

        Args:
            package_name: Name of the Python package

        Returns:
            Loaded ToolPlugin instance or None on failure
        """
        try:
            module = importlib.import_module(package_name)

            if not hasattr(module, "Plugin"):
                # Try submodule
                try:
                    plugin_module = importlib.import_module(f"{package_name}.plugin")
                    module = plugin_module
                except ImportError:
                    raise PluginLoadError(f"No 'Plugin' class found in {package_name}")

            plugin_class = module.Plugin
            if not issubclass(plugin_class, ToolPlugin):
                raise PluginLoadError(f"Plugin class must extend ToolPlugin: {package_name}")

            # Get plugin-specific config
            plugin_config = self.config.get(plugin_class.name, {})

            # Instantiate plugin
            plugin = plugin_class(config=plugin_config)

            logger.info(f"Loaded plugin from package: {plugin.name} v{plugin.version}")
            return plugin

        except ImportError as e:
            logger.error(f"Failed to import plugin package {package_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load plugin from package {package_name}: {e}")
            return None

    def register_plugin(self, plugin: ToolPlugin) -> bool:
        """Register and initialize a plugin.

        Args:
            plugin: ToolPlugin instance to register

        Returns:
            True if registration succeeded
        """
        if plugin.name in self.loaded_plugins:
            logger.warning(f"Plugin '{plugin.name}' already registered, skipping")
            return False

        if plugin.name in self._disabled_plugins:
            logger.info(f"Plugin '{plugin.name}' is disabled, skipping")
            return False

        try:
            # Initialize plugin
            plugin._do_initialize()
            self.loaded_plugins[plugin.name] = plugin
            logger.info(f"Registered plugin: {plugin.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize plugin '{plugin.name}': {e}")
            return False

    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin and cleanup its resources.

        Args:
            plugin_name: Name of the plugin to unload

        Returns:
            True if unload succeeded
        """
        if plugin_name not in self.loaded_plugins:
            logger.warning(f"Plugin '{plugin_name}' not loaded")
            return False

        plugin = self.loaded_plugins[plugin_name]

        try:
            plugin._do_cleanup()
            del self.loaded_plugins[plugin_name]
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to unload plugin '{plugin_name}': {e}")
            return False

    def discover_and_load(self) -> int:
        """Discover and load all plugins from configured directories.

        Returns:
            Number of plugins successfully loaded
        """
        discovered = self.discover_plugins()
        loaded_count = 0

        for plugin_path in discovered:
            plugin = self.load_plugin_from_path(plugin_path)
            if plugin and self.register_plugin(plugin):
                loaded_count += 1

        logger.info(f"Loaded {loaded_count}/{len(discovered)} discovered plugins")
        return loaded_count

    def get_all_tools(self) -> List[BaseTool]:
        """Get all tools from all loaded plugins.

        Returns:
            List of BaseTool instances from all plugins
        """
        tools = []
        for plugin in self.loaded_plugins.values():
            tools.extend(plugin._tools)
        return tools

    def register_tools(self, registry: ToolRegistry) -> int:
        """Register all plugin tools with a ToolRegistry.

        Args:
            registry: ToolRegistry to register tools with

        Returns:
            Number of tools registered
        """
        registered = 0
        for plugin in self.loaded_plugins.values():
            for tool in plugin._tools:
                try:
                    registry.register(tool)
                    plugin.on_tool_registered(tool)
                    registered += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to register tool '{tool.name}' from plugin '{plugin.name}': {e}"
                    )

        logger.info(f"Registered {registered} tools from {len(self.loaded_plugins)} plugins")
        return registered

    def get_plugin_info(self) -> List[PluginMetadata]:
        """Get metadata for all loaded plugins.

        Returns:
            List of PluginMetadata for loaded plugins
        """
        return [plugin.get_metadata() for plugin in self.loaded_plugins.values()]

    def disable_plugin(self, plugin_name: str) -> None:
        """Disable a plugin (prevents loading).

        Args:
            plugin_name: Name of the plugin to disable
        """
        self._disabled_plugins.add(plugin_name)
        if plugin_name in self.loaded_plugins:
            self.unload_plugin(plugin_name)

    def enable_plugin(self, plugin_name: str) -> None:
        """Enable a previously disabled plugin.

        Args:
            plugin_name: Name of the plugin to enable
        """
        self._disabled_plugins.discard(plugin_name)

    def reload_plugin(self, plugin_name: str, force_reimport: bool = True) -> bool:
        """Reload a plugin with proper module cache invalidation.

        Performs true hot-reloading by:
        1. Calling plugin cleanup
        2. Removing from loaded plugins
        3. Invalidating Python module cache
        4. Re-importing the plugin module
        5. Re-registering the plugin

        This ensures code changes in plugin files are properly picked up.

        Args:
            plugin_name: Name of the plugin to reload
            force_reimport: If True, invalidate module cache for true hot-reload

        Returns:
            True if reload succeeded
        """
        if plugin_name not in self.loaded_plugins:
            logger.warning(f"Plugin '{plugin_name}' not loaded")
            return False

        plugin = self.loaded_plugins[plugin_name]
        metadata = plugin.get_metadata()

        if not metadata.path:
            logger.warning(f"Cannot reload plugin '{plugin_name}': no path available")
            return False

        plugin_path = metadata.path
        module_name = f"victor_plugin_{plugin_path.name}"

        # Unload plugin (calls cleanup)
        if not self.unload_plugin(plugin_name):
            return False

        # Invalidate module cache for true hot-reload
        if force_reimport:
            self._invalidate_plugin_modules(module_name, plugin_path)

        # Reload plugin
        new_plugin = self.load_plugin_from_path(plugin_path)
        if new_plugin and self.register_plugin(new_plugin):
            logger.info(f"Hot-reloaded plugin: {plugin_name}")
            return True

        logger.error(f"Failed to hot-reload plugin: {plugin_name}")
        return False

    def _invalidate_plugin_modules(self, module_name: str, plugin_path: Path) -> None:
        """Invalidate Python module cache for a plugin and its submodules.

        This ensures that subsequent imports will load fresh code from disk.

        Args:
            module_name: Base module name for the plugin
            plugin_path: Path to plugin directory
        """
        # Remove the main plugin module
        if module_name in sys.modules:
            del sys.modules[module_name]
            logger.debug(f"Invalidated module: {module_name}")

        # Remove any submodules that were imported from the plugin directory
        plugin_path_str = str(plugin_path.resolve())
        modules_to_remove = []

        for mod_name, mod in list(sys.modules.items()):
            if mod is None:
                continue

            # Check if module comes from plugin directory
            try:
                mod_file = getattr(mod, "__file__", None)
                if mod_file and plugin_path_str in str(Path(mod_file).resolve()):
                    modules_to_remove.append(mod_name)
            except Exception:
                continue

        for mod_name in modules_to_remove:
            del sys.modules[mod_name]
            logger.debug(f"Invalidated submodule: {mod_name}")

        # Clear any cached imports
        importlib.invalidate_caches()

    def watch_plugins(self, callback: Optional[Callable[[str, str], None]] = None) -> bool:
        """Start watching plugin directories for changes.

        When a plugin file changes, automatically reload the affected plugin.
        This is useful during development for immediate feedback.

        Args:
            callback: Optional callback(plugin_name, event_type) for change notifications.
                      event_type is one of: "modified", "created", "deleted"

        Returns:
            True if watching started successfully
        """
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler, FileModifiedEvent
        except ImportError:
            logger.warning("watchdog not installed - plugin hot-reload watching unavailable")
            return False

        if hasattr(self, "_plugin_observer") and self._plugin_observer:
            logger.warning("Plugin watcher already running")
            return False

        class PluginFileHandler(FileSystemEventHandler):
            def __init__(handler_self, registry: "ToolPluginRegistry"):
                super().__init__()
                handler_self.registry = registry
                handler_self._debounce_timers: Dict[str, Any] = {}
                handler_self._debounce_delay = 0.5  # 500ms debounce

            def _get_plugin_for_path(handler_self, file_path: str) -> Optional[str]:
                """Find which plugin a file belongs to."""
                file_path_obj = Path(file_path).resolve()
                for name, plugin in handler_self.registry.loaded_plugins.items():
                    plugin_path = plugin.get_metadata().path
                    if plugin_path and str(file_path_obj).startswith(str(plugin_path.resolve())):
                        return name
                return None

            def _schedule_reload(handler_self, plugin_name: str, event_type: str) -> None:
                """Schedule a debounced reload."""
                import threading

                # Cancel existing timer
                if plugin_name in handler_self._debounce_timers:
                    handler_self._debounce_timers[plugin_name].cancel()

                def do_reload():
                    try:
                        if handler_self.registry.reload_plugin(plugin_name):
                            if callback:
                                callback(plugin_name, event_type)
                            logger.info(
                                f"Auto-reloaded plugin '{plugin_name}' due to file {event_type}"
                            )
                    except Exception as e:
                        logger.error(f"Failed to auto-reload plugin '{plugin_name}': {e}")
                    finally:
                        handler_self._debounce_timers.pop(plugin_name, None)

                timer = threading.Timer(handler_self._debounce_delay, do_reload)
                timer.daemon = True
                timer.start()
                handler_self._debounce_timers[plugin_name] = timer

            def on_modified(handler_self, event) -> None:
                if event.is_directory:
                    return
                if not event.src_path.endswith(".py"):
                    return
                plugin_name = handler_self._get_plugin_for_path(event.src_path)
                if plugin_name:
                    handler_self._schedule_reload(plugin_name, "modified")

            def on_created(handler_self, event) -> None:
                if event.is_directory:
                    return
                if not event.src_path.endswith(".py"):
                    return
                plugin_name = handler_self._get_plugin_for_path(event.src_path)
                if plugin_name:
                    handler_self._schedule_reload(plugin_name, "created")

        self._plugin_observer = Observer()
        self._plugin_handler = PluginFileHandler(self)

        # Watch all plugin directories
        for plugin_dir in self.plugin_dirs:
            if plugin_dir.exists():
                self._plugin_observer.schedule(
                    self._plugin_handler, str(plugin_dir), recursive=True
                )
                logger.debug(f"Watching plugin directory: {plugin_dir}")

        self._plugin_observer.start()
        logger.info(f"Started plugin hot-reload watcher for {len(self.plugin_dirs)} directories")
        return True

    def stop_watching(self) -> None:
        """Stop watching plugin directories for changes."""
        if hasattr(self, "_plugin_observer") and self._plugin_observer:
            self._plugin_observer.stop()
            self._plugin_observer.join(timeout=2.0)
            self._plugin_observer = None
            self._plugin_handler = None
            logger.info("Stopped plugin hot-reload watcher")

    def cleanup_all(self) -> None:
        """Cleanup all loaded plugins and stop watching."""
        # Stop file watcher if running
        self.stop_watching()

        # Unload all plugins
        for plugin_name in list(self.loaded_plugins.keys()):
            self.unload_plugin(plugin_name)

    def load(self, plugin: ToolPlugin) -> bool:
        """Load a plugin instance directly.

        This is useful for programmatic plugin loading without discovery.

        Args:
            plugin: ToolPlugin instance to load

        Returns:
            True if loading succeeded
        """
        return self.register_plugin(plugin)

    def unload(self, plugin_name: str) -> bool:
        """Unload a plugin by name.

        Alias for unload_plugin() for API consistency with load().

        Args:
            plugin_name: Name of the plugin to unload

        Returns:
            True if unload succeeded
        """
        return self.unload_plugin(plugin_name)

    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """List all loaded plugins with their metadata.

        Returns:
            Dictionary mapping plugin name to info dict
        """
        result = {}
        for name, plugin in self.loaded_plugins.items():
            result[name] = {
                "version": plugin.version,
                "description": plugin.description,
                "author": plugin.author,
                "tools": [t.name for t in plugin._tools],
                "enabled": name not in self._disabled_plugins,
            }
        return result

    def __enter__(self) -> "ToolPluginRegistry":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup all plugins."""
        self.cleanup_all()


# Backward compatibility alias
ToolPluginManager = ToolPluginRegistry
