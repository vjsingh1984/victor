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

"""Shared tool registry for memory-efficient concurrent sessions.

This module provides a singleton registry that shares tool class definitions
across multiple AgentOrchestrator instances. This significantly reduces memory
footprint when running concurrent sessions, as tool classes are discovered
and stored only once.

Design Pattern: Singleton + Flyweight
====================================
- Singleton: Ensures only one SharedToolRegistry instance exists
- Flyweight: Tool classes are shared objects, while per-session state is kept separate

Memory Savings:
- Without sharing: 100 sessions x 61 tools x ~150KB = ~900MB
- With sharing: 1 registry + 100 sessions x per-session state = ~20MB

Usage:
    from victor.agent.shared_tool_registry import SharedToolRegistry

    # Get the singleton instance
    registry = SharedToolRegistry.get_instance()

    # Get all tool classes (cached after first call)
    tool_classes = registry.get_tool_classes()

    # Create a new tool instance (for per-session use if needed)
    tool_instance = registry.create_tool_instance("read_file")

    # For test isolation
    SharedToolRegistry.reset_instance()
"""

import importlib
import inspect
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Set, Type

logger = logging.getLogger(__name__)


# Web tools that should be filtered out in airgapped mode
WEB_TOOL_NAMES: Set[str] = {
    "web_search",
    "web_fetch",
    "http_tool",
    "browser_tool",
    "http_request",
}


class SharedToolRegistry:
    """Singleton registry for sharing tool class definitions across sessions.

    This class provides a memory-efficient way to share tool definitions across
    multiple concurrent AgentOrchestrator instances. Instead of each session
    discovering and storing its own copy of tool classes, they all reference
    the same shared definitions.

    Thread-Safety:
        All operations are thread-safe. The singleton instance is protected by
        a lock, and tool discovery is performed atomically.

    Attributes:
        _instance: The singleton instance
        _lock: Threading lock for thread-safe singleton access
        _tool_classes: Cached dictionary mapping tool names to tool classes
        _decorated_tools: Cached dictionary mapping tool names to decorated functions
        _initialized: Flag indicating if tools have been discovered
    """

    _instance: Optional["SharedToolRegistry"] = None
    _lock: threading.Lock = threading.Lock()
    _instantiation_allowed: bool = False

    def __init__(self) -> None:
        """Initialize the SharedToolRegistry.

        Raises:
            RuntimeError: If called directly instead of via get_instance()
        """
        if not SharedToolRegistry._instantiation_allowed:
            raise RuntimeError(
                "SharedToolRegistry is a singleton. Use SharedToolRegistry.get_instance() "
                "to get the singleton instance."
            )

        self._tool_classes: Dict[str, Type[Any]] = {}
        self._decorated_tools: Dict[str, Any] = {}
        self._initialized: bool = False
        self._discovery_lock: threading.Lock = threading.Lock()

        # OPTIMIZATION: Cache tool instances to avoid repeated instantiation
        # This significantly improves performance when get_all_tools_for_registration()
        # is called multiple times
        self._tool_instances_cache: Optional[Dict[str, Any]] = None

        logger.debug("SharedToolRegistry instance created")

    @classmethod
    def get_instance(cls) -> "SharedToolRegistry":
        """Get the singleton instance of SharedToolRegistry.

        This method is thread-safe and ensures only one instance is created
        even when called concurrently from multiple threads.

        Returns:
            The singleton SharedToolRegistry instance
        """
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    cls._instantiation_allowed = True
                    try:
                        cls._instance = cls()
                    finally:
                        cls._instantiation_allowed = False
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance.

        This method is primarily intended for test isolation. It allows
        tests to start with a fresh SharedToolRegistry instance.

        OPTIMIZATION: Also clears the tool instances cache.

        Note:
            This method is thread-safe but should only be called during
            test setup/teardown, not during normal operation.
        """
        with cls._lock:
            # Clear instance cache before resetting
            if cls._instance is not None:
                cls._instance._tool_instances_cache = None
            cls._instance = None
            logger.debug("SharedToolRegistry instance reset")

    def get_tool_classes(self, airgapped_mode: bool = False) -> Dict[str, Type[Any]]:
        """Get all discovered tool classes.

        This method returns the cached dictionary of tool classes. If tools
        haven't been discovered yet, discovery is performed lazily.

        Args:
            airgapped_mode: If True, filter out web tools that require network

        Returns:
            Dictionary mapping tool names to their corresponding classes.
            For tools defined with @tool decorator, the class is retrieved
            from the .Tool attribute.

        Note:
            The returned dictionary should NOT be modified. If you need to
            filter tools, create a copy first.
        """
        if not self._initialized:
            with self._discovery_lock:
                if not self._initialized:
                    self._discover_tools()
                    self._initialized = True

        if airgapped_mode:
            return {
                name: cls for name, cls in self._tool_classes.items() if name not in WEB_TOOL_NAMES
            }

        return self._tool_classes

    def get_decorated_tools(self, airgapped_mode: bool = False) -> Dict[str, Any]:
        """Get all discovered decorated tool functions.

        This method returns decorated functions (those using @tool decorator)
        separately from class-based tools.

        Args:
            airgapped_mode: If True, filter out web tools that require network

        Returns:
            Dictionary mapping tool names to decorated functions
        """
        if not self._initialized:
            with self._discovery_lock:
                if not self._initialized:
                    self._discover_tools()
                    self._initialized = True

        if airgapped_mode:
            return {
                name: func
                for name, func in self._decorated_tools.items()
                if name not in WEB_TOOL_NAMES
            }

        return self._decorated_tools

    def get_tool_names(self, airgapped_mode: bool = False) -> List[str]:
        """Get list of all discovered tool names.

        Args:
            airgapped_mode: If True, filter out web tools that require network

        Returns:
            List of tool names
        """
        return list(self.get_tool_classes(airgapped_mode=airgapped_mode).keys())

    def create_tool_instance(self, tool_name: str) -> Optional[Any]:
        """Create a new instance of a tool.

        This method creates a fresh instance of the specified tool. Use this
        when you need a per-session tool instance with its own state.

        Args:
            tool_name: Name of the tool to instantiate

        Returns:
            New instance of the tool, or None if tool not found
        """
        # Ensure tools are discovered
        self.get_tool_classes()

        tool_class = self._tool_classes.get(tool_name)
        if tool_class is None:
            return None

        try:
            return tool_class()
        except Exception as e:
            logger.debug(f"Failed to create instance of {tool_name}: {e}")
            return None

    def _discover_tools(self) -> None:
        """Discover and cache all tool classes from victor/tools directory.

        This method scans the victor/tools directory for tool definitions and
        stores references to their classes. It handles both:
        - Classes inheriting from BaseTool
        - Functions decorated with @tool

        Note:
            This method is called lazily on first access and only runs once
            (per singleton instance). Results are cached in _tool_classes.
        """
        # Import BaseTool for isinstance checks
        from victor.tools.base import BaseTool as BaseToolClass

        tools_dir = os.path.join(os.path.dirname(__file__), "..", "tools")
        excluded_files = {
            "__init__.py",
            "base.py",
            "decorators.py",
            "semantic_selector.py",
            "registry.py",
            "enums.py",
            "metadata.py",
            "metadata_registry.py",
            "selection_filters.py",
            "selection_common.py",
            "keyword_tool_selector.py",
            "hybrid_tool_selector.py",
            "output_utils.py",
            "shared_ast_utils.py",
            "common.py",
            "subprocess_executor.py",
            "tool_names.py",
            "composition.py",
            "plugin.py",
            "plugin_registry.py",
            "language_analyzer.py",
            "dependency_graph.py",
        }

        discovered_classes = 0
        discovered_decorated = 0

        for filename in os.listdir(tools_dir):
            if filename.endswith(".py") and filename not in excluded_files:
                module_name = f"victor.tools.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for member_name, obj in inspect.getmembers(module):
                        # Handle @tool decorated functions
                        if inspect.isfunction(obj) and getattr(obj, "_is_tool", False):
                            if hasattr(obj, "Tool"):
                                tool_instance = getattr(
                                    obj, "Tool"
                                )  # Dynamically attached attribute
                                tool_name = tool_instance.name
                                # Store the class, not the instance
                                self._tool_classes[tool_name] = type(tool_instance)
                                self._decorated_tools[tool_name] = obj
                                discovered_decorated += 1

                        # Handle BaseTool class instances
                        elif (
                            inspect.isclass(obj)
                            and issubclass(obj, BaseToolClass)
                            and obj is not BaseToolClass
                            and hasattr(obj, "name")
                        ):
                            # Only store the class reference if it's not abstract
                            try:
                                # Attempt to create an instance to get the name
                                temp_instance = obj()
                                tool_name = temp_instance.name
                                self._tool_classes[tool_name] = obj
                                discovered_classes += 1
                            except Exception:
                                # Skip abstract classes or classes that can't be instantiated
                                pass

                except Exception as e:
                    logger.debug(f"Failed to load tools from {module_name}: {e}")

        logger.info(
            f"SharedToolRegistry discovered {len(self._tool_classes)} tools "
            f"({discovered_classes} classes, {discovered_decorated} decorated)"
        )

    def get_all_tools_for_registration(self, airgapped_mode: bool = False) -> List[Any]:
        """Get tool instances and decorated functions for registration.

        This method returns a list suitable for registering with ToolRegistry.
        It includes both class-based tool instances and decorated functions.

        OPTIMIZATION: Caches tool instances to avoid repeated instantiation.

        Args:
            airgapped_mode: If True, filter out web tools that require network

        Returns:
            List of tool instances and decorated functions ready for registration
        """
        from typing import cast

        # Ensure tools are discovered
        self.get_tool_classes()

        # OPTIMIZATION: Use cached instances if available and airgapped_mode matches
        # Check if we need to rebuild the cache (first call or airgapped_mode changed)
        cache_key = "airgapped" if airgapped_mode else "full"

        # Simple cache invalidation check
        if self._tool_instances_cache is None:
            self._tool_instances_cache = {}

        if cache_key not in self._tool_instances_cache:
            result: List[Any] = []

            # Add decorated functions first
            decorated_names = set(self._decorated_tools.keys())
            for name, func in self._decorated_tools.items():
                if not airgapped_mode or not name.startswith("web_"):
                    result.append(func)

            # Add class instances (avoiding duplicates with decorated tools)
            decorated_names = set(self._decorated_tools.keys())
            for name, cls in self.get_tool_classes(airgapped_mode=airgapped_mode).items():
                if name not in decorated_names:
                    try:
                        instance = cls()
                        result.append(instance)
                    except Exception as e:
                        logger.debug(f"Skipped creating instance of {name}: {e}")

            # Cache the result
            self._tool_instances_cache[cache_key] = result
            logger.debug(
                f"Built and cached {len(result)} tools for registration (mode={cache_key})"
            )
        else:
            logger.debug(f"Using cached tool instances (mode={cache_key})")

        return cast(List[Any], self._tool_instances_cache.get(cache_key, []))
