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

"""Migration adapters for backward compatibility.

These adapters allow existing code to use UnifiedToolRegistry
without changes, providing a smooth migration path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SharedToolRegistryAdapter:
    """Adapter to make SharedToolRegistry use UnifiedToolRegistry.

    This adapter provides the same interface as SharedToolRegistry
    but delegates to UnifiedToolRegistry internally.

    Existing code using SharedToolRegistry will continue to work:

        from victor.agent.shared_tool_registry import SharedToolRegistry

        registry = SharedToolRegistry.get_instance()
        tools = registry.get_tool_classes()
        instance = registry.create_tool_instance("read_file")

    Under the hood, it uses UnifiedToolRegistry for all operations.
    """

    def __init__(self):
        """Initialize the adapter.

        Gets the UnifiedToolRegistry singleton.
        """
        from victor.tools.unified import UnifiedToolRegistry

        self._unified = UnifiedToolRegistry.get_instance()
        self._initialized = False

    @classmethod
    def get_instance(cls) -> "SharedToolRegistryAdapter":
        """Get the singleton adapter instance.

        Returns:
            SharedToolRegistryAdapter instance
        """
        # Use a class-level singleton pattern
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        if hasattr(cls, "_instance"):
            delattr(cls, "_instance")

    async def get_tool_classes(
        self,
        airgapped_mode: bool = False,
    ) -> Dict[str, Any]:
        """Get all discovered tool classes.

        Args:
            airgapped_mode: Filter out web tools

        Returns:
            Dict mapping tool names to tool classes
        """
        # Ensure discovery
        if not self._initialized:
            await self._unified.discover(airgapped=airgapped_mode)
            self._initialized = True

        # Return tool classes from unified registry
        tools = {}
        for name in self._unified.list_tools(enabled_only=not airgapped_mode):
            tool = self._unified.get(name)
            if tool:
                tools[name] = type(tool)

        return tools

    def get_decorated_tools(
        self,
        airgapped_mode: bool = False,
    ) -> Dict[str, Any]:
        """Get decorated tool functions.

        Args:
            airgapped_mode: Filter out web tools

        Returns:
            Dict mapping tool names to decorated functions
        """
        # UnifiedToolRegistry doesn't separate decorated tools
        # Return all tools for compatibility
        return {}

    def get_tool_names(
        self,
        airgapped_mode: bool = False,
    ) -> List[str]:
        """Get list of all discovered tool names.

        Args:
            airgapped_mode: Filter out web tools

        Returns:
            List of tool names
        """
        return self._unified.list_tools(enabled_only=not airgapped_mode)

    def create_tool_instance(
        self,
        tool_name: str,
    ) -> Optional[Any]:
        """Create a new instance of a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool instance or None
        """
        return self._unified.get(tool_name)

    def get_all_tools_for_registration(
        self,
        airgapped_mode: bool = False,
    ) -> List[Any]:
        """Get tools for registration with ToolRegistry.

        Args:
            airgapped_mode: Filter out web tools

        Returns:
            List of tool instances
        """
        tool_names = self._unified.list_tools(enabled_only=not airgapped_mode)
        return [self._unified.get(name) for name in tool_names if self._unified.get(name)]


class ToolRegistryAdapter:
    """Adapter to make ToolRegistry use UnifiedToolRegistry.

    This adapter provides backward compatibility for ToolRegistry
    by delegating to UnifiedToolRegistry.

    Existing code using ToolRegistry will continue to work:

        from victor.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(my_tool)
        tools = registry.list_tools()
        schemas = registry.get_tool_schemas()
    """

    def __init__(self):
        """Initialize the adapter.

        Gets the UnifiedToolRegistry singleton.
        """
        from victor.tools.unified import UnifiedToolRegistry

        self._unified = UnifiedToolRegistry.get_instance()

        # Import BaseTool for type checks
        from victor.tools.base import BaseTool, ToolResult

        self._BaseTool = BaseTool
        self._ToolResult = ToolResult

    def register(
        self,
        *args,
        enabled: bool = True,
        **kwargs,
    ) -> None:
        """Register a tool.

        Supports both signatures:
        - register(tool) - Auto-extract name
        - register(key, value) - Explicit key-value

        Args:
            *args: Either (tool,) or (key, value)
            enabled: Whether tool is enabled
        """
        if len(args) == 1:
            # Single argument: register(tool)
            tool = args[0]

            # Convert to UnifiedToolRegistry format
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Running in async context, create task
                    asyncio.create_task(self._unified.register(tool, enabled=enabled))
                else:
                    # Not running, run in sync
                    asyncio.run(self._unified.register(tool, enabled=enabled))

            except Exception as e:
                logger.warning(f"Failed to register tool: {e}")

        elif len(args) == 2:
            # Two arguments: register(key, value)
            key, value = args
            # Store directly for explicit key-value registration
            # This is less common but supported
            self._unified._tools[key] = value
        else:
            raise TypeError(
                f"register() takes 1 or 2 positional arguments but {len(args)} were given"
            )

    def unregister(self, name: str) -> bool:
        """Unregister a tool.

        Args:
            name: Tool name

        Returns:
            True if removed
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._unified.unregister(name))
                return True  # Assume success for async
            else:
                return asyncio.run(self._unified.unregister(name))
        except Exception as e:
            logger.warning(f"Failed to unregister tool {name}: {e}")
            return False

    def enable_tool(self, name: str) -> bool:
        """Enable a tool.

        Args:
            name: Tool name

        Returns:
            True if enabled
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._unified.enable(name))
                return True
            else:
                return asyncio.run(self._unified.enable(name))
        except Exception:
            return False

    def disable_tool(self, name: str) -> bool:
        """Disable a tool.

        Args:
            name: Tool name

        Returns:
            True if disabled
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._unified.disable(name))
                return True
            else:
                return asyncio.run(self._unified.disable(name))
        except Exception:
            return False

    def is_tool_enabled(self, name: str) -> bool:
        """Check if a tool is enabled.

        Args:
            name: Tool name

        Returns:
            True if enabled
        """
        metadata = self._unified.get_metadata(name)
        return metadata.enabled if metadata else False

    def get(self, name: str) -> Optional[Any]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None
        """
        return self._unified.get(name)

    def list_tools(self, only_enabled: bool = True) -> List[Any]:
        """List all registered tools.

        Args:
            only_enabled: Only return enabled tools

        Returns:
            List of tool instances
        """
        tool_names = self._unified.list_tools(enabled_only=only_enabled)
        return [self._unified.get(name) for name in tool_names if self._unified.get(name)]

    def get_tool_schemas(self, only_enabled: bool = True) -> List[Dict[str, Any]]:
        """Get JSON schemas for all tools.

        Args:
            only_enabled: Only include enabled tools

        Returns:
            List of tool JSON schemas
        """
        return self._unified.get_schemas(enabled_only=only_enabled)

    @property
    def _tools(self) -> Dict[str, Any]:
        """Alias for backward compatibility."""
        return self._unified._tools


def migrate_to_unified_registry() -> None:
    """Migrate existing code to use UnifiedToolRegistry.

    This function updates the singleton instances of existing
    registries to point to the UnifiedToolRegistry adapters.

    Call this during application initialization to enable
    the unified registry:

        from victor.tools.unified.adapters import migrate_to_unified_registry

        migrate_to_unified_registry()
    """
    # Update SharedToolRegistry
    from victor.agent import shared_tool_registry

    shared_tool_registry.SharedToolRegistry = SharedToolRegistryAdapter
    shared_tool_registry.SharedToolRegistry.get_instance = SharedToolRegistryAdapter.get_instance

    # Note: ToolRegistry is typically instantiated per-session,
    # so we update it via __init__ instead

    logger.info("Migrated to UnifiedToolRegistry")


__all__ = [
    "SharedToolRegistryAdapter",
    "ToolRegistryAdapter",
    "migrate_to_unified_registry",
]
