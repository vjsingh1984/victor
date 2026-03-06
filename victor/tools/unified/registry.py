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

"""Unified Tool Registry.

This module provides the UnifiedToolRegistry which consolidates tool
discovery, registration, selection, and lifecycle management into a
single, coherent interface.

Design Principles:
- Singleton pattern for memory efficiency
- Thread-safe operations
- Backward compatibility with existing registries
- Extensible plugin system
- Observability built-in
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Type,
    Union,
    runtime_checkable,
)

from victor.framework.tools import ToolCategory
from victor.tools.base import BaseTool, CostTier, ToolResult
from victor.tools.selection.protocol import (
    CrossVerticalToolSelectionContext,
    ToolSelectionStrategy,
    ToolSelectorFeatures,
)

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Tool selection strategy."""

    AUTO = "auto"  # Automatically choose best strategy
    KEYWORD = "keyword"  # Fast keyword matching
    SEMANTIC = "semantic"  # Embedding-based semantic matching
    HYBRID = "hybrid"  # Combined keyword + semantic


class HookPhase(Enum):
    """Hook execution phase."""

    BEFORE = "before"  # Before tool execution
    AFTER = "after"  # After tool execution
    ERROR = "error"  # On tool error


@dataclass
class ToolMetadata:
    """Metadata for a registered tool.

    Attributes:
        name: Tool name
        description: Tool description
        category: Tool category (FILESYSTEM, GIT, WEB, etc.)
        tier: Cost tier (LOW, MEDIUM, HIGH)
        enabled: Whether tool is enabled
        deprecated: Whether tool is deprecated
        deprecation_message: Deprecation warning
        replacement: Replacement tool name
        aliases: Alternative names for this tool
        tags: User-defined tags
        registered_at: When tool was registered
        version: Tool version
        author: Tool author
    """

    name: str
    description: str = ""
    category: ToolCategory = ToolCategory.CUSTOM
    tier: CostTier = CostTier.MEDIUM
    enabled: bool = True
    deprecated: bool = False
    deprecation_message: str = ""
    replacement: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    registered_at: float = field(default_factory=time.time)
    version: str = ""
    author: str = ""


@dataclass
class ToolMetrics:
    """Usage metrics for a tool.

    Attributes:
        name: Tool name
        call_count: Total times tool was called
        success_count: Successful executions
        error_count: Failed executions
        last_used: Last usage timestamp
        avg_duration_ms: Average execution time
    """

    name: str
    call_count: int = 0
    success_count: int = 0
    error_count: int = 0
    last_used: Optional[float] = None
    avg_duration_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.call_count == 0:
            return 0.0
        return self.success_count / self.call_count

    def record_call(
        self,
        success: bool,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Record a tool execution.

        Args:
            success: Whether execution succeeded
            duration_ms: Execution duration in milliseconds
        """
        self.call_count += 1
        self.last_used = time.time()

        if success:
            self.success_count += 1
        else:
            self.error_count += 1

        if duration_ms is not None:
            # Update running average
            alpha = 2 / (self.call_count + 1)
            self.avg_duration_ms = alpha * duration_ms + (1 - alpha) * self.avg_duration_ms


@runtime_checkable
class ToolHook(Protocol):
    """Protocol for tool execution hooks."""

    async def __call__(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: ToolSelectionContext,
    ) -> None:
        """Execute the hook.

        Args:
            tool_name: Name of tool being executed
            arguments: Tool arguments
            context: Selection context
        """
        ...

    @property
    def phase(self) -> HookPhase:
        """Hook phase."""
        ...

    @property
    def name(self) -> str:
        """Hook name."""
        ...


class UnifiedToolRegistry:
    """Unified registry for tool management.

    Consolidates:
    - Tool discovery from modules
    - Tool registration (class-based, decorated functions, plugins)
    - Tool selection (keyword, semantic, hybrid)
    - Lifecycle management (enable, disable, deprecate)
    - Metadata and categorization
    - Observability (metrics, hooks)

    Thread-safe singleton pattern.

    Example:
        registry = UnifiedToolRegistry.get_instance()

        # Discover tools
        await registry.discover()

        # Register custom tool
        await registry.register(my_tool)

        # Select tools
        tool_names = await registry.select_tools(
            "Read and edit files",
            max_tools=5,
        )

        # Execute tool
        tool = registry.get("read_file")
        result = await tool.execute(file_path="main.py")
    """

    _instance: Optional["UnifiedToolRegistry"] = None
    _lock: threading.Lock = threading.Lock()
    _instantiation_allowed: bool = False

    def __init__(self) -> None:
        """Initialize the registry.

        Raises:
            RuntimeError: If called directly (use get_instance())
        """
        if not UnifiedToolRegistry._instantiation_allowed:
            raise RuntimeError("Use UnifiedToolRegistry.get_instance() to get the singleton")

        # Core storage
        self._tools: Dict[str, BaseTool] = {}
        self._metadata: Dict[str, ToolMetadata] = {}
        self._aliases: Dict[str, str] = {}  # alias -> canonical name
        self._metrics: Dict[str, ToolMetrics] = {}
        self._hooks: List[ToolHook] = []

        # Thread safety
        self._lock: threading.RLock = threading.RLock()
        self._discovery_lock: threading.Lock = threading.Lock()

        # State
        self._discovered: bool = False
        self._discovery_paths: List[str] = []

        # Selection strategies
        self._selector: Optional[ToolSelectionStrategy] = None

        logger.debug("UnifiedToolRegistry instance created")

    # ========================================================================
    # Singleton API
    # ========================================================================

    @classmethod
    def get_instance(cls) -> "UnifiedToolRegistry":
        """Get the singleton instance.

        Returns:
            The UnifiedToolRegistry singleton
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instantiation_allowed = True
                    try:
                        cls._instance = cls()
                    finally:
                        cls._instantiation_allowed = False
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                # Close selector if present
                if cls._instance._selector:
                    try:
                        asyncio.run(cls._instance._selector.close())
                    except Exception:
                        pass
            cls._instance = None
        logger.debug("UnifiedToolRegistry instance reset")

    # ========================================================================
    # Tool Discovery
    # ========================================================================

    async def discover(
        self,
        paths: Optional[List[str]] = None,
        airgapped: bool = False,
    ) -> List[str]:
        """Discover and register tools from module paths.

        Args:
            paths: Module paths to scan (default: victor.tools)
            airgapped: If True, filter out web tools

        Returns:
            List of discovered tool names
        """
        if self._discovered:
            return list(self._tools.keys())

        with self._discovery_lock:
            if self._discovered:
                return list(self._tools.keys())

            # Default paths
            if paths is None:
                paths = ["victor.tools"]

            discovered = []

            for path in paths:
                tool_names = await self._discover_from_path(path, airgapped)
                discovered.extend(tool_names)

            self._discovered = True
            self._discovery_paths = paths

            logger.info(f"UnifiedToolRegistry discovered {len(discovered)} tools from {paths}")

            return discovered

    async def _discover_from_path(
        self,
        path: str,
        airgapped: bool = False,
    ) -> List[str]:
        """Discover tools from a specific path.

        Args:
            path: Module path (e.g., "victor.tools")
            airgapped: Filter web tools

        Returns:
            List of discovered tool names
        """
        discovered = []

        # Import the module
        try:
            import importlib

            module = importlib.import_module(path)

            # Scan for BaseTool subclasses
            for member_name, obj in inspect.getmembers(module):
                if self._is_tool_class(obj):
                    try:
                        instance = obj()
                        tool_name = instance.name

                        # Skip web tools in airgapped mode
                        if airgapped and tool_name in self._get_web_tools():
                            continue

                        await self.register(
                            instance,
                            enabled=True,
                            # Metadata from class properties
                        )

                        discovered.append(tool_name)

                    except Exception as e:
                        logger.debug(f"Failed to instantiate {member_name}: {e}")

        except Exception as e:
            logger.warning(f"Failed to discover tools from {path}: {e}")

        return discovered

    def _is_tool_class(self, obj: Any) -> bool:
        """Check if object is a tool class.

        Args:
            obj: Object to check

        Returns:
            True if it's a tool class
        """
        try:
            return inspect.isclass(obj) and issubclass(obj, BaseTool) and obj is not BaseTool
        except TypeError:
            return False

    def _get_web_tools(self) -> Set[str]:
        """Get set of web tool names to filter in airgapped mode."""
        return {
            "web_search",
            "web_fetch",
            "http_tool",
            "browser_tool",
            "http_request",
        }

    # ========================================================================
    # Tool Registration
    # ========================================================================

    async def register(
        self,
        tool: Union[BaseTool, Callable, ToolDefinition],
        *,
        enabled: bool = True,
        category: Optional[ToolCategory] = None,
        tier: CostTier = CostTier.MEDIUM,
    ) -> None:
        """Register a tool.

        Args:
            tool: Tool instance, decorated function, or tool definition
            enabled: Whether tool is enabled by default
            category: Tool category (auto-detected if None)
            tier: Cost tier for this tool
        """
        tool_instance: BaseTool
        tool_name: str

        # Handle different tool types
        if isinstance(tool, BaseTool):
            tool_instance = tool
            tool_name = tool_instance.name

        elif callable(tool) and hasattr(tool, "Tool"):
            # Decorated function
            tool_instance = tool.Tool
            tool_name = tool_instance.name

        elif isinstance(tool, dict):
            # Tool definition dict
            tool_name = tool.get("name", "")
            tool_instance = self._create_tool_from_dict(tool)

        else:
            raise TypeError(f"Unsupported tool type: {type(tool)}")

        # Store tool
        with self._lock:
            self._tools[tool_name] = tool_instance

            # Create metadata
            if category is None:
                category = self._infer_category(tool_instance)

            self._metadata[tool_name] = ToolMetadata(
                name=tool_name,
                description=tool_instance.description,
                category=category,
                tier=tier,
                enabled=enabled,
            )

            # Initialize metrics
            self._metrics[tool_name] = ToolMetrics(name=tool_name)

        logger.debug(f"Registered tool: {tool_name}")

    async def unregister(self, name: str) -> bool:
        """Unregister a tool.

        Args:
            name: Tool name

        Returns:
            True if tool was removed
        """
        with self._lock:
            removed = name in self._tools

            if removed:
                del self._tools[name]
                del self._metadata[name]
                self._metrics.pop(name, None)

                # Remove aliases
                self._aliases = {k: v for k, v in self._aliases.items() if v != name}

                logger.debug(f"Unregistered tool: {name}")

            return removed

    # ========================================================================
    # Tool Selection
    # ========================================================================

    async def select_tools(
        self,
        query: str,
        *,
        context: Optional[CrossVerticalToolSelectionContext] = None,
        max_tools: int = 10,
        strategy: SelectionStrategy = SelectionStrategy.AUTO,
    ) -> List[str]:
        """Select relevant tools for a query.

        Args:
            query: User query or task description
            context: Optional selection context
            max_tools: Maximum tools to return
            strategy: Selection strategy

        Returns:
            List of tool names, ordered by relevance
        """
        # Ensure tools are discovered
        await self.discover()

        # Get selector if needed
        if strategy == SelectionStrategy.AUTO:
            strategy = self._select_strategy(context)

        # Create context if needed
        if context is None:
            context = CrossVerticalToolSelectionContext(
                prompt=query,
                conversation_history=[],
                max_tools=max_tools,
            )

        # Run selection
        selector = self._get_selector_for_strategy(strategy)

        try:
            tool_names = await selector.select_tools(context, max_tools)

            # Filter by enabled state
            enabled_tools = [
                name
                for name in tool_names
                if self._metadata.get(name, ToolMetadata(name="")).enabled
            ]

            return enabled_tools[:max_tools]

        except Exception as e:
            logger.warning(f"Tool selection failed: {e}")
            # Fallback to empty list
            return []

    def _select_strategy(
        self,
        context: Optional[CrossVerticalToolSelectionContext],
    ) -> SelectionStrategy:
        """Auto-select appropriate strategy.

        Args:
            context: Selection context

        Returns:
            Selected strategy
        """
        # Default to hybrid for best results
        return SelectionStrategy.HYBRID

    def _get_selector_for_strategy(
        self,
        strategy: SelectionStrategy,
    ) -> ToolSelectionStrategy:
        """Get selector implementation for strategy.

        Args:
            strategy: Selection strategy enum

        Returns:
            ToolSelectionStrategy instance
        """
        # Lazy import to avoid circular dependency
        if strategy == SelectionStrategy.SEMANTIC:
            from victor.tools.semantic_selector import SemanticToolSelector

            if self._selector is None:
                self._selector = SemanticToolSelector()

            return self._selector

        elif strategy == SelectionStrategy.HYBRID:
            from victor.tools.hybrid_tool_selector import HybridToolSelector

            if self._selector is None or not isinstance(self._selector, HybridToolSelector):
                self._selector = HybridToolSelector()

            return self._selector

        else:  # KEYWORD or default
            from victor.tools.keyword_tool_selector import KeywordToolSelector

            return KeywordToolSelector()

    # ========================================================================
    # Tool Access
    # ========================================================================

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name.

        Args:
            name: Tool name or alias

        Returns:
            Tool instance or None
        """
        # Resolve alias
        canonical_name = self._aliases.get(name, name)

        return self._tools.get(canonical_name)

    def list_tools(
        self,
        *,
        enabled_only: bool = True,
        category: Optional[ToolCategory] = None,
        tier: Optional[CostTier] = None,
    ) -> List[str]:
        """List tools with optional filtering.

        Args:
            enabled_only: Only return enabled tools
            category: Filter by category
            tier: Filter by cost tier

        Returns:
            List of tool names
        """
        with self._lock:
            tool_names = list(self._tools.keys())

            result = []
            for name in tool_names:
                metadata = self._metadata.get(name)

                if metadata is None:
                    continue

                if enabled_only and not metadata.enabled:
                    continue

                if category and metadata.category != category:
                    continue

                if tier and metadata.tier != tier:
                    continue

                result.append(name)

            return result

    # ========================================================================
    # Metadata & Search
    # ========================================================================

    def get_metadata(self, name: str) -> ToolMetadata:
        """Get metadata for a tool.

        Args:
            name: Tool name

        Returns:
            Tool metadata
        """
        return self._metadata.get(name, ToolMetadata(name=name))

    def get_categories(self) -> Dict[ToolCategory, List[str]]:
        """Get tools grouped by category.

        Returns:
            Dict mapping categories to tool names
        """
        result: Dict[ToolCategory, List[str]] = {}

        for name, metadata in self._metadata.items():
            cat = metadata.category
            if cat not in result:
                result[cat] = []
            result[cat].append(name)

        return result

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    async def enable(self, name: str) -> bool:
        """Enable a tool.

        Args:
            name: Tool name

        Returns:
            True if enabled
        """
        with self._lock:
            metadata = self._metadata.get(name)
            if metadata:
                metadata.enabled = True
                return True
            return False

    async def disable(self, name: str) -> bool:
        """Disable a tool.

        Args:
            name: Tool name

        Returns:
            True if disabled
        """
        with self._lock:
            metadata = self._metadata.get(name)
            if metadata:
                metadata.enabled = False
                return True
            return False

    async def deprecate(
        self,
        name: str,
        replacement: Optional[str] = None,
        message: str = "",
    ) -> None:
        """Mark a tool as deprecated.

        Args:
            name: Tool name
            replacement: Replacement tool name
            message: Deprecation message
        """
        with self._lock:
            metadata = self._metadata.get(name)
            if metadata:
                metadata.deprecated = True
                metadata.replacement = replacement
                metadata.deprecation_message = message

                logger.info(f"Deprecated tool: {name} -> {replacement}")

    # ========================================================================
    # Aliases
    # ========================================================================

    def add_alias(self, name: str, alias: str) -> None:
        """Add an alias for a tool.

        Args:
            name: Canonical tool name
            alias: Alternative name
        """
        with self._lock:
            if name in self._tools:
                self._aliases[alias] = name
                logger.debug(f"Added alias: {alias} -> {name}")

    def resolve_alias(self, alias: str) -> str:
        """Resolve an alias to canonical name.

        Args:
            alias: Alias to resolve

        Returns:
            Canonical tool name
        """
        return self._aliases.get(alias, alias)

    # ========================================================================
    # Schemas
    # ========================================================================

    def get_schemas(
        self,
        *,
        enabled_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get JSON schemas for all tools.

        Args:
            enabled_only: Only include enabled tools

        Returns:
            List of tool JSON schemas
        """
        schemas = []

        for name in self.list_tools(enabled_only=enabled_only):
            tool = self.get(name)
            if tool:
                schema = {
                    "name": name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                schemas.append(schema)

        return schemas

    # ========================================================================
    # Observability
    # ========================================================================

    def record_execution(
        self,
        name: str,
        success: bool,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Record tool execution metrics.

        Args:
            name: Tool name
            success: Whether execution succeeded
            duration_ms: Execution duration
        """
        metrics = self._metrics.get(name)
        if metrics:
            metrics.record_call(success, duration_ms)

    def get_metrics(self, name: str) -> ToolMetrics:
        """Get metrics for a tool.

        Args:
            name: Tool name

        Returns:
            Tool metrics
        """
        return self._metrics.get(name, ToolMetrics(name=name))

    def get_all_metrics(self) -> Dict[str, ToolMetrics]:
        """Get metrics for all tools.

        Returns:
            Dict mapping tool names to metrics
        """
        return self._metrics.copy()

    # ========================================================================
    # Hooks
    # ========================================================================

    async def execute_hooks(
        self,
        phase: HookPhase,
        tool_name: str,
        arguments: Dict[str, Any],
        context: ToolSelectionContext,
    ) -> None:
        """Execute registered hooks.

        Args:
            phase: Hook phase
            tool_name: Tool being executed
            arguments: Tool arguments
            context: Selection context
        """
        for hook in self._hooks:
            if hook.phase == phase:
                try:
                    await hook(tool_name, arguments, context)
                except Exception as e:
                    logger.error(f"Hook {hook.name} failed: {e}")

    # ========================================================================
    # Internal Helpers
    # ========================================================================

    def _create_tool_from_dict(self, definition: Dict[str, Any]) -> BaseTool:
        """Create a tool from definition dict.

        Args:
            definition: Tool definition

        Returns:
            BaseTool instance
        """
        name = definition.get("name", "")
        description = definition.get("description", "")
        parameters = definition.get("parameters", {})

        class DictTool(BaseTool):
            @property
            def name(self) -> str:
                return name

            @property
            def description(self) -> str:
                return description

            @property
            def parameters(self) -> Dict[str, Any]:
                return parameters

            async def execute(self, _exec_ctx, **kwargs) -> ToolResult:
                return ToolResult(
                    success=False,
                    output=None,
                    error="Dict tools should be called via specialized executor",
                )

        return DictTool()

    def _infer_category(self, tool: BaseTool) -> ToolCategory:
        """Infer tool category from name/description.

        Args:
            tool: Tool instance

        Returns:
            Inferred category
        """
        name_lower = tool.name.lower()

        # Category inference based on name patterns
        category_patterns = {
            ToolCategory.FILESYSTEM: ["file", "read", "write", "edit", "directory"],
            ToolCategory.GIT: ["git", "commit", "diff", "branch"],
            ToolCategory.WEB: ["web", "http", "fetch", "browser"],
            ToolCategory.DATABASE: ["database", "db", "sql"],
            ToolCategory.TESTING: ["test", "pytest", "unittest"],
            ToolCategory.DOCKER: ["docker", "container"],
        }

        for category, patterns in category_patterns.items():
            if any(pattern in name_lower for pattern in patterns):
                return category

        return ToolCategory.CUSTOM


__all__ = [
    "UnifiedToolRegistry",
    "SelectionStrategy",
    "HookPhase",
    "ToolMetadata",
    "ToolMetrics",
]
