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

"""Tool registration, plugin loading, and MCP integration.

This module encapsulates all tool registration functionality:
- Dynamic tool discovery from victor/tools directory
- Plugin system management
- MCP (Model Context Protocol) integration
- Tool dependency graph setup
- Tool configuration loading

Design Pattern: Registry Pattern + Facade
=========================================
ToolRegistrar acts as a facade for multiple registration systems:
- ToolRegistry: Core tool registration
- ToolPluginRegistry: Plugin management
- MCPRegistry: MCP server discovery
- ToolDependencyGraph: Tool planning

Usage:
    from victor.agent.tool_registrar import ToolRegistrar

    registrar = ToolRegistrar(
        tools=tool_registry,
        settings=settings,
        provider=provider,
        model=model,
    )

    # Register all tools
    await registrar.initialize()

    # Get registered tool count
    stats = registrar.get_stats()
"""

import asyncio
import importlib
import inspect
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from victor.providers.base import BaseProvider, ToolDefinition
from victor.tools.base import ToolRegistry, CostTier

logger = logging.getLogger(__name__)


@dataclass
class ToolRegistrarConfig:
    """Configuration for ToolRegistrar.

    Attributes:
        enable_plugins: Enable plugin system
        enable_mcp: Enable MCP integration
        enable_tool_graph: Enable tool dependency graph
        airgapped_mode: Disable web tools
        plugin_dirs: Additional plugin directories
        disabled_plugins: Plugin names to disable
        plugin_packages: Package names to load plugins from
        max_workers: Max workers for batch processing
        max_complexity: Max complexity for code review
    """

    enable_plugins: bool = True
    enable_mcp: bool = False
    enable_tool_graph: bool = True
    airgapped_mode: bool = False
    plugin_dirs: List[str] = field(default_factory=list)
    disabled_plugins: Set[str] = field(default_factory=set)
    plugin_packages: List[str] = field(default_factory=list)
    max_workers: int = 4
    max_complexity: int = 10


@dataclass
class RegistrationStats:
    """Statistics about tool registration.

    Attributes:
        dynamic_tools: Tools registered from victor/tools
        plugin_tools: Tools registered from plugins
        mcp_tools: Tools registered from MCP servers
        dependency_graph_tools: Tools registered in dependency graph
        total_tools: Total registered tools
        plugins_loaded: Number of plugins loaded
        mcp_servers_connected: Number of MCP servers connected
    """

    dynamic_tools: int = 0
    plugin_tools: int = 0
    mcp_tools: int = 0
    dependency_graph_tools: int = 0
    total_tools: int = 0
    plugins_loaded: int = 0
    mcp_servers_connected: int = 0


class ToolRegistrar:
    """Handles tool registration, plugins, and MCP integration.

    Encapsulates all tool registration functionality:
    - Dynamic tool discovery from victor/tools directory
    - Plugin system management
    - MCP (Model Context Protocol) integration
    - Tool dependency graph setup
    - Tool configuration loading

    This component follows the Registry Pattern combined with Facade pattern,
    providing a unified interface for multiple registration systems.

    Usage:
        registrar = ToolRegistrar(
            tools=tool_registry,
            settings=settings,
            provider=provider,
            model=model,
        )
        await registrar.initialize()
    """

    def __init__(
        self,
        tools: ToolRegistry,
        settings: Any,
        provider: Optional[BaseProvider] = None,
        model: Optional[str] = None,
        tool_graph: Optional[Any] = None,
        config: Optional[ToolRegistrarConfig] = None,
    ):
        """Initialize the tool registrar.

        Args:
            tools: Tool registry to register tools with
            settings: Application settings
            provider: LLM provider for tool configuration
            model: Model name for tool configuration
            tool_graph: Optional tool dependency graph
            config: Optional configuration
        """
        self.tools = tools
        self.settings = settings
        self.provider = provider
        self.model = model
        self.tool_graph = tool_graph
        self.config = config or ToolRegistrarConfig()

        # Plugin system
        self.plugin_manager: Optional[Any] = None

        # MCP integration
        self.mcp_registry: Optional[Any] = None
        self._mcp_tasks: List[asyncio.Task] = []

        # Statistics
        self._stats = RegistrationStats()

        # Background task callback
        self._create_background_task: Optional[Callable] = None

        # Tool configuration for context injection (populated in _setup_providers)
        self._tool_config: Dict[str, Any] = {}

        logger.debug("ToolRegistrar initialized")

    def set_background_task_callback(self, callback: Callable[[Any, str], asyncio.Task]) -> None:
        """Set callback for creating background tasks.

        Args:
            callback: Function that takes (coroutine, name) and returns Task
        """
        self._create_background_task = callback

    def _create_task(self, coro: Any, name: str) -> Optional[asyncio.Task]:
        """Create a background task using callback or directly."""
        if self._create_background_task:
            return self._create_background_task(coro, name)
        else:
            task = asyncio.create_task(coro)
            self._mcp_tasks.append(task)
            return task

    async def initialize(self) -> RegistrationStats:
        """Initialize all tool registration.

        Performs:
        1. Pre-registration setup (providers, configs)
        2. Dynamic tool discovery and registration
        3. Plugin loading (if enabled)
        4. MCP integration (if enabled)
        5. Tool dependency graph setup (if enabled)

        Returns:
            RegistrationStats with registration counts
        """
        # Pre-registration setup
        self._setup_providers()

        # Dynamic tool registration
        self._stats.dynamic_tools = self._register_dynamic_tools()

        # Load tool configurations
        self._load_tool_configurations()

        # Plugin loading
        if self.config.enable_plugins:
            self._stats.plugin_tools = self._initialize_plugins()
            self._stats.plugins_loaded = (
                len(self.plugin_manager.loaded_plugins) if self.plugin_manager else 0
            )

        # MCP integration
        if self.config.enable_mcp or getattr(self.settings, "use_mcp_tools", False):
            self._stats.mcp_tools = self._setup_mcp_integration()

        # Tool dependency graph
        if self.config.enable_tool_graph and self.tool_graph:
            self._stats.dependency_graph_tools = self._register_tool_dependencies()

        # Calculate totals
        self._stats.total_tools = len(self.tools.list_tools())

        logger.info(
            f"Tool registration complete: {self._stats.total_tools} tools "
            f"(dynamic: {self._stats.dynamic_tools}, plugins: {self._stats.plugin_tools}, "
            f"MCP: {self._stats.mcp_tools})"
        )

        return self._stats

    def _setup_providers(self) -> None:
        """Set up provider configuration for tool context injection.

        Note: As of Phase 7, tools receive configuration via context parameter
        at execution time rather than via global setters. This method now stores
        configuration that will be injected into the tool execution context.
        """
        # Store configuration for context injection during tool execution
        # This config will be passed to ToolConfig.from_context() at runtime
        self._tool_config = {
            "provider": self.provider,
            "model": self.model,
            "max_workers": self.config.max_workers,
            "max_complexity": self.config.max_complexity,
        }

        # Load web tool config if not air-gapped
        if not self.config.airgapped_mode:
            try:
                tool_config = self.settings.load_tool_config()
                web_cfg = tool_config.get("web_tools", {}) or tool_config.get("web", {}) or {}
                self._tool_config.update({
                    "web_fetch_top": web_cfg.get("summarize_fetch_top"),
                    "web_fetch_pool": web_cfg.get("summarize_fetch_pool"),
                    "max_content_length": web_cfg.get("summarize_max_content_length"),
                })
            except Exception as exc:
                logger.debug(f"Failed to load web tool config: {exc}")

    def _register_dynamic_tools(self) -> int:
        """Dynamically discover and register tools from victor/tools directory.

        Returns:
            Number of tools registered
        """
        tools_dir = os.path.join(os.path.dirname(__file__), "..", "tools")
        excluded_files = {
            "__init__.py",
            "base.py",
            "decorators.py",
            "semantic_selector.py",
        }
        registered_count = 0

        # Import BaseTool for isinstance checks
        from victor.tools.base import BaseTool as BaseToolClass

        for filename in os.listdir(tools_dir):
            if filename.endswith(".py") and filename not in excluded_files:
                module_name = f"victor.tools.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for _name, obj in inspect.getmembers(module):
                        # Register @tool decorated functions
                        if inspect.isfunction(obj) and getattr(obj, "_is_tool", False):
                            self.tools.register(obj)
                            registered_count += 1
                        # Register BaseTool class instances
                        elif (
                            inspect.isclass(obj)
                            and issubclass(obj, BaseToolClass)
                            and obj is not BaseToolClass
                            and hasattr(obj, "name")
                        ):
                            try:
                                tool_instance = obj()
                                self.tools.register(tool_instance)
                                registered_count += 1
                            except Exception as e:
                                logger.debug(f"Skipped registering {_name}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to load tools from {module_name}: {e}")

        logger.debug(f"Dynamically registered {registered_count} tools")
        return registered_count

    def _load_tool_configurations(self) -> None:
        """Load tool configurations from settings.

        Applies enabled/disabled states to tools based on configuration.
        """
        try:
            tool_config = self.settings.load_tool_config()

            if not tool_config:
                return

            # Handle enabled list
            enabled_list = tool_config.get("enabled", [])
            if enabled_list:
                # Disable all tools not in enabled list
                for tool in self.tools.list_tools():
                    if tool.name not in enabled_list:
                        self.tools.disable_tool(tool.name)
                logger.debug(f"Enabled {len(enabled_list)} tools from config")

            # Handle disabled list
            disabled_list = tool_config.get("disabled", [])
            for tool_name in disabled_list:
                self.tools.disable_tool(tool_name)
            if disabled_list:
                logger.debug(f"Disabled {len(disabled_list)} tools from config")

            # Handle per-tool configurations
            for tool_name, config in tool_config.items():
                if isinstance(config, dict) and "enabled" in config:
                    if not config["enabled"]:
                        self.tools.disable_tool(tool_name)

        except Exception as e:
            logger.warning(f"Failed to load tool configurations: {e}")

    def _initialize_plugins(self) -> int:
        """Initialize and load tool plugins.

        Returns:
            Number of plugin tools registered
        """
        try:
            from victor.tools.plugin_registry import ToolPluginRegistry
            from victor.config.settings import get_project_paths

            # Use centralized path for plugins directory
            plugin_dirs = [get_project_paths().global_plugins_dir]
            plugin_dirs.extend(self.config.plugin_dirs)

            plugin_config = getattr(self.settings, "plugin_config", {})

            self.plugin_manager = ToolPluginRegistry(
                plugin_dirs=plugin_dirs,
                config=plugin_config,
            )

            # Disable specified plugins
            for plugin_name in self.config.disabled_plugins:
                self.plugin_manager.disable_plugin(plugin_name)

            # Discover and load plugins from directories
            loaded_count = self.plugin_manager.discover_and_load()

            # Load plugins from packages
            for package_name in self.config.plugin_packages:
                plugin = self.plugin_manager.load_plugin_from_package(package_name)
                if plugin:
                    self.plugin_manager.register_plugin(plugin)

            # Register plugin tools
            tool_count = 0
            if loaded_count > 0 or self.plugin_manager.loaded_plugins:
                tool_count = self.plugin_manager.register_tools(self.tools)
                logger.info(
                    f"Plugins loaded: {len(self.plugin_manager.loaded_plugins)} plugins, "
                    f"{tool_count} tools"
                )

            return tool_count

        except Exception as e:
            logger.warning(f"Failed to initialize plugin system: {e}")
            self.plugin_manager = None
            return 0

    def _setup_mcp_integration(self) -> int:
        """Set up MCP integration using registry or legacy client.

        Returns:
            Number of MCP tools registered
        """
        mcp_command = getattr(self.settings, "mcp_command", None)
        mcp_tools_registered = 0

        # Try MCPRegistry with auto-discovery first
        try:
            from victor.mcp.registry import MCPRegistry, MCPServerConfig

            # Auto-discover MCP servers from standard locations
            self.mcp_registry = MCPRegistry.discover_servers()

            # Also register command from settings if specified
            if mcp_command:
                cmd_parts = mcp_command.split()
                self.mcp_registry.register_server(
                    MCPServerConfig(
                        name="settings_mcp",
                        command=cmd_parts,
                        description="MCP server from settings",
                        auto_connect=True,
                    )
                )

            # Start registry and connect to servers in background
            if self.mcp_registry.list_servers():
                logger.info(
                    f"MCP Registry initialized with {len(self.mcp_registry.list_servers())} server(s)"
                )
                self._create_task(self._start_mcp_registry(), "mcp_registry_start")

        except ImportError:
            logger.debug("MCPRegistry not available, using legacy client")
            self._setup_legacy_mcp(mcp_command)

        # Register MCP tool definitions
        try:
            from victor.tools.mcp_bridge_tool import get_mcp_tool_definitions

            for mcp_tool in get_mcp_tool_definitions():
                self.tools.register_dict(mcp_tool)
                mcp_tools_registered += 1
        except ImportError:
            logger.debug("MCP bridge tool not available")

        return mcp_tools_registered

    def _setup_legacy_mcp(self, mcp_command: Optional[str]) -> None:
        """Set up legacy single MCP client (backwards compatibility)."""
        if mcp_command:
            try:
                from victor.mcp.client import MCPClient
                from victor.tools.mcp_bridge_tool import configure_mcp_client

                mcp_client = MCPClient()
                cmd_parts = mcp_command.split()
                self._create_task(mcp_client.connect(cmd_parts), "mcp_legacy_connect")
                configure_mcp_client(mcp_client, prefix=getattr(self.settings, "mcp_prefix", "mcp"))
            except Exception as exc:
                logger.warning(f"Failed to start MCP client: {exc}")

    async def _start_mcp_registry(self) -> None:
        """Start MCP registry and connect to discovered servers."""
        try:
            await self.mcp_registry.start()
            results = await self.mcp_registry.connect_all()
            connected = sum(1 for v in results.values() if v)
            self._stats.mcp_servers_connected = connected
            if connected > 0:
                logger.info(f"Connected to {connected} MCP server(s)")
                # Update available tools from MCP
                mcp_tools = self.mcp_registry.get_all_tools()
                if mcp_tools:
                    logger.info(f"Discovered {len(mcp_tools)} MCP tools")
        except Exception as e:
            logger.warning(f"Failed to start MCP registry: {e}")

    def _register_tool_dependencies(self) -> int:
        """Register tool input/output specs for planning with cost tiers.

        Returns:
            Number of tools registered in dependency graph
        """
        if not self.tool_graph:
            return 0

        registered = 0
        try:
            # Search tools - FREE tier (local operations)
            self.tool_graph.add_tool(
                "code_search",
                inputs=["query"],
                outputs=["file_candidates"],
                cost_tier=CostTier.FREE,
            )
            registered += 1

            self.tool_graph.add_tool(
                "semantic_code_search",
                inputs=["query"],
                outputs=["file_candidates"],
                cost_tier=CostTier.FREE,
            )
            registered += 1

            # File operations - FREE tier
            self.tool_graph.add_tool(
                "read_file",
                inputs=["file_candidates"],
                outputs=["file_contents"],
                cost_tier=CostTier.FREE,
            )
            registered += 1

            # Analysis tools - LOW tier
            self.tool_graph.add_tool(
                "analyze_docs",
                inputs=["file_contents"],
                outputs=["summary"],
                cost_tier=CostTier.LOW,
            )
            registered += 1

            self.tool_graph.add_tool(
                "code_review",
                inputs=["file_contents"],
                outputs=["summary"],
                cost_tier=CostTier.LOW,
            )
            registered += 1

            self.tool_graph.add_tool(
                "generate_docs",
                inputs=["file_contents"],
                outputs=["documentation"],
                cost_tier=CostTier.LOW,
            )
            registered += 1

            self.tool_graph.add_tool(
                "security_scan",
                inputs=["file_contents"],
                outputs=["security_report"],
                cost_tier=CostTier.LOW,
            )
            registered += 1

            self.tool_graph.add_tool(
                "analyze_metrics",
                inputs=["file_contents"],
                outputs=["metrics_report"],
                cost_tier=CostTier.LOW,
            )
            registered += 1

        except Exception as exc:
            logger.debug(f"Failed to register tool dependencies: {exc}")

        return registered

    def plan_tools(
        self, goals: List[str], available_inputs: Optional[List[str]] = None
    ) -> List[ToolDefinition]:
        """Plan a sequence of tools to satisfy goals using the dependency graph.

        Args:
            goals: List of goals to achieve
            available_inputs: Already available inputs

        Returns:
            List of ToolDefinition objects for the planned tools
        """
        if not goals or not self.tool_graph:
            return []

        available = available_inputs or []
        plan_names = self.tool_graph.plan(goals, available)
        tool_defs: List[ToolDefinition] = []

        for name in plan_names:
            tool = self.tools.get(name)
            if tool and self.tools.is_tool_enabled(name):
                tool_defs.append(
                    ToolDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters,
                    )
                )
        return tool_defs

    def infer_goals_from_message(self, user_message: str) -> List[str]:
        """Infer planning goals from user request.

        Args:
            user_message: User's message

        Returns:
            List of inferred goal names
        """
        text = user_message.lower()
        goals: List[str] = []

        if any(kw in text for kw in ["summarize", "summary", "analyze", "overview"]):
            goals.append("summary")
        if any(kw in text for kw in ["review", "code review", "audit"]):
            goals.append("summary")
        if any(kw in text for kw in ["doc", "documentation", "readme"]):
            goals.append("documentation")
        if any(kw in text for kw in ["security", "vulnerability", "secret", "scan"]):
            goals.append("security_report")
        if any(kw in text for kw in ["complexity", "metrics", "maintainability", "technical debt"]):
            goals.append("metrics_report")

        return goals

    def get_stats(self) -> RegistrationStats:
        """Get registration statistics.

        Returns:
            RegistrationStats with all counts
        """
        return self._stats

    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool configuration for context injection.

        Returns:
            Dictionary with provider, model, and tool-specific settings
            that should be included in tool execution context.
        """
        return self._tool_config.copy()

    def get_plugin_info(self) -> Dict[str, Any]:
        """Get information about loaded plugins.

        Returns:
            Dictionary with plugin information
        """
        if not self.plugin_manager:
            return {"plugins": [], "total": 0}

        return {
            "plugins": [
                {
                    "name": p.name,
                    "version": getattr(p, "version", "unknown"),
                    "tools": len(p.get_tools()) if hasattr(p, "get_tools") else 0,
                }
                for p in self.plugin_manager.loaded_plugins.values()
            ],
            "total": len(self.plugin_manager.loaded_plugins),
        }

    def get_mcp_info(self) -> Dict[str, Any]:
        """Get information about MCP servers.

        Returns:
            Dictionary with MCP server information
        """
        if not self.mcp_registry:
            return {"servers": [], "connected": 0, "total": 0}

        servers = self.mcp_registry.list_servers()
        return {
            "servers": [
                {
                    "name": s.name,
                    "description": s.description,
                    "connected": s.name in getattr(self.mcp_registry, "_connected_servers", {}),
                }
                for s in servers
            ],
            "connected": self._stats.mcp_servers_connected,
            "total": len(servers),
        }

    async def shutdown(self) -> None:
        """Shutdown MCP connections and cleanup."""
        # Cancel pending MCP tasks
        for task in self._mcp_tasks:
            if not task.done():
                task.cancel()

        # Shutdown MCP registry
        if self.mcp_registry:
            try:
                await self.mcp_registry.shutdown()
            except Exception as e:
                logger.debug(f"Error shutting down MCP registry: {e}")

        logger.debug("ToolRegistrar shutdown complete")
