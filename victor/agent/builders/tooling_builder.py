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

"""Tooling builder for orchestrator initialization.

Part of HIGH-005: Initialization Complexity reduction.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from victor.agent.builders.base import FactoryAwareBuilder

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.orchestrator_factory import OrchestratorFactory
    from victor.providers.base import BaseProvider


class ToolingBuilder(FactoryAwareBuilder):
    """Build tool registry, executor, and access control components."""

    def __init__(self, settings: Any, factory: Optional["OrchestratorFactory"] = None):
        """Initialize the builder.

        Args:
            settings: Application settings
            factory: Optional OrchestratorFactory instance
        """
        super().__init__(settings, factory)

    def build(
        self,
        orchestrator: "AgentOrchestrator",
        provider: "BaseProvider",
        model: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build tooling components and attach them to orchestrator."""
        factory = self._ensure_factory(provider=provider, model=model)
        components: Dict[str, Any] = {}

        # Tool registry (via factory)
        orchestrator.tools = factory.create_tool_registry()
        # Alias for backward compatibility - some code uses tool_registry instead of tools
        orchestrator.tool_registry = orchestrator.tools
        components["tool_registry"] = orchestrator.tools

        # Configure ResponseCoordinator with tool_adapter and mode_coordinator (now available)
        orchestrator._response_coordinator._tool_adapter = orchestrator.tool_adapter
        orchestrator._response_coordinator._tool_registry = orchestrator.tools
        orchestrator._response_coordinator._shell_variant_resolver = orchestrator._mode_coordinator
        orchestrator._response_coordinator._tool_enabled_checker = orchestrator

        # Initialize ToolRegistrar (via factory) - tool registration, plugins, MCP integration
        orchestrator.tool_registrar = factory.create_tool_registrar(
            orchestrator.tools, orchestrator.tool_graph, provider, model
        )
        orchestrator.tool_registrar.set_background_task_callback(
            orchestrator._create_background_task
        )
        components["tool_registrar"] = orchestrator.tool_registrar

        # Register tool dependencies for planning (delegates to ToolRegistrar)
        orchestrator.tool_registrar._register_tool_dependencies()

        # Synchronous registration (dynamic tools, configs)
        orchestrator._register_default_tools()  # Delegates to ToolRegistrar
        orchestrator.tool_registrar._load_tool_configurations()  # Delegates to ToolRegistrar
        orchestrator.tools.register_before_hook(orchestrator._log_tool_call)

        # Plugin system for extensible tools (via factory, delegates to ToolRegistrar)
        orchestrator.plugin_manager = factory.initialize_plugin_system(orchestrator.tool_registrar)
        components["plugin_manager"] = orchestrator.plugin_manager

        # Argument normalizer for handling malformed tool arguments (via factory, DI with fallback)
        orchestrator.argument_normalizer = factory.create_argument_normalizer(provider)
        components["argument_normalizer"] = orchestrator.argument_normalizer

        # Initialize middleware chain from vertical extensions (via factory)
        orchestrator._middleware_chain, orchestrator._code_correction_middleware = (
            factory.create_middleware_chain()
        )
        components["middleware_chain"] = orchestrator._middleware_chain

        # Initialize SafetyChecker with vertical patterns (via factory)
        orchestrator._safety_checker = factory.create_safety_checker()
        components["safety_checker"] = orchestrator._safety_checker

        # Initialize AutoCommitter for AI-assisted commits (via factory)
        orchestrator._auto_committer = factory.create_auto_committer()
        components["auto_committer"] = orchestrator._auto_committer

        # Tool executor for centralized tool execution with retry, caching, and metrics (via factory)
        orchestrator.tool_executor = factory.create_tool_executor(
            tools=orchestrator.tools,
            argument_normalizer=orchestrator.argument_normalizer,
            tool_cache=orchestrator.tool_cache,
            safety_checker=orchestrator._safety_checker,
            code_correction_middleware=orchestrator._code_correction_middleware,
        )
        components["tool_executor"] = orchestrator.tool_executor

        # Parallel tool executor for concurrent independent tool calls (via factory)
        orchestrator.parallel_executor = factory.create_parallel_executor(
            orchestrator.tool_executor
        )
        components["parallel_executor"] = orchestrator.parallel_executor

        # Response completer for ensuring complete responses after tool calls (via factory)
        orchestrator.response_completer = factory.create_response_completer()
        components["response_completer"] = orchestrator.response_completer

        # Initialize UnifiedTaskTracker (via factory, DI with fallback)
        orchestrator.unified_tracker = factory.create_unified_tracker(
            orchestrator._tool_calling_caps_internal
        )
        components["unified_tracker"] = orchestrator.unified_tracker

        # Initialize unified ToolSelector (via factory)
        orchestrator.tool_selector = factory.create_tool_selector(
            tools=orchestrator.tools,
            conversation_state=orchestrator.conversation_state,
            unified_tracker=orchestrator.unified_tracker,
            model=orchestrator.model,
            provider_name=orchestrator.provider_name,
            tool_selection=orchestrator.tool_selection,
            on_selection_recorded=orchestrator._record_tool_selection,
        )
        components["tool_selector"] = orchestrator.tool_selector

        # Initialize ToolAccessController for unified tool access control (via factory)
        orchestrator._tool_access_controller = factory.create_tool_access_controller(
            registry=orchestrator.tools,
        )
        components["tool_access_controller"] = orchestrator._tool_access_controller

        # Initialize ToolAccessConfigCoordinator for tool access configuration (E2 extraction)
        from victor.agent.coordinators.config_coordinator import ToolAccessConfigCoordinator

        orchestrator._tool_access_config_coordinator = ToolAccessConfigCoordinator(
            tool_access_controller=orchestrator._tool_access_controller,
            mode_coordinator=orchestrator._mode_coordinator,
            tool_registry=orchestrator.tools,
        )
        components["tool_access_config_coordinator"] = orchestrator._tool_access_config_coordinator

        # Initialize BudgetManager for unified budget tracking (via factory)
        orchestrator._budget_manager = factory.create_budget_manager()
        components["budget_manager"] = orchestrator._budget_manager

        self._register_components(components)
        return components
