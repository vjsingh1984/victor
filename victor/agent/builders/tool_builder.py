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

"""ToolBuilder for building tool components.

This module provides a builder for tool-related components used by
AgentOrchestrator, including:
- Tool registry
- Tool registrar (plugins, MCP)
- Tool pipeline
- Tool executor
- Tool selector
- Tool cache
- Tool dependency graph

Part of HIGH-005: Initialization Complexity reduction.
"""

from typing import Any, Callable, Dict, Optional
from victor.agent.builders.base import FactoryAwareBuilder


class ToolBuilder(FactoryAwareBuilder):
    """Build tools and tool-related components.

    This builder creates all tool-related components that the orchestrator
    needs for tool execution. It delegates to OrchestratorFactory for the
    actual component creation while providing a cleaner, more focused API.

    Components built:
        - tools: ToolRegistry with registered tools
        - tool_registrar: ToolRegistrar for plugins and MCP
        - tool_pipeline: ToolPipeline for execution coordination
        - tool_executor: ToolExecutor for centralized execution
        - tool_selector: ToolSelector for semantic + keyword selection
        - tool_cache: ResultCache for idempotent tools
        - tool_graph: ToolDependencyGraph
        - argument_normalizer: ArgumentNormalizer
        - semantic_selector: SemanticToolSelector (optional)
        - parallel_executor: ParallelToolExecutor
        - response_completer: ResponseCompleter
        - unified_tracker: UnifiedTaskTracker
        - deduplication_tracker: ToolDeduplicationTracker
        - plugin_manager: PluginManager
        - usage_logger: UsageLogger
        - debug_logger: DebugLogger
        - safety_checker: SafetyChecker
        - auto_committer: AutoCommitter
        - code_correction_middleware: CodeCorrectionMiddleware
        - middleware_chain: List of middleware
        - tool_output_formatter: ToolOutputFormatter
        - workflow_registry: WorkflowRegistry
        - code_manager: CodeExecutionManager
    """

    def build(
        self,
        provider: Any = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        provider_name: Optional[str] = None,
        profile_name: Optional[str] = None,
        tool_selection: Optional[Dict[str, Any]] = None,
        thinking: bool = False,
        tool_calling_caps: Any = None,
        tool_adapter: Any = None,
        service_provider: Any = None,
        conversation_state: Any = None,
        unified_tracker: Any = None,
        context_compactor: Any = None,
        on_tool_start: Optional[Callable] = None,
        on_tool_complete: Optional[Callable] = None,
        background_task_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build all tool components.

        Args:
            provider: LLM provider instance (needed for some tools)
            model: Model identifier (needed for some tools)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            provider_name: Optional provider label from profile
            profile_name: Optional profile name for session tracking
            tool_selection: Optional tool selection configuration
            thinking: Enable extended thinking mode
            tool_calling_caps: ToolCallingCapabilities from provider builder
            tool_adapter: BaseToolCallingAdapter from provider builder
            service_provider: DI container from service builder
            conversation_state: ConversationStateMachine from service builder
            unified_tracker: UnifiedTaskTracker (built if not provided)
            context_compactor: ContextCompactor from service builder
            on_tool_start: Callback for tool start
            on_tool_complete: Callback for tool complete
            background_task_callback: Callback for background tasks
            **kwargs: Additional dependencies

        Returns:
            Dictionary of built tool components with keys:
            - tools: ToolRegistry
            - tool_registrar: ToolRegistrar
            - tool_pipeline: ToolPipeline
            - tool_executor: ToolExecutor
            - tool_selector: ToolSelector
            - tool_cache: ResultCache
            - tool_graph: ToolDependencyGraph
            - argument_normalizer: ArgumentNormalizer
            - semantic_selector: SemanticToolSelector (optional)
            - parallel_executor: ParallelToolExecutor
            - response_completer: ResponseCompleter
            - unified_tracker: UnifiedTaskTracker
            - deduplication_tracker: ToolDeduplicationTracker
            - plugin_manager: PluginManager
            - usage_logger: UsageLogger
            - debug_logger: DebugLogger
            - safety_checker: SafetyChecker
            - auto_committer: AutoCommitter
            - code_correction_middleware: CodeCorrectionMiddleware
            - middleware_chain: List of middleware
            - tool_output_formatter: ToolOutputFormatter
            - workflow_registry: WorkflowRegistry
            - code_manager: CodeExecutionManager
        """
        tool_components = {}

        # Create or reuse factory
        self._ensure_factory(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            provider_name=provider_name,
            profile_name=profile_name,
            tool_selection=tool_selection,
            thinking=thinking,
        )

        # Build tool budget
        if tool_calling_caps:
            tool_budget = self._factory.initialize_tool_budget(tool_calling_caps)
        else:
            tool_budget = kwargs.get("tool_budget", 10)
        tool_components["tool_budget"] = tool_budget

        # Build tool cache
        tool_cache = self._factory.create_tool_cache()
        tool_components["tool_cache"] = tool_cache

        # Build tool dependency graph
        tool_graph = self._factory.create_tool_dependency_graph()
        tool_components["tool_graph"] = tool_graph

        # Build tool registry
        tools = self._factory.create_tool_registry()
        tool_components["tools"] = tools

        # Build usage logger
        usage_logger = self._factory.create_usage_logger()
        tool_components["usage_logger"] = usage_logger

        # Build debug logger
        debug_logger = self._factory.create_debug_logger_configured()
        tool_components["debug_logger"] = debug_logger

        # Build tool registrar
        tool_registrar = self._factory.create_tool_registrar(tools, tool_graph, provider, model)
        if background_task_callback:
            tool_registrar.set_background_task_callback(background_task_callback)
        tool_components["tool_registrar"] = tool_registrar

        # Build plugin manager
        plugin_manager = self._factory.initialize_plugin_system(tool_registrar)
        tool_components["plugin_manager"] = plugin_manager

        # Build argument normalizer
        argument_normalizer = self._factory.create_argument_normalizer(provider)
        tool_components["argument_normalizer"] = argument_normalizer

        # Build middleware chain
        middleware_chain, code_correction_middleware = self._factory.create_middleware_chain()
        tool_components["middleware_chain"] = middleware_chain
        tool_components["code_correction_middleware"] = code_correction_middleware

        # Build safety checker
        safety_checker = self._factory.create_safety_checker()
        tool_components["safety_checker"] = safety_checker

        # Build auto committer
        auto_committer = self._factory.create_auto_committer()
        tool_components["auto_committer"] = auto_committer

        # Build tool executor
        tool_executor = self._factory.create_tool_executor(
            tools=tools,
            argument_normalizer=argument_normalizer,
            tool_cache=tool_cache,
            safety_checker=safety_checker,
            code_correction_middleware=code_correction_middleware,
        )
        tool_components["tool_executor"] = tool_executor

        # Build parallel executor
        parallel_executor = self._factory.create_parallel_executor(tool_executor)
        tool_components["parallel_executor"] = parallel_executor

        # Build response completer
        response_completer = self._factory.create_response_completer()
        tool_components["response_completer"] = response_completer

        # Build unified tracker
        if unified_tracker is None:
            unified_tracker = self._factory.create_unified_tracker(tool_calling_caps)
        tool_components["unified_tracker"] = unified_tracker

        # Build tool selector (uses unified strategy factory with auto/keyword/semantic/hybrid strategies)
        tool_selector = self._factory.create_tool_selector(
            tools=tools,
            conversation_state=conversation_state,
            unified_tracker=unified_tracker,
            model=model or "unknown",
            provider_name=provider_name or "unknown",
            tool_selection=tool_selection or {},
            on_selection_recorded=kwargs.get("on_selection_recorded"),
        )
        tool_components["tool_selector"] = tool_selector

        # Build deduplication tracker
        deduplication_tracker = self._factory.create_tool_deduplication_tracker()
        tool_components["deduplication_tracker"] = deduplication_tracker

        # Build tool pipeline
        tool_pipeline = self._factory.create_tool_pipeline(
            tools=tools,
            tool_executor=tool_executor,
            tool_budget=tool_budget,
            tool_cache=tool_cache,
            argument_normalizer=argument_normalizer,
            on_tool_start=on_tool_start,
            on_tool_complete=on_tool_complete,
            deduplication_tracker=deduplication_tracker,
        )
        tool_components["tool_pipeline"] = tool_pipeline

        # Build tool output formatter
        if context_compactor:
            tool_output_formatter = self._factory.create_tool_output_formatter(context_compactor)
            tool_components["tool_output_formatter"] = tool_output_formatter

        # Build workflow registry
        workflow_registry = self._factory.create_workflow_registry()
        tool_components["workflow_registry"] = workflow_registry

        # Build code execution manager
        code_manager = self._factory.create_code_execution_manager()
        tool_components["code_manager"] = code_manager

        # Register all built components
        self._register_components(tool_components)

        self._logger.info(
            f"ToolBuilder built {len(tool_components)} tool components: "
            f"{', '.join(tool_components.keys())}"
        )

        return tool_components

    def get_tools(self) -> Optional[Any]:
        """Get the tool registry from built components.

        Returns:
            The tool registry if built, None otherwise
        """
        return self.get_component("tools")

    def get_tool_pipeline(self) -> Optional[Any]:
        """Get the tool pipeline from built components.

        Returns:
            The tool pipeline if built, None otherwise
        """
        return self.get_component("tool_pipeline")

    def get_tool_executor(self) -> Optional[Any]:
        """Get the tool executor from built components.

        Returns:
            The tool executor if built, None otherwise
        """
        return self.get_component("tool_executor")

    def get_tool_selector(self) -> Optional[Any]:
        """Get the tool selector from built components.

        Returns:
            The tool selector if built, None otherwise
        """
        return self.get_component("tool_selector")
