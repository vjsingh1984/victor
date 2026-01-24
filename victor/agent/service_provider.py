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

"""Service provider for orchestrator dependencies.

Registers all services required by AgentOrchestrator with the DI container.
This enables:
- Centralized service configuration
- Consistent lifecycle management
- Easy testing via override_services
- Type-safe service resolution

Design Pattern: Service Provider
- Groups related service registrations
- Separates singleton vs scoped lifetimes
- Provides factory functions for complex service creation

Architecture Note (Phase 3 DI Migration):
- 98 protocols defined across victor/agent/protocols.py and victor/protocols/
- 55 protocols registered here (56.1% coverage of all protocols)
- 43 protocols intentionally NOT registered - categorized as:
  1. Orchestrator-specific components (StreamingController, ToolPipeline, ConversationController)
     - These have orchestrator-specific dependencies (callbacks, state, budget)
     - Created in OrchestratorFactory with those specific dependencies
     - Resolved via get_optional() to allow factory flexibility
  2. Framework-level protocols (team coordination, multi-agent, search, etc.)
     - Used by verticals or other systems, not core orchestrator
     - Registered elsewhere or intentionally not registered
  3. Orchestrator-implemented protocols (VerticalStorageProtocol)
     - The orchestrator itself implements these
     - Not separate services to register

The current architecture prioritizes flexibility over pure DI. Components with
orchestrator-specific dependencies are created in OrchestratorFactory rather than
registered in the container, enabling clean dependency injection without forcing
complex registration logic.

Usage:
    from victor.core.container import ServiceContainer
    from victor.agent.service_provider import (
        OrchestratorServiceProvider,
        configure_orchestrator_services,
    )

    # Option 1: Full registration
    container = ServiceContainer()
    configure_orchestrator_services(container, settings)

    # Option 2: Selective registration
    provider = OrchestratorServiceProvider(settings)
    provider.register_singleton_services(container)  # Only singletons
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from victor.core.container import ServiceContainer, ServiceLifetime

# PERFORMANCE OPTIMIZATION (Phase 1):
# Move heavy imports to TYPE_CHECKING where possible to reduce startup overhead
# These imports are only needed for type hints, not runtime execution
if TYPE_CHECKING:
    from victor.config.settings import Settings

# NOTE: IToolSelector and Coordinators are imported at runtime (not TYPE_CHECKING) because they're
# used in container.register() calls, not just for type hints
from victor.protocols.tool_selector import IToolSelector
from victor.agent.coordinators.tool_retry_coordinator import ToolRetryCoordinator
from victor.agent.coordinators.memory_coordinator import MemoryCoordinator
from victor.agent.coordinators.tool_capability_coordinator import ToolCapabilityCoordinator

logger = logging.getLogger(__name__)


class OrchestratorServiceProvider:
    """Service provider for orchestrator dependencies.

    Manages registration of all services required by AgentOrchestrator.
    Services are categorized by lifetime:

    Singleton Services (application lifetime):
        - ToolRegistry: Shared tool definitions
        - ObservabilityIntegration: Event bus
        - TaskAnalyzer: Shared analysis
        - IntentClassifier: Semantic classification
        - ComplexityClassifier: Task complexity
        - ActionAuthorizer: Action authorization
        - SearchRouter: Search routing
        - ResponseSanitizer: Response cleanup
        - ArgumentNormalizer: Argument normalization
        - ProjectContext: Project-specific instructions

    Scoped Services (per-session):
        - ConversationStateMachine: Per-session state
        - UnifiedTaskTracker: Per-session tracking
        - MessageHistory: Per-session messages

    Attributes:
        _settings: Application settings for service configuration

    Example:
        container = ServiceContainer()
        provider = OrchestratorServiceProvider(settings)
        provider.register_services(container)

        # Resolve singletons directly
        sanitizer = container.get(ResponseSanitizerProtocol)

        # Resolve scoped services within a scope
        with container.create_scope() as scope:
            state_machine = scope.get(ConversationStateMachineProtocol)
    """

    def __init__(self, settings: "Settings"):
        """Initialize the service provider.

        Args:
            settings: Application settings for service configuration
        """
        self._settings = settings

    def register_services(self, container: ServiceContainer) -> None:
        """Register all orchestrator services.

        Registers both singleton and scoped services. Call this method
        during application bootstrap to set up all orchestrator dependencies.

        Args:
            container: DI container to register services in
        """
        # Store container reference for factory methods
        self.container = container
        self.register_singleton_services(container)
        self.register_scoped_services(container)
        logger.info("Registered all orchestrator services")

    def register_singleton_services(self, container: ServiceContainer) -> None:
        """Register singleton (application-lifetime) services.

        These services are created once and shared across all sessions.
        Use for stateless services or those with expensive initialization.

        PERFORMANCE OPTIMIZATION (Phase 1):
        Protocol imports are batched to reduce import overhead.
        Coordinator imports are deferred to actual registration to reduce
        unnecessary imports during bootstrap.

        Args:
            container: DI container to register services in
        """
        # Batch import all protocols in one statement to reduce import overhead
        from victor.agent.protocols import (
            ComplexityClassifierProtocol,
            ActionAuthorizerProtocol,
            SearchRouterProtocol,
            ResponseSanitizerProtocol,
            ArgumentNormalizerProtocol,
            TaskAnalyzerProtocol,
            ObservabilityProtocol,
            ToolRegistryProtocol,
            ToolRegistrarProtocol,
            ProjectContextProtocol,
            RecoveryHandlerProtocol,
            CodeExecutionManagerProtocol,
            WorkflowRegistryProtocol,
            UsageAnalyticsProtocol,
            ToolSequenceTrackerProtocol,
            ContextCompactorProtocol,
            ModeControllerProtocol,
            ProviderLifecycleProtocol,
            FileOperationsCapabilityProtocol,
            StageTransitionProtocol,
            ToolDeduplicationTrackerProtocol,
            # Utility service protocols
            DebugLoggerProtocol,
            TaskTypeHinterProtocol,
            ReminderManagerProtocol,
            RLCoordinatorProtocol,
            SafetyCheckerProtocol,
            AutoCommitterProtocol,
            MCPBridgeProtocol,
            # Infrastructure service protocols
            ToolDependencyGraphProtocol,
            ToolPluginRegistryProtocol,
            ProviderRegistryProtocol,
            # Analytics & observability protocols
            ConversationEmbeddingStoreProtocol,
            MetricsCollectorProtocol,
            ToolCacheProtocol,
            UsageLoggerProtocol,
            StreamingMetricsCollectorProtocol,
            IntentClassifierProtocol,
            # Helper/adapter service protocols
            SystemPromptBuilderProtocol,
            ToolExecutorProtocol,
            ToolOutputFormatterProtocol,
            ParallelExecutorProtocol,
            ResponseCompleterProtocol,
            StreamingHandlerProtocol,
            StreamingRecoveryCoordinatorProtocol,
            ChunkGeneratorProtocol,
            ToolPlannerProtocol,
            TaskCoordinatorProtocol,
            # Memory protocols
            UnifiedMemoryCoordinatorProtocol,
            # New coordinator protocols (WS-D)
            ToolCoordinatorProtocol,
            StateCoordinatorProtocol,
            PromptCoordinatorProtocol,
        )

        # Defer coordinator imports to when they're actually needed
        # This saves ~50-80ms during bootstrap

        # ToolRegistry - shared tool definitions
        self._register_tool_registry(container)

        # ObservabilityIntegration - event bus
        self._register_observability(container)

        # TaskAnalyzer - shared analysis facade
        self._register_task_analyzer(container)

        # IntentClassifier - singleton by design (ML model)
        self._register_intent_classifier(container)

        # ComplexityClassifier - stateless
        container.register(
            ComplexityClassifierProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_complexity_classifier(),
            ServiceLifetime.SINGLETON,
        )

        # ActionAuthorizer - stateless
        container.register(
            ActionAuthorizerProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_action_authorizer(),
            ServiceLifetime.SINGLETON,
        )

        # SearchRouter - stateless
        container.register(
            SearchRouterProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_search_router(),
            ServiceLifetime.SINGLETON,
        )

        # ResponseSanitizer - stateless
        container.register(
            ResponseSanitizerProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_response_sanitizer(),
            ServiceLifetime.SINGLETON,
        )

        # ArgumentNormalizer - stateless
        container.register(
            ArgumentNormalizerProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_argument_normalizer(),
            ServiceLifetime.SINGLETON,
        )

        # ProjectContext - shared project instructions
        container.register(
            ProjectContextProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_project_context(),
            ServiceLifetime.SINGLETON,
        )

        # RecoveryHandler - model failure recovery with Q-learning
        self._register_recovery_handler(container)

        # CodeSandbox - manages code execution sandboxes
        container.register(
            CodeExecutionManagerProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_code_execution_manager(),
            ServiceLifetime.SINGLETON,
        )

        # WorkflowRegistry - shared workflow definitions
        container.register(
            WorkflowRegistryProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_workflow_registry(),
            ServiceLifetime.SINGLETON,
        )

        # UsageAnalytics - singleton for data-driven optimization
        container.register(
            UsageAnalyticsProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_usage_analytics(),
            ServiceLifetime.SINGLETON,
        )

        # ToolSequenceTracker - singleton for pattern learning
        container.register(
            ToolSequenceTrackerProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_tool_sequence_tracker(),
            ServiceLifetime.SINGLETON,
        )

        # ContextCompactor - singleton for context window management
        container.register(
            ContextCompactorProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_context_compactor(),
            ServiceLifetime.SINGLETON,
        )

        # ModeController - singleton for agent mode management
        container.register(
            ModeControllerProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_mode_controller(),
            ServiceLifetime.SINGLETON,
        )

        # ProviderLifecycleManager - singleton for provider lifecycle operations (Phase 1.2)
        container.register(
            ProviderLifecycleProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_provider_lifecycle_manager(c),
            ServiceLifetime.SINGLETON,
        )

        # FileOperationsCapability - singleton for shared file operations (Phase 1.4)
        container.register(
            FileOperationsCapabilityProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_file_operations_capability(),
            ServiceLifetime.SINGLETON,
        )

        # StageTransitionEngine - singleton for stage transition management (Phase 2.2)
        container.register(
            StageTransitionProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_stage_transition_engine(c),
            ServiceLifetime.SINGLETON,
        )

        # ToolDeduplicationTracker - singleton for call deduplication
        container.register(
            ToolDeduplicationTrackerProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_tool_deduplication_tracker(),
            ServiceLifetime.SINGLETON,
        )

        # =========================================================================
        # Utility Services
        # =========================================================================

        # DebugLogger - singleton for clean debug output
        container.register(
            DebugLoggerProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_debug_logger(),
            ServiceLifetime.SINGLETON,
        )

        # TaskTypeHinter - singleton for task-specific hints
        container.register(
            TaskTypeHinterProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_task_type_hinter(),
            ServiceLifetime.SINGLETON,
        )

        # ReminderManager - scoped per session, but registered as factory
        container.register(
            ReminderManagerProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_reminder_manager(),
            ServiceLifetime.SCOPED,
        )

        # RLCoordinator - singleton for RL data management
        container.register(
            RLCoordinatorProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_rl_coordinator(),
            ServiceLifetime.SINGLETON,
        )

        # SafetyChecker - singleton for operation safety
        container.register(
            SafetyCheckerProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_safety_checker(),
            ServiceLifetime.SINGLETON,
        )

        # AutoCommitter - singleton for git commits
        container.register(
            AutoCommitterProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_auto_committer(),
            ServiceLifetime.SINGLETON,
        )

        # MCPBridge - singleton for MCP tool integration
        container.register(
            MCPBridgeProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_mcp_bridge(),
            ServiceLifetime.SINGLETON,
        )

        # =========================================================================
        # Infrastructure Services
        # =========================================================================

        # ToolDependencyGraph - singleton for tool dependency management
        container.register(
            ToolDependencyGraphProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_tool_dependency_graph(),
            ServiceLifetime.SINGLETON,
        )

        # ToolPluginRegistry - singleton for plugin management
        container.register(
            ToolPluginRegistryProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_tool_plugin_registry(),
            ServiceLifetime.SINGLETON,
        )

        # ProviderRegistry - singleton for provider management
        container.register(
            ProviderRegistryProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_provider_registry(),
            ServiceLifetime.SINGLETON,
        )

        # =========================================================================
        # Analytics & Observability Services
        # =========================================================================

        # ConversationEmbeddingStore - singleton for semantic search over history
        container.register(
            ConversationEmbeddingStoreProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_conversation_embedding_store(),
            ServiceLifetime.SINGLETON,
        )

        # MetricsCollector - singleton for metrics aggregation
        container.register(
            MetricsCollectorProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_metrics_collector(),
            ServiceLifetime.SINGLETON,
        )

        # ToolCache - singleton for tool result caching
        container.register(
            ToolCacheProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_tool_cache(),
            ServiceLifetime.SINGLETON,
        )

        # ToolCacheManager - singleton for internal tool state (indexes, connections)
        self._register_tool_cache_manager(container)

        # UsageLogger - singleton for usage logging
        container.register(
            UsageLoggerProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_usage_logger(),
            ServiceLifetime.SINGLETON,
        )

        # StreamingMetricsCollector - scoped per session
        container.register(
            StreamingMetricsCollectorProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_streaming_metrics_collector(),
            ServiceLifetime.SCOPED,
        )

        # IntentClassifier - singleton for ML-based intent classification
        container.register(
            IntentClassifierProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_intent_classifier(),
            ServiceLifetime.SINGLETON,
        )

        # =========================================================================
        # Helper/Adapter Services
        # =========================================================================

        # SystemPromptBuilder - factory for prompt construction
        container.register(
            SystemPromptBuilderProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_system_prompt_builder(),
            ServiceLifetime.SINGLETON,
        )

        # ToolSelector - singleton for tool selection logic
        container.register(
            IToolSelector,  # type: ignore[type-abstract]
            lambda c: self._create_tool_selector(),
            ServiceLifetime.SINGLETON,
        )

        # ToolExecutor - singleton for tool execution
        container.register(
            ToolExecutorProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_tool_executor(),
            ServiceLifetime.SINGLETON,
        )

        # ToolOutputFormatter - singleton for formatting tool outputs
        container.register(
            ToolOutputFormatterProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_tool_output_formatter(),
            ServiceLifetime.SINGLETON,
        )

        # ParallelExecutor - factory for parallel execution
        container.register(
            ParallelExecutorProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_parallel_executor(),
            ServiceLifetime.SINGLETON,
        )

        # ResponseCompleter - factory for response completion
        container.register(
            ResponseCompleterProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_response_completer(),
            ServiceLifetime.SINGLETON,
        )

        # StreamingHandler - scoped per session
        container.register(
            StreamingHandlerProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_streaming_handler(),
            ServiceLifetime.SCOPED,
        )

        # StreamingRecoveryCoordinator - singleton for streaming session recovery
        container.register(
            StreamingRecoveryCoordinatorProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_recovery_coordinator(),
            ServiceLifetime.SINGLETON,
        )

        # ChunkGenerator - singleton for chunk generation
        container.register(
            ChunkGeneratorProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_chunk_generator(),
            ServiceLifetime.SINGLETON,
        )

        # ToolPlanner - singleton for tool planning
        container.register(
            ToolPlannerProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_tool_planner(),
            ServiceLifetime.SINGLETON,
        )

        # TaskCoordinator - singleton for task coordination
        container.register(
            TaskCoordinatorProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_task_coordinator(),
            ServiceLifetime.SINGLETON,
        )

        # PathResolver - singleton for centralized path resolution
        self._register_path_resolver(container)

        # UnifiedMemoryCoordinator - singleton for federated memory search
        self._register_unified_memory_coordinator(container)

        # =========================================================================
        # New Coordinators (WS-D: Orchestrator SOLID Fixes)
        # =========================================================================

        # ToolCoordinator - scoped for tool selection/budget/execution coordination
        container.register(
            ToolCoordinatorProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_tool_coordinator(),
            ServiceLifetime.SCOPED,
        )

        # StateCoordinator - scoped for conversation state management
        container.register(
            StateCoordinatorProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_state_coordinator(),
            ServiceLifetime.SCOPED,
        )

        # PromptCoordinator - scoped for system prompt assembly
        container.register(
            PromptCoordinatorProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_prompt_coordinator(),
            ServiceLifetime.SCOPED,
        )

        # =========================================================================
        # Phase 5 Coordinators (Orchestrator Integration)
        # =========================================================================

        # ToolRetryCoordinator - scoped for tool retry logic
        container.register(
            ToolRetryCoordinator,
            lambda c: self._create_tool_retry_coordinator(),
            ServiceLifetime.SCOPED,
        )

        # MemoryCoordinator - scoped for memory management
        container.register(
            MemoryCoordinator,
            lambda c: self._create_memory_coordinator(),
            ServiceLifetime.SCOPED,
        )

        # ToolCapabilityCoordinator - singleton for capability checks
        container.register(
            ToolCapabilityCoordinator,
            lambda c: self._create_tool_capability_coordinator(),
            ServiceLifetime.SINGLETON,
        )

        # =========================================================================
        # Phase 3 Coordinators (Tool Call & Prompt Builder Integration)
        # =========================================================================

        # ToolCallCoordinator - scoped for tool call coordination
        from victor.agent.coordinators.tool_call_protocol import IToolCallCoordinator

        container.register(
            IToolCallCoordinator,  # type: ignore[type-abstract]
            lambda c: self._create_tool_call_coordinator(),
            ServiceLifetime.SCOPED,
        )

        # PromptBuilderCoordinator - scoped for prompt building coordination
        from victor.agent.coordinators.prompt_builder_protocol import IPromptBuilderCoordinator

        container.register(
            IPromptBuilderCoordinator,  # type: ignore[type-abstract]
            lambda c: self._create_prompt_builder_coordinator(),
            ServiceLifetime.SCOPED,
        )

        # =========================================================================
        # Agentic AI Services (Phase 3 Integration)
        # =========================================================================
        # These services enable advanced agentic AI capabilities for autonomous
        # planning, memory, skill discovery, and self-improvement.
        # Only registered if feature flags are enabled in settings.

        if self._settings.enable_hierarchical_planning:
            self._register_hierarchical_planner(container)

        if self._settings.enable_episodic_memory:
            self._register_episodic_memory(container)

        if self._settings.enable_semantic_memory:
            self._register_semantic_memory(container)

        if self._settings.enable_skill_discovery:
            self._register_skill_discovery(container)

        if self._settings.enable_skill_chaining:
            self._register_skill_chainer(container)

        if self._settings.enable_self_improvement:
            self._register_proficiency_tracker(container)

        if self._settings.enable_rl_coordinator:
            self._register_rl_coordinator(container)

        # =========================================================================
        # Presentation Abstraction Layer
        # =========================================================================

        # PresentationAdapter - singleton for icon/formatting concerns
        self._register_presentation_adapter(container)

        # =========================================================================
        # Provider Pool Services (Load Balancing & Health Monitoring)
        # =========================================================================
        # These services enable provider pooling for improved reliability and
        # performance. Only registered if provider pool is enabled in settings.

        if self._settings.enable_provider_pool:
            self._register_provider_pool_services(container)

        logger.debug("Registered singleton orchestrator services")

    def _register_provider_pool_services(self, container: ServiceContainer) -> None:
        """Register provider pool services for load balancing.

        Args:
            container: DI container to register services in
        """
        from victor.providers.health_monitor import (
            ProviderHealthRegistry,
            HealthMonitor,
            get_health_registry,
        )
        from victor.providers.load_balancer import LoadBalancerType, create_load_balancer

        # ProviderHealthRegistry - singleton for health monitoring
        # Note: We register a factory that returns the registry
        # Use async_to_sync wrapper for async function
        container.register(
            ProviderHealthRegistry,
            lambda c: get_health_registry(),  # type: ignore[arg-type, return-value]
            ServiceLifetime.SINGLETON,
        )

        # LoadBalancerFactory - factory for creating load balancers
        # Store as instance variable for later access
        self._load_balancer_factory = self._create_load_balancer_factory()

        logger.info("Registered provider pool services (health monitoring, load balancing)")

    def get_load_balancer_factory(self) -> Any:  # Callable[[], LoadBalancer]
        """Get the load balancer factory instance.

        Returns:
            Factory function for creating load balancers
        """
        return getattr(self, "_load_balancer_factory", None)

    def _create_load_balancer_factory(self) -> Any:  # Callable[[], LoadBalancer]
        """Create factory function for load balancers.

        Returns:
            Factory function that creates LoadBalancer instances
        """
        from victor.providers.load_balancer import LoadBalancerType, create_load_balancer

        def factory(strategy: str, name: Optional[str] = None) -> Any:  # LoadBalancer
            """Create a load balancer instance.

            Args:
                strategy: Load balancing strategy name
                name: Optional custom name

            Returns:
                LoadBalancer instance
            """
            load_balancer_type = LoadBalancerType(strategy)
            return create_load_balancer(load_balancer_type, name=name)

        return factory

    def register_scoped_services(self, container: ServiceContainer) -> None:
        """Register scoped (per-session) services.

        These services are created fresh for each orchestrator session.
        Use for stateful services that need isolation between sessions.

        Args:
            container: DI container to register services in
        """
        from victor.agent.protocols import (
            ConversationStateMachineProtocol,
            TaskTrackerProtocol,
            MessageHistoryProtocol,
        )

        # ConversationStateMachine - per-session state
        container.register(
            ConversationStateMachineProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_conversation_state_machine(),
            ServiceLifetime.SCOPED,
        )

        # UnifiedTaskTracker - per-session tracking
        container.register(
            TaskTrackerProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_unified_task_tracker(),
            ServiceLifetime.SCOPED,
        )

        # MessageHistory - per-session messages
        container.register(
            MessageHistoryProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_message_history(),
            ServiceLifetime.SCOPED,
        )

        # ToolAccessController - per-session tool access control
        self._register_tool_access_controller(container)

        # BudgetManager - per-session budget management
        self._register_budget_manager(container)

        logger.debug("Registered scoped orchestrator services")

    # =========================================================================
    # Factory methods for singleton services
    # =========================================================================

    def _register_tool_registry(self, container: ServiceContainer) -> None:
        """Register ToolRegistry as singleton."""
        from victor.agent.protocols import ToolRegistryProtocol, ToolRegistrarProtocol
        from victor.tools.base import ToolRegistry  # type: ignore[attr-defined]

        container.register(
            ToolRegistryProtocol,  # type: ignore[type-abstract]
            lambda c: ToolRegistry(),  # type: ignore[arg-type, return-value]
            ServiceLifetime.SINGLETON,
        )

        # ToolRegistrar - manages tool registration, plugins, and MCP integration
        container.register(
            ToolRegistrarProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_tool_registrar(c),
            ServiceLifetime.SINGLETON,
        )

    def _create_tool_registrar(self, container: ServiceContainer) -> Any:
        """Create ToolRegistrar instance.

        ToolRegistrar manages:
        - Dynamic tool discovery from victor/tools directory
        - Plugin system management
        - MCP (Model Context Protocol) integration
        - Tool dependency graph setup

        Args:
            container: DI container for resolving dependencies

        Returns:
            ToolRegistrar instance
        """
        from victor.agent.tool_registrar import ToolRegistrar, ToolRegistrarConfig
        from victor.agent.protocols import ToolRegistryProtocol, ToolDependencyGraphProtocol

        # Get ToolRegistry from container
        tool_registry = container.get(ToolRegistryProtocol)  # type: ignore[type-abstract]

        # Get ToolDependencyGraph from container for tool planning
        tool_graph = None
        if getattr(self._settings, "enable_tool_graph", True):
            try:
                tool_graph = container.get(ToolDependencyGraphProtocol)  # type: ignore[type-abstract]
            except Exception:
                pass  # Tool graph is optional

        # Build config from settings
        config = ToolRegistrarConfig(
            enable_plugins=getattr(self._settings, "enable_plugins", True),
            enable_mcp=getattr(self._settings, "use_mcp_tools", False),
            enable_tool_graph=getattr(self._settings, "enable_tool_graph", True),
            airgapped_mode=getattr(self._settings, "airgapped_mode", False),
        )

        return ToolRegistrar(
            tools=tool_registry,  # type: ignore[arg-type]
            settings=self._settings,
            provider=None,  # Will be set later by orchestrator
            model=getattr(self._settings, "model", None),
            tool_graph=tool_graph,  # Pass tool graph for tool planning
            config=config,
        )

    def _register_observability(self, container: ServiceContainer) -> None:
        """Register ObservabilityIntegration as singleton."""
        from victor.agent.protocols import ObservabilityProtocol

        def create_observability(_: ServiceContainer) -> Any:
            enable = getattr(self._settings, "enable_observability", True)
            if not enable:
                return _NullObservability()

            try:
                from victor.observability.integration import ObservabilityIntegration

                return ObservabilityIntegration()
            except ImportError:
                logger.warning("ObservabilityIntegration not available")
                return _NullObservability()

        container.register(
            ObservabilityProtocol,  # type: ignore[type-abstract]
            create_observability,
            ServiceLifetime.SINGLETON,
        )

    def _register_task_analyzer(self, container: ServiceContainer) -> None:
        """Register TaskAnalyzer as singleton."""
        from victor.agent.protocols import TaskAnalyzerProtocol

        def create_task_analyzer(_: ServiceContainer) -> Any:
            try:
                from victor.agent.task_analyzer import get_task_analyzer

                return get_task_analyzer()
            except ImportError:
                logger.warning("TaskAnalyzer not available")
                return _NullTaskAnalyzer()

        container.register(
            TaskAnalyzerProtocol,  # type: ignore[type-abstract]
            create_task_analyzer,
            ServiceLifetime.SINGLETON,
        )

    def _register_intent_classifier(self, container: ServiceContainer) -> None:
        """Register IntentClassifier as singleton."""
        from victor.storage.embeddings.intent_classifier import IntentClassifier

        container.register(
            IntentClassifier,
            lambda c: IntentClassifier.get_instance(),
            ServiceLifetime.SINGLETON,
        )

    def _create_complexity_classifier(self) -> Any:
        """Create ComplexityClassifier instance."""
        from victor.framework.task import TaskComplexityService as ComplexityClassifier

        return ComplexityClassifier()

    def _create_action_authorizer(self) -> Any:
        """Create ActionAuthorizer instance."""
        from victor.agent.action_authorizer import ActionAuthorizer

        return ActionAuthorizer()

    def _create_search_router(self) -> Any:
        """Create SearchRouter instance."""
        from victor.agent.search_router import SearchRouter

        return SearchRouter()

    def _create_response_sanitizer(self) -> Any:
        """Create ResponseSanitizer instance."""
        from victor.agent.response_sanitizer import ResponseSanitizer

        return ResponseSanitizer()

    def _create_argument_normalizer(self) -> Any:
        """Create ArgumentNormalizer instance."""
        from victor.agent.argument_normalizer import ArgumentNormalizer

        # Provider name will be updated when orchestrator is created
        return ArgumentNormalizer(provider_name="unknown")

    def _create_project_context(self) -> Any:
        """Create ProjectContext instance."""
        from victor.context.project_context import ProjectContext

        context = ProjectContext()
        context.load()
        return context

    def _register_recovery_handler(self, container: ServiceContainer) -> None:
        """Register RecoveryHandler as singleton.

        The RecoveryHandler integrates with:
        - Q-learning for adaptive recovery strategy selection
        - UsageAnalytics for telemetry
        - ContextCompactor for proactive context management

        Note: Session-specific state (recent_responses, consecutive_failures)
        is reset via set_session_id() when orchestrator creates a new session.
        """
        from victor.agent.protocols import RecoveryHandlerProtocol

        def create_recovery_handler(_: ServiceContainer) -> Any:
            enabled = getattr(self._settings, "enable_recovery_system", True)
            if not enabled:
                return _NullRecoveryHandler()

            try:
                from victor.agent.recovery import RecoveryHandler

                return RecoveryHandler.create(settings=self._settings)
            except ImportError as e:
                logger.warning(f"RecoveryHandler not available: {e}")
                return _NullRecoveryHandler()
            except Exception as e:
                logger.warning(f"RecoveryHandler creation failed: {e}")
                return _NullRecoveryHandler()

        container.register(
            RecoveryHandlerProtocol,  # type: ignore[type-abstract]
            create_recovery_handler,
            ServiceLifetime.SINGLETON,
        )

    # =========================================================================
    # Factory methods for scoped services
    # =========================================================================

    def _create_conversation_state_machine(self) -> Any:
        """Create ConversationStateMachine instance."""
        from victor.agent.conversation_state import ConversationStateMachine

        return ConversationStateMachine()

    def _create_unified_task_tracker(self) -> Any:
        """Create UnifiedTaskTracker instance."""
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        return UnifiedTaskTracker()

    def _create_message_history(self) -> Any:
        """Create MessageHistory instance."""
        from victor.agent.message_history import MessageHistory

        return MessageHistory(
            system_prompt="",  # Will be set by orchestrator
            max_history_messages=getattr(self._settings, "max_conversation_history", 100000),
        )

    # =========================================================================
    # Factory methods for new framework services
    # =========================================================================

    def _register_path_resolver(self, container: ServiceContainer) -> None:
        """Register PathResolver as singleton."""
        from victor.protocols.path_resolver import PathResolver, IPathResolver

        container.register(
            IPathResolver,  # type: ignore[type-abstract]
            lambda c: PathResolver(),
            ServiceLifetime.SINGLETON,
        )

    def _register_unified_memory_coordinator(self, container: ServiceContainer) -> None:
        """Register UnifiedMemoryCoordinator as singleton.

        The UnifiedMemoryCoordinator provides federated search across all
        memory backends (entity, conversation, graph, embeddings) with
        pluggable ranking strategies.
        """
        from victor.agent.protocols import UnifiedMemoryCoordinatorProtocol

        def create_memory_coordinator(_: ServiceContainer) -> Any:
            try:
                from victor.storage.memory.unified import get_memory_coordinator

                return get_memory_coordinator()
            except ImportError as e:
                logger.warning(f"UnifiedMemoryCoordinator not available: {e}")
                return _NullMemoryCoordinator()

        container.register(
            UnifiedMemoryCoordinatorProtocol,  # type: ignore[type-abstract]
            create_memory_coordinator,
            ServiceLifetime.SINGLETON,
        )

    def _register_tool_access_controller(self, container: ServiceContainer) -> None:
        """Register ToolAccessController as scoped service."""
        from victor.agent.protocols import IToolAccessController, ToolRegistryProtocol
        from victor.agent.tool_access_controller import (
            create_tool_access_controller,
        )

        container.register(
            IToolAccessController,  # type: ignore[type-abstract]
            lambda c: create_tool_access_controller(
                registry=c.get_optional(ToolRegistryProtocol),  # type: ignore[arg-type, type-abstract]
            ),
            ServiceLifetime.SCOPED,
        )

    def _register_tool_cache_manager(self, container: ServiceContainer) -> None:
        """Register ToolCacheManager as singleton.

        The ToolCacheManager provides centralized cache management for tools,
        replacing module-level caches like _INDEX_CACHE in code_search_tool.py
        and _connections in database_tool.py.

        This enables:
        - Proper test isolation (caches can be cleared between tests)
        - DI-based tool configuration
        - Unified cache statistics and monitoring
        """
        from victor.tools.cache_manager import ToolCacheManager

        container.register(
            ToolCacheManager,
            lambda c: ToolCacheManager(),
            ServiceLifetime.SINGLETON,
        )

    def _register_budget_manager(self, container: ServiceContainer) -> None:
        """Register BudgetManager as scoped service."""
        from victor.agent.protocols import IBudgetManager, BudgetConfig
        from victor.agent.budget_manager import create_budget_manager

        # Get budget settings from settings
        base_tool_calls = getattr(self._settings, "tool_budget", 30)
        base_iterations = getattr(self._settings, "max_iterations", 50)
        base_exploration = getattr(self._settings, "max_exploration_iterations", 8)
        base_action = getattr(self._settings, "max_action_iterations", 12)

        config = BudgetConfig(
            base_tool_calls=base_tool_calls,
            base_iterations=base_iterations,
            base_exploration=base_exploration,
            base_action=base_action,
        )

        container.register(
            IBudgetManager,  # type: ignore[type-abstract]
            lambda c: create_budget_manager(config=config),
            ServiceLifetime.SCOPED,
        )

    # =========================================================================
    # Infrastructure service factory methods
    # =========================================================================

    def _create_code_execution_manager(self) -> Any:
        """Create CodeSandbox instance."""
        from victor.tools.code_executor_tool import CodeSandbox

        manager = CodeSandbox()
        manager.start()
        return manager

    def _create_workflow_registry(self) -> Any:
        """Create WorkflowRegistry instance."""
        from victor.workflows.base import WorkflowRegistry

        return WorkflowRegistry()

    def _create_usage_analytics(self) -> Any:
        """Create UsageAnalytics singleton instance."""
        from victor.agent.usage_analytics import UsageAnalytics, AnalyticsConfig
        from pathlib import Path

        # Use settings cache_dir if available
        analytics_cache_dir = (
            Path(self._settings.cache_dir)
            if hasattr(self._settings, "cache_dir") and self._settings.cache_dir
            else None
        )

        return UsageAnalytics.get_instance(
            AnalyticsConfig(
                cache_dir=analytics_cache_dir,
                enable_prometheus_export=getattr(self._settings, "enable_prometheus_export", True),
            )
        )

    def _create_tool_sequence_tracker(self) -> Any:
        """Create ToolSequenceTracker instance."""
        from victor.agent.tool_sequence_tracker import create_sequence_tracker

        return create_sequence_tracker(
            use_predefined=getattr(self._settings, "use_predefined_patterns", True),
            learning_rate=getattr(self._settings, "sequence_learning_rate", 0.3),
        )

    def _create_context_compactor(self) -> Any:
        """Create ContextCompactor instance.

        Note: This is a placeholder that will be replaced when the orchestrator
        creates the actual compactor with a ConversationController instance.
        The DI container registration is primarily for protocol compliance.
        """
        # ContextCompactor requires a ConversationController which is not available
        # at container initialization time. Return None and let the orchestrator
        # create it via the factory method when the controller is available.
        return None

    def _create_mode_controller(self) -> Any:
        """Create ModeControllerAdapter wrapping AgentModeController.

        Phase 1.1: Uses ModeControllerAdapter for DI integration.
        The adapter wraps AgentModeController and implements the protocol
        interface for clean orchestrator integration.
        """
        from victor.agent.mode_controller import AgentModeController, AgentMode
        from victor.agent.coordinators.mode_adapter import ModeControllerAdapter

        initial_mode_str = getattr(self._settings, "default_agent_mode", "build")
        try:
            initial_mode = AgentMode(initial_mode_str.lower())
        except ValueError:
            initial_mode = AgentMode.BUILD

        controller = AgentModeController(initial_mode=initial_mode)
        return ModeControllerAdapter(controller)

    def _create_provider_lifecycle_manager(self, container: ServiceContainer) -> Any:
        """Create ProviderLifecycleManager instance.

        Phase 1.2: Provider lifecycle management for post-switch hooks.
        """
        from victor.agent.provider_lifecycle_manager import ProviderLifecycleManager

        return ProviderLifecycleManager(container)

    def _create_file_operations_capability(self) -> Any:
        """Create FileOperationsCapability instance.

        Phase 1.4: Shared file operations capability for verticals.
        """
        from victor.framework.capabilities import FileOperationsCapability

        return FileOperationsCapability()

    def _create_stage_transition_engine(self, container: ServiceContainer) -> Any:
        """Create StageTransitionEngine instance.

        Phase 2.2: Stage transition management for conversation flow.
        """
        from victor.agent.stage_transition_engine import StageTransitionEngine

        # Get event bus if available
        try:
            from victor.core.events import ObservabilityBus

            event_bus = container.get(ObservabilityBus)
        except Exception:
            event_bus = None

        return StageTransitionEngine(event_bus=event_bus)

    def _create_tool_deduplication_tracker(self) -> Any:
        """Create ToolDeduplicationTracker instance."""
        from victor.agent.tool_deduplication import ToolDeduplicationTracker

        window_size = getattr(self._settings, "dedup_window_size", 10)
        similarity_threshold = getattr(self._settings, "dedup_similarity_threshold", 0.7)

        return ToolDeduplicationTracker(
            window_size=window_size,
            similarity_threshold=similarity_threshold,
        )

    # =========================================================================
    # Utility service factory methods
    # =========================================================================

    def _create_debug_logger(self) -> Any:
        """Create DebugLogger instance."""
        from victor.agent.debug_logger import get_debug_logger

        return get_debug_logger()

    def _create_task_type_hinter(self) -> Any:
        """Create TaskTypeHinter wrapper."""
        from victor.coding.prompts import get_task_type_hint

        class TaskTypeHinter:
            """Wrapper for task type hint retrieval."""

            def get_hint(self, task_type: str) -> str:
                """Get prompt hint for a specific task type."""
                return get_task_type_hint(task_type)

        return TaskTypeHinter()

    def _create_reminder_manager(self) -> Any:
        """Create ContextReminderManager instance."""
        from victor.agent.context_reminder import create_reminder_manager

        provider = getattr(self._settings, "provider", "unknown")
        task_complexity = "medium"
        tool_budget = getattr(self._settings, "tool_budget", 10)

        return create_reminder_manager(
            provider=provider,
            task_complexity=task_complexity,
            tool_budget=tool_budget,
        )

    def _create_rl_coordinator(self) -> Any:
        """Create RLCoordinator instance."""
        from victor.framework.rl.coordinator import get_rl_coordinator

        return get_rl_coordinator()

    def _create_safety_checker(self) -> Any:
        """Create SafetyChecker instance."""
        from victor.agent.safety import get_safety_checker

        return get_safety_checker()

    def _create_auto_committer(self) -> Any:
        """Create AutoCommitter instance."""
        from victor.agent.auto_commit import get_auto_committer

        return get_auto_committer()

    def _create_mcp_bridge(self) -> Any:
        """Create MCP bridge wrapper."""
        from victor.tools.mcp_bridge_tool import get_mcp_tool_definitions

        class MCPBridge:
            """Wrapper for MCP bridge functionality."""

            def get_tool_definitions(self) -> list[dict[str, Any]]:
                """Return MCP tools as Victor tool definitions."""
                return get_mcp_tool_definitions()

        return MCPBridge()

    # =========================================================================
    # Infrastructure service factory methods
    # =========================================================================

    def _create_tool_dependency_graph(self) -> Any:
        """Create ToolDependencyGraph instance."""
        from victor.tools.dependency_graph import ToolDependencyGraph

        return ToolDependencyGraph()

    def _create_tool_plugin_registry(self) -> Any:
        """Create ToolPluginRegistry instance."""
        from victor.tools.plugin_registry import ToolPluginRegistry

        return ToolPluginRegistry()

    def _create_semantic_tool_selector(self) -> Any:
        """Create SemanticToolSelector instance."""
        from victor.tools.semantic_selector import SemanticToolSelector

        # Get unified embedding model from settings for consistency
        # Use unified_embedding_model (default: BAAI/bge-small-en-v1.5)
        # This ensures parity across all LLM providers for semantic search
        embedding_model = getattr(
            self._settings, "unified_embedding_model", "BAAI/bge-small-en-v1.5"
        )
        use_semantic_selection = getattr(self._settings, "use_semantic_tool_selection", True)

        if not use_semantic_selection:
            # Return a no-op selector that just returns all tools
            class NoOpSelector:
                def select_tools(
                    self,
                    query: str,
                    available_tools: list[Any],
                    max_tools: int = 10,
                    threshold: float = 0.3,
                ) -> list[Any]:
                    return available_tools[:max_tools]

                def compute_similarity(self, query: str, tool_description: str) -> float:
                    return 1.0

            return NoOpSelector()

        # IMPORTANT: Always use sentence-transformers for semantic search regardless of LLM provider
        # This ensures consistency in context retrieval quality across all providers (ollama, openai, etc.)
        # Using different embedding models per provider would break parity and cause inconsistent behavior
        return SemanticToolSelector(
            embedding_model=embedding_model,
            embedding_provider="sentence-transformers",
        )

    def _create_provider_registry(self) -> Any:
        """Create ProviderRegistry wrapper.

        Note: ProviderRegistry is a class with class methods,
        so we return the class itself, not an instance.
        """
        from victor.providers.registry import ProviderRegistry

        # Return the class itself - it's used as a class with class methods
        return ProviderRegistry

    # =========================================================================
    # Analytics & observability service factory methods
    # =========================================================================

    def _create_conversation_embedding_store(self) -> Any:
        """Create ConversationEmbeddingStore instance."""
        from victor.agent.conversation_embedding_store import ConversationEmbeddingStore
        from victor.storage.embeddings.service import EmbeddingService
        from pathlib import Path

        embedding_model = getattr(self._settings, "embedding_model", "all-MiniLM-L12-v2")
        cache_dir = Path(getattr(self._settings, "cache_dir", ".victor/cache"))

        # Create embedding service
        embedding_service = EmbeddingService(model_name=embedding_model)

        # Create embedding store with SQLite backend
        sqlite_path = cache_dir / "embeddings" / "conversation.db"

        return ConversationEmbeddingStore(
            embedding_service=embedding_service,
            sqlite_db_path=sqlite_path,
        )

    def _create_metrics_collector(self) -> Any:
        """Create MetricsCollector instance."""
        from victor.agent.metrics_collector import MetricsCollector, MetricsCollectorConfig
        from victor.analytics.logger import UsageLogger
        from pathlib import Path

        # Create config with provider/model info
        model = getattr(self._settings, "model", "unknown")
        provider = getattr(self._settings, "provider", "unknown")
        analytics_enabled = getattr(self._settings, "enable_analytics", True)

        config = MetricsCollectorConfig(
            model=model,
            provider=provider,
            analytics_enabled=analytics_enabled,
        )

        # Create usage logger
        log_file = Path(getattr(self._settings, "usage_log_path", ".victor/logs/usage.log"))
        usage_logger = UsageLogger(log_file=log_file)

        return MetricsCollector(
            config=config,
            usage_logger=usage_logger,
        )

    def _create_tool_cache(self) -> Any:
        """Create ToolCache instance."""
        from victor.storage.cache.tool_cache import ToolCache

        enabled = getattr(self._settings, "enable_tool_cache", True)
        ttl = getattr(self._settings, "tool_cache_ttl", 300)

        if not enabled:
            # Return a no-op cache
            class NoOpCache:
                def get(self, key: str) -> Any:
                    return None

                def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
                    pass

                def invalidate(self, key: str) -> None:
                    pass

                def clear(self) -> None:
                    pass

            return NoOpCache()

        return ToolCache(ttl=ttl)

    def _create_usage_logger(self) -> Any:
        """Create UsageLogger instance."""
        from victor.analytics.logger import UsageLogger
        from pathlib import Path

        log_file = Path(getattr(self._settings, "usage_log_path", ".victor/logs/usage.log"))
        enabled = getattr(self._settings, "enable_usage_logging", True)

        return UsageLogger(log_file=log_file, enabled=enabled)

    def _create_streaming_metrics_collector(self) -> Any:
        """Create StreamingMetricsCollector instance."""
        from victor.analytics.streaming_metrics import StreamingMetricsCollector
        from pathlib import Path

        enabled = getattr(self._settings, "streaming_metrics_enabled", True)
        max_history = getattr(self._settings, "streaming_metrics_history_size", 100)

        if not enabled:
            # Return a no-op collector
            class NoOpStreamingMetrics:
                def record_chunk(self, chunk_size: int, timestamp: float, **metadata: Any) -> None:
                    pass

                def get_metrics(self) -> dict[str, Any]:
                    return {}

                def reset(self) -> None:
                    pass

            return NoOpStreamingMetrics()

        export_path = (
            Path(getattr(self._settings, "cache_dir", ".victor/cache")) / "streaming_metrics.json"
        )

        return StreamingMetricsCollector(max_history=max_history, export_path=export_path)

    def _create_intent_classifier(self) -> Any:
        """Create IntentClassifier instance."""
        from victor.storage.embeddings.intent_classifier import IntentClassifier
        from victor.storage.embeddings.service import EmbeddingService
        from pathlib import Path

        embedding_model = getattr(self._settings, "embedding_model", "all-MiniLM-L12-v2")
        cache_dir = (
            Path(getattr(self._settings, "cache_dir", ".victor/cache")) / "intent_classifier"
        )

        # Create embedding service
        embedding_service = EmbeddingService(model_name=embedding_model)

        return IntentClassifier(
            cache_dir=cache_dir,
            embedding_service=embedding_service,
        )

    # =========================================================================
    # Helper/adapter service factory methods
    # =========================================================================

    def _create_system_prompt_builder(self) -> Any:
        """Create SystemPromptBuilder instance."""
        from victor.agent.prompt_builder import SystemPromptBuilder

        provider_name = getattr(self._settings, "provider", "anthropic")
        model = getattr(self._settings, "model", "claude-opus-4")

        return SystemPromptBuilder(
            provider_name=provider_name,
            model=model,
        )

    def _create_tool_selector(self) -> Any:
        """Create ToolSelector instance."""
        from victor.agent.tool_selection import ToolSelector
        from victor.tools.registry import ToolRegistry

        # Create a ToolRegistry instance for the ToolSelector
        tools = ToolRegistry()
        model = getattr(self._settings, "model", "claude-opus-4")
        provider_name = getattr(self._settings, "provider", "anthropic")
        fallback_max_tools = getattr(self._settings, "fallback_max_tools", 8)

        return ToolSelector(
            tools=tools,
            model=model,
            provider_name=provider_name,
            fallback_max_tools=fallback_max_tools,
        )

    def _create_tool_executor(self) -> Any:
        """Create ToolExecutor instance."""
        from victor.agent.tool_executor import ToolExecutor
        from victor.tools.registry import ToolRegistry

        # Create a ToolRegistry instance for the ToolExecutor
        tool_registry = ToolRegistry()

        return ToolExecutor(tool_registry=tool_registry)

    def _create_tool_output_formatter(self) -> Any:
        """Create ToolOutputFormatter instance."""
        from victor.agent.tool_output_formatter import create_tool_output_formatter

        # Factory function handles all defaults
        return create_tool_output_formatter()

    def _create_parallel_executor(self) -> Any:
        """Create ParallelExecutor instance."""
        from victor.agent.parallel_executor import create_parallel_executor
        from victor.agent.tool_executor import ToolExecutor
        from victor.tools.registry import ToolRegistry

        # Create ToolExecutor instance needed by parallel executor
        tool_registry = ToolRegistry()
        tool_executor = ToolExecutor(tool_registry=tool_registry)

        max_concurrent = getattr(self._settings, "max_concurrent_tools", 5)
        enable = getattr(self._settings, "enable_parallel_execution", True)

        return create_parallel_executor(
            tool_executor=tool_executor,
            max_concurrent=max_concurrent,
            enable=enable,
        )

    def _create_response_completer(self) -> Any:
        """Create ResponseCompleter instance.

        Note: This returns a no-op completer since we don't have access to the provider
        instance here. The actual ResponseCompleter should be created when the provider
        is available.
        """
        from victor.agent.response_completer import create_response_completer

        # Create a mock provider for now - in practice, this would need the actual provider
        class MockProvider:
            """Mock provider for ResponseCompleter."""

            async def chat(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                return {"content": "", "model": "mock"}

        mock_provider = MockProvider()
        max_retries = getattr(self._settings, "max_response_retries", 3)
        force_response = getattr(self._settings, "force_response", True)

        return create_response_completer(
            provider=mock_provider,  # type: ignore[arg-type]
            max_retries=max_retries,
            force_response=force_response,
        )

    def _create_streaming_handler(self) -> Any:
        """Create StreamingHandler instance."""
        from victor.agent.streaming import StreamingChatHandler

        # Create a mock message adder for now
        class MockMessageAdder:
            """Mock message adder for StreamingChatHandler."""

            def add_message(self, role: str, content: str) -> None:
                pass

        message_adder = MockMessageAdder()
        session_idle_timeout = getattr(self._settings, "session_idle_timeout", 180.0)

        return StreamingChatHandler(
            settings=self._settings,
            message_adder=message_adder,
            session_idle_timeout=session_idle_timeout,
        )

    def _create_recovery_coordinator(self) -> Any:
        """Create RecoveryCoordinator instance.

        The RecoveryCoordinator centralizes all recovery and error handling logic
        for streaming chat sessions, including condition checking, action handling,
        and recovery integration.

        Returns:
            StreamingRecoveryCoordinator instance
        """
        from victor.agent.recovery_coordinator import StreamingRecoveryCoordinator
        from victor.agent.protocols import (
            RecoveryHandlerProtocol,
            StreamingHandlerProtocol,
            ContextCompactorProtocol,
            TaskTrackerProtocol,
        )

        # Get recovery handler from DI container (optional)
        recovery_handler = self.container.get_optional(RecoveryHandlerProtocol)  # type: ignore[type-abstract]

        # Get recovery integration (optional)
        # Note: OrchestratorRecoveryIntegration is not in DI yet, will be None for now
        recovery_integration = None

        # Get streaming handler from DI container
        streaming_handler = self.container.get(StreamingHandlerProtocol)  # type: ignore[type-abstract]

        # Get context compactor from DI container (optional)
        context_compactor = self.container.get_optional(ContextCompactorProtocol)  # type: ignore[type-abstract]

        # Get unified tracker from DI container (might be scoped)
        unified_tracker = self.container.get_optional(TaskTrackerProtocol)  # type: ignore[type-abstract]
        if unified_tracker is None:
            # Create directly if not in scope (will be replaced later when in scope)
            from victor.agent.unified_task_tracker import UnifiedTaskTracker

            unified_tracker_instance: Any = UnifiedTaskTracker()

        return StreamingRecoveryCoordinator(
            recovery_handler=recovery_handler,  # type: ignore[arg-type]
            recovery_integration=recovery_integration,
            streaming_handler=streaming_handler,  # type: ignore[arg-type]
            context_compactor=context_compactor,  # type: ignore[arg-type]
            unified_tracker=unified_tracker or unified_tracker_instance,  # type: ignore[arg-type]
            settings=self._settings,
        )

    def _create_chunk_generator(self) -> Any:
        """Create ChunkGenerator instance.

        The ChunkGenerator provides a centralized interface for generating streaming
        chunks for various purposes (tool execution, status updates, metrics, content).

        Returns:
            ChunkGenerator instance
        """
        from victor.agent.chunk_generator import ChunkGenerator
        from victor.agent.protocols import StreamingHandlerProtocol

        # Get streaming handler from DI container
        streaming_handler = self.container.get(StreamingHandlerProtocol)  # type: ignore[type-abstract]

        return ChunkGenerator(
            streaming_handler=streaming_handler,  # type: ignore[arg-type]
            settings=self._settings,
        )

    def _create_tool_planner(self) -> Any:
        """Create ToolPlanner instance.

        The ToolPlanner provides a centralized interface for tool planning operations,
        including goal inference, tool sequence planning, and intent-based filtering.

        Returns:
            ToolPlanner instance
        """
        from victor.agent.tool_planner import ToolPlanner
        from victor.agent.protocols import ToolRegistrarProtocol

        # Get tool registrar from DI container
        tool_registrar = self.container.get(ToolRegistrarProtocol)  # type: ignore[type-abstract]

        return ToolPlanner(
            tool_registrar=tool_registrar,  # type: ignore[arg-type]
            settings=self._settings,
        )

    def _create_task_coordinator(self) -> Any:
        """Create TaskCoordinator instance.

        The TaskCoordinator provides a centralized interface for task coordination,
        including task preparation, intent detection, and task-specific guidance.

        Note: ConversationController is not injected as it hasn't been migrated
        to DI yet. It's passed as a parameter when calling TaskCoordinator methods.

        Returns:
            TaskCoordinator instance
        """
        from victor.agent.task_coordinator import TaskCoordinator
        from victor.agent.protocols import (
            TaskAnalyzerProtocol,
            TaskTrackerProtocol,
            SystemPromptBuilderProtocol,
        )

        # Get dependencies from DI container
        task_analyzer = self.container.get(TaskAnalyzerProtocol)  # type: ignore[type-abstract]

        # TaskTracker might be scoped
        unified_tracker = self.container.get_optional(TaskTrackerProtocol)  # type: ignore[type-abstract]
        if unified_tracker is None:
            # Create directly if not in scope
            from victor.agent.unified_task_tracker import UnifiedTaskTracker

            unified_tracker_instance2: Any = UnifiedTaskTracker()

        prompt_builder = self.container.get(SystemPromptBuilderProtocol)  # type: ignore[type-abstract]

        return TaskCoordinator(
            task_analyzer=task_analyzer,  # type: ignore[arg-type]
            unified_tracker=unified_tracker or unified_tracker_instance2,  # type: ignore[arg-type]
            prompt_builder=prompt_builder,  # type: ignore[arg-type]
            settings=self._settings,
        )

    # =========================================================================
    # New Coordinator Factory Methods (WS-D: Orchestrator SOLID Fixes)
    # =========================================================================

    def _create_tool_coordinator(self) -> Any:
        """Create ToolCoordinator instance.

        The ToolCoordinator provides a centralized interface for tool-related
        operations: selection, budgeting, and execution coordination.

        Returns:
            ToolCoordinator instance
        """
        from victor.agent.coordinators import (
            ToolCoordinator,
            ToolCoordinatorConfig,
        )
        from victor.agent.protocols import (
            ToolPipelineProtocol,
            ToolRegistryProtocol,
            IBudgetManager,
            ToolCacheProtocol,
        )

        # Get dependencies from DI container (optional for some)
        tool_pipeline = self.container.get_optional(ToolPipelineProtocol)  # type: ignore[type-abstract]
        tool_registry = self.container.get(ToolRegistryProtocol)  # type: ignore[type-abstract]
        tool_selector = self.container.get_optional(IToolSelector)  # type: ignore[type-abstract]
        budget_manager = self.container.get_optional(IBudgetManager)  # type: ignore[type-abstract]
        tool_cache = self.container.get_optional(ToolCacheProtocol)  # type: ignore[type-abstract]

        # Build config from settings
        config = ToolCoordinatorConfig(
            default_budget=getattr(self._settings, "tool_budget", 25),
            enable_caching=getattr(self._settings, "enable_tool_cache", True),
            max_tools_per_selection=getattr(self._settings, "max_tools_per_selection", 15),
            selection_threshold=getattr(self._settings, "tool_selection_threshold", 0.3),
        )

        # Note: tool_pipeline may be None if not yet registered
        # The coordinator handles this gracefully
        if tool_pipeline is None:
            logger.debug("ToolPipeline not available for ToolCoordinator")
            return None

        return ToolCoordinator(
            tool_registry=tool_registry,  # type: ignore[arg-type]
            tool_pipeline=tool_pipeline,  # type: ignore[arg-type]
            tool_selector=tool_selector,  # type: ignore[arg-type]
            budget_manager=budget_manager,  # type: ignore[arg-type]
            tool_cache=tool_cache,  # type: ignore[arg-type]
            config=config,
        )

    def _create_state_coordinator(self) -> Any:
        """Create StateCoordinator instance.

        The StateCoordinator provides a centralized interface for conversation
        state and stage transition management.

        Returns:
            StateCoordinator instance
        """
        from victor.agent.state_coordinator import (
            StateCoordinator,
            StateCoordinatorConfig,
        )
        from victor.agent.protocols import (
            ConversationControllerProtocol,
            ConversationStateMachineProtocol,
        )

        # Get dependencies from DI container
        conversation_controller = self.container.get_optional(ConversationControllerProtocol)  # type: ignore[type-abstract]
        state_machine = self.container.get_optional(ConversationStateMachineProtocol)  # type: ignore[type-abstract]

        # Build config from settings
        config = StateCoordinatorConfig(
            enable_auto_transitions=getattr(self._settings, "enable_auto_stage_transitions", True),
            enable_history_tracking=True,
            max_history_length=100,
            emit_events=getattr(self._settings, "enable_observability", True),
        )

        # Note: conversation_controller may be None if not yet registered
        if conversation_controller is None:
            logger.debug("ConversationController not available for StateCoordinator")
            return None

        return StateCoordinator(
            conversation_controller=conversation_controller,  # type: ignore[arg-type]
            state_machine=state_machine,  # type: ignore[arg-type]
            config=config,
        )

    def _create_prompt_coordinator(self) -> Any:
        """Create PromptCoordinator instance.

        The PromptCoordinator provides a centralized interface for system
        prompt assembly using PromptBuilder and vertical context.

        Returns:
            PromptCoordinator instance
        """
        from victor.agent.prompt_coordinator import (
            PromptCoordinator,
            PromptCoordinatorConfig,
        )
        from victor.framework.prompt_builder import PromptBuilder

        # Build config from settings
        config = PromptCoordinatorConfig(
            default_grounding_mode=getattr(self._settings, "grounding_mode", "minimal"),
            enable_task_hints=getattr(self._settings, "enable_task_hints", True),
            enable_vertical_sections=True,
            enable_safety_rules=True,
            max_context_tokens=getattr(self._settings, "max_context_tokens", 2000),
        )

        # Get base identity from settings or use default
        base_identity = getattr(self._settings, "base_identity", None)

        return PromptCoordinator(
            prompt_builder=PromptBuilder(),
            config=config,
            base_identity=base_identity,
        )

    def _create_tool_retry_coordinator(self) -> Any:
        """Create ToolRetryCoordinator instance.

        The ToolRetryCoordinator provides centralized retry logic for tool execution,
        including exponential backoff, cache integration, and error classification.

        Returns:
            ToolRetryCoordinator instance
        """
        from victor.agent.coordinators.tool_retry_coordinator import (
            ToolRetryCoordinator,
            ToolRetryConfig,
            create_tool_retry_coordinator,
        )
        from victor.agent.protocols import ToolExecutorProtocol, ToolCacheProtocol

        # Get dependencies from DI container
        tool_executor = self.container.get(ToolExecutorProtocol)  # type: ignore[type-abstract]
        tool_cache = self.container.get_optional(ToolCacheProtocol)  # type: ignore[type-abstract]

        # Build config from settings
        retry_enabled = getattr(self._settings, "tool_retry_enabled", True)
        max_attempts = getattr(self._settings, "tool_retry_max_attempts", 3)
        base_delay = getattr(self._settings, "tool_retry_base_delay", 1.0)
        max_delay = getattr(self._settings, "tool_retry_max_delay", 10.0)
        cache_enabled = getattr(self._settings, "enable_tool_cache", True)

        config = ToolRetryConfig(
            retry_enabled=retry_enabled,
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            cache_enabled=cache_enabled,
        )

        # Note: task_completion_detector will be set by orchestrator if available
        return ToolRetryCoordinator(
            tool_executor=tool_executor,
            tool_cache=tool_cache,
            task_completion_detector=None,  # Will be set by orchestrator
            config=config,
        )

    def _create_memory_coordinator(self) -> Any:
        """Create MemoryCoordinator instance.

        The MemoryCoordinator provides centralized memory management operations,
        including context retrieval, session statistics, and session recovery.

        Returns:
            MemoryCoordinator instance
        """
        from victor.agent.coordinators.memory_coordinator import (
            MemoryCoordinator,
            create_memory_coordinator,
        )
        from victor.agent.protocols import ToolExecutorProtocol

        # Note: Memory manager and session_id will be set by orchestrator
        # conversation_store will be provided by orchestrator
        return MemoryCoordinator(
            memory_manager=None,  # Will be set by orchestrator
            session_id=None,  # Will be set by orchestrator
            conversation_store=None,  # Will be set by orchestrator
        )

    def _create_tool_capability_coordinator(self) -> Any:
        """Create ToolCapabilityCoordinator instance.

        The ToolCapabilityCoordinator provides centralized tool capability checks,
        including tool calling support, model capability queries, and warnings.

        Returns:
            ToolCapabilityCoordinator instance
        """
        from victor.agent.coordinators.tool_capability_coordinator import (
            ToolCapabilityCoordinator,
            create_tool_capability_coordinator,
        )

        # Get tool capabilities from settings
        tool_capabilities = getattr(self._settings, "tool_capabilities", None)
        if tool_capabilities is None:
            # Create a default capability checker
            from victor.agent.tool_calling.capabilities import ToolCallingCapabilities  # type: ignore[attr-defined]

            tool_capabilities = ToolCallingCapabilities()

        # Console will be set by orchestrator (for user-facing messages)
        return ToolCapabilityCoordinator(
            tool_capabilities=tool_capabilities,
            console=None,  # Will be set by orchestrator
            warn_once=True,
        )

    def _create_tool_call_coordinator(self) -> Any:
        """Create ToolCallCoordinator instance.

        The ToolCallCoordinator provides centralized tool call coordination,
        including validation, parsing, execution with retry logic, and result formatting.

        Returns:
            ToolCallCoordinator instance
        """
        from victor.agent.coordinators.tool_call_coordinator import (
            ToolCallCoordinator,
            create_tool_call_coordinator,
        )
        from victor.agent.coordinators.tool_call_protocol import ToolCallCoordinatorConfig
        from victor.agent.protocols import (
            ToolExecutorProtocol,
            ToolRegistryProtocol,
        )

        # Get dependencies from DI container
        tool_executor = self.container.get(ToolExecutorProtocol)  # type: ignore[type-abstract]
        tool_registry = self.container.get(ToolRegistryProtocol)  # type: ignore[type-abstract]
        tool_retry_coordinator = self.container.get_optional(ToolRetryCoordinator)

        # Build config from settings
        config = ToolCallCoordinatorConfig(
            max_retries=getattr(self._settings, "tool_retry_max_attempts", 3),
            retry_delay=getattr(self._settings, "tool_retry_base_delay", 1.0),
            retry_backoff_multiplier=getattr(self._settings, "tool_retry_backoff_multiplier", 2.0),
            parallel_execution=getattr(self._settings, "enable_parallel_tool_execution", False),
            timeout_seconds=getattr(self._settings, "tool_execution_timeout", 30.0),
            strict_validation=getattr(self._settings, "strict_tool_validation", True),
        )

        # Get sanitizer if available
        from victor.agent.tool_calling.sanitizer import ToolNameSanitizer

        sanitizer = ToolNameSanitizer()

        return create_tool_call_coordinator(
            config=config,
            tool_executor=tool_executor,
            tool_registry=tool_registry,
            tool_retry_coordinator=tool_retry_coordinator,
            sanitizer=sanitizer,
        )

    def _create_prompt_builder_coordinator(self) -> Any:
        """Create PromptBuilderCoordinator instance.

        The PromptBuilderCoordinator provides centralized prompt building,
        including mode-specific prompts, thinking mode handling, and tool hints.

        Returns:
            PromptBuilderCoordinator instance
        """
        from victor.agent.coordinators.prompt_builder_coordinator import (
            PromptBuilderCoordinator,
            create_prompt_builder_coordinator,
        )
        from victor.agent.coordinators.prompt_builder_protocol import PromptBuilderCoordinatorConfig

        # Build config from settings
        config = PromptBuilderCoordinatorConfig(
            cache_enabled=getattr(self._settings, "enable_prompt_cache", True),
            include_tool_hints=getattr(self._settings, "include_tool_hints_in_prompt", True),
            include_thinking_instructions=getattr(
                self._settings, "include_thinking_instructions", True
            ),
            max_prompt_length=getattr(self._settings, "max_system_prompt_length", 50000),
        )

        # Get base system prompt from settings
        base_prompt = getattr(self._settings, "base_system_prompt", "")

        return create_prompt_builder_coordinator(
            config=config,
            base_prompt=base_prompt,
        )

    # =========================================================================
    # Presentation Abstraction Layer Factory Methods
    # =========================================================================

    def _register_presentation_adapter(self, container: ServiceContainer) -> None:
        """Register PresentationAdapter as singleton.

        The PresentationAdapter provides a clean abstraction for icon/emoji
        rendering, decoupling the agent layer from direct UI dependencies.

        Uses EmojiPresentationAdapter by default, which respects the
        `use_emojis` setting from configuration.
        """
        from victor.agent.presentation import (
            PresentationProtocol,
            EmojiPresentationAdapter,
        )

        container.register(
            PresentationProtocol,  # type: ignore[type-abstract]
            lambda c: EmojiPresentationAdapter(),
            ServiceLifetime.SINGLETON,
        )

    # =========================================================================
    # Agentic AI Service Factory Methods (Phase 3 Integration)
    # =========================================================================

    def _register_hierarchical_planner(self, container: ServiceContainer) -> None:
        """Register HierarchicalPlanner as singleton.

        The HierarchicalPlanner provides hierarchical task decomposition
        for complex goal-oriented execution.

        Args:
            container: DI container to register services in
        """
        from victor.agent.protocols_agentic_ai import HierarchicalPlannerProtocol

        container.register(
            HierarchicalPlannerProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_hierarchical_planner(),
            ServiceLifetime.SINGLETON,
        )

    def _create_hierarchical_planner(self) -> Any:
        """Create HierarchicalPlanner instance.

        Returns:
            HierarchicalPlanner instance configured with settings
        """
        from victor.agent.planning import HierarchicalPlanner

        # Orchestrator and provider_manager will be set later by dependency injection
        return HierarchicalPlanner(
            orchestrator=None,  # Will be set by orchestrator
            provider_manager=None,  # Will be set by orchestrator
            event_bus=None,  # Will be set by orchestrator
        )

    def _register_episodic_memory(self, container: ServiceContainer) -> None:
        """Register EpisodicMemory as singleton.

        The EpisodicMemory provides storage and retrieval of agent experiences.

        Args:
            container: DI container to register services in
        """
        from victor.agent.protocols_agentic_ai import EpisodicMemoryProtocol

        container.register(
            EpisodicMemoryProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_episodic_memory(),
            ServiceLifetime.SINGLETON,
        )

    def _create_episodic_memory(self) -> Any:
        """Create EpisodicMemory instance.

        Returns:
            EpisodicMemory instance configured with settings
        """
        from victor.agent.memory import create_episodic_memory

        max_episodes = getattr(self._settings, "episodic_memory_max_episodes", 1000)
        decay_rate = getattr(self._settings, "episodic_memory_decay_rate", 0.01)
        consolidation_threshold = getattr(
            self._settings, "episodic_memory_consolidation_interval", 100
        )

        return create_episodic_memory(
            max_episodes=max_episodes,
            decay_rate=decay_rate,
            consolidation_threshold=consolidation_threshold,
        )

    def _register_semantic_memory(self, container: ServiceContainer) -> None:
        """Register SemanticMemory as singleton.

        The SemanticMemory provides storage and querying of factual knowledge.

        Args:
            container: DI container to register services in
        """
        from victor.agent.protocols_agentic_ai import SemanticMemoryProtocol

        container.register(
            SemanticMemoryProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_semantic_memory(),
            ServiceLifetime.SINGLETON,
        )

    def _create_semantic_memory(self) -> Any:
        """Create SemanticMemory instance.

        Returns:
            SemanticMemory instance configured with settings
        """
        from victor.agent.memory import SemanticMemory

        max_knowledge = getattr(self._settings, "semantic_memory_max_facts", 5000)

        return SemanticMemory(max_knowledge=max_knowledge)

    def _register_skill_discovery(self, container: ServiceContainer) -> None:
        """Register SkillDiscoveryEngine as singleton.

        The SkillDiscoveryEngine provides dynamic tool discovery and composition.

        Args:
            container: DI container to register services in
        """
        from victor.agent.protocols_agentic_ai import SkillDiscoveryProtocol

        container.register(
            SkillDiscoveryProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_skill_discovery(),
            ServiceLifetime.SINGLETON,
        )

    def _create_skill_discovery(self) -> Any:
        """Create SkillDiscoveryEngine instance.

        Returns:
            SkillDiscoveryEngine instance configured with settings
        """
        from victor.agent.skills import SkillDiscoveryEngine
        from victor.agent.protocols import ToolRegistryProtocol

        # Get tool registry from container
        tool_registry = self.container.get(ToolRegistryProtocol)  # type: ignore[type-abstract]

        return SkillDiscoveryEngine(
            tool_registry=tool_registry,
            tool_selector=None,  # Will be set by orchestrator
            event_bus=None,  # Will be set by orchestrator
        )

    def _register_skill_chainer(self, container: ServiceContainer) -> None:
        """Register SkillChainer as singleton.

        The SkillChainer provides multi-step skill chain planning and execution.

        Args:
            container: DI container to register services in
        """
        from victor.agent.protocols_agentic_ai import SkillChainerProtocol

        container.register(
            SkillChainerProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_skill_chainer(),
            ServiceLifetime.SINGLETON,
        )

    def _create_skill_chainer(self) -> Any:
        """Create SkillChainer instance.

        Returns:
            SkillChainer instance configured with settings
        """
        from victor.agent.skills import SkillChainer

        return SkillChainer()

    def _register_proficiency_tracker(self, container: ServiceContainer) -> None:
        """Register ProficiencyTracker as singleton.

        The ProficiencyTracker provides performance tracking and improvement.

        Args:
            container: DI container to register services in
        """
        from victor.agent.protocols_agentic_ai import ProficiencyTrackerProtocol

        container.register(
            ProficiencyTrackerProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_proficiency_tracker(),
            ServiceLifetime.SINGLETON,
        )

    def _create_proficiency_tracker(self) -> Any:
        """Create ProficiencyTracker instance.

        Returns:
            ProficiencyTracker instance configured with settings
        """
        from victor.agent.improvement import ProficiencyTracker

        return ProficiencyTracker()

    def _register_rl_coordinator(self, container: ServiceContainer) -> None:
        """Register RLCoordinator as singleton.

        The RLCoordinator provides reinforcement learning for decision optimization.

        Args:
            container: DI container to register services in
        """
        from victor.agent.protocols_agentic_ai import RLCoordinatorProtocol

        container.register(
            RLCoordinatorProtocol,  # type: ignore[type-abstract]
            lambda c: self._create_rl_coordinator(),
            ServiceLifetime.SINGLETON,
        )

    # =========================================================================
    # Architecture Documentation: Intentionally Unregistered Protocols
    # =========================================================================

    # The following protocols are intentionally NOT registered in the ServiceProvider.
    # This is a deliberate architectural decision to prioritize flexibility over
    # pure DI container usage. See PROTOCOL_ANALYSIS_REPORT.md for details.

    # Category 1: Orchestrator-Specific Components
    # -------------------------------------------
    # These components have orchestrator-specific dependencies (callbacks, state,
    # budget) that are provided during creation in OrchestratorFactory. They are
    # resolved via get_optional() to allow the factory to create them with
    # orchestrator-specific configuration.

    # StreamingControllerProtocol:
    #   - Requires: metrics_collector, on_session_complete callback
    #   - Reason: Callback is orchestrator-specific
    #   - Created in: OrchestratorFactory.create_streaming_controller()
    #   - Resolution: container.get_optional(StreamingControllerProtocol)

    # ToolPipelineProtocol:
    #   - Requires: tool_registry, tool_executor, config, callbacks
    #   - Reason: Many orchestrator-specific dependencies (on_tool_start, on_tool_complete)
    #   - Created in: OrchestratorFactory.create_tool_pipeline()
    #   - Resolution: container.get_optional(ToolPipelineProtocol)

    # ConversationControllerProtocol:
    #   - Requires: provider, model, conversation, state, callbacks
    #   - Reason: Orchestrator-specific configuration and state
    #   - Created in: OrchestratorFactory.create_conversation_controller()
    #   - Resolution: container.get_optional(ConversationControllerProtocol)

    # Category 2: Framework-Level Protocols
    # ------------------------------------
    # These protocols are defined for canonical typing but used by verticals
    # or other systems, not the core orchestrator. They may be registered
    # elsewhere or intentionally left unregistered.

    # Team coordination: ITeamCoordinator, ITeamMember, IObservableCoordinator
    # Multi-agent: IAgentFactory, IAgent
    # Search: ISemanticSearch, IIndexable
    # Provider infrastructure: IProviderAdapter, IProviderHealthMonitor
    # Quality/Grounding: IQualityAssessor, IGroundingStrategy
    # Coordination: ModeWorkflowTeamCoordinatorProtocol, TeamSelectionStrategyProtocol

    # Category 3: Orchestrator-Implemented Protocols
    # ---------------------------------------------
    # These protocols are implemented BY the orchestrator itself, not as
    # separate services.

    # VerticalStorageProtocol:
    #   - Implemented by: AgentOrchestrator
    #   - Methods: set_middleware(), get_middleware(), set_safety_patterns(), etc.
    #   - Reason: Orchestrator is the storage location for vertical data

    # Design Rationale:
    # -----------------
    # 1. Flexibility: Factory can create components with specific dependencies
    # 2. Testing: Components can be mocked via factory or container override
    # 3. Simplicity: Avoid complex registration logic for orchestrator-specific deps
    # 4. Performance: Services are created on-demand in factory, not pre-registered
    #
    # This design prioritizes practical flexibility over pure DI patterns while
    # still maintaining loose coupling through protocols.


# =============================================================================
# Null implementations for graceful degradation
# =============================================================================


class _NullObservability:
    """No-op observability implementation."""

    def on_tool_start(self, tool_name: str, arguments: dict[str, Any], tool_id: str) -> None:
        pass

    def on_tool_end(
        self,
        tool_name: str,
        result: Any,
        success: bool,
        tool_id: str,
        error: Optional[str] = None,
    ) -> None:
        pass

    def wire_state_machine(self, state_machine: Any) -> None:
        pass

    def on_error(self, error: Exception, context: dict[str, Any]) -> None:
        pass


class _NullTaskAnalyzer:
    """No-op task analyzer implementation."""

    def analyze(self, prompt: str) -> dict[str, Any]:
        return {"complexity": "unknown", "intent": "unknown"}

    def classify_complexity(self, prompt: str) -> Any:
        return None

    def detect_intent(self, prompt: str) -> Any:
        return None


class _NullRecoveryHandler:
    """No-op recovery handler implementation for disabled mode."""

    @property
    def enabled(self) -> bool:
        return False

    @property
    def consecutive_failures(self) -> int:
        return 0

    def detect_failure(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def recover(self, *args: Any, **kwargs: Any) -> Any:
        # Return a no-op outcome
        from dataclasses import dataclass, field
        from enum import Enum, auto

        class _RecoveryAction(Enum):
            CONTINUE = auto()

        @dataclass
        class _RecoveryResult:
            action: _RecoveryAction = _RecoveryAction.CONTINUE
            success: bool = True
            strategy_name: str = "disabled"
            reason: str = "Recovery system disabled"

        @dataclass
        class _RecoveryOutcome:
            result: _RecoveryResult = field(default_factory=_RecoveryResult)

        return _RecoveryOutcome()

    def record_outcome(self, success: bool, quality_improvement: float = 0.0) -> None:
        pass

    def track_response(self, content: str) -> None:
        pass

    def reset_session(self, session_id: str) -> None:
        pass

    def set_context_compactor(self, compactor: Any) -> None:
        pass

    def set_session_id(self, session_id: str) -> None:
        pass

    def get_diagnostics(self) -> dict[str, Any]:
        return {"enabled": False}


class _NullMemoryCoordinator:
    """No-op memory coordinator implementation for disabled mode."""

    async def search_all(
        self,
        query: str,
        limit: int = 20,
        memory_types: Optional[list[Any]] = None,
        session_id: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        min_relevance: float = 0.0,
    ) -> list[Any]:
        return []

    async def search_type(
        self,
        memory_type: Any,
        query: str,
        limit: int = 20,
        **kwargs: Any,
    ) -> list[Any]:
        return []

    async def store(
        self,
        memory_type: Any,
        key: str,
        value: Any,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        return False

    async def get(
        self,
        memory_type: Any,
        key: str,
    ) -> Optional[Any]:
        return None

    def register_provider(self, provider: Any) -> None:
        pass

    def unregister_provider(self, memory_type: Any) -> bool:
        return False

    def get_registered_types(self) -> list[Any]:
        return []

    def get_stats(self) -> dict[str, Any]:
        return {"enabled": False, "providers": []}


# =============================================================================
# Convenience function
# =============================================================================


def configure_orchestrator_services(
    container: ServiceContainer,
    settings: "Settings",
) -> None:
    """Configure all orchestrator services in one call.

    Convenience function for application bootstrap. Creates a service
    provider and registers all services.

    Args:
        container: DI container to register services in
        settings: Application settings

    Example:
        from victor.core.container import ServiceContainer
        from victor.agent.service_provider import configure_orchestrator_services

        container = ServiceContainer()
        configure_orchestrator_services(container, settings)
    """
    provider = OrchestratorServiceProvider(settings)
    provider.register_services(container)


__all__ = [
    "OrchestratorServiceProvider",
    "configure_orchestrator_services",
]
