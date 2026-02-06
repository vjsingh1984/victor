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

"""Factory for creating AgentOrchestrator components.

This module extracts component initialization logic from AgentOrchestrator.__init__
to reduce its complexity and improve testability.

Part of CRITICAL-001: Monolithic Orchestrator decomposition.

Usage:
    factory = OrchestratorFactory(settings, provider, model)
    orchestrator = factory.create_orchestrator()

Or use individual creation methods for testing:
    factory = OrchestratorFactory(settings, provider, model)
    sanitizer = factory.create_sanitizer()
    prompt_builder = factory.create_prompt_builder()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
from collections.abc import Callable

from rich.console import Console

# Mode-aware mixin for consistent mode controller access
from victor.protocols.mode_aware import ModeAwareMixin

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings
    from victor.providers.base import BaseProvider
    from victor.agent.tool_calling import BaseToolCallingAdapter, ToolCallingCapabilities
    from victor.agent.response_sanitizer import ResponseSanitizer
    from victor.agent.prompt_builder import SystemPromptBuilder
    from victor.agent.context_project import ProjectContext
    from victor.framework.task import TaskComplexityService as ComplexityClassifier
    from victor.agent.action_authorizer import ActionAuthorizer
    from victor.agent.search_router import SearchRouter
    from victor.agent.metrics_collector import MetricsCollector
    from victor.agent.conversation_controller import ConversationController
    from victor.agent.streaming_controller import StreamingController
    from victor.agent.context_compactor import ContextCompactor
    from victor.agent.usage_analytics import UsageAnalytics
    from victor.agent.tool_sequence_tracker import ToolSequenceTracker
    from victor.agent.tool_output_formatter import ToolOutputFormatter
    from victor.agent.recovery import RecoveryHandler
    from victor.observability.integration import ObservabilityIntegration
    from victor.agent.coordinators.response_coordinator import ResponseCoordinator
    from victor.agent.coordinators.config_coordinator import ToolAccessConfigCoordinator
    from victor.agent.coordinators.state_coordinator import StateCoordinator
    from victor.agent.session_state_manager import SessionStateManager
    from victor.agent.conversation_state import ConversationStateMachine

logger = logging.getLogger(__name__)


@dataclass
class ProviderComponents:
    """Components related to provider and tool calling."""

    provider: "BaseProvider"
    model: str
    provider_name: str
    tool_adapter: "BaseToolCallingAdapter"
    tool_calling_caps: "ToolCallingCapabilities"


@dataclass
class CoreServices:
    """Core service components resolved via DI or fallback."""

    sanitizer: "ResponseSanitizer"
    prompt_builder: "SystemPromptBuilder"
    project_context: "ProjectContext"
    complexity_classifier: "ComplexityClassifier"
    action_authorizer: "ActionAuthorizer"
    search_router: "SearchRouter"


@dataclass
class ConversationComponents:
    """Components for conversation management."""

    conversation_controller: "ConversationController"
    memory_manager: Optional[Any] = None
    memory_session_id: Optional[str] = None
    conversation_state: Optional[Any] = None


@dataclass
class ToolComponents:
    """Components for tool management and execution."""

    tool_registry: Any
    tool_registrar: Any
    tool_executor: Any
    tool_cache: Optional[Any] = None
    tool_graph: Optional[Any] = None
    plugin_manager: Optional[Any] = None


@dataclass
class StreamingComponents:
    """Components for streaming and metrics."""

    streaming_controller: "StreamingController"
    streaming_handler: Any
    metrics_collector: "MetricsCollector"
    streaming_metrics_collector: Optional[Any] = None


@dataclass
class AnalyticsComponents:
    """Components for analytics and tracking."""

    usage_analytics: "UsageAnalytics"
    sequence_tracker: "ToolSequenceTracker"
    unified_tracker: Any


@dataclass
class RecoveryComponents:
    """Components for error recovery and resilience."""

    recovery_handler: Optional["RecoveryHandler"]
    recovery_integration: Any
    context_compactor: "ContextCompactor"


@dataclass
class WorkflowOptimizationComponents:
    """Components for workflow optimizations.

    These components address MODE workflow issues:
    - TaskCompletionDetector: Detects when task objectives are met
    - ReadResultCache: Caches file reads to prevent redundant operations
    - TimeAwareExecutor: Manages execution with time budget awareness
    - ThinkingPatternDetector: Detects and breaks thinking loops
    - ResourceManager: Centralized resource lifecycle management
    - ModeCompletionChecker: Mode-specific early exit detection
    """

    task_completion_detector: Optional[Any] = None
    read_cache: Optional[Any] = None
    time_aware_executor: Optional[Any] = None
    thinking_detector: Optional[Any] = None
    resource_manager: Optional[Any] = None
    mode_completion_criteria: Optional[Any] = None


@dataclass
class CoordinatorComponents:
    """New coordinator components (Stream E4).

    These coordinators extract specialized logic from the orchestrator:
    - ResponseCoordinator: Response processing and sanitization
    - ToolAccessConfigCoordinator: Tool access configuration management
    - StateCoordinator: Unified state management

    Design Patterns:
        - Facade Pattern: Simplified access to complex subsystems
        - SRP: Each coordinator has a single responsibility
        - ISP: Focused protocols for coordinator operations
    """

    response_coordinator: Optional["ResponseCoordinator"] = None
    tool_access_config_coordinator: Optional["ToolAccessConfigCoordinator"] = None
    state_coordinator: Optional["StateCoordinator"] = None


@dataclass
class OrchestratorComponents:
    """All components needed to construct an AgentOrchestrator.

    This dataclass serves as a transfer object containing all initialized
    components, allowing the orchestrator's __init__ to be simplified to
    just assigning these pre-constructed components.
    """

    # Provider
    provider: Optional[ProviderComponents] = field(default=None)

    # Core services
    services: Optional[CoreServices] = field(default=None)

    # Conversation
    conversation: Optional[ConversationComponents] = field(default=None)

    # Tools
    tools: Optional[ToolComponents] = field(default=None)

    # Streaming
    streaming: Optional[StreamingComponents] = field(default=None)

    # Analytics
    analytics: Optional[AnalyticsComponents] = field(default=None)

    # Recovery
    recovery: Optional[RecoveryComponents] = field(default=None)

    # Observability
    observability: Optional["ObservabilityIntegration"] = None

    # Tool output formatter
    tool_output_formatter: Optional["ToolOutputFormatter"] = None

    # Workflow optimizations
    workflow_optimization: WorkflowOptimizationComponents = field(
        default_factory=WorkflowOptimizationComponents
    )

    # New coordinators (Stream E4)
    coordinators: CoordinatorComponents = field(default_factory=CoordinatorComponents)
    # Raw attribute snapshot (Phase 1 compatibility bridge)
    attributes: dict[str, Any] = field(default_factory=dict)


class OrchestratorFactory(ModeAwareMixin):
    """Factory for creating AgentOrchestrator components.

    _container: Optional["ServiceContainer"]  # Lazy-initialized container

    This factory extracts component initialization logic from the orchestrator's
    __init__ method, providing:

    1. Cleaner separation of concerns
    2. Easier testing of individual component creation
    3. Potential for lazy initialization
    4. Reduced orchestrator complexity

    Uses ModeAwareMixin for consistent mode controller access when applying
    mode-specific configurations (e.g., exploration multipliers).

    Example:
        factory = OrchestratorFactory(settings, provider, model)

        # Create all components at once
        components = factory.create_all_components()

        # Or create individual components for testing
        sanitizer = factory.create_sanitizer()
    """

    def __init__(
        self,
        settings: "Settings",
        provider: "BaseProvider",
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        console: Optional["Console"] = None,
        provider_name: Optional[str] = None,
        profile_name: Optional[str] = None,
        tool_selection: Optional[dict[str, Any]] = None,
        thinking: bool = False,
    ):
        """Initialize the factory with core configuration.

        Args:
            settings: Application settings
            provider: LLM provider instance
            model: Model identifier
            temperature: Sampling temperature (uses settings if None)
            max_tokens: Maximum tokens to generate (uses settings if None)
            console: Optional rich console for output
            provider_name: Provider label (for OpenAI-compatible disambiguation)
            profile_name: Profile name for session tracking
            tool_selection: Tool selection configuration
            thinking: Enable extended thinking mode
        """
        self.settings = settings
        self.provider = provider
        self.model = model
        # Use parameter if provided, otherwise fall back to settings
        self.temperature = (
            temperature if temperature is not None else getattr(settings, "temperature", 0.7)
        )
        self.max_tokens = (
            max_tokens if max_tokens is not None else getattr(settings, "max_tokens", 4096)
        )
        self.console = console
        self.provider_name = provider_name
        self.profile_name = profile_name
        self.tool_selection = tool_selection or {}
        self.thinking = thinking

        # Lazy-initialized container
        self._container = None

    def _builder_sequence(self) -> list[type]:
        """Return the ordered builder sequence for orchestrator initialization."""
        from victor.agent.builders.provider_layer_builder import ProviderLayerBuilder
        from victor.agent.builders.prompting_builder import PromptingBuilder
        from victor.agent.builders.session_services_builder import SessionServicesBuilder
        from victor.agent.builders.metrics_logging_builder import MetricsLoggingBuilder
        from victor.agent.builders.workflow_memory_builder import WorkflowMemoryBuilder
        from victor.agent.builders.workflow_chat_builder import WorkflowChatBuilder
        from victor.agent.builders.intelligent_integration_builder import (
            IntelligentIntegrationBuilder,
        )
        from victor.agent.builders.tooling_builder import ToolingBuilder
        from victor.agent.builders.conversation_pipeline_builder import (
            ConversationPipelineBuilder,
        )
        from victor.agent.builders.context_intelligence_builder import (
            ContextIntelligenceBuilder,
        )
        from victor.agent.builders.recovery_observability_builder import (
            RecoveryObservabilityBuilder,
        )
        from victor.agent.builders.config_workflow_builder import ConfigWorkflowBuilder
        from victor.agent.builders.finalization_builder import FinalizationBuilder

        return [
            ProviderLayerBuilder,
            PromptingBuilder,
            SessionServicesBuilder,
            MetricsLoggingBuilder,
            WorkflowMemoryBuilder,
            WorkflowChatBuilder,  # Phase 1: Domain-Agnostic Workflow Chat
            IntelligentIntegrationBuilder,
            ToolingBuilder,
            ConversationPipelineBuilder,
            ContextIntelligenceBuilder,
            RecoveryObservabilityBuilder,
            ConfigWorkflowBuilder,
            FinalizationBuilder,
        ]

    def initialize_orchestrator(self, orchestrator: "AgentOrchestrator") -> None:
        """Initialize an orchestrator instance using the factory."""
        settings = self.settings
        provider = self.provider
        model = self.model
        temperature = self.temperature
        max_tokens = self.max_tokens
        console = self.console
        tool_selection = self.tool_selection
        thinking = self.thinking
        provider_name = self.provider_name
        profile_name = self.profile_name

        # Store profile name for session tracking
        orchestrator._profile_name = profile_name
        # Track active session ID for parallel session support
        if not hasattr(orchestrator, "active_session_id"):
            orchestrator.active_session_id = None
        # Bootstrap DI container - ensures all services are available
        # This is idempotent and will only bootstrap if not already done
        orchestrator._container = self.container
        # Share factory on orchestrator for compatibility
        orchestrator._factory = self

        orchestrator.settings = settings
        orchestrator._settings = settings  # Alias for internal use
        orchestrator.temperature = temperature
        orchestrator.max_tokens = max_tokens
        orchestrator.console = console or Console()
        orchestrator.tool_selection = tool_selection or {}
        orchestrator.thinking = thinking

        builder_context = {
            "provider": provider,
            "model": model,
            "provider_name": provider_name,
            "profile_name": profile_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "console": console,
            "tool_selection": tool_selection,
            "thinking": thinking,
        }

        for builder_cls in self._builder_sequence():
            builder = builder_cls(settings=settings, factory=self)
            builder.build(orchestrator, **builder_context)

    def create_orchestrator(self) -> "AgentOrchestrator":
        """Create a fully initialized AgentOrchestrator.

        This centralizes orchestration creation in a single composition root.
        Initialization is performed via initialize_orchestrator() to avoid
        double factory creation when AgentOrchestrator.__init__ is bypassed.
        """
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
        self.initialize_orchestrator(orchestrator)
        return orchestrator

    def create_all_components(self) -> OrchestratorComponents:
        """Create all orchestrator components.

        Phase 1 builds a fully initialized orchestrator, then extracts
        grouped components plus a raw attribute snapshot for compatibility.
        """
        orchestrator = self.create_orchestrator()

        provider_components = ProviderComponents(
            provider=orchestrator.provider,
            model=orchestrator.model,
            provider_name=orchestrator.provider_name or "",
            tool_adapter=orchestrator.tool_adapter,
            tool_calling_caps=orchestrator._tool_calling_caps_internal,
        )
        core_services = CoreServices(
            sanitizer=orchestrator.sanitizer,
            prompt_builder=orchestrator.prompt_builder,
            project_context=orchestrator.project_context,
            complexity_classifier=orchestrator.task_classifier,
            action_authorizer=orchestrator.intent_detector,
            search_router=orchestrator.search_router,
        )
        conversation_components = ConversationComponents(
            conversation_controller=orchestrator._conversation_controller,
            memory_manager=orchestrator.memory_manager,
            memory_session_id=orchestrator._memory_session_id,
            conversation_state=orchestrator.conversation_state,
        )
        tool_components = ToolComponents(
            tool_registry=orchestrator.tools,
            tool_registrar=orchestrator.tool_registrar,
            tool_executor=orchestrator.tool_executor,
            tool_cache=orchestrator.tool_cache,
            tool_graph=orchestrator.tool_graph,
            plugin_manager=orchestrator.plugin_manager,
        )
        streaming_components = StreamingComponents(
            streaming_controller=orchestrator._streaming_controller,
            streaming_handler=orchestrator._streaming_handler,
            metrics_collector=orchestrator._metrics_collector,
            streaming_metrics_collector=orchestrator.streaming_metrics_collector,
        )
        analytics_components = AnalyticsComponents(
            usage_analytics=orchestrator._usage_analytics,
            sequence_tracker=orchestrator._sequence_tracker,
            unified_tracker=orchestrator.unified_tracker,
        )
        recovery_components = RecoveryComponents(
            recovery_handler=orchestrator._recovery_handler,
            recovery_integration=orchestrator._recovery_integration,
            context_compactor=orchestrator._context_compactor,
        )
        coordinator_components = CoordinatorComponents(
            response_coordinator=orchestrator._response_coordinator,
            tool_access_config_coordinator=orchestrator._tool_access_config_coordinator,
            state_coordinator=orchestrator._state_coordinator,
        )

        return OrchestratorComponents(
            provider=provider_components,
            services=core_services,
            conversation=conversation_components,
            tools=tool_components,
            streaming=streaming_components,
            analytics=analytics_components,
            recovery=recovery_components,
            observability=orchestrator._observability,
            tool_output_formatter=orchestrator._tool_output_formatter,
            workflow_optimization=orchestrator._workflow_optimization,
            coordinators=coordinator_components,
            attributes=dict(orchestrator.__dict__),
        )

    @property
    def container(self) -> Any:
        """Get or create the DI container."""
        if self._container is None:
            from victor.core.bootstrap import ensure_bootstrapped
            from victor.agent.protocols import ResponseSanitizerProtocol
            from victor.core.container import ServiceContainer

            new_container: ServiceContainer = ensure_bootstrapped(self.settings)
            self._container = new_container  # type: ignore[assignment]

            # Verify orchestrator services are registered
            # If ResponseSanitizer is not registered, bootstrapping was incomplete
            if not new_container.is_registered(ResponseSanitizerProtocol):
                # Need to re-bootstrap with orchestrator services
                from victor.core.bootstrap import bootstrap_container

                # Create a new fully-bootstrapped container
                self._container = bootstrap_container(self.settings)  # type: ignore[assignment]

        return self._container

    def create_sanitizer(self) -> "ResponseSanitizer":
        """Create response sanitizer (from DI container)."""
        from victor.agent.protocols import ResponseSanitizerProtocol

        return self.container.get(ResponseSanitizerProtocol)  # type: ignore[no-any-return]

    def create_prompt_builder(
        self,
        tool_adapter: "BaseToolCallingAdapter",
        capabilities: "ToolCallingCapabilities",
    ) -> "SystemPromptBuilder":
        """Create system prompt builder.

        Args:
            tool_adapter: Tool calling adapter
            capabilities: Tool calling capabilities

        Returns:
            Configured SystemPromptBuilder
        """
        from victor.agent.prompt_builder import SystemPromptBuilder
        from victor.core.verticals.protocols import VerticalExtensions

        # Get prompt contributors from vertical extensions
        prompt_contributors = []
        try:
            extensions = self.container.get_optional(VerticalExtensions)
            if extensions and extensions.prompt_contributors:
                prompt_contributors = extensions.prompt_contributors
        except Exception as e:
            logger.debug(f"Could not load vertical prompt contributors: {e}")

        return SystemPromptBuilder(
            provider_name=self.provider_name or self.provider.__class__.__name__.lower(),
            model=self.model,
            tool_adapter=tool_adapter,
            capabilities=capabilities,
            prompt_contributors=prompt_contributors,
        )

    def create_project_context(self) -> "ProjectContext":
        """Create project context loader (from DI container)."""
        from victor.agent.protocols import ProjectContextProtocol

        return self.container.get(ProjectContextProtocol)

    def create_complexity_classifier(self) -> "ComplexityClassifier":
        """Create complexity classifier (from DI container)."""
        from victor.agent.protocols import ComplexityClassifierProtocol

        return self.container.get(ComplexityClassifierProtocol)  # type: ignore[no-any-return]

    def create_action_authorizer(self) -> "ActionAuthorizer":
        """Create action authorizer (from DI container)."""
        from victor.agent.protocols import ActionAuthorizerProtocol

        return self.container.get(ActionAuthorizerProtocol)  # type: ignore[no-any-return]

    def create_search_router(self) -> "SearchRouter":
        """Create search router (from DI container)."""
        from victor.agent.protocols import SearchRouterProtocol

        return self.container.get(SearchRouterProtocol)  # type: ignore[no-any-return]

    def create_presentation_adapter(self) -> Any:
        """Create presentation adapter for icon/emoji rendering.

        The presentation adapter provides a clean abstraction for presentation
        concerns, decoupling the agent layer from direct UI dependencies.

        Returns:
            PresentationProtocol implementation (EmojiPresentationAdapter by default)
        """
        from victor.agent.presentation import PresentationProtocol, EmojiPresentationAdapter

        # Try to get from container, fallback to direct instantiation
        adapter = self.container.get_optional(PresentationProtocol)
        if adapter is None:
            # Container not bootstrapped yet, create directly
            adapter = EmojiPresentationAdapter()
        return adapter

    # ==========================================================================
    # Coordinator Creation Methods (Phase 1.4)
    # ==========================================================================

    def create_config_coordinator(self, config_providers: Optional[list[Any]] = None) -> Any:
        """Create configuration coordinator.

        Args:
            config_providers: Optional list of config providers (IConfigProvider)

        Returns:
            ConfigCoordinator instance
        """
        from victor.agent.coordinators.config_coordinator import ConfigCoordinator

        return ConfigCoordinator(providers=config_providers or [])

    def create_prompt_coordinator(self, prompt_contributors: Optional[list[Any]] = None) -> Any:
        """Create prompt coordinator.

        Args:
            prompt_contributors: Optional list of prompt contributors (IPromptContributor)

        Returns:
            PromptCoordinator instance
        """
        from victor.agent.coordinators.prompt_coordinator import PromptCoordinator

        return PromptCoordinator(contributors=prompt_contributors or [])

    def create_context_coordinator(self, compaction_strategies: Optional[list[Any]] = None) -> Any:
        """Create context coordinator.

        Args:
            compaction_strategies: Optional list of compaction strategies (ICompactionStrategy)

        Returns:
            ContextCoordinator instance
        """
        from victor.agent.coordinators.context_coordinator import ContextCoordinator

        return ContextCoordinator(strategies=compaction_strategies or [])

    def create_analytics_coordinator(
        self,
        analytics_exporters: Optional[list[Any]] = None,
        enable_console_exporter: bool = False,
    ) -> Any:
        """Create analytics coordinator.

        Args:
            analytics_exporters: Optional list of analytics exporters (IAnalyticsExporter)
            enable_console_exporter: If True, adds ConsoleAnalyticsExporter

        Returns:
            AnalyticsCoordinator instance
        """
        from victor.agent.coordinators.analytics_coordinator import (
            AnalyticsCoordinator,
            ConsoleAnalyticsExporter,
        )

        exporters = analytics_exporters or []

        # Add console exporter if enabled
        if enable_console_exporter:
            exporters.append(ConsoleAnalyticsExporter())

        return AnalyticsCoordinator(exporters=exporters)

    # ========================================================================
    # New Coordinator Creation Methods (Stream E4)
    # ========================================================================

    def create_response_coordinator(
        self,
        tool_adapter: Optional["BaseToolCallingAdapter"] = None,
        tool_registry: Optional[Any] = None,
    ) -> "ResponseCoordinator":
        """Create response coordinator for response processing and sanitization.

        The ResponseCoordinator consolidates response processing logic including:
        - Content sanitization (clean malformed content from local models)
        - Tool call parsing and validation
        - Garbage content detection
        - Streaming chunk aggregation

        Args:
            tool_adapter: Optional tool calling adapter for parsing
            tool_registry: Optional tool registry for validation

        Returns:
            ResponseCoordinator instance configured with dependencies
        """
        from victor.agent.coordinators.response_coordinator import (
            ResponseCoordinator,
            ResponseCoordinatorConfig,
        )

        sanitizer = self.create_sanitizer()

        config = ResponseCoordinatorConfig(
            max_garbage_chunks=getattr(self.settings, "max_garbage_chunks", 3),
            enable_tool_call_extraction=getattr(self.settings, "enable_tool_call_extraction", True),
            enable_content_sanitization=getattr(self.settings, "enable_content_sanitization", True),
            min_content_length=getattr(self.settings, "min_content_length", 20),
        )

        coordinator = ResponseCoordinator(
            sanitizer=sanitizer,
            tool_adapter=tool_adapter,
            tool_registry=tool_registry,
            config=config,
        )

        logger.debug(
            f"ResponseCoordinator created with "
            f"sanitization={'enabled' if config.enable_content_sanitization else 'disabled'}, "
            f"max_garbage_chunks={config.max_garbage_chunks}"
        )

        return coordinator

    def create_tool_access_config_coordinator(
        self,
        tool_access_controller: Optional[Any] = None,
        mode_coordinator: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
    ) -> "ToolAccessConfigCoordinator":
        """Create tool access configuration coordinator.

        The ToolAccessConfigCoordinator provides unified interface for:
        - Building tool access context
        - Querying enabled tools
        - Checking individual tool access
        - Setting enabled tools with propagation

        Args:
            tool_access_controller: Optional ToolAccessController instance
            mode_coordinator: Optional ModeCoordinator instance
            tool_registry: Optional ToolRegistry instance

        Returns:
            ToolAccessConfigCoordinator instance configured with dependencies
        """
        from victor.agent.coordinators.config_coordinator import (
            ToolAccessConfigCoordinator,
        )

        coordinator = ToolAccessConfigCoordinator(
            tool_access_controller=tool_access_controller,
            mode_coordinator=mode_coordinator,
            tool_registry=tool_registry,
        )

        logger.debug("ToolAccessConfigCoordinator created")

        return coordinator

    def create_state_coordinator(
        self,
        session_state_manager: "SessionStateManager",
        conversation_state_machine: Optional["ConversationStateMachine"] = None,
        enable_history: bool = True,
        max_history_size: int = 100,
    ) -> "StateCoordinator":
        """Create state coordinator for unified state management.

        The StateCoordinator provides centralized state management for:
        - SessionStateManager: Execution state (tool calls, files, budget)
        - ConversationStateMachine: Conversation stage and flow
        - Checkpoint state: Serialization/deserialization
        - Observer pattern: State change notifications

        Args:
            session_state_manager: Manager for session execution state
            conversation_state_machine: Optional manager for conversation stage
            enable_history: Whether to track state change history
            max_history_size: Maximum number of state changes to track

        Returns:
            StateCoordinator instance configured with dependencies
        """
        from victor.agent.coordinators.state_coordinator import StateCoordinator

        coordinator = StateCoordinator(
            session_state_manager=session_state_manager,
            conversation_state_machine=conversation_state_machine,
            enable_history=enable_history,
            max_history_size=max_history_size,
        )

        logger.debug(
            f"StateCoordinator created with "
            f"history={'enabled' if enable_history else 'disabled'}"
        )

        return coordinator

    def create_coordinators(
        self,
        tool_adapter: Optional["BaseToolCallingAdapter"] = None,
        tool_registry: Optional[Any] = None,
        tool_access_controller: Optional[Any] = None,
        mode_coordinator: Optional[Any] = None,
        session_state_manager: Optional["SessionStateManager"] = None,
        conversation_state_machine: Optional["ConversationStateMachine"] = None,
    ) -> CoordinatorComponents:
        """Create all new coordinator components (Stream E4).

        This method creates all three new coordinators in a single call,
        providing a complete set of coordinator components for the orchestrator.

        Args:
            tool_adapter: Optional tool calling adapter for ResponseCoordinator
            tool_registry: Optional tool registry for coordinators
            tool_access_controller: Optional ToolAccessController
            mode_coordinator: Optional ModeCoordinator
            session_state_manager: Optional SessionStateManager for StateCoordinator
            conversation_state_machine: Optional ConversationStateMachine

        Returns:
            CoordinatorComponents with all three coordinators
        """
        response_coordinator = self.create_response_coordinator(
            tool_adapter=tool_adapter,
            tool_registry=tool_registry,
        )

        tool_access_config_coordinator = self.create_tool_access_config_coordinator(
            tool_access_controller=tool_access_controller,
            mode_coordinator=mode_coordinator,
            tool_registry=tool_registry,
        )

        state_coordinator = None
        if session_state_manager:
            state_coordinator = self.create_state_coordinator(
                session_state_manager=session_state_manager,
                conversation_state_machine=conversation_state_machine,
            )

        components = CoordinatorComponents(
            response_coordinator=response_coordinator,
            tool_access_config_coordinator=tool_access_config_coordinator,
            state_coordinator=state_coordinator,
        )

        logger.debug(
            f"CoordinatorComponents created: "
            f"response={'yes' if response_coordinator else 'no'}, "
            f"tool_access_config={'yes' if tool_access_config_coordinator else 'no'}, "
            f"state={'yes' if state_coordinator else 'no'}"
        )

        return components

    def create_core_services(
        self,
        tool_adapter: "BaseToolCallingAdapter",
        capabilities: "ToolCallingCapabilities",
    ) -> CoreServices:
        """Create all core service components.

        Args:
            tool_adapter: Tool calling adapter
            capabilities: Tool calling capabilities

        Returns:
            CoreServices containing all service components
        """
        return CoreServices(
            sanitizer=self.create_sanitizer(),
            prompt_builder=self.create_prompt_builder(tool_adapter, capabilities),
            project_context=self.create_project_context(),
            complexity_classifier=self.create_complexity_classifier(),
            action_authorizer=self.create_action_authorizer(),
            search_router=self.create_search_router(),
        )

    def create_streaming_metrics_collector(self) -> Optional[Any]:
        """Create streaming metrics collector if enabled."""
        from victor.analytics.streaming_metrics import StreamingMetricsCollector

        if not getattr(self.settings, "streaming_metrics_enabled", True):
            return None

        history_size = getattr(self.settings, "streaming_metrics_history_size", 1000)
        return StreamingMetricsCollector(max_history=history_size)

    def create_metrics_collector(
        self,
        streaming_metrics_collector: Optional[Any],
        usage_logger: Any,
        debug_logger: Any,
        tool_cost_lookup: Callable[[str], Any],
    ) -> "MetricsCollector":
        """Create metrics collector.

        Args:
            streaming_metrics_collector: Optional streaming metrics collector
            usage_logger: Usage logger instance
            debug_logger: Debug logger instance
            tool_cost_lookup: Function to lookup tool cost tier

        Returns:
            Configured MetricsCollector
        """
        from victor.agent.metrics_collector import MetricsCollector, MetricsCollectorConfig

        return MetricsCollector(
            config=MetricsCollectorConfig(
                model=self.model,
                provider=self.provider_name or self.provider.__class__.__name__,
                analytics_enabled=streaming_metrics_collector is not None,
            ),
            usage_logger=usage_logger,
            debug_logger=debug_logger,
            streaming_metrics_collector=streaming_metrics_collector,
            tool_cost_lookup=tool_cost_lookup,
        )

    def create_usage_analytics(self) -> "UsageAnalytics":
        """Create usage analytics (from DI container)."""
        from victor.agent.protocols import UsageAnalyticsProtocol

        return self.container.get(UsageAnalyticsProtocol)  # type: ignore[no-any-return]

    def create_sequence_tracker(self) -> "ToolSequenceTracker":
        """Create tool sequence tracker (from DI container)."""
        from victor.agent.protocols import ToolSequenceTrackerProtocol

        return self.container.get(ToolSequenceTrackerProtocol)  # type: ignore[no-any-return]

    def create_recovery_handler(self) -> Optional["RecoveryHandler"]:
        """Create recovery handler (from DI container)."""
        from victor.agent.protocols import RecoveryHandlerProtocol

        # RecoveryHandler is always registered, but may be disabled via settings
        return self.container.get(RecoveryHandlerProtocol)  # type: ignore[no-any-return]

    def create_observability(self) -> Optional["ObservabilityIntegration"]:
        """Create observability integration if enabled."""
        from victor.observability.integration import ObservabilityIntegration

        if not getattr(self.settings, "enable_observability", True):
            return None

        observability = ObservabilityIntegration()
        logger.debug("Observability integration created")
        return observability

    def create_tracers(self) -> tuple[Any, Any]:
        """Create execution and tool call tracers for debugging.

        Returns:
            Tuple of (ExecutionTracer, ToolCallTracer) or (None, None) if disabled
        """
        if not getattr(self.settings, "enable_tracing", True):
            self._execution_tracer = None
            self._tool_call_tracer = None
            return (None, None)

        from victor.core.events import get_observability_bus
        from victor.observability.tracing import ExecutionTracer, ToolCallTracer

        # Get ObservabilityBus from DI container
        observability_bus = get_observability_bus()
        execution_tracer = ExecutionTracer(observability_bus)
        tool_call_tracer = ToolCallTracer(observability_bus)

        # Store for later use in create_tool_executor
        self._execution_tracer = execution_tracer
        self._tool_call_tracer = tool_call_tracer

        logger.debug("ExecutionTracer and ToolCallTracer created")
        return (execution_tracer, tool_call_tracer)

    def create_tool_cache(self) -> Optional[Any]:
        """Create tool cache if enabled.

        Returns:
            ToolCache instance or None if disabled
        """
        if not getattr(self.settings, "tool_cache_enabled", True):
            return None

        from victor.storage.cache.config import CacheConfig
        from victor.config.settings import get_project_paths
        from victor.storage.cache.tool_cache import ToolCache

        # Allow explicit override of cache_dir, otherwise use centralized path
        cache_dir = getattr(self.settings, "tool_cache_dir", None)
        if cache_dir:
            cache_dir = Path(cache_dir).expanduser()
        else:
            cache_dir = get_project_paths().global_cache_dir

        cache = ToolCache(
            ttl=getattr(self.settings, "tool_cache_ttl", 600),
            allowlist=getattr(self.settings, "tool_cache_allowlist", []),
            cache_config=CacheConfig(disk_path=cache_dir),
        )
        logger.debug(f"ToolCache created with TTL={cache.ttl}s")
        return cache

    def create_reminder_manager(self, provider: str, task_complexity: str, tool_budget: int) -> Any:
        """Create context reminder manager from DI container.

        Args:
            provider: Provider name
            task_complexity: Task complexity level
            tool_budget: Tool call budget

        Returns:
            ReminderManager instance
        """
        from victor.agent.protocols import ReminderManagerProtocol

        # Resolve from DI container (scoped service)
        with self.container.create_scope() as scope:
            reminder_manager = scope.get(ReminderManagerProtocol)

        logger.debug(f"ReminderManager created for {provider} with complexity {task_complexity}")
        return reminder_manager

    def create_tool_dependency_graph(self) -> Any:
        """Create tool dependency graph from DI container.

        Returns:
            ToolDependencyGraph instance
        """
        from victor.agent.protocols import ToolDependencyGraphProtocol

        # Resolve from DI container
        tool_graph = self.container.get(ToolDependencyGraphProtocol)

        logger.debug("ToolDependencyGraph created")
        return tool_graph

    def create_memory_components(
        self, provider_name: str, tool_capable: bool = True
    ) -> tuple[Optional[Any], Optional[str]]:
        """Create conversation memory components.

        Args:
            provider_name: Provider name for session metadata
            tool_capable: Whether the model supports tool calling

        Returns:
            Tuple of (ConversationStore or None, session_id or None)
        """
        if not getattr(self.settings, "conversation_memory_enabled", True):
            return None, None

        try:
            from victor.config.settings import get_project_paths
            from victor.agent.conversation_memory import ConversationStore

            paths = get_project_paths()
            # Ensure .victor directory exists
            paths.project_victor_dir.mkdir(parents=True, exist_ok=True)
            db_path = paths.conversation_db
            max_context = getattr(self.settings, "max_context_tokens", 100000)
            response_reserve = getattr(self.settings, "response_token_reserve", 4096)

            memory_manager = ConversationStore(
                db_path=db_path,
                max_context_tokens=max_context,
                response_reserve=response_reserve,
            )

            # Create a session for this orchestrator instance
            project_path = str(paths.project_root)
            session = memory_manager.create_session(
                project_path=project_path,
                provider=provider_name,
                model=self.model,
                max_tokens=max_context,
                profile=self.profile_name,
                tool_capable=tool_capable,
            )
            session_id = session.session_id
            logger.info(
                f"ConversationStore initialized via factory. "
                f"Session: {session_id[:8]}..., DB: {db_path}"
            )
            return memory_manager, session_id

        except Exception as e:
            logger.warning(f"Failed to initialize ConversationStore: {e}")
            return None, None

    def create_usage_logger(self) -> Any:
        """Create usage logger (DI with fallback).

        Returns:
            UsageLogger instance
        """
        from victor.core.bootstrap import UsageLoggerProtocol
        from victor.analytics.logger import UsageLogger
        from victor.config.settings import get_project_paths

        usage_logger = self.container.get_optional(UsageLoggerProtocol)
        if usage_logger is not None:
            logger.debug("Using enhanced usage logger from DI container")
            return usage_logger

        # Fallback to basic usage logger
        analytics_log_file = get_project_paths().global_logs_dir / "usage.jsonl"
        usage_logger = UsageLogger(
            analytics_log_file, enabled=getattr(self.settings, "analytics_enabled", True)
        )
        logger.debug("Using basic usage logger (enhanced version not available)")
        return usage_logger

    def create_middleware_chain(self) -> tuple[Optional[Any], Optional[Any]]:
        """Create middleware chain with vertical extensions.

        Returns:
            Tuple of (MiddlewareChain or None, CodeCorrectionMiddleware or None)
        """
        middleware_chain: Optional[Any] = None
        code_correction_middleware: Optional[Any] = None
        code_correction_enabled = getattr(self.settings, "code_correction_enabled", True)

        try:
            from victor.agent.middleware_chain import MiddlewareChain
            from victor.core.verticals.protocols import VerticalExtensions

            middleware_chain = MiddlewareChain()

            # Get vertical extensions from DI container if available
            extensions = self.container.get_optional(VerticalExtensions)
            if extensions and extensions.middleware:
                for middleware in extensions.middleware:
                    middleware_chain.add(middleware)
                    logger.debug(f"Added middleware from vertical: {type(middleware).__name__}")

                # For backward compatibility, find CodeCorrectionMiddleware
                for mw in extensions.middleware:
                    if "CodeCorrection" in type(mw).__name__:
                        code_correction_middleware = mw
                        break
            else:
                # Fallback: Load default code correction middleware for backward compatibility
                if code_correction_enabled:
                    try:
                        from victor.agent.code_correction_middleware import (
                            CodeCorrectionMiddleware,
                            CodeCorrectionConfig,
                        )

                        code_correction_auto_fix = getattr(
                            self.settings, "code_correction_auto_fix", True
                        )
                        code_correction_max_iterations = getattr(
                            self.settings, "code_correction_max_iterations", 1
                        )

                        code_correction_middleware = CodeCorrectionMiddleware(
                            config=CodeCorrectionConfig(
                                enabled=True,
                                auto_fix=code_correction_auto_fix,
                                max_iterations=code_correction_max_iterations,
                            )
                        )
                        logger.debug("Using fallback CodeCorrectionMiddleware (no vertical loaded)")
                    except ImportError as e:
                        logger.warning(f"CodeCorrectionMiddleware unavailable: {e}")
        except ImportError as e:
            logger.warning(f"Middleware chain unavailable: {e}")

        return middleware_chain, code_correction_middleware

    def create_safety_checker(self) -> Any:
        """Create safety checker with vertical safety patterns.

        Returns:
            SafetyChecker instance with registered patterns
        """
        from victor.agent.protocols import SafetyCheckerProtocol

        # Resolve from DI container
        safety_checker = self.container.get(SafetyCheckerProtocol)

        # Register vertical safety patterns with the safety checker
        try:
            from victor.core.verticals.protocols import VerticalExtensions

            extensions = self.container.get_optional(VerticalExtensions)
            if extensions and extensions.safety_extensions:
                for safety_ext in extensions.safety_extensions:
                    # Add all bash patterns from the extension
                    for pattern in safety_ext.get_bash_patterns():
                        safety_checker.add_custom_pattern(
                            pattern.pattern,
                            pattern.description,
                            pattern.risk_level,
                            pattern.category,
                        )
                    logger.debug(
                        f"Added safety patterns from vertical: {safety_ext.get_category()}"
                    )
        except Exception as e:
            logger.debug(f"Could not load vertical safety extensions: {e}")

        return safety_checker

    def create_auto_committer(self) -> Optional[Any]:
        """Create auto committer for AI-assisted commits.

        Returns:
            AutoCommitter instance or None
        """
        from victor.agent.protocols import AutoCommitterProtocol
        from victor.agent.auto_commit import get_auto_committer

        auto_commit_enabled = getattr(self.settings, "auto_commit_enabled", False)

        if auto_commit_enabled:
            # Resolve from DI container
            auto_committer = self.container.get(AutoCommitterProtocol)
            logger.debug("AutoCommitter enabled for AI-assisted commits")
            return auto_committer
        else:
            # Still create instance for manual use, just not auto-commit
            return get_auto_committer()

    def create_parallel_executor(self, tool_executor: Any) -> Any:
        """Create parallel tool executor for concurrent independent tool calls.

        Args:
            tool_executor: The ToolExecutor instance to wrap

        Returns:
            ParallelExecutor instance wrapping the tool executor
        """
        from victor.agent.parallel_executor import create_parallel_executor

        parallel_enabled = getattr(self.settings, "parallel_tool_execution", True)
        max_concurrent = getattr(self.settings, "max_concurrent_tools", 5)

        parallel_executor = create_parallel_executor(
            tool_executor=tool_executor,
            max_concurrent=max_concurrent,
            enable=parallel_enabled,
        )
        logger.debug(
            f"ParallelExecutor created (enabled={parallel_enabled}, max_concurrent={max_concurrent})"
        )
        return parallel_executor

    def create_response_completer(self) -> Any:
        """Create response completer for ensuring complete responses after tool calls.

        Returns:
            ResponseCompleter instance
        """
        from victor.agent.response_completer import create_response_completer

        response_completer = create_response_completer(
            provider=self.provider,
            max_retries=getattr(self.settings, "response_completion_retries", 3),
            force_response=getattr(self.settings, "force_response_on_error", True),
        )
        logger.debug("ResponseCompleter created")
        return response_completer

    def create_unified_tracker(self, tool_calling_caps: "ToolCallingCapabilities") -> Any:
        """Create unified task tracker with model-specific exploration settings.

        The UnifiedTaskTracker is the single source of truth for task progress,
        milestones, and loop detection across the orchestration lifecycle.

        Args:
            tool_calling_caps: Tool calling capabilities for model-specific settings

        Returns:
            UnifiedTaskTracker instance configured with model exploration parameters
        """
        from victor.agent.protocols import TaskTrackerProtocol, ModeControllerProtocol
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        # Resolve from DI container (might be scoped)
        unified_tracker = self.container.get_optional(TaskTrackerProtocol)
        if unified_tracker is None:
            # Container not bootstrapped or not in scope, create directly
            unified_tracker = UnifiedTaskTracker()

        # Apply model-specific exploration settings
        unified_tracker.set_model_exploration_settings(
            exploration_multiplier=tool_calling_caps.exploration_multiplier,
            continuation_patience=tool_calling_caps.continuation_patience,
        )

        # Apply agent mode exploration multiplier (plan/explore modes get more iterations)
        try:
            mode_controller = self.container.get_optional(ModeControllerProtocol)
            if mode_controller:
                from victor.agent.mode_controller import MODE_CONFIGS

                current_mode = mode_controller.current_mode
                mode_config = MODE_CONFIGS.get(current_mode)
                if mode_config and hasattr(mode_config, "exploration_multiplier"):
                    unified_tracker.set_mode_exploration_multiplier(
                        mode_config.exploration_multiplier
                    )
                    logger.info(
                        f"Applied mode exploration multiplier: mode={current_mode.value}, "
                        f"multiplier={mode_config.exploration_multiplier}"
                    )
        except Exception as e:
            logger.debug(f"Could not apply mode exploration multiplier: {e}")

        logger.debug(
            f"UnifiedTaskTracker created with exploration_multiplier="
            f"{tool_calling_caps.exploration_multiplier}, "
            f"continuation_patience={tool_calling_caps.continuation_patience}"
        )

        return unified_tracker

    def create_tool_output_formatter(self, context_compactor: Any) -> Any:
        """Create tool output formatter for LLM-context-aware output formatting.

        Args:
            context_compactor: ContextCompactor instance for smart truncation

        Returns:
            ToolOutputFormatter instance configured with settings
        """
        from victor.agent.tool_output_formatter import (
            create_tool_output_formatter,
            ToolOutputFormatterConfig,
        )

        formatter = create_tool_output_formatter(
            config=ToolOutputFormatterConfig(
                max_output_chars=getattr(self.settings, "max_tool_output_chars", 15000),
                file_structure_threshold=getattr(self.settings, "file_structure_threshold", 50000),
            ),
            truncator=context_compactor,
        )
        logger.debug("ToolOutputFormatter created")
        return formatter

    def create_recovery_integration(self, recovery_handler: Optional[Any]) -> Any:
        """Create recovery integration submodule for clean delegation.

        Args:
            recovery_handler: RecoveryHandler instance (may be None if disabled)

        Returns:
            RecoveryIntegration instance
        """
        from victor.agent.orchestrator_recovery import create_recovery_integration

        integration = create_recovery_integration(
            recovery_handler=recovery_handler,
            settings=self.settings,
        )
        logger.debug("RecoveryIntegration created")
        return integration

    def create_recovery_coordinator(self) -> Any:
        """Create RecoveryCoordinator via DI container.

        The RecoveryCoordinator centralizes all recovery and error handling logic
        for streaming chat sessions. It's resolved from the DI container to enable
        proper dependency management and testability.

        Returns:
            StreamingRecoveryCoordinator instance for recovery coordination
        """
        from victor.agent.protocols import StreamingRecoveryCoordinatorProtocol

        recovery_coordinator = self.container.get(StreamingRecoveryCoordinatorProtocol)
        logger.debug("StreamingRecoveryCoordinator created via DI")
        return recovery_coordinator

    def create_chunk_generator(self) -> Any:
        """Create ChunkGenerator via DI container.

        The ChunkGenerator provides a centralized interface for generating streaming
        chunks for various purposes (tool execution, status updates, metrics, content).
        It's resolved from the DI container to enable proper dependency management
        and testability.

        Returns:
            ChunkGenerator instance for chunk generation
        """
        from victor.agent.protocols import ChunkGeneratorProtocol

        chunk_generator = self.container.get(ChunkGeneratorProtocol)
        logger.debug("ChunkGenerator created via DI")
        return chunk_generator

    def create_tool_planner(self) -> Any:
        """Create ToolPlanner via DI container.

        The ToolPlanner provides a centralized interface for tool planning operations,
        including goal inference, tool sequence planning, and intent-based filtering.
        It's resolved from the DI container to enable proper dependency management
        and testability.

        Returns:
            ToolPlanner instance for tool planning
        """
        from victor.agent.protocols import ToolPlannerProtocol

        tool_planner = self.container.get(ToolPlannerProtocol)
        logger.debug("ToolPlanner created via DI")
        return tool_planner

    def create_task_coordinator(self) -> Any:
        """Create TaskCoordinator via DI container.

        The TaskCoordinator provides a centralized interface for task coordination,
        including task preparation, intent detection, and task-specific guidance.
        It's resolved from the DI container to enable proper dependency management
        and testability.

        Returns:
            TaskCoordinator instance for task coordination
        """
        from victor.agent.protocols import TaskCoordinatorProtocol

        task_coordinator = self.container.get(TaskCoordinatorProtocol)
        logger.debug("TaskCoordinator created via DI")
        return task_coordinator

    def create_streaming_controller(
        self,
        streaming_metrics_collector: Optional[Any],
        on_session_complete: Callable[..., Any],
    ) -> Any:
        """Create streaming controller for managing streaming sessions and metrics.

        Args:
            streaming_metrics_collector: Optional metrics collector for session tracking
            on_session_complete: Callback invoked when streaming session completes

        Returns:
            StreamingController instance configured for session management
        """
        from victor.agent.streaming_controller import (
            StreamingController,
            StreamingControllerConfig,
        )

        controller = StreamingController(
            config=StreamingControllerConfig(
                max_history=100,
                enable_metrics_collection=streaming_metrics_collector is not None,
            ),
            metrics_collector=streaming_metrics_collector,
            on_session_complete=on_session_complete,
        )
        logger.debug("StreamingController created")
        return controller

    def create_streaming_coordinator(
        self,
        streaming_controller: Any,
    ) -> Any:
        """Create streaming coordinator for response processing.

        Args:
            streaming_controller: StreamingController for session management

        Returns:
            StreamingCoordinator instance for processing streaming responses
        """
        from victor.agent.streaming.streaming_coordinator import StreamingCoordinator

        coordinator = StreamingCoordinator(
            streaming_controller=streaming_controller,
        )
        logger.debug("StreamingCoordinator created")
        return coordinator

    def create_provider_switch_coordinator(
        self,
        provider_switcher: Any,
        health_monitor: Optional[Any] = None,
    ) -> Any:
        """Create provider switch coordinator for switching workflow.

        Args:
            provider_switcher: ProviderSwitcher for switching logic
            health_monitor: Optional ProviderHealthMonitor for pre-switch checks

        Returns:
            ProviderSwitchCoordinator instance for coordinating switches
        """
        from victor.agent.provider_switch_coordinator import ProviderSwitchCoordinator

        coordinator = ProviderSwitchCoordinator(
            provider_switcher=provider_switcher,
            health_monitor=health_monitor,
        )
        logger.debug("ProviderSwitchCoordinator created")
        return coordinator

    def create_lifecycle_manager(
        self,
        conversation_controller: Any,
        metrics_collector: Optional[Any] = None,
        context_compactor: Optional[Any] = None,
        sequence_tracker: Optional[Any] = None,
        usage_analytics: Optional[Any] = None,
        reminder_manager: Optional[Any] = None,
    ) -> Any:
        """Create lifecycle manager for session lifecycle and resource cleanup.

        Args:
            conversation_controller: Controller for conversation management
            metrics_collector: Optional metrics collector for stats
            context_compactor: Optional context compactor for cleanup
            sequence_tracker: Optional sequence tracker for pattern learning
            usage_analytics: Optional usage analytics for session tracking
            reminder_manager: Optional reminder manager for context reminders

        Returns:
            LifecycleManager instance for managing lifecycle operations
        """
        from victor.agent.lifecycle_manager import LifecycleManager

        lifecycle_manager = LifecycleManager(
            conversation_controller=conversation_controller,
            metrics_collector=metrics_collector,
            context_compactor=context_compactor,
            sequence_tracker=sequence_tracker,
            usage_analytics=usage_analytics,
            reminder_manager=reminder_manager,
        )
        logger.debug("LifecycleManager created")
        return lifecycle_manager

    def create_tool_pipeline(
        self,
        tools: Any,
        tool_executor: Any,
        tool_budget: int,
        tool_cache: Optional[Any],
        argument_normalizer: Any,
        on_tool_start: Callable[..., Any],
        on_tool_complete: Callable[..., Any],
        deduplication_tracker: Optional[Any],
        middleware_chain: Optional[Any] = None,
        semantic_cache: Optional[Any] = None,
    ) -> Any:
        """Create tool pipeline for coordinating tool execution flow.

        Args:
            tools: Tool registry containing available tools
            tool_executor: ToolExecutor for executing tool calls
            tool_budget: Maximum tool calls allowed per session
            tool_cache: Optional cache for tool results
            argument_normalizer: Normalizer for tool arguments
            on_tool_start: Callback invoked when tool execution starts
            on_tool_complete: Callback invoked when tool execution completes
            deduplication_tracker: Optional tracker for preventing duplicate calls
            middleware_chain: Optional middleware chain for processing tool calls
            semantic_cache: Optional FAISS-based semantic cache for tool results

        Returns:
            ToolPipeline instance coordinating tool execution
        """
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig

        pipeline = ToolPipeline(
            tool_registry=tools,
            tool_executor=tool_executor,
            config=ToolPipelineConfig(
                tool_budget=tool_budget,
                enable_caching=tool_cache is not None,
                enable_analytics=True,
                enable_failed_signature_tracking=True,
                enable_semantic_caching=semantic_cache is not None,
            ),
            tool_cache=tool_cache,
            argument_normalizer=argument_normalizer,
            on_tool_start=on_tool_start,
            on_tool_complete=on_tool_complete,
            deduplication_tracker=deduplication_tracker,
            middleware_chain=middleware_chain,
            semantic_cache=semantic_cache,
        )
        logger.debug(
            "ToolPipeline created%s%s",
            " with middleware chain" if middleware_chain else "",
            " with semantic cache" if semantic_cache else "",
        )
        return pipeline

    def create_conversation_controller(
        self,
        provider: "BaseProvider",
        model: str,
        conversation: Any,
        conversation_state: Any,
        memory_manager: Any,
        memory_session_id: str,
        system_prompt: str,
    ) -> Any:
        """Create conversation controller for managing message history and state.

        Args:
            provider: LLM provider instance for context calculations
            model: Model identifier for context limits
            conversation: Message history list
            conversation_state: ConversationStateMachine instance
            memory_manager: Memory manager for persistent storage
            memory_session_id: Unique session identifier
            system_prompt: System prompt to set on the controller

        Returns:
            ConversationController instance configured with model-aware settings
        """
        from victor.agent.conversation_controller import (
            ConversationController,
            ConversationConfig,
            CompactionStrategy,
        )
        from victor.agent.orchestrator_utils import calculate_max_context_chars

        # Calculate model-aware context limit
        model_context_chars = calculate_max_context_chars(self.settings, provider, model)

        # Parse compaction strategy from settings
        compaction_strategy_str = getattr(
            self.settings, "context_compaction_strategy", "tiered"
        ).lower()
        compaction_strategy_map = {
            "simple": CompactionStrategy.SIMPLE,
            "tiered": CompactionStrategy.TIERED,
            "semantic": CompactionStrategy.SEMANTIC,
            "hybrid": CompactionStrategy.HYBRID,
        }
        compaction_strategy = compaction_strategy_map.get(
            compaction_strategy_str, CompactionStrategy.TIERED
        )

        # Create controller with model-aware configuration
        controller = ConversationController(
            config=ConversationConfig(
                max_context_chars=model_context_chars,
                enable_stage_tracking=True,
                enable_context_monitoring=True,
                # Smart compaction settings from Settings
                compaction_strategy=compaction_strategy,
                min_messages_to_keep=getattr(self.settings, "context_min_messages_to_keep", 6),
                tool_result_retention_weight=getattr(
                    self.settings, "context_tool_retention_weight", 1.5
                ),
                recent_message_weight=getattr(self.settings, "context_recency_weight", 2.0),
                semantic_relevance_threshold=getattr(
                    self.settings, "context_semantic_threshold", 0.3
                ),
            ),
            message_history=conversation,
            state_machine=conversation_state,
            # Pass SQLite store for persistent semantic memory
            conversation_store=memory_manager,
            session_id=memory_session_id,
        )
        controller.set_system_prompt(system_prompt)

        logger.debug(
            f"ConversationController created with max_context_chars={model_context_chars}, "
            f"compaction_strategy={compaction_strategy}"
        )
        return controller

    def create_tool_deduplication_tracker(self) -> Optional[Any]:
        """Create tool deduplication tracker for preventing redundant calls.

        Returns None if deduplication is disabled in settings, otherwise returns
        configured ToolDeduplicationTracker instance.

        Returns:
            ToolDeduplicationTracker instance if enabled, None otherwise
        """
        # Check if deduplication is enabled
        if not getattr(self.settings, "enable_tool_deduplication", False):
            logger.debug("Tool deduplication disabled")
            return None

        try:
            from victor.agent.tool_deduplication import ToolDeduplicationTracker

            window_size = getattr(self.settings, "tool_deduplication_window_size", 10)
            tracker = ToolDeduplicationTracker(window_size=window_size)
            logger.info(f"ToolDeduplicationTracker initialized (window: {window_size})")
            return tracker
        except Exception as e:
            logger.warning(f"Failed to initialize ToolDeduplicationTracker: {e}")
            return None

    def create_streaming_chat_handler(self, message_adder: Any) -> Any:
        """Create streaming chat handler for testable streaming loop logic.

        This component encapsulates limit checking, response processing, and
        iteration control for streaming conversations.

        Args:
            message_adder: Object implementing add_message() interface (typically orchestrator)

        Returns:
            StreamingChatHandler instance configured with settings
        """
        from victor.agent.streaming import StreamingChatHandler

        session_idle_timeout = getattr(self.settings, "session_idle_timeout", 180.0)
        # Get presentation adapter from message_adder (orchestrator) if available
        presentation = getattr(message_adder, "_presentation", None)
        handler = StreamingChatHandler(
            settings=self.settings,
            message_adder=message_adder,
            session_idle_timeout=session_idle_timeout,
            presentation=presentation,
        )
        logger.debug(f"StreamingChatHandler created (idle_timeout={session_idle_timeout})")
        return handler

    def create_rl_coordinator(self) -> Optional[Any]:
        """Create RL coordinator for reinforcement learning framework.

        The RL coordinator manages all learners (continuation_prompts,
        continuation_patience, model_selector, semantic_threshold) with
        unified SQLite storage.

        Returns None if RL learning is disabled in settings, otherwise returns
        configured RL coordinator instance.

        Returns:
            RL coordinator instance if enabled, None otherwise
        """
        # Check if RL learning is enabled
        if not getattr(self.settings, "enable_continuation_rl_learning", False):
            logger.debug("RL learning disabled")
            return None

        try:
            from victor.agent.protocols import RLCoordinatorProtocol

            # Resolve from DI container
            coordinator = self.container.get(RLCoordinatorProtocol)
            logger.info("RL: Coordinator initialized with unified database")
            return coordinator
        except Exception as e:
            logger.warning(f"RL: Failed to initialize RL coordinator: {e}")
            return None

    def create_context_compactor(
        self,
        conversation_controller: Any,
        pruning_learner: Optional[Any] = None,
    ) -> Any:
        """Create context compactor for proactive context management.

        This component provides:
        - Proactive compaction before context overflow (triggers at 70% utilization)
        - Smart tool result truncation with content-aware strategies
        - Token estimation with content-type-specific factors
        - RL-based adaptive pruning (when learner is provided)

        Args:
            conversation_controller: ConversationController instance for context tracking
            pruning_learner: Optional RL learner for adaptive pruning decisions

        Returns:
            ContextCompactor instance configured with settings
        """
        from victor.agent.context_compactor import (
            create_context_compactor,
            TruncationStrategy,
        )

        # Parse truncation strategy from settings
        truncation_strategy_str = getattr(
            self.settings, "tool_truncation_strategy", "smart"
        ).lower()
        truncation_strategy_map = {
            "head": TruncationStrategy.HEAD,
            "tail": TruncationStrategy.TAIL,
            "both": TruncationStrategy.BOTH,
            "smart": TruncationStrategy.SMART,
        }
        truncation_strategy = truncation_strategy_map.get(
            truncation_strategy_str, TruncationStrategy.SMART
        )

        # Determine provider type for RL state
        provider_name = getattr(self.settings, "provider", "").lower()
        local_providers = {"ollama", "lmstudio", "vllm", "llamacpp", "local"}
        provider_type = "local" if any(p in provider_name for p in local_providers) else "cloud"

        # Create compactor with all settings
        compactor = create_context_compactor(
            controller=conversation_controller,
            proactive_threshold=getattr(self.settings, "context_proactive_threshold", 0.90),
            min_messages_after_compact=getattr(
                self.settings, "context_min_messages_after_compact", 8
            ),
            tool_result_max_chars=getattr(self.settings, "max_tool_output_chars", 8192),
            tool_result_max_lines=getattr(self.settings, "max_tool_output_lines", 200),
            truncation_strategy=truncation_strategy,
            preserve_code_blocks=True,
            enable_proactive=getattr(self.settings, "context_proactive_compaction", True),
            enable_tool_truncation=getattr(self.settings, "tool_result_truncation", True),
            pruning_learner=pruning_learner,
            provider_type=provider_type,
        )

        rl_status = "with RL learner" if pruning_learner else "without RL learner"
        logger.debug(
            f"ContextCompactor created {rl_status}, "
            f"truncation_strategy={truncation_strategy}, provider_type={provider_type}"
        )
        return compactor

    def create_argument_normalizer(self, provider: "BaseProvider") -> Any:
        """Create argument normalizer for handling malformed tool arguments.

        Uses DI-first resolution with fallback to direct instantiation.

        Args:
            provider: Provider instance for extracting provider name

        Returns:
            ArgumentNormalizer instance configured with provider name
        """
        from victor.agent.protocols import ArgumentNormalizerProtocol

        # Resolve from DI container (guaranteed to be registered)
        return self.container.get(ArgumentNormalizerProtocol)

    def create_tool_executor(
        self,
        tools: Any,
        argument_normalizer: Any,
        tool_cache: Optional[Any],
        safety_checker: Any,
        code_correction_middleware: Optional[Any],
    ) -> Any:
        """Create tool executor for centralized tool execution.

        Handles retry logic, caching, validation, and metrics collection.

        Args:
            tools: Tool registry containing available tools
            argument_normalizer: ArgumentNormalizer for handling malformed arguments
            tool_cache: Optional cache for tool results
            safety_checker: SafetyChecker for validating tool execution
            code_correction_middleware: Optional middleware for code correction

        Returns:
            ToolExecutor instance configured with settings
        """
        from victor.agent.tool_executor import ToolExecutor, ValidationMode

        # Parse validation mode from settings
        validation_mode_str = getattr(self.settings, "tool_validation_mode", "lenient").lower()
        validation_mode_map = {
            "strict": ValidationMode.STRICT,
            "lenient": ValidationMode.LENIENT,
            "off": ValidationMode.OFF,
        }
        validation_mode = validation_mode_map.get(validation_mode_str, ValidationMode.LENIENT)

        # Create executor with all settings
        executor = ToolExecutor(
            tool_registry=tools,
            argument_normalizer=argument_normalizer,
            tool_cache=tool_cache,
            max_retries=getattr(self.settings, "tool_retry_max_attempts", 3),
            retry_delay=getattr(self.settings, "tool_retry_base_delay", 1.0),
            validation_mode=validation_mode,
            safety_checker=safety_checker,
            code_correction_middleware=code_correction_middleware,
            enable_code_correction=getattr(self.settings, "code_correction_enabled", True),
            tool_call_tracer=getattr(self, "_tool_call_tracer", None),  # Pass tracer for debugging
        )

        # Inject ToolConfig into executor context for DI-style tool configuration
        # Tools can access this via ToolConfig.from_context(context) instead of globals
        from victor.tools.base import ToolConfig

        tool_config = ToolConfig(
            provider=self.provider,
            model=getattr(self.settings, "model", None),
            max_complexity=getattr(self.settings, "max_complexity", 10),
            web_fetch_top=getattr(self.settings, "web_fetch_top", 5),
            web_fetch_pool=getattr(self.settings, "web_fetch_pool", 3),
            max_content_length=getattr(self.settings, "max_content_length", 5000),
            batch_concurrency=getattr(self.settings, "batch_concurrency", 5),
            batch_max_files=getattr(self.settings, "batch_max_files", 100),
        )
        executor.update_context(tool_config=tool_config)

        logger.debug(f"ToolExecutor created with validation_mode={validation_mode}")
        return executor

    def create_code_execution_manager(self) -> Any:
        """Create code execution manager for Docker-based code execution.

        Uses DI-first resolution with fallback to direct instantiation.
        Automatically starts the manager after creation.

        Returns:
            CodeExecutionManager instance (started)
        """
        from victor.agent.protocols import CodeExecutionManagerProtocol

        # Resolve from DI container (guaranteed to be registered and started)
        return self.container.get(CodeExecutionManagerProtocol)

    def create_workflow_registry(self) -> Any:
        """Create workflow registry for managing workflow patterns.

        Uses DI-first resolution with fallback to direct instantiation.

        Returns:
            WorkflowRegistry instance
        """
        from victor.agent.protocols import WorkflowRegistryProtocol

        # Resolve from DI container (guaranteed to be registered)
        return self.container.get(WorkflowRegistryProtocol)

    def create_conversation_state_machine(self) -> Any:
        """Create conversation state machine for intelligent stage detection.

        Uses DI-first resolution with fallback to direct instantiation.

        Returns:
            ConversationStateMachine instance
        """
        from victor.agent.protocols import ConversationStateMachineProtocol
        from victor.agent.conversation_state import ConversationStateMachine

        # Try to resolve from DI container (might be scoped)
        # Use get_optional since we're not in a scope yet
        state_machine = self.container.get_optional(ConversationStateMachineProtocol)
        if state_machine is None:
            # Container not bootstrapped or not in scope, create directly
            state_machine = ConversationStateMachine()
        return state_machine

    def create_integration_config(self) -> Any:
        """Create intelligent pipeline integration configuration.

        Extracts all intelligent_* settings from Settings and constructs
        an IntegrationConfig for the intelligent pipeline integration layer.

        Returns:
            IntegrationConfig instance with intelligent pipeline settings
        """
        from victor.agent.orchestrator_integration import IntegrationConfig

        config = IntegrationConfig(
            enable_resilient_calls=getattr(self.settings, "intelligent_pipeline_enabled", True),
            enable_quality_scoring=getattr(self.settings, "intelligent_quality_scoring", True),
            enable_mode_learning=getattr(self.settings, "intelligent_mode_learning", True),
            enable_prompt_optimization=getattr(
                self.settings, "intelligent_prompt_optimization", True
            ),
            min_quality_threshold=getattr(self.settings, "intelligent_min_quality_threshold", 0.5),
            grounding_confidence_threshold=getattr(
                self.settings, "intelligent_grounding_threshold", 0.7
            ),
        )

        logger.debug(
            f"IntegrationConfig created with resilient_calls={config.enable_resilient_calls}, "
            f"quality_scoring={config.enable_quality_scoring}"
        )
        return config

    def create_tool_registrar(
        self, tools: Any, tool_graph: Any, provider: "BaseProvider", model: str
    ) -> Any:
        """Create tool registrar for dynamic tool discovery and registration.

        Encapsulates dynamic tool discovery, plugin loading, and MCP integration.

        Args:
            tools: ToolRegistry instance for tool storage
            tool_graph: ToolGraph instance for dependency tracking
            provider: LLM provider instance
            model: Model identifier

        Returns:
            ToolRegistrar instance configured with settings
        """
        from victor.agent.tool_registrar import ToolRegistrar, ToolRegistrarConfig

        registrar = ToolRegistrar(
            tools=tools,
            settings=self.settings,
            provider=provider,
            model=model,
            tool_graph=tool_graph,
            config=ToolRegistrarConfig(
                enable_plugins=getattr(self.settings, "plugin_enabled", True),
                enable_mcp=getattr(self.settings, "use_mcp_tools", False),
                enable_tool_graph=True,
                airgapped_mode=getattr(self.settings, "airgapped_mode", False),
                plugin_dirs=getattr(self.settings, "plugin_dirs", []),
                disabled_plugins=set(getattr(self.settings, "disabled_plugins", [])),
                plugin_packages=getattr(self.settings, "plugin_packages", []),
                max_workers=4,
                max_complexity=10,
            ),
        )

        logger.debug(
            f"ToolRegistrar created with plugins={registrar.config.enable_plugins}, "
            f"mcp={registrar.config.enable_mcp}"
        )
        return registrar

    def create_message_history(self, system_prompt: str) -> Any:
        """Create message history for conversation tracking.

        Args:
            system_prompt: System prompt to initialize the conversation with

        Returns:
            MessageHistory instance configured with settings
        """
        from victor.agent.message_history import MessageHistory

        max_history = getattr(self.settings, "max_conversation_history", 100000)
        history = MessageHistory(
            system_prompt=system_prompt,
            max_history_messages=max_history,
        )

        logger.debug(f"MessageHistory created with max_history={max_history}")
        return history

    def create_tool_registry(self) -> Any:
        """Create tool registry for managing available tools.

        Returns:
            ToolRegistry instance for tool storage and management
        """
        from victor.tools.registry import ToolRegistry

        registry = ToolRegistry()
        logger.debug("ToolRegistry created")
        return registry

    def initialize_plugin_system(self, tool_registrar: Any) -> Optional[Any]:
        """Initialize plugin system for extensible tools.

        Delegates to ToolRegistrar for plugin discovery and loading.

        Args:
            tool_registrar: ToolRegistrar instance for plugin management

        Returns:
            ToolPluginRegistry instance if plugins enabled, None otherwise
        """
        # Check if plugins are enabled
        if not getattr(self.settings, "plugin_enabled", True):
            logger.debug("Plugin system disabled")
            return None

        # Initialize plugins via ToolRegistrar
        tool_count = tool_registrar._initialize_plugins()
        plugin_manager = tool_registrar.plugin_manager

        if tool_count > 0:
            logger.info(f"Plugins initialized via ToolRegistrar: {tool_count} tools")
        else:
            logger.debug("No plugins loaded")

        return plugin_manager

    def create_tool_selector(
        self,
        tools: Any,
        conversation_state: Any,
        unified_tracker: Any,
        model: str,
        provider_name: str,
        tool_selection: Any,
        on_selection_recorded: Any,
    ) -> Any:
        """Create unified tool selector using the new strategy factory.

        This method uses the unified tool selection strategy factory from
        victor.agent.tool_selector_factory, which supports:
        - auto: Automatic selection based on environment (default)
        - keyword: Fast metadata-based selection (<1ms)
        - semantic: ML-based embedding similarity (~50ms)
        - hybrid: Blends semantic + keyword for best results (~30ms)

        Args:
            tools: ToolRegistry instance
            conversation_state: ConversationStateMachine instance
            unified_tracker: UnifiedTaskTracker instance
            model: Model identifier
            provider_name: Provider name
            tool_selection: Tool selection configuration
            on_selection_recorded: Callback for recording tool selections

        Returns:
            IToolSelector implementation from the new strategy factory
        """
        from victor.agent.tool_selector_factory import create_tool_selector_strategy
        from victor.storage.embeddings.service import get_embedding_service

        # Get strategy from settings (migrated from use_semantic_tool_selection)
        strategy = getattr(self.settings, "tool_selection_strategy", "auto")

        # Get embedding service for semantic/hybrid strategies
        embedding_service = None
        if strategy in ("semantic", "hybrid", "auto"):
            try:
                embedding_service = get_embedding_service()
            except Exception as e:
                logger.debug(f"Could not get embedding service: {e}")

        # Get enabled tools from tool selection config
        enabled_tools = None
        if tool_selection and isinstance(tool_selection, dict):
            enabled_tools = tool_selection.get("enabled_tools")

        # Create selector using the new factory
        selector = create_tool_selector_strategy(
            strategy=strategy,
            tools=tools,
            conversation_state=conversation_state,
            model=model,
            provider_name=provider_name,
            enabled_tools=enabled_tools,
            embedding_service=embedding_service,
            settings=self.settings,
        )

        logger.debug(
            f"ToolSelector created using strategy='{strategy}' "
            f"(migrated from use_semantic_tool_selection setting)"
        )
        return selector

    def create_intent_classifier(self) -> Any:
        """Create intent classifier for semantic continuation/completion detection.

        Uses singleton pattern via get_instance(). Intent classifier uses
        embeddings instead of hardcoded phrase matching.

        Returns:
            IntentClassifier singleton instance
        """
        from victor.storage.embeddings.intent_classifier import IntentClassifier

        classifier = IntentClassifier.get_instance()
        logger.debug("IntentClassifier singleton retrieved")
        return classifier

    def setup_subagent_orchestration(self) -> tuple[Optional[Any], bool]:
        """Setup sub-agent orchestration with lazy initialization.

        Enables spawning specialized sub-agents for parallel task delegation.
        Actual SubAgentOrchestrator is created lazily via property getter.

        Returns:
            Tuple of (None, enabled_flag) for lazy initialization pattern
        """
        enabled = getattr(self.settings, "subagent_orchestration_enabled", True)
        logger.debug(f"Sub-agent orchestration setup: enabled={enabled}")
        return (None, enabled)

    def setup_semantic_selection(self) -> tuple[bool, Optional[Any]]:
        """Setup semantic tool selection and background embedding preload.

        Initializes the semantic selection flag and prepares for background
        embedding preload task (lazy initialization pattern).

        Returns:
            Tuple of (use_semantic_selection, embedding_preload_task_placeholder)
            where embedding_preload_task_placeholder is None (lazy init)
        """
        use_semantic = getattr(self.settings, "use_semantic_tool_selection", False)
        logger.debug(f"Semantic selection setup: enabled={use_semantic}")
        # Embedding preload task is lazily initialized by ToolSelector
        return (use_semantic, None)

    def wire_component_dependencies(
        self,
        recovery_handler: Any,
        context_compactor: Any,
        observability: Any,
        conversation_state: Any,
    ) -> None:
        """Wire component dependencies after initialization.

        Handles post-initialization wiring of components to ensure proper
        integration between recovery handler, context compactor, observability,
        and conversation state.

        Args:
            recovery_handler: RecoveryHandler instance
            context_compactor: ContextCompactor instance
            observability: ObservabilityIntegration instance
            conversation_state: ConversationStateMachine instance
        """
        # Wire context compactor to recovery handler
        if recovery_handler and hasattr(recovery_handler, "set_context_compactor"):
            recovery_handler.set_context_compactor(context_compactor)
            logger.debug("RecoveryHandler wired with ContextCompactor")

        # Wire conversation state to observability
        if observability and conversation_state:
            observability.wire_state_machine(conversation_state)
            logger.debug("Observability integration wired with ConversationStateMachine")

    def create_provider_manager_with_adapter(
        self,
        provider: "BaseProvider",
        model: str,
        provider_name: str,
    ) -> tuple[Any, Any, str, str, Any, Any]:
        """Create ProviderManager and initialize tool adapter.

        Creates ProviderManager for unified provider/model management,
        initializes the tool adapter, and returns exposed attributes.

        Args:
            provider: Initial LLM provider instance
            model: Initial model identifier
            provider_name: Provider name string

        Returns:
            Tuple of (provider_manager, provider, model, provider_name,
                     tool_adapter, tool_calling_caps)
        """
        from victor.agent.provider_manager import ProviderManager, ProviderManagerConfig

        # Create ProviderManager with health checks and auto-fallback
        manager = ProviderManager(
            settings=self.settings,
            initial_provider=provider,
            initial_model=model,
            provider_name=provider_name,
            config=ProviderManagerConfig(
                enable_health_checks=getattr(self.settings, "provider_health_checks", True),
                auto_fallback=getattr(self.settings, "provider_auto_fallback", True),
                fallback_providers=getattr(self.settings, "fallback_providers", []),
            ),
        )

        # Initialize tool adapter through ProviderManager
        manager.initialize_tool_adapter()

        # Log adapter configuration
        caps = manager.capabilities
        adapter = manager.tool_adapter
        if caps and adapter:
            logger.info(
                f"Tool calling adapter: {adapter.provider_name}, "
                f"native={caps.native_tool_calls}, "
                f"format={caps.tool_call_format.value}"
            )

        # Return manager and exposed attributes for backward compatibility
        return (
            manager,
            manager.provider,
            manager.model,
            manager.provider_name,
            manager.tool_adapter,
            manager.capabilities,
        )

    async def create_provider_pool_if_enabled(
        self,
        base_provider: "BaseProvider",
    ) -> tuple["BaseProvider", bool]:
        """Create provider pool if enabled in settings.

        Args:
            base_provider: Base provider to wrap in pool

        Returns:
            Tuple of (provider or pool, is_pool_enabled)
        """
        from victor.providers.provider_pool import (
            ProviderPoolConfig,
            PoolStrategy,
            create_provider_pool,
        )
        from victor.providers.load_balancer import LoadBalancerType
        from victor.providers.health_monitor import HealthCheckConfig
        from victor.providers.circuit_breaker import CircuitBreakerConfig

        # Check if provider pool is enabled
        if not getattr(self.settings, "enable_provider_pool", False):
            logger.debug("Provider pool disabled, using single provider")
            return base_provider, False

        # For now, only support pooling when we have multiple endpoints
        # This is a simplified implementation - full pooling would require
        # creating multiple provider instances from different base URLs
        provider_name = getattr(base_provider, "name", "unknown")

        # Check if provider has multiple base URLs (for load balancing)
        base_urls = getattr(self.settings, f"{provider_name}_base_urls", None)
        if not base_urls or not isinstance(base_urls, list) or len(base_urls) <= 1:
            logger.debug(
                f"Provider pool enabled but {provider_name} has only one endpoint, "
                "using single provider"
            )
            return base_provider, False

        # Create pool configuration from settings
        pool_config = ProviderPoolConfig(
            pool_size=getattr(self.settings, "pool_size", 3),
            min_instances=getattr(self.settings, "pool_min_instances", 1),
            load_balancer=LoadBalancerType(
                getattr(self.settings, "pool_load_balancer", "adaptive")
            ),
            pool_strategy=PoolStrategy.ACTIVE_ACTIVE,
            enable_warmup=getattr(self.settings, "pool_enable_warmup", True),
            warmup_concurrency=getattr(self.settings, "pool_warmup_concurrency", 3),
            health_check_config=HealthCheckConfig(
                check_interval_seconds=getattr(self.settings, "pool_health_check_interval", 30),
            ),
            circuit_breaker_config=CircuitBreakerConfig(),
            max_retries=getattr(self.settings, "pool_max_retries", 3),
        )

        # Create provider instances from each base URL
        providers = {}
        for i, base_url in enumerate(base_urls[: pool_config.pool_size]):
            # Create provider instance with specific base URL
            provider_instance = await self._create_provider_for_url(provider_name, base_url)
            provider_id = f"{provider_name}-{i}"
            providers[provider_id] = provider_instance

        # Create and initialize pool
        logger.info(
            f"Creating provider pool with {len(providers)} instances "
            f"using {pool_config.load_balancer.value} load balancing"
        )

        pool = await create_provider_pool(
            name=f"{provider_name}-pool",
            providers=providers,
            config=pool_config,
        )

        # Log pool stats
        stats = pool.get_pool_stats()
        logger.info(f"Provider pool stats: {stats}")

        # Return pool as BaseProvider (ProviderPool is a BaseProvider subclass)
        return pool, True  # type: ignore[return-value]

    async def _create_provider_for_url(
        self,
        provider_name: str,
        base_url: str,
    ) -> "BaseProvider":
        """Create a provider instance for a specific base URL.

        Args:
            provider_name: Name of the provider class
            base_url: Base URL for this provider instance

        Returns:
            BaseProvider instance configured with the base URL
        """
        from victor.providers.registry import ProviderRegistry

        # Get provider class
        provider_class = ProviderRegistry.get(provider_name)

        # Create instance with base URL override
        # This is a simplified approach - full implementation would
        # need to handle provider-specific initialization
        provider = provider_class(api_key=None, base_url=base_url)

        return provider

    def create_tool_calling_matrix(self) -> tuple[Any, Any]:
        """Create ToolCallingMatrix for managing tool calling capabilities.

        Extracts tool_calling_models from settings and creates matrix with
        always-allowed providers for native tool calling support.

        Returns:
            Tuple of (tool_calling_models, tool_capabilities)
        """
        from victor.config.model_capabilities import ToolCallingMatrix

        tool_calling_models = getattr(self.settings, "tool_calling_models", {})

        tool_capabilities = ToolCallingMatrix(
            tool_calling_models,
            always_allow_providers=[
                "openai",
                "anthropic",
                "google",
                "xai",
                # Cloud providers with native tool calling support
                "cerebras",
                "groq",
                "deepseek",
                "mistral",
                "together",
                "fireworks",
                "openrouter",
                "cohere",
                "moonshot",
            ],
        )

        logger.debug(f"ToolCallingMatrix created with {len(tool_calling_models)} model configs")
        return (tool_calling_models, tool_capabilities)

    def create_system_prompt_builder(
        self,
        provider_name: str,
        model: str,
        tool_adapter: Any,
        tool_calling_caps: Any,
    ) -> Any:
        """Create SystemPromptBuilder with vertical prompt contributors.

        Extracts prompt contributors from vertical extensions and creates
        SystemPromptBuilder for provider-specific prompt generation.

        Args:
            provider_name: Provider name string
            model: Model identifier
            tool_adapter: Tool calling adapter instance
            tool_calling_caps: Tool calling capabilities

        Returns:
            SystemPromptBuilder instance configured with prompt contributors
        """
        from victor.agent.prompt_builder import SystemPromptBuilder

        # Get prompt contributors from vertical extensions
        prompt_contributors = []
        try:
            from victor.core.verticals.protocols import VerticalExtensions

            extensions = self.container.get_optional(VerticalExtensions)
            if extensions and extensions.prompt_contributors:
                prompt_contributors = extensions.prompt_contributors
                logger.debug(
                    f"Loaded {len(prompt_contributors)} prompt contributors from verticals"
                )
        except Exception as e:
            logger.debug(f"Could not load vertical prompt contributors: {e}")

        # Create SystemPromptBuilder with all configuration
        prompt_builder = SystemPromptBuilder(
            provider_name=provider_name,
            model=model,
            tool_adapter=tool_adapter,
            capabilities=tool_calling_caps,
            prompt_contributors=prompt_contributors,
        )

        logger.debug(f"SystemPromptBuilder created for {provider_name}/{model}")
        return prompt_builder

    def initialize_tool_budget(self, tool_calling_caps: Any) -> int:
        """Initialize tool call budget with adapter recommendations.

        Uses adapter's recommended budget with settings override and ensures
        minimum budget for meaningful work.

        Args:
            tool_calling_caps: Tool calling capabilities from adapter

        Returns:
            Tool call budget (integer)
        """
        # Use adapter's recommended budget, with settings override
        # Note: For analysis tasks, this may be increased dynamically in stream_chat()
        default_budget: int = tool_calling_caps.recommended_tool_budget

        # Ensure minimum budget of 50 for meaningful work
        default_budget = max(default_budget, 50)

        # Allow settings to override
        tool_budget: int = getattr(self.settings, "tool_call_budget", default_budget)

        logger.debug(
            f"Tool budget initialized: {tool_budget} "
            f"(adapter recommended: {tool_calling_caps.recommended_tool_budget}, "
            f"minimum: 50)"
        )
        return tool_budget

    def initialize_execution_state(self) -> tuple[list[str], list[str], set[tuple[str, str]], bool]:
        """Initialize execution state containers.

        Creates empty containers for tracking tool execution state during
        conversation sessions.

        Returns:
            Tuple of (observed_files, executed_tools, failed_tool_signatures,
                     tool_capability_warned)
        """

        observed_files: list[str] = []
        executed_tools: list[str] = []
        failed_tool_signatures: set[tuple[str, str]] = set()
        tool_capability_warned = False

        logger.debug("Execution state containers initialized")
        return (observed_files, executed_tools, failed_tool_signatures, tool_capability_warned)

    def create_debug_logger_configured(self) -> Any:
        """Create and configure debug logger for conversation tracking.

        Creates debug logger with enabled flag based on settings or log level.
        Used for incremental output and conversation tracking.

        Returns:
            Configured debug logger instance
        """
        import logging
        from victor.agent.protocols import DebugLoggerProtocol
        from victor.config.config_loaders import get_provider_limits

        # Resolve from DI container
        debug_logger = self.container.get(DebugLoggerProtocol)

        # Enable if debug_logging setting is True OR victor logger is at DEBUG level
        debug_logger.enabled = (
            getattr(self.settings, "debug_logging", False)
            or logging.getLogger("victor").level <= logging.DEBUG
        )

        # Set context window from provider/model configuration
        try:
            provider = getattr(self, "provider_name", "")
            model = getattr(self, "model", "")
            limits = get_provider_limits(provider, model)
            debug_logger.context_window = limits.context_window
            logger.debug(
                f"Debug logger configured with context_window={debug_logger.context_window:,} "
                f"(provider={provider}, model={model})"
            )
        except Exception as e:
            # Fall back to default context_window if unable to get provider limits
            logger.debug(f"Unable to set context_window from provider limits: {e}")

        logger.debug(
            f"Debug logger initialized: enabled={debug_logger.enabled}, context_window={debug_logger.context_window:,}"
        )
        return debug_logger

    def create_tool_access_controller(self, registry: Any = None) -> Any:
        """Create ToolAccessController for unified tool access control.

        Creates a layered tool access controller that consolidates 6 independent
        access control systems with clear precedence:
        Safety (L0) > Mode (L1) > Session (L2) > Vertical (L3) > Stage (L4) > Intent (L5)

        Args:
            registry: Optional tool registry for tool lookup

        Returns:
            ToolAccessController instance
        """
        from victor.agent.tool_access_controller import create_tool_access_controller

        controller = create_tool_access_controller(registry=registry)

        # Apply mode exploration multiplier if available (via ModeAwareMixin)
        mc = self.mode_controller
        if mc is not None and hasattr(mc.config, "exploration_multiplier"):
            multiplier = mc.config.exploration_multiplier
            logger.debug(f"ToolAccessController created with mode multiplier: {multiplier}")

        return controller

    def create_budget_manager(self) -> Any:
        """Create BudgetManager for unified budget tracking.

        Creates a budget manager that centralizes all budget tracking with
        consistent multiplier composition:
        effective_max = base × model_multiplier × mode_multiplier × productivity_multiplier

        Returns:
            BudgetManager instance with settings-derived configuration
        """
        from victor.agent.budget_manager import create_budget_manager
        from victor.agent.protocols import BudgetConfig

        # Get budget settings from settings
        base_tool_calls = getattr(self.settings, "tool_budget", 30)
        base_iterations = getattr(self.settings, "max_iterations", 50)
        base_exploration = getattr(self.settings, "max_exploration_iterations", 8)
        base_action = getattr(self.settings, "max_action_iterations", 12)

        config = BudgetConfig(
            base_tool_calls=base_tool_calls,
            base_iterations=base_iterations,
            base_exploration=base_exploration,
            base_action=base_action,
        )

        manager = create_budget_manager(config=config)

        # Apply mode multiplier if available (via ModeAwareMixin)
        mc = self.mode_controller
        if mc is not None and hasattr(mc.config, "exploration_multiplier"):
            manager.set_mode_multiplier(mc.config.exploration_multiplier)
            logger.debug(
                f"BudgetManager created with mode multiplier: "
                f"{mc.config.exploration_multiplier}"
            )

        return manager

    # =========================================================================
    # Workflow Optimization Components
    # =========================================================================

    def create_task_completion_detector(self) -> Any:
        """Create TaskCompletionDetector for detecting task completion.

        Issue Reference: workflow-test-issues-v2.md Issue #1

        Returns:
            TaskCompletionDetector instance
        """
        from victor.agent.task_completion import create_task_completion_detector

        detector = create_task_completion_detector()
        logger.debug("TaskCompletionDetector created")
        return detector

    def create_read_cache(self) -> Any:
        """Create ReadResultCache for file read deduplication.

        Issue Reference: workflow-test-issues-v2.md Issue #2

        Returns:
            ReadResultCache instance with settings-derived configuration
        """
        from victor.agent.read_cache import create_read_cache

        # Get cache settings from settings
        ttl_seconds = getattr(self.settings, "read_cache_ttl", 300.0)
        max_entries = getattr(self.settings, "read_cache_max_entries", 100)

        cache = create_read_cache(ttl_seconds=ttl_seconds, max_entries=max_entries)
        logger.debug(f"ReadResultCache created (ttl={ttl_seconds}s, max={max_entries})")
        return cache

    def create_time_aware_executor(self, timeout_seconds: Optional[float] = None) -> Any:
        """Create TimeAwareExecutor for time-aware execution management.

        Issue Reference: workflow-test-issues-v2.md Issue #3

        Args:
            timeout_seconds: Execution time budget (None for unlimited)

        Returns:
            TimeAwareExecutor instance
        """
        from victor.agent.time_aware_executor import create_time_aware_executor

        # Get timeout from settings if not specified
        if timeout_seconds is None:
            timeout_seconds = getattr(self.settings, "execution_timeout", None)

        executor = create_time_aware_executor(timeout_seconds=timeout_seconds)
        if timeout_seconds:
            logger.debug(f"TimeAwareExecutor created with {timeout_seconds}s budget")
        else:
            logger.debug("TimeAwareExecutor created (no timeout)")
        return executor

    def create_thinking_detector(self) -> Any:
        """Create ThinkingPatternDetector for detecting thinking loops.

        Issue Reference: workflow-test-issues-v2.md Issue #4

        Returns:
            ThinkingPatternDetector instance
        """
        from victor.agent.thinking_detector import create_thinking_detector

        # Get detector settings from settings
        repetition_threshold = getattr(self.settings, "thinking_repetition_threshold", 3)
        similarity_threshold = getattr(self.settings, "thinking_similarity_threshold", 0.65)

        detector = create_thinking_detector(
            repetition_threshold=repetition_threshold,
            similarity_threshold=similarity_threshold,
        )
        logger.debug(
            f"ThinkingPatternDetector created "
            f"(repetition={repetition_threshold}, similarity={similarity_threshold})"
        )
        return detector

    def create_resource_manager(self) -> Any:
        """Get ResourceManager for resource lifecycle management.

        Issue Reference: workflow-test-issues-v2.md Issue #5

        Returns:
            ResourceManager singleton instance
        """
        from victor.agent.resource_manager import get_resource_manager

        manager = get_resource_manager()
        logger.debug("ResourceManager retrieved (singleton)")
        return manager

    def create_mode_completion_criteria(self) -> Any:
        """Create ModeCompletionChecker for mode-specific early exit.

        Issue Reference: workflow-test-issues-v2.md Issue #6

        Returns:
            ModeCompletionChecker instance
        """
        from victor.agent.budget_manager import create_mode_completion_criteria

        criteria = create_mode_completion_criteria()
        logger.debug("ModeCompletionChecker created")
        return criteria

    def create_checkpoint_manager(self) -> Optional[Any]:
        """Create ConversationCheckpointManager for time-travel debugging.

        Creates the checkpoint manager based on settings configuration.
        Returns None if checkpointing is disabled.

        Returns:
            ConversationCheckpointManager instance or None if disabled
        """
        if not getattr(self.settings, "checkpoint_enabled", True):
            logger.debug("Checkpoint system disabled via settings")
            return None

        try:
            from victor.storage.checkpoints import (
                ConversationCheckpointManager,
                SQLiteCheckpointBackend,
            )
            from victor.config.settings import get_project_paths

            # Get project paths for checkpoint storage
            paths = get_project_paths()

            # Create SQLite backend with directory path (not full file path)
            # SQLiteCheckpointBackend expects storage_path as Path, db_name as filename
            backend = SQLiteCheckpointBackend(
                storage_path=paths.project_victor_dir,
                db_name="checkpoints.db",
            )

            # Get settings
            auto_interval = getattr(self.settings, "checkpoint_auto_interval", 5)
            max_per_session = getattr(self.settings, "checkpoint_max_per_session", 50)

            manager = ConversationCheckpointManager(
                backend=backend,
                auto_checkpoint_interval=auto_interval,
                max_checkpoints_per_session=max_per_session,
            )

            logger.info(
                f"ConversationCheckpointManager created (auto_interval={auto_interval}, "
                f"max_per_session={max_per_session})"
            )
            return manager

        except Exception as e:
            logger.warning(f"Failed to create ConversationCheckpointManager: {e}")
            return None

    def create_workflow_optimization_components(
        self, timeout_seconds: Optional[float] = None
    ) -> WorkflowOptimizationComponents:
        """Create all workflow optimization components.

        Args:
            timeout_seconds: Execution timeout for time-aware executor

        Returns:
            WorkflowOptimizationComponents with all optimization components
        """
        return WorkflowOptimizationComponents(
            task_completion_detector=self.create_task_completion_detector(),
            read_cache=self.create_read_cache(),
            time_aware_executor=self.create_time_aware_executor(timeout_seconds),
            thinking_detector=self.create_thinking_detector(),
            resource_manager=self.create_resource_manager(),
            mode_completion_criteria=self.create_mode_completion_criteria(),
        )

    def create_mode_workflow_team_coordinator(
        self,
        vertical_context: Any,
    ) -> Any:
        """Create ModeWorkflowTeamCoordinator for intelligent task coordination.

        The coordinator bridges agent modes, team specifications, and workflows
        to provide intelligent suggestions for task execution.

        Args:
            vertical_context: VerticalContext with team specs and workflows

        Returns:
            ModeWorkflowTeamCoordinator instance
        """
        from victor.agent.mode_workflow_team_coordinator import create_coordinator

        # Get team learner from RL coordinator if available
        team_learner = None
        try:
            from victor.agent.protocols import RLCoordinatorProtocol

            rl_coordinator = self.container.get_optional(RLCoordinatorProtocol)
            if rl_coordinator:
                team_learner = rl_coordinator.get_learner("team_composition")
        except Exception as e:
            logger.debug(f"Could not get team composition learner: {e}")

        # Determine selection strategy from settings
        selection_strategy = getattr(self.settings, "team_selection_strategy", "hybrid")

        coordinator = create_coordinator(
            vertical_context=vertical_context,
            team_learner=team_learner,
            selection_strategy=selection_strategy,
        )

        logger.debug(f"ModeWorkflowTeamCoordinator created with strategy={selection_strategy}")
        return coordinator

    # =========================================================================
    # Orchestrator Decomposition Components (Phase 1)
    # =========================================================================

    def create_response_processor(
        self,
        tool_adapter: Any,
        tool_registry: Any,
        sanitizer: Any,
        shell_resolver: Optional[Any] = None,
        output_formatter: Optional[Any] = None,
    ) -> Any:
        """Create ResponseProcessor for tool call parsing and response handling.

        This processor is extracted from AgentOrchestrator to reduce class size
        while maintaining the same functionality.

        Args:
            tool_adapter: Tool calling adapter for parsing
            tool_registry: Registry for checking enabled tools
            sanitizer: Validator for tool names and content
            shell_resolver: Optional resolver for shell variants
            output_formatter: Optional formatter for tool output

        Returns:
            ResponseProcessor instance
        """
        from victor.agent.response_processor import ResponseProcessor

        processor = ResponseProcessor(
            tool_adapter=tool_adapter,
            tool_registry=tool_registry,
            sanitizer=sanitizer,
            shell_resolver=shell_resolver,
            output_formatter=output_formatter,
        )

        logger.debug("ResponseProcessor created")
        return processor

    # =========================================================================
    # Unified Agent Creation (Phase 4)
    # =========================================================================

    async def create_agent(
        self,
        mode: str = "foreground",
        config: Optional[Any] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Create ANY agent type using shared factory infrastructure.

        This is the ONLY method that should create agents. All other entrypoints
        (Agent.create, BackgroundAgentManager.start_agent, Vertical.create_agent)
        must delegate here to ensure consistent code maintenance and eliminate
        code proliferation (SOLID SRP, DIP).

        Args:
            mode: Agent creation mode
                - "foreground": Interactive Agent instance (default)
                - "background": BackgroundAgent for async task execution
                - "team_member": TeamMember/SubAgent for multi-agent teams
            config: Optional unified agent configuration (UnifiedAgentConfig)
            task: Optional task description (for background agents)
            **kwargs: Additional agent-specific parameters
                - For foreground: provider, model, temperature, max_tokens, tools, etc.
                - For background: mode_type ("build", "plan", "explore"), websocket, etc.
                - For team_member: role, capabilities, description, etc.

        Returns:
            Agent instance based on mode:
            - mode="foreground": Agent (victor.framework.agent.Agent)
            - mode="background": BackgroundAgent (victor.agent.background_agent.BackgroundAgent)
            - mode="team_member": TeamMember/SubAgent (victor.teams.types.TeamMember)

        Raises:
            ValueError: If mode is invalid or required parameters are missing
            ProviderError: If provider initialization fails

        Examples:
            # Foreground agent (simple)
            agent = await factory.create_agent(mode="foreground")

            # Foreground agent with UnifiedAgentConfig

        This is the ONLY method that should create agents. All other entrypoints
        (Agent.create, BackgroundAgentManager.start_agent, Vertical.create_agent)
        must delegate here to ensure consistent code maintenance and eliminate
        code proliferation (SOLID SRP, DIP).

        Args:
            mode: Agent creation mode
                - "foreground": Interactive Agent instance (default)
                - "background": BackgroundAgent for async task execution
                - "team_member": TeamMember/SubAgent for multi-agent teams
            config: Optional unified agent configuration (UnifiedAgentConfig)
            task: Optional task description (for background agents)
            **kwargs: Additional agent-specific parameters
                - For foreground: provider, model, temperature, max_tokens, tools, etc.
                - For background: mode_type ("build", "plan", "explore"), websocket, etc.
                - For team_member: role, capabilities, description, etc.

        Returns:
            Agent instance based on mode:
            - mode="foreground": Agent (victor.framework.agent.Agent)
            - mode="background": BackgroundAgent (victor.agent.background_agent.BackgroundAgent)
            - mode="team_member": TeamMember/SubAgent (victor.teams.types.TeamMember)

        Raises:
            ValueError: If mode is invalid or required parameters are missing
            ProviderError: If provider initialization fails

        Examples:
            # Foreground agent (simple)
            agent = await factory.create_agent(mode="foreground")

            # Foreground agent with UnifiedAgentConfig
            from victor.agent.config import UnifiedAgentConfig
            config = UnifiedAgentConfig.foreground(
                provider="openai",
                model="gpt-4-turbo",
                tool_budget=100
            )
            agent = await factory.create_agent(config=config)

            # Foreground agent with options (legacy)
            agent = await factory.create_agent(
                mode="foreground",
                provider="openai",
                model="gpt-4-turbo",
                tools=ToolSet.coding()
            )

            # Background agent with task
            agent = await factory.create_agent(
                mode="background",
                task="Implement feature X",
                mode_type="build"
            )

            # Background agent with UnifiedAgentConfig
            config = UnifiedAgentConfig.background(
                task="Implement feature X",
                tool_budget=100
            )
            agent = await factory.create_agent(config=config)

            # Team member
            agent = await factory.create_agent(
                mode="team_member",
                role="researcher",
                capabilities=["search", "analyze"]
            )
        """
        from victor.agent.orchestrator import AgentOrchestrator

        # Validate mode early (before expensive setup)
        valid_modes = {"foreground", "background", "team_member"}
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid agent mode: {mode!r}. " f"Must be one of {sorted(valid_modes)}"
            )

        # Extract parameters from UnifiedAgentConfig if provided
        if config:
            # Import here to avoid circular dependency
            from victor.agent.config import UnifiedAgentConfig

            if isinstance(config, UnifiedAgentConfig):
                # Apply config settings to kwargs (config takes precedence)
                mode = config.mode
                task = config.task or task

                # Merge config settings with kwargs (kwargs override config)
                merged_kwargs: dict[str, Any] = {}

                # Common settings
                if config.provider != "anthropic":
                    merged_kwargs["provider"] = config.provider
                if config.model:
                    merged_kwargs["model"] = config.model
                if config.temperature != 0.7:
                    merged_kwargs["temperature"] = config.temperature
                if config.max_tokens != 4096:
                    merged_kwargs["max_tokens"] = config.max_tokens

                # Foreground-specific
                if mode == "foreground":
                    merged_kwargs["tool_budget"] = config.tool_budget
                    merged_kwargs["max_iterations"] = config.max_iterations
                    merged_kwargs["enable_parallel_tools"] = config.enable_parallel_tools
                    merged_kwargs["max_concurrent_tools"] = config.max_concurrent_tools
                    merged_kwargs["enable_context_compaction"] = config.enable_context_compaction
                    merged_kwargs["enable_semantic_search"] = config.enable_semantic_search
                    merged_kwargs["enable_analytics"] = config.enable_analytics

                # Background-specific
                elif mode == "background":
                    merged_kwargs["task"] = config.task or task
                    merged_kwargs["mode_type"] = config.mode_type
                    merged_kwargs["websocket"] = config.websocket
                    merged_kwargs["timeout_seconds"] = config.timeout_seconds
                    merged_kwargs["tool_budget"] = config.tool_budget

                # Team member-specific
                elif mode == "team_member":
                    merged_kwargs["role"] = config.role
                    merged_kwargs["capabilities"] = config.capabilities
                    merged_kwargs["description"] = config.description
                    merged_kwargs["allowed_tools"] = config.allowed_tools
                    merged_kwargs["can_spawn_subagents"] = config.can_spawn_subagents

                # Apply extra settings
                merged_kwargs.update(config.extra)

                # Merge with provided kwargs (kwargs take precedence)
                merged_kwargs.update(kwargs)
                kwargs = merged_kwargs

        # Create orchestrator using existing factory methods
        # Note: We re-use the existing from_settings pattern which internally
        # uses this factory's create_* methods
        orchestrator = await AgentOrchestrator.from_settings(
            self.settings,
            profile_name=self.profile_name or "default",
            thinking=self.thinking,
        )

        # Override provider/model if different from defaults
        provider = kwargs.get("provider")
        model = kwargs.get("model")
        if provider or model:
            if hasattr(orchestrator, "_provider_manager") or hasattr(
                orchestrator, "provider_manager"
            ):
                pm = getattr(orchestrator, "_provider_manager", None) or getattr(
                    orchestrator, "provider_manager", None
                )
                if pm:
                    await pm.switch_provider(
                        provider or pm.provider_name,
                        model or self.model,
                    )

        # Create agent based on mode
        if mode == "foreground":
            # Import Agent to avoid circular dependency
            from victor.framework.agent import Agent

            # Create foreground Agent
            agent = Agent(orchestrator)  # type: ignore[arg-type]

            # Apply tools if specified
            tools = kwargs.get("tools")
            if tools:
                from victor.framework.tool_config import configure_tools

                configure_tools(orchestrator, tools)  # type: ignore[arg-type]

            # Apply vertical configuration if specified
            vertical = kwargs.get("vertical")
            if vertical:
                # Apply vertical configuration
                vertical_config = vertical.get_config()
                if vertical_config.system_prompt:
                    orchestrator.system_prompt = vertical_config.system_prompt

            return agent

        elif mode == "background":
            # Import BackgroundAgent
            from victor.agent.background_agent import BackgroundAgent
            import uuid
            import time

            # Create background agent
            background_agent = BackgroundAgent(
                id=str(uuid.uuid4()),
                name=f"Background-{int(time.time())}",
                description=task or kwargs.get("task", "Background task"),
                task=task or kwargs.get("task", ""),
                mode=kwargs.get("mode_type", "build"),
                orchestrator=orchestrator,
            )
            return background_agent

        elif mode == "team_member":
            # Import TeamMember
            from victor.teams.types import TeamMember
            import uuid

            # Create team member (wraps existing agent/team member)
            member = TeamMember(
                id=str(uuid.uuid4()),
                role=kwargs.get("role", "team_member"),
                name=kwargs.get("name", kwargs.get("role", "team_member")),
                goal=kwargs.get("description", ""),
            )
            return member

        else:
            # This should never be reached due to early validation above
            raise ValueError(
                f"Invalid agent mode: {mode!r}. " f"Must be one of {sorted(valid_modes)}"
            )


# Convenience function for creating factory
def create_orchestrator_factory(
    settings: "Settings",
    provider: "BaseProvider",
    model: str,
    **kwargs: Any,
) -> OrchestratorFactory:
    """Create an OrchestratorFactory instance.

    Args:
        settings: Application settings
        provider: LLM provider instance
        model: Model identifier
        **kwargs: Additional factory configuration

    Returns:
        Configured OrchestratorFactory instance

    Example:
        factory = create_orchestrator_factory(settings, provider, "gpt-4")
    """
    logger.debug("Creating OrchestratorFactory")

    return OrchestratorFactory(
        settings=settings,
        provider=provider,
        model=model,
        **kwargs,
    )
