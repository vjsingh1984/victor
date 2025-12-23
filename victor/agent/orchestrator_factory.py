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
    components = factory.create_all_components()
    orchestrator = AgentOrchestrator._from_components(components)

Or use individual creation methods for testing:
    factory = OrchestratorFactory(settings, provider, model)
    sanitizer = factory.create_sanitizer()
    prompt_builder = factory.create_prompt_builder()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.config.settings import Settings
    from victor.providers.base import BaseProvider
    from victor.agent.tool_calling import BaseToolCallingAdapter, ToolCallingCapabilities
    from victor.agent.response_sanitizer import ResponseSanitizer
    from victor.agent.prompt_builder import SystemPromptBuilder
    from victor.agent.context_project import ProjectContext
    from victor.agent.complexity_classifier import ComplexityClassifier
    from victor.agent.action_authorizer import ActionAuthorizer
    from victor.agent.search_router import SearchRouter
    from victor.agent.metrics_collector import MetricsCollector
    from victor.agent.conversation_controller import ConversationController
    from victor.agent.tool_pipeline import ToolPipeline
    from victor.agent.streaming_controller import StreamingController
    from victor.agent.context_compactor import ContextCompactor
    from victor.agent.usage_analytics import UsageAnalytics
    from victor.agent.tool_sequence_tracker import ToolSequenceTracker
    from victor.agent.tool_output_formatter import ToolOutputFormatter
    from victor.agent.recovery import RecoveryHandler
    from victor.observability.integration import ObservabilityIntegration

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
class OrchestratorComponents:
    """All components needed to construct an AgentOrchestrator.

    This dataclass serves as a transfer object containing all initialized
    components, allowing the orchestrator's __init__ to be simplified to
    just assigning these pre-constructed components.
    """

    # Provider
    provider: ProviderComponents = field(default_factory=lambda: None)  # type: ignore

    # Core services
    services: CoreServices = field(default_factory=lambda: None)  # type: ignore

    # Conversation
    conversation: ConversationComponents = field(default_factory=lambda: None)  # type: ignore

    # Tools
    tools: ToolComponents = field(default_factory=lambda: None)  # type: ignore

    # Streaming
    streaming: StreamingComponents = field(default_factory=lambda: None)  # type: ignore

    # Analytics
    analytics: AnalyticsComponents = field(default_factory=lambda: None)  # type: ignore

    # Recovery
    recovery: RecoveryComponents = field(default_factory=lambda: None)  # type: ignore

    # Observability
    observability: Optional["ObservabilityIntegration"] = None

    # Tool output formatter
    tool_output_formatter: Optional["ToolOutputFormatter"] = None


class OrchestratorFactory:
    """Factory for creating AgentOrchestrator components.

    This factory extracts component initialization logic from the orchestrator's
    __init__ method, providing:

    1. Cleaner separation of concerns
    2. Easier testing of individual component creation
    3. Potential for lazy initialization
    4. Reduced orchestrator complexity

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
        temperature: float = 0.7,
        max_tokens: int = 4096,
        provider_name: Optional[str] = None,
        profile_name: Optional[str] = None,
        tool_selection: Optional[Dict[str, Any]] = None,
        thinking: bool = False,
    ):
        """Initialize the factory with core configuration.

        Args:
            settings: Application settings
            provider: LLM provider instance
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            provider_name: Provider label (for OpenAI-compatible disambiguation)
            profile_name: Profile name for session tracking
            tool_selection: Tool selection configuration
            thinking: Enable extended thinking mode
        """
        self.settings = settings
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider_name = provider_name
        self.profile_name = profile_name
        self.tool_selection = tool_selection or {}
        self.thinking = thinking

        # Lazy-initialized container
        self._container = None

    @property
    def container(self):
        """Get or create the DI container."""
        if self._container is None:
            from victor.core.bootstrap import ensure_bootstrapped
            self._container = ensure_bootstrapped(self.settings)
        return self._container

    def create_sanitizer(self) -> "ResponseSanitizer":
        """Create response sanitizer (from DI container)."""
        from victor.agent.protocols import ResponseSanitizerProtocol

        return self.container.get(ResponseSanitizerProtocol)

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
        from victor.verticals.protocols import VerticalExtensions

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

        return self.container.get(ComplexityClassifierProtocol)

    def create_action_authorizer(self) -> "ActionAuthorizer":
        """Create action authorizer (from DI container)."""
        from victor.agent.protocols import ActionAuthorizerProtocol

        return self.container.get(ActionAuthorizerProtocol)

    def create_search_router(self) -> "SearchRouter":
        """Create search router (from DI container)."""
        from victor.agent.protocols import SearchRouterProtocol

        return self.container.get(SearchRouterProtocol)

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

        return self.container.get(UsageAnalyticsProtocol)

    def create_sequence_tracker(self) -> "ToolSequenceTracker":
        """Create tool sequence tracker (from DI container)."""
        from victor.agent.protocols import ToolSequenceTrackerProtocol

        return self.container.get(ToolSequenceTrackerProtocol)

    def create_recovery_handler(self) -> Optional["RecoveryHandler"]:
        """Create recovery handler (from DI container)."""
        from victor.agent.protocols import RecoveryHandlerProtocol

        # RecoveryHandler is always registered, but may be disabled via settings
        return self.container.get(RecoveryHandlerProtocol)

    def create_observability(self) -> Optional["ObservabilityIntegration"]:
        """Create observability integration if enabled."""
        from victor.observability.integration import ObservabilityIntegration

        if not getattr(self.settings, "enable_observability", True):
            return None

        observability = ObservabilityIntegration()
        logger.debug("Observability integration created")
        return observability

    def create_tool_cache(self) -> Optional[Any]:
        """Create tool cache if enabled.

        Returns:
            ToolCache instance or None if disabled
        """
        if not getattr(self.settings, "tool_cache_enabled", True):
            return None

        from victor.cache.config import CacheConfig
        from victor.config.settings import get_project_paths
        from victor.cache.tool_cache import ToolCache

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
    ) -> Tuple[Optional[Any], Optional[str]]:
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

    def create_middleware_chain(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Create middleware chain with vertical extensions.

        Returns:
            Tuple of (MiddlewareChain or None, CodeCorrectionMiddleware or None)
        """
        middleware_chain: Optional[Any] = None
        code_correction_middleware: Optional[Any] = None
        code_correction_enabled = getattr(self.settings, "code_correction_enabled", True)

        try:
            from victor.agent.middleware_chain import MiddlewareChain
            from victor.verticals.protocols import VerticalExtensions

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
            from victor.verticals.protocols import VerticalExtensions

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

    def create_semantic_selector(self) -> Optional[Any]:
        """Create semantic tool selector if enabled.

        Returns:
            SemanticToolSelector instance or None
        """
        use_semantic_selection = getattr(self.settings, "use_semantic_tool_selection", False)

        if not use_semantic_selection:
            return None

        from victor.tools.semantic_selector import SemanticToolSelector

        # Use settings-configured embedding provider and model
        # Default: sentence-transformers with unified_embedding_model (local, fast, air-gapped)
        # Both tool selection and codebase search use the same model for:
        # - 40% memory reduction (120MB vs 200MB)
        # - Better OS page cache utilization
        # - Improved CPU L2/L3 cache hit rates
        semantic_selector = SemanticToolSelector(
            embedding_model=self.settings.embedding_model,
            embedding_provider=self.settings.embedding_provider,
            ollama_base_url=self.settings.ollama_base_url,
            cache_embeddings=True,
        )
        logger.debug("SemanticToolSelector created")
        return semantic_selector

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

    def create_unified_tracker(
        self, tool_calling_caps: "ToolCallingCapabilities"
    ) -> Any:
        """Create unified task tracker with model-specific exploration settings.

        The UnifiedTaskTracker is the single source of truth for task progress,
        milestones, and loop detection across the orchestration lifecycle.

        Args:
            tool_calling_caps: Tool calling capabilities for model-specific settings

        Returns:
            UnifiedTaskTracker instance configured with model exploration parameters
        """
        from victor.agent.protocols import TaskTrackerProtocol

        # Resolve from DI container (guaranteed to be registered)
        unified_tracker = self.container.get(TaskTrackerProtocol)

        # Apply model-specific exploration settings
        unified_tracker.set_model_exploration_settings(
            exploration_multiplier=tool_calling_caps.exploration_multiplier,
            continuation_patience=tool_calling_caps.continuation_patience,
        )

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
                file_structure_threshold=getattr(
                    self.settings, "file_structure_threshold", 50000
                ),
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
            RecoveryCoordinator instance for recovery coordination
        """
        from victor.agent.protocols import RecoveryCoordinatorProtocol

        recovery_coordinator = self.container.get(RecoveryCoordinatorProtocol)
        logger.debug("RecoveryCoordinator created via DI")
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
        on_session_complete: Callable,
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

    def create_tool_pipeline(
        self,
        tools: Any,
        tool_executor: Any,
        tool_budget: int,
        tool_cache: Optional[Any],
        argument_normalizer: Any,
        on_tool_start: Callable,
        on_tool_complete: Callable,
        deduplication_tracker: Optional[Any],
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
            ),
            tool_cache=tool_cache,
            argument_normalizer=argument_normalizer,
            on_tool_start=on_tool_start,
            on_tool_complete=on_tool_complete,
            deduplication_tracker=deduplication_tracker,
        )
        logger.debug("ToolPipeline created")
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
        handler = StreamingChatHandler(
            settings=self.settings,
            message_adder=message_adder,
            session_idle_timeout=session_idle_timeout,
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

    def create_context_compactor(self, conversation_controller: Any) -> Any:
        """Create context compactor for proactive context management.

        This component provides:
        - Proactive compaction before context overflow (triggers at 70% utilization)
        - Smart tool result truncation with content-aware strategies
        - Token estimation with content-type-specific factors

        Args:
            conversation_controller: ConversationController instance for context tracking

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
        )

        logger.debug(
            f"ContextCompactor created with truncation_strategy={truncation_strategy}"
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
        )

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

        # Resolve from DI container (guaranteed to be registered)
        return self.container.get(ConversationStateMachineProtocol)

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

        max_history = getattr(self.settings, "max_conversation_history", 100)
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
        from victor.tools.base import ToolRegistry

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
        semantic_selector: Any,
        conversation_state: Any,
        unified_tracker: Any,
        model: str,
        provider_name: str,
        tool_selection: Any,
        on_selection_recorded: Any,
    ) -> Any:
        """Create unified tool selector for semantic and keyword-based selection.

        Args:
            tools: ToolRegistry instance
            semantic_selector: SemanticToolSelector instance
            conversation_state: ConversationStateMachine instance
            unified_tracker: UnifiedTaskTracker instance
            model: Model identifier
            provider_name: Provider name
            tool_selection: Tool selection configuration
            on_selection_recorded: Callback for recording tool selections

        Returns:
            ToolSelector instance configured with all dependencies
        """
        from victor.agent.tool_selection import ToolSelector

        fallback_max_tools = getattr(self.settings, "fallback_max_tools", 8)

        selector = ToolSelector(
            tools=tools,
            semantic_selector=semantic_selector,
            conversation_state=conversation_state,
            task_tracker=unified_tracker,
            model=model,
            provider_name=provider_name,
            tool_selection_config=tool_selection,
            fallback_max_tools=fallback_max_tools,
            on_selection_recorded=on_selection_recorded,
        )

        logger.debug(
            f"ToolSelector created with fallback_max_tools={fallback_max_tools}"
        )
        return selector

    def create_intent_classifier(self) -> Any:
        """Create intent classifier for semantic continuation/completion detection.

        Uses singleton pattern via get_instance(). Intent classifier uses
        embeddings instead of hardcoded phrase matching.

        Returns:
            IntentClassifier singleton instance
        """
        from victor.embeddings.intent_classifier import IntentClassifier

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
        logger.info(
            f"Tool calling adapter: {manager.tool_adapter.provider_name}, "
            f"native={manager.capabilities.native_tool_calls}, "
            f"format={manager.capabilities.tool_call_format.value}"
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

        logger.debug(
            f"ToolCallingMatrix created with {len(tool_calling_models)} model configs"
        )
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
            from victor.verticals.protocols import VerticalExtensions

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
        default_budget = tool_calling_caps.recommended_tool_budget

        # Ensure minimum budget of 50 for meaningful work
        default_budget = max(default_budget, 50)

        # Allow settings to override
        tool_budget = getattr(self.settings, "tool_call_budget", default_budget)

        logger.debug(
            f"Tool budget initialized: {tool_budget} "
            f"(adapter recommended: {tool_calling_caps.recommended_tool_budget}, "
            f"minimum: 50)"
        )
        return tool_budget

    def initialize_execution_state(self) -> tuple[list, list, set, bool]:
        """Initialize execution state containers.

        Creates empty containers for tracking tool execution state during
        conversation sessions.

        Returns:
            Tuple of (observed_files, executed_tools, failed_tool_signatures,
                     tool_capability_warned)
        """
        from typing import List

        observed_files: List[str] = []
        executed_tools: List[str] = []
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

        # Resolve from DI container
        debug_logger = self.container.get(DebugLoggerProtocol)

        # Enable if debug_logging setting is True OR victor logger is at DEBUG level
        debug_logger.enabled = (
            getattr(self.settings, "debug_logging", False)
            or logging.getLogger("victor").level <= logging.DEBUG
        )

        logger.debug(f"Debug logger initialized: enabled={debug_logger.enabled}")
        return debug_logger


# Convenience function for creating factory
def create_orchestrator_factory(
    settings: "Settings",
    provider: "BaseProvider",
    model: str,
    **kwargs,
) -> OrchestratorFactory:
    """Create an OrchestratorFactory instance.

    Args:
        settings: Application settings
        provider: LLM provider instance
        model: Model identifier
        **kwargs: Additional factory configuration

    Returns:
        Configured OrchestratorFactory
    """
    return OrchestratorFactory(
        settings=settings,
        provider=provider,
        model=model,
        **kwargs,
    )
