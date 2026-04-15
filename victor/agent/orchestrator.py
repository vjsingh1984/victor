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

"""Agent orchestrator for managing conversations and tool execution.

Architecture: Facade Pattern
============================
AgentOrchestrator acts as a facade coordinating several extracted components:

Extracted Components (separate modules):
- ConversationController: Message history, context tracking, stage management
- ToolPipeline: Tool validation, execution coordination, budget enforcement
- StreamingController: Session lifecycle, metrics collection, cancellation
- StreamingCoordinator: Response processing, chunk aggregation, event dispatch (NEW)
- ProviderSwitchCoordinator: Provider/model switching workflow coordination (NEW)
- LifecycleManager: Session lifecycle and resource cleanup coordination (NEW)
- TaskAnalyzer: Unified facade for complexity/task/intent classification
- ToolSelector: Semantic and keyword-based tool selection
- ToolRegistrar: Tool registration, plugins, MCP integration (NEW)

Remaining Orchestrator Responsibilities:
- High-level chat flow coordination
- Configuration loading and validation
- Post-switch hooks (prompt rebuilding, tracker updates)

Recently Integrated:
- ProviderManager: Provider initialization, switching, health checks (NEW)
- StreamingCoordinator: Simple response processing for streaming use cases (NEW)
- ProviderSwitchCoordinator: Switch validation, health checks, retry logic (NEW)
- LifecycleManager: Session reset, recovery, graceful shutdown, resource cleanup (NEW)

Note: Keep orchestrator as a thin facade. New logic should go into
appropriate extracted components, not added here.

Recent Refactoring (December 2025 - January 2025):
- Extracted ToolRegistrar from _register_default_tools, _initialize_plugins,
    _setup_mcp_integration, _plan_tools, and _goal_hints_for_message
- Added ProviderHealthChecker for proactive health monitoring
- Added ResilienceMetricsExporter for dashboard integration
- Added classification-aware tool selection in SemanticToolSelector
- Added StreamingCoordinator for response processing (January 2025)
- Added ProviderSwitchCoordinator for switching workflow coordination (January 2025)
- Added LifecycleManager for lifecycle management (January 2025)
"""

import ast
import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from rich.console import Console

# Coordinators (Phase 2 refactoring - being integrated)
# NOTE: These are runtime imports, not type-checking only
from victor.agent.coordinators.metrics_coordinator import (
    MetricsCoordinator,
)  # noqa: F401  # imported for runtime use

if TYPE_CHECKING:
    # Type-only imports (created by factory, only used for type hints)
    from victor.agent.orchestrator_integration import OrchestratorIntegration
    from victor.agent.recovery_coordinator import StreamingRecoveryCoordinator
    from victor.agent.chunk_generator import ChunkGenerator
    from victor.agent.tool_planner import ToolPlanner
    from victor.agent.task_coordinator import TaskCoordinator
    from victor.agent.protocols import ToolAccessContext
    from victor.evaluation.protocol import TokenUsage

    # Factory-created components (type hints only)
    from victor.agent.response_sanitizer import ResponseSanitizer
    from victor.agent.search_router import SearchRouter
    from victor.framework.task import TaskComplexityService as ComplexityClassifier
    from victor.agent.metrics_collector import MetricsCollector
    from victor.agent.conversation_controller import ConversationController
    from victor.agent.context_compactor import ContextCompactor
    from victor.agent.usage_analytics import UsageAnalytics
    from victor.agent.tool_sequence_tracker import ToolSequenceTracker
    from victor.agent.recovery import RecoveryHandler
    from victor.agent.orchestrator_recovery import OrchestratorRecoveryIntegration
    from victor.agent.tool_output_formatter import ToolOutputFormatter
    from victor.agent.tool_pipeline import ToolPipeline
    from victor.runtime.context import ExecutionContext
    from victor.agent.streaming_controller import StreamingController
    from victor.agent.task_analyzer import TaskAnalyzer
    from victor.agent.tool_registrar import ToolRegistrar
    from victor.agent.provider_manager import ProviderManager
    from victor.agent.provider_coordinator import ProviderCoordinator
    from victor.agent.tool_selection import ToolSelector
    from victor.agent.tool_executor import ToolExecutor
    from victor.agent.safety import SafetyChecker
    from victor.agent.auto_commit import AutoCommitter
    from victor.agent.conversation_manager import (
        ConversationManager,
        create_conversation_manager,
    )

# Runtime imports - used for instantiation, enums, constants, or function calls
from victor.agent.argument_normalizer import ArgumentNormalizer, NormalizationStrategy
from victor.agent.message_history import MessageHistory
from victor.agent.conversation_memory import ConversationStore

# DI container bootstrap
from victor.core.bootstrap import ensure_bootstrapped, get_service_optional
from victor.core.container import MetricsServiceProtocol, LoggerServiceProtocol

# Service protocols for DI resolution
from victor.agent.protocols import (
    ResponseSanitizerProtocol,
    ComplexityClassifierProtocol,
    ActionAuthorizerProtocol,
    SearchRouterProtocol,
    ProjectContextProtocol,
    ArgumentNormalizerProtocol,
    ConversationStateMachineProtocol,
    TaskTrackerProtocol,
    CodeExecutionManagerProtocol,
    WorkflowRegistryProtocol,
    UsageAnalyticsProtocol,
    ToolSequenceTrackerProtocol,
    ContextCompactorProtocol,
)

# Mixins (used at class definition time)
from victor.protocols.mode_aware import ModeAwareMixin
from victor.agent.capability_registry import CapabilityRegistryMixin

# Config and enums (used at runtime)
from victor.config.config_loaders import get_provider_limits
from victor.agent.conversation_embedding_store import ConversationEmbeddingStore
from victor.agent.conversation_state import ConversationStateMachine, ConversationStage
from victor.agent.action_authorizer import ActionIntent, INTENT_BLOCKED_TOOLS
from victor.agent.prompt_builder import get_task_type_hint, SystemPromptBuilder
from victor.agent.search_router import SearchRoute, SearchType
from victor.framework.task import TaskComplexity, DEFAULT_BUDGETS
from victor.agent.stream_handler import StreamMetrics
from victor.agent.metrics_collector import MetricsCollectorConfig
from victor.agent.unified_task_tracker import TrackerTaskType, UnifiedTaskTracker
from victor.agent.prompt_requirement_extractor import extract_prompt_requirements

# Decomposed components - configs, strategies, functions
from victor.core.context import session_id as ctx_session_id, set_session_id
from victor.agent.conversation_controller import (
    ConversationConfig,
    ContextMetrics,
    CompactionStrategy,
)
from victor.agent.context_compactor import (
    TruncationStrategy,
    create_context_compactor,
    calculate_parallel_read_budget,
)
from victor.agent.context_manager import (
    ContextManager,
    ContextManagerConfig,
    create_context_manager,
)
from victor.agent.continuation_strategy import ContinuationStrategy
from victor.agent.tool_call_extractor import ExtractedToolCall
from victor.framework.rl.coordinator import get_rl_coordinator
from victor.agent.usage_analytics import AnalyticsConfig
from victor.agent.tool_sequence_tracker import create_sequence_tracker
from victor.agent.session_state_accessor import SessionStateAccessor
from victor.agent.session_state_manager import (
    SessionStateManager,
    create_session_state_manager,
)

# Recovery - enums and functions used at runtime
from victor.agent.recovery import RecoveryOutcome, FailureType, RecoveryAction
from victor.agent.vertical_context import VerticalContext, create_vertical_context
from victor.agent.vertical_integration_adapter import VerticalIntegrationAdapter
from victor.agent.protocols import RecoveryHandlerProtocol
from victor.agent.orchestrator_recovery import (
    create_recovery_integration,
    RecoveryAction as OrchestratorRecoveryAction,
)
from victor.agent.tool_output_formatter import (
    ToolOutputFormatterConfig,
    FormattingContext,
    create_tool_output_formatter,
)

# Pipeline - configs and results used at runtime
from victor.agent.tool_pipeline import ToolPipelineConfig, ToolCallResult
from victor.agent.streaming_controller import (
    StreamingControllerConfig,
    StreamingSession,
)
from victor.agent.task_analyzer import get_task_analyzer
from victor.agent.coordinators.system_prompt_coordinator import SystemPromptCoordinator
from victor.agent.tool_registrar import ToolRegistrarConfig
from victor.agent.provider_manager import ProviderManagerConfig, ProviderState

# Observability
# from victor.observability.event_bus import EventBus, EventCategory, VictorEvent  # DELETED
from victor.observability.integration import ObservabilityIntegration
from victor.agent.orchestrator_integration import IntegrationConfig

# Tool execution - functions and enums
from victor.agent.tool_selection import get_critical_tools
from victor.agent.tool_calling import ToolCallParseResult
from victor.agent.tool_executor import ValidationMode
from victor.agent.orchestrator_utils import (
    calculate_max_context_chars,
    infer_git_operation,
    get_tool_status_message,
)
from victor.agent.orchestrator_factory import OrchestratorFactory
from victor.agent.parallel_executor import create_parallel_executor
from victor.agent.response_completer import (
    ToolFailureContext,
    create_response_completer,
)
from victor.observability.analytics.logger import UsageLogger
from victor.observability.analytics.streaming_metrics import StreamingMetricsCollector
from victor.storage.cache.tool_cache import ToolCache
from victor.config.model_capabilities import ToolCallingMatrix
from victor.config.settings import Settings
from victor.context.project_context import ProjectContext
from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.registry import ProviderRegistry
from victor.core.errors import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ToolNotFoundError,
    ToolValidationError,
)
from victor.tools.base import CostTier, ToolRegistry
from victor.tools.code_executor_tool import CodeSandbox
from victor.tools.mcp_bridge_tool import get_mcp_tool_definitions
from victor.tools.plugin_registry import ToolPluginRegistry
from victor.tools.semantic_selector import SemanticToolSelector
from victor.tools.tool_names import ToolNames, TOOL_ALIASES
from victor.tools.alias_resolver import get_alias_resolver
from victor.tools.progressive_registry import get_progressive_registry
from victor.workflows.base import WorkflowRegistry
from victor.workflows.discovery import register_builtin_workflows

# Streaming submodule - extracted for testability
from victor.agent.streaming import (
    CoordinatorConfig,
    ContinuationHandler,
    ContinuationResult,
    IntentClassificationHandler,
    IntentClassificationResult,
    IterationCoordinator,
    StreamingChatContext,
    StreamingChatHandler,
    ToolExecutionHandler,
    ToolExecutionResult,
    TrackingState,
    apply_tracking_state_updates,
    create_continuation_handler,
    create_coordinator,
    create_intent_classification_handler,
    create_stream_context,
    create_tool_execution_handler,
    create_tracking_state,
)

logger = logging.getLogger(__name__)

# Tools with progressive parameters - different params = progress, not a loop
# Format: tool_name -> list of param names that indicate progress
# NOTE: Includes both canonical short names and legacy names for LLM compatibility
# NOTE: These values are registered with ProgressiveToolsRegistry for extensibility
_PROGRESSIVE_TOOLS_CONFIG = {
    # Canonical short names
    "read": ["path", "offset", "limit"],
    "grep": ["query", "directory"],
    "search": ["query", "directory"],
    "ls": ["path", "recursive"],
    "shell": ["command"],
    "git": ["operation", "files", "branch"],
    "http": ["url", "method"],
    "web": ["query"],
    "summarize": ["query"],
    "fetch": ["url"],
    # Legacy names (backward compatibility - LLMs may still use these)
    "read_file": ["path", "offset", "limit"],
    "code_search": ["query", "directory"],
    "semantic_code_search": ["query", "directory"],
    "list_directory": ["path", "recursive"],
    "execute_bash": ["command"],
    "http_request": ["url", "method"],
    "web_search": ["query"],
    "web_summarize": ["query"],
    "web_fetch": ["url"],
}


def _register_progressive_tools() -> None:
    """Register all progressive tools with the ProgressiveToolsRegistry.

    This function is idempotent and can be called multiple times safely.
    Tools are only registered if not already present in the registry.
    """
    registry = get_progressive_registry()
    for tool_name, params in _PROGRESSIVE_TOOLS_CONFIG.items():
        if not registry.is_progressive(tool_name):
            # Convert list of param names to dict format expected by registry
            progressive_params = dict.fromkeys(params, "any")
            registry.register(tool_name, progressive_params)


def _ensure_progressive_tools_registered() -> None:
    """Ensure progressive tools are registered in the registry.

    This is called lazily on first access to handle cases where
    the registry singleton was reset (e.g., during testing).
    """
    registry = get_progressive_registry()
    # Quick check if already registered by checking for one known tool
    if not registry.is_progressive("read"):
        _register_progressive_tools()


# Register tools at module load time
_register_progressive_tools()


class _ProgressiveToolsProxy:
    """Proxy class that delegates to ProgressiveToolsRegistry for backward compatibility.

    This allows existing code using PROGRESSIVE_TOOLS dict-like access to continue working
    while the actual data is managed by the registry.

    The proxy ensures tools are re-registered if the registry was reset (e.g., during tests).
    """

    def __getitem__(self, key: str) -> List[str]:
        _ensure_progressive_tools_registered()
        registry = get_progressive_registry()
        config = registry.get_config(key)
        if config is None:
            raise KeyError(key)
        return list(config.progressive_params.keys())

    def __contains__(self, key: str) -> bool:
        _ensure_progressive_tools_registered()
        registry = get_progressive_registry()
        return registry.is_progressive(key)

    def get(self, key: str, default: Any = None) -> Any:
        _ensure_progressive_tools_registered()
        registry = get_progressive_registry()
        config = registry.get_config(key)
        if config is None:
            return default
        return list(config.progressive_params.keys())

    def keys(self) -> Set[str]:
        _ensure_progressive_tools_registered()
        registry = get_progressive_registry()
        return registry.list_progressive_tools()

    def items(self):
        _ensure_progressive_tools_registered()
        registry = get_progressive_registry()
        for tool_name in registry.list_progressive_tools():
            config = registry.get_config(tool_name)
            if config:
                yield tool_name, list(config.progressive_params.keys())

    def __iter__(self):
        _ensure_progressive_tools_registered()
        registry = get_progressive_registry()
        return iter(registry.list_progressive_tools())


# Backward-compatible constant that delegates to the registry
PROGRESSIVE_TOOLS = _ProgressiveToolsProxy()

# Build set of all known tool names (canonical + aliases) for detection
_ALL_TOOL_NAMES: Set[str] = set()
for attr in dir(ToolNames):
    if not attr.startswith("_"):
        val = getattr(ToolNames, attr)
        if isinstance(val, str):
            _ALL_TOOL_NAMES.add(val)
_ALL_TOOL_NAMES.update(TOOL_ALIASES.keys())


def _detect_mentioned_tools(text: str) -> List[str]:
    """Detect tool names mentioned in text that model said it would call.

    Looks for patterns like:
    - "let me call read()"
    - "I'll use web_search to"
    - "calling the ls tool"
    - "execute grep"

    Args:
        text: Model response text

    Returns:
        List of mentioned tool names (canonical form)
    """
    import re

    mentioned: List[str] = []
    text_lower = text.lower()

    # Look for tool names followed by common patterns
    for tool_name in _ALL_TOOL_NAMES:
        # Match patterns like: call read, use read, execute read, run read
        # Also: read() or read( with args
        patterns = [
            rf"\b(?:call|use|execute|run|invoke|perform)\s+{re.escape(tool_name)}\b",
            rf"\b{re.escape(tool_name)}\s*\(",  # tool_name( or tool_name (
            rf"\bthe\s+{re.escape(tool_name)}\s+tool\b",  # "the read tool"
        ]
        for pattern in patterns:
            if re.search(pattern, text_lower):
                # Resolve to canonical name
                canonical = TOOL_ALIASES.get(tool_name, tool_name)
                if canonical not in mentioned:
                    mentioned.append(canonical)
                break

    return mentioned


class AgentOrchestrator(ModeAwareMixin, CapabilityRegistryMixin):
    """Orchestrates agent interactions, tool execution, and provider communication.

    Uses ModeAwareMixin for consistent mode controller access (via self.is_build_mode,
    self.mode_controller, self.exploration_multiplier, etc.).

    Uses CapabilityRegistryMixin for explicit capability discovery, replacing hasattr
    duck-typing with type-safe protocol conformance. See capability_registry.py.
    """

    @staticmethod
    def _calculate_max_context_chars(
        settings: "Settings",
        provider: "BaseProvider",
        model: str,
    ) -> int:
        """Calculate maximum context size in characters for a model.

        Delegates to orchestrator_utils.calculate_max_context_chars.
        """
        return calculate_max_context_chars(settings, provider, model)

    def _initialize_provider_runtime(self) -> None:
        """Initialize provider runtime boundaries with lazy coordinator loading."""
        from victor.agent.runtime.provider_runtime import (
            create_provider_runtime_components,
        )

        self._provider_runtime = create_provider_runtime_components(
            factory=self._factory,
            settings=self.settings,
            provider_manager=self._provider_manager,
        )
        # Keep legacy attributes for compatibility while deferring heavy coordinator init.
        self._provider_coordinator = self._provider_runtime.provider_coordinator
        self._provider_switch_coordinator = self._provider_runtime.provider_switch_coordinator

    def _initialize_memory_runtime(self) -> None:
        """Initialize memory/session runtime boundaries."""
        from victor.agent.runtime.memory_runtime import create_memory_runtime_components

        self._memory_runtime = create_memory_runtime_components(
            factory=self._factory,
            provider_name=self.provider_name,
            native_tool_calls=self.tool_calling_caps.native_tool_calls,
        )
        self.memory_manager = self._memory_runtime.memory_manager
        self._memory_session_id = self._memory_runtime.memory_session_id

        # Initialize LanceDB embedding store for efficient semantic retrieval if memory enabled.
        if self.memory_manager and getattr(self.settings, "conversation_embeddings_enabled", True):
            try:
                self._init_conversation_embedding_store()
            except ImportError as embed_err:
                logger.debug(f"ConversationEmbeddingStore dependencies not available: {embed_err}")
            except (OSError, IOError) as embed_err:
                logger.warning(
                    f"Failed to initialize ConversationEmbeddingStore (I/O error): {embed_err}"
                )
            except (ValueError, TypeError) as embed_err:
                logger.warning(
                    f"Failed to initialize ConversationEmbeddingStore (config error): {embed_err}"
                )

    def _initialize_metrics_runtime(self) -> None:
        """Initialize metrics/analytics runtime boundaries."""
        from victor.agent.runtime.metrics_runtime import (
            create_metrics_runtime_components,
        )

        self._metrics_runtime = create_metrics_runtime_components(
            factory=self._factory,
            provider=self.provider,
            model=self.model,
            debug_logger=self.debug_logger,
            cumulative_token_usage=self._cumulative_token_usage,
            tool_cost_lookup=lambda name: (
                self.tools.get_tool_cost(name) if hasattr(self, "tools") else CostTier.FREE
            ),
        )
        self.usage_logger = self._metrics_runtime.usage_logger

        # Wrap with trace enricher for GEPA ASI when prompt optimization enabled
        try:
            from victor.observability.analytics.trace_enrichment import (
                create_trace_enricher,
            )

            po_cfg = getattr(self.settings, "prompt_optimization", None)
            gepa_cfg = po_cfg.gepa if po_cfg and po_cfg.enabled else None
            self.usage_logger = create_trace_enricher(self.usage_logger, gepa_settings=gepa_cfg)
        except Exception as exc:
            logger.debug("Trace enrichment unavailable: %s", exc)

        self.streaming_metrics_collector = self._metrics_runtime.streaming_metrics_collector
        self._metrics_collector = self._metrics_runtime.metrics_collector
        self._session_cost_tracker = self._metrics_runtime.session_cost_tracker
        self._metrics_coordinator = self._metrics_runtime.metrics_coordinator

        self.usage_logger.log_event(
            "session_start",
            {"model": self.model, "provider": self.provider.__class__.__name__},
        )
        if self.streaming_metrics_collector:
            logger.info("StreamingMetricsCollector initialized via runtime boundary")

    def _initialize_workflow_runtime(self) -> None:
        """Initialize workflow runtime boundaries with lazy registry loading."""
        from victor.agent.runtime.workflow_runtime import (
            create_workflow_runtime_components,
        )

        self._workflow_runtime = create_workflow_runtime_components(factory=self._factory)
        self._workflow_registry = self._workflow_runtime.workflow_registry

    def _initialize_coordination_runtime(self) -> None:
        """Initialize coordination runtime boundaries with lazy components."""
        from victor.agent.runtime.coordination_runtime import (
            create_coordination_runtime_components,
        )

        self._coordination_runtime = create_coordination_runtime_components(factory=self._factory)
        self._recovery_coordinator = self._coordination_runtime.recovery_coordinator
        self._chunk_generator = self._coordination_runtime.chunk_generator
        self._tool_planner = self._coordination_runtime.tool_planner
        self._task_coordinator = self._coordination_runtime.task_coordinator

    def _initialize_resilience_runtime(self, *, context_compactor: Any) -> None:
        """Initialize resilience runtime boundaries with lazy recovery components."""
        from victor.agent.runtime.resilience_runtime import (
            create_resilience_runtime_components,
        )

        self._resilience_runtime = create_resilience_runtime_components(
            factory=self._factory,
            context_compactor=context_compactor,
        )
        self._recovery_handler = self._resilience_runtime.recovery_handler
        self._recovery_integration = self._resilience_runtime.recovery_integration

    def _initialize_interaction_runtime(self) -> None:
        """Initialize interaction runtime boundaries for chat/tool/session coordinators."""
        from victor.agent.runtime.interaction_runtime import (
            create_interaction_runtime_components,
        )

        self._interaction_runtime = create_interaction_runtime_components(
            orchestrator=self,
            factory=self._factory,
            tool_pipeline=self._tool_pipeline,
            tool_registry=self.tools,
            tool_selector=(self.tool_selector if hasattr(self, "tool_selector") else None),
            tool_access_controller=getattr(self, "_tool_access_controller", None),
            mode_controller=(self.mode_controller if hasattr(self, "mode_controller") else None),
            session_state_manager=self._session_state,
            lifecycle_manager=self._lifecycle_manager,
            memory_manager=self.memory_manager,
            checkpoint_manager=self._checkpoint_manager,
            cost_tracker=self._session_cost_tracker,
        )
        self._chat_coordinator = self._interaction_runtime.chat_coordinator
        self._tool_coordinator = self._interaction_runtime.tool_coordinator
        self._session_coordinator = self._interaction_runtime.session_coordinator

    def _initialize_credit_runtime(self) -> None:
        """Initialize credit assignment runtime (FEP-0001 Phase 3).

        Creates CreditTrackingService and attaches it to the ToolPipeline.
        Gated by settings.credit_assignment.enabled (default: False).
        """
        self._credit_tracking_service = None
        ca_settings = getattr(self.settings, "credit_assignment", None)
        if ca_settings is None or not getattr(ca_settings, "enabled", False):
            return

        try:
            from victor.framework.rl.credit_tracking_service import CreditTrackingService

            obs_bus = getattr(self, "_observability_bus", None)
            self._credit_tracking_service = CreditTrackingService.from_settings(
                self.settings, observability_bus=obs_bus
            )

            # Attach to ToolPipeline
            if hasattr(self, "_tool_pipeline") and self._tool_pipeline is not None:
                self._tool_pipeline._credit_tracking_service = self._credit_tracking_service

            logger.info("Credit tracking service initialized")
        except Exception as e:
            logger.warning("Failed to initialize credit tracking: %s", e)

    def _initialize_services(self) -> None:
        """Initialize service layer by registering coordinators and bootstrapping services.

        This implements the Strangler Fig pattern:
        1. Register coordinators in the container (for service dependencies)
        2. Bootstrap services using the registered coordinators
        3. Resolve service instances from the container

        When USE_SERVICE_LAYER flag is enabled, orchestrator methods delegate to
        services instead of calling coordinators directly. When disabled, services
        are None and methods fall through to coordinators as before.
        """
        if not hasattr(self, "_container"):
            logger.warning("ServiceContainer not available — services will be None")
            self._chat_service = None
            self._tool_service = None
            self._session_service = None
            self._context_service = None
            self._provider_service = None
            self._recovery_service = None
            return

        # Register coordinators in container for service dependencies
        self._register_coordinators_for_services()

        # Bootstrap services using the registered coordinators
        self._bootstrap_service_layer()

        # Resolve service instances from container
        from victor.agent.services.protocols import (
            ChatServiceProtocol,
            ContextServiceProtocol,
            ProviderServiceProtocol,
            RecoveryServiceProtocol,
            SessionServiceProtocol,
            ToolServiceProtocol,
        )

        self._chat_service = self._container.get_optional(ChatServiceProtocol)
        self._tool_service = self._container.get_optional(ToolServiceProtocol)
        self._session_service = self._container.get_optional(SessionServiceProtocol)
        self._context_service = self._container.get_optional(ContextServiceProtocol)
        self._provider_service = self._container.get_optional(ProviderServiceProtocol)
        self._recovery_service = self._container.get_optional(RecoveryServiceProtocol)

        logger.info(
            "Service layer initialized: chat=%s, tool=%s, session=%s, "
            "context=%s, provider=%s, recovery=%s",
            self._chat_service is not None,
            self._tool_service is not None,
            self._session_service is not None,
            self._context_service is not None,
            self._provider_service is not None,
            self._recovery_service is not None,
        )

    def _register_coordinators_for_services(self) -> None:
        """Register coordinators in the container for service layer dependencies.

        Services need access to coordinators through the container. This method
        registers the coordinator protocols so services can resolve them.
        """
        from victor.agent.protocols import (
            ConversationControllerProtocol,
            StreamingCoordinatorProtocol,
        )
        from victor.core.container import ServiceLifetime

        # Register conversation controller if not already registered
        if not self._container.is_registered(ConversationControllerProtocol):
            self._container.register(
                ConversationControllerProtocol,
                lambda c: self._conversation_controller,
                ServiceLifetime.SINGLETON,
            )

        # Register streaming coordinator if not already registered
        if not self._container.is_registered(StreamingCoordinatorProtocol):
            self._container.register(
                StreamingCoordinatorProtocol,
                lambda c: self._streaming_controller,
                ServiceLifetime.SINGLETON,
            )

        logger.debug("Registered coordinators in container for service layer")

    def _bootstrap_service_layer(self) -> None:
        """Bootstrap the service layer using registered coordinators.

        This creates and registers all services (ChatService, ToolService, etc.)
        using the coordinators that were registered in the container.
        """
        from victor.core.bootstrap_services import bootstrap_new_services

        # Bootstrap services with the registered coordinators
        bootstrap_new_services(
            self._container,
            conversation_controller=self._conversation_controller,
            streaming_coordinator=self._streaming_controller,
        )

        logger.debug("Bootstrapped service layer")

    def _create_execution_context(self) -> "ExecutionContext":
        """Create an ExecutionContext carrying all runtime dependencies.

        Replaces scattered get_global_manager() / get_instance() calls
        with explicit context passing. Created after services are
        bootstrapped so all dependencies are available.

        Returns:
            ExecutionContext with settings, container, state manager,
            and lazy service accessor.
        """
        from victor.runtime.context import ExecutionContext

        session_id = getattr(self, "_memory_session_id", "") or ""
        return ExecutionContext.create(
            settings=self.settings,
            container=self._container,
            session_id=session_id,
        )

    def __init__(
        self,
        settings: Settings,
        provider: BaseProvider,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        console: Optional[Console] = None,
        tool_selection: Optional[Dict[str, Any]] = None,
        thinking: bool = False,
        provider_name: Optional[str] = None,
        profile_name: Optional[str] = None,
    ):
        """Initialize orchestrator.

        Args:
            settings: The application settings.
            provider: LLM provider instance
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            console: Rich console for output
            tool_selection: Optional tool selection configuration (base_threshold, base_max_tools)
            thinking: Enable extended thinking mode (Claude models only)
            provider_name: Optional provider label from profile (e.g., lmstudio, vllm) to disambiguate OpenAI-compatible providers
            profile_name: Optional profile name (e.g., "groq-fast", "claude-sonnet") for session tracking
        """
        # Store profile name for session tracking
        self._profile_name = profile_name
        # Track active session ID for parallel session support
        self.active_session_id: Optional[str] = None
        # Bootstrap DI container - ensures all services are available
        # This is idempotent and will only bootstrap if not already done
        self._container = ensure_bootstrapped(settings)

        # Create factory for component initialization (CRITICAL-001)
        # This enables DI-aware creation with fallback and reduces __init__ complexity
        self._factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            provider_name=provider_name,
            profile_name=profile_name,
            tool_selection=tool_selection,
            thinking=thinking,
        )
        self._factory._container = self._container  # Share container

        self.settings = settings
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.console = console or Console()
        self.tool_selection = tool_selection or {}
        self.thinking = thinking

        # Tool calling matrix for managing provider capabilities (via factory)
        self.tool_calling_models, self.tool_capabilities = (
            self._factory.create_tool_calling_matrix()
        )

        # Initialize ProviderManager with tool adapter (via factory)
        (
            self._provider_manager,
            self.provider,
            self.model,
            self.provider_name,
            self.tool_adapter,
            self.tool_calling_caps,
        ) = self._factory.create_provider_manager_with_adapter(provider, model, provider_name)

        # Provider runtime boundary: coordinator services are created lazily on first use.
        self._initialize_provider_runtime()

        # Response sanitizer for cleaning model output (via factory - DI with fallback)
        self.sanitizer = self._factory.create_sanitizer()

        # System prompt builder with vertical prompt contributors (via factory)
        self.prompt_builder = self._factory.create_system_prompt_builder(
            provider_name=self.provider_name,
            model=model,
            tool_adapter=self.tool_adapter,
            tool_calling_caps=self.tool_calling_caps,
        )

        # UnifiedPromptPipeline: consolidated prompt assembly (replaces
        # PromptComposer + SystemPromptCoordinator + frozen-prompt glue).
        try:
            from victor.agent.content_registry import create_default_registry
            from victor.agent.optimization_injector import OptimizationInjector
            from victor.agent.prompt_pipeline import UnifiedPromptPipeline

            self._optimization_injector = OptimizationInjector()
            self._prompt_pipeline = UnifiedPromptPipeline(
                provider=self.provider,
                builder=self.prompt_builder,
                registry=create_default_registry(),
                optimizer=self._optimization_injector,
                get_context_window=self._get_model_context_window,
            )
            # Backward compat alias — old code referencing _prompt_composer
            self._prompt_composer = self._prompt_pipeline
        except Exception as e:
            logger.debug("UnifiedPromptPipeline unavailable: %s", e)
            self._optimization_injector = None
            self._prompt_pipeline = None
            self._prompt_composer = None

        # Load project context from .victor/init.md (via factory - DI with fallback)
        self.project_context = self._factory.create_project_context()

        # Build system prompt using adapter hints
        base_system_prompt = self._build_system_prompt_with_adapter()

        # Inject project context if available
        if self.project_context.content:
            self._system_prompt = (
                base_system_prompt + "\n\n" + self.project_context.get_system_prompt_addition()
            )
            logger.info(f"Loaded project context from {self.project_context.context_file}")
        else:
            self._system_prompt = base_system_prompt

        self._system_added = False

        # Streaming loop tracking counters (previously injected via hasattr in pipeline.py)
        self._continuation_prompts: int = 0
        self._asking_input_prompts: int = 0
        self._consecutive_blocked_attempts: int = 0
        self._cumulative_prompt_interventions: int = 0

        # Initialize tool call budget (via factory) - uses adapter recommendations with settings override
        self.tool_budget = self._factory.initialize_tool_budget(self.tool_calling_caps)

        # Initialize SessionStateManager for consolidated execution state tracking (TD-002)
        # Replaces scattered state variables: tool_calls_used, observed_files, executed_tools,
        # failed_tool_signatures, _read_files_session, _required_files, _required_outputs, etc.
        self._session_state = create_session_state_manager(tool_budget=self.tool_budget)
        self._session_accessor = SessionStateAccessor(self._session_state)

        # Gap implementations: Complexity classifier, action authorizer, search router (via factory)
        self.task_classifier = self._factory.create_complexity_classifier()
        self.intent_detector = self._factory.create_action_authorizer()
        self.search_router = self._factory.create_search_router()

        # Presentation adapter for icon/emoji rendering (via factory, DI)
        # Decouples agent layer from direct UI dependencies
        self._presentation = self._factory.create_presentation_adapter()

        # Task Completion Detection: Signal-based completion detection
        # Uses explicit markers (_DONE_, _TASK_DONE_, _SUMMARY_) for deterministic completion
        from victor.agent.task_completion import TaskCompletionDetector

        self._task_completion_detector = TaskCompletionDetector()
        logger.info("TaskCompletionDetector initialized (signal-based completion)")

        # Context reminder manager for intelligent system message injection (via factory, DI)
        # Reduces token waste by consolidating reminders and only injecting when context changes
        self.reminder_manager = self._factory.create_reminder_manager(
            provider=self.provider_name,
            task_complexity="medium",  # Will be updated per-task
            tool_budget=self.tool_budget,
        )

        # Debug logger for incremental output and conversation tracking (via factory)
        self.debug_logger = self._factory.create_debug_logger_configured()

        # Metrics/analytics runtime boundary with lazy collector/coordinator loading.
        self._initialize_metrics_runtime()

        # CallbackCoordinator: centralized callback delegation for tool/streaming events.
        self._callback_coordinator = self._build_callback_coordinator()

        # Cancellation support for streaming
        self._cancel_event: Optional[asyncio.Event] = None
        self._is_streaming = False

        # Background task tracking for graceful shutdown
        self._background_tasks: set[asyncio.Task] = set()
        self._bg_task_lock = threading.Lock()

        # Tool error dedup (bounded to prevent memory leak in long sessions)
        self._shown_tool_errors: set = set()
        self._tool_context_cache: Optional[dict] = None

        # Workflow runtime boundary (lazy registry + default workflow registration).
        self._initialize_workflow_runtime()

        # Conversation history (via factory) - MessageHistory for better encapsulation
        self.conversation = self._factory.create_message_history(self._system_prompt)

        # Memory/session runtime boundary with embedding-store initialization.
        self._initialize_memory_runtime()

        # Conversation state machine for intelligent stage detection
        self.conversation_state = self._factory.create_conversation_state_machine()

        # Intent classifier for semantic continuation/completion detection
        self.intent_classifier = self._factory.create_intent_classifier()

        # Intelligent pipeline integration (lazy initialization)
        self._intelligent_integration: Optional["OrchestratorIntegration"] = None
        self._intelligent_integration_config = self._factory.create_integration_config()
        self._intelligent_pipeline_enabled = getattr(settings, "intelligent_pipeline_enabled", True)

        # Sub-agent orchestration (lazy initialization)
        self._subagent_orchestrator, self._subagent_orchestration_enabled = (
            self._factory.setup_subagent_orchestration()
        )

        # Component assembly phases (tools → conversation → intelligence)
        from victor.agent.runtime.component_assembler import ComponentAssembler

        ComponentAssembler.assemble_tools(self, provider, model)
        ComponentAssembler.assemble_conversation(self, provider, model)
        ComponentAssembler.assemble_intelligence(self)

        # Post-factory component preparation and facade assembly
        from victor.agent.runtime.bootstrapper import AgentRuntimeBootstrapper

        AgentRuntimeBootstrapper.prepare_components(self, settings)
        AgentRuntimeBootstrapper.finalize(self)

        # Create ExecutionContext — explicit context object replacing global singletons.
        # Available after services are bootstrapped; passed to coordinators and workflows.
        self._execution_context = self._create_execution_context()

        # Cache optimization flags once (provider is fully initialized)
        self._kv_opt_cached: Optional[bool] = None
        self._cache_opt_cached: Optional[bool] = None
        self._last_sorted_tool_names: Optional[frozenset] = None
        self._last_sorted_tools: Optional[list] = None
        self._session_semantic_tools: Optional[list] = None
        self._compute_cache_flags()

    # =====================================================================
    # Callbacks for decomposed components — delegated to CallbackCoordinator
    # =====================================================================

    def _build_callback_coordinator(self) -> Any:
        """Lazily construct the CallbackCoordinator."""
        from victor.agent.callback_coordinator import CallbackCoordinator

        return CallbackCoordinator(
            metrics_coordinator=self._metrics_coordinator,
            get_tool_coordinator=lambda: self._tool_coordinator,
            get_observability=lambda: getattr(self, "_observability", None),
            get_pipeline_calls_used=lambda: (
                self._tool_pipeline.calls_used if hasattr(self, "_tool_pipeline") else 0
            ),
            get_usage_analytics=lambda: (
                self._usage_analytics
                if hasattr(self, "_usage_analytics") and self._usage_analytics
                else None
            ),
            get_rl_coordinator=lambda: self._rl_coordinator,
            get_vertical_context=lambda: self._vertical_context,
        )

    def _on_tool_start_callback(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Callback when tool execution starts (from ToolPipeline)."""
        self._callback_coordinator.on_tool_start(tool_name, arguments)

    def _on_tool_complete_callback(self, result: ToolCallResult) -> None:
        """Callback when tool execution completes (from ToolPipeline)."""
        nudge_flag = [getattr(self, "_all_files_read_nudge_sent", False)]
        self._callback_coordinator.on_tool_complete(
            result,
            read_files_session=self._read_files_session,
            required_files=self._required_files,
            required_outputs=self._required_outputs,
            nudge_sent_flag=nudge_flag,
            add_message=self.add_message,
        )
        self._all_files_read_nudge_sent = nudge_flag[0]

    def _on_streaming_session_complete(self, session: StreamingSession) -> None:
        """Callback when streaming session completes (from StreamingController)."""
        self._callback_coordinator.on_streaming_session_complete(session)

    @property
    def workflow_registry(self) -> Any:
        """Get workflow registry, materializing lazy runtime on first access."""
        registry = getattr(self, "_workflow_registry", None)
        if hasattr(registry, "get_instance"):
            resolved = registry.get_instance()
            self._workflow_registry = resolved
            return resolved
        return registry

    @workflow_registry.setter
    def workflow_registry(self, value: Any) -> None:
        """Set workflow registry (supports test overrides)."""
        self._workflow_registry = value

    @property
    def observability(self) -> Optional[ObservabilityIntegration]:
        """Get the observability integration component.

        Returns:
            ObservabilityIntegration instance for event bus access, or None if disabled
        """
        return getattr(self, "_observability", None)

    @observability.setter
    def observability(self, value: Optional[ObservabilityIntegration]) -> None:
        """Set the observability integration component.

        This allows FrameworkShim to inject an externally-configured
        ObservabilityIntegration instance for unified event handling.

        Args:
            value: ObservabilityIntegration instance or None to disable
        """
        self._observability = value

    def set_observability(self, value: Optional[ObservabilityIntegration]) -> None:
        """Set observability integration via explicit port method.

        Args:
            value: ObservabilityIntegration instance or None.
        """
        self.observability = value

    @property
    def container(self) -> Any:
        """Get the service container via a public property."""
        return self._container

    def get_service_container(self) -> Any:
        """Get the DI service container via explicit port method."""
        return self._container

    def get_capability_config_scope_key(self) -> Optional[str]:
        """Get stable scope key for framework capability-config service storage."""
        if self.active_session_id:
            normalized_active = str(self.active_session_id).strip()
            if normalized_active:
                return normalized_active

        memory_session_id = getattr(self, "_memory_session_id", None)
        if memory_session_id:
            normalized_memory = str(memory_session_id).strip()
            if normalized_memory:
                return normalized_memory

        return f"orchestrator:{id(self)}"

    def get_capability_loader(self) -> Optional[Any]:
        """Get the cached framework CapabilityLoader if available."""
        return getattr(self, "_capability_loader", None)

    def set_capability_loader(self, loader: Any) -> None:
        """Set the framework CapabilityLoader instance."""
        self._capability_loader = loader

    def get_or_create_capability_loader(self) -> Any:
        """Get or lazily create the framework CapabilityLoader."""
        loader = self.get_capability_loader()
        if loader is None:
            from victor.framework.capability_loader import CapabilityLoader

            loader = CapabilityLoader()
            self.set_capability_loader(loader)
        return loader

    def get_vertical_context(self) -> VerticalContext:
        """Get vertical context via explicit protocol getter."""
        return self._vertical_context

    @property
    def lsp(self) -> Optional[Any]:
        """Get the LSP capability for code intelligence.

        Returns:
            LSPCapability instance or None if not configured.
        """
        return getattr(self, "_lsp", None)

    def set_lsp(self, lsp_capability: Any) -> None:
        """Set the LSP capability (LSPServiceProtocol/LSPPoolProtocol).

        This enables framework-level language intelligence for all verticals.

        Args:
            lsp_capability: LSPCapability instance
        """
        self._lsp = lsp_capability
        logger.debug("LSP capability registered with orchestrator")

    def get_team_suggestions(
        self,
        task_type: str,
        complexity: str,
    ) -> Any:
        """Get team and workflow suggestions for a task.

        Queries the ModeWorkflowTeamCoordinator to get recommendations for
        teams and workflows based on task classification and current mode.

        Args:
            task_type: Classified task type (e.g., "feature", "bugfix", "refactor")
            complexity: Complexity level (e.g., "low", "medium", "high", "extreme")

        Returns:
            CoordinationSuggestion with team and workflow recommendations
        """
        from victor.agent.mode_controller import MODE_CONFIGS

        # Get current mode
        current_mode = "build"  # Default
        if self.mode_controller:
            current_mode = self.mode_controller.current_mode.value

        return self.coordination.suggest_for_task(
            task_type=task_type,
            complexity=complexity,
            mode=current_mode,
        )

    # =========================================================================
    # Vertical Protocol Methods
    # These implement OrchestratorVerticalProtocol for proper vertical integration
    # =========================================================================

    def set_vertical_context(self, context: VerticalContext) -> None:
        """Set the vertical context (OrchestratorVerticalProtocol).

        This replaces direct assignment to _vertical_context and provides
        a proper API for framework integration.

        Args:
            context: VerticalContext to set
        """
        self._vertical_context = context

        # Sync coordinator with new vertical context (if already initialized)
        if self._mode_workflow_team_coordinator is not None:
            self._mode_workflow_team_coordinator.set_vertical_context(context)
            logger.debug(f"Coordinator synced with vertical context: {context.vertical_name}")

        # Sync tool selector with vertical context for vertical-specific tool selection (DIP)
        if hasattr(self, "tool_selector") and self.tool_selector is not None:
            self.tool_selector.set_vertical_context(context)
            if context.has_tool_selection_strategy:
                logger.debug(
                    f"Tool selector synced with vertical tool selection strategy: "
                    f"{context.vertical_name}"
                )

        # Sync middleware chain with vertical context for vertical-aware middleware (DIP)
        if hasattr(self, "_middleware_chain") and self._middleware_chain is not None:
            self._middleware_chain.set_vertical_context(context)
            logger.debug(f"Middleware chain synced with vertical context: {context.vertical_name}")

        logger.debug(f"Vertical context set: {context.vertical_name}")

    def set_tiered_tool_config(self, config: Any) -> None:
        """Set tiered tool configuration (Phase 1: Gap fix).

        Applies tiered tool config from vertical to:
        1. VerticalContext for storage
        2. ToolAccessController.VerticalLayer for access filtering

        Args:
            config: TieredToolConfig from the active vertical
        """
        # Store in vertical context
        if self._vertical_context is not None:
            self._vertical_context.apply_tiered_config(config)

        # Apply to tool access controller
        if self._tool_access_controller is not None:
            self._tool_access_controller.set_tiered_config(config)
            logger.debug("Tiered config applied to ToolAccessController")

        logger.debug("Tiered tool config set")

    def set_workspace(self, workspace_dir: Path) -> None:
        """Set the workspace directory for task execution.

        Updates the global project root and orchestrator's project context
        to work in the specified directory. This is essential for benchmark
        evaluations where each task operates in a different workspace.

        Args:
            workspace_dir: Path to the workspace directory
        """
        from victor.config.settings import set_project_root
        from victor.context.project_context import ProjectContext

        # Update global project root
        set_project_root(workspace_dir)
        logger.info(f"Project root set to: {workspace_dir}")

        # Create new project context for this workspace
        self.project_context = ProjectContext(root_path=str(workspace_dir))
        self.project_context.load()

        # Set RL repo context for per-repo isolation
        try:
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            coordinator.set_repo_context(workspace_dir.name)
        except Exception as exc:
            logger.debug("RL coordinator repo context unavailable: %s", exc)

        # Rebuild system prompt with new workspace context (replaces old init.md)
        # Workspace switch is a session reset — unfreeze and re-sample GEPA sections
        if getattr(self, "_prompt_pipeline", None):
            self._prompt_pipeline.unfreeze()
        self._system_prompt_frozen = False

        base_prompt = self._build_system_prompt_with_adapter()
        if self.project_context.content:
            self._system_prompt = (
                base_prompt + "\n\n" + self.project_context.get_system_prompt_addition()
            )
            logger.info(f"Loaded project context from {self.project_context.context_file}")
        else:
            self._system_prompt = base_prompt

        if self._kv_optimization_enabled:
            self._system_prompt_frozen = True
        if self._cache_optimization_enabled:
            self._session_tools = None  # Re-lock tools for new workspace

        # Clear conversation history to remove old workspace context
        # This ensures the model doesn't see init.md from the previous project
        if hasattr(self, "reset_conversation"):
            self.reset_conversation()

    def _apply_vertical_tools(self, tools: Set[str]) -> None:
        """Apply enabled tools to vertical context and access controller.

        Internal helper called by set_enabled_tools. Separated to avoid
        duplication and maintain single method for protocol compliance.

        Args:
            tools: Set of tool names to enable
        """
        self._vertical_context.apply_enabled_tools(tools)
        logger.debug(f"Applied {len(tools)} tools to vertical context")

    async def save_checkpoint(
        self,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Save a manual checkpoint of the current conversation state.

        Delegates to service layer when enabled, otherwise SessionCoordinator.

        Args:
            description: Human-readable description for the checkpoint
            tags: Optional tags for categorization

        Returns:
            Checkpoint ID if saved, None if checkpointing is disabled
        """
        if self._session_service:
            return await self._session_service.save_checkpoint(description, tags)
        return await self._session_coordinator.save_checkpoint(description, tags)

    async def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore conversation state from a checkpoint.

        Delegates to service layer when enabled, otherwise SessionCoordinator.

        Args:
            checkpoint_id: ID of checkpoint to restore

        Returns:
            True if restored successfully, False otherwise
        """
        if self._session_service:
            return await self._session_service.restore_checkpoint(checkpoint_id)
        return await self._session_coordinator.restore_checkpoint(checkpoint_id)

    async def maybe_auto_checkpoint(self) -> Optional[str]:
        """Trigger auto-checkpoint if interval threshold is met.

        Delegates to SessionCoordinator.

        Returns:
            Checkpoint ID if auto-checkpoint was created, None otherwise
        """
        return await self._session_coordinator.maybe_auto_checkpoint()

    async def _prepare_intelligent_request(
        self, task: str, task_type: str
    ) -> Optional[Dict[str, Any]]:
        """Pre-request hook for intelligent pipeline integration.

        Delegates to OrchestratorIntegration.prepare_intelligent_request().

        Args:
            task: The user's task/query
            task_type: Detected task type (analysis, edit, etc.)

        Returns:
            Dictionary with recommendations, or None if pipeline disabled
        """
        integration = self.intelligent_integration
        if not integration:
            return None

        return await integration.prepare_intelligent_request(
            task=task,
            task_type=task_type,
            conversation_state=self.conversation_state,
            unified_tracker=self.unified_tracker,
        )

    async def _validate_intelligent_response(
        self,
        response: str,
        query: str,
        tool_calls: int,
        task_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Post-response hook for intelligent pipeline integration.

        Delegates to OrchestratorIntegration.validate_intelligent_response().

        Args:
            response: The model's response content
            query: Original user query
            tool_calls: Number of tool calls made so far
            task_type: Task type for context

        Returns:
            Dictionary with quality/grounding scores, or None if pipeline disabled
        """
        integration = self.intelligent_integration
        if not integration:
            return None

        return await integration.validate_intelligent_response(
            response=response,
            query=query,
            tool_calls=tool_calls,
            task_type=task_type,
        )

    def _record_intelligent_outcome(
        self,
        success: bool,
        quality_score: float = 0.5,
        user_satisfied: bool = True,
        completed: bool = True,
    ) -> None:
        """Record outcome for Q-learning feedback.

        Delegates to OrchestratorIntegration.record_intelligent_outcome().

        Args:
            success: Whether the task was completed successfully
            quality_score: Final quality score (0.0-1.0)
            user_satisfied: Whether user seemed satisfied
            completed: Whether task reached completion
        """
        integration = self.intelligent_integration
        if not integration:
            return

        # Get orchestrator state to pass to integration
        stream_context = getattr(self, "_current_stream_context", None)
        continuation_prompts = getattr(self, "_continuation_prompts", 0)
        max_continuation_prompts_used = getattr(self, "_max_continuation_prompts_used", 6)
        stuck_loop_detected = getattr(self, "_stuck_loop_detected", False)

        try:
            integration.record_intelligent_outcome(
                success=success,
                quality_score=quality_score,
                user_satisfied=user_satisfied,
                completed=completed,
                rl_coordinator=self._rl_coordinator,
                stream_context=stream_context,
                vertical_context=self._vertical_context,
                provider_name=self.provider.name,
                model=self.model,
                tool_calls_used=self.tool_calls_used,
                continuation_prompts=continuation_prompts,
                max_continuation_prompts_used=max_continuation_prompts_used,
                stuck_loop_detected=stuck_loop_detected,
            )
        except Exception as e:
            logger.debug(f"IntelligentPipeline record_outcome failed: {e}")

        # EvoTest: trigger session-end GEPA evolution (E12)
        if self.tool_calls_used >= 3:
            try:
                rl_coord = getattr(self, "_rl_coordinator", None)
                if rl_coord and hasattr(rl_coord, "try_evolve_on_session_end"):
                    result = rl_coord.try_evolve_on_session_end(
                        getattr(self.provider, "name", "unknown"), self.model
                    )
                    if result and isinstance(result, dict):
                        # Display evolution report via debug logger
                        debug_log = getattr(self, "debug_logger", None)
                        if debug_log and hasattr(debug_log, "log_evolution_report"):
                            debug_log.log_evolution_report(result)
            except Exception as e:
                logger.debug(f"[evotest] Session-end evolution failed: {e}")

    def _should_continue_intelligent(self) -> tuple[bool, str]:
        """Check if processing should continue using learned behaviors.

        Delegates to OrchestratorIntegration.should_continue_intelligent().

        Returns:
            Tuple of (should_continue, reason)
        """
        integration = self.intelligent_integration
        if not integration:
            return True, "Pipeline disabled"

        return integration.should_continue_intelligent()

    @property
    def safety_checker(self) -> "SafetyChecker":
        """Get the safety checker for dangerous operation detection.

        UI layers can use this to set confirmation callbacks:
            orchestrator.safety_checker.confirmation_callback = my_callback

        Returns:
            SafetyChecker instance for dangerous operation detection
        """
        return self._safety_checker

    @property
    def auto_committer(self) -> Optional["AutoCommitter"]:
        """Get the auto-committer for AI-assisted code change commits.

        Usage:
            result = orchestrator.auto_committer.commit_changes(
                files=["src/api.py"],
                description="Add input validation",
                change_type="feat"
            )

        Returns:
            AutoCommitter instance or None if not enabled
        """
        return self._auto_committer

    # =========================================================================
    # Vertical Extension Support
    # =========================================================================

    def apply_vertical_middleware(self, middleware_list: List[Any]) -> None:
        """Apply middleware from vertical extensions.

        Called by FrameworkShim after orchestrator creation to inject
        vertical-specific middleware. This enables the middleware chain
        pattern for tool execution.

        Delegates to VerticalIntegrationAdapter for single-source implementation.

        Args:
            middleware_list: List of MiddlewareProtocol implementations.
        """
        self._vertical_integration_adapter.apply_middleware(middleware_list)

    def apply_vertical_safety_patterns(self, patterns: List[Any]) -> None:
        """Apply safety patterns from vertical extensions.

        Called by FrameworkShim to inject vertical-specific danger patterns
        into the safety checker.

        Delegates to VerticalIntegrationAdapter for single-source implementation.

        Args:
            patterns: List of SafetyPattern objects.
        """
        self._vertical_integration_adapter.apply_safety_patterns(patterns)

    def get_middleware_chain(self) -> Optional[Any]:
        """Get the middleware chain for tool execution.

        Returns:
            MiddlewareChain instance or None if not initialized.
        """
        return getattr(self, "_middleware_chain", None)

    def set_middleware_chain(self, chain: Any) -> None:
        """Store middleware chain via public runtime port."""
        self._middleware_chain = chain

    # =========================================================================
    # Internal Storage Setters (DIP Compliance)
    # These methods provide controlled access for adapter implementations,
    # replacing direct private attribute writes. Only called by
    # VerticalIntegrationAdapter - not for general use.
    # =========================================================================

    def _set_vertical_middleware_storage(self, middleware: List[Any]) -> None:
        """Internal: Set vertical middleware storage.

        DIP Compliance: Provides controlled setter instead of direct
        private attribute access. Called by VerticalIntegrationAdapter.

        Args:
            middleware: List of middleware instances
        """
        self.set_middleware(middleware)

    def _set_middleware_chain_storage(self, chain: Any) -> None:
        """Internal: Set middleware chain storage.

        DIP Compliance: Provides controlled setter instead of direct
        private attribute access. Called by VerticalIntegrationAdapter.

        Args:
            chain: MiddlewareChain instance
        """
        self.set_middleware_chain(chain)

    def _set_safety_patterns_storage(self, patterns: List[Any]) -> None:
        """Internal: Set safety patterns storage.

        DIP Compliance: Provides controlled setter instead of direct
        private attribute access. Called by VerticalIntegrationAdapter.

        Args:
            patterns: List of safety pattern instances
        """
        self.set_safety_patterns(patterns)

    # =========================================================================
    # VerticalStorageProtocol Implementation (DIP Compliance)
    # These public methods implement VerticalStorageProtocol, providing a clean
    # interface for vertical data storage and retrieval. This replaces direct
    # private attribute access with protocol-compliant methods.
    # =========================================================================

    def set_middleware(self, middleware: List[Any]) -> None:
        """Store middleware configuration.

        Implements VerticalStorageProtocol.set_middleware().
        Provides a clean public interface for setting vertical middleware,
        replacing direct private attribute access.

        Args:
            middleware: List of MiddlewareProtocol implementations
        """
        self._vertical_middleware = middleware

    def get_middleware(self) -> List[Any]:
        """Retrieve middleware configuration.

        Implements VerticalStorageProtocol.get_middleware().
        Returns the list of middleware instances configured by vertical integration.

        Returns:
            List of middleware instances, or empty list if not set
        """
        return getattr(self, "_vertical_middleware", [])

    def set_safety_patterns(self, patterns: List[Any]) -> None:
        """Store safety patterns.

        Implements VerticalStorageProtocol.set_safety_patterns().
        Provides a clean public interface for setting vertical safety patterns,
        replacing direct private attribute access.

        Args:
            patterns: List of SafetyPattern instances from vertical extensions
        """
        self._vertical_safety_patterns = patterns

    def get_safety_patterns(self) -> List[Any]:
        """Retrieve safety patterns.

        Implements VerticalStorageProtocol.get_safety_patterns().
        Returns the list of safety patterns configured by vertical integration.

        Returns:
            List of safety pattern instances, or empty list if not set
        """
        return getattr(self, "_vertical_safety_patterns", [])

    def set_team_specs(self, specs: Dict[str, Any]) -> None:
        """Store team specifications.

        Implements VerticalStorageProtocol.set_team_specs().
        Provides a clean public interface for setting team specs,
        replacing direct private attribute access.

        Args:
            specs: Dictionary mapping team names to TeamSpec instances
        """
        self._team_specs = specs

    def get_team_specs(self) -> Dict[str, Any]:
        """Retrieve team specifications.

        Implements VerticalStorageProtocol.get_team_specs().
        Returns the dictionary of team specs configured by vertical integration.

        Returns:
            Dictionary of team specs, or empty dict if not set
        """
        return getattr(self, "_team_specs", {})

    @property
    def messages(self) -> List[Message]:
        """Get conversation messages (backward compatibility property).

        Returns:
            List of messages in conversation history
        """
        return self.conversation.messages

    def _get_model_context_window(self) -> int:
        """Get context window size for the current model.

        Delegates to ContextManager when available (TD-002 refactoring).
        Falls back to direct implementation during __init__ before
        ContextManager is created.

        Returns:
            Context window size in tokens
        """
        # Delegate to ContextManager if available
        if hasattr(self, "_context_manager") and self._context_manager is not None:
            return self._context_manager.get_model_context_window()

        # Fallback for calls during __init__ before ContextManager is created
        try:
            from victor.config.config_loaders import get_provider_limits

            limits = get_provider_limits(self.provider_name, self.model)
            return limits.context_window
        except Exception as e:
            logger.warning(f"Could not load provider limits from config: {e}")
            return 128000  # Default safe value

    def _get_max_context_chars(self) -> int:
        """Get maximum context size in characters.

        Delegates to ContextManager (TD-002 refactoring).

        Returns:
            Maximum context size in characters
        """
        return self._context_manager.get_max_context_chars()

    def _check_cache_setting_enabled(self) -> bool:
        """Check if cache optimization is enabled in settings."""
        ctx = getattr(self, "settings", None)
        if ctx is not None:
            context = getattr(ctx, "context", None)
            if context is not None:
                if not getattr(context, "cache_optimization_enabled", True):
                    return False
        return True

    @property
    def _cache_optimization_enabled(self) -> bool:
        """Full API-level cache optimization — session-lock all tools + freeze prompt.

        Uses cached flag from _compute_cache_flags() when available.
        """
        cached = getattr(self, "_cache_opt_cached", None)
        if cached is not None:
            return cached
        try:
            if not self._check_cache_setting_enabled():
                return False
            provider = getattr(self, "provider", None)
            if provider is not None and hasattr(provider, "supports_prompt_caching"):
                return provider.supports_prompt_caching()
            return False
        except Exception:
            return False

    def _compute_cache_flags(self) -> None:
        """Compute and cache optimization flags once (called after provider init).

        Caches _kv_opt_cached and _cache_opt_cached so the @property accessors
        don't repeat try/except/getattr/hasattr chains on every access.
        """
        try:
            if not self._check_cache_setting_enabled():
                self._kv_opt_cached = False
                self._cache_opt_cached = False
                return

            provider = getattr(self, "provider", None)
            # Cache optimization (API billing discount)
            self._cache_opt_cached = (
                provider is not None
                and hasattr(provider, "supports_prompt_caching")
                and provider.supports_prompt_caching()
            )
            # KV optimization (prefix latency savings) — controlled by separate setting
            kv_setting = True
            ctx = getattr(self, "settings", None)
            if ctx is not None:
                context = getattr(ctx, "context", None)
                if context is not None:
                    kv_setting = getattr(context, "kv_optimization_enabled", True)

            if not kv_setting:
                self._kv_opt_cached = False
            elif provider is not None and hasattr(provider, "supports_kv_prefix_caching"):
                self._kv_opt_cached = provider.supports_kv_prefix_caching()
            elif self._cache_opt_cached:
                self._kv_opt_cached = True  # API caching implies KV
            else:
                self._kv_opt_cached = False
        except Exception:
            self._kv_opt_cached = False
            self._cache_opt_cached = False

    @property
    def _kv_optimization_enabled(self) -> bool:
        """KV prefix cache optimization — freeze prompt for prefix reuse.

        Uses cached flag from _compute_cache_flags() when available.
        """
        cached = getattr(self, "_kv_opt_cached", None)
        if cached is not None:
            return cached
        # Fallback: compute on the fly (before _compute_cache_flags is called)
        try:
            if not self._check_cache_setting_enabled():
                return False
            # Check separate kv_optimization_enabled setting
            ctx = getattr(self, "settings", None)
            if ctx is not None:
                context = getattr(ctx, "context", None)
                if context is not None:
                    if not getattr(context, "kv_optimization_enabled", True):
                        return False
            provider = getattr(self, "provider", None)
            if provider is not None:
                if hasattr(provider, "supports_kv_prefix_caching"):
                    return provider.supports_kv_prefix_caching()
                if hasattr(provider, "supports_prompt_caching"):
                    return provider.supports_prompt_caching()
            return False
        except Exception:
            return False

    async def warm_up_kv_cache(self) -> None:
        """Prime the KV cache by sending a minimal request with the system prompt.

        For KV prefix caching providers (Ollama, LMStudio), the first API call
        is always cold — full prefill computation. This method sends a 1-token
        completion to prime the KV cache so subsequent calls reuse the prefix.

        No-op when KV optimization is disabled.
        """
        if not self._kv_optimization_enabled:
            return
        try:
            from victor.providers.base import Message as Msg

            messages = [Msg(role="system", content=self._system_prompt or "")]
            await self.provider.chat(
                messages=messages,
                model=self.model,
                max_tokens=1,
            )
            logger.info("[kv-cache] Warm-up complete — KV prefix primed")
        except Exception as e:
            logger.debug("[kv-cache] Warm-up failed (non-fatal): %s", e)

    def _kv_prefix_fingerprint(self) -> str:
        """Compute a short fingerprint of the system prompt for KV cache observability.

        Returns a hex hash of the first 500 chars of the system prompt. Identical
        hashes across turns indicate the KV prefix is stable (likely cache hit).
        """
        prompt = getattr(self, "_system_prompt", "") or ""
        import hashlib

        return hashlib.md5(prompt[:500].encode()).hexdigest()[:12]

    def get_session_tools(self) -> Optional[list]:
        """Get session-locked tools for cache-friendly API calls.

        Returns the FULL tool set (frozen at session start) so the tools
        prefix remains byte-identical across all API calls in a session.
        When cache optimization is disabled, returns None (use per-turn).
        """
        if not self._cache_optimization_enabled:
            return None
        session_tools = getattr(self, "_session_tools", None)
        if session_tools is None:
            try:
                all_tools = self._get_all_available_tools()
                if all_tools:
                    self._session_tools = all_tools
                    logger.info(
                        "[cache] Session tools locked: %d tools (prefix-stable)",
                        len(all_tools),
                    )
                    return all_tools
            except Exception:
                return None
        return session_tools

    def get_assembled_messages(self, current_query: Optional[str] = None) -> List["Message"]:
        """Get context-assembled messages for provider calls.

        When context assembler is available: keeps system prompt + ledger +
        last N full turns + score-selected older messages within budget.
        Falls back to raw self.messages when assembler unavailable.

        When cache_optimization is enabled, dynamic content (skills, reminders,
        task guidance) is prepended to the last user message instead of being
        injected as system messages. This keeps the system prompt byte-identical
        across turns for provider prefix caching (90% discount).
        """
        assembler = getattr(self, "_context_assembler", None)
        if assembler is None:
            messages = list(self.messages)
        else:
            max_chars = self._get_max_context_chars()
            messages = assembler.assemble(self.messages, max_chars, current_query=current_query)

            if len(messages) < len(self.messages):
                total_original = sum(len(m.content) for m in self.messages)
                total_assembled = sum(len(m.content) for m in messages)
                logger.info(
                    "[context] Assembled %d/%d messages (%dK/%dK chars, budget=%dK)",
                    len(messages),
                    len(self.messages),
                    total_assembled // 1024,
                    total_original // 1024,
                    max_chars // 1024,
                )

        # Log KV prefix fingerprint for observability
        if self._kv_optimization_enabled:
            logger.debug(
                "[kv-cache] prefix=%s frozen=%s tools_cached=%s",
                self._kv_prefix_fingerprint(),
                getattr(self, "_system_prompt_frozen", False),
                getattr(self, "_last_sorted_tool_names", None) is not None,
            )

        # Cache-friendly: prepend dynamic content to last user message
        # Gate on _kv_optimization_enabled so KV-only providers (Ollama) also benefit
        if self._kv_optimization_enabled and messages:
            # Use UnifiedPromptPipeline for per-turn prefix composition
            pipeline = getattr(self, "_prompt_pipeline", None)
            if pipeline:
                from victor.agent.prompt_pipeline import TurnContext

                # Build turn context for the pipeline
                injector = getattr(self, "_optimization_injector", None)
                turn_ctx = TurnContext(
                    provider_name=self.provider_name or "",
                    model=getattr(self, "_model", "") or "",
                    task_type=getattr(self, "_current_task_type", "default"),
                    active_skill_prompt=(
                        self.get_skill_user_prefix()
                        if hasattr(self, "get_skill_user_prefix")
                        else None
                    ),
                    last_turn_failed=(
                        injector._last_failure_category is not None if injector else False
                    ),
                    last_failure_category=(injector._last_failure_category if injector else None),
                    last_failure_error=(injector._last_failure_error if injector else None),
                )

                # Get context reminders
                reminder_mgr = getattr(self, "_reminder_manager", None)
                if reminder_mgr:
                    turn_ctx.reminder_text = reminder_mgr.get_user_message_prefix()

                # Get last user message text for KNN few-shot matching
                last_user_msg = ""
                for m in reversed(messages):
                    if m.role == "user":
                        last_user_msg = m.content[:200]
                        break

                prefix = pipeline.compose_turn_prefix(last_user_msg, turn_ctx)

                # Clear failure state after injecting hint
                if injector and turn_ctx.last_turn_failed:
                    injector._last_failure_category = None
                    injector._last_failure_error = None

            if prefix:
                from victor.providers.base import Message as Msg

                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].role == "user":
                        messages[i] = Msg(
                            role="user",
                            content=prefix + messages[i].content,
                        )
                        break

        return messages

    def _check_context_overflow(self, max_context_chars: int = 200000) -> bool:
        """Check if context is at risk of overflow.

        Args:
            max_context_chars: Maximum allowed context size in chars

        Returns:
            True if context is dangerously large
        """
        if self._context_service:
            return self._context_service.check_context_overflow(max_context_chars)
        return self._context_manager.check_context_overflow(max_context_chars)

    def get_context_metrics(self) -> ContextMetrics:
        """Get detailed context metrics.

        Returns:
            ContextMetrics with size and overflow information
        """
        if self._context_service:
            return self._context_service.get_context_metrics()
        return self._context_manager.get_context_metrics()

    def _init_conversation_embedding_store(self) -> None:
        """Initialize embedding store. Delegates to SessionCoordinator."""
        from victor.agent.runtime.memory_runtime import (
            initialize_conversation_embedding_store,
        )

        store, cache = initialize_conversation_embedding_store(
            memory_manager=self.memory_manager,
        )
        self._conversation_embedding_store = store
        if cache is not None:
            self._pending_semantic_cache = cache

    def _finalize_stream_metrics(
        self, usage_data: Optional[Dict[str, int]] = None
    ) -> Optional[StreamMetrics]:
        """Finalize stream metrics at end of streaming session.

        Delegates to MetricsCoordinator (Phase 2 refactoring).

        Args:
            usage_data: Optional cumulative token usage from provider API.
                       When provided, enables accurate token counts.
        """
        return self._metrics_coordinator.finalize_stream_metrics(usage_data)

    def get_last_stream_metrics(self) -> Optional[StreamMetrics]:
        """Get metrics from the last streaming session.

        Delegates to MetricsCoordinator (Phase 2 refactoring).

        Returns:
            StreamMetrics from the last session or None if no metrics available
        """
        return self._metrics_coordinator.get_last_stream_metrics()

    def get_streaming_metrics_summary(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive streaming metrics summary.

        Delegates to MetricsCoordinator (Phase 2 refactoring).

        Returns:
            Dictionary with aggregated metrics or None if metrics disabled.
        """
        return self._metrics_coordinator.get_streaming_metrics_summary()

    def get_streaming_metrics_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent streaming metrics history.

        Delegates to MetricsCoordinator (Phase 2 refactoring).

        Args:
            limit: Maximum number of recent metrics to return

        Returns:
            List of recent metrics dictionaries
        """
        return self._metrics_coordinator.get_streaming_metrics_history(limit)

    def get_session_cost_summary(self) -> Dict[str, Any]:
        """Get session cost summary.

        Delegates to MetricsCoordinator (Phase 2 refactoring).

        Returns:
            Dictionary with session cost statistics
        """
        return self._metrics_coordinator.get_session_cost_summary()

    def get_session_cost_formatted(self) -> str:
        """Get formatted session cost string.

        Delegates to MetricsCoordinator (Phase 2 refactoring).

        Returns:
            Cost string like "$0.0123" or "cost n/a"
        """
        return self._metrics_coordinator.get_session_cost_formatted()

    def export_session_costs(self, path: str, format: str = "json") -> None:
        """Export session costs to file.

        Delegates to MetricsCoordinator (Phase 2 refactoring).

        Args:
            path: Output file path
            format: Export format ("json" or "csv")
        """
        self._metrics_coordinator.export_session_costs(path, format)

    async def _preload_embeddings(self) -> None:
        """Preload tool embeddings in background to avoid blocking first query.

        This is called asynchronously during initialization if semantic tool
        selection is enabled. Errors are logged but don't crash the app.

        Note: The ToolSelector owns the _embeddings_initialized state to avoid
        DRY violations and consistency issues.
        """
        if not self.semantic_selector:
            return
        # ToolSelector owns the initialization state
        if self.tool_selector._embeddings_initialized:
            return

        try:
            logger.info("Starting background embedding preload...")
            await self.semantic_selector.initialize_tool_embeddings(self.tools)
            # Mark initialization complete in ToolSelector (single source of truth)
            self.tool_selector._embeddings_initialized = True
            logger.info(
                f"{self._presentation.icon('success')} Tool embeddings preloaded successfully in background"
            )
        except Exception as e:
            logger.warning(
                f"Failed to preload embeddings in background: {e}. "
                "Will load on first query (may add ~5s latency)."
            )

    def _create_background_task(
        self, coro: Any, name: str = "background_task"
    ) -> Optional[asyncio.Task]:
        """Create and track a background task for graceful shutdown."""
        from victor.agent.coordinators.session_coordinator import SessionCoordinator

        return SessionCoordinator.create_background_task(
            coro, name, self._background_tasks, self._bg_task_lock
        )

    async def _run_runtime_preload(self) -> None:
        """Run feature-flagged runtime preload tasks via PreloadManager."""
        from victor.framework.preload import (
            PreloadManager,
            PreloadPriority,
            preload_configuration,
        )

        preload_manager = PreloadManager(
            enable_parallel=bool(getattr(self.settings, "framework_preload_parallel", True))
        )
        preload_manager.add_task(
            "configuration",
            preload_configuration,
            priority=PreloadPriority.CRITICAL.value,
            required=False,
        )

        if self.use_semantic_selection:
            preload_manager.add_task(
                "tool_embeddings",
                self._preload_embeddings,
                priority=PreloadPriority.HIGH.value,
                required=False,
            )

        if getattr(self.settings, "http_connection_pool_enabled", False):

            async def _preload_http_pool() -> None:
                from victor.tools.http_pool import ConnectionPoolConfig, get_http_pool

                await get_http_pool(
                    ConnectionPoolConfig(
                        max_connections=int(
                            getattr(
                                self.settings,
                                "http_connection_pool_max_connections",
                                100,
                            )
                        ),
                        max_connections_per_host=int(
                            getattr(
                                self.settings,
                                "http_connection_pool_max_connections_per_host",
                                10,
                            )
                        ),
                        connection_timeout=int(
                            getattr(
                                self.settings,
                                "http_connection_pool_connection_timeout",
                                30,
                            )
                        ),
                        total_timeout=int(
                            getattr(self.settings, "http_connection_pool_total_timeout", 60)
                        ),
                    )
                )

            preload_manager.add_task(
                "http_pool",
                _preload_http_pool,
                priority=PreloadPriority.HIGH.value,
                required=False,
            )

        if not preload_manager.list_tasks():
            return

        if getattr(self.settings, "framework_preload_parallel", True):
            stats = await preload_manager.preload_parallel()
        else:
            stats = await preload_manager.preload_all()
        logger.info(
            "Runtime preload completed: %s/%s tasks in %.2fs",
            stats.completed_tasks,
            stats.total_tasks,
            stats.duration,
        )

    def start_embedding_preload(self) -> None:
        """Start background embedding preload task.

        Should be called after orchestrator initialization to avoid blocking
        the main thread. Safe to call multiple times (no-op if already started).

        Embedding preload is deferred by default to avoid +694MB memory spike
        on first task. Set `preload_embeddings=True` in settings to eagerly load.
        The embedding model will still lazy-load on first semantic query via
        EmbeddingService._ensure_model_loaded().
        """
        if getattr(self.settings, "framework_preload_enabled", False):
            if self._runtime_preload_task is not None:
                return
            task = self._create_background_task(self._run_runtime_preload(), name="runtime_preload")
            if task:
                self._runtime_preload_task = task
                logger.info("Started runtime preload task")
            return

        # Only preload embeddings if explicitly requested (default: deferred)
        if not getattr(self.settings, "preload_embeddings", False):
            return

        if not self.use_semantic_selection or self._embedding_preload_task:
            return

        task = self._create_background_task(self._preload_embeddings(), name="embedding_preload")
        if task:
            self._embedding_preload_task = task
            logger.info("Started background task for embedding preload")

    def _record_tool_selection(self, method: str, num_tools: int) -> None:
        """Record tool selection statistics.

        Args:
            method: Selection method used ('semantic', 'keyword', 'fallback')
            num_tools: Number of tools selected
        """
        self._metrics_coordinator.record_tool_selection(method, num_tools)

    def route_search_query(self, query: str) -> Dict[str, Any]:
        """Route a search query to the optimal search tool using SearchRouter.

        Analyzes the query to determine whether keyword search (code_search)
        semantic search (semantic_code_search), or graph traversal (graph)
        would yield better results.

        Args:
            query: The search query

        Returns:
            Dictionary with routing recommendation:
                - recommended_tool: "code_search" or "semantic_code_search" or "graph" or "both"
                - recommended_args: Suggested tool arguments (for example {"mode": "bugs"})
                - confidence: Confidence in the recommendation (0.0-1.0)
                - reason: Human-readable explanation
                - search_type: SearchType enum value

        Example:
            route = orchestrator.route_search_query("class BaseTool")
            # Returns: {"recommended_tool": "code_search", "confidence": 1.0, ...}

            route = orchestrator.route_search_query("how does error handling work")
            # Returns: {"recommended_tool": "semantic_code_search", "confidence": 0.9, ...}
        """
        route: SearchRoute = self.search_router.route(query)

        # Map SearchType to tool name
        tool_map = {
            SearchType.KEYWORD: "code_search",
            SearchType.SEMANTIC: "semantic_code_search",
            SearchType.HYBRID: "both",
        }

        recommended_tool = route.tool_name or tool_map.get(route.search_type, "code_search")

        return {
            "recommended_tool": recommended_tool,
            "recommended_args": route.tool_arguments,
            "confidence": route.confidence,
            "reason": route.reason,
            "search_type": route.search_type.value,
            "matched_patterns": route.matched_patterns,
            "transformed_query": route.transformed_query,
        }

    def get_recommended_search_tool(self, query: str) -> str:
        """Get the recommended search tool name for a query.

        Convenience method that returns just the tool name.

        Args:
            query: The search query

        Returns:
            Tool name: "code_search", "semantic_code_search", "graph", or "both"
        """
        return self.route_search_query(query)["recommended_tool"]

    def _record_tool_execution(
        self,
        tool_name: str,
        success: bool,
        elapsed_ms: float,
        error_type: Optional[str] = None,
    ) -> None:
        """Record tool execution statistics. Delegates to MetricsCoordinator."""
        self._metrics_coordinator.record_tool_execution_full(
            tool_name=tool_name,
            success=success,
            elapsed_ms=elapsed_ms,
            error_type=error_type,
            usage_analytics=getattr(self, "_usage_analytics", None),
            sequence_tracker=getattr(self, "_sequence_tracker", None),
            tool_selector=getattr(self, "tool_selector", None),
            rl_coordinator=self._rl_coordinator,
            conversation_controller=getattr(self, "_conversation_controller", None),
            provider_name=getattr(self.current_provider, "name", "unknown"),
            model_name=getattr(self, "_current_model", "unknown"),
            task_type=getattr(self, "_task_type", "general"),
            vertical_name=getattr(self, "_vertical_name", None),
        )

    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive tool usage statistics.

        Delegates to MetricsCoordinator (Phase 2 refactoring).

        Returns:
            Dictionary with usage analytics including:
            - Selection stats (semantic/keyword/fallback counts)
            - Per-tool execution stats (calls, success rate, timing)
            - Cost tracking (by tier and total)
            - Overall metrics
        """
        return self._metrics_coordinator.get_tool_usage_stats(
            conversation_state_summary=self.conversation_state.get_state_summary()
        )

    def get_token_usage(self) -> "TokenUsage":
        """Get cumulative token usage for evaluation tracking.

        Delegates to MetricsCoordinator (Phase 2 refactoring).

        Returns cumulative tokens used across all stream_chat calls.
        Used by VictorAgentAdapter for benchmark token tracking.

        Returns:
            TokenUsage dataclass with input/output/total token counts
        """
        return self._metrics_coordinator.get_token_usage()

    def reset_token_usage(self) -> None:
        """Reset cumulative token usage tracking.

        Delegates to MetricsCoordinator (Phase 2 refactoring).

        Call this at the start of a new evaluation task to get fresh counts.
        """
        # Reset through coordinator (which updates the cumulative dict)
        for key in self._cumulative_token_usage:
            self._cumulative_token_usage[key] = 0

    def get_conversation_stage(self) -> ConversationStage:
        """Get the current conversation stage.

        Returns:
            Current ConversationStage enum value
        """
        return self.conversation_state.get_stage()

    def get_stage_recommended_tools(self) -> Set[str]:
        """Get tools recommended for the current conversation stage.

        Returns:
            Set of tool names recommended for current stage
        """
        return self.conversation_state.get_stage_tools()

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all integrated optimization components.

        Delegates to MetricsCoordinator which owns the reporting logic.
        """
        return self._metrics_coordinator.get_optimization_status(
            context_compactor=self._context_compactor,
            usage_analytics=self._usage_analytics,
            sequence_tracker=self._sequence_tracker,
            code_correction_middleware=self._code_correction_middleware,
            safety_checker=self._safety_checker,
            auto_committer=self._auto_committer,
            search_router=self.search_router,
        )

    def flush_analytics(self) -> Dict[str, bool]:
        """Flush all analytics and cached data. Delegates to MetricsCoordinator."""
        return self._metrics_coordinator.flush_analytics(
            usage_analytics=getattr(self, "_usage_analytics", None),
            sequence_tracker=getattr(self, "_sequence_tracker", None),
            tool_cache=getattr(self, "tool_cache", None),
        )

    async def start_health_monitoring(self) -> bool:
        """Start background provider health monitoring.

        Delegates to ProviderCoordinator (TD-002).

        Returns:
            True if monitoring started, False if already running or unavailable
        """
        try:
            await self._provider_coordinator.start_health_monitoring()
            return True
        except Exception as e:
            logger.warning(f"Failed to start health monitoring: {e}")
            return False

    async def stop_health_monitoring(self) -> bool:
        """Stop background provider health monitoring.

        Delegates to ProviderCoordinator (TD-002).

        Returns:
            True if monitoring stopped, False if not running or error
        """
        try:
            await self._provider_coordinator.stop_health_monitoring()
            return True
        except Exception as e:
            logger.warning(f"Failed to stop health monitoring: {e}")
            return False

    async def get_provider_health(self) -> Dict[str, Any]:
        """Get health status of all registered providers.

        Delegates to ProviderCoordinator (TD-002).

        Returns:
            Dictionary with provider health information
        """
        return await self._provider_coordinator.get_health()

    async def graceful_shutdown(self) -> Dict[str, bool]:
        """Perform graceful shutdown of all orchestrator components.

        Delegates to LifecycleManager for core shutdown logic.

        Flushes analytics, stops health monitoring, and cleans up resources.
        Call this before application exit.

        Returns:
            Dictionary with shutdown status for each component
        """
        # Shutdown provider pool if active
        pool = getattr(getattr(self, "_provider_runtime", None), "pool", None)
        if pool is not None:
            try:
                await pool.shutdown()
                logger.info("ProviderPool shutdown complete")
            except Exception as e:
                logger.warning("ProviderPool shutdown error: %s", e)

        # Delegate to LifecycleManager for graceful shutdown
        return await self._lifecycle_manager.graceful_shutdown()

    # =========================================================================
    # Provider/Model Hot-Swap Methods
    # =========================================================================

    def get_current_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider and model.

        Combines provider info with orchestrator-specific runtime state.

        Returns:
            Dictionary with provider/model info and capabilities
        """
        if self._provider_service:
            info = self._provider_service.get_current_provider_info()
            if hasattr(info, "__dict__"):
                info = {
                    "provider_name": getattr(info, "provider_name", ""),
                    "model_name": getattr(info, "model_name", ""),
                }
            elif not isinstance(info, dict):
                info = {}
        else:
            info = self._provider_manager.get_info()
        info.update(
            {
                "tool_budget": self.tool_budget,
                "tool_calls_used": self.tool_calls_used,
            }
        )
        return info

    def _parse_tool_calls_with_adapter(
        self, content: str, raw_tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> ToolCallParseResult:
        """Parse tool calls using the tool calling adapter.

        This is the unified method for parsing tool calls that handles:
        1. Native tool calls from provider
        2. JSON fallback parsing
        3. XML fallback parsing
        4. Tool name validation

        Args:
            content: Response content text
            raw_tool_calls: Native tool_calls from provider (if any)

        Returns:
            ToolCallParseResult with parsed tool calls and metadata
        """
        result = self.tool_adapter.parse_tool_calls(content, raw_tool_calls)

        # Log any warnings
        for warning in result.warnings:
            logger.warning(f"Tool call parse warning: {warning}")

        # Log parse method for debugging
        if result.tool_calls:
            logger.debug(
                f"Parsed {len(result.tool_calls)} tool calls via {result.parse_method} "
                f"(confidence={result.confidence})"
            )

        return result

    def _apply_skill_for_turn(self, user_message: str) -> None:
        """Apply skill auto-selection for a turn (used by both sync and streaming paths).

        Clears previous skill, matches new skill(s), injects into prompt,
        and records analytics.
        """
        self.clear_active_skills()

        matcher = getattr(self, "_skill_matcher", None)
        if (
            matcher is None
            or not getattr(matcher, "_initialized", False)
            or getattr(self, "_skill_auto_disabled", False)
            or getattr(self, "_manual_skill_active", False)
        ):
            return

        try:
            matches = matcher.match_multiple_sync(user_message)
            if matches:
                if len(matches) == 1:
                    skill, score = matches[0]
                    logger.info("Auto-selected skill: %s (score=%.2f)", skill.name, score)
                    self.inject_skill(skill)
                    self._last_skill_match_info = {
                        "auto_skill": skill.name,
                        "auto_skill_score": round(score, 2),
                    }
                else:
                    names = [s.name for s, _ in matches]
                    logger.info("Auto-selected %d skills: %s", len(matches), " → ".join(names))
                    self.inject_skills(matches)
                    self._last_skill_match_info = {
                        "auto_skills": [
                            {"name": s.name, "score": round(sc, 2)} for s, sc in matches
                        ],
                    }

                # Record analytics
                analytics = getattr(self, "_skill_analytics", None)
                if analytics:
                    if len(matches) == 1:
                        analytics.record_selection(matches[0][0].name, matches[0][1])
                    else:
                        analytics.record_multi_selection([(s.name, sc) for s, sc in matches])
            else:
                analytics = getattr(self, "_skill_analytics", None)
                if analytics:
                    analytics.record_miss()
                self._last_skill_match_info = None
        except Exception:
            logger.debug("Skill auto-selection failed", exc_info=True)

    def get_last_skill_match_info(self) -> Optional[Dict[str, Any]]:
        """Return metadata about the last skill match for response attachment."""
        return getattr(self, "_last_skill_match_info", None)

    def clear_active_skills(self) -> None:
        """Remove any active skill injection.

        Called at the start of each turn to ensure skills don't accumulate.
        When cache_optimization enabled: just clears _active_skill_prompt
        (system prompt was never touched). Otherwise: restores base prompt.
        """
        self._active_skill_prompt = ""

        # Cache-friendly: system prompt was never mutated, nothing to restore
        if self._kv_optimization_enabled:
            return

        # Legacy: restore base system prompt
        base = getattr(self, "_base_system_prompt", None)
        if base is not None:
            self._system_prompt = base

        if hasattr(self, "conversation") and self.conversation is not None:
            self.conversation.system_prompt = self._system_prompt
            if self.conversation._system_added and self.conversation._messages:
                if self.conversation._messages[0].role == "system":
                    from victor.agent.message_history import Message

                    self.conversation._messages[0] = Message(
                        role="system", content=self._system_prompt
                    )

    def get_skill_user_prefix(self) -> str:
        """Get active skill prompt as user message prefix (cache-friendly).

        When cache_optimization is enabled, skills are injected into the
        user message instead of mutating the system prompt. This keeps
        the system prompt byte-identical across turns for prefix caching.

        Returns:
            Skill prompt prefix string, or empty string if no active skill.
        """
        return getattr(self, "_active_skill_prompt", "") or ""

    def inject_skill(self, skill: Any) -> None:
        """Inject a skill's prompt fragment.

        When cache_optimization is enabled: stores skill text for user
        message injection (via get_skill_user_prefix). System prompt
        stays frozen.

        When disabled: legacy behavior — prepends to system prompt.

        Args:
            skill: A SkillDefinition with name, description, prompt_fragment.
        """
        skill_prompt = (
            f"ACTIVE SKILL: {skill.name}\n"
            f"Description: {skill.description}\n"
            f"{skill.prompt_fragment}\n\n"
        )
        self._active_skill_prompt = skill_prompt

        # Cache-friendly: store for user message injection, don't touch system prompt
        if self._kv_optimization_enabled:
            logger.info("Skill '%s' stored for user message injection (cache-friendly)", skill.name)
            return

        # Legacy: mutate system prompt directly
        if not getattr(self, "_base_system_prompt", None):
            self._base_system_prompt = self._system_prompt or ""

        self._system_prompt = skill_prompt + (self._base_system_prompt or "")

        # Sync to conversation's live message
        if hasattr(self, "conversation") and self.conversation is not None:
            self.conversation.system_prompt = self._system_prompt
            if self.conversation._system_added and self.conversation._messages:
                if self.conversation._messages[0].role == "system":
                    from victor.agent.message_history import Message

                    self.conversation._messages[0] = Message(
                        role="system", content=self._system_prompt
                    )

        logger.info("Injected skill '%s' into system prompt", skill.name)

    def inject_skills(self, skills: List[Any]) -> None:
        """Inject multiple skills' prompt fragments.

        Cache-friendly: when cache_optimization enabled, stores for user
        message injection. Otherwise, legacy system prompt mutation.

        Args:
            skills: List of (SkillDefinition, score) tuples.
        """
        if not skills:
            return

        skills = skills[:3]

        skill_names = []
        fragments = []
        for item in skills:
            skill = item[0] if isinstance(item, tuple) else item
            skill_names.append(skill.name)
            fragments.append(
                f"ACTIVE SKILL: {skill.name}\n"
                f"Description: {skill.description}\n"
                f"{skill.prompt_fragment}\n"
            )

        composed = (
            f"ACTIVE SKILLS ({len(skill_names)}): {' → '.join(skill_names)}\n"
            f"Execute these skills in the listed order.\n\n" + "\n".join(fragments) + "\n"
        )

        self._active_skill_prompt = composed

        # Cache-friendly: store for user message injection
        if self._kv_optimization_enabled:
            logger.info("Skills %s stored for user message injection", skill_names)
            return

        # Legacy: mutate system prompt
        if not getattr(self, "_base_system_prompt", None):
            self._base_system_prompt = self._system_prompt or ""

        self._system_prompt = composed + (self._base_system_prompt or "")

        if hasattr(self, "conversation") and self.conversation is not None:
            self.conversation.system_prompt = self._system_prompt
            if self.conversation._system_added and self.conversation._messages:
                if self.conversation._messages[0].role == "system":
                    from victor.agent.message_history import Message

                    self.conversation._messages[0] = Message(
                        role="system", content=self._system_prompt
                    )

        logger.info("Injected %d skills: %s", len(skill_names), " → ".join(skill_names))

    def update_system_prompt_for_query(self, query_classification=None) -> None:
        """Rebuild system prompt with query-specific classification context.

        Called by coordinators when a query classification is available,
        injecting task-aware guidance and tool constraints into the prompt.
        Updates both the orchestrator's cached prompt AND the conversation's
        live system message so the provider receives the updated prompt.

        When cache_optimization_enabled is True and the system prompt has
        already been built, this is a no-op to preserve prefix cache stability.
        Query-specific guidance should be injected via user messages instead.

        Args:
            query_classification: Optional QueryClassification for task-aware prompting
        """
        # Freeze system prompt after first build for prefix cache stability
        # Gate on _kv_optimization_enabled so both API-caching and KV-caching providers benefit
        # Check if system prompt is frozen (pipeline owns this state when available)
        pipeline = getattr(self, "_prompt_pipeline", None)
        is_frozen = (
            pipeline.is_frozen if pipeline else getattr(self, "_system_prompt_frozen", False)
        )
        if self._kv_optimization_enabled and is_frozen:
            logger.debug("[cache] System prompt frozen — skipping rebuild for query classification")
            return

        if query_classification is not None:
            self.prompt_builder.query_classification = query_classification
        base_system_prompt = self._build_system_prompt_with_adapter()
        if self.project_context and self.project_context.content:
            self._system_prompt = (
                base_system_prompt + "\n\n" + self.project_context.get_system_prompt_addition()
            )
        else:
            self._system_prompt = base_system_prompt

        # Mark as frozen after first build (KV or API cache optimization)
        if self._kv_optimization_enabled:
            self._system_prompt_frozen = True

        # Sync to conversation's live message list so provider receives it
        if hasattr(self, "conversation") and self.conversation is not None:
            self.conversation.system_prompt = self._system_prompt
            # Replace the existing system message if already inserted
            if self.conversation._system_added and self.conversation._messages:
                if self.conversation._messages[0].role == "system":
                    from victor.agent.message_history import Message

                    self.conversation._messages[0] = Message(
                        role="system", content=self._system_prompt
                    )

    def _build_system_prompt_with_adapter(self) -> str:
        """Build system prompt using the unified pipeline.

        Delegates to UnifiedPromptPipeline.build_system_prompt() when
        available. Falls back to SystemPromptCoordinator, then inline logic
        during __init__ before the pipeline is created.
        """
        pipeline = getattr(self, "_prompt_pipeline", None)
        if pipeline is not None:
            return pipeline.build_system_prompt()

        if hasattr(self, "_system_prompt_coordinator"):
            return self._system_prompt_coordinator.build_system_prompt()

        # Fallback for calls during __init__ before pipeline is created
        base_prompt = self.prompt_builder.build()
        context_window = self._get_model_context_window()
        budget = calculate_parallel_read_budget(context_window)
        if context_window >= 32768:
            return f"{base_prompt}\n\n{budget.to_prompt_hint()}"
        return base_prompt

    def _emit_prompt_used_event(self, prompt: str) -> None:
        """Emit PROMPT_USED event for RL prompt template learner.

        Delegates to UnifiedPromptPipeline or SystemPromptCoordinator.
        """
        pipeline = getattr(self, "_prompt_pipeline", None)
        if pipeline is not None:
            pipeline._emit_prompt_used_event(prompt)
        elif hasattr(self, "_system_prompt_coordinator"):
            self._system_prompt_coordinator._emit_prompt_used_event(prompt)

    def _resolve_shell_variant(self, tool_name: str) -> str:
        """Resolve shell aliases to the appropriate enabled shell variant.

        Delegates to UnifiedPromptPipeline or SystemPromptCoordinator.
        """
        pipeline = getattr(self, "_prompt_pipeline", None)
        if pipeline is not None:
            return pipeline.resolve_shell_variant(tool_name)
        return self._system_prompt_coordinator.resolve_shell_variant(tool_name)

    def _get_thinking_disabled_prompt(self, base_prompt: str) -> str:
        """Prefix a prompt with the thinking disable prefix if supported.

        IMPORTANT: This should ONLY be used in RECOVERY scenarios where:
        - Model returned empty response (stuck in thinking)
        - Context overflow forced completion
        - Iteration limit forced completion

        Normal model calls should NOT use this - thinking mode produces
        better quality results. This is a last-resort recovery mechanism.

        For models with thinking mode (e.g., Qwen3), prepends the configured
        disable prefix (e.g., "/no_think") to get direct responses without
        internal reasoning overhead.

        Args:
            base_prompt: The base prompt text

        Returns:
            Prompt with thinking disable prefix if available, otherwise base_prompt
        """
        prefix = getattr(self.tool_calling_caps, "thinking_disable_prefix", None)
        if prefix:
            return f"{prefix}\n{base_prompt}"
        return base_prompt

    def _log_tool_call(self, name: str, kwargs: dict) -> None:
        """A hook that logs information before a tool is called."""
        # Move verbose argument logging to debug level - not user-facing
        logger.debug(f"Tool call: {name} with args: {kwargs}")

    def _classify_task_keywords(self, user_message: str) -> Dict[str, Any]:
        """Classify task type based on keywords in the user message.

        Delegates to SystemPromptCoordinator.classify_task_keywords().

        Args:
            user_message: The user's input message

        Returns:
            Dictionary with classification results
        """
        pipeline = getattr(self, "_prompt_pipeline", None)
        if pipeline is not None:
            return pipeline.classify_task_keywords(user_message)
        return self._system_prompt_coordinator.classify_task_keywords(user_message)

    def _classify_task_with_context(
        self, user_message: str, history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Classify task with conversation context for improved accuracy.

        Delegates to UnifiedPromptPipeline or SystemPromptCoordinator.

        Args:
            user_message: The user's input message
            history: Optional conversation history for context boosting

        Returns:
            Dictionary with classification results
        """
        pipeline = getattr(self, "_prompt_pipeline", None)
        if pipeline is not None:
            return pipeline.classify_task_with_context(user_message, history)
        return self._system_prompt_coordinator.classify_task_with_context(user_message, history)

    def _format_tool_output(self, tool_name: str, args: Dict[str, Any], output: Any) -> str:
        """Format tool output with clear boundaries to prevent model hallucination.

        Delegates to ToolOutputFormatter for:
        - Structured output serialization (lists, dicts -> compact formats)
        - Anti-hallucination markers (TOOL_OUTPUT tags)
        - Smart truncation for large outputs
        - File structure extraction for very large files

        Args:
            tool_name: Name of the tool that was executed
            args: Arguments passed to the tool
            output: Raw output from the tool

        Returns:
            Formatted string with clear TOOL_OUTPUT boundaries
        """
        # Build formatting context from current orchestrator state
        context_metrics = self._conversation_controller.get_context_metrics()
        context = FormattingContext(
            provider_name=self.provider.name if hasattr(self, "provider") else None,
            model=getattr(self.settings, "model", None),
            remaining_tokens=context_metrics.remaining_tokens,
            max_tokens=context_metrics.max_tokens,
            response_token_reserve=getattr(self.settings, "response_token_reserve", 4096),
        )

        # Delegate to the extracted formatter
        return self._tool_output_formatter.format_tool_output(
            tool_name=tool_name,
            args=args,
            output=output,
            context=context,
        )

    def _register_default_workflows(self) -> None:
        """Register default workflows via dynamic discovery.

        Uses DIP-compliant workflow discovery to avoid hardcoded imports.
        Workflows are discovered from victor.workflows package automatically.
        """
        count = register_builtin_workflows(self.workflow_registry)
        logger.debug(f"Dynamically registered {count} workflows")

    def _register_default_tools(self) -> None:
        """Dynamically discovers and registers all tools.

        Delegates to ToolRegistrar for:
        - Pre-registration provider setup
        - Dynamic tool discovery from victor/tools directory
        - MCP integration (if enabled)

        Note: This is a thin wrapper that delegates to ToolRegistrar.
        The method is kept for backwards compatibility.
        """
        # Delegate to ToolRegistrar for provider setup
        self.tool_registrar._setup_providers()

        # Delegate to ToolRegistrar for dynamic tool registration
        registered_count = self.tool_registrar._register_dynamic_tools()
        logger.debug(f"Dynamically registered {registered_count} tools via ToolRegistrar")

        # MCP integration - delegate to ToolRegistrar
        if getattr(self.settings, "use_mcp_tools", False):
            mcp_tools_count = self.tool_registrar._setup_mcp_integration()
            # Copy mcp_registry reference for backwards compatibility
            self.mcp_registry = getattr(self.tool_registrar, "mcp_registry", None)
            if mcp_tools_count > 0:
                logger.debug(
                    f"MCP integration registered {mcp_tools_count} tools via ToolRegistrar"
                )

    def _initialize_plugins(self) -> None:
        """Initialize and load tool plugins from configured directories.

        Delegates to ToolRegistrar for plugin discovery and loading.
        The method is kept for backwards compatibility.
        """
        tool_count = self.tool_registrar._initialize_plugins()
        # Store reference to plugin_manager for backwards compatibility
        self.plugin_manager = self.tool_registrar.plugin_manager
        if tool_count > 0:
            logger.info(f"Plugins initialized via ToolRegistrar: {tool_count} tools")

    def _should_use_tools(self) -> bool:
        """Always return True - tool selection is handled by _select_relevant_tools()."""
        return True

    def _model_supports_tool_calls(self) -> bool:
        """Check provider/model combo against the capability matrix.

        Fallback chain:
        1. Static YAML config (model_capabilities.yaml) — fast, no network
        2. Ollama /api/show capabilities — authoritative runtime detection
        3. Default: False (warn user)
        """
        provider_key = self.provider_name or getattr(self.provider, "name", "")
        if not provider_key:
            return True

        # 1. Check static YAML config first (fast path)
        supported = self.tool_capabilities.is_tool_call_supported(provider_key, self.model)
        if supported:
            return True

        # 2. For Ollama, query the API as authoritative fallback
        if provider_key.lower() == "ollama":
            try:
                from victor.providers.ollama_capability_detector import (
                    get_global_detector,
                )

                base_url = getattr(
                    self.settings.provider,
                    "ollama_base_url",
                    "http://localhost:11434",
                )
                detector = get_global_detector(base_url)
                tool_support = detector.get_tool_support(self.model)
                if tool_support.supports_tools:
                    logger.info(
                        "Model '%s' supports tools (detected via Ollama API, method=%s)",
                        self.model,
                        tool_support.detection_method,
                    )
                    return True
            except Exception as e:
                logger.debug("Ollama capability detection failed: %s", e)

        # 3. Not supported — warn user
        if not self._tool_capability_warned:
            known = ", ".join(self.tool_capabilities.get_supported_models(provider_key)) or "none"
            logger.warning(
                f"Model '{self.model}' is not marked as tool-call-capable for provider '{provider_key}'. "
                f"Known tool-capable models: {known}"
            )
            self.console.print(
                f"[yellow]{self._presentation.icon('warning', with_color=False)} Model '{self.model}' is not marked as tool-call-capable for provider '{provider_key}'. "
                f"Running without tools.[/]"
            )
            self._tool_capability_warned = True
        return False

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history.

        Enforces max_conversation_history ceiling by removing oldest
        non-system messages when the limit is reached.  Persistence
        and usage logging are delegated to ``ChatCoordinator.persist_message``.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        max_history = getattr(self.settings, "max_conversation_history", 100)
        if len(self.conversation.messages) >= max_history:
            for i, msg in enumerate(self.conversation.messages):
                if msg.get("role") != "system":
                    self.conversation.messages.pop(i)
                    break

        self.conversation.add_message(role, content)

        # Delegate persistence + usage logging to ChatCoordinator
        from victor.agent.coordinators.chat_coordinator import ChatCoordinator

        ChatCoordinator.persist_message(
            role=role,
            content=content,
            memory_manager=self.memory_manager,
            memory_session_id=self._memory_session_id,
            usage_logger=self.usage_logger,
        )

    async def chat(
        self,
        user_message: str,
        use_planning: Optional[bool] = False,
    ) -> CompletionResponse:
        """Send a chat message and get response with full agentic loop.

        Delegates to service layer when USE_SERVICE_LAYER is enabled,
        otherwise falls through to ChatCoordinator.

        Args:
            user_message: User's message
            use_planning: Whether to use planning. None = auto-detect,
                True = always, False = never.

        Returns:
            CompletionResponse from the model with complete response
        """
        if self._chat_service:
            return await self._chat_service.chat(user_message)
        return await self._chat_coordinator.chat(user_message, use_planning=use_planning)

    async def chat_with_planning(
        self,
        user_message: str,
        use_planning: Optional[bool] = None,
    ) -> CompletionResponse:
        """Chat with automatic planning for complex multi-step tasks.

        This method extends regular chat by automatically detecting when a task
        is complex enough to benefit from structured planning. It then:

        1. Generates a structured plan using ReadableTaskPlan (token-efficient)
        2. Executes the plan step-by-step with context-aware tools
        3. Provides a comprehensive summary of results

        For simple tasks, it falls back to regular direct chat.

        Args:
            user_message: User's message
            use_planning: Force planning on/off. None = auto-detect

        Returns:
            CompletionResponse from the model

        Example:
            # Auto-detect if planning is needed (recommended)
            response = await agent.chat_with_planning(
                "Analyze the codebase architecture and provide SOLID evaluation"
            )

            # Force planning mode
            response = await agent.chat_with_planning(
                "Implement user auth",
                use_planning=True
            )

            # Disable planning (use direct chat)
            response = await agent.chat_with_planning(
                "Quick question",
                use_planning=False
            )
        """
        if self._chat_service:
            return await self._chat_service.chat_with_planning(user_message, use_planning)
        return await self._chat_coordinator.chat_with_planning(user_message, use_planning)

    async def _handle_context_and_iteration_limits(
        self,
        user_message: str,
        max_total_iterations: int,
        max_context: int,
        total_iterations: int,
        last_quality_score: float,
    ) -> tuple[bool, Optional[StreamChunk]]:
        """Handle context overflow and hard iteration limits.

        Delegates to ChatCoordinator which owns the full implementation.
        """
        return await self._chat_coordinator._handle_context_and_iteration_limits(
            user_message,
            max_total_iterations,
            max_context,
            total_iterations,
            last_quality_score,
        )

    def _prepare_task(
        self, user_message: str, unified_task_type: TrackerTaskType
    ) -> tuple[Any, int]:
        """Prepare task-specific guidance and budget adjustments.

        Delegates to TaskCoordinator for centralized task preparation.

        Extracted from CRITICAL-001 Phase 2D.
        """
        # Wire reminder_manager dependency if not already set
        if self.task_coordinator._reminder_manager is None:
            self.task_coordinator.set_reminder_manager(self.reminder_manager)

        # Delegate to TaskCoordinator
        return self.task_coordinator.prepare_task(
            user_message, unified_task_type, self.conversation_controller
        )

    def _apply_intent_guard(self, user_message: str) -> None:
        """Detect intent and inject prompt guards for read-only tasks.

        Delegates to TaskCoordinator for centralized intent detection.

        Extracted from CRITICAL-001 Phase 2D.
        """
        # Delegate to TaskCoordinator
        self.task_coordinator.apply_intent_guard(user_message, self.conversation_controller)

        # Sync current_intent back to orchestrator
        self._current_intent = self.task_coordinator.current_intent

    def _apply_task_guidance(
        self,
        user_message: str,
        unified_task_type: TrackerTaskType,
        is_analysis_task: bool,
        is_action_task: bool,
        needs_execution: bool,
        max_exploration_iterations: int,
    ) -> None:
        """Apply guidance and budget tweaks for analysis/action tasks.

        Delegates to TaskCoordinator for centralized task guidance.

        Extracted from CRITICAL-001 Phase 2D.
        """
        # Set initial temperature and tool_budget in TaskCoordinator
        self.task_coordinator.temperature = self.temperature
        self.task_coordinator.tool_budget = self.tool_budget

        # Delegate to TaskCoordinator
        self.task_coordinator.apply_task_guidance(
            user_message=user_message,
            unified_task_type=unified_task_type,
            is_analysis_task=is_analysis_task,
            is_action_task=is_action_task,
            needs_execution=needs_execution,
            max_exploration_iterations=max_exploration_iterations,
            conversation_controller=self.conversation_controller,
        )

        # Sync temperature and tool_budget back to orchestrator
        self.temperature = self.task_coordinator.temperature
        self.tool_budget = self.task_coordinator.tool_budget

    async def _select_tools_for_turn(self, context_msg: str, goals: Any) -> Any:
        """Select and prioritize tools for the current turn."""
        provider_supports_tools = self.provider.supports_tools()
        tooling_allowed = provider_supports_tools and self._model_supports_tool_calls()

        if not tooling_allowed:
            return None

        planned_tools = None
        if goals:
            available_inputs = ["query"]
            if self.observed_files:
                available_inputs.append("file_contents")
            planned_tools = self._tool_planner.plan_tools(goals, available_inputs)
            logger.info(f"available_inputs={available_inputs}")

        conversation_depth = self.conversation.message_count()
        conversation_history = (
            [msg.model_dump() for msg in self.messages] if self.messages else None
        )
        tools = await self.tool_selector.select_tools(
            context_msg,
            use_semantic=self.use_semantic_selection,
            conversation_history=conversation_history,
            conversation_depth=conversation_depth,
            planned_tools=planned_tools,
        )
        logger.info(
            f"context_msg={context_msg}\nuse_semantic={self.use_semantic_selection}\nconversation_depth={conversation_depth}"
        )
        tools = self.tool_selector.prioritize_by_stage(context_msg, tools)
        current_intent = getattr(self, "_current_intent", None)
        tools = self._tool_planner.filter_tools_by_intent(tools, current_intent)
        tools = self._apply_kv_tool_strategy(tools)
        return self._sort_tools_for_kv_stability(tools)

    def _sort_tools_for_kv_stability(self, tools):
        """Sort tools deterministically by name for KV prefix cache stability.

        When KV prefix caching is active, the tool definitions are part of the
        prompt prefix. Sorting ensures the same subset of tools always produces
        the same byte sequence, maximizing KV cache reuse across turns.

        Caches the sorted result keyed on tool names to avoid redundant sorting.
        """
        if tools is None:
            return None
        if not self._kv_optimization_enabled:
            return tools

        # Check cache: same tool names → same sorted result
        current_names = frozenset(t.name for t in tools)
        last_names = getattr(self, "_last_sorted_tool_names", None)
        last_tools = getattr(self, "_last_sorted_tools", None)
        if last_names == current_names and last_tools is not None:
            return last_tools

        sorted_tools = sorted(tools, key=lambda t: t.name)
        self._last_sorted_tool_names = current_names
        self._last_sorted_tools = sorted_tools
        return sorted_tools

    def _apply_kv_tool_strategy(self, tools):
        """Apply the configured KV tool selection strategy.

        Strategies (controlled by settings.context.kv_tool_strategy):
          'per_turn'       — Return tools as-is (fresh selection each turn)
          'session_stable' — Lock tools after first selection for KV prefix stability

        Only active when _kv_optimization_enabled is True and provider does NOT
        use API-level caching (which already session-locks the full tool set).
        """
        if tools is None:
            return None
        if not self._kv_optimization_enabled:
            return tools

        strategy = "per_turn"
        try:
            ctx = getattr(self, "settings", None)
            if ctx is not None:
                context = getattr(ctx, "context", None)
                if context is not None:
                    strategy = getattr(context, "kv_tool_strategy", "per_turn")
        except Exception:
            pass

        if strategy == "session_stable":
            cached = getattr(self, "_session_semantic_tools", None)
            if cached is not None:
                return cached
            self._session_semantic_tools = tools
            return tools

        # per_turn: return as-is, don't cache
        return tools

    # =====================================================================
    # Recovery Coordination Helper
    # =====================================================================

    def _create_recovery_context(
        self,
        stream_ctx: "StreamingChatContext",
    ) -> Any:
        """Create RecoveryContext from current orchestrator state.

        Helper method to construct RecoveryContext for all recovery-related
        method calls. Centralizes context creation to avoid duplication.

        Args:
            stream_ctx: Streaming context

        Returns:
            StreamingRecoveryContext with all necessary state
        """
        from victor.agent.recovery_coordinator import StreamingRecoveryContext

        # Get elapsed time from streaming controller
        elapsed_time = 0.0
        if self._streaming_controller.current_session:
            elapsed_time = time.time() - self._streaming_controller.current_session.start_time

        return StreamingRecoveryContext(
            iteration=stream_ctx.total_iterations,
            elapsed_time=elapsed_time,
            tool_calls_used=self.tool_calls_used,
            tool_budget=self.tool_budget,
            max_iterations=stream_ctx.max_total_iterations,
            session_start_time=(
                self._streaming_controller.current_session.start_time
                if self._streaming_controller.current_session
                else time.time()
            ),
            last_quality_score=stream_ctx.last_quality_score,
            streaming_context=stream_ctx,
            provider_name=self.provider_name,
            model=self.model,
            temperature=self.temperature,
            unified_task_type=stream_ctx.unified_task_type,
            is_analysis_task=stream_ctx.is_analysis_task,
            is_action_task=stream_ctx.is_action_task,
        )

    # =====================================================================
    # Recovery Facade Methods
    # These methods create RecoveryContext from StreamingChatContext and
    # delegate to the appropriate coordinator. They centralize context
    # creation and provide a clean API for stream_chat.
    # =====================================================================

    async def _handle_recovery_with_integration(
        self,
        stream_ctx: StreamingChatContext,
        full_content: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        mentioned_tools: Optional[List[str]] = None,
    ) -> "OrchestratorRecoveryAction":
        """Handle response using the recovery coordinator.

        Creates RecoveryContext and delegates to recovery_coordinator viahandle_recovery_with_integration() directly.


        Args:
            stream_ctx: The streaming context
            full_content: Full response content
            tool_calls: Tool calls made (if any)
            mentioned_tools: Tools mentioned but not called

        Returns:
            RecoveryAction with action to take (continue, retry, abort, force_summary)
        """
        # Create recovery context from current state
        recovery_ctx = self._create_recovery_context(stream_ctx)

        # Delegate to RecoveryCoordinator
        return await self._recovery_coordinator.handle_recovery_with_integration(
            recovery_ctx,
            full_content,
            tool_calls,
            mentioned_tools,
            message_adder=self.add_message,
        )

    def _apply_recovery_action(
        self,
        recovery_action: "OrchestratorRecoveryAction",
        stream_ctx: StreamingChatContext,
    ) -> Optional[StreamChunk]:
        """Apply a recovery action using the recovery coordinator.

        Creates RecoveryContext and delegates to recovery_coordinator viaapply_recovery_action() directly.


        Args:
            recovery_action: The recovery action to apply
            stream_ctx: The streaming context

        Returns:
            StreamChunk if action requires immediate yield, None otherwise
        """
        # Create recovery context from current state
        recovery_ctx = self._create_recovery_context(stream_ctx)

        # Delegate to RecoveryCoordinator
        return self._recovery_coordinator.apply_recovery_action(
            recovery_action, recovery_ctx, message_adder=self.add_message
        )

    def _parse_and_validate_tool_calls(
        self,
        tool_calls: Optional[List[Dict[str, Any]]],
        full_content: str,
    ) -> tuple[Optional[List[Dict[str, Any]]], str]:
        """Parse, validate, and normalize tool calls from provider response.

        Delegates to ToolCoordinator which consolidates all tool-call processing.
        """
        return self._tool_coordinator.parse_and_validate_tool_calls(
            tool_calls, full_content, self.tool_adapter
        )

    async def stream_chat(self, user_message: str) -> AsyncIterator[StreamChunk]:
        """Stream a chat response (public entrypoint).

        Applies skill auto-selection before streaming, then delegates
        to service layer or ChatCoordinator.

        Args:
            user_message: User's input message

        Returns:
            AsyncIterator yielding StreamChunk objects with incremental response
        """
        # Skill auto-selection for streaming path (mirrors SyncChatCoordinator)
        self._apply_skill_for_turn(user_message)

        if self._chat_service:
            async for chunk in self._chat_service.stream_chat(user_message):
                yield chunk
            return
        async for chunk in self._chat_coordinator.stream_chat(user_message):
            yield chunk

    async def _execute_tool_with_retry(
        self, tool_name: str, tool_args: Dict[str, Any], context: Dict[str, Any]
    ) -> tuple[Any, bool, Optional[str]]:
        """Execute a tool with retry logic and exponential backoff.

        Delegates to service layer when USE_SERVICE_LAYER is enabled,
        otherwise falls through to ToolCoordinator.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            context: Execution context

        Returns:
            Tuple of (result, success, error_message or None)
        """
        if self._tool_service:
            return await self._tool_service.execute_tool_with_retry(tool_name, tool_args, context)
        return await self._tool_coordinator.execute_tool_with_retry(tool_name, tool_args, context)

    async def _handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle tool calls from the model.

        Delegates execution to ToolPipeline, then post-processes results
        via ToolCoordinator.process_tool_results().

        Args:
            tool_calls: List of tool call requests

        Returns:
            List of result dicts with keys: name, success, elapsed, args,
            error, follow_up_suggestions.
        """
        if not tool_calls:
            return []

        # Delegate execution to ToolPipeline
        pipeline_result = await self._tool_pipeline.execute_tool_calls(
            tool_calls=tool_calls,
            context=self._get_tool_context(),
        )

        # Sync budget from pipeline
        self.tool_calls_used = self._tool_pipeline.calls_used

        # Delegate post-processing to ToolCoordinator
        from victor.agent.coordinators.tool_coordinator import ToolResultContext

        ctx = ToolResultContext(
            executed_tools=self.executed_tools,
            observed_files=self.observed_files,
            failed_tool_signatures=self.failed_tool_signatures,
            shown_tool_errors=self._shown_tool_errors,
            continuation_prompts=self._continuation_prompts,
            asking_input_prompts=self._asking_input_prompts,
            tool_calls_used=self.tool_calls_used,
            record_tool_execution=self._record_tool_execution,
            conversation_state=self.conversation_state,
            unified_tracker=self.unified_tracker,
            usage_logger=self.usage_logger,
            add_message=self.add_message,
            format_tool_output=self._format_tool_output,
            console=self.console,
            presentation=self._presentation,
            stream_context=(
                self._current_stream_context if hasattr(self, "_current_stream_context") else None
            ),
        )

        results = self._tool_coordinator.process_tool_results(pipeline_result, ctx)

        # Sync mutable state back from context
        self._continuation_prompts = ctx.continuation_prompts
        self._asking_input_prompts = ctx.asking_input_prompts

        return results

    def _get_tool_context(self) -> dict:
        """Get cached tool execution context dict.

        Reuses the same dict across tool calls within a conversation turn
        to avoid repeated allocation in the hot path.
        """
        if self._tool_context_cache is None:
            self._tool_context_cache = {
                "code_manager": self.code_manager,
                "provider": self.provider,
                "model": self.model,
                "tool_registry": self.tools,
                "workflow_registry": self.workflow_registry,
                "settings": self.settings,
            }
        return self._tool_context_cache

    def reset_conversation(self) -> None:
        """Clear conversation history and session state.

        Delegates to LifecycleManager for core reset logic.

        Resets:
        - Conversation history
        - Tool call counter
        - Failed tool signatures cache
        - Observed files list
        - Executed tools list
        - Conversation state machine
        - Context reminder manager
        - Metrics collector stats
        - Context compactor statistics
        - Sequence tracker history (preserves learned patterns)
        - Usage analytics session (ends current, starts fresh)
        """
        # Delegate to LifecycleManager for core reset
        self._lifecycle_manager.reset_conversation()

        # Reset orchestrator-specific state
        self._system_added = False
        self.tool_calls_used = 0
        self.failed_tool_signatures.clear()
        self.observed_files.clear()
        self.executed_tools.clear()
        self._consecutive_blocked_attempts = 0
        self._total_blocked_attempts = 0
        self._shown_tool_errors.clear()
        self._tool_context_cache = None

        logger.debug("Conversation and session state reset (via LifecycleManager)")

    def request_cancellation(self) -> None:
        """Request cancellation of the current streaming operation.

        Safe to call even if not streaming. The cancellation will take
        effect at the next check point in the stream loop.
        """
        if self._cancel_event:
            self._cancel_event.set()
            logger.info("Cancellation requested for streaming operation")

    def is_streaming(self) -> bool:
        """Check if a streaming operation is currently in progress.

        Returns:
            True if streaming, False otherwise.
        """
        return self._is_streaming

    def _check_cancellation(self) -> bool:
        """Check if cancellation has been requested.

        Returns:
            True if cancelled, False otherwise.
        """
        if self._cancel_event and self._cancel_event.is_set():
            return True
        return False

    # =========================================================================
    # Conversation Memory Management
    # =========================================================================

    def get_memory_session_id(self) -> Optional[str]:
        """Get the current memory session ID.

        Returns:
            Session ID or None if memory manager not enabled.
        """
        return self._memory_session_id

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation sessions for recovery.

        Delegates to service layer when enabled, otherwise SessionCoordinator.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session metadata dictionaries
        """
        if self._session_service:
            return self._session_service.get_recent_sessions(limit)
        return self._session_coordinator.get_recent_sessions(limit)

    def recover_session(self, session_id: str) -> bool:
        """Recover a previous conversation session.

        Delegates to SessionCoordinator.

        Args:
            session_id: ID of the session to recover

        Returns:
            True if session was recovered successfully
        """
        success = self._session_coordinator.recover_session(session_id)
        if success:
            self._memory_session_id = session_id
        return success

    def get_memory_context(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get token-aware context messages from memory manager.

        Delegates to SessionCoordinator.

        Args:
            max_tokens: Override max tokens for this retrieval.

        Returns:
            List of messages in provider format.
        """
        return self._session_coordinator.get_memory_context(
            max_tokens=max_tokens,
            messages=self.messages,
        )

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for the current memory session.

        Delegates to service layer when enabled, otherwise SessionCoordinator.

        Returns:
            Dictionary with session statistics.
        """
        if self._session_service:
            return self._session_service.get_session_stats()
        return self._session_coordinator.get_session_stats()

    async def shutdown(self) -> None:
        """Clean up resources and shutdown gracefully.

        Delegates to LifecycleManager for core shutdown logic.

        Should be called when the orchestrator is no longer needed.
        Cleans up:
        - Background async tasks
        - Provider connections
        - Code execution manager (Docker containers)
        - Semantic selector resources
        - HTTP clients
        """
        # Delegate to LifecycleManager for shutdown
        await self._lifecycle_manager.shutdown()
        logger.info("AgentOrchestrator shutdown complete")

    # =========================================================================
    # Protocol Conformance Methods (OrchestratorProtocol)
    # =========================================================================
    # These methods implement the stable interface contract defined in
    # victor/framework/protocols.py, enabling the framework layer to
    # interact with the orchestrator without duck-typing.

    # --- ConversationStateProtocol ---

    def get_stage(self) -> "ConversationStage":
        """Get current conversation stage (protocol method).

        Returns:
            Current ConversationStage enum value

        Note:
            Framework layer converts this to framework.state.Stage
        """
        if self.conversation_state:
            return self.conversation_state.get_stage()
        return ConversationStage.INITIAL

    def get_tool_calls_count(self) -> int:
        """Get total tool calls made (protocol method).

        Returns:
            Non-negative count of tool calls in this session
        """
        if self.unified_tracker:
            return self.unified_tracker.tool_calls_used
        return getattr(self, "tool_calls_used", 0)

    def get_tool_budget(self) -> int:
        """Get tool call budget (protocol method).

        Returns:
            Maximum allowed tool calls
        """
        if self.unified_tracker:
            return self.unified_tracker.tool_budget
        return getattr(self, "tool_budget", 50)

    def get_observed_files(self) -> Set[str]:
        """Get files observed/read during conversation (protocol method).

        Returns:
            Set of absolute file paths
        """
        return set(getattr(self, "observed_files", []))

    def get_modified_files(self) -> Set[str]:
        """Get files modified during conversation (protocol method).

        Returns:
            Set of absolute file paths
        """
        if self.conversation_state and hasattr(self.conversation_state, "state"):
            return set(getattr(self.conversation_state.state, "modified_files", []))
        return set()

    def get_iteration_count(self) -> int:
        """Get current agent loop iteration count (protocol method).

        Returns:
            Non-negative iteration count
        """
        if self.unified_tracker:
            return self.unified_tracker.iteration_count
        return 0

    def get_max_iterations(self) -> int:
        """Get maximum allowed iterations (protocol method).

        Returns:
            Max iteration limit
        """
        if self.unified_tracker:
            return self.unified_tracker.max_iterations
        return 25

    # --- ProviderProtocol ---

    @property
    def current_provider(self) -> str:
        """Get current provider name (protocol property).

        Returns:
            Provider identifier (e.g., "anthropic", "openai")
        """
        return self.provider_name

    @property
    def current_model(self) -> str:
        """Get current model name (protocol property).

        Returns:
            Model identifier
        """
        return self.model

    async def switch_provider(
        self,
        provider_name: str,
        model: Optional[str] = None,
        on_switch: Optional[Any] = None,
    ) -> bool:
        """Switch to a different provider/model (protocol method).

        Delegates to ProviderManager's async switch_provider directly
        for proper exception handling in async context (Phase 2 refactoring fix).

        Args:
            provider_name: Target provider name
            model: Optional specific model
            on_switch: Optional callback(provider_name, model) after switch

        Returns:
            True if switch was successful, False otherwise

        Raises:
            ProviderNotFoundError: If provider not found
        """
        if self._provider_service:
            await self._provider_service.switch_provider(provider_name, model)
        else:
            await self._provider_coordinator.switch_provider_async(
                provider_name=provider_name, model=model,
            )
        self.model = self._provider_manager.model
        self.provider_name = self._provider_manager.provider_name
        if on_switch:
            on_switch(self.provider_name, self.model)
        return True

    async def switch_model(self, model: str) -> bool:
        """Switch to a different model on the current provider (protocol method).

        Delegates to ProviderManager's async switch_model directly
        for proper exception handling in async context (Phase 2 refactoring fix).

        Args:
            model: Target model name

        Returns:
            True if switch was successful, False otherwise
        """
        result = await self._provider_coordinator.switch_model_async(model)
        if result:
            # Sync orchestrator's model attribute with provider manager state
            self.model = self._provider_manager.model
        return result

    # --- ToolsProtocol ---

    def get_available_tools(self) -> Set[str]:
        """Get all registered tool names (protocol method).

        Delegates to service layer when enabled, otherwise ToolCoordinator.

        Returns:
            Set of tool names available in registry
        """
        if self._tool_service:
            return self._tool_service.get_available_tools()
        return self._tool_coordinator.get_available_tools()

    def _build_tool_access_context(self) -> "ToolAccessContext":
        """Build ToolAccessContext for unified access control checks.

        Delegates to ToolCoordinator.

        Returns:
            ToolAccessContext with session tools and current mode
        """
        return self._tool_coordinator._build_tool_access_context()

    def get_enabled_tools(self) -> Set[str]:
        """Get currently enabled tool names (protocol method).

        Delegates to service layer when enabled, otherwise ToolCoordinator.

        Returns:
            Set of enabled tool names for this session
        """
        if self._tool_service:
            return self._tool_service.get_enabled_tools()
        return self._tool_coordinator.get_enabled_tools()

    def set_enabled_tools(self, tools: Set[str], tiered_config: Any = None) -> None:
        """Set which tools are enabled for this session (protocol method).

        Delegates core logic to service layer when enabled, otherwise
        ToolCoordinator with orchestrator-specific propagation.

        Args:
            tools: Set of tool names to enable
            tiered_config: Optional TieredToolConfig to propagate for stage filtering.
        """
        self._enabled_tools = tools
        if self._tool_service:
            self._tool_service.set_enabled_tools(tools)
        elif hasattr(self, "_tool_coordinator") and self._tool_coordinator.initialized:
            self._tool_coordinator.set_enabled_tools(tools)

        # Apply to vertical context and tool access controller
        self._apply_vertical_tools(tools)

        # Propagate TieredToolConfig for stage-aware filtering
        if hasattr(self, "tool_selector") and self.tool_selector:
            if tiered_config is None:
                tiered_config = self._get_vertical_tiered_config()
            if tiered_config is not None:
                self.tool_selector.set_tiered_config(tiered_config)
                logger.info(
                    f"Tiered config propagated to selector: "
                    f"mandatory={sorted(tiered_config.mandatory)}, "
                    f"vertical_core={sorted(tiered_config.vertical_core)}"
                )

    def _get_vertical_tiered_config(self) -> Any:
        """Get TieredToolConfig from active vertical canonical API."""
        from victor.agent.vertical_context import VerticalContext

        return VerticalContext.get_vertical_tiered_config()

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled (protocol method).

        Delegates to service layer when enabled, otherwise ToolCoordinator.

        Args:
            tool_name: Name of tool to check

        Returns:
            True if tool is enabled
        """
        if self._tool_service:
            return self._tool_service.is_tool_enabled(tool_name)
        return self._tool_coordinator.is_tool_enabled(tool_name)

    # --- SystemPromptProtocol ---

    def get_system_prompt(self) -> str:
        """Get current system prompt (protocol method).

        Returns:
            Complete system prompt string
        """
        if self.prompt_builder:
            return self.prompt_builder.build()
        return ""

    def set_system_prompt(self, prompt: str) -> None:
        """Set custom system prompt (protocol method).

        Args:
            prompt: New system prompt (replaces existing)
        """
        if self.prompt_builder and hasattr(self.prompt_builder, "set_custom_prompt"):
            self.prompt_builder.set_custom_prompt(prompt)

    def append_to_system_prompt(self, content: str) -> None:
        """Append content to system prompt (protocol method).

        Args:
            content: Content to append
        """
        current = self.get_system_prompt()
        self.set_system_prompt(current + "\n\n" + content)

    # --- MessagesProtocol ---

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get conversation messages (protocol method).

        Returns:
            List of message dictionaries
        """
        return [{"role": m.role, "content": m.content} for m in self.conversation.messages]

    def get_message_count(self) -> int:
        """Get message count (protocol method).

        Returns:
            Number of messages in conversation
        """
        return len(self.conversation.messages)

    # --- Lifecycle Methods ---
    # Note: is_streaming() already exists at line ~5285
    # Note: reset() is provided via reset_conversation() at line ~5221

    def reset(self) -> None:
        """Reset conversation state (protocol method).

        Alias for reset_conversation() for protocol conformance.
        """
        self.reset_conversation()

    async def __aenter__(self) -> "AgentOrchestrator":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - ensures cleanup."""
        await self.shutdown()

    @classmethod
    async def from_settings(
        cls,
        settings: Settings,
        profile_name: str = "default",
        thinking: bool = False,
    ) -> "AgentOrchestrator":
        """Create orchestrator from settings. Delegates to orchestrator_creation module."""
        from victor.agent.orchestrator_creation import create_orchestrator_from_settings

        return await create_orchestrator_from_settings(
            cls, settings, profile_name=profile_name, thinking=thinking
        )


# Install extracted property descriptors onto the class.
# This runs once at import time and keeps full mock.patch.object compatibility.
from victor.agent.orchestrator_properties import (
    install_properties as _install_properties,
)

_install_properties(AgentOrchestrator)
del _install_properties
