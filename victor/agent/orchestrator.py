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
    from victor.agent.streaming_controller import StreamingController
    from victor.agent.task_analyzer import TaskAnalyzer
    from victor.agent.tool_registrar import ToolRegistrar
    from victor.agent.provider_manager import ProviderManager
    from victor.agent.provider_coordinator import ProviderCoordinator
    from victor.agent.tool_selection import ToolSelector
    from victor.agent.tool_executor import ToolExecutor
    from victor.agent.safety import SafetyChecker
    from victor.agent.auto_commit import AutoCommitter
    from victor.agent.conversation_manager import ConversationManager, create_conversation_manager

# Runtime imports - used for instantiation, enums, constants, or function calls
from victor.agent.argument_normalizer import ArgumentNormalizer, NormalizationStrategy
from victor.agent.message_history import MessageHistory
from victor.agent.conversation_memory import ConversationStore, MessageRole

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
from victor.agent.session_state_manager import SessionStateManager, create_session_state_manager

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
from victor.agent.streaming_controller import StreamingControllerConfig, StreamingSession
from victor.agent.task_analyzer import get_task_analyzer
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
from victor.agent.response_completer import ToolFailureContext, create_response_completer
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
from victor.storage.embeddings.intent_classifier import IntentClassifier, IntentType
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

        # ProviderCoordinator: Wraps ProviderManager with rate limiting and health monitoring (TD-002)
        from victor.agent.provider_coordinator import (
            ProviderCoordinator,
            ProviderCoordinatorConfig,
        )

        self._provider_coordinator = ProviderCoordinator(
            provider_manager=self._provider_manager,
            config=ProviderCoordinatorConfig(
                max_rate_limit_retries=getattr(settings, "max_rate_limit_retries", 3),
                enable_health_monitoring=getattr(settings, "provider_health_checks", True),
            ),
        )

        # ProviderSwitchCoordinator: Coordinate provider/model switching workflow (via factory)
        # Wraps ProviderSwitcher with validation, health checks, retry logic
        self._provider_switch_coordinator = self._factory.create_provider_switch_coordinator(
            provider_switcher=self._provider_manager._provider_switcher,
            health_monitor=self._provider_manager._health_monitor,
        )

        # Response sanitizer for cleaning model output (via factory - DI with fallback)
        self.sanitizer = self._factory.create_sanitizer()

        # System prompt builder with vertical prompt contributors (via factory)
        self.prompt_builder = self._factory.create_system_prompt_builder(
            provider_name=self.provider_name,
            model=model,
            tool_adapter=self.tool_adapter,
            tool_calling_caps=self.tool_calling_caps,
        )

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

        # Initialize tool call budget (via factory) - uses adapter recommendations with settings override
        self.tool_budget = self._factory.initialize_tool_budget(self.tool_calling_caps)

        # Initialize SessionStateManager for consolidated execution state tracking (TD-002)
        # Replaces scattered state variables: tool_calls_used, observed_files, executed_tools,
        # failed_tool_signatures, _read_files_session, _required_files, _required_outputs, etc.
        self._session_state = create_session_state_manager(tool_budget=self.tool_budget)

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

        # Analytics - usage logger with DI support (via factory)
        self.usage_logger = self._factory.create_usage_logger()
        self.usage_logger.log_event(
            "session_start", {"model": self.model, "provider": self.provider.__class__.__name__}
        )

        # Streaming metrics collector for performance monitoring (via factory)
        self.streaming_metrics_collector = self._factory.create_streaming_metrics_collector()
        if self.streaming_metrics_collector:
            logger.info("StreamingMetricsCollector initialized via factory")

        # Debug logger for incremental output and conversation tracking (via factory)
        self.debug_logger = self._factory.create_debug_logger_configured()

        # Cancellation support for streaming
        self._cancel_event: Optional[asyncio.Event] = None
        self._is_streaming = False

        # Background task tracking for graceful shutdown
        self._background_tasks: set[asyncio.Task] = set()

        # Metrics collection (via factory)
        self._metrics_collector = self._factory.create_metrics_collector(
            streaming_metrics_collector=self.streaming_metrics_collector,
            usage_logger=self.usage_logger,
            debug_logger=self.debug_logger,
            tool_cost_lookup=lambda name: (
                self.tools.get_tool_cost(name) if hasattr(self, "tools") else CostTier.FREE
            ),
        )

        # Session cost tracking (for LLM API cost monitoring)
        from victor.agent.session_cost_tracker import SessionCostTracker

        self._session_cost_tracker = SessionCostTracker(
            provider=self.provider.name,
            model=self.model,
        )

        # Metrics coordinator (Phase 2 refactoring - aggregates metrics/cost/token tracking)
        from victor.agent.coordinators.metrics_coordinator import MetricsCoordinator

        self._metrics_coordinator = MetricsCoordinator(
            metrics_collector=self._metrics_collector,
            session_cost_tracker=self._session_cost_tracker,
            cumulative_token_usage=self._cumulative_token_usage,
        )

        # Result cache for pure/idempotent tools (via factory)
        self.tool_cache = self._factory.create_tool_cache()
        # Minimal dependency graph (used for planning search→read→analyze) (via factory, DI)
        # Tool dependencies are registered via ToolRegistrar after it's created
        self.tool_graph = self._factory.create_tool_dependency_graph()

        # Stateful managers (DI with fallback)
        # Code execution manager for Docker-based code execution (via factory, DI with fallback)
        self.code_manager = self._factory.create_code_execution_manager()

        # Workflow registry (via factory, DI with fallback)
        self.workflow_registry = self._factory.create_workflow_registry()
        self._register_default_workflows()

        # Conversation history (via factory) - MessageHistory for better encapsulation
        self.conversation = self._factory.create_message_history(self._system_prompt)

        # Persistent conversation memory with SQLite backing (via factory)
        # Provides session recovery, token-aware pruning, and multi-turn context retention
        self.memory_manager, self._memory_session_id = self._factory.create_memory_components(
            self.provider_name, self.tool_calling_caps.native_tool_calls
        )

        # Initialize LanceDB embedding store for efficient semantic retrieval if memory enabled
        if self.memory_manager and getattr(settings, "conversation_embeddings_enabled", True):
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

        # Conversation state machine for intelligent stage detection (via factory, DI with fallback)
        self.conversation_state = self._factory.create_conversation_state_machine()

        # Intent classifier for semantic continuation/completion detection (via factory)
        self.intent_classifier = self._factory.create_intent_classifier()

        # Intelligent pipeline integration (lazy initialization via factory)
        # Provides RL-based mode learning, quality scoring, prompt optimization
        self._intelligent_integration: Optional["OrchestratorIntegration"] = None
        self._intelligent_integration_config = self._factory.create_integration_config()
        self._intelligent_pipeline_enabled = getattr(settings, "intelligent_pipeline_enabled", True)

        # Sub-agent orchestration (via factory) - lazy initialization for parallel task delegation
        self._subagent_orchestrator, self._subagent_orchestration_enabled = (
            self._factory.setup_subagent_orchestration()
        )

        # Tool registry (via factory)
        self.tools = self._factory.create_tool_registry()
        # Alias for backward compatibility - some code uses tool_registry instead of tools
        self.tool_registry = self.tools

        # Initialize ToolRegistrar (via factory) - tool registration, plugins, MCP integration
        self.tool_registrar = self._factory.create_tool_registrar(
            self.tools, self.tool_graph, provider, model
        )
        self.tool_registrar.set_background_task_callback(self._create_background_task)

        # Register tool dependencies for planning (delegates to ToolRegistrar)
        self.tool_registrar._register_tool_dependencies()

        # Synchronous registration (dynamic tools, configs)
        self._register_default_tools()  # Delegates to ToolRegistrar
        self.tool_registrar._load_tool_configurations()  # Delegates to ToolRegistrar
        self.tools.register_before_hook(self._log_tool_call)

        # Plugin system for extensible tools (via factory, delegates to ToolRegistrar)
        self.plugin_manager = self._factory.initialize_plugin_system(self.tool_registrar)

        # Argument normalizer for handling malformed tool arguments (via factory, DI with fallback)
        self.argument_normalizer = self._factory.create_argument_normalizer(provider)

        # Initialize middleware chain from vertical extensions (via factory)
        # Middleware provides code validation, safety checks, and domain-specific processing
        self._middleware_chain, self._code_correction_middleware = (
            self._factory.create_middleware_chain()
        )

        # Initialize SafetyChecker with vertical patterns (via factory)
        # Exposes via property for UI layer to set confirmation callback
        self._safety_checker = self._factory.create_safety_checker()

        # Initialize AutoCommitter for AI-assisted commits (via factory)
        # Provides conventional commits with co-authorship attribution
        self._auto_committer = self._factory.create_auto_committer()

        # Tool executor for centralized tool execution with retry, caching, and metrics (via factory)
        self.tool_executor = self._factory.create_tool_executor(
            tools=self.tools,
            argument_normalizer=self.argument_normalizer,
            tool_cache=self.tool_cache,
            safety_checker=self._safety_checker,
            code_correction_middleware=self._code_correction_middleware,
        )

        # Parallel tool executor for concurrent independent tool calls (via factory)
        self.parallel_executor = self._factory.create_parallel_executor(self.tool_executor)

        # Response completer for ensuring complete responses after tool calls (via factory)
        self.response_completer = self._factory.create_response_completer()

        # Semantic tool selector (via factory - optional, configured via settings)
        self.use_semantic_selection, self._embedding_preload_task = (
            self._factory.setup_semantic_selection()
        )
        self.semantic_selector = self._factory.create_semantic_selector()

        # Initialize UnifiedTaskTracker (via factory, DI with fallback)
        # This is the single source of truth for task progress, milestones, and loop detection
        self.unified_tracker = self._factory.create_unified_tracker(self.tool_calling_caps)

        # Initialize unified ToolSelector (via factory) - semantic + keyword selection
        self.tool_selector = self._factory.create_tool_selector(
            tools=self.tools,
            semantic_selector=self.semantic_selector,
            conversation_state=self.conversation_state,
            unified_tracker=self.unified_tracker,
            model=self.model,
            provider_name=self.provider_name,
            tool_selection=self.tool_selection,
            on_selection_recorded=self._record_tool_selection,
        )

        # Initialize ToolAccessController for unified tool access control (via factory)
        # Replaces scattered is_tool_enabled/get_enabled_tools logic with layered precedence
        self._tool_access_controller = self._factory.create_tool_access_controller(
            registry=self.tools,
        )

        # Initialize BudgetManager for unified budget tracking (via factory)
        # Centralizes budget management with consistent multiplier composition
        self._budget_manager = self._factory.create_budget_manager()

        # =================================================================
        # NEW: Decomposed component facades for orchestrator responsibilities
        # These provide cleaner interfaces while maintaining backward compatibility
        # =================================================================

        # ConversationController: Manages message history and conversation state (via factory)
        self._conversation_controller = self._factory.create_conversation_controller(
            provider=provider,
            model=model,
            conversation=self.conversation,
            conversation_state=self.conversation_state,
            memory_manager=self.memory_manager,
            memory_session_id=self._memory_session_id,
            system_prompt=self._system_prompt,
        )

        # LifecycleManager: Coordinate session lifecycle and resource cleanup (via factory)
        # Handles conversation reset, session recovery, graceful shutdown
        # Must be created AFTER conversation_controller and other dependencies
        self._lifecycle_manager = self._factory.create_lifecycle_manager(
            conversation_controller=self._conversation_controller,
            metrics_collector=(
                self._metrics_collector if hasattr(self, "_metrics_collector") else None
            ),
            context_compactor=(
                self._context_compactor if hasattr(self, "_context_compactor") else None
            ),
            sequence_tracker=self._sequence_tracker if hasattr(self, "_sequence_tracker") else None,
            usage_analytics=self._usage_analytics if hasattr(self, "_usage_analytics") else None,
            reminder_manager=self._reminder_manager if hasattr(self, "_reminder_manager") else None,
        )

        # Tool deduplication tracker for preventing redundant calls (via factory)
        self._deduplication_tracker = self._factory.create_tool_deduplication_tracker()

        # ToolPipeline: Coordinates tool execution flow (via factory)
        # Middleware chain enables vertical-specific tool processing (code correction, validation, etc.)
        self._tool_pipeline = self._factory.create_tool_pipeline(
            tools=self.tools,
            tool_executor=self.tool_executor,
            tool_budget=self.tool_budget,
            tool_cache=self.tool_cache,
            argument_normalizer=self.argument_normalizer,
            on_tool_start=self._on_tool_start_callback,
            on_tool_complete=self._on_tool_complete_callback,
            deduplication_tracker=self._deduplication_tracker,
            middleware_chain=self._middleware_chain,
        )

        # Wire pending semantic cache to tool pipeline (deferred from embedding store init)
        if hasattr(self, "_pending_semantic_cache") and self._pending_semantic_cache is not None:
            self._tool_pipeline.set_semantic_cache(self._pending_semantic_cache)
            logger.info("[AgentOrchestrator] Semantic tool result cache enabled")
            self._pending_semantic_cache = None  # Clear reference

        # StreamingController: Manages streaming sessions and metrics (via factory)
        self._streaming_controller = self._factory.create_streaming_controller(
            streaming_metrics_collector=self.streaming_metrics_collector,
            on_session_complete=self._on_streaming_session_complete,
        )

        # StreamingCoordinator: Coordinates streaming response processing (via factory)
        self._streaming_coordinator = self._factory.create_streaming_coordinator(
            streaming_controller=self._streaming_controller,
        )

        # StreamingChatHandler: Testable extraction of streaming loop logic (via factory)
        self._streaming_handler = self._factory.create_streaming_chat_handler(message_adder=self)

        # IterationCoordinator: Loop control for streaming chat (using handler)
        self._iteration_coordinator: Optional[IterationCoordinator] = None

        # TaskAnalyzer: Unified task analysis facade
        self._task_analyzer = get_task_analyzer()

        # RLCoordinator: Framework-level RL with unified SQLite storage (via factory)
        self._rl_coordinator = self._factory.create_rl_coordinator()

        # Get context pruning learner from RL coordinator (if available)
        pruning_learner = None
        if self._rl_coordinator is not None:
            try:
                pruning_learner = self._rl_coordinator.get_learner("context_pruning")
            except KeyError:
                logger.debug("context_pruning learner not registered in RL coordinator")
            except AttributeError:
                logger.debug("RL coordinator interface mismatch (missing get_learner)")

        # ContextCompactor: Proactive context management and tool result truncation (via factory)
        self._context_compactor = self._factory.create_context_compactor(
            conversation_controller=self._conversation_controller,
            pruning_learner=pruning_learner,
        )

        # ContextManager: Centralized context window management (TD-002 refactoring)
        # Consolidates _get_model_context_window, _get_max_context_chars,
        # _check_context_overflow, and _handle_compaction
        self._context_manager = create_context_manager(
            provider_name=self.provider_name,
            model=self.model,
            conversation_controller=self._conversation_controller,
            context_compactor=self._context_compactor,
            debug_logger=self.debug_logger,
            settings=self.settings,
        )

        # ToolOutputFormatter: LLM-context-aware formatting of tool results
        # (Initialization consolidated: the definitive initialization occurs
        # later in this method to avoid accidental double-initialization.)
        # Previous duplicate initialization removed to reduce boilerplate.

        # Initialize UsageAnalytics singleton for data-driven optimization (via factory)
        self._usage_analytics = self._factory.create_usage_analytics()

        # Token usage tracking now managed by SessionStateManager (TD-002)
        # Access via self._cumulative_token_usage property or self._session_state.get_token_usage()

        # Initialize ToolSequenceTracker for intelligent next-tool suggestions (via factory)
        self._sequence_tracker = self._factory.create_sequence_tracker()

        # Initialize ToolOutputFormatter for LLM-context-aware output formatting (via factory)
        self._tool_output_formatter = self._factory.create_tool_output_formatter(
            self._context_compactor
        )

        # Initialize RecoveryHandler for handling model failures and stuck states (via factory)
        self._recovery_handler = self._factory.create_recovery_handler()

        # Create recovery integration submodule for clean delegation (via factory)
        self._recovery_integration = self._factory.create_recovery_integration(
            self._recovery_handler
        )

        # Initialize RecoveryCoordinator for centralized recovery logic (via factory, DI)
        # Consolidates all recovery/error handling methods from orchestrator
        self._recovery_coordinator = self._factory.create_recovery_coordinator()

        # Initialize ChunkGenerator for centralized chunk generation (via factory, DI)
        # Consolidates all chunk generation methods from orchestrator
        self._chunk_generator = self._factory.create_chunk_generator()

        # Initialize ToolPlanner for centralized tool planning (via factory, DI)
        # Consolidates all tool planning methods from orchestrator
        self._tool_planner = self._factory.create_tool_planner()

        # Initialize TaskCoordinator for centralized task coordination (via factory, DI)
        # Consolidates task preparation, intent detection, and guidance methods
        self._task_coordinator = self._factory.create_task_coordinator()

        # Initialize ObservabilityIntegration for unified event bus (via factory)
        self._observability = self._factory.create_observability()

        # Initialize ConversationCheckpointManager for time-travel debugging (via factory)
        # Provides save/restore/fork capabilities for conversation state
        self._checkpoint_manager = self._factory.create_checkpoint_manager()

        # =================================================================
        # Workflow Optimization Components - MODE workflow optimizations
        # These address issues identified during EXPLORE/PLAN/BUILD testing
        # =================================================================

        # Initialize workflow optimization components (via factory)
        self._workflow_optimization = self._factory.create_workflow_optimization_components(
            timeout_seconds=getattr(settings, "execution_timeout", None)
        )

        # Wire component dependencies (via factory)
        self._factory.wire_component_dependencies(
            recovery_handler=self._recovery_handler,
            context_compactor=self._context_compactor,
            observability=self._observability,
            conversation_state=self.conversation_state,
        )

        # Initialize VerticalContext for unified vertical state management
        # This replaces scattered _vertical_* attributes with a proper container
        self._vertical_context: VerticalContext = create_vertical_context()

        # Initialize VerticalIntegrationAdapter for single-source vertical methods
        # Eliminates duplicate apply_vertical_* implementations (SRP compliance)
        self._vertical_integration_adapter = VerticalIntegrationAdapter(self)

        # Initialize ModeWorkflowTeamCoordinator for intelligent team/workflow suggestions
        # Lazy initialization pattern - coordinator is created on first access
        self._mode_workflow_team_coordinator: Optional[Any] = None

        # Initialize ChatCoordinator for delegated chat/stream_chat operations
        from victor.agent.coordinators.chat_coordinator import ChatCoordinator

        self._chat_coordinator = ChatCoordinator(self)

        # Initialize ToolCoordinator for delegated tool access operations
        from victor.agent.coordinators.tool_coordinator import ToolCoordinator as _ToolCoordinator

        self._tool_coordinator = _ToolCoordinator(
            tool_pipeline=self._tool_pipeline,
            tool_registry=self.tools,
            tool_selector=self.tool_selector if hasattr(self, "tool_selector") else None,
            tool_access_controller=getattr(self, "_tool_access_controller", None),
        )
        self._tool_coordinator.set_mode_controller(
            self.mode_controller if hasattr(self, "mode_controller") else None
        )
        self._tool_coordinator.set_orchestrator_reference(self)

        # Initialize SessionCoordinator for delegated session management
        from victor.agent.coordinators.session_coordinator import (
            SessionCoordinator as _SessionCoordinator,
            create_session_coordinator,
        )

        self._session_coordinator = create_session_coordinator(
            session_state_manager=self._session_state,
            lifecycle_manager=self._lifecycle_manager,
            memory_manager=self.memory_manager,
            checkpoint_manager=self._checkpoint_manager,
            cost_tracker=self._session_cost_tracker,
        )

        # Initialize capability registry for explicit capability discovery
        # This replaces hasattr duck-typing with type-safe protocol conformance
        self.__init_capability_registry__()

        # Wire up LifecycleManager with dependencies for shutdown
        # (must be done after all components are initialized)
        self._lifecycle_manager.set_provider(self.provider)
        self._lifecycle_manager.set_code_manager(
            self.code_manager if hasattr(self, "code_manager") else None
        )
        self._lifecycle_manager.set_semantic_selector(
            self.semantic_selector if hasattr(self, "semantic_selector") else None
        )
        self._lifecycle_manager.set_usage_logger(
            self.usage_logger if hasattr(self, "usage_logger") else None
        )
        # Note: background_tasks is a set, convert to list for lifecycle manager
        self._lifecycle_manager.set_background_tasks(list(self._background_tasks))
        # Set callbacks for orchestrator-specific shutdown logic
        self._lifecycle_manager.set_flush_analytics_callback(self.flush_analytics)
        self._lifecycle_manager.set_stop_health_monitoring_callback(self.stop_health_monitoring)

        # Debug-mode conformance assertion for ChatOrchestratorProtocol
        # Placed at end of __init__ so all attributes are initialized
        if __debug__:
            from victor.agent.coordinators.chat_protocols import ChatOrchestratorProtocol
            from victor.framework.protocols import verify_protocol_conformance

            _conforms, _missing = verify_protocol_conformance(self, ChatOrchestratorProtocol)
            assert (
                _conforms
            ), f"AgentOrchestrator missing ChatOrchestratorProtocol members: {_missing}"

        logger.info(
            "Orchestrator initialized with decomposed components: "
            "ConversationController, ToolPipeline, StreamingController, StreamingChatHandler, "
            "TaskAnalyzer, ContextCompactor, UsageAnalytics, ToolSequenceTracker, "
            "ToolOutputFormatter, RecoveryCoordinator, ChunkGenerator, ToolPlanner, TaskCoordinator, "
            "ObservabilityIntegration, WorkflowOptimization, VerticalContext, ModeWorkflowTeamCoordinator, "
            "CapabilityRegistry"
        )

    # =====================================================================
    # Callbacks for decomposed components
    # =====================================================================

    def _on_tool_start_callback(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Callback when tool execution starts (from ToolPipeline)."""
        iteration = self._tool_pipeline.calls_used if hasattr(self, "_tool_pipeline") else 0
        self._metrics_collector.on_tool_start(tool_name, arguments, iteration)

        # Emit observability event for tool start
        if hasattr(self, "_observability") and self._observability:
            tool_id = f"tool-{iteration}"
            self._observability.on_tool_start(tool_name, arguments, tool_id)

    def _on_tool_complete_callback(self, result: ToolCallResult) -> None:
        """Callback when tool execution completes (from ToolPipeline).

        Delegates to ToolCoordinator.on_tool_complete with orchestrator state.
        """
        # Use a mutable list as a flag so ToolCoordinator can update it
        nudge_flag = [getattr(self, "_all_files_read_nudge_sent", False)]

        self._tool_coordinator.on_tool_complete(
            result=result,
            metrics_collector=self._metrics_collector,
            read_files_session=self._read_files_session,
            required_files=self._required_files,
            required_outputs=self._required_outputs,
            nudge_sent_flag=nudge_flag,
            add_message=self.add_message,
            observability=getattr(self, "_observability", None),
            pipeline_calls_used=(
                self._tool_pipeline.calls_used if hasattr(self, "_tool_pipeline") else 0
            ),
        )

        # Sync the nudge flag back
        self._all_files_read_nudge_sent = nudge_flag[0]

    def _on_streaming_session_complete(self, session: StreamingSession) -> None:
        """Callback when streaming session completes (from StreamingController).

        This callback:
        1. Records metrics via MetricsCollector
        2. Ends UsageAnalytics session
        3. Sends RL reward signal to update provider Q-values
        """
        self._metrics_collector.on_streaming_session_complete(session)

        # End UsageAnalytics session
        if hasattr(self, "_usage_analytics") and self._usage_analytics:
            self._usage_analytics.end_session()

        # Send RL reward signal for Q-learning model selection
        self._send_rl_reward_signal(session)

    def _send_rl_reward_signal(self, session: StreamingSession) -> None:
        """Send reward signal to RL model selector. Delegates to MetricsCoordinator."""
        self._metrics_coordinator.send_rl_reward_signal(
            session=session,
            rl_coordinator=self._rl_coordinator,
            vertical_context=self._vertical_context,
        )

    @property
    def conversation_controller(self) -> "ConversationController":
        """Get the conversation controller component.

        Returns:
            ConversationController instance for managing conversation state
        """
        return self._conversation_controller

    @property
    def tool_pipeline(self) -> "ToolPipeline":
        """Get the tool pipeline component.

        Returns:
            ToolPipeline instance for coordinating tool execution
        """
        return self._tool_pipeline

    @property
    def streaming_controller(self) -> "StreamingController":
        """Get the streaming controller component.

        Returns:
            StreamingController instance for managing streaming sessions
        """
        return self._streaming_controller

    @property
    def streaming_handler(self) -> StreamingChatHandler:
        """Get the streaming chat handler component.

        Returns:
            StreamingChatHandler instance for testable streaming loop logic
        """
        return self._streaming_handler

    @property
    def task_analyzer(self) -> "TaskAnalyzer":
        """Get the task analyzer component.

        Returns:
            TaskAnalyzer instance for unified task analysis
        """
        return self._task_analyzer

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

    @property
    def provider_manager(self) -> "ProviderManager":
        """Get the provider manager component.

        Returns:
            ProviderManager instance for unified provider management
        """
        return self._provider_manager

    @property
    def context_compactor(self) -> "ContextCompactor":
        """Get the context compactor component.

        Returns:
            ContextCompactor instance for proactive context management
        """
        return self._context_compactor

    @property
    def tool_output_formatter(self) -> "ToolOutputFormatter":
        """Get the tool output formatter for LLM-context-aware formatting.

        Returns:
            ToolOutputFormatter instance for formatting tool outputs
        """
        return self._tool_output_formatter

    @property
    def usage_analytics(self) -> "UsageAnalytics":
        """Get the usage analytics singleton.

        Returns:
            UsageAnalytics instance for data-driven optimization
        """
        return self._usage_analytics

    @property
    def sequence_tracker(self) -> "ToolSequenceTracker":
        """Get the tool sequence tracker for intelligent next-tool suggestions.

        Returns:
            ToolSequenceTracker instance for pattern learning
        """
        return self._sequence_tracker

    @property
    def recovery_handler(self) -> Optional["RecoveryHandler"]:
        """Get the recovery handler for model failure recovery.

        Returns:
            RecoveryHandler instance or None if not enabled
        """
        return self._recovery_handler

    @property
    def recovery_integration(self) -> "OrchestratorRecoveryIntegration":
        """Get the recovery integration submodule.

        Returns:
            OrchestratorRecoveryIntegration for delegated recovery handling
        """
        return self._recovery_integration

    @property
    def recovery_coordinator(self) -> "StreamingRecoveryCoordinator":
        """Get the recovery coordinator for centralized recovery logic.

        The StreamingRecoveryCoordinator consolidates all recovery and error handling
        logic for streaming sessions, including condition checking, action
        handling, and recovery integration.

        Extracted from CRITICAL-001 Phase 2A.

        Returns:
            StreamingRecoveryCoordinator instance for recovery coordination
        """
        return self._recovery_coordinator

    @property
    def chunk_generator(self) -> "ChunkGenerator":
        """Get the chunk generator for streaming output.

        The ChunkGenerator provides a centralized interface for generating
        streaming chunks for various purposes (tool execution, status updates,
        metrics, content).

        Extracted from CRITICAL-001 Phase 2B.

        Returns:
            ChunkGenerator instance for chunk generation
        """
        return self._chunk_generator

    @property
    def tool_planner(self) -> "ToolPlanner":
        """Get the tool planner for tool planning operations.

        The ToolPlanner provides a centralized interface for tool planning,
        including goal inference, tool sequence planning, and intent-based
        filtering.

        Extracted from CRITICAL-001 Phase 2C.

        Returns:
            ToolPlanner instance for tool planning
        """
        return self._tool_planner

    @property
    def task_coordinator(self) -> "TaskCoordinator":
        """Get the task coordinator for task coordination operations.

        The TaskCoordinator provides a centralized interface for task
        preparation, intent detection, and task-specific guidance.

        Extracted from CRITICAL-001 Phase 2D.

        Returns:
            TaskCoordinator instance for task coordination
        """
        return self._task_coordinator

    @property
    def code_correction_middleware(self) -> Optional[Any]:
        """Get the code correction middleware for automatic code validation/fixing.

        Returns:
            CodeCorrectionMiddleware instance or None if not enabled
        """
        return self._code_correction_middleware

    # =====================================================================
    # Session state delegation properties (TD-002)
    # These delegate to SessionStateManager for consolidated state tracking
    # =====================================================================

    @property
    def session_state(self) -> SessionStateManager:
        """Get the session state manager.

        Returns:
            SessionStateManager instance for consolidated state tracking
        """
        return self._session_state

    @property
    def tool_calls_used(self) -> int:
        """Get the number of tool calls used in this session.

        Delegates to SessionStateManager.
        """
        return self._session_state.tool_calls_used

    @tool_calls_used.setter
    def tool_calls_used(self, value: int) -> None:
        """Set the number of tool calls used (for backward compatibility)."""
        self._session_state.execution_state.tool_calls_used = value

    @property
    def observed_files(self) -> Set[str]:
        """Get set of files observed/read during this session.

        Delegates to SessionStateManager.
        """
        return self._session_state.execution_state.observed_files

    @observed_files.setter
    def observed_files(self, value: Set[str]) -> None:
        """Set observed files (for checkpoint restore)."""
        self._session_state.execution_state.observed_files = set(value) if value else set()

    @property
    def executed_tools(self) -> List[str]:
        """Get list of executed tool names in order.

        Delegates to SessionStateManager.
        """
        return self._session_state.execution_state.executed_tools

    @executed_tools.setter
    def executed_tools(self, value: List[str]) -> None:
        """Set executed tools (for checkpoint restore)."""
        self._session_state.execution_state.executed_tools = list(value) if value else []

    @property
    def failed_tool_signatures(self) -> Set[Tuple[str, str]]:
        """Get set of (tool_name, args_hash) tuples for failed calls.

        Delegates to SessionStateManager.
        """
        return self._session_state.execution_state.failed_tool_signatures

    @failed_tool_signatures.setter
    def failed_tool_signatures(self, value: Set[Tuple[str, str]]) -> None:
        """Set failed tool signatures (for checkpoint restore)."""
        self._session_state.execution_state.failed_tool_signatures = set(value) if value else set()

    @property
    def _tool_capability_warned(self) -> bool:
        """Get whether we've warned about tool capability limitations.

        Delegates to SessionStateManager.
        """
        return self._session_state.session_flags.tool_capability_warned

    @_tool_capability_warned.setter
    def _tool_capability_warned(self, value: bool) -> None:
        """Set tool capability warning flag."""
        self._session_state.session_flags.tool_capability_warned = value

    @property
    def _read_files_session(self) -> Set[str]:
        """Get files read during this session for task completion detection.

        Delegates to SessionStateManager.
        """
        return self._session_state.execution_state.read_files_session

    @property
    def _required_files(self) -> List[str]:
        """Get required files extracted from user prompts.

        Delegates to SessionStateManager.
        """
        return self._session_state.execution_state.required_files

    @_required_files.setter
    def _required_files(self, value: List[str]) -> None:
        """Set required files list."""
        self._session_state.execution_state.required_files = list(value)

    @property
    def _required_outputs(self) -> List[str]:
        """Get required outputs extracted from user prompts.

        Delegates to SessionStateManager.
        """
        return self._session_state.execution_state.required_outputs

    @_required_outputs.setter
    def _required_outputs(self, value: List[str]) -> None:
        """Set required outputs list."""
        self._session_state.execution_state.required_outputs = list(value)

    @property
    def _all_files_read_nudge_sent(self) -> bool:
        """Get whether we've sent a nudge that all required files are read.

        Delegates to SessionStateManager.
        """
        return self._session_state.session_flags.all_files_read_nudge_sent

    @_all_files_read_nudge_sent.setter
    def _all_files_read_nudge_sent(self, value: bool) -> None:
        """Set all files read nudge flag."""
        self._session_state.session_flags.all_files_read_nudge_sent = value

    @property
    def _cumulative_token_usage(self) -> Dict[str, int]:
        """Get cumulative token usage for evaluation/benchmarking.

        Delegates to SessionStateManager.
        """
        return self._session_state.get_token_usage()

    @_cumulative_token_usage.setter
    def _cumulative_token_usage(self, value: Dict[str, int]) -> None:
        """Set cumulative token usage (for backward compatibility)."""
        self._session_state.execution_state.token_usage = dict(value)

    @property
    def intelligent_integration(self) -> Optional["OrchestratorIntegration"]:
        """Get the intelligent pipeline integration (lazy initialization).

        Use this for:
        - RL-based mode learning (explore → plan → build → review)
        - Response quality scoring
        - Provider resilience integration
        - Embedding-based prompt optimization

        Returns:
            OrchestratorIntegration instance or None if disabled or failed to initialize
        """
        if not self._intelligent_pipeline_enabled:
            return None

        if self._intelligent_integration is None:
            try:
                from victor.agent.orchestrator_integration import OrchestratorIntegration

                # Synchronous initialization (async version available via enhance_orchestrator)
                from victor.agent.intelligent_pipeline import IntelligentAgentPipeline

                # Determine project root for grounding verification
                # Context file is at .victor/init.md, so project root is grandparent
                from victor.config.settings import get_project_paths
                from victor.context.project_context import VICTOR_DIR_NAME

                if self.project_context.context_file:
                    # Context file: /project/.victor/init.md
                    # Parent: /project/.victor/ (not what we want)
                    # If parent is .victor dir, use grandparent as project root
                    parent_dir = self.project_context.context_file.parent
                    if parent_dir.name == VICTOR_DIR_NAME:
                        intelligent_project_root = str(parent_dir.parent)
                    else:
                        # Legacy case: context file directly in project root
                        intelligent_project_root = str(parent_dir)
                else:
                    # Fall back to project paths (which uses cwd)
                    intelligent_project_root = str(get_project_paths().project_root)

                pipeline = IntelligentAgentPipeline(
                    provider_name=self.provider_name,
                    model=self.model,
                    profile_name=f"{self.provider_name}:{self.model}",
                    project_root=intelligent_project_root,
                )
                self._intelligent_integration = OrchestratorIntegration(
                    orchestrator=self,
                    pipeline=pipeline,
                    config=self._intelligent_integration_config,
                )
                logger.info(
                    f"IntelligentPipeline initialized for {self.provider_name}:{self.model}"
                )
            except ImportError as e:
                logger.debug(f"IntelligentPipeline dependencies not available: {e}")
                self._intelligent_pipeline_enabled = False
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Failed to initialize IntelligentPipeline (config error): {e}")
                self._intelligent_pipeline_enabled = False

        return self._intelligent_integration

    @property
    def subagent_orchestrator(self) -> Optional["SubAgentOrchestrator"]:  # noqa: F821
        """Get the sub-agent orchestrator (lazy initialization).

        Use this for:
        - Spawning specialized sub-agents (researcher, planner, executor, etc.)
        - Parallel task delegation via fan_out()
        - Hierarchical task decomposition

        Returns:
            SubAgentOrchestrator instance or None if disabled or failed to initialize
        """
        if not self._subagent_orchestration_enabled:
            return None

        if self._subagent_orchestrator is None:
            try:
                from victor.agent.subagents import SubAgentOrchestrator

                self._subagent_orchestrator = SubAgentOrchestrator(parent=self)
                logger.info("SubAgentOrchestrator initialized")
            except ImportError as e:
                logger.debug(f"SubAgentOrchestrator module not available: {e}")
                self._subagent_orchestration_enabled = False
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Failed to initialize SubAgentOrchestrator (config error): {e}")
                self._subagent_orchestration_enabled = False

        return self._subagent_orchestrator

    @property
    def checkpoint_manager(self) -> Optional[Any]:
        """Get the checkpoint manager for time-travel debugging.

        Use this for:
        - Saving conversation state snapshots
        - Restoring to previous states
        - Forking sessions from checkpoints
        - Comparing state differences

        Returns:
            ConversationCheckpointManager instance or None if disabled
        """
        return self._checkpoint_manager

    @property
    def vertical_context(self) -> VerticalContext:
        """Get the vertical context for unified vertical state access.

        The VerticalContext provides structured access to all vertical-related
        configuration, replacing scattered _vertical_* attributes.

        Use this for:
        - Accessing vertical name and configuration
        - Querying enabled tools from vertical
        - Getting middleware, safety patterns, task hints
        - Mode configuration and tool dependencies

        Returns:
            VerticalContext instance (never None, may be empty)
        """
        return self._vertical_context

    @property
    def coordination(self) -> Any:
        """Get the mode-workflow-team coordinator (lazy initialization).

        The ModeWorkflowTeamCoordinator bridges agent modes, team specifications,
        and workflow definitions to provide intelligent suggestions for task execution.

        Use this for:
        - Getting team suggestions for complex tasks
        - Workflow recommendations based on task type
        - Mode-specific coordination configuration

        Returns:
            ModeWorkflowTeamCoordinator instance
        """
        if self._mode_workflow_team_coordinator is None:
            self._mode_workflow_team_coordinator = (
                self._factory.create_mode_workflow_team_coordinator(self._vertical_context)
            )
            logger.debug("ModeWorkflowTeamCoordinator initialized on first access")

        return self._mode_workflow_team_coordinator

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

        # Update system prompt if new context has content
        if self.project_context.content:
            base_prompt = self._build_system_prompt_with_adapter()
            self._system_prompt = (
                base_prompt + "\n\n" + self.project_context.get_system_prompt_addition()
            )
            logger.info(f"Loaded project context from {self.project_context.context_file}")

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

        Delegates to SessionCoordinator.

        Args:
            description: Human-readable description for the checkpoint
            tags: Optional tags for categorization

        Returns:
            Checkpoint ID if saved, None if checkpointing is disabled
        """
        return await self._session_coordinator.save_checkpoint(description, tags)

    async def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore conversation state from a checkpoint.

        Delegates to SessionCoordinator.

        Args:
            checkpoint_id: ID of checkpoint to restore

        Returns:
            True if restored successfully, False otherwise
        """
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
        self._vertical_middleware = middleware

    def _set_middleware_chain_storage(self, chain: Any) -> None:
        """Internal: Set middleware chain storage.

        DIP Compliance: Provides controlled setter instead of direct
        private attribute access. Called by VerticalIntegrationAdapter.

        Args:
            chain: MiddlewareChain instance
        """
        self._middleware_chain = chain

    def _set_safety_patterns_storage(self, patterns: List[Any]) -> None:
        """Internal: Set safety patterns storage.

        DIP Compliance: Provides controlled setter instead of direct
        private attribute access. Called by VerticalIntegrationAdapter.

        Args:
            patterns: List of safety pattern instances
        """
        self._vertical_safety_patterns = patterns

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

    def _check_context_overflow(self, max_context_chars: int = 200000) -> bool:
        """Check if context is at risk of overflow.

        Delegates to ContextManager (TD-002 refactoring).

        Args:
            max_context_chars: Maximum allowed context size in chars

        Returns:
            True if context is dangerously large
        """
        return self._context_manager.check_context_overflow(max_context_chars)

    def get_context_metrics(self) -> ContextMetrics:
        """Get detailed context metrics.

        Delegates to ContextManager (TD-002 refactoring).

        Returns:
            ContextMetrics with size and overflow information
        """
        return self._context_manager.get_context_metrics()

    def _init_conversation_embedding_store(self) -> None:
        """Initialize embedding store. Delegates to SessionCoordinator."""
        from victor.agent.coordinators.session_coordinator import SessionCoordinator

        store, cache = SessionCoordinator.init_conversation_embedding_store(
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
        """Create and track a background task for graceful shutdown.

        Args:
            coro: The coroutine to run as a background task.
            name: Name for the task (for logging).

        Returns:
            The created task, or None if no event loop is available.
        """
        try:
            loop = asyncio.get_event_loop()
            task = loop.create_task(coro, name=name)

            # Track the task
            self._background_tasks.add(task)

            # Remove from tracking when done
            task.add_done_callback(self._background_tasks.discard)

            logger.debug(f"Created background task: {name}")
            return task
        except RuntimeError:
            logger.debug(f"No event loop available for background task: {name}")
            return None

    def start_embedding_preload(self) -> None:
        """Start background embedding preload task.

        Should be called after orchestrator initialization to avoid blocking
        the main thread. Safe to call multiple times (no-op if already started).
        """
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
        self._metrics_collector.record_tool_selection(method, num_tools)

    def route_search_query(self, query: str) -> Dict[str, Any]:
        """Route a search query to the optimal search tool using SearchRouter.

        Analyzes the query to determine whether keyword search (code_search)
        or semantic search (semantic_code_search) would yield better results.

        Args:
            query: The search query

        Returns:
            Dictionary with routing recommendation:
                - recommended_tool: "code_search" or "semantic_code_search" or "both"
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

        return {
            "recommended_tool": tool_map.get(route.search_type, "code_search"),
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
            Tool name: "code_search", "semantic_code_search", or "both"
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

        Returns:
            Dictionary with usage analytics including:
            - Selection stats (semantic/keyword/fallback counts)
            - Per-tool execution stats (calls, success rate, timing)
            - Cost tracking (by tier and total)
            - Overall metrics
        """
        return self._metrics_collector.get_tool_usage_stats(
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
        # Delegate to LifecycleManager for graceful shutdown
        return await self._lifecycle_manager.graceful_shutdown()

    # =========================================================================
    # Provider/Model Hot-Swap Methods
    # =========================================================================

    def switch_provider(
        self,
        provider_name: str,
        model: Optional[str] = None,
        **provider_kwargs: Any,
    ) -> bool:
        """Switch to a different provider mid-conversation.

        .. deprecated::
            This method is deprecated. Use the async version instead:
            await orchestrator.switch_provider(provider_name, model)

        This synchronous method is kept for backward compatibility but
        delegates to ProviderCoordinator (Phase 2 refactoring).

        Args:
            provider_name: Name of the provider (ollama, lmstudio, anthropic, etc.)
            model: Optional model name. If not provided, uses current model.
            **provider_kwargs: Additional provider-specific arguments (base_url, api_key, etc.)

        Returns:
            True if switch was successful, False otherwise
        """
        import warnings

        warnings.warn(
            "switch_provider() is deprecated and will be removed. "
            "Use async switch_provider() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # For backward compatibility, run the async version in an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we can't use run_until_complete
                # Fall back to the old implementation for now
                logger.warning(
                    "switch_provider() called from async context - "
                    "deprecated sync path will be removed in future version"
                )
                return asyncio.run(self._provider_coordinator.switch_provider(provider_name, model))
            else:
                return asyncio.run(self._provider_coordinator.switch_provider(provider_name, model))
        except Exception as e:
            logger.error(f"Failed to switch provider to {provider_name}: {e}")
            return False

    def switch_model(self, model: str) -> bool:
        """Switch to a different model on the current provider.

        Delegates to ProviderCoordinator (Phase 2 refactoring).

        This is a lighter-weight switch than switch_provider() - it only
        updates the model and reinitializes the tool adapter.

        Args:
            model: New model name

        Returns:
            True if switch was successful, False otherwise

        Example:
            orchestrator.switch_model("qwen2.5-coder:32b")
        """
        import warnings

        # For backward compatibility with sync API, run the async version
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we can't use run_until_complete
                # Fall back to a simpler implementation
                logger.warning(
                    "switch_model() called from async context - "
                    "consider using await coordinator.switch_model() instead"
                )
                return asyncio.run(self._provider_coordinator.switch_model(model))
            else:
                return asyncio.run(self._provider_coordinator.switch_model(model))
        except Exception as e:
            logger.error(f"Failed to switch model to {model}: {e}")
            return False

    def get_current_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider and model.

        Combines ProviderManager's provider info with orchestrator-specific
        runtime state (tool budget, tool calls used).

        Returns:
            Dictionary with provider/model info and capabilities
        """
        # Get base info from ProviderManager
        info = self._provider_manager.get_info()

        # Add orchestrator-specific runtime state
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

    def _build_system_prompt_with_adapter(self) -> str:
        """Build system prompt using the tool calling adapter.

        Includes dynamic parallel read budget based on model's context window.
        """
        base_prompt = self.prompt_builder.build()

        # Calculate dynamic parallel read budget based on model context window
        context_window = self._get_model_context_window()
        budget = calculate_parallel_read_budget(context_window)

        # Inject dynamic budget hint for models with reasonable context
        # Only add for models with >= 32K context (smaller models benefit from sequential reads)
        if context_window >= 32768:
            budget_hint = budget.to_prompt_hint()
            final_prompt = f"{base_prompt}\n\n{budget_hint}"
        else:
            final_prompt = base_prompt

        # Emit prompt_used event for RL learning
        self._emit_prompt_used_event(final_prompt)

        return final_prompt

    def _emit_prompt_used_event(self, prompt: str) -> None:
        """Emit PROMPT_USED event for RL prompt template learner.

        Args:
            prompt: The final system prompt that was built
        """
        try:
            from victor.framework.rl.hooks import get_rl_hooks, RLEvent, RLEventType

            hooks = get_rl_hooks()
            if hooks is None:
                return

            # Determine prompt style based on provider type
            # Cloud providers use concise style, local uses detailed
            provider_name = getattr(self.provider, "name", "unknown")
            is_local = provider_name.lower() in {"ollama", "lmstudio", "vllm"}
            prompt_style = "detailed" if is_local else "structured"

            # Calculate prompt characteristics
            has_examples = "example" in prompt.lower() or "e.g." in prompt.lower()
            has_thinking = "step by step" in prompt.lower() or "think" in prompt.lower()
            has_constraints = "must" in prompt.lower() or "always" in prompt.lower()

            event = RLEvent(
                type=RLEventType.PROMPT_USED,
                success=True,  # Prompt was successfully built
                quality_score=0.5,  # Neutral until we get outcome feedback
                provider=provider_name,
                model=self.model,
                task_type="general",  # Will be updated with actual task type
                metadata={
                    "prompt_style": prompt_style,
                    "prompt_length": len(prompt),
                    "has_examples": has_examples,
                    "has_thinking_prompt": has_thinking,
                    "has_constraints": has_constraints,
                    "session_id": getattr(self, "_session_id", ""),
                },
            )
            hooks.emit(event)
            logger.debug(f"Emitted prompt_used event: style={prompt_style}")

        except Exception as e:
            # RL hook failure should never block prompt building
            logger.debug(f"Failed to emit prompt_used event: {e}")

    def _resolve_shell_variant(self, tool_name: str) -> str:
        """Resolve shell aliases to the appropriate enabled shell variant.

        LLMs often hallucinate shell tool names like 'run', 'bash', 'execute'.
        These map to 'shell' canonically, but in INITIAL stage only 'shell_readonly'
        may be enabled. This method resolves to whichever shell variant is available.

        This method now delegates to ToolAliasResolver for extensibility while
        maintaining backward compatibility with existing behavior.

        Args:
            tool_name: Original tool name (may be alias like 'run')

        Returns:
            The appropriate enabled shell tool name, or original if not a shell alias
        """
        # Shell-related aliases that should resolve intelligently
        # Also include shell_readonly so it can be upgraded to shell in BUILD mode
        shell_aliases = {"run", "bash", "execute", "cmd", "execute_bash", "shell_readonly", "shell"}

        if tool_name not in shell_aliases:
            return tool_name

        # Get alias resolver and register shell aliases with our custom resolver
        # We always register to ensure this orchestrator's resolver is used (handles
        # multiple orchestrator instances correctly by updating the resolver reference)
        resolver = get_alias_resolver()
        resolver.register(
            ToolNames.SHELL,
            aliases=list(shell_aliases - {ToolNames.SHELL}),
            resolver=self._shell_alias_resolver,
        )

        # Use the alias resolver - it will call our custom resolver
        return resolver.resolve(tool_name, enabled_tools=[])

    def _shell_alias_resolver(self, tool_name: str) -> str:
        """Custom resolver for shell aliases that checks mode and tool availability.

        This is registered with ToolAliasResolver to handle shell-related resolution.
        It encapsulates the mode-aware logic for choosing between shell and shell_readonly.

        Args:
            tool_name: The shell-related tool name being resolved.

        Returns:
            The appropriate shell variant based on mode and tool availability.
        """
        from victor.tools.tool_names import get_canonical_name

        # Check mode controller for BUILD mode (allows all tools including shell)
        # Uses ModeAwareMixin for consistent access
        mc = self.mode_controller
        if mc is not None:
            config = mc.config
            # If mode allows all tools and shell isn't explicitly disallowed, use full shell
            if config.allow_all_tools and "shell" not in config.disallowed_tools:
                logger.debug(f"Resolved '{tool_name}' to 'shell' (BUILD mode allows all tools)")
                return ToolNames.SHELL

        # Check if full shell is enabled first
        if self.tools.is_tool_enabled(ToolNames.SHELL):
            logger.debug(f"Resolved '{tool_name}' to 'shell' (shell enabled)")
            return ToolNames.SHELL

        # Fall back to shell_readonly if enabled
        if self.tools.is_tool_enabled(ToolNames.SHELL_READONLY):
            logger.debug(f"Resolved '{tool_name}' to 'shell_readonly' (readonly mode)")
            return ToolNames.SHELL_READONLY

        # Neither enabled - return canonical name (will fail validation)
        canonical = get_canonical_name(tool_name)
        logger.debug(f"No shell variant enabled for '{tool_name}', using canonical '{canonical}'")
        return canonical

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

        Delegates to TaskAnalyzer.classify_task_keywords().

        Args:
            user_message: The user's input message

        Returns:
            Dictionary with classification results (see TaskAnalyzer.classify_task_keywords)
        """
        return self._task_analyzer.classify_task_keywords(user_message)

    def _classify_task_with_context(
        self, user_message: str, history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Classify task with conversation context for improved accuracy.

        Delegates to TaskAnalyzer.classify_task_with_context().

        Args:
            user_message: The user's input message
            history: Optional conversation history for context boosting

        Returns:
            Dictionary with classification results (see TaskAnalyzer.classify_task_with_context)
        """
        return self._task_analyzer.classify_task_with_context(user_message, history)

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
        """Check provider/model combo against the capability matrix."""
        provider_key = self.provider_name or getattr(self.provider, "name", "")
        if not provider_key:
            return True

        supported = self.tool_capabilities.is_tool_call_supported(provider_key, self.model)
        if not supported and not self._tool_capability_warned:
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
        return supported

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        self.conversation.add_message(role, content)

        # Persist to memory manager if available
        if self.memory_manager and self._memory_session_id:
            try:
                role_map = {
                    "user": MessageRole.USER,
                    "assistant": MessageRole.ASSISTANT,
                    "system": MessageRole.SYSTEM,
                }
                msg_role = role_map.get(role, MessageRole.USER)
                self.memory_manager.add_message(
                    session_id=self._memory_session_id,
                    role=msg_role,
                    content=content,
                )
            except Exception as e:
                logger.debug(f"Failed to persist message to memory manager: {e}")

        if role == "user":
            self.usage_logger.log_event("user_prompt", {"content": content})
        elif role == "assistant":
            self.usage_logger.log_event("assistant_response", {"content": content})

    async def chat(self, user_message: str) -> CompletionResponse:
        """Send a chat message and get response with full agentic loop.

        Delegates to ChatCoordinator for the full agentic loop implementation.

        Args:
            user_message: User's message

        Returns:
            CompletionResponse from the model with complete response
        """
        return await self._chat_coordinator.chat(user_message)

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
        return await self._chat_coordinator.chat_with_planning(
            user_message, use_planning
        )

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

        Delegates to ChatCoordinator for the full streaming implementation.

        Args:
            user_message: User's input message

        Returns:
            AsyncIterator yielding StreamChunk objects with incremental response
        """
        async for chunk in self._chat_coordinator.stream_chat(user_message):
            yield chunk

    async def _execute_tool_with_retry(
        self, tool_name: str, tool_args: Dict[str, Any], context: Dict[str, Any]
    ) -> tuple[Any, bool, Optional[str]]:
        """Execute a tool with retry logic and exponential backoff.

        Delegates to ToolCoordinator.execute_tool_with_retry with orchestrator-specific
        execution backend and task completion detection callback.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            context: Execution context

        Returns:
            Tuple of (result, success, error_message or None)
        """

        async def _executor(name: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Any:
            return await self.tools.execute(name, context=ctx, **args)

        def _on_success(name: str, args: Dict[str, Any], result: Any) -> None:
            if self._task_completion_detector:
                tool_result = {"success": True}
                if "path" in args:
                    tool_result["path"] = args["path"]
                elif "file_path" in args:
                    tool_result["file_path"] = args["file_path"]
                self._task_completion_detector.record_tool_result(name, tool_result)

        return await self._tool_coordinator.execute_tool_with_retry(
            tool_name,
            tool_args,
            context,
            tool_executor=_executor,
            cache=self.tool_cache,
            on_success=_on_success,
            retry_config={
                "retry_enabled": getattr(self.settings, "tool_retry_enabled", True),
                "max_attempts": getattr(self.settings, "tool_retry_max_attempts", 3),
                "base_delay": getattr(self.settings, "tool_retry_base_delay", 1.0),
                "max_delay": getattr(self.settings, "tool_retry_max_delay", 10.0),
            },
        )

    async def _handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle tool calls from the model.

        Thin facade that delegates validation and normalization to
        ToolCoordinator, keeping state mutations and message injection local.

        Args:
            tool_calls: List of tool call requests
        """
        if not tool_calls:
            return []

        results: List[Dict[str, Any]] = []
        warn_icon = self._presentation.icon("warning", with_color=False)

        for tool_call in tool_calls:
            # --- Validate via ToolCoordinator ---
            validation = self._tool_coordinator.validate_tool_call(
                tool_call, self.sanitizer, is_tool_enabled_fn=self.is_tool_enabled
            )
            if not validation.valid:
                if validation.skip_reason:
                    self.console.print(f"[yellow]{warn_icon} {validation.skip_reason}[/]")
                if validation.error_result:
                    results.append(validation.error_result)
                continue

            # --- Budget check (reads orchestrator state) ---
            if self.tool_calls_used >= self.tool_budget:
                self.console.print(
                    f"[yellow]{warn_icon} Tool budget reached ({self.tool_budget}); skipping remaining tool calls.[/]"
                )
                break

            original_tool_name = validation.original_name
            tool_name = validation.canonical_name

            # --- Normalize via ToolCoordinator ---
            norm = self._tool_coordinator.normalize_arguments_full(
                tool_name,
                original_tool_name,
                tool_call.get("arguments", {}),
                self.argument_normalizer,
                self.tool_adapter,
                failed_signatures=self.failed_tool_signatures,
            )
            normalized_args = norm.args

            # Skip repeated failing calls
            if norm.is_repeated_failure:
                self.console.print(
                    f"[yellow]{warn_icon} Skipping repeated failing call to '{tool_name}' with same arguments[/]"
                )
                continue

            # Log normalization if applied
            if norm.strategy != NormalizationStrategy.DIRECT:
                logger.warning(
                    f"Applied {norm.strategy.value} normalization to {tool_name} arguments"
                )
                gear_icon = self._presentation.icon("gear", with_color=False)
                self.console.print(
                    f"[yellow]{gear_icon} Normalized arguments via {norm.strategy.value}[/]"
                )

            # --- Execution (uses orchestrator context) ---
            self.usage_logger.log_event(
                "tool_call", {"tool_name": tool_name, "tool_args": normalized_args}
            )
            logger.debug(f"Executing tool: {tool_name}")

            start = time.monotonic()

            context = {
                "code_manager": self.code_manager,
                "provider": self.provider,
                "model": self.model,
                "tool_registry": self.tools,
                "workflow_registry": self.workflow_registry,
                "settings": self.settings,
            }

            exec_result = await self.tool_executor.execute(
                tool_name=tool_name,
                arguments=normalized_args,
                context=context,
                skip_normalization=True,
            )
            success = exec_result.success
            error_msg = exec_result.error

            # Reset activity timer
            if hasattr(self, "_current_stream_context") and self._current_stream_context:
                self._current_stream_context.reset_activity_timer()

            # --- State updates (orchestrator-local) ---
            self.tool_calls_used += 1
            self.executed_tools.append(tool_name)
            if tool_name == "read" and "path" in normalized_args:
                self.observed_files.add(str(normalized_args.get("path")))

            # Reset continuation/input prompts on successful tool call
            if hasattr(self, "_continuation_prompts") and self._continuation_prompts > 0:
                logger.debug(
                    f"Resetting continuation prompts counter (was {self._continuation_prompts}) after successful tool call"
                )
                self._continuation_prompts = 0
                if hasattr(self, "_tool_calls_at_continuation_start"):
                    self._tool_calls_at_continuation_start = self.tool_calls_used
            if hasattr(self, "_asking_input_prompts") and self._asking_input_prompts > 0:
                logger.debug(
                    f"Resetting asking input prompts counter (was {self._asking_input_prompts}) after successful tool call"
                )
                self._asking_input_prompts = 0

            elapsed_ms = (time.monotonic() - start) * 1000

            error_type = (
                type(exec_result.error).__name__ if exec_result.error and not success else None
            )
            self._record_tool_execution(tool_name, success, elapsed_ms, error_type=error_type)
            self.conversation_state.record_tool_execution(tool_name, normalized_args)

            result_dict = {"success": success}
            if hasattr(exec_result, "result") and exec_result.result:
                result_dict["result"] = exec_result.result
            self.unified_tracker.update_from_tool_call(tool_name, normalized_args, result_dict)

            # --- Result formatting + conversation injection ---
            output = exec_result.result if success else None

            # Check for semantic failure
            semantic_success = success
            if success and isinstance(output, dict) and output.get("success") is False:
                semantic_success = False
                error_msg = output.get("error", "Operation returned success=False")

            error_display = None if semantic_success else (error_msg or "Unknown error")

            self.usage_logger.log_event(
                "tool_result",
                {
                    "tool_name": tool_name,
                    "success": semantic_success,
                    "result": output,
                    "error": error_display,
                },
            )

            if semantic_success:
                logger.debug(f"Tool {tool_name} executed successfully ({elapsed_ms:.0f}ms)")
                formatted_output = self._format_tool_output(tool_name, normalized_args, output)
                output_preview = str(output)[:500] if output else "<empty>"
                if len(str(output)) > 500:
                    output_preview += f"... [truncated, total {len(str(output))} chars]"
                logger.debug(f"Tool '{tool_name}' actual output:\n{output_preview}")

                self.add_message("user", formatted_output)
                results.append(
                    {
                        "name": tool_name,
                        "success": True,
                        "elapsed": time.monotonic() - start,
                        "args": normalized_args,
                    }
                )
            else:
                self.failed_tool_signatures.add(norm.signature)
                self.console.print(
                    f"[red]{self._presentation.icon('error', with_color=False)} Tool execution failed: {error_display}[/] [dim]({elapsed_ms:.0f}ms)[/dim]"
                )

                error_output = output if isinstance(output, dict) else {"error": error_display}
                formatted_error = self._format_tool_output(tool_name, normalized_args, error_output)
                self.add_message("user", formatted_error)
                logger.debug(f"Sent error feedback to model for {tool_name}: {error_display}")

                results.append(
                    {
                        "name": tool_name,
                        "success": False,
                        "elapsed": time.monotonic() - start,
                        "error": error_display,
                    }
                )
        return results

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

        Delegates to SessionCoordinator.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session metadata dictionaries
        """
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

        Delegates to SessionCoordinator.

        Returns:
            Dictionary with session statistics.
        """
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
        result = await self._provider_coordinator._manager.switch_provider(
            provider_name=provider_name,
            model=model,
        )

        if result:
            self._provider_coordinator._notify_post_switch_hooks()
            # Sync orchestrator's attributes with provider manager state
            self.model = self._provider_manager.model
            self.provider_name = self._provider_manager.provider_name
            if on_switch:
                on_switch(self.provider_name, self.model)

        return result

    async def switch_model(self, model: str) -> bool:
        """Switch to a different model on the current provider (protocol method).

        Delegates to ProviderManager's async switch_model directly
        for proper exception handling in async context (Phase 2 refactoring fix).

        Args:
            model: Target model name

        Returns:
            True if switch was successful, False otherwise
        """
        result = await self._provider_manager.switch_model(model)
        if result:
            self._provider_coordinator._notify_post_switch_hooks()
            # Sync orchestrator's model attribute with provider manager state
            self.model = self._provider_manager.model
        return result

    # --- ToolsProtocol ---

    def get_available_tools(self) -> Set[str]:
        """Get all registered tool names (protocol method).

        Delegates to ToolCoordinator.

        Returns:
            Set of tool names available in registry
        """
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

        Delegates to ToolCoordinator.

        Returns:
            Set of enabled tool names for this session
        """
        return self._tool_coordinator.get_enabled_tools()

    def set_enabled_tools(self, tools: Set[str], tiered_config: Any = None) -> None:
        """Set which tools are enabled for this session (protocol method).

        Delegates core logic to ToolCoordinator, handles orchestrator-specific
        propagation (vertical context, tiered config).

        Args:
            tools: Set of tool names to enable
            tiered_config: Optional TieredToolConfig to propagate for stage filtering.
        """
        self._enabled_tools = tools
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
        """Get TieredToolConfig from active vertical if available.

        Returns:
            TieredToolConfig or None
        """
        try:
            from victor.core.verticals.vertical_loader import get_vertical_loader

            loader = get_vertical_loader()
            if loader.active_vertical:
                return loader.active_vertical.get_tiered_tools()
        except Exception as e:
            logger.debug(f"Could not get tiered config from vertical: {e}")
        return None

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled (protocol method).

        Delegates to ToolCoordinator.

        Args:
            tool_name: Name of tool to check

        Returns:
            True if tool is enabled
        """
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
        """Create orchestrator from settings.

        Args:
            settings: Application settings
            profile_name: Profile to use
            thinking: Enable extended thinking mode (Claude models only)

        Returns:
            Configured AgentOrchestrator instance

        Note:
            The orchestrator reads settings.one_shot_mode to determine whether to
            auto-continue on ASKING_INPUT (one-shot) or return to user (interactive).
        """
        # Load profile
        profiles = settings.load_profiles()
        profile = profiles.get(profile_name)

        if not profile:
            available = list(profiles.keys())
            # Use difflib for similar name suggestions
            import difflib

            suggestions = difflib.get_close_matches(profile_name, available, n=3, cutoff=0.4)

            error_msg = f"Profile not found: '{profile_name}'"
            if suggestions:
                error_msg += f"\n  Did you mean: {', '.join(suggestions)}?"
            if available:
                error_msg += f"\n  Available profiles: {', '.join(sorted(available))}"
            else:
                error_msg += "\n  No profiles configured. Run 'victor init' or create ~/.victor/profiles.yaml"
            raise ValueError(error_msg)

        # Get provider-level settings
        provider_settings = settings.get_provider_settings(profile.provider)

        # Merge profile-level overrides (base_url, timeout, api_key, etc.)
        # ProfileConfig uses extra="allow" so extra fields are in __pydantic_extra__
        if hasattr(profile, "__pydantic_extra__") and profile.__pydantic_extra__:
            # Profile-level settings override provider-level settings
            provider_settings.update(profile.__pydantic_extra__)
            logger.debug(
                f"Profile '{profile_name}' overrides: {list(profile.__pydantic_extra__.keys())}"
            )

        # Apply timeout multiplier from model capabilities
        # Slow local models (Ollama, LMStudio) get longer timeouts
        from victor.agent.tool_calling.capabilities import ModelCapabilityLoader

        cap_loader = ModelCapabilityLoader()
        caps = cap_loader.get_capabilities(profile.provider, profile.model)
        if caps and caps.timeout_multiplier > 1.0:
            base_timeout = provider_settings.get("timeout", 300)
            adjusted_timeout = int(base_timeout * caps.timeout_multiplier)
            provider_settings["timeout"] = adjusted_timeout
            logger.info(
                f"Adjusted timeout for {profile.provider}/{profile.model}: "
                f"{base_timeout}s -> {adjusted_timeout}s (multiplier: {caps.timeout_multiplier}x)"
            )

        # Create provider instance using registry
        provider = ProviderRegistry.create(profile.provider, **provider_settings)

        orchestrator = cls(
            settings=settings,
            provider=provider,
            model=profile.model,
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            tool_selection=profile.tool_selection,
            thinking=thinking,
            provider_name=profile.provider,
            profile_name=profile_name,
        )

        # Setup JSONL exporter if enabled
        if getattr(settings, "enable_observability_logging", False):
            from victor.observability.bridge import ObservabilityBridge
            from victor.core.events import get_observability_bus

            try:
                bridge = ObservabilityBridge.get_instance()
                log_path = getattr(settings, "observability_log_path", None)
                bridge.setup_jsonl_exporter(log_path=log_path)
                logger.info(
                    f"JSONL event logging enabled: {log_path or '~/.victor/metrics/victor.jsonl'}"
                )
            except Exception as e:
                logger.warning(f"Failed to setup JSONL exporter: {e}")

        return orchestrator
