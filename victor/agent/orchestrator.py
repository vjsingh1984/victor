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
from victor.agent.rl.coordinator import get_rl_coordinator
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

        # Initialize CheckpointManager for time-travel debugging (via factory)
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
        """Callback when tool execution completes (from ToolPipeline)."""
        self._metrics_collector.on_tool_complete(result)

        # Emit tool complete event
        from victor.core.events import get_observability_bus

        bus = get_observability_bus()
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                bus.emit(
                    topic="tool.complete",
                    data={
                        "tool_name": result.tool_name,
                        "success": result.success,
                        "result_length": len(str(result.result or "")) if result.result else 0,
                        "error": str(result.error) if result.error else None,
                        "category": "tool",
                    },
                )
            )
        except RuntimeError:
            # No event loop running
            pass
        except Exception:
            # Ignore errors during event emission
            pass

        # Track read files for task completion detection
        if result.success and result.tool_name in ("read", "Read", "read_file"):
            # Extract file path from arguments
            if result.arguments:
                file_path = result.arguments.get("path") or result.arguments.get("file_path")
                if file_path:
                    self._read_files_session.add(file_path)
                    logger.debug(f"Tracked read file: {file_path}")

                    # Check if all required files have been read - nudge to produce output
                    if (
                        self._required_files
                        and self._read_files_session.issuperset(set(self._required_files))
                        and not getattr(self, "_all_files_read_nudge_sent", False)
                    ):
                        self._all_files_read_nudge_sent = True
                        logger.info(
                            f"All {len(self._required_files)} required files have been read. "
                            "Agent should now produce the required output."
                        )

                        # Emit nudge event
                        from victor.core.events import get_observability_bus

                        event_bus = get_observability_bus()
                        event_bus.emit(
                            topic="state.task.all_files_read_nudge",
                            data={
                                "required_files": list(self._required_files),
                                "read_files": list(self._read_files_session),
                                "required_outputs": self._required_outputs,
                                "action": "nudge_output_production",
                                "category": "state",
                            },
                        )

                        # Inject nudge message to encourage output production
                        if self._required_outputs:
                            outputs_str = ", ".join(self._required_outputs)
                            self.add_message(
                                "system",
                                f"[REMINDER] All required files have been read. "
                                f"Please now produce the required output: {outputs_str}. "
                                f"Avoid further exploration - focus on synthesizing findings.",
                            )

        # Emit observability event for tool completion
        if hasattr(self, "_observability") and self._observability:
            iteration = self._tool_pipeline.calls_used if hasattr(self, "_tool_pipeline") else 0
            tool_id = f"tool-{iteration}"
            self._observability.on_tool_end(
                tool_name=result.tool_name,
                result=result.result,
                success=result.success,
                tool_id=tool_id,
                error=result.error,
            )

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
        """Send reward signal to RL model selector for Q-value updates.

        Converts StreamingSession data into RLOutcome and updates Q-values
        based on session outcome (success, latency, throughput, tool usage).
        """
        try:
            from victor.agent.rl.coordinator import get_rl_coordinator
            from victor.agent.rl.base import RLOutcome

            if not self._rl_coordinator:
                return

            # Extract metrics from session
            token_count = 0
            if session.metrics:
                # Estimate tokens from chunks (streaming metrics)
                token_count = session.metrics.total_chunks or 0

            # Get tool execution count from metrics collector
            tool_calls_made = 0
            if hasattr(self, "_metrics_collector") and self._metrics_collector:
                tool_calls_made = self._metrics_collector._selection_stats.total_tools_executed

            # Determine success: no error and not cancelled
            success = session.error is None and not session.cancelled

            # Compute quality score (0-1) based on success and metrics
            quality_score = 0.5
            if success:
                quality_score = 0.8
                # Bonus for fast responses
                if session.duration < 10:
                    quality_score += 0.1
                # Bonus for tool usage
                if tool_calls_made > 0:
                    quality_score += 0.1
            quality_score = min(1.0, quality_score)

            # Create outcome
            outcome = RLOutcome(
                provider=session.provider,
                model=session.model,
                task_type=getattr(session, "task_type", "unknown"),
                success=success,
                quality_score=quality_score,
                metadata={
                    "latency_seconds": session.duration,
                    "token_count": token_count,
                    "tool_calls_made": tool_calls_made,
                    "session_id": session.session_id,
                },
                vertical=getattr(self._vertical_context, "vertical_name", None) or "default",
            )

            # Record outcome for model selector - use vertical from context
            vertical_name = getattr(self._vertical_context, "vertical_name", None) or "default"
            self._rl_coordinator.record_outcome("model_selector", outcome, vertical_name)

            logger.debug(
                f"RL feedback: provider={session.provider} success={success} "
                f"quality={quality_score:.2f} duration={session.duration:.1f}s"
            )

        except ImportError:
            # RL module not available - skip silently
            pass
        except (KeyError, AttributeError) as e:
            # RL coordinator not properly initialized
            logger.debug(f"RL reward signal skipped (not configured): {e}")
        except (ValueError, TypeError) as e:
            # Invalid reward data
            logger.warning(f"Failed to send RL reward signal (invalid data): {e}")

    def _extract_required_files_from_prompt(self, user_message: str) -> List[str]:
        """Extract file paths mentioned in user prompt for task completion tracking.

        Delegates to TaskAnalyzer.extract_required_files_from_prompt().

        Args:
            user_message: The user's prompt text

        Returns:
            List of file paths mentioned in the prompt
        """
        return self._task_analyzer.extract_required_files_from_prompt(user_message)

    def _extract_required_outputs_from_prompt(self, user_message: str) -> List[str]:
        """Extract output requirements from user prompt.

        Delegates to TaskAnalyzer.extract_required_outputs_from_prompt().

        Args:
            user_message: The user's prompt text

        Returns:
            List of required output types (e.g., ["findings table", "top-3 fixes"])
        """
        return self._task_analyzer.extract_required_outputs_from_prompt(user_message)

    # =====================================================================
    # Component accessors for external use
    # =====================================================================

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
            CheckpointManager instance or None if disabled
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

        Args:
            description: Human-readable description for the checkpoint
            tags: Optional tags for categorization

        Returns:
            Checkpoint ID if saved, None if checkpointing is disabled
        """
        if not self._checkpoint_manager:
            logger.debug("Checkpoint save skipped - manager not initialized")
            return None

        # Build conversation state for checkpointing
        state = self._get_checkpoint_state()

        try:
            checkpoint_id = await self._checkpoint_manager.save_checkpoint(
                session_id=self._memory_session_id or "default",
                state=state,
                description=description,
                tags=tags,
            )
            logger.info(f"Manual checkpoint saved: {checkpoint_id[:20]}...")
            return checkpoint_id
        except (OSError, IOError) as e:
            logger.warning(f"Failed to save checkpoint (I/O error): {e}")
            return None
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to save checkpoint (serialization error): {e}")
            return None

    async def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore conversation state from a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to restore

        Returns:
            True if restored successfully, False otherwise
        """
        if not self._checkpoint_manager:
            logger.warning("Cannot restore - checkpoint manager not initialized")
            return False

        try:
            state = await self._checkpoint_manager.restore_checkpoint(checkpoint_id)
            self._apply_checkpoint_state(state)
            logger.info(f"Restored checkpoint: {checkpoint_id[:20]}...")
            return True
        except (OSError, IOError) as e:
            logger.warning(f"Failed to restore checkpoint (I/O error): {e}")
            return False
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to restore checkpoint (invalid data): {e}")
            return False

    async def maybe_auto_checkpoint(self) -> Optional[str]:
        """Trigger auto-checkpoint if interval threshold is met.

        This should be called after tool executions to maintain
        automatic checkpointing based on configured interval.

        Returns:
            Checkpoint ID if auto-checkpoint was created, None otherwise
        """
        if not self._checkpoint_manager:
            return None

        state = self._get_checkpoint_state()

        try:
            return await self._checkpoint_manager.maybe_auto_checkpoint(
                session_id=self._memory_session_id or "default",
                state=state,
            )
        except (OSError, IOError) as e:
            logger.debug(f"Auto-checkpoint failed (I/O error): {e}")
            return None
        except (ValueError, TypeError) as e:
            logger.debug(f"Auto-checkpoint failed (serialization error): {e}")
            return None

    def _get_checkpoint_state(self) -> dict:
        """Build a dictionary representing current conversation state for checkpointing."""
        return {
            "stage": (
                self.conversation_state.get_current_stage().name
                if self.conversation_state.get_current_stage()
                else "INITIAL"
            ),
            "tool_history": list(self.executed_tools),
            "observed_files": list(self.observed_files),
            "modified_files": list(
                getattr(self._conversation_controller, "_modified_files", set())
            ),
            "message_count": len(self.conversation.messages) if self.conversation else 0,
            "tool_calls_used": self.tool_calls_used,
            "tool_budget": self.tool_budget,
        }

    def _apply_checkpoint_state(self, state: dict) -> None:
        """Apply a checkpoint state to restore the orchestrator.

        Args:
            state: State dictionary from checkpoint
        """
        # Restore execution tracking
        self.executed_tools = set(state.get("tool_history", []))
        self.observed_files = set(state.get("observed_files", []))
        self.tool_calls_used = state.get("tool_calls_used", 0)

        # Restore stage if present
        stage_name = state.get("stage", "INITIAL")
        try:
            stage = ConversationStage[stage_name]
            self.conversation_state.set_stage(stage)
        except (KeyError, AttributeError):
            logger.debug(f"Could not restore stage: {stage_name}")

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
        """Initialize LanceDB embedding store for semantic conversation retrieval.

        Uses the module-level singleton to prevent duplicate initialization.
        The singleton pattern ensures that intelligent_prompt_builder and other
        components share the same instance.

        The ConversationEmbeddingStore provides:
        - Pre-computed message embeddings in LanceDB
        - O(log n) vector search instead of O(n) on-the-fly embedding
        - Automatic sync when messages are added to ConversationStore
        """
        if self.memory_manager is None:
            return

        try:
            from victor.storage.embeddings.service import EmbeddingService
            import victor.agent.conversation_embedding_store as ces_module

            # Get the shared embedding service
            embedding_service = EmbeddingService.get_instance()

            # Use singleton pattern - check if already exists
            if ces_module._embedding_store is not None:
                self._conversation_embedding_store = ces_module._embedding_store
                logger.debug(
                    "[AgentOrchestrator] Reusing existing ConversationEmbeddingStore singleton"
                )
            else:
                # Create new instance and register as singleton
                self._conversation_embedding_store = ConversationEmbeddingStore(
                    embedding_service=embedding_service,
                )
                ces_module._embedding_store = self._conversation_embedding_store
                logger.debug("[AgentOrchestrator] Created ConversationEmbeddingStore singleton")

            # Wire it to the memory manager for automatic sync
            self.memory_manager.set_embedding_store(self._conversation_embedding_store)

            # Also set the embedding service for fallback
            self.memory_manager.set_embedding_service(embedding_service)

            # Initialize async (fire and forget for faster startup).
            # If there's no running event loop (e.g., unit tests), fall back
            # to synchronous initialization to avoid 'coroutine was never awaited' warnings.
            if not self._conversation_embedding_store.is_initialized:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._conversation_embedding_store.initialize())
                except RuntimeError:
                    try:
                        asyncio.run(self._conversation_embedding_store.initialize())
                    except Exception as e:
                        logger.debug(
                            "Failed to run ConversationEmbeddingStore.initialize() synchronously: %s",
                            e,
                        )

            logger.info(
                "[AgentOrchestrator] ConversationEmbeddingStore configured. "
                "Message embeddings will sync to LanceDB."
            )

            # Set up semantic tool result cache using the embedding service
            # This enables FAISS-based semantic caching with mtime invalidation
            try:
                from victor.agent.tool_result_cache import ToolResultCache

                semantic_cache = ToolResultCache(
                    embedding_service=embedding_service,
                    max_entries=500,
                    cleanup_interval=60.0,
                )
                # Store semantic cache for later wiring to tool pipeline
                # (tool pipeline is created after this method runs)
                self._pending_semantic_cache = semantic_cache
                logger.debug(
                    "[AgentOrchestrator] Semantic tool cache created, pending wire to pipeline"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize semantic tool cache: {e}")

        except Exception as e:
            logger.warning(f"Failed to initialize ConversationEmbeddingStore: {e}")
            self._conversation_embedding_store = None

    def _finalize_stream_metrics(
        self, usage_data: Optional[Dict[str, int]] = None
    ) -> Optional[StreamMetrics]:
        """Finalize stream metrics at end of streaming session.

        Args:
            usage_data: Optional cumulative token usage from provider API.
                       When provided, enables accurate token counts.
        """
        metrics = self._metrics_collector.finalize_stream_metrics(usage_data)

        # Record to session cost tracker for cumulative tracking
        if metrics and hasattr(self, "_session_cost_tracker"):
            self._session_cost_tracker.record_request(
                prompt_tokens=metrics.prompt_tokens,
                completion_tokens=metrics.completion_tokens,
                cache_read_tokens=metrics.cache_read_tokens,
                cache_write_tokens=metrics.cache_write_tokens,
                duration_seconds=metrics.total_duration,
                tool_calls=metrics.tool_calls_count,
            )

        return metrics

    def get_last_stream_metrics(self) -> Optional[StreamMetrics]:
        """Get metrics from the last streaming session.

        Returns:
            StreamMetrics from the last session or None if no metrics available
        """
        return self._metrics_collector.get_last_stream_metrics()

    def get_streaming_metrics_summary(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive streaming metrics summary.

        Returns:
            Dictionary with aggregated metrics or None if metrics disabled.
        """
        return self._metrics_collector.get_streaming_metrics_summary()

    def get_streaming_metrics_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent streaming metrics history.

        Args:
            limit: Maximum number of recent metrics to return

        Returns:
            List of recent metrics dictionaries
        """
        return self._metrics_collector.get_streaming_metrics_history(limit)

    def get_session_cost_summary(self) -> Dict[str, Any]:
        """Get session cost summary.

        Returns:
            Dictionary with session cost statistics
        """
        if hasattr(self, "_session_cost_tracker"):
            return self._session_cost_tracker.get_summary()
        return {}

    def get_session_cost_formatted(self) -> str:
        """Get formatted session cost string.

        Returns:
            Cost string like "$0.0123" or "cost n/a"
        """
        if hasattr(self, "_session_cost_tracker"):
            return self._session_cost_tracker.format_inline_cost()
        return "cost n/a"

    def export_session_costs(self, path: str, format: str = "json") -> None:
        """Export session costs to file.

        Args:
            path: Output file path
            format: Export format ("json" or "csv")
        """
        from pathlib import Path

        if not hasattr(self, "_session_cost_tracker"):
            return

        output_path = Path(path)
        if format == "csv":
            self._session_cost_tracker.export_csv(output_path)
        else:
            self._session_cost_tracker.export_json(output_path)

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
            logger.info("✓ Tool embeddings preloaded successfully in background")
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
        """Record tool execution statistics.

        Args:
            tool_name: Name of the tool executed
            success: Whether execution succeeded
            elapsed_ms: Execution time in milliseconds
            error_type: Type of error if execution failed
        """
        self._metrics_collector.record_tool_execution(tool_name, success, elapsed_ms)

        # Also record to UsageAnalytics for data-driven optimization
        if hasattr(self, "_usage_analytics") and self._usage_analytics:
            context_metrics = self._conversation_controller.get_context_metrics()
            self._usage_analytics.record_tool_execution(
                tool_name=tool_name,
                success=success,
                execution_time_ms=elapsed_ms,
                error_type=error_type,
                context_tokens=context_metrics.estimated_tokens,
            )

        # Record to ToolSequenceTracker for intelligent next-tool suggestions
        if hasattr(self, "_sequence_tracker") and self._sequence_tracker:
            self._sequence_tracker.record_execution(
                tool_name=tool_name,
                success=success,
                execution_time=elapsed_ms / 1000.0,  # Convert to seconds
            )

        # Also record to SemanticToolSelector's internal tracker for confidence boosting
        # This enables the 15-20% accuracy improvement via workflow pattern detection
        if hasattr(self, "tool_selector") and hasattr(self.tool_selector, "record_tool_execution"):
            self.tool_selector.record_tool_execution(tool_name, success=success)

        # Record to RL tool_selector learner for Q-learning optimization
        if self._rl_coordinator:
            try:
                from victor.agent.rl.base import RLOutcome

                # Get current context
                provider_name = getattr(self.current_provider, "name", "unknown")
                model_name = getattr(self, "_current_model", "unknown")
                task_type = getattr(self, "_task_type", "general")
                vertical_name = getattr(self, "_vertical_name", None)

                tool_outcome = RLOutcome(
                    success=success,
                    quality_score=1.0 if success else 0.0,
                    provider=provider_name,
                    model=model_name,
                    task_type=task_type,
                    metadata={
                        "tool_name": tool_name,
                        "execution_time_ms": elapsed_ms,
                        "error_type": error_type,
                    },
                )
                self._rl_coordinator.record_outcome("tool_selector", tool_outcome, vertical_name)
            except ImportError:
                logger.debug("RLOutcome not available, skipping RL recording")
            except KeyError as e:
                logger.debug(f"RL learner not registered: {e}")
            except ValueError as e:
                logger.warning(f"Invalid outcome data for RL recording: {e}")

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

        Returns cumulative tokens used across all stream_chat calls.
        Used by VictorAgentAdapter for benchmark token tracking.

        Returns:
            TokenUsage dataclass with input/output/total token counts
        """
        from victor.evaluation.protocol import TokenUsage

        return TokenUsage(
            input_tokens=self._cumulative_token_usage.get("prompt_tokens", 0),
            output_tokens=self._cumulative_token_usage.get("completion_tokens", 0),
            total_tokens=self._cumulative_token_usage.get("total_tokens", 0),
        )

    def reset_token_usage(self) -> None:
        """Reset cumulative token usage tracking.

        Call this at the start of a new evaluation task to get fresh counts.
        """
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

        Provides visibility into the health and statistics of all optimization
        components for debugging, monitoring, and observability.

        Returns:
            Dictionary with component status and statistics:
            - context_compactor: Compaction stats, utilization, threshold
            - usage_analytics: Tool/provider metrics, session info
            - sequence_tracker: Pattern learning stats, suggestions
            - code_correction: Enabled status, correction stats
            - safety_checker: Enabled status, pattern counts
            - auto_committer: Enabled status, commit history
            - search_router: Routing stats, pattern matches
        """
        status: Dict[str, Any] = {
            "timestamp": time.time(),
            "components": {},
        }

        # Context Compactor
        if self._context_compactor:
            status["components"]["context_compactor"] = self._context_compactor.get_statistics()

        # Usage Analytics
        if self._usage_analytics:
            try:
                status["components"]["usage_analytics"] = {
                    "session_active": self._usage_analytics._current_session is not None,
                    "tool_records_count": len(self._usage_analytics._tool_records),
                    "provider_records_count": len(self._usage_analytics._provider_records),
                }
            except Exception:
                status["components"]["usage_analytics"] = {"status": "error"}

        # Sequence Tracker
        if self._sequence_tracker:
            try:
                status["components"]["sequence_tracker"] = self._sequence_tracker.get_statistics()
            except Exception:
                status["components"]["sequence_tracker"] = {"status": "error"}

        # Code Correction Middleware
        status["components"]["code_correction"] = {
            "enabled": self._code_correction_middleware is not None,
        }
        if self._code_correction_middleware:
            # Support both old-style (with config) and new vertical middleware (without config)
            if hasattr(self._code_correction_middleware, "config"):
                status["components"]["code_correction"]["config"] = {
                    "auto_fix": self._code_correction_middleware.config.auto_fix,
                    "max_iterations": self._code_correction_middleware.config.max_iterations,
                }
            else:
                # New vertical middleware - use get_config() if available or default values
                status["components"]["code_correction"]["config"] = {
                    "auto_fix": getattr(self._code_correction_middleware, "auto_fix", True),
                    "max_iterations": getattr(
                        self._code_correction_middleware, "max_iterations", 1
                    ),
                }

        # Safety Checker
        status["components"]["safety_checker"] = {
            "enabled": self._safety_checker is not None,
            "has_confirmation_callback": (
                self._safety_checker.confirmation_callback is not None
                if self._safety_checker
                else False
            ),
        }

        # Auto Committer
        status["components"]["auto_committer"] = {
            "enabled": self._auto_committer is not None,
        }
        if self._auto_committer:
            status["components"]["auto_committer"]["auto_commit"] = self._auto_committer.auto_commit

        # Search Router
        status["components"]["search_router"] = {
            "enabled": self.search_router is not None,
        }

        # Overall health
        enabled_count = sum(
            1
            for c in status["components"].values()
            if c.get("enabled", True) and c.get("status") != "error"
        )
        status["health"] = {
            "enabled_components": enabled_count,
            "total_components": len(status["components"]),
            "status": "healthy" if enabled_count >= 4 else "degraded",
        }

        return status

    def flush_analytics(self) -> Dict[str, bool]:
        """Flush all analytics and cached data to persistent storage.

        Call this method before shutdown or when you need to ensure
        all analytics data is persisted to disk. Useful for graceful
        shutdown scenarios.

        Returns:
            Dictionary indicating success/failure for each component:
            - usage_analytics: Whether analytics were flushed
            - sequence_tracker: Whether patterns were saved
            - tool_cache: Whether cache was flushed
        """
        results: Dict[str, bool] = {}

        # Flush usage analytics
        if hasattr(self, "_usage_analytics") and self._usage_analytics:
            try:
                self._usage_analytics.flush()
                results["usage_analytics"] = True
                logger.debug("UsageAnalytics flushed to disk")
            except Exception as e:
                logger.warning(f"Failed to flush usage analytics: {e}")
                results["usage_analytics"] = False
        else:
            results["usage_analytics"] = False

        # Flush sequence tracker patterns
        if hasattr(self, "_sequence_tracker") and self._sequence_tracker:
            try:
                # SequenceTracker learns patterns in memory; no explicit flush needed
                # but we capture statistics for reporting
                stats = self._sequence_tracker.get_statistics()
                results["sequence_tracker"] = True
                logger.debug(
                    f"SequenceTracker has {stats.get('unique_transitions', 0)} learned patterns"
                )
            except Exception as e:
                logger.warning(f"Failed to get sequence tracker stats: {e}")
                results["sequence_tracker"] = False
        else:
            results["sequence_tracker"] = False

        # Flush tool cache
        if hasattr(self, "tool_cache") and self.tool_cache:
            try:
                # Tool cache is already persistent; just report status
                results["tool_cache"] = True
            except Exception as e:
                logger.warning(f"Failed to access tool cache: {e}")
                results["tool_cache"] = False
        else:
            results["tool_cache"] = False

        logger.info(f"Analytics flush complete: {results}")
        return results

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

        This method reinitializes the provider, tool calling adapter, and
        prompt builder while preserving the conversation history.

        Delegates core switching to ProviderManager while handling
        orchestrator-specific post-switch hooks.

        Args:
            provider_name: Name of the provider (ollama, lmstudio, anthropic, etc.)
            model: Optional model name. If not provided, uses current model.
            **provider_kwargs: Additional provider-specific arguments (base_url, api_key, etc.)

        Returns:
            True if switch was successful, False otherwise

        Example:
            # Switch from ollama to lmstudio
            orchestrator.switch_provider("lmstudio", "qwen2.5-coder:14b", base_url="http://localhost:1234")

            # Switch provider with profile config
            orchestrator.switch_provider("anthropic", "claude-sonnet-4-20250514")
        """
        try:
            # Get provider settings from settings if not provided
            if not provider_kwargs:
                provider_kwargs = self.settings.get_provider_settings(provider_name)

            # Create new provider instance
            new_provider = ProviderRegistry.create(provider_name, **provider_kwargs)

            # Determine model to use
            new_model = model or self.model

            # Store old state for analytics
            old_provider_name = self.provider_name
            old_model = self.model

            # Update ProviderManager internal state directly
            # (Using sync update instead of async switch_provider for backward compatibility)
            # Update both _current_state (legacy) and ProviderSwitcher state
            self._provider_manager._current_state = ProviderState(
                provider=new_provider,
                provider_name=provider_name.lower(),
                model=new_model,
            )

            # Also update ProviderSwitcher's state (the source of truth)
            from victor.agent.provider.switcher import ProviderSwitcherState

            switcher_state = self._provider_manager._provider_switcher.get_current_state()
            old_switch_count = switcher_state.switch_count if switcher_state else 0

            self._provider_manager._provider_switcher._current_state = ProviderSwitcherState(
                provider=new_provider,
                provider_name=provider_name.lower(),
                model=new_model,
                switch_count=old_switch_count + 1,
            )

            self._provider_manager.initialize_tool_adapter()

            # Sync local attributes from ProviderManager
            self.provider = self._provider_manager.provider
            self.model = self._provider_manager.model
            self.provider_name = self._provider_manager.provider_name
            self.tool_adapter = self._provider_manager.tool_adapter
            self.tool_calling_caps = self._provider_manager.capabilities

            # Apply post-switch hooks (exploration settings, prompt builder, system prompt, tool budget)
            self._apply_post_switch_hooks(respect_sticky_budget=True)

            # Log the switch
            logger.info(
                f"Switched provider: {old_provider_name}:{old_model} -> "
                f"{self.provider_name}:{new_model} "
                f"(native_tools={self.tool_calling_caps.native_tool_calls})"
            )

            # Log analytics event
            self.usage_logger.log_event(
                "provider_switch",
                {
                    "old_provider": old_provider_name,
                    "old_model": old_model,
                    "new_provider": self.provider_name,
                    "new_model": new_model,
                    "native_tool_calls": self.tool_calling_caps.native_tool_calls,
                },
            )

            # Update metrics collector with new model info
            self._metrics_collector.update_model_info(new_model, self.provider_name)

            return True

        except Exception as e:
            logger.error(f"Failed to switch provider to {provider_name}: {e}")
            return False

    def switch_model(self, model: str) -> bool:
        """Switch to a different model on the current provider.

        This is a lighter-weight switch than switch_provider() - it only
        updates the model and reinitializes the tool adapter.

        Delegates core switching to ProviderManager while handling
        orchestrator-specific post-switch hooks.

        Args:
            model: New model name

        Returns:
            True if switch was successful, False otherwise

        Example:
            orchestrator.switch_model("qwen2.5-coder:32b")
        """
        try:
            old_model = self.model

            # Update ProviderManager's model and reinitialize adapter
            if self._provider_manager._current_state:
                # Update both _current_state (legacy) and ProviderSwitcher state
                self._provider_manager._current_state.model = model

                # Also update ProviderSwitcher's state (the source of truth)
                switcher_state = self._provider_manager._provider_switcher.get_current_state()
                if switcher_state:
                    switcher_state.model = model

                self._provider_manager.initialize_tool_adapter()

            # Sync local attributes from ProviderManager
            self.model = self._provider_manager.model
            self.tool_adapter = self._provider_manager.tool_adapter
            self.tool_calling_caps = self._provider_manager.capabilities

            # Apply post-switch hooks (exploration settings, prompt builder, system prompt, tool budget)
            self._apply_post_switch_hooks()

            logger.info(
                f"Switched model: {old_model} -> {model} "
                f"(native_tools={self.tool_calling_caps.native_tool_calls})"
            )

            self.usage_logger.log_event(
                "model_switch",
                {
                    "provider": self.provider_name,
                    "old_model": old_model,
                    "new_model": model,
                    "native_tool_calls": self.tool_calling_caps.native_tool_calls,
                },
            )

            # Update metrics collector with new model info
            self._metrics_collector.update_model_info(model, self.provider_name)

            return True

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

    def _apply_post_switch_hooks(self, respect_sticky_budget: bool = False) -> None:
        """Apply post-switch hooks after provider/model switch.

        Consolidates common post-switch logic used by both switch_provider()
        and switch_model() methods:
        - Apply model-specific exploration settings to unified tracker
        - Reinitialize prompt builder with new capabilities
        - Rebuild system prompt
        - Update tool budget

        Args:
            respect_sticky_budget: If True, don't reset tool budget when user
                override is sticky (used by switch_provider).

        Note:
            This method assumes self.model, self.provider_name, self.tool_adapter,
            and self.tool_calling_caps are already updated before calling.
        """
        # Apply model-specific exploration settings to unified tracker
        self.unified_tracker.set_model_exploration_settings(
            exploration_multiplier=self.tool_calling_caps.exploration_multiplier,
            continuation_patience=self.tool_calling_caps.continuation_patience,
        )

        # Get prompt contributors from vertical extensions
        prompt_contributors = []
        try:
            from victor.core.verticals.protocols import VerticalExtensions

            extensions = self._container.get_optional(VerticalExtensions)
            if extensions and extensions.prompt_contributors:
                prompt_contributors = extensions.prompt_contributors
        except ImportError:
            logger.debug("VerticalExtensions module not available")
        except AttributeError as e:
            logger.warning(f"VerticalExtensions missing expected attributes: {e}")

        # Reinitialize prompt builder
        self.prompt_builder = SystemPromptBuilder(
            provider_name=self.provider_name,
            model=self.model,
            tool_adapter=self.tool_adapter,
            capabilities=self.tool_calling_caps,
            prompt_contributors=prompt_contributors,
        )

        # Rebuild system prompt with new adapter hints
        base_system_prompt = self._build_system_prompt_with_adapter()
        if self.project_context.content:
            self._system_prompt = (
                base_system_prompt + "\n\n" + self.project_context.get_system_prompt_addition()
            )
        else:
            self._system_prompt = base_system_prompt

        # Update tool budget based on new adapter's recommendation
        if respect_sticky_budget:
            sticky_budget = getattr(self.unified_tracker, "_sticky_user_budget", False)
            if sticky_budget:
                logger.debug("Skipping tool budget reset on provider switch (sticky user override)")
                return

        default_budget = max(self.tool_calling_caps.recommended_tool_budget, 50)
        self.tool_budget = getattr(self.settings, "tool_call_budget", default_budget)

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
            from victor.agent.rl.hooks import get_rl_hooks, RLEvent, RLEventType

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

    def _determine_continuation_action(
        self,
        intent_result: Any,  # IntentClassificationResult
        is_analysis_task: bool,
        is_action_task: bool,
        content_length: int,
        full_content: Optional[str],
        continuation_prompts: int,
        asking_input_prompts: int,
        one_shot_mode: bool,
        mentioned_tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Determine what continuation action to take when model doesn't call tools.

        DEPRECATED: This method delegates to ContinuationStrategy.determine_continuation_action().
        Kept for backward compatibility with existing tests.

        Args:
            intent_result: Result from intent classifier (has .intent, .confidence)
            is_analysis_task: Whether task is analysis-oriented
            is_action_task: Whether task is action-oriented
            content_length: Length of model's response content
            full_content: Full response content (for structure detection)
            continuation_prompts: Current count of continuation prompts sent
            asking_input_prompts: Current count of asking-input auto-responses
            one_shot_mode: Whether running in non-interactive mode
            mentioned_tools: Tools mentioned but not executed (hallucinated tool calls)

        Returns:
            Dictionary with action, message, reason, and updates.
        """
        # Delegate to ContinuationStrategy (extracted in Phase 2E)
        strategy = ContinuationStrategy()
        return strategy.determine_continuation_action(
            intent_result=intent_result,
            is_analysis_task=is_analysis_task,
            is_action_task=is_action_task,
            content_length=content_length,
            full_content=full_content,
            continuation_prompts=continuation_prompts,
            asking_input_prompts=asking_input_prompts,
            one_shot_mode=one_shot_mode,
            mentioned_tools=mentioned_tools,
            # Context from orchestrator
            max_prompts_summary_requested=getattr(self, "_max_prompts_summary_requested", False),
            settings=self.settings,
            rl_coordinator=self._rl_coordinator,
            provider_name=self.provider.name,
            model=self.model,
            tool_budget=self.tool_budget,
            unified_tracker_config=self.unified_tracker.config,
            task_completion_signals=None,  # Legacy caller doesn't use this
            # Task Completion Detection Enhancement (Phase 2 - Feature Flag Protected)
            task_completion_detector=self._task_completion_detector,
        )

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
                f"[yellow]⚠ Model '{self.model}' is not marked as tool-call-capable for provider '{provider_key}'. "
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

        This method implements a proper agentic loop that:
        1. Gets model response
        2. Executes any tool calls
        3. Continues until model provides a final response (no tool calls)
        4. Ensures non-empty response on tool failures

        Args:
            user_message: User's message

        Returns:
            CompletionResponse from the model with complete response
        """
        # Ensure system prompt is included once at start of conversation
        self.conversation.ensure_system_prompt()
        self._system_added = True
        # Add user message to history
        self.add_message("user", user_message)

        # Initialize tracking for this conversation turn
        self.tool_calls_used = 0
        failure_context = ToolFailureContext()
        max_iterations = getattr(self.settings, "chat_max_iterations", 10)
        iteration = 0

        # Classify task complexity for appropriate budgeting
        task_classification = self.task_classifier.classify(user_message)
        iteration_budget = min(
            task_classification.tool_budget * 2, max_iterations  # Allow 2x budget for iterations
        )

        # Agentic loop: continue until no tool calls or budget exhausted
        final_response: Optional[CompletionResponse] = None

        while iteration < iteration_budget:
            iteration += 1

            # Get tool definitions if provider supports them
            tools = None
            if self.provider.supports_tools() and self.tool_calls_used < self.tool_budget:
                conversation_depth = self.conversation.message_count()
                conversation_history = (
                    [msg.model_dump() for msg in self.messages] if self.messages else None
                )
                tools = await self.tool_selector.select_tools(
                    user_message,
                    use_semantic=self.use_semantic_selection,
                    conversation_history=conversation_history,
                    conversation_depth=conversation_depth,
                )
                tools = self.tool_selector.prioritize_by_stage(user_message, tools)

            # Prepare optional thinking parameter
            provider_kwargs = {}
            if self.thinking:
                provider_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}

            # Check context and compact before API call to prevent overflow
            if self._context_compactor:
                compaction_action = self._context_compactor.check_and_compact(
                    current_query=user_message,
                    force=False,
                    tool_call_count=self.tool_calls_used,
                    task_complexity=task_classification.complexity.value,
                )
                if compaction_action.action_taken:
                    logger.info(
                        f"Compacted context before API call: {compaction_action.messages_removed} messages removed, "
                        f"{compaction_action.tokens_freed} tokens freed"
                    )

            # Get response from provider
            response = await self.provider.chat(
                messages=self.messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=tools,
                **provider_kwargs,
            )

            # Accumulate token usage for evaluation tracking (P1: Token Tracking Fix)
            if response.usage:
                self._cumulative_token_usage["prompt_tokens"] += response.usage.get(
                    "prompt_tokens", 0
                )
                self._cumulative_token_usage["completion_tokens"] += response.usage.get(
                    "completion_tokens", 0
                )
                self._cumulative_token_usage["total_tokens"] += response.usage.get(
                    "total_tokens", 0
                )

            # Add assistant response to history if has content
            if response.content:
                self.add_message("assistant", response.content)

                # Check compaction after adding assistant response
                if self._context_compactor:
                    compaction_action = self._context_compactor.check_and_compact(
                        current_query=user_message,
                        force=False,
                        tool_call_count=self.tool_calls_used,
                        task_complexity=task_classification.complexity.value,
                    )
                    if compaction_action.action_taken:
                        logger.info(
                            f"Compacted context after response: {compaction_action.messages_removed} messages removed, "
                            f"{compaction_action.tokens_freed} tokens freed"
                        )

            # Check if model wants to use tools
            if response.tool_calls:
                # Handle tool calls and track results
                tool_results = await self._handle_tool_calls(response.tool_calls)

                # Update failure context
                for result in tool_results:
                    if result.get("success"):
                        failure_context.successful_tools.append(result)
                    else:
                        failure_context.failed_tools.append(result)
                        failure_context.last_error = result.get("error")

                # Continue loop to get follow-up response
                continue

            # No tool calls - this is the final response
            final_response = response
            break

        # Ensure we have a complete response
        if final_response is None or not final_response.content:
            # Use response completer to generate a response
            completion_result = await self.response_completer.ensure_response(
                messages=self.messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                failure_context=failure_context if failure_context.failed_tools else None,
            )

            if completion_result.content:
                self.add_message("assistant", completion_result.content)
                # Create a synthetic response
                final_response = CompletionResponse(
                    content=completion_result.content,
                    role="assistant",
                    tool_calls=None,
                )
            else:
                # Last resort fallback
                fallback_content = (
                    "I was unable to generate a complete response. "
                    "Please try rephrasing your request."
                )
                if failure_context.failed_tools:
                    fallback_content = self.response_completer.format_tool_failure_message(
                        failure_context
                    )
                # Add fallback to history and return synthetic response
                self.add_message("assistant", fallback_content)
                final_response = CompletionResponse(
                    content=fallback_content,
                    role="assistant",
                    tool_calls=None,
                )

        return final_response

    def _handle_cancellation(self, last_quality_score: float) -> Optional[StreamChunk]:
        """Handle user cancellation if requested."""
        if not self._check_cancellation():
            return None

        logger.info("Stream cancelled by user request")
        self._is_streaming = False
        # Record outcome for Q-learning (cancelled = incomplete)
        self._record_intelligent_outcome(
            success=False,
            quality_score=last_quality_score,
            user_satisfied=False,
            completed=False,
        )
        return StreamChunk(
            content="\n\n[Cancelled by user]\n",
            is_final=True,
        )

    async def _handle_compaction_async(self, user_message: str) -> Optional[StreamChunk]:
        """Perform proactive compaction asynchronously if enabled.

        Non-blocking version for async hot paths.
        Delegates to ContextManager (TD-002 refactoring).
        """
        return await self._context_manager.handle_compaction_async(user_message)

    async def _handle_context_and_iteration_limits(
        self,
        user_message: str,
        max_total_iterations: int,
        max_context: int,
        total_iterations: int,
        last_quality_score: float,
    ) -> tuple[bool, Optional[StreamChunk]]:
        """Handle context overflow and hard iteration limits.

        Returns:
            handled (bool): True if the caller should stop processing
            chunk (Optional[StreamChunk]): Chunk to yield if produced
        """
        # Context overflow handling
        if self._check_context_overflow(max_context):
            logger.warning("Context overflow detected. Attempting smart compaction...")
            removed = self._conversation_controller.smart_compact_history(
                current_query=user_message
            )
            if removed > 0:
                logger.info(f"Smart compaction removed {removed} messages")
                chunk = StreamChunk(
                    content=f"\n[context] Compacted history ({removed} messages) to continue.\n"
                )
                self._conversation_controller.inject_compaction_context()
                return False, chunk

            # If still overflowing, force completion
            if self._check_context_overflow(max_context):
                logger.warning("Still overflowing after compaction. Forcing completion.")
                chunk = StreamChunk(
                    content="\n[tool] ⚠ Context size limit reached. Providing summary.\n"
                )
                completion_prompt = self._get_thinking_disabled_prompt(
                    "Context limit reached. Summarize in 2-3 sentences."
                )
                recent_messages = self.messages[-8:] if len(self.messages) > 8 else self.messages[:]
                completion_messages = recent_messages + [
                    Message(role="user", content=completion_prompt)
                ]

                try:
                    response = await self.provider.chat(
                        messages=completion_messages,
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=min(self.max_tokens, 1024),
                        tools=None,
                    )
                    if response and response.content:
                        sanitized = self.sanitizer.sanitize(response.content)
                        if sanitized:
                            self.add_message("assistant", sanitized)
                            chunk = StreamChunk(content=sanitized, is_final=True)
                            self._record_intelligent_outcome(
                                success=True,
                                quality_score=last_quality_score,
                                user_satisfied=True,
                                completed=True,
                            )
                            return True, chunk
                except Exception as e:
                    logger.warning(f"Final response after context overflow failed: {e}")
                self._record_intelligent_outcome(
                    success=True,
                    quality_score=last_quality_score,
                    user_satisfied=True,
                    completed=True,
                )
                return True, StreamChunk(content="", is_final=True)

        # Iteration limit handling
        if total_iterations > max_total_iterations:
            logger.warning(
                f"Hard iteration limit reached ({max_total_iterations}). Forcing completion."
            )
            iteration_prompt = self._get_thinking_disabled_prompt(
                "Max iterations reached. Summarize key findings in 3-4 sentences. "
                "Do NOT attempt any more tool calls."
            )
            recent_messages = self.messages[-10:] if len(self.messages) > 10 else self.messages[:]
            completion_messages = recent_messages + [Message(role="user", content=iteration_prompt)]

            chunk = StreamChunk(
                content=f"\n[tool] ⚠ Maximum iterations ({max_total_iterations}) reached. Providing summary.\n"
            )

            try:
                response = await self.provider.chat(
                    messages=completion_messages,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=min(self.max_tokens, 1024),
                    tools=None,
                )
                if response and response.content:
                    sanitized = self.sanitizer.sanitize(response.content)
                    if sanitized:
                        self.add_message("assistant", sanitized)
                        chunk = StreamChunk(content=sanitized)
            except (ProviderRateLimitError, ProviderTimeoutError) as e:
                logger.error(f"Rate limit/timeout during final response: {e}")
                chunk = StreamChunk(content="Rate limited or timeout. Please retry in a moment.\n")
            except ProviderAuthError as e:
                logger.error(f"Auth error during final response: {e}")
                chunk = StreamChunk(content="Authentication error. Check API credentials.\n")
            except (ConnectionError, TimeoutError) as e:
                logger.error(f"Network error during final response: {e}")
                chunk = StreamChunk(content="Network error. Check connection.\n")
            except Exception:
                logger.exception("Unexpected error during final response generation")
                chunk = StreamChunk(
                    content="Unable to generate final summary due to iteration limit.\n"
                )

            self._record_intelligent_outcome(
                success=True,
                quality_score=last_quality_score,
                user_satisfied=True,
                completed=True,
            )
            return True, StreamChunk(content="", is_final=True)

        return False, None

    async def _prepare_stream(self, user_message: str) -> tuple[
        Any,
        float,
        float,
        Dict[str, int],
        int,
        int,
        int,
        bool,
        TrackerTaskType,
        Any,
        int,
    ]:
        """Prepare streaming state and return commonly used values."""
        # Initialize cancellation support
        self._cancel_event = asyncio.Event()
        self._is_streaming = True

        # Track performance metrics using StreamMetrics
        stream_metrics = self._metrics_collector.init_stream_metrics()
        start_time = stream_metrics.start_time
        total_tokens: float = 0

        # Cumulative token usage from provider (more accurate than estimates)
        cumulative_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

        # Ensure system prompt is included once at start of conversation
        self.conversation.ensure_system_prompt()
        self._system_added = True
        # Reset session state for new stream via SessionStateManager
        self._session_state.reset_for_new_turn()

        # Reset unified tracker for new conversation (single source of truth)
        self.unified_tracker.reset()

        # Reset context reminder manager for new conversation turn
        self.reminder_manager.reset()

        # Start UsageAnalytics session for this conversation
        if hasattr(self, "_usage_analytics") and self._usage_analytics:
            self._usage_analytics.start_session()

        # Clear ToolSequenceTracker history for new conversation (keep learned patterns)
        if hasattr(self, "_sequence_tracker") and self._sequence_tracker:
            self._sequence_tracker.clear_history()

        # PERF: Start background compaction for async context management
        # This runs compaction checks periodically without blocking the main loop
        if self._context_manager and hasattr(self._context_manager, "start_background_compaction"):
            await self._context_manager.start_background_compaction(interval_seconds=15.0)

        # Local aliases for frequently-used values
        max_total_iterations = self.unified_tracker.config.get("max_total_iterations", 50)
        total_iterations = 0
        force_completion = False

        # Add user message to history
        self.add_message("user", user_message)

        # Record this turn in UsageAnalytics
        if hasattr(self, "_usage_analytics") and self._usage_analytics:
            self._usage_analytics.record_turn()

        # Detect task type using unified tracker (single source of truth)
        unified_task_type = self.unified_tracker.detect_task_type(user_message)
        logger.info(f"Task type detected: {unified_task_type.value}")

        # Extract prompt requirements for dynamic budgets (e.g., "read 9 files", "top 3 fixes")
        prompt_requirements = extract_prompt_requirements(user_message)
        if prompt_requirements.has_explicit_requirements():
            # Mark tracker as having prompt requirements (enables lenient limits)
            self.unified_tracker._progress.has_prompt_requirements = True

            # Apply dynamic tool budget if larger than current
            if (
                prompt_requirements.tool_budget
                and prompt_requirements.tool_budget > self.unified_tracker._progress.tool_budget
            ):
                self.unified_tracker.set_tool_budget(prompt_requirements.tool_budget)
                logger.info(
                    f"Dynamic budget from prompt: {prompt_requirements.tool_budget} "
                    f"(files={prompt_requirements.file_count}, fixes={prompt_requirements.fix_count})"
                )

            # Apply dynamic iteration budget if larger than current
            if (
                prompt_requirements.iteration_budget
                and prompt_requirements.iteration_budget
                > self.unified_tracker._task_config.max_exploration_iterations
            ):
                self.unified_tracker.set_max_iterations(prompt_requirements.iteration_budget)
                logger.info(
                    f"Dynamic iterations from prompt: {prompt_requirements.iteration_budget}"
                )

        # Intelligent pipeline pre-request hook: get Q-learning recommendations
        # This enables RL-based mode transitions and optimal tool budget selection
        # PERF: Start as background task to run in parallel with sync work below
        intelligent_task = asyncio.create_task(
            self._prepare_intelligent_request(
                task=user_message,
                task_type=unified_task_type.value,
            )
        )

        # Get exploration iterations from unified tracker (replaces TASK_CONFIGS lookup)
        max_exploration_iterations = self.unified_tracker.max_exploration_iterations

        # Task prep: hints, complexity, reminders
        # This runs while intelligent_task executes in background
        task_classification, complexity_tool_budget = self._prepare_task(
            user_message, unified_task_type
        )

        # PERF: Await intelligent request after sync work completes
        intelligent_context = await intelligent_task
        if intelligent_context:
            # Inject optimized system prompt if provided
            if intelligent_context.get("system_prompt_addition"):
                self.add_message("system", intelligent_context["system_prompt_addition"])
                logger.debug("Injected intelligent pipeline optimized prompt")

        return (
            stream_metrics,
            start_time,
            total_tokens,
            cumulative_usage,
            max_total_iterations,
            max_exploration_iterations,
            total_iterations,
            force_completion,
            unified_task_type,
            task_classification,
            complexity_tool_budget,
        )

    async def _create_stream_context(self, user_message: str) -> StreamingChatContext:
        """Create a StreamingChatContext with all prepared data.

        This method encapsulates _prepare_stream results into a StreamingChatContext,
        enabling use of the StreamingChatHandler for testable iteration logic.

        Args:
            user_message: The user's message

        Returns:
            Populated StreamingChatContext ready for the streaming loop
        """
        # Get all prepared data from _prepare_stream
        (
            stream_metrics,
            start_time,
            total_tokens,
            cumulative_usage,
            max_total_iterations,
            max_exploration_iterations,
            total_iterations,
            force_completion,
            unified_task_type,
            task_classification,
            complexity_tool_budget,
        ) = await self._prepare_stream(user_message)

        # Classify task type based on keywords
        task_keywords = self._classify_task_keywords(user_message)

        # Create and populate context
        ctx = create_stream_context(
            user_message=user_message,
            max_iterations=max_total_iterations,
            max_exploration=max_exploration_iterations,
            tool_budget=complexity_tool_budget,
        )

        # Populate context with prepared data
        ctx.stream_metrics = stream_metrics
        ctx.start_time = start_time
        ctx.total_tokens = total_tokens
        ctx.cumulative_usage = cumulative_usage
        ctx.total_iterations = total_iterations
        ctx.force_completion = force_completion
        ctx.unified_task_type = unified_task_type
        ctx.task_classification = task_classification
        ctx.complexity_tool_budget = complexity_tool_budget

        # Add task keyword results
        # Reconcile is_analysis_task from both UnifiedClassifier (keyword-based) and
        # UnifiedTaskTracker (semantic-based). Either source detecting analysis = analysis task.
        # This fixes the mismatch where unified_task_type=ANALYZE but is_analysis_task=False
        ctx.is_analysis_task = task_keywords["is_analysis_task"] or unified_task_type.value in (
            "analyze",
            "analysis",
        )
        ctx.is_action_task = task_keywords["is_action_task"]
        ctx.needs_execution = task_keywords["needs_execution"]
        ctx.coarse_task_type = task_keywords["coarse_task_type"]

        # GAP-16: Set is_complex_task from ComplexityClassifier for lenient progress checking
        # This allows COMPLEX tasks (like refactoring, multi-file changes) to continue exploring
        # even when is_analysis_task and is_action_task are False
        if task_classification and hasattr(task_classification, "complexity"):
            ctx.is_complex_task = task_classification.complexity in (
                TaskComplexity.COMPLEX,
                TaskComplexity.ANALYSIS,
            )

        # Set goals for tool selection
        ctx.goals = self._tool_planner.infer_goals_from_message(user_message)

        # Sync tool tracking from orchestrator to context
        ctx.tool_budget = self.tool_budget
        ctx.tool_calls_used = self.tool_calls_used

        # Task Completion Detection Enhancement (Phase 2 - Feature Flag Protected)
        # Make detector available to intent classification for priority checks
        ctx.task_completion_detector = self._task_completion_detector

        return ctx

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

    def _check_time_limit_with_handler(
        self,
        stream_ctx: StreamingChatContext,
    ) -> Optional[StreamChunk]:
        """Check time limit and record Q-learning outcome.

        Creates RecoveryContext and delegates to recovery_coordinator.
        Also records Q-learning outcome when time limit is reached.

        Args:
            stream_ctx: The streaming context

        Returns:
            StreamChunk if time limit reached, None otherwise
        """
        # Create recovery context from current state
        recovery_ctx = self._create_recovery_context(stream_ctx)

        # Delegate to RecoveryCoordinator
        chunk = self._recovery_coordinator.check_time_limit(recovery_ctx)

        # Record Q-learning outcome (orchestrator-specific logic)
        if chunk:
            self._record_intelligent_outcome(
                success=stream_ctx.total_accumulated_chars > 200,
                quality_score=0.4 if stream_ctx.total_accumulated_chars > 200 else 0.2,
                user_satisfied=False,
                completed=False,
            )

        return chunk

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

    def _handle_force_tool_execution_with_handler(
        self,
        stream_ctx: StreamingChatContext,
        mentioned_tools: List[str],
        force_message: Optional[str] = None,
    ) -> None:
        """Handle force tool execution using the recovery coordinator.

        Creates RecoveryContext and delegates to recovery_coordinator viahandle_force_tool_execution() directly.


        Args:
            stream_ctx: The streaming context
            mentioned_tools: Tools that were mentioned but not executed
            force_message: Optional pre-crafted message to use instead of default
        """
        # Create recovery context from current state (reserved for RecoveryCoordinator)
        _recovery_ctx = self._create_recovery_context(stream_ctx)  # noqa: F841

        # Delegate to RecoveryCoordinator
        # Note: RecoveryCoordinator's implementation is currently a stub
        # For now, use the streaming handler directly until RecoveryCoordinator is updated
        self._streaming_handler.handle_force_tool_execution(
            stream_ctx, mentioned_tools, force_message
        )

        # Orchestrator-specific: increment turn counter
        self.unified_tracker.increment_turn()

    async def _execute_extracted_tool_call(
        self,
        stream_ctx: StreamingChatContext,
        extracted_call: ExtractedToolCall,
    ) -> AsyncIterator[StreamChunk]:
        """Execute a tool call that was extracted from model text.

        When the model mentions a tool in text but doesn't properly execute it,
        and we successfully extract the intended call, this method executes it.

        Args:
            stream_ctx: The streaming context
            extracted_call: The extracted tool call to execute

        Yields:
            StreamChunk objects for progress updates
        """
        tool_name = extracted_call.tool_name
        tool_args = extracted_call.arguments

        logger.info(
            f"[ExtractedToolExecution] Executing {tool_name} with args: "
            f"{list(tool_args.keys())}"
        )

        # Show status to user
        yield self._chunk_generator.generate_tool_start_chunk(
            tool_name=tool_name,
            status_msg=f"🔧 Auto-executing {tool_name} from model intent...",
        )

        # Emit extracted tool execution event
        from victor.core.events import get_observability_bus

        event_bus = get_observability_bus()
        event_bus.emit(
            topic="tool.extracted_execution",
            data={
                "tool_name": tool_name,
                "arguments": {k: str(v)[:100] for k, v in tool_args.items()},
                "confidence": extracted_call.confidence,
                "category": "tool",
            },
        )

        # Execute the tool
        try:
            context = {
                "cwd": getattr(self.settings, "cwd", "."),
                "provider": self.provider_name,
                "model": self.model,
            }

            result, success, error_msg = await self._execute_tool_with_retry(
                tool_name, tool_args, context
            )

            # Update tool usage tracking
            self.tool_calls_used += 1
            stream_ctx.tool_calls_used = self.tool_calls_used

            if success:
                # Format the result
                result_str = str(result.result) if hasattr(result, "result") else str(result)
                truncated_result = (
                    result_str[:2000] + "..." if len(result_str) > 2000 else result_str
                )

                # Add to messages as if the model had called it
                # First add a synthetic assistant message indicating the tool call
                self.add_message(
                    "assistant",
                    f"[Auto-executed {tool_name} based on my intent]",
                )
                # Then add the tool result
                self.add_message(
                    "tool",
                    truncated_result,
                    tool_call_id=f"extracted_{tool_name}_{self.tool_calls_used}",
                    name=tool_name,
                )

                # Yield success chunk
                yield self._chunk_generator.generate_tool_complete_chunk(
                    tool_name=tool_name,
                    elapsed_time=0.0,  # We don't track this for extracted calls
                    success=True,
                )

                logger.info(
                    f"[ExtractedToolExecution] {tool_name} succeeded. "
                    f"Result length: {len(result_str)} chars"
                )

                # Callback for tracking
                if hasattr(self, "_on_tool_complete_callback"):
                    from victor.tools.tool_pipeline import ToolCallResult

                    callback_result = ToolCallResult(
                        tool_name=tool_name,
                        tool_call_id=f"extracted_{tool_name}_{self.tool_calls_used}",
                        result=truncated_result,
                        success=True,
                        error=None,
                        arguments=tool_args,
                    )
                    self._on_tool_complete_callback(callback_result)

            else:
                # Tool failed - add error message
                self.add_message(
                    "tool",
                    f"Error executing {tool_name}: {error_msg}",
                    tool_call_id=f"extracted_{tool_name}_{self.tool_calls_used}",
                    name=tool_name,
                )

                yield self._chunk_generator.generate_tool_complete_chunk(
                    tool_name=tool_name,
                    elapsed_time=0.0,
                    success=False,
                )

                logger.warning(f"[ExtractedToolExecution] {tool_name} failed: {error_msg}")

        except Exception as e:
            logger.error(f"[ExtractedToolExecution] Exception executing {tool_name}: {e}")
            self.add_message(
                "tool",
                f"Error: {str(e)}",
                tool_call_id=f"extracted_{tool_name}_{self.tool_calls_used}",
                name=tool_name,
            )
            yield self._chunk_generator.generate_tool_complete_chunk(
                tool_name=tool_name,
                elapsed_time=0.0,
                success=False,
            )

        # Increment turn counter
        self.unified_tracker.increment_turn()

    def _check_progress_with_handler(self, stream_ctx: StreamingChatContext) -> bool:
        """Check progress using the recovery coordinator.

        Creates RecoveryContext and delegates to recovery_coordinator viacheck_progress() directly.


        Args:
            stream_ctx: The streaming context

        Returns:
            True if stuck/should force completion, False if making progress
        """
        # Create recovery context from current state
        recovery_ctx = self._create_recovery_context(stream_ctx)

        # Delegate to RecoveryCoordinator (uses UnifiedTaskTracker internally)
        # RecoveryCoordinator.check_progress returns True if making progress,
        # but callers expect True if stuck. Invert the result.
        is_making_progress = self._recovery_coordinator.check_progress(recovery_ctx)

        if not is_making_progress:
            # Stuck - set force_completion flag
            stream_ctx.force_completion = True
            return True

        return False

    def _handle_force_completion_with_handler(
        self,
        stream_ctx: StreamingChatContext,
    ) -> Optional[StreamChunk]:
        """Handle force completion using the recovery coordinator.

        Creates RecoveryContext and delegates to recovery_coordinator viahandle_force_completion() directly.


        Args:
            stream_ctx: The streaming context

        Returns:
            Warning chunk if force_completion is set, None otherwise
        """
        if not stream_ctx.force_completion:
            return None

        # Create recovery context from current state (reserved for RecoveryCoordinator)
        _recovery_ctx = self._create_recovery_context(stream_ctx)  # noqa: F841

        # Get stop decision from unified tracker for context
        stop_decision = self.unified_tracker.should_stop()
        stop_reason_value = stop_decision.reason.value
        stop_hint = stop_decision.hint

        # Delegate to streaming handler (RecoveryCoordinator implementation is incomplete)
        result = self._streaming_handler.handle_force_completion(
            stream_ctx, stop_reason_value, stop_hint
        )
        if result and result.chunks:
            return result.chunks[0]
        return None

    def _get_iteration_coordinator(self) -> IterationCoordinator:
        """Get or create the iteration coordinator.

        Creates the coordinator lazily to ensure unified_tracker is available.

        Returns:
            The iteration coordinator instance.
        """
        if self._iteration_coordinator is None:
            # Create coordinator with unified tracker as loop detector
            self._iteration_coordinator = create_coordinator(
                handler=self._streaming_handler,
                loop_detector=self.unified_tracker,
                settings=self.settings,
                config=CoordinatorConfig(
                    session_idle_timeout=getattr(self.settings, "session_idle_timeout", 180.0),
                    budget_warning_threshold=getattr(
                        self.settings, "tool_call_budget_warning_threshold", 250
                    ),
                ),
            )
        return self._iteration_coordinator

    def _run_pre_iteration_checks(
        self,
        stream_ctx: StreamingChatContext,
    ) -> Optional[StreamChunk]:
        """Run pre-iteration checks using the coordinator.

        Combines time limit, iteration limit, and force completion checks.

        Args:
            stream_ctx: The streaming context.

        Returns:
            StreamChunk if iteration should be skipped, None otherwise.
        """
        # Use handler's handle_iteration_start which combines all pre-checks
        result = self._streaming_handler.handle_iteration_start(stream_ctx)
        if result is not None:
            if result.chunks:
                return result.chunks[0]
            # If result says to break but no chunks, return a marker
            if result.should_break:
                return StreamChunk(content="", is_final=True)
        return None

    def _should_continue_streaming(
        self,
        stream_ctx: StreamingChatContext,
        has_tool_calls: bool,
        has_content: bool,
    ) -> tuple[bool, Optional[StreamChunk]]:
        """Determine if streaming loop should continue.

        Uses the coordinator for the continuation decision.

        Args:
            stream_ctx: The streaming context.
            has_tool_calls: Whether response has tool calls.
            has_content: Whether response has content.

        Returns:
            Tuple of (should_continue, optional_chunk_to_yield).
        """
        # Use handler's handle_continuation method
        result = self._streaming_handler.handle_continuation(
            stream_ctx, has_tool_calls, has_content
        )

        if result is not None:
            chunk = result.chunks[0] if result.chunks else None
            return not result.should_break, chunk

        return True, None

    async def _run_iteration_pre_checks(
        self,
        stream_ctx: StreamingChatContext,
        user_message: str,
    ) -> AsyncIterator[StreamChunk]:
        """Run all pre-iteration checks using coordinator.

        Combines cancellation, compaction, time limit, and iteration checks.
        Yields any notification chunks and handles early termination.

        Args:
            stream_ctx: The streaming context.
            user_message: The user's message (for compaction).

        Yields:
            StreamChunk notifications from checks.

        Note:
            Sets stream_ctx.force_completion if time limit reached.
            Caller should check stream_ctx after this method.
        """
        # 1. Cancellation check
        cancellation_chunk = self._handle_cancellation(stream_ctx.last_quality_score)
        if cancellation_chunk:
            yield cancellation_chunk
            stream_ctx.force_completion = True
            return

        # 2. Compaction (skip if background compaction active)
        if not self._context_manager.is_background_compaction_running:
            compaction_chunk = await self._handle_compaction_async(user_message)
            if compaction_chunk:
                yield compaction_chunk

        # 3. Time limit check via handler
        time_limit_chunk = self._check_time_limit_with_handler(stream_ctx)
        if time_limit_chunk:
            yield time_limit_chunk
            # Handler already set stream_ctx.force_completion = True

        # 4. Increment iteration
        stream_ctx.increment_iteration()

        # 5. Inject grounding feedback if pending
        if stream_ctx.pending_grounding_feedback:
            logger.info("Injecting pending grounding feedback as system message")
            self.add_message("system", stream_ctx.pending_grounding_feedback)
            stream_ctx.pending_grounding_feedback = ""

    def _log_iteration_debug(
        self,
        stream_ctx: StreamingChatContext,
        max_total_iterations: int,
    ) -> None:
        """Log iteration debug information.

        Args:
            stream_ctx: The streaming context.
            max_total_iterations: Maximum iterations allowed.
        """
        unique_resources = self.unified_tracker.unique_resources
        logger.debug(
            f"Iteration {stream_ctx.total_iterations}/{max_total_iterations}: "
            f"tool_calls_used={self.tool_calls_used}/{self.tool_budget}, "
            f"unique_resources={len(unique_resources)}, "
            f"force_completion={stream_ctx.force_completion}"
        )

        self.debug_logger.log_iteration_start(
            stream_ctx.total_iterations,
            tool_calls=self.tool_calls_used,
            files_read=len(unique_resources),
        )
        self.debug_logger.log_limits(
            tool_budget=self.tool_budget,
            tool_calls_used=self.tool_calls_used,
            max_iterations=max_total_iterations,
            current_iteration=stream_ctx.total_iterations,
            is_analysis_task=stream_ctx.is_analysis_task,
        )

    async def _handle_budget_exhausted(
        self,
        stream_ctx: StreamingChatContext,
    ) -> AsyncIterator[StreamChunk]:
        """Handle budget exhaustion by generating final summary.

        Args:
            stream_ctx: The streaming context.

        Yields:
            StreamChunk notifications and final content.
        """
        # Yield budget exhausted chunks
        for chunk in self._chunk_generator.get_budget_exhausted_chunks(stream_ctx):
            yield chunk

        # Try to generate final summary
        try:
            self.add_message(
                "system",
                "Tool budget reached. Provide a brief summary of what you found based on "
                "the information gathered. Do NOT attempt any more tool calls.",
            )
            response = await self.provider.chat(
                messages=self.messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=None,
            )
            if response and response.content:
                sanitized = self.sanitizer.sanitize(response.content)
                if sanitized:
                    yield self._chunk_generator.generate_content_chunk(sanitized, suffix="\n")
        except Exception as e:
            logger.warning(f"Failed to generate final summary: {e}")
            yield self._chunk_generator.generate_budget_error_chunk()

        # Finalize metrics
        final_metrics = self._finalize_stream_metrics(stream_ctx.cumulative_usage)
        elapsed_time = (
            final_metrics.total_duration if final_metrics else time.time() - stream_ctx.start_time
        )
        ttft = final_metrics.time_to_first_token if final_metrics else None
        cost_str = None
        if self.settings.show_cost_metrics and final_metrics:
            cost_str = final_metrics.format_cost()
        metrics_line = self._chunk_generator.format_budget_exhausted_metrics(
            stream_ctx, elapsed_time, ttft, cost_str
        )

        # Record Q-learning outcome
        self._record_intelligent_outcome(
            success=True,
            quality_score=stream_ctx.last_quality_score,
            user_satisfied=True,
            completed=True,
        )
        yield self._chunk_generator.generate_metrics_chunk(metrics_line, is_final=True, prefix="\n")

    async def _handle_force_final_response(
        self,
        stream_ctx: StreamingChatContext,
    ) -> AsyncIterator[StreamChunk]:
        """Force a final response without tools.

        Args:
            stream_ctx: The streaming context.

        Yields:
            StreamChunk with final content.
        """
        try:
            response = await self.provider.chat(
                messages=self.messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=None,  # No tools - force text response
            )
            if response and response.content:
                sanitized = self.sanitizer.sanitize(response.content)
                if sanitized:
                    self.add_message("assistant", sanitized)
                    yield self._chunk_generator.generate_content_chunk(sanitized)
        except Exception as e:
            logger.warning(f"Error forcing final response: {e}")
            yield self._chunk_generator.generate_force_response_error_chunk()

    async def _handle_empty_response_recovery(
        self,
        stream_ctx: StreamingChatContext,
        tools: Optional[List[Dict[str, Any]]],
    ) -> tuple[bool, Optional[List[Dict[str, Any]]], Optional[StreamChunk]]:
        """Handle empty response with retry recovery attempts.

        Attempts multiple recovery strategies with increasing temperature
        to get a useful response from the model.

        Args:
            stream_ctx: The streaming context.
            tools: Available tools for recovery attempts.

        Returns:
            Tuple of (recovery_success, recovered_tool_calls, final_chunk).
            - recovery_success: True if recovery produced content or tool calls
            - recovered_tool_calls: Tool calls extracted during recovery (or None)
            - final_chunk: Final chunk to yield if recovery produced text response
        """
        # Get recovery prompts via streaming handler
        has_thinking_mode = getattr(self.tool_calling_caps, "thinking_mode", False)
        thinking_prefix = getattr(self.tool_calling_caps, "thinking_disable_prefix", None)
        recovery_prompts = self._streaming_handler.get_recovery_prompts(
            ctx=stream_ctx,
            base_temperature=self.temperature,
            has_thinking_mode=has_thinking_mode,
            thinking_disable_prefix=thinking_prefix,
        )

        for attempt, (prompt, temp) in enumerate(recovery_prompts, 1):
            logger.info(f"Recovery attempt {attempt}/3 with temp={temp:.1f}")

            # Create temporary message list with recent context
            recent_messages = self.messages[-10:] if len(self.messages) > 10 else self.messages[:]
            recovery_messages = recent_messages + [Message(role="user", content=prompt)]

            # Check if tools should be enabled
            use_tools = self._streaming_handler.should_use_tools_for_recovery(stream_ctx, attempt)
            recovery_tools = tools if use_tools else None

            try:
                response = await self.provider.chat(
                    messages=recovery_messages,
                    model=self.model,
                    temperature=temp,
                    max_tokens=min(self.max_tokens, 1024),
                    tools=recovery_tools,
                )

                # Check for tool calls in recovery response
                if use_tools and response and response.tool_calls:
                    logger.info(
                        f"Recovery attempt {attempt}: model made {len(response.tool_calls)} tool call(s)"
                    )
                    self.add_message("user", prompt)
                    if response.content:
                        self.add_message("assistant", response.content)
                    return True, response.tool_calls, None

                if response and response.content:
                    logger.debug(f"Recovery attempt {attempt}: got {len(response.content)} chars")

                    # Try to extract tool calls from text
                    tool_calls = self._try_extract_tool_calls_from_text(response.content, prompt)
                    if tool_calls:
                        return True, tool_calls, None

                    # Check if we have useful text content
                    sanitized = self.sanitizer.sanitize(response.content)
                    if sanitized and len(sanitized) > 20:
                        self.add_message("assistant", sanitized)
                        final_chunk = self._chunk_generator.generate_content_chunk(
                            sanitized, is_final=True
                        )
                        return True, None, final_chunk
                    elif response.content and len(response.content) > 20:
                        self.add_message("assistant", response.content)
                        final_chunk = self._chunk_generator.generate_content_chunk(
                            response.content, is_final=True
                        )
                        return True, None, final_chunk
                else:
                    logger.debug(f"Recovery attempt {attempt}: empty response")

            except Exception as exc:
                await self._handle_recovery_exception(exc, attempt)

        # All recovery attempts failed
        return False, None, None

    def _try_extract_tool_calls_from_text(
        self,
        content: str,
        prompt: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Try to extract tool calls from text response.

        Args:
            content: Response content to parse.
            prompt: Recovery prompt used.

        Returns:
            List of tool call dicts if extraction succeeded, None otherwise.
        """
        try:
            from victor.agent.tool_calling.text_extractor import (
                extract_tool_calls_from_text,
            )

            valid_tool_names = {t.name for t in self.tools.list_tools(only_enabled=True)}
            extraction_result = extract_tool_calls_from_text(
                content, valid_tool_names=valid_tool_names
            )

            if extraction_result.success and extraction_result.tool_calls:
                logger.info(
                    f"Recovery: Extracted {len(extraction_result.tool_calls)} "
                    f"tool calls from text output"
                )
                tool_calls = [
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "id": f"recovery_{idx}",
                    }
                    for idx, tc in enumerate(extraction_result.tool_calls)
                ]
                self.add_message("user", prompt)
                if extraction_result.remaining_content:
                    self.add_message("assistant", extraction_result.remaining_content)
                return tool_calls
        except Exception as e:
            logger.debug(f"Text extraction failed during recovery: {e}")

        return None

    async def _handle_recovery_exception(
        self,
        exc: Exception,
        attempt: int,
    ) -> None:
        """Handle exception during recovery attempt.

        Args:
            exc: The exception that occurred.
            attempt: Current attempt number.
        """
        exc_str = str(exc)
        logger.warning(f"Recovery attempt {attempt} failed: {exc}")

        # Check for rate limit errors and extract wait time
        if "rate_limit" in exc_str.lower() or "429" in exc_str:
            import re

            wait_match = re.search(r"try again in (\d+(?:\.\d+)?)\s*s", exc_str, re.I)
            if wait_match:
                wait_time = float(wait_match.group(1))
                logger.info(f"Rate limited. Waiting {wait_time:.1f}s before retry...")
                await asyncio.sleep(min(wait_time + 0.5, 30.0))
            else:
                backoff = min(2**attempt, 15)
                logger.info(f"Rate limited. Waiting {backoff}s before retry...")
                await asyncio.sleep(backoff)

    def _parse_and_validate_tool_calls(
        self,
        tool_calls: Optional[List[Dict[str, Any]]],
        full_content: str,
    ) -> tuple[Optional[List[Dict[str, Any]]], str]:
        """Parse, validate, and normalize tool calls from provider response.

        Handles:
        1. Fallback parsing from content if no native tool calls
        2. Normalization to ensure tool_calls are dicts
        3. Filtering out disabled/invalid tool names
        4. Coercing arguments to dicts (some providers send JSON strings)

        Args:
            tool_calls: Native tool calls from provider (may be None)
            full_content: Full response content for fallback parsing

        Returns:
            Tuple of (validated_tool_calls, remaining_content)
            - validated_tool_calls: List of valid tool call dicts, or None
            - remaining_content: Content after extracting any embedded tool calls
        """
        # Use unified adapter-based tool call parsing with fallbacks
        if not tool_calls and full_content:
            logger.debug(
                f"No native tool_calls, attempting fallback parsing on content len={len(full_content)}"
            )
            parse_result = self._parse_tool_calls_with_adapter(full_content, tool_calls)
            if parse_result.tool_calls:
                # Convert ToolCall objects to dicts for compatibility
                tool_calls = [tc.to_dict() for tc in parse_result.tool_calls]
                logger.debug(
                    f"Fallback parser found {len(tool_calls)} tool calls: {[tc.get('name') for tc in tool_calls]}"
                )
                full_content = parse_result.remaining_content
            else:
                logger.debug("Fallback parser found no tool calls")

        # Ensure tool_calls is a list of dicts to avoid type errors from malformed provider output
        if tool_calls:
            normalized_tool_calls = [tc for tc in tool_calls if isinstance(tc, dict)]
            if len(normalized_tool_calls) != len(tool_calls):
                logger.warning(f"Dropped non-dict tool_calls: {tool_calls}")
            tool_calls = normalized_tool_calls or None
            logger.debug(f"After normalization: {len(tool_calls) if tool_calls else 0} tool_calls")

        # Filter out invalid/hallucinated tool names early (adapter already validates, but double-check for enabled)
        if tool_calls:
            valid_tool_calls = []
            for tc in tool_calls:
                name = tc.get("name", "")
                # Resolve shell aliases to appropriate enabled variant (run → shell_readonly in INITIAL)
                resolved_name = self._resolve_shell_variant(name)
                if resolved_name != name:
                    tc["name"] = resolved_name
                    name = resolved_name
                # Use ToolAccessController (with tiered config) instead of registry directly
                is_enabled = self.is_tool_enabled(name)
                logger.debug(f"Tool '{name}' enabled={is_enabled}")
                if is_enabled:
                    valid_tool_calls.append(tc)
                else:
                    logger.debug(f"Filtered out disabled tool: {name}")
            if len(valid_tool_calls) != len(tool_calls):
                logger.warning(
                    f"Filtered {len(tool_calls) - len(valid_tool_calls)} invalid tool calls"
                )
            tool_calls = valid_tool_calls or None
            logger.debug(
                f"After filtering: {len(tool_calls) if tool_calls else 0} valid tool_calls"
            )

        # Coerce arguments to dicts early (providers may stream JSON strings)
        if tool_calls:
            for tc in tool_calls:
                args = tc.get("arguments")
                if isinstance(args, str):
                    try:
                        tc["arguments"] = json.loads(args)
                    except Exception:
                        try:
                            tc["arguments"] = ast.literal_eval(args)
                        except Exception:
                            tc["arguments"] = {"value": args}
                elif args is None:
                    tc["arguments"] = {}

        return tool_calls, full_content

    async def _stream_provider_response(
        self,
        tools: Any,
        provider_kwargs: Dict[str, Any],
        stream_ctx: "StreamingChatContext",
    ) -> tuple[str, Any, float, bool]:
        """
        Stream response from provider and return the full content, tool_calls, total tokens, and garbage detection flag.

        Includes automatic retry with exponential backoff for rate limit errors (HTTP 429).

        Args:
            tools: Available tools for the provider
            provider_kwargs: Additional kwargs for the provider
            stream_ctx: The streaming context containing metrics and usage tracking

        Returns:
            tuple:
                - full_content (str): The accumulated content from the provider stream.
                - tool_calls (Any): Tool calls detected in the stream, if any.
                - total_tokens (float): Estimated total tokens in the streamed content.
                - garbage_detected (bool): True if garbage content was detected and the stream was stopped early, otherwise False.
        """
        # Delegate to rate-limit-aware wrapper
        return await self._stream_with_rate_limit_retry(tools, provider_kwargs, stream_ctx)

    def _get_rate_limit_wait_time(self, exc: Exception, attempt: int) -> float:
        """Get wait time for rate limit retry.

        Delegates to ProviderCoordinator for base wait time calculation (TD-002).

        Args:
            exc: The rate limit exception
            attempt: Current retry attempt number (0-indexed)

        Returns:
            Number of seconds to wait before retrying
        """
        # Get base wait time from coordinator (handles parsing retry_after, patterns, etc.)
        base_wait = self._provider_coordinator.get_rate_limit_wait_time(exc)

        # Apply exponential backoff based on attempt number
        # For attempt 0: base_wait * 1, attempt 1: base_wait * 2, etc.
        backoff_multiplier = 2**attempt
        wait_time = base_wait * backoff_multiplier

        # Cap at 5 minutes (300 seconds) - matches coordinator's max_rate_limit_wait
        return min(wait_time, 300.0)

    async def _stream_with_rate_limit_retry(
        self,
        tools: Any,
        provider_kwargs: Dict[str, Any],
        stream_ctx: "StreamingChatContext",
        max_retries: int = 3,
    ) -> tuple[str, Any, float, bool]:
        """Stream provider response with automatic rate limit retry.

        Wraps _stream_provider_response_inner with retry logic for rate limit errors.

        Args:
            tools: Available tools for the provider
            provider_kwargs: Additional kwargs for the provider
            stream_ctx: The streaming context containing metrics and usage tracking
            max_retries: Maximum number of retry attempts for rate limits

        Returns:
            Same tuple as _stream_provider_response
        """
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await self._stream_provider_response_inner(
                    tools, provider_kwargs, stream_ctx
                )
            except ProviderRateLimitError as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = self._get_rate_limit_wait_time(e, attempt)
                    logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Waiting {wait_time:.1f}s before retry..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Rate limit persisted after {max_retries + 1} attempts")
            except Exception as e:
                # Check for rate limit errors that aren't wrapped in ProviderRateLimitError
                exc_str = str(e).lower()
                if "rate_limit" in exc_str or "429" in exc_str or "rate limit" in exc_str:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = self._get_rate_limit_wait_time(e, attempt)
                        logger.warning(
                            f"Rate limit detected (attempt {attempt + 1}/{max_retries + 1}). "
                            f"Waiting {wait_time:.1f}s before retry..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Rate limit persisted after {max_retries + 1} attempts")
                else:
                    raise  # Re-raise non-rate-limit errors immediately

        # All retries exhausted - re-raise the last exception
        if last_exception:
            raise last_exception
        raise RuntimeError("Rate limit retry exhausted without exception")

    async def _stream_provider_response_inner(
        self,
        tools: Any,
        provider_kwargs: Dict[str, Any],
        stream_ctx: "StreamingChatContext",
    ) -> tuple[str, Any, float, bool]:
        """Inner implementation of stream_provider_response without retry logic.

        This is the actual streaming logic, called by _stream_with_rate_limit_retry.
        """
        full_content = ""
        tool_calls = None
        garbage_detected = False
        consecutive_garbage_chunks = 0
        max_garbage_chunks = 3
        total_tokens: float = 0

        async for chunk in self.provider.stream(
            messages=self.messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=tools,
            **provider_kwargs,
        ):
            # Garbage detection
            chunk, consecutive_garbage_chunks, garbage_detected = self._handle_stream_chunk(
                chunk, consecutive_garbage_chunks, max_garbage_chunks, garbage_detected
            )
            if chunk is None:
                continue

            full_content += chunk.content
            stream_ctx.stream_metrics.total_chunks += 1
            if chunk.content:
                self._metrics_collector.record_first_token()
                total_tokens += len(chunk.content) / 4
                stream_ctx.stream_metrics.total_content_length += len(chunk.content)

            if chunk.tool_calls:
                logger.debug(f"Received tool_calls in chunk: {chunk.tool_calls}")
                tool_calls = chunk.tool_calls
                stream_ctx.stream_metrics.tool_calls_count += len(chunk.tool_calls)

            if chunk.usage:
                for key in stream_ctx.cumulative_usage:
                    stream_ctx.cumulative_usage[key] += chunk.usage.get(key, 0)
                logger.debug(
                    f"Chunk usage: in={chunk.usage.get('prompt_tokens', 0)} "
                    f"out={chunk.usage.get('completion_tokens', 0)} "
                    f"cache_read={chunk.usage.get('cache_read_input_tokens', 0)}"
                )

            if tool_calls:
                break

        if garbage_detected and not tool_calls:
            logger.info("Setting force_completion due to garbage detection")

        stream_ctx.total_tokens = total_tokens
        return full_content, tool_calls, total_tokens, garbage_detected

    def _handle_stream_chunk(
        self,
        chunk: Any,
        consecutive_garbage_chunks: int,
        max_garbage_chunks: int,
        garbage_detected: bool,
    ) -> tuple[Any, int, bool]:
        """Handle garbage detection for a streaming chunk."""
        if chunk.content and self.sanitizer.is_garbage_content(chunk.content):
            consecutive_garbage_chunks += 1
            if consecutive_garbage_chunks >= max_garbage_chunks:
                if not garbage_detected:
                    garbage_detected = True
                    logger.warning(
                        f"Garbage content detected after {len(chunk.content)} chars - stopping stream early"
                    )
                return None, consecutive_garbage_chunks, garbage_detected
        else:
            consecutive_garbage_chunks = 0
        return chunk, consecutive_garbage_chunks, garbage_detected

    async def stream_chat(self, user_message: str) -> AsyncIterator[StreamChunk]:
        """Stream a chat response (public entrypoint).

        This method wraps the implementation to make phased refactors safer.

        Args:
            user_message: User's input message

        Returns:
            AsyncIterator yielding StreamChunk objects with incremental response
        """
        try:
            async for chunk in self._stream_chat_impl(user_message):
                yield chunk
        finally:
            # Update cumulative token usage after stream completes
            # This enables accurate token tracking for evaluations/benchmarks
            if hasattr(self, "_current_stream_context") and self._current_stream_context:
                ctx = self._current_stream_context
                if hasattr(ctx, "cumulative_usage"):
                    for key in self._cumulative_token_usage:
                        if key in ctx.cumulative_usage:
                            self._cumulative_token_usage[key] += ctx.cumulative_usage[key]
                    # Calculate total if not tracked by provider
                    if self._cumulative_token_usage["total_tokens"] == 0:
                        self._cumulative_token_usage["total_tokens"] = (
                            self._cumulative_token_usage["prompt_tokens"]
                            + self._cumulative_token_usage["completion_tokens"]
                        )

    async def _stream_chat_impl(self, user_message: str) -> AsyncIterator[StreamChunk]:
        """Implementation for streaming chat.

        Args:
            user_message: User's message

        Yields:
            StreamChunk objects with incremental response

        Note:
            Stream metrics (TTFT, throughput) are available via get_last_stream_metrics()
            after the stream completes.

        The stream can be cancelled by calling request_cancellation(). When cancelled,
        the stream will yield a final chunk indicating cancellation and stop.

        This method now uses StreamingChatContext to centralize state management,
        enabling testable iteration logic via StreamingChatHandler.
        """
        # Initialize and prepare using StreamingChatContext
        stream_ctx = await self._create_stream_context(user_message)

        # Store context reference for handler delegation methods
        self._current_stream_context = stream_ctx

        # Extract required files and outputs from user prompt for task completion tracking
        # This enables early termination when all requirements are met
        self._required_files = self._extract_required_files_from_prompt(user_message)
        self._required_outputs = self._extract_required_outputs_from_prompt(user_message)
        self._read_files_session.clear()  # Reset for new session
        self._all_files_read_nudge_sent = False  # Reset nudge flag for new session
        logger.debug(
            f"Task requirements extracted - files: {self._required_files}, "
            f"outputs: {self._required_outputs}"
        )

        # Emit task requirements extracted event
        if self._required_files or self._required_outputs:
            from victor.core.events import get_observability_bus

            event_bus = get_observability_bus()
            event_bus.emit(
                topic="state.task.requirements_extracted",
                data={
                    "required_files": self._required_files,
                    "required_outputs": self._required_outputs,
                    "file_count": len(self._required_files),
                    "output_count": len(self._required_outputs),
                    "category": "state",
                },
            )

        # Iteration limits - kept as read-only local references for readability
        # (These are configuration values that don't change during the loop)
        max_total_iterations = stream_ctx.max_total_iterations
        max_exploration_iterations = stream_ctx.max_exploration_iterations
        # Metrics aliases removed - use stream_ctx.stream_metrics, stream_ctx.start_time,
        # stream_ctx.total_tokens, stream_ctx.cumulative_usage directly
        # total_iterations removed - use stream_ctx.total_iterations directly
        # force_completion already moved to context-only access (stream_ctx.force_completion)
        # Task classification aliases removed - use stream_ctx.* directly:
        #   - stream_ctx.unified_task_type
        #   - stream_ctx.task_classification (unused in loop)
        #   - stream_ctx.complexity_tool_budget (unused in loop)
        #   - stream_ctx.coarse_task_type
        #   - stream_ctx.context_msg (updates via update_context_message())
        # Task classification flags - use stream_ctx.* directly for access
        # Aliases removed for: is_action_task, is_analysis_task, needs_execution

        # Detect intent and inject prompt guard for non-write tasks
        self._apply_intent_guard(user_message)

        # For compound analysis+edit tasks, unified_tracker handles exploration limits
        if stream_ctx.is_analysis_task and stream_ctx.unified_task_type.value in ("edit", "create"):
            logger.info(
                f"Compound task detected (analysis+{stream_ctx.unified_task_type.value}): "
                f"unified_tracker will use appropriate exploration limits"
            )

        logger.info(
            f"Task type classification: coarse={stream_ctx.coarse_task_type}, "
            f"unified={stream_ctx.unified_task_type.value}, is_analysis={stream_ctx.is_analysis_task}, "
            f"is_action={stream_ctx.is_action_task}"
        )

        # Apply guidance for analysis/action tasks
        self._apply_task_guidance(
            user_message,
            stream_ctx.unified_task_type,
            stream_ctx.is_analysis_task,
            stream_ctx.is_action_task,
            stream_ctx.needs_execution,
            max_exploration_iterations,
        )

        # Add guidance for action-oriented tasks
        if stream_ctx.is_action_task:
            logger.info(
                f"Detected action-oriented task - allowing up to {max_exploration_iterations} exploration iterations"
            )

            # needs_execution is already computed by _classify_task_keywords
            if stream_ctx.needs_execution:
                self.add_message(
                    "system",
                    "This is an action-oriented task requiring execution. "
                    "Follow this workflow: "
                    "1. CREATE the file/script with write_file or edit_files "
                    "2. EXECUTE it immediately with execute_bash (don't skip this step!) "
                    "3. SHOW the output to the user. "
                    "Minimize exploration and proceed directly to create→execute→show results.",
                )
            else:
                self.add_message(
                    "system",
                    "This is an action-oriented task (create/write/build). "
                    "Minimize exploration and proceed directly to creating what was requested. "
                    "Only explore if absolutely necessary to complete the task.",
                )

        goals = self._tool_planner.infer_goals_from_message(user_message)

        # Log all limits for debugging
        logger.info(
            f"Stream chat limits: "
            f"tool_budget={self.tool_budget}, "
            f"max_total_iterations={max_total_iterations}, "
            f"max_exploration_iterations={max_exploration_iterations}, "
            f"is_analysis_task={stream_ctx.is_analysis_task}, "
            f"is_action_task={stream_ctx.is_action_task}"
        )

        # Reset debug logger for new conversation turn
        self.debug_logger.reset()

        # Quality score and content threshold - accessed directly from stream_ctx
        # Quality updates use stream_ctx.update_quality_score()
        # Content accumulation is handled by stream_ctx.accumulate_content()
        # Aliases removed: last_quality_score, substantial_content_threshold

        while True:
            # === PRE-ITERATION CHECKS (via coordinator helper) ===
            # Handles: cancellation, compaction, time limit, iteration increment, grounding feedback
            cancelled = False
            async for pre_chunk in self._run_iteration_pre_checks(stream_ctx, user_message):
                yield pre_chunk
                # Check if cancellation occurred (force_completion set with empty content)
                if pre_chunk.content == "" and getattr(pre_chunk, "is_final", False):
                    cancelled = True
            if cancelled:
                return

            # Log iteration debug info
            self._log_iteration_debug(stream_ctx, max_total_iterations)

            # === CONTEXT AND ITERATION LIMIT CHECKS ===
            max_context = self._get_max_context_chars()
            handled, iter_chunk = await self._handle_context_and_iteration_limits(
                user_message,
                max_total_iterations,
                max_context,
                stream_ctx.total_iterations,
                stream_ctx.last_quality_score,
            )
            if iter_chunk:
                yield iter_chunk
            if handled:
                break

            tools = await self._select_tools_for_turn(stream_ctx.context_msg, goals)

            # Prepare optional thinking parameter for providers that support it (Anthropic)
            provider_kwargs = {}
            if self.thinking:
                # Anthropic extended thinking format
                provider_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}

            full_content, tool_calls, _, garbage_detected = await self._stream_provider_response(
                tools=tools,
                provider_kwargs=provider_kwargs,
                stream_ctx=stream_ctx,
            )

            # Debug: Log response details
            content_preview = full_content[:200] if full_content else "(empty)"
            logger.debug(
                f"_stream_provider_response returned: content_len={len(full_content) if full_content else 0}, "
                f"native_tool_calls={len(tool_calls) if tool_calls else 0}, tokens={stream_ctx.total_tokens}, "
                f"garbage={garbage_detected}, content_preview={content_preview!r}"
            )

            # If garbage was detected, force completion on next iteration
            if garbage_detected and not tool_calls:
                stream_ctx.force_completion = True
                logger.info("Setting force_completion due to garbage detection")

            # Parse, validate, and normalize tool calls (fallback parsing, filtering, arg coercion)
            tool_calls, full_content = self._parse_and_validate_tool_calls(tool_calls, full_content)

            # Task Completion Detection Enhancement (Phase 2 - Feature Flag Protected)
            # Analyze response for explicit completion signals when feature flag is enabled
            if self._task_completion_detector and full_content:
                from victor.agent.task_completion import CompletionConfidence

                self._task_completion_detector.analyze_response(full_content)
                confidence = self._task_completion_detector.get_completion_confidence()

                # HIGH confidence (active signal) triggers immediate completion
                if confidence == CompletionConfidence.HIGH:
                    logger.info(
                        "Task completion: HIGH confidence detected (active signal), "
                        "forcing completion after this response"
                    )
                    stream_ctx.force_completion = True

                # MEDIUM confidence (file mods + passive) logs info but doesn't force
                elif confidence == CompletionConfidence.MEDIUM:
                    logger.info(
                        "Task completion: MEDIUM confidence detected (file mods + passive signal)"
                    )

            # DEBUG: Log complete tool calls from LLM for diagnosis
            if tool_calls:
                logger.debug(f"LLM tool calls ({len(tool_calls)} total):")
                for i, tc in enumerate(tool_calls):
                    tc_name = tc.get("name", "unknown")
                    tc_args = tc.get("arguments", {})
                    # Truncate large args for readability
                    args_str = str(tc_args)
                    if len(args_str) > 500:
                        args_str = args_str[:500] + "...(truncated)"
                    logger.debug(f"  [{i+1}] {tc_name}: {args_str}")

            # Initialize mentioned_tools_detected for later use in continuation action
            mentioned_tools_detected: List[str] = []

            # Check for mentioned tools early for recovery integration
            if full_content and not tool_calls:
                mentioned_tools_detected = ContinuationStrategy.detect_mentioned_tools(
                    full_content, list(_ALL_TOOL_NAMES), TOOL_ALIASES
                )

            # Use recovery integration to detect and handle failures
            # This runs after each response to check for stuck states, empty responses, etc.
            recovery_action = await self._handle_recovery_with_integration(
                stream_ctx=stream_ctx,
                full_content=full_content,
                tool_calls=tool_calls,
                mentioned_tools=mentioned_tools_detected or None,
            )

            # Apply recovery action if not just "continue"
            if recovery_action.action != "continue":
                recovery_chunk = self._apply_recovery_action(recovery_action, stream_ctx)
                if recovery_chunk:
                    yield recovery_chunk
                    if recovery_chunk.is_final:
                        # Record outcome for Q-learning
                        self._recovery_integration.record_outcome(success=False)
                        return
                # If action was retry/force_summary, continue the loop with updated state
                if recovery_action.action in ("retry", "force_summary"):
                    continue

            if full_content:
                # Sanitize response to remove malformed patterns from local models
                sanitized = self.sanitizer.sanitize(full_content)
                if sanitized:
                    self.add_message("assistant", sanitized)
                else:
                    # If sanitization removed everything, use stripped markup as fallback
                    plain_text = self.sanitizer.strip_markup(full_content)
                    if plain_text:
                        self.add_message("assistant", plain_text)

                # Log info if model mentioned tools but didn't execute them
                # (mentioned_tools_detected was already computed earlier for recovery integration)
                # This is common with local models that struggle with native tool calling
                if mentioned_tools_detected:
                    tools_str = ", ".join(mentioned_tools_detected)
                    logger.info(
                        f"Model mentioned tool(s) [{tools_str}] in text without executing. "
                        "Common with local models - tool syntax detected in response content."
                    )
            elif not tool_calls:
                # No content and no tool calls - check for natural completion
                # via recovery coordinator directly (checks if substantial content was provided)
                recovery_ctx = self._create_recovery_context(stream_ctx)
                final_chunk = self._recovery_coordinator.check_natural_completion(
                    recovery_ctx, has_tool_calls=False, content_length=0
                )
                if final_chunk:
                    yield final_chunk
                    return

                # No substantial content yet - attempt aggressive recovery
                logger.warning("Model returned empty response - attempting aggressive recovery")

                # Track empty responses - delegates to recovery coordinator
                recovery_ctx = self._create_recovery_context(stream_ctx)
                recovery_chunk, should_force = self._recovery_coordinator.handle_empty_response(
                    recovery_ctx
                )
                if recovery_chunk:
                    yield recovery_chunk
                    # CRITICAL: Handler already set stream_ctx.force_completion if needed
                    # The should_force flag confirms the handler's decision
                    continue

                # Delegate empty response recovery to helper method
                recovery_success, recovered_tool_calls, final_chunk = (
                    await self._handle_empty_response_recovery(stream_ctx, tools)
                )

                if recovery_success:
                    if final_chunk:
                        # Recovery produced text response - yield and return
                        yield final_chunk
                        return
                    elif recovered_tool_calls:
                        # Recovery produced tool calls - continue main loop
                        tool_calls = recovered_tool_calls
                        logger.info(
                            f"Recovery produced {len(tool_calls)} tool call(s) - continuing main loop"
                        )
                else:
                    # All recovery attempts failed - get fallback message
                    recovery_ctx = self._create_recovery_context(stream_ctx)
                    fallback_msg = self._recovery_coordinator.get_recovery_fallback_message(
                        recovery_ctx
                    )
                    self._record_intelligent_outcome(
                        success=False,
                        quality_score=0.3,
                        user_satisfied=False,
                        completed=False,
                    )
                    yield self._chunk_generator.generate_content_chunk(fallback_msg, is_final=True)
                    return

            # Record tool calls in progress tracker for loop detection
            # Progress tracker handles unique resource tracking internally
            for tc in tool_calls or []:
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})

                # Record tool call in unified tracker (single source of truth)
                self.unified_tracker.record_tool_call(tool_name, tool_args)

            content_length = len(full_content.strip())

            # Record iteration in unified tracker (single source of truth)
            self.unified_tracker.record_iteration(content_length)

            # Intelligent pipeline post-iteration hook: validate response quality
            # This enables quality scoring, hallucination detection, and Q-learning feedback
            if full_content and len(full_content.strip()) > 50:
                quality_result = await self._validate_intelligent_response(
                    response=full_content,
                    query=user_message,
                    tool_calls=self.tool_calls_used,
                    task_type=stream_ctx.unified_task_type.value,
                )
                if quality_result and not quality_result.get("is_grounded", True):
                    # Log grounding issues for debugging
                    issues = quality_result.get("grounding_issues", [])
                    if issues:
                        logger.warning(
                            f"IntelligentPipeline detected grounding issues: {issues[:3]}"
                        )
                    # If retry is allowed, inject grounding feedback as system message
                    if quality_result.get("should_retry"):
                        grounding_feedback = quality_result.get("grounding_feedback", "")
                        if grounding_feedback:
                            logger.info(
                                f"Injecting grounding feedback for retry: {len(grounding_feedback)} chars"
                            )
                            # Store feedback for injection in next iteration
                            stream_ctx.pending_grounding_feedback = grounding_feedback

                # Update quality score for Q-learning outcome recording
                if quality_result:
                    new_score = quality_result.get("quality_score", stream_ctx.last_quality_score)
                    stream_ctx.update_quality_score(new_score)

                # Check for force finalize from grounding failures
                # This prevents infinite loops when grounding keeps failing
                if quality_result and quality_result.get("should_finalize"):
                    finalize_reason = quality_result.get(
                        "finalize_reason", "grounding limit exceeded"
                    )
                    logger.warning(
                        f"Force finalize triggered: {finalize_reason}. "
                        "Stopping continuation to prevent infinite loop."
                    )
                    # Set flag to force completion in continuation logic
                    self._force_finalize = True

            # Check for loop warning via streaming handler directly
            unified_loop_warning = self.unified_tracker.check_loop_warning()
            loop_warning_chunk = self._streaming_handler.handle_loop_warning(
                stream_ctx, unified_loop_warning
            )
            if loop_warning_chunk:
                logger.warning(f"UnifiedTaskTracker loop warning: {unified_loop_warning}")
                yield loop_warning_chunk
            else:
                # PRIMARY: Check UnifiedTaskTracker for stop decision via recovery coordinator
                recovery_ctx = self._create_recovery_context(stream_ctx)
                was_triggered, hint = self._recovery_coordinator.check_force_action(recovery_ctx)
                if was_triggered:
                    logger.info(
                        f"UnifiedTaskTracker forcing action: {hint}, "
                        f"metrics={self.unified_tracker.get_metrics()}"
                    )

                logger.debug(f"After streaming pass, tool_calls = {tool_calls}")

                if not tool_calls:
                    # === INTENT CLASSIFICATION (P0 SRP refactor) ===
                    # Delegated to IntentClassificationHandler for testability and SRP compliance.
                    # The handler manages content yielding, intent classification (with caching),
                    # response loop detection, and continuation action determination.

                    # Create intent classification handler lazily (reused across iterations)
                    if not hasattr(self, "_intent_classification_handler"):
                        self._intent_classification_handler = create_intent_classification_handler(
                            self
                        )

                    # Ensure tracking variables are initialized
                    if not hasattr(self, "_continuation_prompts"):
                        self._continuation_prompts = 0
                    if not hasattr(self, "_asking_input_prompts"):
                        self._asking_input_prompts = 0
                    if not hasattr(self, "_consecutive_blocked_attempts"):
                        self._consecutive_blocked_attempts = 0
                    if not hasattr(self, "_cumulative_prompt_interventions"):
                        self._cumulative_prompt_interventions = 0

                    # Create tracking state from orchestrator
                    tracking_state = create_tracking_state(self)

                    # Delegate to IntentClassificationHandler
                    intent_result = (
                        self._intent_classification_handler.classify_and_determine_action(
                            stream_ctx=stream_ctx,
                            full_content=full_content,
                            content_length=content_length,
                            mentioned_tools=mentioned_tools_detected,
                            tracking_state=tracking_state,
                        )
                    )

                    # Yield chunks from handler (content yielded to UI)
                    for chunk in intent_result.chunks:
                        yield chunk

                    # Clear full_content if handler yielded it
                    if intent_result.content_cleared:
                        full_content = ""

                    # Apply state updates back to orchestrator
                    force_finalize_used = (
                        tracking_state.force_finalize and intent_result.action == "finish"
                    )
                    apply_tracking_state_updates(
                        self, intent_result.state_updates, force_finalize_used
                    )

                    # Get action result for ContinuationHandler
                    action_result = intent_result.action_result
                    action = intent_result.action

                    # Log the action
                    logger.info(
                        f"Continuation action: {action} - {action_result.get('reason', 'unknown')}"
                    )

                    # === CONTINUATION ACTION HANDLING (P0 SRP refactor) ===
                    # Delegated to ContinuationHandler for testability and SRP compliance.
                    # The handler processes actions like prompt_tool_call, request_summary,
                    # execute_extracted_tool, force_tool_execution, finish, etc.

                    # Create continuation handler lazily (reused across iterations)
                    if not hasattr(self, "_continuation_handler"):
                        self._continuation_handler = create_continuation_handler(self)

                    # Update action in action_result if it was overridden
                    action_result["action"] = action

                    # Delegate to ContinuationHandler
                    continuation_result = await self._continuation_handler.handle_action(
                        action_result=action_result,
                        stream_ctx=stream_ctx,
                        full_content=full_content,
                    )

                    # Yield chunks from handler
                    for chunk in continuation_result.chunks:
                        yield chunk

                    # Apply state updates from handler
                    if "cumulative_prompt_interventions" in continuation_result.state_updates:
                        self._cumulative_prompt_interventions = continuation_result.state_updates[
                            "cumulative_prompt_interventions"
                        ]

                    # Check control flags
                    if continuation_result.should_return:
                        return
                    # If should_skip_rest, continue to tool execution section
                    # (which will be a no-op since tool_calls is empty)

                # === TOOL EXECUTION PHASE (P0 SRP refactor) ===
                # Delegated to ToolExecutionHandler for testability and SRP compliance.
                # The handler manages budget checks, filtering, execution, and result generation.

                # Create tool execution handler lazily (reused across iterations)
                if not hasattr(self, "_tool_execution_handler"):
                    self._tool_execution_handler = create_tool_execution_handler(self)

                # Update observed files for reminder tracking
                self._tool_execution_handler.update_observed_files(
                    set(self.observed_files) if self.observed_files else set()
                )

                # Delegate to ToolExecutionHandler
                tool_exec_result = await self._tool_execution_handler.execute_tools(
                    stream_ctx=stream_ctx,
                    tool_calls=tool_calls,
                    user_message=user_message,
                    full_content=full_content,
                    tool_calls_used=self.tool_calls_used,
                    tool_budget=self.tool_budget,
                )

                # Yield chunks from handler
                for chunk in tool_exec_result.chunks:
                    yield chunk

                # Update tool calls counter
                self.tool_calls_used += tool_exec_result.tool_calls_executed

                # Check control flags
                if tool_exec_result.should_return:
                    return

    async def _execute_tool_with_retry(
        self, tool_name: str, tool_args: Dict[str, Any], context: Dict[str, Any]
    ) -> tuple[Any, bool, Optional[str]]:
        """Execute a tool with retry logic and exponential backoff.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            context: Execution context

        Returns:
            Tuple of (result, success, error_message or None)
        """
        # Try cache first for allowlisted tools
        if self.tool_cache:
            cached = self.tool_cache.get(tool_name, tool_args)
            if cached is not None:
                logger.debug(f"Cache hit for tool '{tool_name}'")
                return cached, True, None

        retry_enabled = getattr(self.settings, "tool_retry_enabled", True)
        max_attempts = getattr(self.settings, "tool_retry_max_attempts", 3) if retry_enabled else 1
        base_delay = getattr(self.settings, "tool_retry_base_delay", 1.0)
        max_delay = getattr(self.settings, "tool_retry_max_delay", 10.0)

        last_error = None
        for attempt in range(max_attempts):
            try:
                result = await self.tools.execute(tool_name, context=context, **tool_args)

                if result.success:
                    if self.tool_cache:
                        self.tool_cache.set(tool_name, tool_args, result)
                        invalidating_tools = {
                            "write_file",
                            "edit_files",
                            "execute_bash",
                            "git",
                            "docker",
                        }
                        if tool_name in invalidating_tools:
                            touched_paths = []
                            if "path" in tool_args:
                                touched_paths.append(tool_args["path"])
                            if "paths" in tool_args and isinstance(tool_args["paths"], list):
                                touched_paths.extend(tool_args["paths"])
                            if touched_paths:
                                self.tool_cache.invalidate_paths(touched_paths)
                            else:
                                namespaces_to_clear = [
                                    "code_search",
                                    "semantic_code_search",
                                    "list_directory",
                                ]
                                self.tool_cache.clear_namespaces(namespaces_to_clear)
                    if attempt > 0:
                        logger.info(
                            f"Tool '{tool_name}' succeeded on retry attempt {attempt + 1}/{max_attempts}"
                        )

                    # Task Completion Detection Enhancement (Phase 2 - Feature Flag Protected)
                    # Record successful tool execution for completion detection
                    if self._task_completion_detector:
                        tool_result = {"success": True}
                        # Include path if available
                        if "path" in tool_args:
                            tool_result["path"] = tool_args["path"]
                        elif "file_path" in tool_args:
                            tool_result["file_path"] = tool_args["file_path"]
                        self._task_completion_detector.record_tool_result(tool_name, tool_result)

                    return result, True, None
                else:
                    # Tool returned failure - check if retryable
                    error_msg = result.error or "Unknown error"

                    # Don't retry validation errors or permanent failures
                    non_retryable_errors = ["Invalid", "Missing required", "Not found", "disabled"]
                    if any(err in error_msg for err in non_retryable_errors):
                        logger.debug(
                            f"Tool '{tool_name}' failed with non-retryable error: {error_msg}"
                        )
                        return result, False, error_msg

                    last_error = error_msg
                    if attempt < max_attempts - 1:
                        # Calculate exponential backoff delay
                        delay = min(base_delay * (2**attempt), max_delay)
                        logger.warning(
                            f"Tool '{tool_name}' failed (attempt {attempt + 1}/{max_attempts}): {error_msg}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Tool '{tool_name}' failed after {max_attempts} attempts: {error_msg}"
                        )
                        return result, False, error_msg

            except (ToolNotFoundError, ToolValidationError, PermissionError) as e:
                # Non-retryable errors - fail immediately
                logger.error(f"Tool '{tool_name}' permanent failure: {e}")
                return None, False, str(e)
            except (TimeoutError, ConnectionError, asyncio.TimeoutError) as e:
                # Retryable transient errors
                last_error = str(e)
                if attempt < max_attempts - 1:
                    delay = min(base_delay * (2**attempt), max_delay)
                    logger.warning(
                        f"Tool '{tool_name}' transient error (attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Tool '{tool_name}' failed after {max_attempts} attempts: {e}")
                    return None, False, last_error
            except Exception as e:
                # Unknown errors - log and retry with caution
                last_error = str(e)
                if attempt < max_attempts - 1:
                    delay = min(base_delay * (2**attempt), max_delay)
                    logger.warning(
                        f"Tool '{tool_name}' unexpected error (attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Tool '{tool_name}' raised exception after {max_attempts} attempts: {e}"
                    )
                    return None, False, last_error

        # Should not reach here, but handle it anyway
        return None, False, last_error or "Unknown error"

    async def _handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle tool calls from the model.

        Args:
            tool_calls: List of tool call requests
        """
        if not tool_calls:
            return []

        results: List[Dict[str, Any]] = []

        for tool_call in tool_calls:
            # Validate tool call structure
            if not isinstance(tool_call, dict):
                self.console.print(
                    f"[yellow]⚠ Skipping invalid tool call (not a dict): {tool_call}[/]"
                )
                continue

            tool_name = tool_call.get("name")
            if not tool_name:
                self.console.print(f"[yellow]⚠ Skipping tool call without name: {tool_call}[/]")
                # GAP-5 FIX: Add feedback so model learns from missing tool name
                results.append(
                    {
                        "tool_name": "",
                        "success": False,
                        "result": None,
                        "error": "Tool call missing name. Each tool call must include a 'name' field. "
                        "Please specify which tool you want to use.",
                    }
                )
                continue

            # Validate tool name format (reject hallucinated/malformed names)
            if not self.sanitizer.is_valid_tool_name(tool_name):
                self.console.print(
                    f"[yellow]⚠ Skipping invalid/hallucinated tool name: {tool_name}[/]"
                )
                # GAP-5 FIX: Add feedback so model learns from invalid tool name
                # Instead of silently dropping, return an error result the model can learn from
                results.append(
                    {
                        "tool_name": tool_name,
                        "success": False,
                        "result": None,
                        "error": f"Invalid tool name '{tool_name}'. This tool does not exist. "
                        "Use only tools from the provided tool list. "
                        "Check for typos or hallucinated tool names.",
                    }
                )
                continue

            # Resolve legacy/alias names to canonical form before checks
            try:
                from victor.tools.decorators import resolve_tool_name

                canonical_tool_name = resolve_tool_name(tool_name)
            except Exception:
                canonical_tool_name = tool_name

            # Skip unknown tools immediately (no retries, no budget cost)
            # Use ToolAccessController (with tiered config) instead of registry directly
            if not self.is_tool_enabled(canonical_tool_name):
                # Log original and canonical names to aid debugging in tests
                self.console.print(
                    f"[yellow]⚠ Skipping unknown or disabled tool: {tool_name} (resolved: {canonical_tool_name})[/]"
                )
                # GAP-5 FIX: Add feedback so model learns from unknown/disabled tool
                # Instead of silently dropping, return an error result the model can learn from
                results.append(
                    {
                        "tool_name": tool_name,
                        "success": False,
                        "result": None,
                        "error": f"Tool '{tool_name}' is not available. It may be disabled, not registered, "
                        "or not included in the current tool selection. "
                        "Use only the tools listed in your available tools.",
                    }
                )
                continue

            if self.tool_calls_used >= self.tool_budget:
                self.console.print(
                    f"[yellow]⚠ Tool budget reached ({self.tool_budget}); skipping remaining tool calls.[/]"
                )
                break

            # Use the canonical name for execution and downstream bookkeeping
            original_tool_name = tool_name  # Preserve for alias-based inference
            tool_name = canonical_tool_name

            tool_args = tool_call.get("arguments", {})

            # Providers sometimes stream arguments as JSON strings; normalize to dict early
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except Exception:
                    try:
                        tool_args = ast.literal_eval(tool_args)
                    except Exception:
                        tool_args = {"value": tool_args}
            elif tool_args is None:
                tool_args = {}

            # Normalize arguments to handle malformed JSON (e.g., Python vs JSON syntax)
            normalized_args, strategy = self.argument_normalizer.normalize_arguments(
                tool_args, tool_name
            )

            # DEBUG: Log normalized arguments with types
            original_types = {k: type(v).__name__ for k, v in tool_args.items()}
            normalized_types = {k: type(v).__name__ for k, v in normalized_args.items()}
            logger.debug(
                f"[NORMALIZE] {tool_name}: Original types={original_types}, "
                f"Normalized types={normalized_types}"
            )

            # Apply adapter-based normalization for missing required parameters
            # This handles provider-specific quirks like Gemini omitting 'path' for list_directory
            before_adapter = normalized_args.copy()
            normalized_args = self.tool_adapter.normalize_arguments(normalized_args, tool_name)

            # DEBUG: Log if adapter changed the types
            if before_adapter != normalized_args:
                before_types = {k: type(v).__name__ for k, v in before_adapter.items()}
                after_types = {k: type(v).__name__ for k, v in normalized_args.items()}
                logger.debug(
                    f"[ADAPTER] {tool_name}: Changed arguments - "
                    f"Before={before_types}, After={after_types}"
                )

            # Infer operation from alias for git tool (e.g., git_log → git with operation=log)
            normalized_args = infer_git_operation(
                original_tool_name, canonical_tool_name, normalized_args
            )

            # Skip repeated failing calls with identical signature to avoid tight loops
            try:
                signature = (tool_name, json.dumps(normalized_args, sort_keys=True, default=str))
            except Exception:
                signature = (tool_name, str(normalized_args))
            if signature in self.failed_tool_signatures:
                self.console.print(
                    f"[yellow]⚠ Skipping repeated failing call to '{tool_name}' with same arguments[/]"
                )
                continue

            # Log normalization if applied (for debugging and monitoring)
            if strategy != NormalizationStrategy.DIRECT:
                logger.warning(
                    f"Applied {strategy.value} normalization to {tool_name} arguments. "
                    f"Original: {tool_args} → Normalized: {normalized_args}"
                )
                self.console.print(f"[yellow]⚙ Normalized arguments via {strategy.value}[/]")
            else:
                # Log type coercion even when strategy is DIRECT
                args_changed = tool_args != normalized_args
                if args_changed:
                    logger.debug(
                        f"Type coercion applied to {tool_name} arguments. "
                        f"Original: {tool_args} → Normalized: {normalized_args}"
                    )

            self.usage_logger.log_event(
                "tool_call", {"tool_name": tool_name, "tool_args": normalized_args}
            )

            # Tool execution message handled by streaming - avoid duplicate
            logger.debug(f"Executing tool: {tool_name}")

            start = time.monotonic()

            # Create context for the tool
            context = {
                "code_manager": self.code_manager,
                "provider": self.provider,
                "model": self.model,
                "tool_registry": self.tools,
                "workflow_registry": self.workflow_registry,
                "settings": self.settings,
            }

            # Execute tool via centralized ToolExecutor (handles retry, caching, metrics)
            # Note: skip_normalization=True since we already normalized above
            final_types = {k: type(v).__name__ for k, v in normalized_args.items()}
            logger.debug(f"[EXECUTE] {tool_name}: Final argument types={final_types}")
            exec_result = await self.tool_executor.execute(
                tool_name=tool_name,
                arguments=normalized_args,
                context=context,
                skip_normalization=True,
            )
            success = exec_result.success
            error_msg = exec_result.error

            # Reset activity timer after tool execution to prevent idle timeout during active work
            if hasattr(self, "_current_stream_context") and self._current_stream_context:
                self._current_stream_context.reset_activity_timer()

            # Update counters and tracking
            self.tool_calls_used += 1
            self.executed_tools.append(tool_name)
            if tool_name == "read" and "path" in normalized_args:
                self.observed_files.add(str(normalized_args.get("path")))

            # Reset continuation prompts counter on successful tool call
            # This allows the model to get fresh continuation prompts if it pauses again
            if hasattr(self, "_continuation_prompts") and self._continuation_prompts > 0:
                logger.debug(
                    f"Resetting continuation prompts counter (was {self._continuation_prompts}) after successful tool call"
                )
                self._continuation_prompts = 0
                # Also reset the tool call tracking for stuck loop detection
                if hasattr(self, "_tool_calls_at_continuation_start"):
                    self._tool_calls_at_continuation_start = self.tool_calls_used

            # Reset asking input prompts counter on successful tool call
            if hasattr(self, "_asking_input_prompts") and self._asking_input_prompts > 0:
                logger.debug(
                    f"Resetting asking input prompts counter (was {self._asking_input_prompts}) after successful tool call"
                )
                self._asking_input_prompts = 0

            # Calculate execution time
            elapsed_ms = (time.monotonic() - start) * 1000  # Convert to milliseconds

            # Record tool execution analytics (including error type for UsageAnalytics)
            error_type = (
                type(exec_result.error).__name__ if exec_result.error and not success else None
            )
            self._record_tool_execution(tool_name, success, elapsed_ms, error_type=error_type)

            # Update conversation state machine for stage detection
            self.conversation_state.record_tool_execution(tool_name, normalized_args)

            # Update unified tracker for milestone tracking (single source of truth)
            result_dict = {"success": success}
            if hasattr(exec_result, "result") and exec_result.result:
                result_dict["result"] = exec_result.result
            self.unified_tracker.update_from_tool_call(tool_name, normalized_args, result_dict)

            # ToolExecutionResult stores actual output in .result field
            output = exec_result.result if success else None

            # Check for semantic failure in result dict (e.g., edit_files returning success=False)
            semantic_success = success
            if success and isinstance(output, dict) and output.get("success") is False:
                semantic_success = False
                # Extract error from result dict
                error_msg = output.get("error", "Operation returned success=False")

            # Only set error_display for failures; keep None for successes
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
                # Success message handled by streaming - log for debug only
                logger.debug(f"Tool {tool_name} executed successfully ({elapsed_ms:.0f}ms)")

                # Format tool output with clear boundaries to prevent hallucination
                # Use structured format that models recognize as authoritative
                formatted_output = self._format_tool_output(tool_name, normalized_args, output)

                # Log actual tool output for debugging (truncated for readability)
                output_preview = str(output)[:500] if output else "<empty>"
                if len(str(output)) > 500:
                    output_preview += f"... [truncated, total {len(str(output))} chars]"
                logger.debug(f"Tool '{tool_name}' actual output:\n{output_preview}")

                # Add tool result to conversation with proper role
                # Use "user" role but with clear TOOL_OUTPUT markers that models recognize
                self.add_message(
                    "user",
                    formatted_output,
                )
                results.append(
                    {
                        "name": tool_name,
                        "success": True,
                        "elapsed": time.monotonic() - start,
                        "args": normalized_args,  # Include args for diff display
                    }
                )
            else:
                self.failed_tool_signatures.add(signature)
                # error_display was already set above from exec_result.error or semantic failure
                self.console.print(
                    f"[red]✗ Tool execution failed: {error_display}[/] [dim]({elapsed_ms:.0f}ms)[/dim]"
                )

                # ALWAYS pass error details back to the model so it can understand and continue
                # This handles both:
                # 1. Execution failures (success=False): tool raised exception
                # 2. Semantic failures (success=True, semantic_success=False): tool returned error
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

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session metadata dictionaries
        """
        if not self.memory_manager:
            return []

        try:
            sessions = self.memory_manager.list_sessions(limit=limit)
            return [
                {
                    "session_id": s.session_id,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "last_activity": s.last_activity.isoformat() if s.last_activity else None,
                    "project_path": s.project_path,
                    "provider": s.provider,
                    "model": s.model,
                    "message_count": len(s.messages),
                }
                for s in sessions
            ]
        except Exception as e:
            logger.warning(f"Failed to get recent sessions: {e}")
            return []

    def recover_session(self, session_id: str) -> bool:
        """Recover a previous conversation session.

        Delegates to LifecycleManager for core recovery logic.

        Args:
            session_id: ID of the session to recover

        Returns:
            True if session was recovered successfully
        """
        if not self.memory_manager:
            logger.warning("Memory manager not enabled, cannot recover session")
            return False

        # Delegate to LifecycleManager for recovery
        success = self._lifecycle_manager.recover_session(
            session_id=session_id,
            memory_manager=self.memory_manager,
        )

        if success:
            # Update orchestrator-specific session tracking
            self._memory_session_id = session_id
            logger.info(f"Recovered session {session_id[:8]}... ")
        else:
            logger.warning(f"Failed to recover session {session_id}")

        return success

    def get_memory_context(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get token-aware context messages from memory manager.

        Uses intelligent pruning to select the most relevant messages
        within token budget. Useful for long conversations.

        Args:
            max_tokens: Override max tokens for this retrieval. If None,
                       uses the default token limit from memory manager.

        Returns:
            List of messages in provider format, where each message is a
            dictionary containing 'role' and 'content' keys.

        Note:
            If memory manager is not enabled or no session is active,
            falls back to returning messages from in-memory conversation.
            If memory retrieval fails, logs a warning and uses in-memory
            messages as fallback.
        """
        if not self.memory_manager or not self._memory_session_id:
            # Fall back to in-memory conversation
            return [msg.model_dump() for msg in self.messages]

        try:
            return self.memory_manager.get_context_messages(
                session_id=self._memory_session_id,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.warning(f"Failed to get memory context: {e}, using in-memory")
            return [msg.model_dump() for msg in self.messages]

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for the current memory session.

        Delegates to memory_manager.get_session_stats() when available.

        Returns:
            Dictionary with session statistics including:
            - enabled: Whether memory manager is active
            - session_id: Current session ID
            - message_count: Number of messages
            - total_tokens: Total token usage
            - max_tokens: Maximum token budget
            - available_tokens: Remaining token budget
            - Other session metadata
        """
        if not self.memory_manager or not self._memory_session_id:
            return {
                "enabled": False,
                "session_id": None,
                "message_count": len(self.messages),
            }

        try:
            # Delegate to memory_manager.get_session_stats()
            stats = self.memory_manager.get_session_stats(self._memory_session_id)
            if not stats:
                return {
                    "enabled": True,
                    "session_id": self._memory_session_id,
                    "error": "Session not found",
                }

            # Add orchestrator-specific fields
            stats["enabled"] = True
            return stats
        except Exception as e:
            logger.warning(f"Failed to get session stats: {e}")
            return {"enabled": True, "session_id": self._memory_session_id, "error": str(e)}

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
        provider: str,
        model: Optional[str] = None,
        on_switch: Optional[Any] = None,
    ) -> None:
        """Switch to a different provider/model (protocol method).

        This is an async protocol method that delegates to the sync implementation.

        Args:
            provider: Target provider name
            model: Optional specific model
            on_switch: Optional callback(provider, model) after switch
        """
        # Import ProviderSwitcherState for state update
        from victor.agent.provider.switcher import ProviderSwitcherState

        # Get provider settings from settings if not provided
        provider_kwargs = self.settings.get_provider_settings(provider)

        # Create new provider instance
        # Note: ProviderRegistry is already imported at module level (line 210)
        # This ensures patches work correctly
        new_provider = ProviderRegistry.create(provider, **provider_kwargs)

        # Determine model to use
        new_model = model or self.model

        # Store old state for analytics
        old_provider_name = self.provider_name
        old_model = self.model

        # Update ProviderManager internal state directly
        # Update both _current_state (legacy) and ProviderSwitcher state
        self._provider_manager._current_state = ProviderState(
            provider=new_provider,
            provider_name=provider.lower(),
            model=new_model,
        )

        # Also update ProviderSwitcher's state (the source of truth)
        switcher_state = self._provider_manager._provider_switcher.get_current_state()
        old_switch_count = switcher_state.switch_count if switcher_state else 0

        self._provider_manager._provider_switcher._current_state = ProviderSwitcherState(
            provider=new_provider,
            provider_name=provider.lower(),
            model=new_model,
            switch_count=old_switch_count + 1,
        )

        self._provider_manager.initialize_tool_adapter()

        # Sync local attributes from ProviderManager
        self.provider = self._provider_manager.provider
        self.model = self._provider_manager.model
        self.provider_name = self._provider_manager.provider_name
        self.tool_adapter = self._provider_manager.tool_adapter
        self.tool_calling_caps = self._provider_manager.capabilities

        # Apply post-switch hooks (exploration settings, prompt builder, system prompt, tool budget)
        self._apply_post_switch_hooks(respect_sticky_budget=True)

        # Log the switch
        logger.info(
            f"Switched provider: {old_provider_name}:{old_model} -> "
            f"{self.provider_name}:{new_model} "
            f"(native_tools={self.tool_calling_caps.native_tool_calls})"
        )

        # Log analytics event
        self.usage_logger.log_event(
            "provider_switch",
            {
                "old_provider": old_provider_name,
                "old_model": old_model,
                "new_provider": self.provider_name,
                "new_model": new_model,
                "native_tool_calls": self.tool_calling_caps.native_tool_calls,
            },
        )

        # Update metrics collector with new model info
        self._metrics_collector.update_model_info(new_model, self.provider_name)

        if on_switch:
            on_switch(self.provider_name, self.model)

    # --- ToolsProtocol ---

    def get_available_tools(self) -> Set[str]:
        """Get all registered tool names (protocol method).

        Returns:
            Set of tool names available in registry
        """
        if self.tools:
            return set(self.tools.list_tools())
        return set()

    def _build_tool_access_context(self) -> "ToolAccessContext":
        """Build ToolAccessContext for unified access control checks.

        Consolidates context construction used by get_enabled_tools() and
        is_tool_enabled() to ensure consistent access control decisions.

        Returns:
            ToolAccessContext with session tools and current mode
        """
        from victor.agent.protocols import ToolAccessContext

        return ToolAccessContext(
            session_enabled_tools=getattr(self, "_enabled_tools", None),
            current_mode=self.current_mode_name if self.mode_controller else None,
        )

    def get_enabled_tools(self) -> Set[str]:
        """Get currently enabled tool names (protocol method).

        Uses ToolAccessController if available for unified access control.
        In BUILD mode (allow_all_tools=True), expands to all available tools
        regardless of vertical restrictions.

        Returns:
            Set of enabled tool names for this session
        """
        # Use ToolAccessController if available (new unified approach)
        if hasattr(self, "_tool_access_controller") and self._tool_access_controller:
            context = self._build_tool_access_context()
            return self._tool_access_controller.get_allowed_tools(context)

        # Legacy fallback: Check mode controller for BUILD mode (allows all tools)
        # Uses ModeAwareMixin for consistent access
        mc = self.mode_controller
        if mc is not None:
            config = mc.config
            # BUILD mode expands to all available tools (minus disallowed)
            if config.allow_all_tools:
                all_tools = self.get_available_tools()
                # Remove any explicitly disallowed tools
                enabled = all_tools - config.disallowed_tools
                return enabled

        # Check for framework-set tools (vertical filtering)
        if hasattr(self, "_enabled_tools") and self._enabled_tools:
            return self._enabled_tools

        # Fall back to all available tools
        return self.get_available_tools()

    def set_enabled_tools(self, tools: Set[str], tiered_config: Any = None) -> None:
        """Set which tools are enabled for this session (protocol method).

        This is the single source of truth for enabled tools configuration.
        It updates all relevant components: tool selector, vertical context,
        tool access controller, and tiered configuration.

        Args:
            tools: Set of tool names to enable
            tiered_config: Optional TieredToolConfig to propagate for stage filtering.
                          If None, will attempt to retrieve from active vertical.
        """
        self._enabled_tools = tools

        # Apply to vertical context and tool access controller
        self._apply_vertical_tools(tools)

        # Propagate to tool_selector for selection-time filtering
        if hasattr(self, "tool_selector") and self.tool_selector:
            self.tool_selector.set_enabled_tools(tools)
            logger.info(f"Enabled tools filter propagated to selector: {sorted(tools)}")

            # Also propagate TieredToolConfig for stage-aware filtering
            if tiered_config is None:
                # Try to get tiered config from active vertical
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

        Uses ToolAccessController for unified layered access control:
        Safety (L0) > Mode (L1) > Session (L2) > Vertical (L3) > Stage (L4) > Intent (L5)

        Falls back to legacy logic if controller not available.

        Args:
            tool_name: Name of tool to check

        Returns:
            True if tool is enabled
        """
        # Use ToolAccessController if available (new unified approach)
        if hasattr(self, "_tool_access_controller") and self._tool_access_controller:
            context = self._build_tool_access_context()
            decision = self._tool_access_controller.check_access(tool_name, context)
            return decision.allowed

        # Legacy fallback: Check mode controller restrictions first
        # Uses ModeAwareMixin for consistent access
        mc = self.mode_controller
        if mc is not None:
            config = mc.config

            # If tool is in mode's disallowed list, it's disabled regardless of other settings
            if tool_name in config.disallowed_tools:
                return False

            # If mode allows all tools, it's enabled (unless in disallowed list above)
            if config.allow_all_tools:
                # Check if tool exists in registry
                if self.tools and tool_name in self.tools.list_tools():
                    return True

        # Fall back to session/vertical restrictions
        enabled = self.get_enabled_tools()
        return tool_name in enabled

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
