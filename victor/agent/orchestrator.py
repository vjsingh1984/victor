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

# Standard library imports
import ast
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

# Consolidated imports - all imports organized by category
# See victor/agent/orchestrator_imports.py for the complete import structure
from victor.agent.orchestrator_imports import (
    # Third-party
    Console,
    # Coordinators
    CheckpointCoordinator,
    ToolAccessConfigCoordinator,
    EvaluationCoordinator,
    MetricsCoordinator,
    ResponseCoordinator,
    StateCoordinator,
    StateScope,
    WorkflowCoordinator,
    ConversationCoordinator,
    SearchCoordinator,
    TeamCoordinator,
    # Agent components
    ArgumentNormalizer,
    NormalizationStrategy,
    MessageHistory,
    ConversationStore,
    MessageRole,
    # Mixins
    ModeAwareMixin,
    CapabilityRegistryMixin,
    ComponentAccessorMixin,
    StateDelegationMixin,
    LegacyAPIMixin,
    # DI container
    ensure_bootstrapped,
    get_service_optional,
    MetricsServiceProtocol,
    LoggerServiceProtocol,
    # Protocols
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
    RecoveryHandlerProtocol,
    # Configuration and enums
    get_provider_limits,
    Settings,
    ToolCallingMatrix,
    ConversationStateMachine,
    ConversationStage,
    ActionIntent,
    INTENT_BLOCKED_TOOLS,
    get_task_type_hint,
    SystemPromptBuilder,
    SearchRoute,
    SearchType,
    TaskComplexity,
    DEFAULT_BUDGETS,
    StreamMetrics,
    MetricsCollectorConfig,
    TrackerTaskType,
    UnifiedTaskTracker,
    extract_prompt_requirements,
    # Decomposed components
    ConversationConfig,
    ContextMetrics,
    CompactionStrategy,
    TruncationStrategy,
    create_context_compactor,
    calculate_parallel_read_budget,
    ContextManager,
    ContextManagerConfig,
    create_context_manager,
    ContinuationStrategy,
    ExtractedToolCall,
    get_rl_coordinator,
    AnalyticsConfig,
    create_sequence_tracker,
    SessionStateManager,
    create_session_state_manager,
    # Phase 1 extractions
    ConfigurationManager,
    create_configuration_manager,
    MemoryManager,
    SessionRecoveryManager,
    create_memory_manager,
    create_session_recovery_manager,
    # Recovery
    RecoveryOutcome,
    FailureType,
    RecoveryAction,
    VerticalContext,
    create_vertical_context,
    VerticalIntegrationAdapter,
    create_recovery_integration,
    OrchestratorRecoveryAction,
    # Tool output formatting
    ToolOutputFormatterConfig,
    FormattingContext,
    create_tool_output_formatter,
    # Pipeline
    ToolPipelineConfig,
    ToolCallResult,
    StreamingControllerConfig,
    StreamingSession,
    get_task_analyzer,
    ToolRegistrarConfig,
    ProviderManagerConfig,
    ProviderState,
    # Observability
    ObservabilityIntegration,
    IntegrationConfig,
    # Tool execution
    get_critical_tools,
    ToolCallParseResult,
    ValidationMode,
    calculate_max_context_chars,
    infer_git_operation,
    get_tool_status_message,
    OrchestratorFactory,
    create_parallel_executor,
    ToolFailureContext,
    create_response_completer,
    # Analytics and logging
    UsageLogger,
    StreamingMetricsCollector,
    # Storage and caching
    ToolCache,
    # Providers
    BaseProvider,
    CompletionResponse,
    Message,
    StreamChunk,
    ToolDefinition,
    ProviderRegistry,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ToolNotFoundError,
    ToolValidationError,
    # Tools
    CostTier,
    ToolRegistry,
    CodeSandbox,
    get_mcp_tool_definitions,
    ToolPluginRegistry,
    SemanticToolSelector,
    ToolNames,
    TOOL_ALIASES,
    get_alias_resolver,
    get_progressive_registry,
    # Workflows and embeddings
    IntentClassifier,
    IntentType,
    WorkflowRegistry,
    register_builtin_workflows,
    # Project context
    ProjectContext,
    # Streaming submodule
    CoordinatorConfig,
    ContinuationHandler,
    ContinuationResult,
    IntentClassificationHandler,
    IntentClassificationResult,
    IterationCoordinator,
    ProgressMetrics,
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
    # Adapters
    CoordinatorAdapter,
    IntelligentPipelineAdapter,
    ResultConverters,
    OrchestratorProtocolAdapter,
    create_orchestrator_protocol_adapter,
    # Logger
    logger,
)

# Type-checking only imports
if TYPE_CHECKING:
    from victor.agent.orchestrator_integration import OrchestratorIntegration
    from victor.agent.recovery_coordinator import StreamingRecoveryCoordinator
    from victor.agent.chunk_generator import ChunkGenerator
    from victor.agent.tool_planner import ToolPlanner
    from victor.agent.task_coordinator import TaskCoordinator
    from victor.agent.protocols import ToolAccessContext
    from victor.evaluation.protocol import TokenUsage
    from victor.agent.subagents import SubAgentOrchestrator

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
    from victor.agent.orchestrator_factory import OrchestratorComponents
    from victor.agent.provider_manager import ProviderManager
    from victor.agent.provider_coordinator import ProviderCoordinator
    from victor.agent.tool_selection import ToolSelector
    from victor.agent.tool_executor import ToolExecutor
    from victor.agent.safety import SafetyChecker
    from victor.agent.auto_commit import AutoCommitter
    from victor.agent.conversation_manager import ConversationManager

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


class AgentOrchestrator(
    ComponentAccessorMixin,
    StateDelegationMixin,
    ModeAwareMixin,
    CapabilityRegistryMixin,
    LegacyAPIMixin,  # Phase 4: Backward compatibility mixin
):
    """Facade orchestrator for agent interactions, tool execution, and provider communication.

    **Delegation Patterns:**
    - Component access via ComponentAccessorMixin (e.g., `self.tool_registry`, `self.provider_manager`)
    - State properties via StateDelegationMixin (e.g., `self.messages`, `self.stage`, `self.current_mode`)
    - Mode control via ModeAwareMixin (e.g., `self.is_build_mode`, `self.exploration_multiplier`)
    - Capability discovery via CapabilityRegistryMixin (type-safe protocol conformance)
    - Legacy API support via LegacyAPIMixin (deprecated methods with warnings)

    **Key Public APIs:**
    - `chat()`, `stream_chat()`: Main entry points for conversations
    - `switch_provider()`: Change provider/model mid-conversation
    - `reset_session()`: Clear conversation state
    - Component properties (e.g., `tool_selector`, `conversation_controller`): Access extracted components

    **Deprecated APIs:**
    - Methods from LegacyAPIMixin issue deprecation warnings
    - Will be removed in v0.7.0
    - See migration guide in docs/migration/legacy_api_migration.md
    """

    # State delegations for StateDelegationMixin
    _state_delegations = {
        "messages": ("_conversation_coordinator", "messages"),
        "current_mode": ("_mode_coordinator", "current_mode"),
        "stage": ("_conversation_coordinator", "stage"),
        "context_size": ("_context_manager", "context_size"),
        "max_context_size": ("_context_manager", "max_context_size"),
    }

    @staticmethod
    def _calculate_max_context_chars(
        settings: "Settings",
        provider: "BaseProvider",
        model: str,
    ) -> int:
        """Calculate maximum context size in characters for a model."""
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
    ) -> None:
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

        # Type annotations for attributes set by factory.initialize_orchestrator()
        # These are declared here to satisfy mypy without changing the factory pattern
        self._settings: Settings
        self._provider: BaseProvider
        self._model: str
        self._temperature: float
        self._max_tokens: int
        self._console: Optional[Console]
        self._tool_selection: Optional[Dict[str, Any]]
        self._thinking: bool
        self._provider_name: Optional[str]
        self._profile_name: Optional[str]

        # Factory-created components (initialized by factory.initialize_orchestrator)
        self._metrics_collector: "MetricsCollector"
        self._progress_metrics: "ProgressMetrics"
        self._evaluation_coordinator: "EvaluationCoordinator"
        self._task_analyzer: "TaskAnalyzer"
        self._conversation_controller: "ConversationController"
        self._tool_pipeline: "ToolPipeline"
        self._streaming_controller: "StreamingController"
        self._streaming_handler: "StreamingChatHandler"  # Created by create_streaming_handler()
        self._provider_manager: "ProviderManager"
        self._context_compactor: "ContextCompactor"
        self._tool_output_formatter: Optional["ToolOutputFormatter"]
        self._sequence_tracker: "ToolSequenceTracker"
        self._recovery_handler: Optional["RecoveryHandler"]
        self._observability: Optional["ObservabilityIntegration"]
        self._response_sanitizer: "ResponseSanitizer"
        self._search_router: "SearchRouter"
        self._complexity_classifier: "ComplexityClassifier"
        self._usage_analytics: "UsageAnalytics"
        self._conversation_manager: "ConversationManager"
        self._tool_selector: "ToolSelector"
        self._tool_executor: "ToolExecutor"
        self._safety_checker: "SafetyChecker"
        self._auto_committer: "AutoCommitter"

        # Additional runtime state (properties that delegate to _session_state)
        # NOTE: _read_files_session, _required_files, _required_outputs are @properties
        self._all_files_read_nudge_sent: bool

        # Additional factory-created components that may not always be present
        self._recovery_integration: Optional["OrchestratorRecoveryIntegration"]
        self._recovery_coordinator: Optional["StreamingRecoveryCoordinator"]
        self._chunk_generator: Optional["ChunkGenerator"]
        self._tool_planner: Optional["ToolPlanner"]
        self._task_coordinator: Optional["TaskCoordinator"]
        self._code_correction_middleware: Any
        self._state_coordinator: Optional["StateCoordinator"]
        self._session_state: Optional["SessionStateManager"]
        self._vertical_context: "VerticalContext"

        # Lazy-initialized attributes
        self._intelligent_pipeline_enabled: bool = False
        self._intelligent_integration: Optional["OrchestratorIntegration"] = None
        self._subagent_orchestration_enabled: bool = False
        self._subagent_orchestrator: Optional["SubAgentOrchestrator"] = None
        self._mode_workflow_team_coordinator: Optional[Any] = None
        self._team_coordinator: Optional[Any] = None
        self._embedding_preload_task: Optional[asyncio.Task[Any]] = None
        self._continuation_prompts: Dict[str, str] = {}
        self._asking_input_prompts: Dict[str, str] = {}

        # Create factory for component initialization (composition root)
        self._factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            console=console,
            provider_name=provider_name,
            profile_name=profile_name,
            tool_selection=tool_selection,
            thinking=thinking,
        )
        self._factory.initialize_orchestrator(self)

    def _on_tool_start_callback(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Callback when tool execution starts."""
        iteration = self._tool_pipeline.calls_used if hasattr(self, "_tool_pipeline") else 0
        self._metrics_collector.on_tool_start(tool_name, arguments, iteration)

        # Emit observability event for tool start
        if hasattr(self, "_observability") and self._observability:
            tool_id = f"tool-{iteration}"
            self._observability.on_tool_start(tool_name, arguments, tool_id)

    def _on_tool_complete_callback(self, result: ToolCallResult) -> None:
        """Callback when tool execution completes."""
        self._metrics_collector.on_tool_complete(result)

        # Track progress metrics for intelligent continuation decisions
        if result.success:
            self._progress_metrics.record_tool_used(result.tool_name)

            # Track read files for progress-aware continuation
            if result.tool_name in ("read", "Read", "read_file"):
                # Extract file path from arguments
                if result.arguments:
                    file_path = result.arguments.get("path") or result.arguments.get("file_path")
                    if file_path:
                        self._progress_metrics.record_file_read(file_path)
                        logger.debug(f"ProgressMetrics: Recorded file read: {file_path}")

        # Emit tool complete event
        from victor.core.events import get_observability_bus

        bus = get_observability_bus()
        try:
            loop = asyncio.get_running_loop()
            _ = loop.create_task(  # Intentionally discard - fire and forget
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

        # Track read files for task completion detection (legacy support)
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
                        from victor.core.events.emit_helper import emit_event_sync

                        event_bus = get_observability_bus()
                        emit_event_sync(
                            event_bus,
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
        """Callback when streaming session completes. Records metrics and sends RL reward signal."""
        self._metrics_collector.on_streaming_session_complete(session)

        # End UsageAnalytics session
        if hasattr(self, "_usage_analytics") and self._usage_analytics:
            self._usage_analytics.end_session()

        # Send RL reward signal for Q-learning model selection
        self._send_rl_reward_signal(session)

    def _send_rl_reward_signal(self, session: StreamingSession) -> None:
        """Send reward signal to RL model selector for Q-value updates."""
        self._coordinator_adapter.send_rl_reward_signal(session)

    def _extract_required_files_from_prompt(self, user_message: str) -> List[str]:
        """Extract file paths mentioned in user prompt for task completion tracking."""
        return self._task_analyzer.extract_required_files_from_prompt(user_message)

    def _extract_required_outputs_from_prompt(self, user_message: str) -> List[str]:
        """Extract output requirements from user prompt."""
        return self._task_analyzer.extract_required_outputs_from_prompt(user_message)

    # =====================================================================
    # Component accessors for external use
    # =====================================================================

    @property
    def streaming_handler(self) -> StreamingChatHandler:
        """Streaming chat handler for testable streaming loop logic."""
        return self._streaming_handler

    @property
    def observability(self) -> Optional[ObservabilityIntegration]:
        """Observability integration for event bus access."""
        return getattr(self, "_observability", None)

    @observability.setter
    def observability(self, value: Optional[ObservabilityIntegration]) -> None:
        """Set observability integration (for FrameworkShim injection)."""
        self._observability = value

    @property
    def memory_manager(self) -> Optional["MemoryManager"]:
        """Memory manager wrapper for session persistence and recovery."""
        return self._memory_manager_wrapper

    @memory_manager.setter
    def memory_manager(self, value: Optional["MemoryManager"]) -> None:
        """Set memory manager (for test injection)."""
        self._memory_manager_wrapper = value if value is not None else None
        if (
            hasattr(self, "_session_recovery_manager")
            and self._session_recovery_manager is not None
        ):
            self._session_recovery_manager._memory_manager = value if value is not None else None

    @property
    def tool_calling_caps(self) -> Any:
        """Tool calling capabilities with adapter configuration."""
        return self._tool_calling_caps_internal

    @tool_calling_caps.setter
    def tool_calling_caps(self, value: Any) -> None:
        """Set tool calling capabilities (for test injection)."""
        self._tool_calling_caps_internal = value
        if hasattr(self, "_prompt_coordinator") and self._prompt_coordinator is not None:
            self._prompt_coordinator.set_tool_calling_caps(value)

    @property
    def usage_analytics(self) -> Optional["UsageAnalytics"]:
        """Usage analytics for data-driven optimization."""
        return self._evaluation_coordinator.usage_analytics

    @property
    def sequence_tracker(self) -> Optional["ToolSequenceTracker"]:
        """Tool sequence tracker for intelligent next-tool suggestions."""
        return self._sequence_tracker

    @property
    def recovery_handler(self) -> Optional["RecoveryHandler"]:
        """Recovery handler for model failure recovery."""
        return self._recovery_handler

    @property
    def chunk_generator(self) -> Optional["ChunkGenerator"]:
        """Chunk generator for streaming output."""
        return self._chunk_generator

    @property
    def tool_planner(self) -> Optional["ToolPlanner"]:
        """Tool planner for planning operations."""
        return self._tool_planner

    @property
    def task_coordinator(self) -> Optional["TaskCoordinator"]:
        """Task coordinator for task preparation and guidance."""
        return self._task_coordinator

    @property
    def code_correction_middleware(self) -> Optional[Any]:
        """Code correction middleware for automatic validation/fixing."""
        return self._code_correction_middleware

    # =====================================================================
    # State Coordinator - Unified state management
    # =====================================================================

    @property
    def state_coordinator(self) -> Optional[StateCoordinator]:
        """State coordinator for unified state management (SessionStateManager, ConversationStateMachine, checkpoints)."""
        return self._state_coordinator

    # =====================================================================
    # Session state delegation properties (TD-002)
    # These delegate to SessionStateManager for consolidated state tracking
    # =====================================================================

    @property
    def session_state(self) -> Optional[SessionStateManager]:
        """Session state manager for consolidated state tracking."""
        return self._session_state

    # =====================================================================
    # Phase 5 Coordinators (Orchestrator Integration)
    # =====================================================================

    @property
    def tool_retry_coordinator(self) -> Optional[Any]:
        """Tool retry coordinator for centralized retry logic."""
        if hasattr(self, "_workflow_optimization") and self._workflow_optimization:
            return self._workflow_optimization.get("tool_retry_coordinator")
        return None

    @property
    def memory_coordinator(self) -> Optional[Any]:
        """Memory coordinator for centralized memory management."""
        if hasattr(self, "_workflow_optimization") and self._workflow_optimization:
            return self._workflow_optimization.get("memory_coordinator")
        return None

    @property
    def tool_capability_coordinator(self) -> Optional[Any]:
        """Tool capability coordinator for capability checks."""
        if hasattr(self, "_workflow_optimization") and self._workflow_optimization:
            return self._workflow_optimization.get("tool_capability_coordinator")
        return None

    # =====================================================================
    # Session state delegation properties (TD-002)
    # These delegate to SessionStateManager for consolidated state tracking
    # =====================================================================

    @property
    def tool_calls_used(self) -> int:
        """Number of tool calls used in this session."""
        if self._session_state is None:
            return 0
        return self._session_state.tool_calls_used

    @tool_calls_used.setter
    def tool_calls_used(self, value: int) -> None:
        """Set tool calls used (for backward compatibility)."""
        if self._session_state is not None:
            self._session_state.execution_state.tool_calls_used = value

    @property
    def observed_files(self) -> Set[str]:
        """Files observed/read during this session."""
        if self._session_state is None:
            return set()
        return self._session_state.execution_state.observed_files

    @observed_files.setter
    def observed_files(self, value: Set[str]) -> None:
        """Set observed files (for checkpoint restore)."""
        if self._session_state is not None:
            self._session_state.execution_state.observed_files = set(value) if value else set()

    @property
    def executed_tools(self) -> List[str]:
        """Executed tool names in order."""
        if self._session_state is None:
            return []
        return self._session_state.execution_state.executed_tools

    @executed_tools.setter
    def executed_tools(self, value: List[str]) -> None:
        """Set executed tools (for checkpoint restore)."""
        if self._session_state is not None:
            self._session_state.execution_state.executed_tools = list(value) if value else []

    @property
    def failed_tool_signatures(self) -> Set[Tuple[str, str]]:
        """Failed tool call signatures (tool_name, args_hash)."""
        if self._session_state is None:
            return set()
        return self._session_state.execution_state.failed_tool_signatures

    @failed_tool_signatures.setter
    def failed_tool_signatures(self, value: Set[Tuple[str, str]]) -> None:
        """Set failed tool signatures (for checkpoint restore)."""
        if self._session_state is not None:
            self._session_state.execution_state.failed_tool_signatures = set(value) if value else set()

    @property
    def _tool_capability_warned(self) -> bool:
        """Whether we've warned about tool capability limitations."""
        if self._session_state is None:
            return False
        return self._session_state.session_flags.tool_capability_warned

    @_tool_capability_warned.setter
    def _tool_capability_warned(self, value: bool) -> None:
        """Set tool capability warning flag."""
        if self._session_state is not None:
            self._session_state.session_flags.tool_capability_warned = value

    @property
    def _read_files_session(self) -> Set[str]:
        """Files read during this session for task completion detection."""
        if self._session_state is None:
            return set()
        return self._session_state.execution_state.read_files_session

    @property
    def _required_files(self) -> List[str]:
        """Required files extracted from user prompts."""
        if self._session_state is None:
            return []
        return self._session_state.execution_state.required_files

    @_required_files.setter
    def _required_files(self, value: List[str]) -> None:
        """Set required files list."""
        if self._session_state is not None:
            self._session_state.execution_state.required_files = list(value)

    @property
    def _required_outputs(self) -> List[str]:
        """Required outputs extracted from user prompts."""
        if self._session_state is None:
            return []
        return self._session_state.execution_state.required_outputs

    @_required_outputs.setter
    def _required_outputs(self, value: List[str]) -> None:
        """Set required outputs list."""
        if self._session_state is not None:
            self._session_state.execution_state.required_outputs = list(value)

    @property
    def _all_files_read_nudge_sent(self) -> bool:
        """Whether we've sent a nudge that all required files are read."""
        if self._session_state is None:
            return False
        return self._session_state.session_flags.all_files_read_nudge_sent

    @_all_files_read_nudge_sent.setter
    def _all_files_read_nudge_sent(self, value: bool) -> None:
        """Set all files read nudge flag."""
        if self._session_state is not None:
            self._session_state.session_flags.all_files_read_nudge_sent = value

    @property
    def _cumulative_token_usage(self) -> Dict[str, int]:
        """Cumulative token usage for evaluation/benchmarking."""
        if self._session_state is None:
            return {}
        return self._session_state.get_token_usage()

    @_cumulative_token_usage.setter
    def _cumulative_token_usage(self, value: Dict[str, int]) -> None:
        """Set cumulative token usage (for backward compatibility)."""
        if self._session_state is not None:
            self._session_state.execution_state.token_usage = dict(value)

    @property
    def intelligent_integration(self) -> Optional["OrchestratorIntegration"]:
        """Intelligent pipeline integration (RL-based mode learning, quality scoring, embeddings)."""
        if not self._intelligent_pipeline_enabled:
            return None

        if self._intelligent_integration is None:
            try:
                from victor.agent.orchestrator_integration import OrchestratorIntegration

                # Synchronous initialization (async version available via enhance_orchestrator)
                from victor.agent.intelligent_pipeline import IntelligentAgentPipeline

                # Determine project root for grounding verification
                # Context file is at .victor/init.md, so project root is grandparent
                from victor.config.settings import get_project_paths, VICTOR_DIR_NAME

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
                self._intelligent_integration = OrchestratorIntegration(  # type: ignore[arg-type]
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
        """Sub-agent orchestrator for spawning specialized sub-agents and parallel task delegation."""
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
        """Checkpoint manager for time-travel debugging (snapshots, restore, forking)."""
        return self._checkpoint_coordinator.checkpoint_manager

    @property
    def vertical_context(self) -> VerticalContext:
        """Vertical context for unified vertical state access (config, tools, middleware, mode)."""
        return self._vertical_context

    # =====================================================================
    # Adapter Properties (Phase 3 Refactoring)
    # =====================================================================

    @property
    def _coordinator_adapter(self) -> CoordinatorAdapter:
        """Adapter for coordinator operations (checkpointing, RL rewards)."""
        if not hasattr(self, "_coordinator_adapter_instance"):
            self._coordinator_adapter_instance = CoordinatorAdapter(
                state_coordinator=self._state_coordinator,
                evaluation_coordinator=self._evaluation_coordinator,
                conversation_controller=self._conversation_controller,
            )
        return self._coordinator_adapter_instance

    @property
    def _intelligent_pipeline_adapter(self) -> IntelligentPipelineAdapter:
        """Adapter for intelligent pipeline integration."""
        if not hasattr(self, "_intelligent_pipeline_adapter_instance"):
            self._intelligent_pipeline_adapter_instance = IntelligentPipelineAdapter(
                intelligent_integration=None,  # Set on access
                validation_coordinator=self._validation_coordinator,
            )
        return self._intelligent_pipeline_adapter_instance

    @property
    def _protocol_adapter(self) -> OrchestratorProtocolAdapter:
        """Adapter for protocol implementations (extracted from orchestrator)."""
        if not hasattr(self, "_protocol_adapter_instance"):
            if self._state_coordinator is None:
                raise RuntimeError("State coordinator is required for protocol adapter")
            self._protocol_adapter_instance = create_orchestrator_protocol_adapter(
                orchestrator=self,
                state_coordinator=self._state_coordinator,
                provider_coordinator=self._provider_coordinator,
                tools=self.tools,
                conversation=self.conversation,
                mode_controller=getattr(self, "mode_controller", None),
                unified_tracker=getattr(self, "unified_tracker", None),
                tool_selector=getattr(self, "tool_selector", None),
                tool_access_config_coordinator=self._tool_access_config_coordinator,
                vertical_context=self._vertical_context,
                conversation_state=self.conversation_state,
                prompt_builder=getattr(self, "prompt_builder", None),
            )
        return self._protocol_adapter_instance

    def __getattr__(self, name: str) -> Any:
        """Dynamic delegation to coordinators.

        This method provides automatic delegation to coordinators for methods
        that were extracted during refactoring, maintaining backward compatibility
        without requiring explicit wrapper methods.

        Delegates to:
        - _chat_coordinator for chat/streaming methods
        - _state_coordinator for state management methods

        Args:
            name: Attribute name being accessed

        Returns:
            The attribute from the appropriate coordinator

        Raises:
            AttributeError: If attribute not found in any coordinator
        """
        # Prevent infinite recursion during __init__
        if name.startswith("_") and not name.startswith("__"):
            # For private attributes, raise AttributeError normally
            pass

        # Delegate to ChatCoordinator for chat/streaming methods
        if "_chat_coordinator" in self.__dict__:
            chat_coordinator = self._chat_coordinator
            if hasattr(chat_coordinator, name):
                return getattr(chat_coordinator, name)

        # Delegate to StateCoordinator for state methods
        if "_state_coordinator" in self.__dict__:
            state_coordinator = self._state_coordinator
            if hasattr(state_coordinator, name):
                return getattr(state_coordinator, name)

        # Not found in any coordinator
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # =====================================================================
    # Mode-Workflow-Team Coordination
    # =====================================================================

    @property
    def coordination(self) -> Any:
        """Mode-workflow-team coordinator for intelligent task execution suggestions."""
        if self._mode_workflow_team_coordinator is None:
            self._mode_workflow_team_coordinator = (
                self._factory.create_mode_workflow_team_coordinator(self._vertical_context)
            )
            logger.debug("ModeWorkflowTeamCoordinator initialized on first access")

            # Initialize TeamCoordinator now that mode_workflow_team_coordinator is available
            if self._team_coordinator is None:
                self._team_coordinator = TeamCoordinator(
                    orchestrator=self,
                    mode_coordinator=self._mode_coordinator,
                    mode_workflow_team_coordinator=self._mode_workflow_team_coordinator,
                )
                logger.debug("TeamCoordinator initialized after ModeWorkflowTeamCoordinator")

        return self._mode_workflow_team_coordinator

    def get_team_suggestions(
        self,
        task_type: str,
        complexity: str,
    ) -> Any:
        """Get team and workflow suggestions for a task.

        Args:
            task_type: Classified task type (e.g., "feature", "bugfix", "refactor")
            complexity: Complexity level (e.g., "low", "medium", "high", "extreme")

        Returns:
            CoordinationSuggestion with team and workflow recommendations
        """
        # Ensure team coordinator is initialized
        if self._team_coordinator is None:
            # Access mode_workflow_team_coordinator to trigger lazy initialization
            _ = self.mode_workflow_team_coordinator

        return self._team_coordinator.get_team_suggestions(task_type, complexity)

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
        3. ConfigurationManager for centralized access

        Args:
            config: TieredToolConfig from the active vertical
        """
        # Delegate to ConfigurationManager for centralized management
        self._configuration_manager.set_tiered_tool_config(
            config=config,
            vertical_context=self._vertical_context,
            tool_access_controller=self._tool_access_controller,
        )

        logger.debug("Tiered tool config set via ConfigurationManager")

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
        # Update session ID in coordinator
        self._checkpoint_coordinator.update_session_id(self._memory_session_id)

        # Delegate to coordinator
        return await self._checkpoint_coordinator.save_checkpoint(  # type: ignore[no-any-return]
            description=description,
            tags=tags,
        )

    async def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore conversation state from a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to restore

        Returns:
            True if restored successfully, False otherwise
        """
        # Update session ID in coordinator
        self._checkpoint_coordinator.update_session_id(self._memory_session_id)

        # Delegate to coordinator
        return await self._checkpoint_coordinator.restore_checkpoint(checkpoint_id)  # type: ignore[no-any-return]

    async def maybe_auto_checkpoint(self) -> Optional[str]:
        """Trigger auto-checkpoint if interval threshold is met.

        This should be called after tool executions to maintain
        automatic checkpointing based on configured interval.

        Returns:
            Checkpoint ID if auto-checkpoint was created, None otherwise
        """
        # Update session ID in coordinator
        self._checkpoint_coordinator.update_session_id(self._memory_session_id)

        # Delegate to coordinator
        return await self._checkpoint_coordinator.maybe_auto_checkpoint()  # type: ignore[no-any-return]

    def _get_checkpoint_state(self) -> dict[str, Any]:
        """Build a dictionary representing current conversation state for checkpointing."""
        return self._coordinator_adapter.get_checkpoint_state()

    def _apply_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Apply a checkpoint state to restore the orchestrator."""
        self._coordinator_adapter.apply_checkpoint_state(state)

    async def _prepare_intelligent_request(
        self, task: str, task_type: str
    ) -> Optional[Dict[str, Any]]:
        """Pre-request hook for intelligent pipeline integration."""
        # Update adapter with current intelligent integration
        adapter = self._intelligent_pipeline_adapter
        adapter._intelligent_integration = self.intelligent_integration

        return await adapter.prepare_intelligent_request(
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

        Args:
            response: The model's response content
            query: Original user query
            tool_calls: Number of tool calls made so far
            task_type: Task type for context

        Returns:
            Dictionary with quality/grounding scores, or None if pipeline disabled
        """
        # Update adapter with current intelligent integration
        adapter = self._intelligent_pipeline_adapter
        adapter._intelligent_integration = self.intelligent_integration

        return await adapter.validate_intelligent_response(
            response=response,
            query=query,
            tool_calls=tool_calls,
            task_type=task_type,
        )

    async def _record_intelligent_outcome(
        self,
        success: bool,
        quality_score: float = 0.5,
        user_satisfied: bool = True,
        completed: bool = True,
    ) -> None:
        """Record outcome for Q-learning feedback (async event-driven)."""
        # Update adapter with current intelligent integration
        adapter = self._intelligent_pipeline_adapter
        adapter._intelligent_integration = self.intelligent_integration

        await adapter.record_intelligent_outcome(
            success=success,
            quality_score=quality_score,
            user_satisfied=user_satisfied,
            completed=completed,
        )

    def _should_continue_intelligent(self) -> tuple[bool, str]:
        """Check if processing should continue using learned behaviors.

        Returns:
            Tuple of (should_continue, reason)
        """
        # Update adapter with current intelligent integration
        adapter = self._intelligent_pipeline_adapter
        adapter._intelligent_integration = self.intelligent_integration

        return adapter.should_continue_intelligent()

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
        """Apply middleware from vertical extensions (called by FrameworkShim)."""
        self._vertical_integration_adapter.apply_middleware(middleware_list)

    def apply_vertical_safety_patterns(self, patterns: List[Any]) -> None:
        """Apply safety patterns from vertical extensions (called by FrameworkShim)."""
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
        # Ensure team coordinator is initialized
        if self._team_coordinator is None:
            # Access mode_workflow_team_coordinator to trigger lazy initialization
            _ = self.mode_workflow_team_coordinator

        self._team_coordinator.set_team_specs(specs)

    def get_team_specs(self) -> Dict[str, Any]:
        """Retrieve team specifications.

        Implements VerticalStorageProtocol.get_team_specs().
        Returns the dictionary of team specs configured by vertical integration.

        Returns:
            Dictionary of team specs, or empty dict if not set
        """
        # Ensure team coordinator is initialized
        if self._team_coordinator is None:
            # Access mode_workflow_team_coordinator to trigger lazy initialization
            _ = self.mode_workflow_team_coordinator

        return self._team_coordinator.get_team_specs()  # type: ignore[no-any-return]

    def _get_model_context_window(self) -> int:
        """Get context window size for the current model.


        Returns:
            Context window size in tokens
        """
        # Delegate to ContextManager if available
        if hasattr(self, "_context_manager") and self._context_manager is not None:
            return self._context_manager.get_model_context_window()  # type: ignore[no-any-return]

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


        Returns:
            Maximum context size in characters
        """
        return self._validation_coordinator.get_max_context_chars()  # type: ignore[no-any-return]

    def _check_context_overflow(self, max_context_chars: int = 200000) -> bool:
        """Check if context is at risk of overflow.


        Args:
            max_context_chars: Maximum allowed context size in chars

        Returns:
            True if context is dangerously large
        """
        return self._validation_coordinator.check_context_overflow(max_context_chars).is_overflow  # type: ignore[no-any-return]

    def get_context_metrics(self) -> ContextMetrics:
        """Get detailed context metrics.

        Returns:
            ContextMetrics with size and overflow information
        """
        return self._context_manager.get_context_metrics()  # type: ignore[no-any-return]

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
            from victor.agent.conversation_embedding_store import ConversationEmbeddingStore
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
            # self.memory_manager.set_embedding_store(self._conversation_embedding_store)  # Method removed

            # Also set the embedding service for fallback
            # self.memory_manager.set_embedding_service(embedding_service)  # Method removed

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
        return self._metrics_coordinator.finalize_stream_metrics(usage_data)  # type: ignore[no-any-return]

    def get_last_stream_metrics(self) -> Optional[StreamMetrics]:
        """Get metrics from the last streaming session.

        Returns:
            StreamMetrics from the last session or None if no metrics available
        """
        return self._metrics_coordinator.get_last_stream_metrics()  # type: ignore[no-any-return]

    def get_streaming_metrics_summary(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive streaming metrics summary.

        Returns:
            Dictionary with aggregated metrics or None if metrics disabled.
        """
        return self._metrics_coordinator.get_streaming_metrics_summary()  # type: ignore[no-any-return]

    def get_streaming_metrics_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent streaming metrics history.

        Args:
            limit: Maximum number of recent metrics to return

        Returns:
            List of recent metrics dictionaries
        """
        return self._metrics_coordinator.get_streaming_metrics_history(limit)  # type: ignore[no-any-return]

    def get_session_cost_summary(self) -> Dict[str, Any]:
        """Get session cost summary."""
        return self._metrics_coordinator.get_session_cost_summary()  # type: ignore[no-any-return]

    def get_session_cost_formatted(self) -> str:
        """Get formatted session cost string (e.g., "$0.0123")."""
        return self._metrics_coordinator.get_session_cost_formatted()  # type: ignore[no-any-return]

    def export_session_costs(self, path: str, format: str = "json") -> None:
        """Export session costs to file."""
        self._metrics_coordinator.export_session_costs(path, format)

    async def _preload_embeddings(self) -> None:
        """Preload tool embeddings in background to avoid blocking first query.

        This is called asynchronously during initialization if semantic tool
        selection is enabled. Errors are logged but don't crash the app.

        Note: The ToolSelector owns the _embeddings_initialized state to avoid
        DRY violations and consistency issues.
        """
        # Check if tool_selector needs initialization (has initialize_tool_embeddings method)
        if not hasattr(self.tool_selector, "initialize_tool_embeddings"):
            return
        # ToolSelector owns the initialization state
        if (
            hasattr(self.tool_selector, "_embeddings_initialized")
            and self.tool_selector._embeddings_initialized
        ):
            return

        try:
            logger.info("Starting background embedding preload...")
            # Initialize the actual tool_selector (not the deprecated semantic_selector)
            await self.tool_selector.initialize_tool_embeddings(self.tools)
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
    ) -> Optional[asyncio.Task[Any]]:
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
        # Check if tool_selector needs embedding initialization
        if (
            not hasattr(self.tool_selector, "initialize_tool_embeddings")
            or self._embedding_preload_task
        ):
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
        return self._search_coordinator.route_search_query(query)  # type: ignore[no-any-return]

    def get_recommended_search_tool(self, query: str) -> str:
        """Get the recommended search tool name for a query.

        Convenience method that returns just the tool name.

        Args:
            query: The search query

        Returns:
            Tool name: "code_search", "semantic_code_search", or "both"
        """
        return self._search_coordinator.get_recommended_search_tool(query)  # type: ignore[no-any-return]

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
                from victor.framework.rl.base import RLOutcome

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
        return self._metrics_coordinator.get_tool_usage_stats(  # type: ignore[no-any-return]
            conversation_state_summary=self.conversation_state.get_state_summary()
        )

    def get_token_usage(self) -> "TokenUsage":
        """Get cumulative token usage for evaluation tracking.

        Returns cumulative tokens used across all stream_chat calls.
        Used by VictorAgentAdapter for benchmark token tracking.

        Returns:
            TokenUsage dataclass with input/output/total token counts
        """
        return self._metrics_coordinator.get_token_usage()  # type: ignore[no-any-return]

    def reset_token_usage(self) -> None:
        """Reset cumulative token usage tracking.

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
        # StateCoordinator returns stage name, convert to enum
        if self._state_coordinator is None:
            return ConversationStage.INITIAL
        stage_name = self._state_coordinator.get_stage()
        if stage_name:
            return ConversationStage[stage_name]
        return ConversationStage.INITIAL

    def get_stage_recommended_tools(self) -> Set[str]:
        """Get tools recommended for the current conversation stage.

        Returns:
            Set of tool names recommended for current stage
        """
        if self._state_coordinator is None:
            return set()
        return self._state_coordinator.get_stage_tools()

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all integrated optimization components.

        Creates AnalyticsCoordinator inline for unified status reporting.

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
        from victor.agent.coordinators.analytics_coordinator import AnalyticsCoordinator

        coordinator = AnalyticsCoordinator(
            exporters=[],  # No exporters needed for status reporting
        )
        return coordinator.get_optimization_status(
            context_compactor=self._context_compactor,
            usage_analytics=self._usage_analytics,
            sequence_tracker=self._sequence_tracker,
            code_correction_middleware=self._code_correction_middleware,
            safety_checker=self._safety_checker,
            auto_committer=self._auto_committer,
            search_router=self.search_router,
        )

    async def flush_analytics(self) -> Dict[str, bool]:
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
        # Flush evaluation coordinator analytics (async)
        results = await self._evaluation_coordinator.flush_analytics()

        # Flush tool cache if available
        tool_cache = getattr(self, "tool_cache", None)
        if tool_cache:
            try:
                # ToolCache doesn't have flush(), but we can mark it as synced
                # The cache uses persistent storage via TieredCache
                results["tool_cache"] = True
            except Exception as e:
                logger.error(f"Failed to flush tool cache: {e}")
                results["tool_cache"] = False

        logger.info(f"Analytics flush complete: {results}")
        return results

    async def start_health_monitoring(self) -> bool:
        """Start background provider health monitoring.

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

        Returns:
            Dictionary with provider health information
        """
        return await self._provider_coordinator.get_health()  # type: ignore[no-any-return]

    async def graceful_shutdown(self) -> Dict[str, bool]:
        """Perform graceful shutdown of all orchestrator components.

        Flushes analytics, stops health monitoring, and cleans up resources.
        Call this before application exit.

        Returns:
            Dictionary with shutdown status for each component
        """
        # Delegate to LifecycleManager for graceful shutdown
        return await self._lifecycle_manager.graceful_shutdown()  # type: ignore[no-any-return]

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
                return asyncio.run(self._provider_coordinator.switch_provider(provider_name, model))  # type: ignore[no-any-return]
            else:
                return asyncio.run(self._provider_coordinator.switch_provider(provider_name, model))  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(f"Failed to switch provider to {provider_name}: {e}")
            return False

    def switch_model(self, model: str) -> bool:
        """Switch to a different model on the current provider.

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
                return asyncio.run(self._provider_coordinator.switch_model(model))  # type: ignore[no-any-return]
            else:
                return asyncio.run(self._provider_coordinator.switch_model(model))  # type: ignore[no-any-return]
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

        Delegates to ProviderLifecycleManager for:
        - Apply model-specific exploration settings to unified tracker
        - Get prompt contributors from vertical extensions
        - Create prompt builder with new capabilities
        - Calculate tool budget

        Args:
            respect_sticky_budget: If True, don't reset tool budget when user
                override is sticky (used by switch_provider).

        Note:
            This method assumes self.model, self.provider_name, self.tool_adapter,
            and self._tool_calling_caps_internal are already updated before calling.

        Raises:
            RuntimeError: If ProviderLifecycleManager is not initialized.
        """
        # Get the provider lifecycle manager (set by ProviderLayerBuilder)
        manager = getattr(self, "_provider_lifecycle_manager", None)
        if manager is None:
            raise RuntimeError(
                "ProviderLifecycleManager not initialized. "
                "Ensure ProviderLayerBuilder runs before calling this method."
            )

        # Delegate to ProviderLifecycleManager
        manager.apply_exploration_settings(self.unified_tracker, self._tool_calling_caps_internal)
        prompt_contributors = manager.get_prompt_contributors()
        self.prompt_builder = manager.create_prompt_builder(
            provider_name=self.provider_name,
            model=self.model,
            tool_adapter=self.tool_adapter,
            capabilities=self._tool_calling_caps_internal,
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

        # Update tool budget
        if respect_sticky_budget and manager.should_respect_sticky_budget(self.unified_tracker):
            logger.debug("Skipping tool budget reset on provider switch (sticky user override)")
            return

        self.tool_budget = manager.calculate_tool_budget(
            self._tool_calling_caps_internal, self.settings
        )

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
        result = self._response_coordinator._parse_tool_calls_with_adapter(content, raw_tool_calls)

        # Log any warnings
        for warning in result.warnings:
            logger.warning(f"Tool call parse warning: {warning}")

        # Log parse method for debugging
        if result.tool_calls:
            logger.debug(
                f"Parsed {len(result.tool_calls)} tool calls via {result.parse_method} "
                f"(confidence={result.confidence})"
            )

        return result  # type: ignore[no-any-return]

    def _build_system_prompt_with_adapter(self) -> str:
        """Build system prompt using the tool calling adapter.
        Includes dynamic parallel read budget based on model's context window.

        Returns:
            Built system prompt with dynamic budget hint if applicable
        """
        return self._prompt_coordinator.build_system_prompt_with_adapter(  # type: ignore[no-any-return]
            prompt_builder=self.prompt_builder,
            get_model_context_window=self._get_model_context_window,
            model=self.model,
            session_id=getattr(self, "_session_id", ""),
            provider_name=self.provider_name,
        )

    def _get_thinking_disabled_prompt(self, base_prompt: str) -> str:
        """Prefix a prompt with the thinking disable prefix if supported.

        Args:
            base_prompt: The base prompt text

        Returns:
            Prompt with thinking disable prefix if available, otherwise base_prompt
        """
        return self._prompt_coordinator.get_thinking_disabled_prompt(base_prompt)  # type: ignore[no-any-return]

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
        # Delegate to ModeCoordinator for mode-aware shell variant resolution
        return self._mode_coordinator.resolve_shell_variant(tool_name)  # type: ignore[no-any-return]

    def _log_tool_call(self, name: str, kwargs: dict[str, Any]) -> None:
        """A hook that logs information before a tool is called."""
        # Move verbose argument logging to debug level - not user-facing
        logger.debug(f"Tool call: {name} with args: {kwargs}")

    def _classify_task_keywords(self, user_message: str) -> Dict[str, Any]:
        """Classify task type based on keywords in the user message.


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

        # Get task complexity from task analyzer if available
        task_complexity = None
        try:
            if hasattr(self, "_task_analyzer"):
                # Get the last analyzed complexity from the task analyzer
                # Note: This requires the task analyzer to have analyzed the current task
                task_complexity = getattr(self._task_analyzer, "_last_complexity", None)
                if task_complexity:
                    task_complexity = (
                        task_complexity.value
                        if hasattr(task_complexity, "value")
                        else str(task_complexity)
                    )
        except Exception as e:
            logger.debug(f"Unable to get task complexity: {e}")

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
            # Progress Metrics for intelligent continuation decisions
            progress_metrics=self._progress_metrics,
            # Complexity-based continuation thresholds
            task_complexity=task_complexity,
        )

    def _format_tool_output(self, tool_name: str, args: Dict[str, Any], output: Any) -> str:
        """Format tool output with clear boundaries to prevent model hallucination.


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
        return self._tool_output_formatter.format_tool_output(  # type: ignore[union-attr]
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
        count = self._workflow_coordinator.register_default_workflows()
        logger.debug(f"Dynamically registered {count} workflows")

    def _register_default_tools(self) -> None:
        """Dynamically discovers and registers all tools.


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
        return supported  # type: ignore[no-any-return]

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        self._conversation_coordinator.add_message(role, content)

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
        return await self._chat_coordinator.chat(user_message)  # type: ignore[no-any-return]

    # NOTE: Dead code removed - chat logic delegated to ChatCoordinator
    # Lines 3592-4849 removed (~1258 lines of unused chat-related helper methods)
    # ChatCoordinator now handles all streaming logic, iteration management, and recovery

    async def stream_chat(self, user_message: str) -> AsyncIterator[StreamChunk]:
        """Stream a chat response (public entrypoint).

        This method wraps the implementation to make phased refactors safer.

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


        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            context: Execution context

        Returns:
            Tuple of (result, success, error_message or None)
        """
        # Use coordinator if available
        if self.tool_retry_coordinator:
            result = await self.tool_retry_coordinator.execute_tool(
                tool_name=tool_name,
                tool_args=tool_args,
                context=context,
            )
            if result.success:
                return result.result, True, None
            else:
                return result.result, False, result.error_message

        # Fallback to inline implementation for backward compatibility
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
                    f"[yellow]{self._presentation.icon('warning', with_color=False)} Skipping invalid tool call (not a dict): {tool_call}[/]"
                )
                continue

            tool_name = tool_call.get("name")
            if not tool_name:
                self.console.print(
                    f"[yellow]{self._presentation.icon('warning', with_color=False)} Skipping tool call without name: {tool_call}[/]"
                )
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
                    f"[yellow]{self._presentation.icon('warning', with_color=False)} Skipping invalid/hallucinated tool name: {tool_name}[/]"
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
                    f"[yellow]{self._presentation.icon('warning', with_color=False)} Skipping unknown or disabled tool: {tool_name} (resolved: {canonical_tool_name})[/]"
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
                    f"[yellow]{self._presentation.icon('warning', with_color=False)} Tool budget reached ({self.tool_budget}); skipping remaining tool calls.[/]"
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
                    f"[yellow]{self._presentation.icon('warning', with_color=False)} Skipping repeated failing call to '{tool_name}' with same arguments[/]"
                )
                continue

            # Log normalization if applied (for debugging and monitoring)
            if strategy != NormalizationStrategy.DIRECT:
                logger.warning(
                    f"Applied {strategy.value} normalization to {tool_name} arguments. "
                    f"Original: {tool_args} → Normalized: {normalized_args}"
                )
                gear_icon = self._presentation.icon("gear", with_color=False)
                self.console.print(
                    f"[yellow]{gear_icon} Normalized arguments via {strategy.value}[/]"
                )
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
                    f"[red]{self._presentation.icon('error', with_color=False)} Tool execution failed: {error_display}[/] [dim]({elapsed_ms:.0f}ms)[/dim]"
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
        self._conversation_coordinator.reset_conversation()

        # Reset orchestrator-specific state
        self._system_added = False
        self.tool_calls_used = 0
        self.failed_tool_signatures.clear()
        self.observed_files.clear()
        self.executed_tools.clear()
        self._consecutive_blocked_attempts = 0
        self._total_blocked_attempts = 0

    def request_cancellation(self) -> None:
        """Request cancellation of the current streaming operation.

        Safe to call even if not streaming. The cancellation will take
        effect at the next check point in the stream loop.
        """
        self._metrics_coordinator.request_cancellation()

    def is_streaming(self) -> bool:
        """Check if a streaming operation is currently in progress.

        Returns:
            True if streaming, False otherwise.
        """
        return self._metrics_coordinator.is_streaming()  # type: ignore[no-any-return]

    def _check_cancellation(self) -> bool:
        """Check if cancellation has been requested.


        Returns:
            True if cancelled, False otherwise.
        """
        return self._validation_coordinator.is_cancelled()  # type: ignore[no-any-return]

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

        Note:
            If memory manager is not enabled or retrieval fails, returns empty list.
        """
        # Return empty list if memory manager is not enabled
        if self._memory_manager_wrapper is None:
            return []
        # Delegate to MemoryManager wrapper with exception handling
        try:
            return self._memory_manager_wrapper.get_recent_sessions(limit=limit)
        except Exception as e:
            logger.warning(f"Failed to get recent sessions: {e}")
            return []

    def recover_session(self, session_id: str) -> bool:
        """Recover a previous conversation session.

        Args:
            session_id: ID of the session to recover

        Returns:
            True if session was recovered successfully
        """
        # Return False if memory manager is not enabled
        if self._memory_manager_wrapper is None:
            logger.warning("Cannot recover session: memory manager not enabled")
            return False

        # Update lifecycle manager reference if needed
        if self._session_recovery_manager._lifecycle_manager is None:
            self._session_recovery_manager._lifecycle_manager = self._lifecycle_manager

        # Delegate to SessionRecoveryManager with exception handling
        try:
            success = self._session_recovery_manager.recover_session(session_id)

            if success:
                # Update orchestrator-specific session tracking
                self._memory_session_id = session_id
                if self._memory_manager_wrapper is not None:
                    self._memory_manager_wrapper.session_id = session_id
                logger.info(f"Recovered session {session_id[:8]}... ")
            else:
                logger.warning(f"Failed to recover session {session_id}")

            return success  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning(f"Failed to recover session {session_id}: {e}")
            return False

    # =====================================================================
    # =====================================================================
    # Memory Management
    # =====================================================================

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
        # Fall back to in-memory messages if memory manager is not enabled or no session
        if self._memory_manager_wrapper is None or not self._memory_session_id:
            return [msg.model_dump() for msg in self.conversation.messages]
        # Delegate to MemoryManager wrapper with exception handling
        try:
            return self._memory_manager_wrapper.get_context(max_tokens=max_tokens)
        except Exception as e:
            logger.warning(f"Failed to get memory context, falling back to in-memory: {e}")
            return [msg.model_dump() for msg in self.conversation.messages]

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for the current memory session.

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
                "session_id": self._memory_session_id,
                "message_count": len(self.messages),
            }

        try:
            stats = self.memory_manager.get_session_stats(self._memory_session_id)
            # Handle empty stats (session not found)
            if not stats or not any(k in stats for k in ("message_count", "total_tokens", "found")):
                return {
                    "enabled": True,
                    "session_id": self._memory_session_id,
                    "message_count": len(self.messages),
                    "found": False,
                    "error": "Session not found",
                }
            # Add enabled and session_id fields if not present (for backward compatibility)
            if "enabled" not in stats:
                stats["enabled"] = True
            if "session_id" not in stats:
                stats["session_id"] = self._memory_session_id
            return stats
        except Exception as e:
            logger.warning(f"Failed to get session stats: {e}")
            return {
                "enabled": True,
                "session_id": self._memory_session_id,
                "message_count": len(self.messages),
                "error": str(e),
            }

    async def shutdown(self) -> None:
        """Clean up resources and shutdown gracefully.

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
    #
    # Protocol implementations are delegated to OrchestratorProtocolAdapter
    # to reduce orchestrator size and improve modularity.

    # --- ConversationStateProtocol ---

    def get_stage(self) -> "ConversationStage":
        """Get current conversation stage (protocol method).

        Returns:
            Current ConversationStage enum value
        """
        return self._protocol_adapter.get_stage()

    def get_tool_calls_count(self) -> int:
        """Get total tool calls made (protocol method).

        Returns:
            Non-negative count of tool calls in this session
        """
        return self._protocol_adapter.get_tool_calls_count()

    def get_tool_budget(self) -> int:
        """Get tool call budget (protocol method).

        Returns:
            Maximum allowed tool calls
        """
        return self._protocol_adapter.get_tool_budget()

    def get_observed_files(self) -> Set[str]:
        """Get files observed/read during conversation (protocol method).

        Returns:
            Set of absolute file paths
        """
        return self._protocol_adapter.get_observed_files()

    def get_modified_files(self) -> Set[str]:
        """Get files modified during conversation (protocol method).

        Returns:
            Set of absolute file paths
        """
        return self._protocol_adapter.get_modified_files()

    def get_iteration_count(self) -> int:
        """Get current agent loop iteration count (protocol method).

        Returns:
            Non-negative iteration count
        """
        return self._protocol_adapter.get_iteration_count()

    def get_max_iterations(self) -> int:
        """Get maximum allowed iterations (protocol method).

        Returns:
            Max iteration limit
        """
        return self._protocol_adapter.get_max_iterations()

    # --- ProviderProtocol ---

    @property
    def current_provider(self) -> str:
        """Get current provider name (protocol property).

        Returns:
            Provider identifier (e.g., "anthropic", "openai")
        """
        return self._protocol_adapter.current_provider

    @property
    def current_model(self) -> str:
        """Get current model name (protocol property).

        Returns:
            Model identifier
        """
        return self._protocol_adapter.current_model

    async def switch_provider(
        self,
        provider: str,
        model: Optional[str] = None,
        on_switch: Optional[Any] = None,
    ) -> bool:
        """Switch to a different provider/model (protocol method).

        Args:
            provider: Target provider name
            model: Optional specific model
            on_switch: Optional callback(provider, model) after switch

        Returns:
            True if switch was successful, False otherwise

        Raises:
            ProviderNotFoundError: If provider not found
        """
        return await self._protocol_adapter.switch_provider(provider, model, on_switch)

    # --- ToolsProtocol ---

    def get_available_tools(self) -> Set[str]:
        """Get all registered tool names (protocol method).

        Returns:
            Set of tool names available in registry
        """
        return self._protocol_adapter.get_available_tools()

    def get_enabled_tools(self) -> Set[str]:
        """Get currently enabled tool names (protocol method).

        Returns:
            Set of enabled tool names for this session
        """
        return self._protocol_adapter.get_enabled_tools()

    def set_enabled_tools(self, tools: Set[str], tiered_config: Any = None) -> None:
        """Set which tools are enabled for this session (protocol method).

        Args:
            tools: Set of tool names to enable
            tiered_config: Optional TieredToolConfig to propagate for stage filtering.
        """
        self._protocol_adapter.set_enabled_tools(tools, tiered_config)

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled (protocol method).

        Args:
            tool_name: Name of tool to check

        Returns:
            True if tool is enabled
        """
        return self._protocol_adapter.is_tool_enabled(tool_name)

    # --- SystemPromptProtocol ---

    def get_system_prompt(self) -> str:
        """Get current system prompt (protocol method).

        Returns:
            Complete system prompt string
        """
        return self._protocol_adapter.get_system_prompt()

    def set_system_prompt(self, prompt: str) -> None:
        """Set custom system prompt (protocol method).

        Args:
            prompt: New system prompt (replaces existing)
        """
        self._protocol_adapter.set_system_prompt(prompt)

    def append_to_system_prompt(self, content: str) -> None:
        """Append content to system prompt (protocol method).

        Args:
            content: Content to append
        """
        self._protocol_adapter.append_to_system_prompt(content)

    # --- MessagesProtocol ---

    def get_message_count(self) -> int:
        """Get message count (protocol method).

        Returns:
            Number of messages in conversation
        """
        return self._protocol_adapter.get_message_count()

    # --- Health Check Methods ---

    def check_tool_selector_health(self) -> Dict[str, Any]:
        """Check if tool selector is properly initialized.

        This health check prevents the critical bug where SemanticToolSelector
        was never initialized, blocking ALL chat functionality.

        Returns:
            Dictionary with health status
        """
        return self._protocol_adapter.check_tool_selector_health()

    async def ensure_tool_selector_initialized(self) -> None:
        """Ensure tool selector is initialized before first use.

        This is a health check recovery mechanism that prevents the critical bug
        where SemanticToolSelector was never initialized.

        Should be called before chat() or stream_chat() if health check fails.

        Raises:
            RuntimeError: If initialization fails
        """
        await self._protocol_adapter.ensure_tool_selector_initialized()

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
    def _from_components(cls, components: "OrchestratorComponents") -> "AgentOrchestrator":
        """Create orchestrator from pre-built components."""
        orchestrator = cls.__new__(cls)
        if components.attributes:
            orchestrator.__dict__.update(components.attributes)
        else:
            orchestrator.provider = components.provider.provider
            orchestrator.model = components.provider.model
            orchestrator.provider_name = components.provider.provider_name
            orchestrator.tool_adapter = components.provider.tool_adapter
            orchestrator._tool_calling_caps_internal = components.provider.tool_calling_caps

            orchestrator.sanitizer = components.services.sanitizer
            orchestrator.prompt_builder = components.services.prompt_builder
            orchestrator.project_context = components.services.project_context
            orchestrator.task_classifier = components.services.complexity_classifier
            orchestrator.intent_detector = components.services.action_authorizer
            orchestrator.search_router = components.services.search_router

            orchestrator._conversation_controller = components.conversation.conversation_controller
            orchestrator.memory_manager = components.conversation.memory_manager
            orchestrator._memory_session_id = components.conversation.memory_session_id
            orchestrator.conversation_state = components.conversation.conversation_state

            orchestrator.tools = components.tools.tool_registry
            orchestrator.tool_registrar = components.tools.tool_registrar
            orchestrator.tool_executor = components.tools.tool_executor
            orchestrator.tool_cache = components.tools.tool_cache
            orchestrator.tool_graph = components.tools.tool_graph
            orchestrator.plugin_manager = components.tools.plugin_manager

            orchestrator._streaming_controller = components.streaming.streaming_controller
            orchestrator._streaming_handler = components.streaming.streaming_handler
            orchestrator._metrics_collector = components.streaming.metrics_collector
            orchestrator.streaming_metrics_collector = (
                components.streaming.streaming_metrics_collector
            )

            orchestrator._usage_analytics = components.analytics.usage_analytics
            orchestrator._sequence_tracker = components.analytics.sequence_tracker
            orchestrator.unified_tracker = components.analytics.unified_tracker

            orchestrator._recovery_handler = components.recovery.recovery_handler
            orchestrator._recovery_integration = components.recovery.recovery_integration
            orchestrator._context_compactor = components.recovery.context_compactor

            orchestrator._observability = components.observability
            orchestrator._tool_output_formatter = components.tool_output_formatter
            orchestrator._workflow_optimization = components.workflow_optimization

            orchestrator._response_coordinator = components.coordinators.response_coordinator
            orchestrator._tool_access_config_coordinator = (
                components.coordinators.tool_access_config_coordinator
            )
            orchestrator._state_coordinator = components.coordinators.state_coordinator

        if not hasattr(orchestrator, "_capabilities"):
            orchestrator.__init_capability_registry__()

        return orchestrator

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
        # Use ConfigCoordinator for configuration loading (Phase 2 refactoring)
        from victor.agent.coordinators.config_coordinator import (
            ConfigCoordinator,
            ProfileConfigProvider,
        )

        # Create coordinator with profile provider
        profile_provider = ProfileConfigProvider(settings, profile_name)
        coordinator = ConfigCoordinator(providers=[profile_provider])

        # Load configuration from profile
        # Note: ProfileConfigProvider will raise ValueError if profile not found
        config = await coordinator.load_config(session_id="from_settings")

        # Validate that we got the expected config
        if not config or "provider" not in config:
            raise ValueError(f"Failed to load configuration for profile '{profile_name}'")

        # Create provider instance using coordinator
        provider = await coordinator.create_provider_from_config(config, settings)

        # Create orchestrator via factory (composition root)
        from victor.agent.orchestrator_factory import OrchestratorFactory

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model=config.get("model", ""),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 4096),
            tool_selection=config.get("tool_selection"),
            thinking=thinking,
            provider_name=config.get("provider"),
            profile_name=profile_name,
        )
        orchestrator = factory.create_orchestrator()

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
