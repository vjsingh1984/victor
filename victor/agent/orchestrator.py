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
- TaskAnalyzer: Unified facade for complexity/task/intent classification
- ToolSelector: Semantic and keyword-based tool selection
- ToolRegistrar: Tool registration, plugins, MCP integration (NEW)

Remaining Orchestrator Responsibilities:
- High-level chat flow coordination
- Configuration loading and validation
- Post-switch hooks (prompt rebuilding, tracker updates)

Recently Integrated:
- ProviderManager: Provider initialization, switching, health checks (NEW)

Note: Keep orchestrator as a thin facade. New logic should go into
appropriate extracted components, not added here.

Recent Refactoring (December 2025):
- Extracted ToolRegistrar from _register_default_tools, _initialize_plugins,
    _setup_mcp_integration, _plan_tools, and _goal_hints_for_message
- Added ProviderHealthChecker for proactive health monitoring
- Added ResilienceMetricsExporter for dashboard integration
- Added classification-aware tool selection in SemanticToolSelector
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
    from victor.agent.orchestrator_integration import OrchestratorIntegration

from victor.agent.argument_normalizer import ArgumentNormalizer, NormalizationStrategy
from victor.agent.message_history import MessageHistory
from victor.agent.conversation_memory import (
    ConversationStore,
    MessageRole,
)

# DI container bootstrap - ensures services are available
from victor.core.bootstrap import ensure_bootstrapped, get_service_optional
from victor.core.container import (
    MetricsServiceProtocol,
    LoggerServiceProtocol,
)

# Service protocols for DI resolution (Phase 10)
from victor.agent.protocols import (
    ResponseSanitizerProtocol,
    ComplexityClassifierProtocol,
    ActionAuthorizerProtocol,
    SearchRouterProtocol,
    ProjectContextProtocol,
    ArgumentNormalizerProtocol,
    ConversationStateMachineProtocol,
    TaskTrackerProtocol,
)

# Config loaders for externalized configuration
from victor.config.config_loaders import get_provider_limits
from victor.agent.conversation_embedding_store import (
    ConversationEmbeddingStore,
)
from victor.agent.conversation_state import ConversationStateMachine, ConversationStage
from victor.agent.debug_logger import get_debug_logger
from victor.agent.action_authorizer import (
    ActionAuthorizer,
    ActionIntent,
    INTENT_BLOCKED_TOOLS,
)
from victor.agent.prompt_builder import SystemPromptBuilder, get_task_type_hint
from victor.agent.response_sanitizer import ResponseSanitizer
from victor.agent.search_router import SearchRouter, SearchRoute, SearchType
from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity, DEFAULT_BUDGETS
from victor.agent.context_reminder import create_reminder_manager
from victor.agent.stream_handler import StreamMetrics
from victor.agent.metrics_collector import (
    MetricsCollector,
    MetricsCollectorConfig,
)
from victor.agent.unified_task_tracker import (
    UnifiedTaskTracker,
    TaskType,
)

# New decomposed components (facades for orchestrator responsibilities)
from victor.agent.conversation_controller import (
    ConversationController,
    ConversationConfig,
    ContextMetrics,
    CompactionStrategy,
)
from victor.agent.context_compactor import (
    ContextCompactor,
    TruncationStrategy,
    create_context_compactor,
    calculate_parallel_read_budget,
)
from victor.agent.usage_analytics import (
    UsageAnalytics,
    AnalyticsConfig,
)
from victor.agent.tool_sequence_tracker import (
    ToolSequenceTracker,
    create_sequence_tracker,
)
from victor.agent.recovery import (
    RecoveryHandler,
    RecoveryOutcome,
    FailureType,
    RecoveryAction,
)
from victor.agent.protocols import RecoveryHandlerProtocol
from victor.agent.orchestrator_recovery import (
    OrchestratorRecoveryIntegration,
    create_recovery_integration,
    RecoveryAction as OrchestratorRecoveryAction,
)
from victor.agent.tool_output_formatter import (
    ToolOutputFormatter,
    ToolOutputFormatterConfig,
    FormattingContext,
    create_tool_output_formatter,
)

# CodeCorrectionMiddleware imported lazily to avoid circular import
# (code_correction_middleware -> evaluation.correction -> evaluation.__init__ -> agent_adapter -> orchestrator)
from victor.agent.tool_pipeline import (
    ToolPipeline,
    ToolPipelineConfig,
    ToolCallResult,
)
from victor.agent.streaming_controller import (
    StreamingController,
    StreamingControllerConfig,
    StreamingSession,
)
from victor.agent.task_analyzer import TaskAnalyzer, get_task_analyzer
from victor.agent.tool_registrar import ToolRegistrar, ToolRegistrarConfig
from victor.agent.provider_manager import ProviderManager, ProviderManagerConfig, ProviderState

# Observability integration (EventBus, hooks, exporters)
from victor.observability.integration import ObservabilityIntegration

# Intelligent pipeline integration (lazy initialization to avoid circular imports)
# These enable RL-based mode learning, quality scoring, and prompt optimization
from victor.agent.orchestrator_integration import IntegrationConfig

from victor.agent.tool_selection import (
    get_critical_tools,
    ToolSelector,
)
from victor.agent.tool_calling import (
    ToolCallParseResult,
)
from victor.agent.tool_executor import ToolExecutor, ValidationMode
from victor.agent.safety import SafetyChecker, get_safety_checker
from victor.agent.auto_commit import AutoCommitter, get_auto_committer
from victor.agent.parallel_executor import (
    create_parallel_executor,
)
from victor.agent.response_completer import (
    ToolFailureContext,
    create_response_completer,
)
from victor.analytics.logger import UsageLogger
from victor.analytics.streaming_metrics import StreamingMetricsCollector
from victor.cache.tool_cache import ToolCache
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
from victor.tools.base import CostTier, ToolRegistry
from victor.tools.code_executor_tool import CodeExecutionManager
from victor.tools.dependency_graph import ToolDependencyGraph
from victor.tools.mcp_bridge_tool import configure_mcp_client, get_mcp_tool_definitions
from victor.tools.plugin_registry import ToolPluginRegistry
from victor.tools.semantic_selector import SemanticToolSelector
from victor.tools.tool_names import ToolNames, TOOL_ALIASES
from victor.embeddings.intent_classifier import IntentClassifier, IntentType
from victor.workflows.base import WorkflowRegistry
from victor.workflows.new_feature_workflow import NewFeatureWorkflow

# Streaming submodule - extracted for testability
from victor.agent.streaming import (
    StreamingChatContext,
    StreamingChatHandler,
    create_stream_context,
)

logger = logging.getLogger(__name__)

# Tools with progressive parameters - different params = progress, not a loop
# Format: tool_name -> list of param names that indicate progress
# NOTE: Includes both canonical short names and legacy names for LLM compatibility
PROGRESSIVE_TOOLS = {
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


class AgentOrchestrator:
    """Orchestrates agent interactions, tool execution, and provider communication."""

    # Legacy fallback context windows (in tokens)
    # Prefer config loader: get_provider_limits(provider, model).context_window
    PROVIDER_CONTEXT_WINDOWS = {
        "anthropic": 200000,  # Claude models
        "openai": 128000,  # GPT-4 Turbo
        "google": 1000000,  # Gemini 1.5 Pro
        "xai": 131072,  # Grok models
        "deepseek": 131072,  # DeepSeek V3
        "moonshot": 262144,  # Kimi K2
        "ollama": 32768,  # Local models (conservative)
        "lmstudio": 32768,  # Local models (conservative)
        "vllm": 32768,  # Local models (conservative)
    }

    @staticmethod
    def _calculate_max_context_chars(
        settings: "Settings",
        provider: "BaseProvider",
        model: str,
    ) -> int:
        """Calculate maximum context size in characters for a model.

        Uses externalized config from provider_context_limits.yaml with
        per-provider and per-model overrides. Falls back to hardcoded
        defaults if config unavailable.

        Args:
            settings: Application settings
            provider: LLM provider instance
            model: Model identifier

        Returns:
            Maximum context size in characters
        """
        # Check settings override first
        settings_max = getattr(settings, "max_context_chars", None)
        if settings_max and settings_max > 0:
            return settings_max

        # Try to get context window from provider API
        context_tokens = None
        if hasattr(provider, "get_context_window"):
            try:
                context_tokens = provider.get_context_window(model)
            except Exception:
                pass

        # Fall back to externalized config (YAML-based)
        if context_tokens is None:
            provider_name = getattr(provider, "name", "").lower()
            try:
                limits = get_provider_limits(provider_name, model)
                context_tokens = limits.context_window
                logger.debug(
                    f"Using YAML config for {provider_name}/{model}: {context_tokens} tokens"
                )
            except Exception:
                # Final fallback to hardcoded defaults
                context_tokens = AgentOrchestrator.PROVIDER_CONTEXT_WINDOWS.get(
                    provider_name, 100000
                )

        # Convert tokens to chars: ~3.5 chars per token with 80% safety margin
        try:
            # Try to coerce numeric-like values (including strings) to float
            token_count = float(context_tokens)
            max_chars = int(token_count * 3.5 * 0.8)
        except Exception:
            # Fallback to a conservative default if parsing fails (e.g., MagicMock)
            provider_name = getattr(provider, "name", "").lower()
            fallback_tokens = AgentOrchestrator.PROVIDER_CONTEXT_WINDOWS.get(provider_name, 100000)
            logger.debug(
                "Could not parse context token value %r; using fallback %d tokens",
                context_tokens,
                fallback_tokens,
            )
            max_chars = int(fallback_tokens * 3.5 * 0.8)

        # Log safely depending on whether context_tokens is numeric
        if isinstance(context_tokens, (int, float)):
            logger.info(
                f"Model context: {int(context_tokens):,} tokens -> {max_chars:,} chars limit"
            )
        else:
            logger.info(
                "Model context: %r tokens -> %s chars limit", context_tokens, f"{max_chars:,}"
            )

        return max_chars

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
        # Bootstrap DI container - ensures all services are available
        # This is idempotent and will only bootstrap if not already done
        self._container = ensure_bootstrapped(settings)

        self.settings = settings
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.console = console or Console()
        self.tool_selection = tool_selection or {}
        self.thinking = thinking
        self.tool_calling_models = getattr(settings, "tool_calling_models", {})
        self.tool_capabilities = ToolCallingMatrix(
            self.tool_calling_models,
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

        # Initialize ProviderManager for unified provider/model management
        # This encapsulates provider state, switching, and health monitoring
        self._provider_manager = ProviderManager(
            settings=settings,
            initial_provider=provider,
            initial_model=model,
            provider_name=provider_name,
            config=ProviderManagerConfig(
                enable_health_checks=getattr(settings, "provider_health_checks", True),
                auto_fallback=getattr(settings, "provider_auto_fallback", True),
                fallback_providers=getattr(settings, "fallback_providers", []),
            ),
        )

        # Initialize tool adapter through ProviderManager
        self._provider_manager.initialize_tool_adapter()

        # Expose provider attributes for backward compatibility
        # These delegate to ProviderManager
        self.provider = self._provider_manager.provider
        self.model = self._provider_manager.model
        self.provider_name = self._provider_manager.provider_name
        self.tool_adapter = self._provider_manager.tool_adapter
        self.tool_calling_caps = self._provider_manager.capabilities

        logger.info(
            f"Tool calling adapter: {self.tool_adapter.provider_name}, "
            f"native={self.tool_calling_caps.native_tool_calls}, "
            f"format={self.tool_calling_caps.tool_call_format.value}"
        )

        # Response sanitizer for cleaning model output (DI with fallback)
        self.sanitizer = (
            self._container.get_optional(ResponseSanitizerProtocol) or ResponseSanitizer()
        )

        # System prompt builder for provider-specific prompts
        self.prompt_builder = SystemPromptBuilder(
            provider_name=self.provider_name,
            model=model,
            tool_adapter=self.tool_adapter,
            capabilities=self.tool_calling_caps,
        )

        # Load project context from .victor/init.md (DI with fallback)
        self.project_context = self._container.get_optional(ProjectContextProtocol)
        if self.project_context is None:
            self.project_context = ProjectContext()
            self.project_context.load()

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
        # Use adapter's recommended budget, with settings override
        # Note: For analysis tasks, this may be increased dynamically in stream_chat()
        default_budget = self.tool_calling_caps.recommended_tool_budget
        # Ensure minimum budget of 50 for meaningful work
        default_budget = max(default_budget, 50)
        self.tool_budget = getattr(settings, "tool_call_budget", default_budget)
        self.tool_calls_used = 0

        # Gap implementations: Complexity classifier, action authorizer, search router (DI with fallback)
        self.task_classifier = (
            self._container.get_optional(ComplexityClassifierProtocol) or ComplexityClassifier()
        )
        self.intent_detector = (
            self._container.get_optional(ActionAuthorizerProtocol) or ActionAuthorizer()
        )
        self.search_router = self._container.get_optional(SearchRouterProtocol) or SearchRouter()

        self.observed_files: List[str] = []
        self.executed_tools: List[str] = []
        self.failed_tool_signatures: set[tuple[str, str]] = set()
        self._tool_capability_warned = False

        # Context reminder manager for intelligent system message injection
        # Reduces token waste by consolidating reminders and only injecting when context changes
        self.reminder_manager = create_reminder_manager(
            provider=self.provider_name,
            task_complexity="medium",  # Will be updated per-task
            tool_budget=self.tool_budget,
        )

        # Analytics - try DI container first for enhanced logger with rotation/encryption/PII scrubbing
        # Falls back to basic UsageLogger if container doesn't have enhanced version
        from victor.config.settings import get_project_paths
        from victor.core.bootstrap import UsageLoggerProtocol

        usage_logger_from_container = self._container.get_optional(UsageLoggerProtocol)
        if usage_logger_from_container is not None:
            self.usage_logger = usage_logger_from_container
            logger.debug("Using enhanced usage logger from DI container")
        else:
            analytics_log_file = get_project_paths().global_logs_dir / "usage.jsonl"
            self.usage_logger = UsageLogger(
                analytics_log_file, enabled=self.settings.analytics_enabled
            )
            logger.debug("Using basic usage logger (enhanced version not available)")

        self.usage_logger.log_event(
            "session_start", {"model": self.model, "provider": self.provider.__class__.__name__}
        )

        # Streaming metrics collector for performance monitoring
        self.streaming_metrics_collector: Optional[StreamingMetricsCollector] = None
        if getattr(settings, "streaming_metrics_enabled", True):
            history_size = getattr(settings, "streaming_metrics_history_size", 1000)
            self.streaming_metrics_collector = StreamingMetricsCollector(max_history=history_size)
            logger.info(f"StreamingMetricsCollector initialized (history: {history_size})")

        # Debug logger for incremental output and conversation tracking
        self.debug_logger = get_debug_logger()
        self.debug_logger.enabled = (
            getattr(settings, "debug_logging", False)
            or logging.getLogger("victor").level <= logging.DEBUG
        )

        # Cancellation support for streaming
        self._cancel_event: Optional[asyncio.Event] = None
        self._is_streaming = False

        # Background task tracking for graceful shutdown
        self._background_tasks: set[asyncio.Task] = set()

        # Metrics collection (extracted component)
        self._metrics_collector = MetricsCollector(
            config=MetricsCollectorConfig(
                model=model,
                provider=self.provider_name,
                analytics_enabled=bool(self.streaming_metrics_collector),
            ),
            usage_logger=self.usage_logger,
            debug_logger=self.debug_logger,
            streaming_metrics_collector=self.streaming_metrics_collector,
            tool_cost_lookup=lambda name: (
                self.tools.get_tool_cost(name) if hasattr(self, "tools") else CostTier.FREE
            ),
        )
        # Result cache for pure/idempotent tools
        self.tool_cache = None
        if getattr(self.settings, "tool_cache_enabled", True):
            from victor.cache.config import CacheConfig
            from victor.config.settings import get_project_paths

            # Allow explicit override of cache_dir, otherwise use centralized path
            cache_dir = getattr(self.settings, "tool_cache_dir", None)
            if cache_dir:
                cache_dir = Path(cache_dir).expanduser()
            else:
                cache_dir = get_project_paths().global_cache_dir
            self.tool_cache = ToolCache(
                ttl=getattr(self.settings, "tool_cache_ttl", 600),
                allowlist=getattr(self.settings, "tool_cache_allowlist", []),
                cache_config=CacheConfig(disk_path=cache_dir),
            )
        # Minimal dependency graph (used for planning search→read→analyze)
        self.tool_graph = ToolDependencyGraph()
        self._register_default_tool_dependencies()

        # Stateful managers
        self.code_manager = CodeExecutionManager()
        self.code_manager.start()

        # Workflow registry
        self.workflow_registry = WorkflowRegistry()
        self._register_default_workflows()

        # Conversation history (using MessageHistory for better encapsulation)
        self.conversation = MessageHistory(
            system_prompt=self._system_prompt,
            max_history_messages=getattr(settings, "max_conversation_history", 100),
        )

        # Persistent conversation memory with SQLite backing (optional)
        # Provides session recovery, token-aware pruning, and multi-turn context retention
        self.memory_manager: Optional[ConversationStore] = None
        self._memory_session_id: Optional[str] = None
        if getattr(settings, "conversation_memory_enabled", True):
            try:
                from victor.config.settings import get_project_paths

                paths = get_project_paths()
                # Ensure .victor directory exists
                paths.project_victor_dir.mkdir(parents=True, exist_ok=True)
                db_path = paths.conversation_db
                max_context = getattr(settings, "max_context_tokens", 100000)
                response_reserve = getattr(settings, "response_token_reserve", 4096)
                self.memory_manager = ConversationStore(
                    db_path=db_path,
                    max_context_tokens=max_context,
                    response_reserve=response_reserve,
                )
                # Create a session for this orchestrator instance
                # Pass ML-friendly metadata for RL/analytics aggregation
                project_path = str(paths.project_root)
                session = self.memory_manager.create_session(
                    project_path=project_path,
                    provider=self.provider_name,
                    model=model,
                    max_tokens=max_context,
                    profile=self._profile_name,
                    tool_capable=self.tool_calling_caps.native_tool_calls,
                )
                self._memory_session_id = session.session_id
                logger.info(
                    f"ConversationStore initialized. "
                    f"Session: {session.session_id[:8]}..., DB: {db_path}"
                )

                # Initialize LanceDB embedding store for efficient semantic retrieval
                # This stores message embeddings for O(log n) vector search
                if getattr(settings, "conversation_embeddings_enabled", True):
                    try:
                        self._init_conversation_embedding_store()
                    except Exception as embed_err:
                        logger.warning(
                            f"Failed to initialize ConversationEmbeddingStore: {embed_err}"
                        )
            except Exception as e:
                logger.warning(f"Failed to initialize ConversationStore: {e}")
                self.memory_manager = None

        # Conversation state machine for intelligent stage detection (DI with fallback)
        self.conversation_state = (
            self._container.get_optional(ConversationStateMachineProtocol)
            or ConversationStateMachine()
        )

        # Intent classifier for semantic continuation/completion detection
        # Uses embeddings instead of hardcoded phrase matching
        self.intent_classifier = IntentClassifier.get_instance()

        # Intelligent pipeline integration (lazy initialization)
        # Provides RL-based mode learning, quality scoring, prompt optimization
        self._intelligent_integration: Optional["OrchestratorIntegration"] = None
        self._intelligent_integration_config = IntegrationConfig(
            enable_resilient_calls=getattr(settings, "intelligent_pipeline_enabled", True),
            enable_quality_scoring=getattr(settings, "intelligent_quality_scoring", True),
            enable_mode_learning=getattr(settings, "intelligent_mode_learning", True),
            enable_prompt_optimization=getattr(settings, "intelligent_prompt_optimization", True),
            min_quality_threshold=getattr(settings, "intelligent_min_quality_threshold", 0.5),
            grounding_confidence_threshold=getattr(
                settings, "intelligent_grounding_threshold", 0.7
            ),
        )
        self._intelligent_pipeline_enabled = getattr(settings, "intelligent_pipeline_enabled", True)

        # Tool registry
        self.tools = ToolRegistry()

        # Initialize ToolRegistrar (extracted component for tool registration)
        # Encapsulates: dynamic tool discovery, plugins, MCP integration
        self.tool_registrar = ToolRegistrar(
            tools=self.tools,
            settings=settings,
            provider=provider,
            model=model,
            tool_graph=self.tool_graph,
            config=ToolRegistrarConfig(
                enable_plugins=getattr(settings, "plugin_enabled", True),
                enable_mcp=getattr(settings, "use_mcp_tools", False),
                enable_tool_graph=True,
                airgapped_mode=getattr(settings, "airgapped_mode", False),
                plugin_dirs=getattr(settings, "plugin_dirs", []),
                disabled_plugins=set(getattr(settings, "disabled_plugins", [])),
                plugin_packages=getattr(settings, "plugin_packages", []),
                max_workers=4,
                max_complexity=10,
            ),
        )
        self.tool_registrar.set_background_task_callback(self._create_background_task)

        # Synchronous registration (dynamic tools, configs)
        self._register_default_tools()  # Delegates to ToolRegistrar
        self._load_tool_configurations()  # Load tool enable/disable states from config
        self.tools.register_before_hook(self._log_tool_call)

        # Plugin system for extensible tools (delegates to ToolRegistrar)
        self.plugin_manager: Optional[ToolPluginRegistry] = None
        if getattr(settings, "plugin_enabled", True):
            self._initialize_plugins()

        # Argument normalizer for handling malformed tool arguments (DI with fallback)
        self.argument_normalizer = self._container.get_optional(ArgumentNormalizerProtocol)
        if self.argument_normalizer is None:
            provider_name = provider.__class__.__name__ if provider else "unknown"
            self.argument_normalizer = ArgumentNormalizer(provider_name=provider_name)

        # Tool executor for centralized tool execution with retry, caching, and metrics
        # Parse validation mode from settings
        validation_mode_str = getattr(settings, "tool_validation_mode", "lenient").lower()
        validation_mode_map = {
            "strict": ValidationMode.STRICT,
            "lenient": ValidationMode.LENIENT,
            "off": ValidationMode.OFF,
        }
        validation_mode = validation_mode_map.get(validation_mode_str, ValidationMode.LENIENT)

        # Initialize CodeCorrectionMiddleware for automatic code validation and fixing
        # This reduces failed tool calls by ~15-20% by fixing common syntax issues
        # Lazy import to avoid circular import chain
        code_correction_enabled = getattr(settings, "code_correction_enabled", True)
        code_correction_auto_fix = getattr(settings, "code_correction_auto_fix", True)
        code_correction_max_iterations = getattr(settings, "code_correction_max_iterations", 1)

        self._code_correction_middleware: Optional[Any] = None
        if code_correction_enabled:
            try:
                from victor.agent.code_correction_middleware import (
                    CodeCorrectionMiddleware,
                    CodeCorrectionConfig,
                )

                self._code_correction_middleware = CodeCorrectionMiddleware(
                    config=CodeCorrectionConfig(
                        enabled=True,
                        auto_fix=code_correction_auto_fix,
                        max_iterations=code_correction_max_iterations,
                    )
                )
                logger.debug(
                    f"CodeCorrectionMiddleware enabled (auto_fix={code_correction_auto_fix}, "
                    f"max_iterations={code_correction_max_iterations})"
                )
            except ImportError as e:
                logger.warning(f"CodeCorrectionMiddleware unavailable: {e}")
                code_correction_enabled = False

        # Initialize SafetyChecker for dangerous operation detection
        # Exposes via property for UI layer to set confirmation callback
        self._safety_checker = get_safety_checker()

        # Initialize AutoCommitter for AI-assisted code change commits
        # Provides conventional commits with co-authorship attribution
        auto_commit_enabled = getattr(settings, "auto_commit_enabled", False)
        self._auto_committer: Optional[AutoCommitter] = None
        if auto_commit_enabled:
            workspace_root = (
                Path(settings.workspace_root)
                if hasattr(settings, "workspace_root") and settings.workspace_root
                else None
            )
            self._auto_committer = AutoCommitter(
                workspace_root=workspace_root,
                auto_commit=True,
                use_conventional_commits=getattr(settings, "use_conventional_commits", True),
            )
            logger.debug("AutoCommitter enabled for AI-assisted commits")
        else:
            # Still create instance for manual use, just not auto-commit
            self._auto_committer = get_auto_committer()

        self.tool_executor = ToolExecutor(
            tool_registry=self.tools,
            argument_normalizer=self.argument_normalizer,
            tool_cache=self.tool_cache,
            max_retries=getattr(settings, "tool_retry_max_attempts", 3),
            retry_delay=getattr(settings, "tool_retry_base_delay", 1.0),
            validation_mode=validation_mode,
            safety_checker=self._safety_checker,
            code_correction_middleware=self._code_correction_middleware,
            enable_code_correction=code_correction_enabled,
        )

        # Parallel tool executor for concurrent independent tool calls
        parallel_enabled = getattr(settings, "parallel_tool_execution", True)
        max_concurrent = getattr(settings, "max_concurrent_tools", 5)
        self.parallel_executor = create_parallel_executor(
            tool_executor=self.tool_executor,
            max_concurrent=max_concurrent,
            enable=parallel_enabled,
        )

        # Response completer for ensuring complete responses after tool calls
        self.response_completer = create_response_completer(
            provider=self.provider,
            max_retries=getattr(settings, "response_completion_retries", 3),
            force_response=getattr(settings, "force_response_on_error", True),
        )

        # Semantic tool selector (optional, configured via settings)
        self.use_semantic_selection = getattr(settings, "use_semantic_tool_selection", False)
        self.semantic_selector: Optional[SemanticToolSelector] = None

        if self.use_semantic_selection:
            # Use settings-configured embedding provider and model
            # Default: sentence-transformers with unified_embedding_model (local, fast, air-gapped)
            # Both tool selection and codebase search use the same model for:
            # - 40% memory reduction (120MB vs 200MB)
            # - Better OS page cache utilization
            # - Improved CPU L2/L3 cache hit rates
            self.semantic_selector = SemanticToolSelector(
                embedding_model=settings.embedding_model,  # Defaults to unified_embedding_model
                embedding_provider=settings.embedding_provider,  # Defaults to sentence-transformers
                ollama_base_url=settings.ollama_base_url,
                cache_embeddings=True,
            )

        # Background embedding preload task (ToolSelector owns the _embeddings_initialized state)
        self._embedding_preload_task: Optional[asyncio.Task[None]] = None

        # Initialize UnifiedTaskTracker (DI with fallback)
        # This is the single source of truth for task progress, milestones, and loop detection
        self.unified_tracker = (
            self._container.get_optional(TaskTrackerProtocol) or UnifiedTaskTracker()
        )
        # Apply model-specific exploration settings
        self.unified_tracker.set_model_exploration_settings(
            exploration_multiplier=self.tool_calling_caps.exploration_multiplier,
            continuation_patience=self.tool_calling_caps.continuation_patience,
        )

        # Initialize unified ToolSelector (handles semantic + keyword selection)
        # Use unified_tracker as the single source of truth for task tracking
        self.tool_selector = ToolSelector(
            tools=self.tools,
            semantic_selector=self.semantic_selector,
            conversation_state=self.conversation_state,
            task_tracker=self.unified_tracker,
            model=self.model,
            provider_name=self.provider_name,
            tool_selection_config=self.tool_selection,
            fallback_max_tools=getattr(settings, "fallback_max_tools", 8),
            on_selection_recorded=self._record_tool_selection,
        )

        # =================================================================
        # NEW: Decomposed component facades for orchestrator responsibilities
        # These provide cleaner interfaces while maintaining backward compatibility
        # =================================================================

        # ConversationController: Manages message history and conversation state
        # Calculate model-aware context limit
        model_context_chars = self._calculate_max_context_chars(settings, provider, model)

        # Parse compaction strategy from settings
        compaction_strategy_str = getattr(settings, "context_compaction_strategy", "tiered").lower()
        compaction_strategy_map = {
            "simple": CompactionStrategy.SIMPLE,
            "tiered": CompactionStrategy.TIERED,
            "semantic": CompactionStrategy.SEMANTIC,
            "hybrid": CompactionStrategy.HYBRID,
        }
        compaction_strategy = compaction_strategy_map.get(
            compaction_strategy_str, CompactionStrategy.TIERED
        )

        self._conversation_controller = ConversationController(
            config=ConversationConfig(
                max_context_chars=model_context_chars,
                enable_stage_tracking=True,
                enable_context_monitoring=True,
                # Smart compaction settings from Settings
                compaction_strategy=compaction_strategy,
                min_messages_to_keep=getattr(settings, "context_min_messages_to_keep", 6),
                tool_result_retention_weight=getattr(
                    settings, "context_tool_retention_weight", 1.5
                ),
                recent_message_weight=getattr(settings, "context_recency_weight", 2.0),
                semantic_relevance_threshold=getattr(settings, "context_semantic_threshold", 0.3),
            ),
            message_history=self.conversation,
            state_machine=self.conversation_state,
            # Pass SQLite store for persistent semantic memory
            conversation_store=self.memory_manager,
            session_id=self._memory_session_id,
        )
        self._conversation_controller.set_system_prompt(self._system_prompt)

        # ToolPipeline: Coordinates tool execution flow
        self._tool_pipeline = ToolPipeline(
            tool_registry=self.tools,
            tool_executor=self.tool_executor,
            config=ToolPipelineConfig(
                tool_budget=self.tool_budget,
                enable_caching=self.tool_cache is not None,
                enable_analytics=True,
                enable_failed_signature_tracking=True,
            ),
            tool_cache=self.tool_cache,
            argument_normalizer=self.argument_normalizer,
            on_tool_start=self._on_tool_start_callback,
            on_tool_complete=self._on_tool_complete_callback,
        )

        # StreamingController: Manages streaming sessions and metrics
        self._streaming_controller = StreamingController(
            config=StreamingControllerConfig(
                max_history=100,
                enable_metrics_collection=self.streaming_metrics_collector is not None,
            ),
            metrics_collector=self.streaming_metrics_collector,
            on_session_complete=self._on_streaming_session_complete,
        )

        # StreamingChatHandler: Testable extraction of streaming loop logic
        # This component encapsulates limit checking, response processing, and iteration control
        # Uses orchestrator as message_adder since we implement add_message()
        session_time_limit = getattr(settings, "session_time_limit", 240.0)
        self._streaming_handler = StreamingChatHandler(
            settings=settings,
            message_adder=self,  # Orchestrator implements add_message()
            session_time_limit=session_time_limit,
        )

        # TaskAnalyzer: Unified task analysis facade
        self._task_analyzer = get_task_analyzer()

        # ContextCompactor: Proactive context management and tool result truncation
        # This component provides:
        # - Proactive compaction before context overflow (triggers at 70% utilization)
        # - Smart tool result truncation with content-aware strategies
        # - Token estimation with content-type-specific factors
        truncation_strategy_str = getattr(settings, "tool_truncation_strategy", "smart").lower()
        truncation_strategy_map = {
            "head": TruncationStrategy.HEAD,
            "tail": TruncationStrategy.TAIL,
            "both": TruncationStrategy.BOTH,
            "smart": TruncationStrategy.SMART,
        }
        truncation_strategy = truncation_strategy_map.get(
            truncation_strategy_str, TruncationStrategy.SMART
        )

        self._context_compactor = create_context_compactor(
            controller=self._conversation_controller,
            proactive_threshold=getattr(settings, "context_proactive_threshold", 0.90),
            min_messages_after_compact=getattr(settings, "context_min_messages_after_compact", 8),
            tool_result_max_chars=getattr(settings, "max_tool_output_chars", 8192),
            tool_result_max_lines=getattr(settings, "max_tool_output_lines", 200),
            truncation_strategy=truncation_strategy,
            preserve_code_blocks=True,
            enable_proactive=getattr(settings, "context_proactive_compaction", True),
            enable_tool_truncation=getattr(settings, "tool_result_truncation", True),
        )

        # ToolOutputFormatter: LLM-context-aware formatting of tool results
        # (Initialization consolidated: the definitive initialization occurs
        # later in this method to avoid accidental double-initialization.)
        # Previous duplicate initialization removed to reduce boilerplate.

        # Initialize UsageAnalytics singleton for data-driven optimization
        analytics_cache_dir = (
            Path(settings.cache_dir)
            if hasattr(settings, "cache_dir") and settings.cache_dir
            else None
        )
        self._usage_analytics = UsageAnalytics.get_instance(
            AnalyticsConfig(
                cache_dir=analytics_cache_dir,
                enable_prometheus_export=getattr(settings, "enable_prometheus_export", True),
            )
        )

        # Initialize ToolSequenceTracker for intelligent next-tool suggestions
        self._sequence_tracker = create_sequence_tracker(
            use_predefined=getattr(settings, "use_predefined_patterns", True),
            learning_rate=getattr(settings, "sequence_learning_rate", 0.3),
        )

        # Initialize ToolOutputFormatter for LLM-context-aware output formatting
        self._tool_output_formatter = create_tool_output_formatter(
            config=ToolOutputFormatterConfig(
                max_output_chars=getattr(settings, "max_tool_output_chars", 15000),
                file_structure_threshold=getattr(settings, "file_structure_threshold", 50000),
            ),
            truncator=self._context_compactor,  # Use context compactor for smart truncation
        )

        # Initialize RecoveryHandler for handling model failures and stuck states
        # Uses DI container with fallback to direct instantiation
        # Integrates with existing framework components:
        # - QLearningStore from adaptive_mode_controller for recovery action learning
        # - UsageAnalytics singleton for recording recovery metrics
        # - ContextCompactor for proactive context management
        self._recovery_handler = self._container.get_optional(RecoveryHandlerProtocol)
        if self._recovery_handler is None:
            # Fallback to direct instantiation if DI container not configured
            enable_recovery = getattr(settings, "enable_recovery_system", True)
            if enable_recovery:
                try:
                    self._recovery_handler = RecoveryHandler.create(settings=settings)
                    logger.debug("RecoveryHandler created via fallback (direct instantiation)")
                except Exception as e:
                    logger.warning(f"RecoveryHandler creation failed: {e}")
                    self._recovery_handler = None
            else:
                self._recovery_handler = None

        # Wire context compactor if recovery handler exists
        if self._recovery_handler and hasattr(self._recovery_handler, "set_context_compactor"):
            self._recovery_handler.set_context_compactor(self._context_compactor)
            logger.debug("RecoveryHandler wired with ContextCompactor")

        # Create recovery integration submodule for clean delegation
        # This encapsulates recovery logic, making stream_chat cleaner
        self._recovery_integration = create_recovery_integration(
            recovery_handler=self._recovery_handler,
            settings=settings,
        )

        # Initialize ObservabilityIntegration for unified event bus
        # This provides automatic event emission for tool execution, state changes, and errors
        enable_observability = getattr(settings, "enable_observability", True)
        if enable_observability:
            self._observability = ObservabilityIntegration()
            # Wire the state machine hooks for automatic state change events
            if hasattr(self, "conversation_state") and self.conversation_state:
                self._observability.wire_state_machine(self.conversation_state)
            logger.debug("Observability integration enabled")
        else:
            self._observability = None

        logger.info(
            "Orchestrator initialized with decomposed components: "
            "ConversationController, ToolPipeline, StreamingController, StreamingChatHandler, "
            "TaskAnalyzer, ContextCompactor, UsageAnalytics, ToolSequenceTracker, "
            "ToolOutputFormatter, RecoveryCoordinator, ObservabilityIntegration"
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

        Converts StreamingSession data into SessionReward and updates Q-values
        based on session outcome (success, latency, throughput, tool usage).
        """
        try:
            from victor.agent.rl_model_selector import get_model_selector, SessionReward

            selector = get_model_selector()
            if selector is None:
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

            # Create reward signal
            reward = SessionReward(
                session_id=session.session_id,
                provider=session.provider,
                model=session.model,
                success=success,
                latency_seconds=session.duration,
                token_count=token_count,
                tool_calls_made=tool_calls_made,
            )

            # Update Q-value with reward signal
            new_q = selector.update_q_value(reward)
            logger.debug(
                f"RL feedback: provider={session.provider} reward={reward.reward:.3f} "
                f"new_q={new_q:.3f} success={success} duration={session.duration:.1f}s"
            )

        except ImportError:
            # RL module not available - skip silently
            pass
        except Exception as e:
            # Don't let RL errors affect main flow
            logger.warning(f"Failed to send RL reward signal: {e}")

    # =====================================================================
    # Component accessors for external use
    # =====================================================================

    @property
    def conversation_controller(self) -> ConversationController:
        """Get the conversation controller component.

        Returns:
            ConversationController instance for managing conversation state
        """
        return self._conversation_controller

    @property
    def tool_pipeline(self) -> ToolPipeline:
        """Get the tool pipeline component.

        Returns:
            ToolPipeline instance for coordinating tool execution
        """
        return self._tool_pipeline

    @property
    def streaming_controller(self) -> StreamingController:
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
    def task_analyzer(self) -> TaskAnalyzer:
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
    def provider_manager(self) -> ProviderManager:
        """Get the provider manager component.

        Returns:
            ProviderManager instance for unified provider management
        """
        return self._provider_manager

    @property
    def context_compactor(self) -> ContextCompactor:
        """Get the context compactor component.

        Returns:
            ContextCompactor instance for proactive context management
        """
        return self._context_compactor

    @property
    def tool_output_formatter(self) -> ToolOutputFormatter:
        """Get the tool output formatter for LLM-context-aware formatting.

        Returns:
            ToolOutputFormatter instance for formatting tool outputs
        """
        return self._tool_output_formatter

    @property
    def usage_analytics(self) -> UsageAnalytics:
        """Get the usage analytics singleton.

        Returns:
            UsageAnalytics instance for data-driven optimization
        """
        return self._usage_analytics

    @property
    def sequence_tracker(self) -> ToolSequenceTracker:
        """Get the tool sequence tracker for intelligent next-tool suggestions.

        Returns:
            ToolSequenceTracker instance for pattern learning
        """
        return self._sequence_tracker

    @property
    def recovery_handler(self) -> Optional[RecoveryHandler]:
        """Get the recovery handler for model failure recovery.

        Returns:
            RecoveryHandler instance or None if not enabled
        """
        return self._recovery_handler

    @property
    def recovery_integration(self) -> OrchestratorRecoveryIntegration:
        """Get the recovery integration submodule.

        Returns:
            OrchestratorRecoveryIntegration for delegated recovery handling
        """
        return self._recovery_integration

    @property
    def code_correction_middleware(self) -> Optional[Any]:
        """Get the code correction middleware for automatic code validation/fixing.

        Returns:
            CodeCorrectionMiddleware instance or None if not enabled
        """
        return self._code_correction_middleware

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
            except Exception as e:
                logger.warning(f"Failed to initialize IntelligentPipeline: {e}")
                self._intelligent_pipeline_enabled = False

        return self._intelligent_integration

    async def _prepare_intelligent_request(
        self, task: str, task_type: str
    ) -> Optional[Dict[str, Any]]:
        """Pre-request hook for intelligent pipeline integration.

        Called at the start of stream_chat to:
        - Get mode transition recommendations (Q-learning)
        - Get optimal tool budget for task type
        - Enable prompt optimization if configured

        Args:
            task: The user's task/query
            task_type: Detected task type (analysis, edit, etc.)

        Returns:
            Dictionary with recommendations, or None if pipeline disabled
        """
        integration = self.intelligent_integration
        if not integration:
            return None

        try:
            # Get current mode from conversation state
            # Note: ConversationStage uses auto() which returns int values,
            # so we use stage.name.lower() to get a string mode name
            stage = self.conversation_state.get_current_stage()
            current_mode = stage.name.lower() if stage else "explore"

            # Prepare request context (async call to pipeline)
            context = await integration.prepare_request(
                task=task,
                task_type=task_type,
                current_mode=current_mode,
            )

            # Apply recommended tool budget if available (skip if user made a sticky override)
            # NOTE: We no longer reduce the budget based on pipeline recommendations.
            # The pipeline may suggest a smaller budget based on Q-learning, but this
            # caused premature stopping (e.g., 50 -> 5). The user's budget is authoritative.
            # We only log the recommendation for debugging purposes.
            if context.recommended_tool_budget:
                sticky_budget = getattr(self.unified_tracker, "_sticky_user_budget", False)
                if not sticky_budget:
                    current_budget = self.unified_tracker.progress.tool_budget
                    # Only log significant differences, but don't reduce the budget
                    if abs(context.recommended_tool_budget - current_budget) > 10:
                        logger.debug(
                            f"IntelligentPipeline recommended budget {context.recommended_tool_budget} "
                            f"differs from current {current_budget}, keeping current budget"
                        )

            return {
                "recommended_mode": context.recommended_mode,
                "recommended_tool_budget": context.recommended_tool_budget,
                "should_continue": context.should_continue,
                "system_prompt_addition": context.system_prompt if context.system_prompt else None,
            }
        except Exception as e:
            logger.debug(f"IntelligentPipeline prepare_request failed: {e}")
            return None

    async def _validate_intelligent_response(
        self,
        response: str,
        query: str,
        tool_calls: int,
        task_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Post-response hook for intelligent pipeline integration.

        Called after each streaming iteration to:
        - Score response quality (coherence, completeness, relevance)
        - Verify grounding (detect hallucinations)
        - Record feedback for Q-learning

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

        # Skip validation for empty or very short responses
        if not response or len(response.strip()) < 50:
            return None

        try:
            result = await integration.validate_response(
                response=response,
                query=query,
                tool_calls=tool_calls,
                success=True,
                task_type=task_type,
            )

            # Log quality warnings if below threshold
            if not result.is_valid:
                logger.warning(
                    f"IntelligentPipeline: Response below quality threshold "
                    f"(quality={result.quality_score:.2f}, grounded={result.is_grounded})"
                )

            return {
                "quality_score": result.quality_score,
                "grounding_score": result.grounding_score,
                "is_grounded": result.is_grounded,
                "is_valid": result.is_valid,
                "grounding_issues": result.grounding_issues,
            }
        except Exception as e:
            logger.debug(f"IntelligentPipeline validate_response failed: {e}")
            return None

    def _record_intelligent_outcome(
        self,
        success: bool,
        quality_score: float = 0.5,
        user_satisfied: bool = True,
        completed: bool = True,
    ) -> None:
        """Record outcome for Q-learning feedback.

        Called at the end of a conversation to record the outcome
        for reinforcement learning. This helps the system learn
        optimal mode transitions and tool budgets.

        Args:
            success: Whether the task was completed successfully
            quality_score: Final quality score (0.0-1.0)
            user_satisfied: Whether user seemed satisfied
            completed: Whether task reached completion
        """
        integration = self.intelligent_integration
        if not integration:
            return

        try:
            # Access the mode controller through the pipeline
            pipeline = integration.pipeline
            if hasattr(pipeline, "_mode_controller") and pipeline._mode_controller:
                pipeline._mode_controller.record_outcome(
                    success=success,
                    quality_score=quality_score,
                    user_satisfied=user_satisfied,
                    completed=completed,
                )
                logger.debug(
                    f"IntelligentPipeline recorded outcome: "
                    f"success={success}, quality={quality_score:.2f}"
                )
        except Exception as e:
            logger.debug(f"IntelligentPipeline record_outcome failed: {e}")

    def _should_continue_intelligent(self) -> tuple[bool, str]:
        """Check if processing should continue using learned behaviors.

        Uses Q-learning based decisions to determine if the agent
        should continue processing or transition to completion.

        Returns:
            Tuple of (should_continue, reason)
        """
        integration = self.intelligent_integration
        if not integration:
            return True, "Pipeline disabled"

        try:
            return integration.should_continue()
        except Exception as e:
            logger.debug(f"IntelligentPipeline should_continue failed: {e}")
            return True, "Fallback to continue"

    @property
    def safety_checker(self) -> SafetyChecker:
        """Get the safety checker for dangerous operation detection.

        UI layers can use this to set confirmation callbacks:
            orchestrator.safety_checker.confirmation_callback = my_callback

        Returns:
            SafetyChecker instance for dangerous operation detection
        """
        return self._safety_checker

    @property
    def auto_committer(self) -> Optional[AutoCommitter]:
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

        Args:
            middleware_list: List of MiddlewareProtocol implementations.
        """
        if not middleware_list:
            return

        # Store middleware for use during tool execution
        self._vertical_middleware = middleware_list

        # Initialize middleware chain if not present
        if not hasattr(self, "_middleware_chain") or self._middleware_chain is None:
            try:
                from victor.agent.middleware_chain import MiddlewareChain

                self._middleware_chain = MiddlewareChain()
            except ImportError:
                logger.warning("MiddlewareChain not available")
                return

        # Add all middleware to chain
        for middleware in middleware_list:
            self._middleware_chain.add(middleware)

        logger.debug(f"Applied {len(middleware_list)} vertical middleware")

    def apply_vertical_safety_patterns(self, patterns: List[Any]) -> None:
        """Apply safety patterns from vertical extensions.

        Called by FrameworkShim to inject vertical-specific danger patterns
        into the safety checker.

        Args:
            patterns: List of SafetyPattern objects.
        """
        if not patterns:
            return

        # Store patterns for reference
        self._vertical_safety_patterns = patterns

        # Add patterns to safety checker
        if self._safety_checker is not None:
            try:
                # Convert SafetyPattern to the format SafetyChecker expects
                for pattern in patterns:
                    self._safety_checker.add_custom_pattern(
                        pattern=pattern.pattern,
                        description=pattern.description,
                        risk_level=pattern.risk_level,
                        category=pattern.category,
                    )
                logger.debug(f"Applied {len(patterns)} vertical safety patterns")
            except AttributeError:
                # SafetyChecker might not support add_custom_pattern
                logger.debug("SafetyChecker does not support add_custom_pattern")

    def get_middleware_chain(self) -> Optional[Any]:
        """Get the middleware chain for tool execution.

        Returns:
            MiddlewareChain instance or None if not initialized.
        """
        return getattr(self, "_middleware_chain", None)

    @property
    def messages(self) -> List[Message]:
        """Get conversation messages (backward compatibility property).

        Returns:
            List of messages in conversation history
        """
        return self.conversation.messages

    def _get_context_size(self) -> tuple[int, int]:
        """Calculate current context size in chars and estimated tokens.

        Returns:
            Tuple of (char_count, estimated_token_count)
        """
        # Delegate to ConversationController for centralized context tracking
        metrics = self._conversation_controller.get_context_metrics()
        return metrics.char_count, metrics.estimated_tokens

    def _get_model_context_window(self) -> int:
        """Get context window size for the current model.

        Queries the provider for model-specific context window.
        Falls back to settings or default if provider doesn't support it.

        Returns:
            Context window size in tokens
        """
        # Try to get context window from provider
        if hasattr(self.provider, "get_context_window"):
            try:
                return self.provider.get_context_window(self.model)
            except Exception:
                pass

        # Known provider defaults (in tokens)
        provider_defaults = {
            "anthropic": 200000,  # Claude models
            "openai": 128000,  # GPT-4 Turbo
            "google": 1000000,  # Gemini 1.5 Pro
            "xai": 131072,  # Grok models
            "deepseek": 131072,  # DeepSeek V3
            "moonshot": 262144,  # Kimi K2
            "ollama": 32768,  # Local models (conservative)
            "lmstudio": 32768,  # Local models (conservative)
        }

        return provider_defaults.get(self.provider_name, 100000)

    def _get_max_context_chars(self) -> int:
        """Get maximum context size in characters.

        Derives from model context window, converting tokens to chars.
        Average ~4 chars per token, with safety margin.

        Returns:
            Maximum context size in characters
        """
        # Check settings override first
        settings_max = getattr(self.settings, "max_context_chars", None)
        if settings_max and settings_max > 0:
            return settings_max

        # Calculate from model context window
        # Use ~3.5 chars per token with 80% safety margin
        context_tokens = self._get_model_context_window()
        return int(context_tokens * 3.5 * 0.8)

    def _check_context_overflow(self, max_context_chars: int = 200000) -> bool:
        """Check if context is at risk of overflow.

        Args:
            max_context_chars: Maximum allowed context size in chars

        Returns:
            True if context is dangerously large
        """
        # Delegate to ConversationController
        metrics = self._conversation_controller.get_context_metrics()

        # Update debug logger
        self.debug_logger.log_context_size(metrics.char_count, metrics.estimated_tokens)

        if metrics.is_overflow_risk:
            logger.warning(
                f"Context overflow risk: {metrics.char_count:,} chars "
                f"(~{metrics.estimated_tokens:,} tokens). "
                f"Max: {metrics.max_context_chars:,} chars"
            )
            return True

        return False

    def get_context_metrics(self) -> ContextMetrics:
        """Get detailed context metrics.

        Returns:
            ContextMetrics with size and overflow information
        """
        return self._conversation_controller.get_context_metrics()

    def _init_stream_metrics(self) -> StreamMetrics:
        """Initialize fresh stream metrics for a new streaming session."""
        return self._metrics_collector.init_stream_metrics()

    def _init_conversation_embedding_store(self) -> None:
        """Initialize LanceDB embedding store for semantic conversation retrieval.

        This creates a ConversationEmbeddingStore that:
        - Stores pre-computed message embeddings in LanceDB
        - Enables O(log n) vector search instead of O(n) on-the-fly embedding
        - Syncs automatically when messages are added to ConversationStore
        """
        if self.memory_manager is None:
            return

        try:
            from victor.embeddings.service import EmbeddingService

            # Get the shared embedding service
            embedding_service = EmbeddingService.get_instance()

            # Create the embedding store
            self._conversation_embedding_store = ConversationEmbeddingStore(
                embedding_service=embedding_service,
            )

            # Wire it to the memory manager for automatic sync
            self.memory_manager.set_embedding_store(self._conversation_embedding_store)

            # Also set the embedding service for fallback
            self.memory_manager.set_embedding_service(embedding_service)

            # Initialize async (fire and forget for faster startup).
            # If there's no running event loop (e.g., unit tests), fall back
            # to synchronous initialization to avoid 'coroutine was never awaited' warnings.
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
        except Exception as e:
            logger.warning(f"Failed to initialize ConversationEmbeddingStore: {e}")
            self._conversation_embedding_store = None

    def _record_first_token(self) -> None:
        """Record the time of first token received."""
        self._metrics_collector.record_first_token()

    def _finalize_stream_metrics(self) -> Optional[StreamMetrics]:
        """Finalize stream metrics at end of streaming session."""
        return self._metrics_collector.finalize_stream_metrics()

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
            status["components"]["code_correction"]["config"] = {
                "auto_fix": self._code_correction_middleware.config.auto_fix,
                "max_iterations": self._code_correction_middleware.config.max_iterations,
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

        Enables automatic health checks at configured intervals and
        auto-failover to healthy providers when enabled.

        Returns:
            True if monitoring started, False if already running or unavailable
        """
        if not hasattr(self, "_provider_manager") or not self._provider_manager:
            logger.warning("Provider manager not available for health monitoring")
            return False

        try:
            await self._provider_manager.start_health_monitoring()
            logger.info("Provider health monitoring started")
            return True
        except Exception as e:
            logger.warning(f"Failed to start health monitoring: {e}")
            return False

    async def stop_health_monitoring(self) -> bool:
        """Stop background provider health monitoring.

        Call this during graceful shutdown to clean up monitoring tasks.

        Returns:
            True if monitoring stopped, False if not running or error
        """
        if not hasattr(self, "_provider_manager") or not self._provider_manager:
            return False

        try:
            await self._provider_manager.stop_health_monitoring()
            logger.debug("Provider health monitoring stopped")
            return True
        except Exception as e:
            logger.warning(f"Failed to stop health monitoring: {e}")
            return False

    async def get_provider_health(self) -> Dict[str, Any]:
        """Get health status of all registered providers.

        Returns:
            Dictionary with provider health information:
            - current_provider: Name of current provider
            - is_healthy: Current provider health status
            - healthy_providers: List of healthy provider names
            - can_failover: Whether failover is possible
        """
        result: Dict[str, Any] = {
            "current_provider": self.provider_name,
            "is_healthy": True,
            "healthy_providers": [self.provider_name] if self.provider_name else [],
            "can_failover": False,
        }

        if hasattr(self, "_provider_manager") and self._provider_manager:
            try:
                # Get current state
                state = self._provider_manager.get_current_state()
                if state:
                    result["is_healthy"] = state.is_healthy
                    result["switch_count"] = state.switch_count

                # Get healthy providers for failover
                healthy = await self._provider_manager.get_healthy_providers()
                result["healthy_providers"] = healthy
                result["can_failover"] = len(healthy) > 1

            except Exception as e:
                logger.warning(f"Failed to get provider health: {e}")
                result["error"] = str(e)

        return result

    async def graceful_shutdown(self) -> Dict[str, bool]:
        """Perform graceful shutdown of all orchestrator components.

        Flushes analytics, stops health monitoring, and cleans up resources.
        Call this before application exit.

        Returns:
            Dictionary with shutdown status for each component
        """
        results: Dict[str, bool] = {}

        # Flush analytics data
        try:
            flush_results = self.flush_analytics()
            results["analytics_flushed"] = all(flush_results.values())
        except Exception as e:
            logger.warning(f"Failed to flush analytics during shutdown: {e}")
            results["analytics_flushed"] = False

        # Stop health monitoring
        try:
            results["health_monitoring_stopped"] = await self.stop_health_monitoring()
        except Exception as e:
            logger.warning(f"Failed to stop health monitoring: {e}")
            results["health_monitoring_stopped"] = False

        # End usage analytics session
        if hasattr(self, "_usage_analytics") and self._usage_analytics:
            try:
                if self._usage_analytics._current_session is not None:
                    self._usage_analytics.end_session()
                results["session_ended"] = True
            except Exception as e:
                logger.warning(f"Failed to end analytics session: {e}")
                results["session_ended"] = False
        else:
            results["session_ended"] = True

        logger.info(f"Graceful shutdown complete: {results}")
        return results

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
            self._provider_manager._current_state = ProviderState(
                provider=new_provider,
                provider_name=provider_name.lower(),
                model=new_model,
            )
            self._provider_manager.initialize_tool_adapter()

            # Sync local attributes from ProviderManager
            self.provider = self._provider_manager.provider
            self.model = self._provider_manager.model
            self.provider_name = self._provider_manager.provider_name
            self.tool_adapter = self._provider_manager.tool_adapter
            self.tool_calling_caps = self._provider_manager.capabilities

            # Apply model-specific exploration settings to unified tracker
            self.unified_tracker.set_model_exploration_settings(
                exploration_multiplier=self.tool_calling_caps.exploration_multiplier,
                continuation_patience=self.tool_calling_caps.continuation_patience,
            )

            # Reinitialize prompt builder
            self.prompt_builder = SystemPromptBuilder(
                provider_name=self.provider_name,
                model=new_model,
                tool_adapter=self.tool_adapter,
                capabilities=self.tool_calling_caps,
            )

            # Rebuild system prompt with new adapter hints
            base_system_prompt = self._build_system_prompt_with_adapter()
            if self.project_context.content:
                self._system_prompt = (
                    base_system_prompt + "\n\n" + self.project_context.get_system_prompt_addition()
                )
            else:
                self._system_prompt = base_system_prompt

            # Update tool budget based on new adapter's recommendation unless user override is sticky
            sticky_budget = getattr(self.unified_tracker, "_sticky_user_budget", False)
            if sticky_budget:
                logger.debug("Skipping tool budget reset on provider switch (sticky user override)")
            else:
                default_budget = max(self.tool_calling_caps.recommended_tool_budget, 50)
                self.tool_budget = getattr(self.settings, "tool_call_budget", default_budget)

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
                self._provider_manager._current_state.model = model
                self._provider_manager.initialize_tool_adapter()

            # Sync local attributes from ProviderManager
            self.model = self._provider_manager.model
            self.tool_adapter = self._provider_manager.tool_adapter
            self.tool_calling_caps = self._provider_manager.capabilities

            # Apply model-specific exploration settings to unified tracker
            self.unified_tracker.set_model_exploration_settings(
                exploration_multiplier=self.tool_calling_caps.exploration_multiplier,
                continuation_patience=self.tool_calling_caps.continuation_patience,
            )

            # Reinitialize prompt builder
            self.prompt_builder = SystemPromptBuilder(
                provider_name=self.provider_name,
                model=model,
                tool_adapter=self.tool_adapter,
                capabilities=self.tool_calling_caps,
            )

            # Rebuild system prompt
            base_system_prompt = self._build_system_prompt_with_adapter()
            if self.project_context.content:
                self._system_prompt = (
                    base_system_prompt + "\n\n" + self.project_context.get_system_prompt_addition()
                )
            else:
                self._system_prompt = base_system_prompt

            # Update tool budget
            default_budget = max(self.tool_calling_caps.recommended_tool_budget, 50)
            self.tool_budget = getattr(self.settings, "tool_call_budget", default_budget)

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
            return f"{base_prompt}\n\n{budget_hint}"

        return base_prompt

    def _strip_markup(self, text: str) -> str:
        """Remove simple XML/HTML-like tags to salvage plain text."""
        return self.sanitizer.strip_markup(text)

    def _sanitize_response(self, text: str) -> str:
        """Sanitize model response by removing malformed patterns."""
        return self.sanitizer.sanitize(text)

    def _is_garbage_content(self, content: str) -> bool:
        """Detect if content is garbage/malformed output from local models."""
        return self.sanitizer.is_garbage_content(content)

    def _is_valid_tool_name(self, name: str) -> bool:
        """Check if a tool name is valid and not a hallucination."""
        return self.sanitizer.is_valid_tool_name(name)

    def _infer_git_operation(
        self, original_name: str, canonical_name: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Infer git operation from alias when not explicitly provided.

        When the model calls 'git_log' instead of 'git' with operation='log',
        we need to infer the operation from the alias.

        Args:
            original_name: Original tool name from model (e.g., 'git_log')
            canonical_name: Resolved canonical name (e.g., 'git')
            args: Tool arguments

        Returns:
            Updated args with inferred operation if applicable
        """
        # Only process git tool with operation-based aliases
        if canonical_name != "git":
            return args

        # If operation already provided, no inference needed
        if args.get("operation"):
            return args

        # Map alias suffixes to operation names
        alias_to_operation = {
            "git_status": "status",
            "git_diff": "diff",
            "git_log": "log",
            "git_commit": "commit",
            "git_branch": "branch",
            "git_stage": "stage",
        }

        inferred_op = alias_to_operation.get(original_name)
        if inferred_op:
            logger.debug(
                f"Inferred git operation '{inferred_op}' from alias '{original_name}'"
            )
            args = dict(args)  # Copy to avoid mutation
            args["operation"] = inferred_op

        return args

    def _resolve_shell_variant(self, tool_name: str) -> str:
        """Resolve shell aliases to the appropriate enabled shell variant.

        LLMs often hallucinate shell tool names like 'run', 'bash', 'execute'.
        These map to 'shell' canonically, but in INITIAL stage only 'shell_readonly'
        may be enabled. This method resolves to whichever shell variant is available.

        Args:
            tool_name: Original tool name (may be alias like 'run')

        Returns:
            The appropriate enabled shell tool name, or original if not a shell alias
        """
        from victor.tools.tool_names import get_canonical_name, ToolNames

        # Shell-related aliases that should resolve intelligently
        shell_aliases = {"run", "bash", "execute", "cmd", "execute_bash"}

        if tool_name not in shell_aliases:
            return tool_name

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

    def _filter_tools_by_intent(self, tools: List[Any]) -> List[Any]:
        """Filter tools based on detected user intent.

        This method enforces intent-based tool restrictions:
        - DISPLAY_ONLY: Blocks write tools (write_file, edit_files, etc.)
        - READ_ONLY: Blocks write tools AND generation tools
        - WRITE_ALLOWED: No restrictions
        - AMBIGUOUS: No restrictions (relies on prompt guard)

        The blocked tools are defined in action_authorizer.INTENT_BLOCKED_TOOLS,
        which is the single source of truth for tool filtering.

        Args:
            tools: List of tool definitions (ToolDefinition objects or dicts)

        Returns:
            Filtered list of tools, excluding blocked tools for current intent
        """
        if not hasattr(self, "_current_intent"):
            return tools

        blocked_tools = INTENT_BLOCKED_TOOLS.get(self._current_intent, frozenset())
        if not blocked_tools:
            return tools

        def get_tool_name(tool: Any) -> str:
            """Extract tool name from ToolDefinition object or dict.

            Returns:
                Tool name string or empty string if not found
            """
            if hasattr(tool, "name"):
                return tool.name
            elif isinstance(tool, dict):
                return tool.get("name", "")
            return ""

        original_count = len(tools)
        filtered = [t for t in tools if get_tool_name(t) not in blocked_tools]
        filtered_count = original_count - len(filtered)

        if filtered_count > 0:
            blocked_names = blocked_tools & {get_tool_name(t) for t in tools}
            logger.info(
                f"Intent {self._current_intent.value}: filtered {filtered_count} "
                f"write/generation tools (blocked: {blocked_names})"
            )

        return filtered

    def _classify_task_keywords(self, user_message: str) -> Dict[str, Any]:
        """Classify task type based on keywords in the user message.

        Uses UnifiedTaskClassifier for robust classification with:
        - Negation detection (handles "don't analyze", "skip the review")
        - Confidence scoring for better decisions
        - Weighted keyword matching

        Args:
            user_message: The user's input message

        Returns:
            Dictionary with:
            - is_action_task: bool - True if task requires action (create/execute/run)
            - is_analysis_task: bool - True if task requires analysis/exploration
            - needs_execution: bool - True if task specifically requires execution
            - coarse_task_type: str - "analysis", "action", or "default"
            - confidence: float - Classification confidence (0.0-1.0)
            - source: str - Classification source ("keyword", "context", "ensemble")
            - task_type: str - Detailed task type
        """
        from victor.agent.unified_classifier import get_unified_classifier

        classifier = get_unified_classifier()
        result = classifier.classify(user_message)

        # Log negated keywords for debugging
        if result.negated_keywords:
            negated_strs = [f"{m.keyword}" for m in result.negated_keywords]
            logger.debug(f"Negated keywords detected: {negated_strs}")

        return result.to_legacy_dict()

    def _classify_task_with_context(
        self, user_message: str, history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Classify task with conversation context for improved accuracy.

        Uses conversation history to boost classification confidence when
        the current message is ambiguous but context suggests a task type.

        Args:
            user_message: The user's input message
            history: Optional conversation history for context boosting

        Returns:
            Dictionary with classification results (same as _classify_task_keywords
            but with potential context boosting applied)
        """
        from victor.agent.unified_classifier import get_unified_classifier

        classifier = get_unified_classifier()

        if history:
            result = classifier.classify_with_context(user_message, history)
        else:
            result = classifier.classify(user_message)

        if result.context_signals:
            logger.debug(f"Context signals applied: {result.context_signals}")

        return result.to_legacy_dict()

    def _get_tool_status_message(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Generate a user-friendly status message for a tool execution.

        Provides context-aware status messages showing relevant details
        (command, path, query, etc.) for different tool types.

        Args:
            tool_name: Name of the tool being executed
            tool_args: Arguments passed to the tool

        Returns:
            Status message string with emoji prefix
        """
        if tool_name == "execute_bash" and "command" in tool_args:
            cmd = tool_args["command"]
            cmd_display = cmd[:80] + "..." if len(cmd) > 80 else cmd
            return f"🔧 Running {tool_name}: `{cmd_display}`"

        if tool_name == "list_directory":
            path = tool_args.get("path", ".")
            return f"🔧 Listing directory: {path}"

        if tool_name == "read":
            path = tool_args.get("path", "file")
            return f"🔧 Reading file: {path}"

        if tool_name == "edit_files":
            files = tool_args.get("files", [])
            if files and isinstance(files, list):
                paths = [f.get("path", "?") for f in files[:3]]
                path_display = ", ".join(paths)
                if len(files) > 3:
                    path_display += f" (+{len(files) - 3} more)"
                return f"🔧 Editing: {path_display}"
            return f"🔧 Running {tool_name}..."

        if tool_name == "write":
            path = tool_args.get("path", "file")
            return f"🔧 Writing file: {path}"

        if tool_name == "code_search":
            query = tool_args.get("query", "")
            query_display = query[:50] + "..." if len(query) > 50 else query
            return f"🔧 Searching: {query_display}"

        return f"🔧 Running {tool_name}..."

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

        Encapsulates the complex decision logic for handling responses without tool
        calls, including intent classification, continuation prompting, and summary
        requests.

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
            Dictionary with:
            - action: str - One of: "continue_asking_input", "return_to_user",
                          "prompt_tool_call", "request_summary",
                          "request_completion", "finish", "force_tool_execution"
            - message: Optional[str] - System message to inject (if any)
            - reason: str - Human-readable reason for the action
            - updates: Dict - State updates (continuation_prompts, asking_input_prompts)
        """

        # Extract intent type
        intends_to_continue = intent_result.intent == IntentType.CONTINUATION
        is_completion = intent_result.intent == IntentType.COMPLETION
        is_asking_input = intent_result.intent == IntentType.ASKING_INPUT
        is_stuck_loop = intent_result.intent == IntentType.STUCK_LOOP

        # CRITICAL FIX: Handle stuck loop immediately - model is planning but not executing
        if is_stuck_loop:
            logger.warning(
                f"Detected STUCK_LOOP intent - model is planning but not executing. "
                f"Forcing summary."
            )
            return {
                "action": "request_summary",
                "message": (
                    "You appear to be stuck in a planning loop - you keep describing what "
                    "you will do but are not making actual tool calls.\n\n"
                    "Please either:\n"
                    "1. Make an ACTUAL tool call NOW (not just describe it), OR\n"
                    "2. Provide your response based on what you already know.\n\n"
                    "Do not describe what you will do - just do it or provide your answer."
                ),
                "reason": "STUCK_LOOP detected - forcing summary",
                "updates": {"continuation_prompts": 99},  # Prevent further prompting
            }

        # Configuration - use configurable thresholds from settings
        max_asking_input_prompts = 3
        requires_continuation_support = is_analysis_task or is_action_task or intends_to_continue
        # Get continuation prompt limits from settings (default: 4 for analysis, 3 for action, 2 for others)
        max_cont_analysis = getattr(self.settings, "max_continuation_prompts_analysis", 4)
        max_cont_action = getattr(self.settings, "max_continuation_prompts_action", 3)
        max_cont_default = getattr(self.settings, "max_continuation_prompts_default", 2)
        max_continuation_prompts = (
            max_cont_analysis
            if is_analysis_task
            else (max_cont_action if is_action_task else max_cont_default)
        )

        # Budget/iteration thresholds
        budget_threshold = (
            self.tool_budget // 4 if requires_continuation_support else self.tool_budget // 2
        )
        max_iterations = self.unified_tracker.config.get("max_total_iterations", 50)
        iteration_threshold = (
            max_iterations * 3 // 4 if requires_continuation_support else max_iterations // 2
        )

        # State updates to apply
        updates: Dict[str, int] = {}

        # CRITICAL FIX: Handle tool mention without execution (hallucinated tool calls)
        # If model says "let me call search()" but didn't actually call it, force action
        if mentioned_tools and len(mentioned_tools) > 0:
            # Track consecutive hallucinated tool mentions
            if not hasattr(self, "_hallucinated_tool_count"):
                self._hallucinated_tool_count = 0
            self._hallucinated_tool_count += 1

            # After 2 consecutive hallucinations, force a more aggressive response
            if self._hallucinated_tool_count >= 2:
                self._hallucinated_tool_count = 0  # Reset counter
                updates["continuation_prompts"] = continuation_prompts + 2  # Double increment
                return {
                    "action": "force_tool_execution",
                    "message": (
                        f"CRITICAL: You mentioned using {', '.join(mentioned_tools)} but did NOT "
                        "actually execute any tool call. Your response contained TEXT describing "
                        "what you would do, but no actual tool invocation.\n\n"
                        "You MUST respond with an ACTUAL tool call in the proper format. "
                        "Do NOT describe what you will do - just DO it.\n\n"
                        "Example correct format:\n"
                        '{"name": "read", "arguments": {"path": "some/file.py"}}\n\n'
                        "If you cannot call tools, provide your final answer NOW."
                    ),
                    "reason": f"Forcing tool execution after {self._hallucinated_tool_count + 2} hallucinated tool mentions",
                    "updates": updates,
                    "mentioned_tools": mentioned_tools,  # Pass tools for handler delegation
                }
            else:
                updates["continuation_prompts"] = continuation_prompts + 1
                return {
                    "action": "prompt_tool_call",
                    "message": (
                        f"You mentioned {', '.join(mentioned_tools)} but did not call any tool. "
                        "Please ACTUALLY call the tool now, or provide your analysis.\n\n"
                        "DO NOT just describe what you will do - make the actual tool call."
                    ),
                    "reason": f"Tool mentioned but not executed: {mentioned_tools}",
                    "updates": updates,
                }

        # Reset hallucinated tool count on successful non-hallucination
        if hasattr(self, "_hallucinated_tool_count"):
            self._hallucinated_tool_count = 0

        # Handle model asking for user input
        if is_asking_input:
            if one_shot_mode and asking_input_prompts < max_asking_input_prompts:
                updates["asking_input_prompts"] = asking_input_prompts + 1
                return {
                    "action": "continue_asking_input",
                    "message": (
                        "Yes, please continue with the implementation. "
                        "Proceed with the most sensible approach based on what you've learned."
                    ),
                    "reason": f"Model asking input (one-shot) - auto-continuing ({asking_input_prompts + 1}/{max_asking_input_prompts})",
                    "updates": updates,
                }
            elif not one_shot_mode:
                return {
                    "action": "return_to_user",
                    "message": None,
                    "reason": "Model asking input (interactive) - returning to user",
                    "updates": updates,
                }

        # Check for structured content that indicates completion
        has_substantial_structured_content = content_length > 500 and any(
            marker in (full_content or "")
            for marker in ["## ", "**Summary", "**Strengths", "**Weaknesses"]
        )
        content_looks_incomplete = content_length < 800 and not has_substantial_structured_content

        # Determine if we should prompt for continuation
        should_prompt_continuation = intends_to_continue or (
            intent_result.intent == IntentType.NEUTRAL
            and content_looks_incomplete
            and not is_completion
        )

        # CRITICAL FIX: Detect stuck continuation loop pattern
        # If model keeps saying "let me read" but never calls tools, after 2 prompts
        # stop prompting and force a summary instead
        if (
            intends_to_continue
            and continuation_prompts >= 2
            and self.tool_calls_used == 0
        ):
            # Model is in a continuation loop without making any progress
            logger.warning(
                f"Detected stuck continuation loop: {continuation_prompts} prompts, "
                f"0 tool calls, intent={intent_result.intent.name}"
            )
            updates["continuation_prompts"] = 99  # Prevent further prompting
            return {
                "action": "request_summary",
                "message": (
                    "You have been saying you will examine files but have not made any tool calls. "
                    "Please provide your response NOW based on what you know, or explain "
                    "specifically what is preventing you from making tool calls."
                ),
                "reason": "Stuck continuation loop - no tool calls after multiple prompts",
                "updates": updates,
            }

        # Prompt model to make tool call if appropriate
        if (
            requires_continuation_support
            and should_prompt_continuation
            and self.tool_calls_used < budget_threshold
            and self.unified_tracker.iterations < iteration_threshold
            and continuation_prompts < max_continuation_prompts
        ):
            updates["continuation_prompts"] = continuation_prompts + 1
            return {
                "action": "prompt_tool_call",
                "message": (
                    "You said you would examine more files but did not call any tool. "
                    "Either:\n"
                    "1. Call ls(path='...') to explore a directory, OR\n"
                    "2. Call read(path='...') to read a specific file, OR\n"
                    "3. Provide your analysis NOW if you have enough information.\n\n"
                    "Make a tool call or provide your final response."
                ),
                "reason": f"Prompting for tool call ({continuation_prompts + 1}/{max_continuation_prompts})",
                "updates": updates,
            }

        # Request summary if max continuation prompts reached
        if (
            requires_continuation_support
            and continuation_prompts >= max_continuation_prompts
            and self.tool_calls_used > 0
        ):
            updates["continuation_prompts"] = 99  # Prevent further prompting
            return {
                "action": "request_summary",
                "message": (
                    "Please complete the task NOW based on what you have done so far. "
                    "Provide a summary of your progress and any remaining steps."
                ),
                "reason": f"Max continuation prompts ({max_continuation_prompts}) reached",
                "updates": updates,
            }

        # Request completion for incomplete output
        if (
            requires_continuation_support
            and self.tool_calls_used > 0
            and content_looks_incomplete
            and not is_completion
            and not getattr(self, "_final_summary_requested", False)
        ):
            return {
                "action": "request_completion",
                "message": (
                    "You have examined several files. Please provide a complete summary "
                    "of your analysis including:\n"
                    "1. **Strengths** - What the codebase does well\n"
                    "2. **Weaknesses** - Areas that need improvement\n"
                    "3. **Recommendations** - Specific suggestions for improvement\n\n"
                    "Provide your analysis NOW."
                ),
                "reason": "Incomplete output - requesting final summary",
                "updates": updates,
                "set_final_summary_requested": True,
            }

        # No more tool calls needed - finish
        return {
            "action": "finish",
            "message": None,
            "reason": "No more tool calls requested",
            "updates": updates,
        }

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
        """Register default workflows."""
        self.workflow_registry.register(NewFeatureWorkflow())

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

        # MCP integration (handled separately via registrar config)
        if getattr(self.settings, "use_mcp_tools", False):
            self._setup_mcp_integration()

    def _setup_mcp_integration(self) -> None:
        """Set up MCP integration using registry or legacy client.

        Uses MCPRegistry for auto-discovery if available, falls back to
        legacy single-client approach for backwards compatibility.
        """
        mcp_command = getattr(self.settings, "mcp_command", None)

        # Try MCPRegistry with auto-discovery first
        try:
            from victor.mcp.registry import MCPRegistry

            # Auto-discover MCP servers from standard locations
            self.mcp_registry = MCPRegistry.discover_servers()

            # Also register command from settings if specified
            if mcp_command:
                from victor.mcp.registry import MCPServerConfig

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
                self._create_background_task(self._start_mcp_registry(), name="mcp_registry_start")
            else:
                logger.debug("No MCP servers configured")

        except ImportError:
            logger.debug("MCPRegistry not available, using legacy client")
            self._setup_legacy_mcp(mcp_command)

        # Register MCP tool definitions
        for mcp_tool in get_mcp_tool_definitions():
            self.tools.register_dict(mcp_tool)

    def _setup_legacy_mcp(self, mcp_command: Optional[str]) -> None:
        """Set up legacy single MCP client (backwards compatibility)."""
        if mcp_command:
            try:
                from victor.mcp.client import MCPClient

                mcp_client = MCPClient()
                cmd_parts = mcp_command.split()
                self._create_background_task(
                    mcp_client.connect(cmd_parts), name="mcp_legacy_connect"
                )
                configure_mcp_client(mcp_client, prefix=getattr(self.settings, "mcp_prefix", "mcp"))
            except Exception as exc:
                logger.warning(f"Failed to start MCP client: {exc}")

    async def _start_mcp_registry(self) -> None:
        """Start MCP registry and connect to discovered servers."""
        try:
            await self.mcp_registry.start()
            results = await self.mcp_registry.connect_all()
            connected = sum(1 for v in results.values() if v)
            if connected > 0:
                logger.info(f"Connected to {connected} MCP server(s)")
                # Update available tools from MCP
                mcp_tools = self.mcp_registry.get_all_tools()
                if mcp_tools:
                    logger.info(f"Discovered {len(mcp_tools)} MCP tools")
        except Exception as e:
            logger.warning(f"Failed to start MCP registry: {e}")

    def _register_default_tool_dependencies(self) -> None:
        """Register minimal tool input/output specs for planning with cost tiers."""
        try:
            # Search tools - FREE tier (local operations)
            self.tool_graph.add_tool(
                "code_search",
                inputs=["query"],
                outputs=["file_candidates"],
                cost_tier=CostTier.FREE,
            )
            self.tool_graph.add_tool(
                "semantic_code_search",
                inputs=["query"],
                outputs=["file_candidates"],
                cost_tier=CostTier.FREE,
            )
            # File operations - FREE tier
            self.tool_graph.add_tool(
                "read_file",
                inputs=["file_candidates"],
                outputs=["file_contents"],
                cost_tier=CostTier.FREE,
            )

            # Analysis tools - LOW tier (more compute but local)
            self.tool_graph.add_tool(
                "analyze_docs",
                inputs=["file_contents"],
                outputs=["summary"],
                cost_tier=CostTier.LOW,
            )
            self.tool_graph.add_tool(
                "code_review",
                inputs=["file_contents"],
                outputs=["summary"],
                cost_tier=CostTier.LOW,
            )
            self.tool_graph.add_tool(
                "generate_docs",
                inputs=["file_contents"],
                outputs=["documentation"],
                cost_tier=CostTier.LOW,
            )
            self.tool_graph.add_tool(
                "security_scan",
                inputs=["file_contents"],
                outputs=["security_report"],
                cost_tier=CostTier.LOW,
            )
            self.tool_graph.add_tool(
                "analyze_metrics",
                inputs=["file_contents"],
                outputs=["metrics_report"],
                cost_tier=CostTier.LOW,
            )
        except Exception as exc:
            logger.debug(f"Failed to register tool dependencies: {exc}")

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

    def _plan_tools(
        self, goals: List[str], available_inputs: Optional[List[str]] = None
    ) -> List[ToolDefinition]:
        """Plan a sequence of tools to satisfy goals using the dependency graph.

        Delegates to ToolRegistrar for tool planning.
        """
        return self.tool_registrar.plan_tools(goals, available_inputs)

    def _goal_hints_for_message(self, user_message: str) -> List[str]:
        """Infer planning goals from the user request.

        Delegates to ToolRegistrar for goal inference.
        """
        return self.tool_registrar.infer_goals_from_message(user_message)

    def _load_tool_configurations(self) -> None:
        """Load tool configurations from profiles.yaml.

        Loads tool enable/disable states from the 'tools' section in profiles.yaml.
        Expected format:

        tools:
          enabled:
            - read_file
            - write_file
            - execute_bash
          disabled:
            - code_review
            - security_scan

        Or:

        tools:
          code_review:
            enabled: false
          security_scan:
            enabled: false
        """
        try:
            tool_config = self.settings.load_tool_config()
            if not tool_config:
                return

            # Get all registered tool names for validation
            registered_tools = {tool.name for tool in self.tools.list_tools(only_enabled=False)}

            # Get critical tools dynamically from registry (priority=Priority.CRITICAL)
            core_tools = get_critical_tools(self.tools)

            # Format 1: Lists of enabled/disabled tools
            if "enabled" in tool_config:
                enabled_tools = tool_config.get("enabled", [])

                # Validate tool names
                invalid_tools = [t for t in enabled_tools if t not in registered_tools]
                if invalid_tools:
                    logger.warning(
                        f"Configuration contains invalid tool names in 'enabled' list: {', '.join(invalid_tools)}. "
                        f"Available tools: {', '.join(sorted(registered_tools))}"
                    )

                # Check if core tools are included (use dynamically discovered core tools)
                missing_core = core_tools - set(enabled_tools)
                if missing_core:
                    logger.warning(
                        f"'enabled' list is missing recommended core tools: {', '.join(missing_core)}. "
                        f"This may limit agent functionality."
                    )

                # First disable all tools
                for tool in self.tools.list_tools(only_enabled=False):
                    self.tools.disable_tool(tool.name)
                # Then enable only the specified ones
                for tool_name in enabled_tools:
                    if tool_name in registered_tools:
                        self.tools.enable_tool(tool_name)

            if "disabled" in tool_config:
                disabled_tools = tool_config.get("disabled", [])

                # Validate tool names
                invalid_tools = [t for t in disabled_tools if t not in registered_tools]
                if invalid_tools:
                    logger.warning(
                        f"Configuration contains invalid tool names in 'disabled' list: {', '.join(invalid_tools)}. "
                        f"Available tools: {', '.join(sorted(registered_tools))}"
                    )

                # Warn if disabling core tools
                disabled_core = core_tools & set(disabled_tools)
                if disabled_core:
                    logger.warning(
                        f"Disabling core tools: {', '.join(disabled_core)}. "
                        f"This may limit agent functionality."
                    )

                for tool_name in disabled_tools:
                    if tool_name in registered_tools:
                        self.tools.disable_tool(tool_name)

            # Format 2: Individual tool settings
            for tool_name, config in tool_config.items():
                if isinstance(config, dict) and "enabled" in config:
                    if tool_name not in registered_tools:
                        logger.warning(
                            f"Configuration contains invalid tool name: '{tool_name}'. "
                            f"Available tools: {', '.join(sorted(registered_tools))}"
                        )
                        continue

                    if config["enabled"]:
                        self.tools.enable_tool(tool_name)
                    else:
                        self.tools.disable_tool(tool_name)
                        # Warn if disabling core tools
                        if tool_name in core_tools:
                            logger.warning(
                                f"Disabling core tool '{tool_name}'. This may limit agent functionality."
                            )

            # Log tool states
            disabled_tools = [
                name for name, enabled in self.tools.get_tool_states().items() if not enabled
            ]
            if disabled_tools:
                logger.info(f"Disabled tools: {', '.join(sorted(disabled_tools))}")

            # Log enabled tool count
            enabled_count = sum(1 for enabled in self.tools.get_tool_states().values() if enabled)
            logger.info(f"Enabled tools: {enabled_count}/{len(registered_tools)}")

        except Exception as e:
            logger.warning(f"Failed to load tool configurations: {e}")

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

    def _ensure_system_message(self) -> None:
        """Ensure the system prompt is included once at the start of the conversation."""
        self.conversation.ensure_system_prompt()
        self._system_added = True

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
        self._ensure_system_message()
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

            # Get response from provider
            response = await self.provider.chat(
                messages=self.messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=tools,
                **provider_kwargs,
            )

            # Add assistant response to history if has content
            if response.content:
                self.add_message("assistant", response.content)

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

    def _handle_compaction(self, user_message: str) -> Optional[StreamChunk]:
        """Perform proactive compaction if enabled."""
        if not (hasattr(self, "_context_compactor") and self._context_compactor):
            return None

        compaction_action = self._context_compactor.check_and_compact(current_query=user_message)
        if not compaction_action.action_taken:
            return None

        logger.info(
            f"Proactive compaction: {compaction_action.trigger.value}, "
            f"removed {compaction_action.messages_removed} messages, "
            f"freed {compaction_action.chars_freed:,} chars"
        )
        chunk: Optional[StreamChunk] = None
        if compaction_action.messages_removed > 0:
            chunk = StreamChunk(
                content=(
                    f"\n[context] Proactively compacted history "
                    f"({compaction_action.messages_removed} messages, "
                    f"{compaction_action.chars_freed:,} chars freed).\n"
                )
            )
            # Inject context reminder about compacted content
            self._conversation_controller.inject_compaction_context()

        return chunk

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
                        sanitized = self._sanitize_response(response.content)
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
                    sanitized = self._sanitize_response(response.content)
                    if sanitized:
                        self.add_message("assistant", sanitized)
                        chunk = StreamChunk(content=sanitized)
            except Exception as e:
                logger.warning(f"Final response generation failed: {e}")
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
        TaskType,
        Any,
        int,
    ]:
        """Prepare streaming state and return commonly used values."""
        # Initialize cancellation support
        self._cancel_event = asyncio.Event()
        self._is_streaming = True

        # Track performance metrics using StreamMetrics
        stream_metrics = self._init_stream_metrics()
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

        self._ensure_system_message()
        self.tool_calls_used = 0
        self.observed_files = []
        self.executed_tools = []
        self.failed_tool_signatures = set()

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

        # Intelligent pipeline pre-request hook: get Q-learning recommendations
        # This enables RL-based mode transitions and optimal tool budget selection
        intelligent_context = await self._prepare_intelligent_request(
            task=user_message,
            task_type=unified_task_type.value,
        )
        if intelligent_context:
            # Inject optimized system prompt if provided
            if intelligent_context.get("system_prompt_addition"):
                self.add_message("system", intelligent_context["system_prompt_addition"])
                logger.debug("Injected intelligent pipeline optimized prompt")

        # Get exploration iterations from unified tracker (replaces TASK_CONFIGS lookup)
        max_exploration_iterations = self.unified_tracker.max_exploration_iterations

        # Task prep: hints, complexity, reminders
        task_classification, complexity_tool_budget = self._prepare_task(
            user_message, unified_task_type
        )

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
        ctx.is_analysis_task = task_keywords["is_analysis_task"]
        ctx.is_action_task = task_keywords["is_action_task"]
        ctx.needs_execution = task_keywords["needs_execution"]
        ctx.coarse_task_type = task_keywords["coarse_task_type"]

        # Set goals for tool selection
        ctx.goals = self._goal_hints_for_message(user_message)

        # Sync tool tracking from orchestrator to context
        ctx.tool_budget = self.tool_budget
        ctx.tool_calls_used = self.tool_calls_used

        return ctx

    def _prepare_task(self, user_message: str, unified_task_type: TaskType) -> tuple[Any, int]:
        """Prepare task-specific guidance and budget adjustments."""
        # Inject task-specific prompt hint for better guidance
        task_hint = get_task_type_hint(unified_task_type.value)
        if task_hint:
            self.add_message("system", task_hint.strip())
            logger.debug(f"Injected task hint for task type: {unified_task_type.value}")

        # Classify task complexity and adjust tool budget
        task_classification = self.task_classifier.classify(user_message)
        complexity_tool_budget = DEFAULT_BUDGETS.get(task_classification.complexity, 15)
        if task_classification.complexity == TaskComplexity.SIMPLE:
            # Override with simpler budget for simple tasks
            current_max = self.unified_tracker.config.get("max_total_iterations", 50)
            new_max = min(complexity_tool_budget, current_max)
            self.unified_tracker.set_tool_budget(new_max)
            logger.info(
                f"Task complexity: {task_classification.complexity.value}, "
                f"adjusted max_iterations to {complexity_tool_budget}"
            )
        elif task_classification.complexity == TaskComplexity.GENERATION:
            # Generation tasks should complete in 1-2 tool calls
            current_max = self.unified_tracker.config.get("max_total_iterations", 50)
            new_max = min(complexity_tool_budget + 1, current_max)
            self.unified_tracker.set_tool_budget(new_max)
            logger.info(
                f"Generation task detected, limiting iterations to {complexity_tool_budget + 1}"
            )
        else:
            logger.info(
                f"Task complexity: {task_classification.complexity.value}, "
                f"confidence: {task_classification.confidence:.2f}"
            )

        # Update reminder manager with task complexity and hint
        self.reminder_manager.update_state(
            task_complexity=task_classification.complexity.value,
            task_hint=task_classification.prompt_hint,
            tool_budget=complexity_tool_budget,
        )

        return task_classification, complexity_tool_budget

    def _apply_intent_guard(self, user_message: str) -> None:
        """Detect intent and inject prompt guards for read-only tasks."""
        intent_result = self.intent_detector.detect(user_message)
        self._current_intent = intent_result.intent
        if intent_result.intent in (ActionIntent.DISPLAY_ONLY, ActionIntent.READ_ONLY):
            if intent_result.prompt_guard:
                self.add_message("system", intent_result.prompt_guard.strip())
                logger.info(f"Intent: {intent_result.intent.value}, injected prompt guard")
        elif intent_result.intent == ActionIntent.WRITE_ALLOWED:
            logger.info("Intent: write_allowed, no prompt guard needed")

    def _apply_task_guidance(
        self,
        user_message: str,
        unified_task_type: TaskType,
        is_analysis_task: bool,
        is_action_task: bool,
        needs_execution: bool,
        max_exploration_iterations: int,
    ) -> None:
        """Apply guidance and budget tweaks for analysis/action tasks."""
        if is_analysis_task:
            analysis_temp = min(self.temperature + 0.2, 1.0)
            logger.info(
                f"Analysis task: increasing temperature {self.temperature:.1f} -> {analysis_temp:.1f}"
            )
            self.temperature = analysis_temp

            self.add_message(
                "system",
                "ANALYSIS APPROACH: Work through the codebase one module at a time. "
                "For each module: 1) List its files, 2) Read 2-3 key files, 3) Note observations. "
                "After examining 3-4 modules, provide your summary. Keep responses concise.",
            )

            original_budget = self.tool_budget
            self.tool_budget = max(self.tool_budget, 200)
            if self.tool_budget != original_budget:
                logger.info(
                    f"Analysis task: increased tool_budget from {original_budget} to {self.tool_budget}"
                )

            self.add_message(
                "system",
                "This is an ANALYSIS task requiring thorough exploration of the codebase. "
                "You MUST systematically examine multiple modules and files using tools like "
                "read_file, list_directory, and code_search. "
                "DO NOT stop after examining just a few files. "
                "Continue using tools until you have gathered comprehensive information about "
                "all major components of the codebase. "
                "Only provide your final analysis AFTER you have examined all relevant modules.",
            )

        if is_action_task:
            logger.info(
                f"Detected action-oriented task - allowing up to {max_exploration_iterations} exploration iterations"
            )

            if needs_execution:
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
            planned_tools = self._plan_tools(goals, available_inputs=available_inputs)

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
        tools = self.tool_selector.prioritize_by_stage(context_msg, tools)
        tools = self._filter_tools_by_intent(tools)
        return tools

    def _check_time_limit_with_handler(
        self,
        stream_ctx: StreamingChatContext,
    ) -> Optional[StreamChunk]:
        """Check time limit using the streaming handler.

        This delegates to the StreamingChatHandler for testable time limit logic.

        Args:
            stream_ctx: The streaming context

        Returns:
            StreamChunk if time limit reached, None otherwise
        """
        result = self._streaming_handler.check_time_limit(stream_ctx)
        if result:
            # Record Q-learning outcome
            self._record_intelligent_outcome(
                success=stream_ctx.total_accumulated_chars > 200,
                quality_score=0.4 if stream_ctx.total_accumulated_chars > 200 else 0.2,
                user_satisfied=False,
                completed=False,
            )
            # Return the chunk from the result
            if result.chunks:
                return result.chunks[0]
            return StreamChunk(
                content=f"\n\n⏱️ Session time limit reached. "
                "Providing summary of progress so far.\n",
                is_final=False,
            )
        return None

    def _check_iteration_limit_with_handler(
        self,
        stream_ctx: StreamingChatContext,
    ) -> Optional[StreamChunk]:
        """Check iteration limit using the streaming handler.

        Args:
            stream_ctx: The streaming context

        Returns:
            StreamChunk if iteration limit reached, None otherwise
        """
        result = self._streaming_handler.check_iteration_limit(stream_ctx)
        if result and result.chunks:
            return result.chunks[0]
        return None

    def _check_natural_completion_with_handler(
        self,
        stream_ctx: StreamingChatContext,
        has_tool_calls: bool,
        content_length: int,
    ) -> Optional[StreamChunk]:
        """Check for natural completion using the streaming handler.

        Args:
            stream_ctx: The streaming context
            has_tool_calls: Whether there are tool calls
            content_length: Length of current content

        Returns:
            StreamChunk if natural completion detected, None otherwise
        """
        result = self._streaming_handler.check_natural_completion(
            stream_ctx, has_tool_calls, content_length
        )
        if result:
            return StreamChunk(content="", is_final=True)
        return None

    def _handle_empty_response_with_handler(
        self,
        stream_ctx: StreamingChatContext,
    ) -> Tuple[Optional[StreamChunk], bool]:
        """Handle empty response using the streaming handler.

        Args:
            stream_ctx: The streaming context

        Returns:
            Tuple of (StreamChunk if threshold exceeded, should_force_completion flag)
        """
        result = self._streaming_handler.handle_empty_response(stream_ctx)
        if result and result.chunks:
            # Handler sets ctx.force_completion = True when threshold exceeded
            return result.chunks[0], stream_ctx.force_completion
        return None, False

    def _handle_blocked_tool_with_handler(
        self,
        stream_ctx: StreamingChatContext,
        tool_name: str,
        tool_args: Dict[str, Any],
        block_reason: str,
    ) -> StreamChunk:
        """Handle blocked tool call using the streaming handler.

        Args:
            stream_ctx: The streaming context
            tool_name: Name of blocked tool
            tool_args: Arguments that were passed
            block_reason: Reason for blocking

        Returns:
            StreamChunk with block notification
        """
        return self._streaming_handler.handle_blocked_tool_call(
            stream_ctx, tool_name, tool_args, block_reason
        )

    def _check_blocked_threshold_with_handler(
        self,
        stream_ctx: StreamingChatContext,
        all_blocked: bool,
    ) -> Optional[Tuple[StreamChunk, bool]]:
        """Check blocked threshold using the streaming handler.

        Args:
            stream_ctx: The streaming context
            all_blocked: Whether all tool calls were blocked

        Returns:
            Tuple of (chunk, should_clear_tools) if threshold exceeded, None otherwise
        """
        consecutive_limit = getattr(
            self.settings, "recovery_blocked_consecutive_threshold", 4
        )
        total_limit = getattr(self.settings, "recovery_blocked_total_threshold", 6)

        result = self._streaming_handler.check_blocked_threshold(
            stream_ctx, all_blocked, consecutive_limit, total_limit
        )
        if result:
            chunk = result.chunks[0] if result.chunks else StreamChunk(
                content="\n[loop] ⚠️ Multiple blocked attempts - forcing completion\n"
            )
            return (chunk, result.clear_tool_calls)
        return None

    def _handle_force_tool_execution_with_handler(
        self,
        stream_ctx: StreamingChatContext,
        mentioned_tools: List[str],
        force_message: Optional[str] = None,
    ) -> None:
        """Handle force tool execution using the streaming handler.

        Args:
            stream_ctx: The streaming context
            mentioned_tools: Tools that were mentioned but not executed
            force_message: Optional pre-crafted message to use instead of default
        """
        self._streaming_handler.handle_force_tool_execution(
            stream_ctx, mentioned_tools, force_message
        )
        self.unified_tracker.increment_turn()

    def _check_tool_budget_with_handler(
        self, stream_ctx: StreamingChatContext
    ) -> Optional[StreamChunk]:
        """Check tool budget and return warning chunk if approaching limit.

        Args:
            stream_ctx: The streaming context

        Returns:
            StreamChunk with warning if approaching limit, None otherwise
        """
        warning_threshold = getattr(
            self.settings, "tool_call_budget_warning_threshold", 250
        )
        result = self._streaming_handler.check_tool_budget(stream_ctx, warning_threshold)
        if result and result.chunks:
            return result.chunks[0]
        return None

    def _check_progress_with_handler(
        self, stream_ctx: StreamingChatContext
    ) -> bool:
        """Check progress and set force_completion if stuck.

        Args:
            stream_ctx: The streaming context

        Returns:
            True if force_completion was set, False otherwise
        """
        base_max = getattr(self.settings, "max_consecutive_tool_calls", 8)
        return self._streaming_handler.check_progress_and_force(stream_ctx, base_max)

    def _truncate_tool_calls_with_handler(
        self,
        tool_calls: List[Dict[str, Any]],
        stream_ctx: StreamingChatContext,
    ) -> List[Dict[str, Any]]:
        """Truncate tool calls to fit remaining budget.

        Args:
            tool_calls: List of tool calls
            stream_ctx: The streaming context

        Returns:
            Truncated list of tool calls
        """
        return self._streaming_handler.truncate_tool_calls(tool_calls, stream_ctx)

    def _handle_force_completion_with_handler(
        self,
        stream_ctx: StreamingChatContext,
    ) -> Optional[StreamChunk]:
        """Handle force completion when the model is stuck.

        Uses the unified_tracker to get stop decision info for determining
        whether this is a research loop or exploration limit.

        Args:
            stream_ctx: The streaming context

        Returns:
            Warning chunk if force_completion is set, None otherwise
        """
        if not stream_ctx.force_completion:
            return None

        # Get stop decision from unified tracker for context
        stop_decision = self.unified_tracker.should_stop()
        stop_reason_value = stop_decision.reason.value
        stop_hint = stop_decision.hint

        result = self._streaming_handler.handle_force_completion(
            stream_ctx, stop_reason_value, stop_hint
        )
        if result and result.chunks:
            return result.chunks[0]
        return None

    def _format_completion_metrics_with_handler(
        self,
        stream_ctx: StreamingChatContext,
        elapsed_time: float,
    ) -> str:
        """Format completion metrics using handler delegation.

        Args:
            stream_ctx: The streaming context
            elapsed_time: Elapsed time in seconds

        Returns:
            Formatted metrics line string
        """
        return self._streaming_handler.format_completion_metrics(stream_ctx, elapsed_time)

    def _format_budget_exhausted_metrics_with_handler(
        self,
        stream_ctx: StreamingChatContext,
        elapsed_time: float,
        time_to_first_token: Optional[float] = None,
    ) -> str:
        """Format budget exhausted metrics using handler delegation.

        Args:
            stream_ctx: The streaming context
            elapsed_time: Elapsed time in seconds
            time_to_first_token: Optional time to first token

        Returns:
            Formatted metrics line string
        """
        return self._streaming_handler.format_budget_exhausted_metrics(
            stream_ctx, elapsed_time, time_to_first_token
        )

    def _generate_tool_result_chunks_with_handler(
        self,
        result: Dict[str, Any],
    ) -> List[StreamChunk]:
        """Generate tool result chunks using handler delegation.

        Args:
            result: The tool execution result dictionary

        Returns:
            List of StreamChunks for the tool result
        """
        return self._streaming_handler.generate_tool_result_chunks(result)

    def _get_recovery_prompts_with_handler(
        self,
        stream_ctx: "StreamingChatContext",
    ) -> List[tuple[str, float]]:
        """Get recovery prompts using handler delegation.

        Args:
            stream_ctx: The streaming context

        Returns:
            List of (prompt, temperature) tuples for recovery attempts
        """
        # Get thinking mode settings
        has_thinking_mode = getattr(self.tool_calling_caps, "thinking_mode", False)
        thinking_prefix = getattr(self.tool_calling_caps, "thinking_disable_prefix", None)

        return self._streaming_handler.get_recovery_prompts(
            ctx=stream_ctx,
            base_temperature=self.temperature,
            has_thinking_mode=has_thinking_mode,
            thinking_disable_prefix=thinking_prefix,
        )

    def _should_use_tools_for_recovery_with_handler(
        self,
        stream_ctx: "StreamingChatContext",
        attempt: int,
    ) -> bool:
        """Check if tools should be enabled for recovery using handler delegation.

        Args:
            stream_ctx: The streaming context
            attempt: The recovery attempt number (1-indexed)

        Returns:
            True if tools should be enabled, False otherwise
        """
        return self._streaming_handler.should_use_tools_for_recovery(stream_ctx, attempt)

    def _get_recovery_fallback_message_with_handler(
        self,
        stream_ctx: "StreamingChatContext",
    ) -> str:
        """Get recovery fallback message using handler delegation.

        Args:
            stream_ctx: The streaming context

        Returns:
            Fallback message string
        """
        unique_resources = list(self.unified_tracker.unique_resources)
        return self._streaming_handler.get_recovery_fallback_message(
            stream_ctx, unique_resources
        )

    def _handle_loop_warning_with_handler(
        self,
        stream_ctx: "StreamingChatContext",
        warning_message: Optional[str],
    ) -> Optional[StreamChunk]:
        """Handle loop warning using handler delegation.

        Args:
            stream_ctx: The streaming context
            warning_message: The warning message from unified tracker

        Returns:
            StreamChunk with warning if applicable, None otherwise
        """
        return self._streaming_handler.handle_loop_warning(stream_ctx, warning_message)

    def _generate_tool_start_chunk_with_handler(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        status_msg: str,
    ) -> StreamChunk:
        """Generate tool start chunk using handler delegation.

        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments
            status_msg: Status message to display

        Returns:
            StreamChunk with tool start metadata
        """
        return self._streaming_handler.generate_tool_start_chunk(
            tool_name, tool_args, status_msg
        )

    def _get_budget_exhausted_chunks_with_handler(
        self,
        stream_ctx: "StreamingChatContext",
    ) -> List[StreamChunk]:
        """Get budget exhausted warning chunks using handler delegation.

        Args:
            stream_ctx: The streaming context

        Returns:
            List of StreamChunks for budget exhausted warning
        """
        return self._streaming_handler.get_budget_exhausted_chunks(stream_ctx)

    def _generate_thinking_status_chunk_with_handler(self) -> StreamChunk:
        """Generate thinking status chunk using handler delegation.

        Returns:
            StreamChunk with thinking status metadata
        """
        return self._streaming_handler.generate_thinking_status_chunk()

    def _generate_budget_error_chunk_with_handler(self) -> StreamChunk:
        """Generate budget limit error chunk using handler delegation.

        Returns:
            StreamChunk with budget limit error message
        """
        return self._streaming_handler.generate_budget_error_chunk()

    def _generate_force_response_error_chunk_with_handler(self) -> StreamChunk:
        """Generate force response error chunk using handler delegation.

        Returns:
            StreamChunk with force response error message
        """
        return self._streaming_handler.generate_force_response_error_chunk()

    def _generate_final_marker_chunk_with_handler(self) -> StreamChunk:
        """Generate final marker chunk using handler delegation.

        Returns:
            StreamChunk with is_final=True
        """
        return self._streaming_handler.generate_final_marker_chunk()

    def _generate_metrics_chunk_with_handler(
        self, metrics_line: str, is_final: bool = False, prefix: str = "\n\n"
    ) -> StreamChunk:
        """Generate metrics display chunk using handler delegation.

        Args:
            metrics_line: The formatted metrics line
            is_final: Whether this is the final chunk
            prefix: Prefix before metrics line (default: double newline)

        Returns:
            StreamChunk with formatted metrics content
        """
        return self._streaming_handler.generate_metrics_chunk(
            metrics_line, is_final=is_final, prefix=prefix
        )

    def _generate_content_chunk_with_handler(
        self, content: str, is_final: bool = False, suffix: str = ""
    ) -> StreamChunk:
        """Generate content chunk using handler delegation.

        Args:
            content: The sanitized content to display
            is_final: Whether this is the final chunk
            suffix: Optional suffix to append

        Returns:
            StreamChunk with content and optional suffix
        """
        return self._streaming_handler.generate_content_chunk(
            content, is_final=is_final, suffix=suffix
        )

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
                is_enabled = self.tools.is_tool_enabled(name)
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
                self._record_first_token()
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
            # Caller will set force_completion on return

        # Store total_tokens back to context for later use
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
        if chunk.content and self._is_garbage_content(chunk.content):
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
        async for chunk in self._stream_chat_impl(user_message):
            yield chunk

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

        goals = self._goal_hints_for_message(user_message)

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
            cancellation_chunk = self._handle_cancellation(stream_ctx.last_quality_score)
            if cancellation_chunk:
                yield cancellation_chunk
                return

            compaction_chunk = self._handle_compaction(user_message)
            if compaction_chunk:
                yield compaction_chunk

            # Context state is maintained directly - no sync needed
            # force_completion and total_accumulated_chars are accessed via stream_ctx

            # Check session time limit using handler delegation (testable)
            time_limit_chunk = self._check_time_limit_with_handler(stream_ctx)
            if time_limit_chunk:
                yield time_limit_chunk
                # Handler already set stream_ctx.force_completion = True
                # Force the model to summarize (handler already added message)

            # Increment iteration count using context method (single source of truth)
            stream_ctx.increment_iteration()
            unique_resources = self.unified_tracker.unique_resources
            logger.debug(
                f"Iteration {stream_ctx.total_iterations}/{max_total_iterations}: "
                f"tool_calls_used={self.tool_calls_used}/{self.tool_budget}, "
                f"unique_resources={len(unique_resources)}, "
                f"force_completion={stream_ctx.force_completion}"
            )

            # Use debug logger for incremental tracking
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

            full_content, tool_calls, _, garbage_detected = (
                await self._stream_provider_response(
                    tools=tools,
                    provider_kwargs=provider_kwargs,
                    stream_ctx=stream_ctx,
                )
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

            if full_content:
                # Sanitize response to remove malformed patterns from local models
                sanitized = self._sanitize_response(full_content)
                if sanitized:
                    self.add_message("assistant", sanitized)
                else:
                    # If sanitization removed everything, use stripped markup as fallback
                    plain_text = self._strip_markup(full_content)
                    if plain_text:
                        self.add_message("assistant", plain_text)

                # Check if model mentioned tools but didn't execute them
                # Store for use in continuation action decision
                if not tool_calls:
                    mentioned_tools_detected = _detect_mentioned_tools(full_content)
                    if mentioned_tools_detected:
                        tools_str = ", ".join(mentioned_tools_detected)
                        logger.warning(
                            f"Model mentioned tool(s) [{tools_str}] but did not execute them. "
                            "This may indicate the model is hallucinating tool calls."
                        )
            elif not tool_calls:
                # No content and no tool calls - check for natural completion using handler
                # Handler checks if substantial content was already provided
                final_chunk = self._check_natural_completion_with_handler(
                    stream_ctx, has_tool_calls=False, content_length=0
                )
                if final_chunk:
                    yield final_chunk
                    return

                # No substantial content yet - attempt aggressive recovery
                logger.warning("Model returned empty response - attempting aggressive recovery")

                # Track empty responses using handler delegation for testable logic
                # Handler tracks consecutive empty responses and forces summary if threshold exceeded
                recovery_chunk, should_force = self._handle_empty_response_with_handler(stream_ctx)
                if recovery_chunk:
                    yield recovery_chunk
                    # CRITICAL: Handler already set stream_ctx.force_completion if needed
                    # The should_force flag confirms the handler's decision
                    continue

                # Get recovery prompts using handler delegation (testable)
                # Handler handles thinking mode prefix and task-aware prompts
                recovery_prompts = self._get_recovery_prompts_with_handler(stream_ctx)

                recovery_success = False
                for attempt, (prompt, temp) in enumerate(recovery_prompts, 1):
                    logger.info(f"Recovery attempt {attempt}/3 with temp={temp:.1f}")

                    # Create temporary message list to avoid polluting conversation history
                    # Include only recent context (last 5 exchanges) to reduce token load
                    recent_messages = (
                        self.messages[-10:] if len(self.messages) > 10 else self.messages[:]
                    )
                    recovery_messages = recent_messages + [Message(role="user", content=prompt)]

                    # Use handler to decide if tools should be enabled (testable)
                    use_tools = self._should_use_tools_for_recovery_with_handler(stream_ctx, attempt)
                    recovery_tools = tools if use_tools else None

                    try:
                        response = await self.provider.chat(
                            messages=recovery_messages,
                            model=self.model,
                            temperature=temp,
                            max_tokens=min(self.max_tokens, 1024),  # Limit output for recovery
                            tools=recovery_tools,
                        )

                        # Check for tool calls in recovery response
                        if use_tools and response and response.tool_calls:
                            logger.info(
                                f"Recovery attempt {attempt}: model made {len(response.tool_calls)} tool call(s)"
                            )
                            # Re-inject the recovery prompt into conversation and let main loop handle
                            self.add_message("user", prompt)
                            if response.content:
                                self.add_message("assistant", response.content)
                            # Set tool_calls for main loop to process
                            tool_calls = response.tool_calls
                            recovery_success = True
                            # Don't yield final chunk - let main loop continue with tool execution
                            break

                        if response and response.content:
                            logger.debug(
                                f"Recovery attempt {attempt}: got {len(response.content)} chars"
                            )
                            sanitized = self._sanitize_response(response.content)
                            if sanitized and len(sanitized) > 20:
                                self.add_message("assistant", sanitized)
                                yield self._generate_content_chunk_with_handler(
                                    sanitized, is_final=True
                                )
                                recovery_success = True
                                break
                            elif response.content and len(response.content) > 20:
                                # Use raw if sanitization failed but content exists
                                self.add_message("assistant", response.content)
                                yield self._generate_content_chunk_with_handler(
                                    response.content, is_final=True
                                )
                                recovery_success = True
                                break
                        else:
                            logger.debug(f"Recovery attempt {attempt}: empty response")
                    except Exception as exc:
                        exc_str = str(exc)
                        logger.warning(f"Recovery attempt {attempt} failed: {exc}")

                        # Check for rate limit errors and extract wait time
                        if "rate_limit" in exc_str.lower() or "429" in exc_str:
                            import re

                            # Try to extract "try again in X.XXs" or similar patterns
                            wait_match = re.search(
                                r"try again in (\d+(?:\.\d+)?)\s*s", exc_str, re.I
                            )
                            if wait_match:
                                wait_time = float(wait_match.group(1))
                                logger.info(
                                    f"Rate limited. Waiting {wait_time:.1f}s before retry..."
                                )
                                await asyncio.sleep(
                                    min(wait_time + 0.5, 30.0)
                                )  # Add 0.5s buffer, cap at 30s
                            else:
                                # Default exponential backoff for rate limits
                                backoff = min(2**attempt, 15)  # 2, 4, 8 seconds
                                logger.info(f"Rate limited. Waiting {backoff}s before retry...")
                                await asyncio.sleep(backoff)

                if recovery_success:
                    # If recovery produced tool_calls, continue the main loop to execute them
                    # Otherwise, we've already yielded the text response, so return
                    if not tool_calls:
                        return
                    # Fall through to tool execution with recovered tool_calls
                    logger.info(
                        f"Recovery produced {len(tool_calls)} tool call(s) - continuing main loop"
                    )
                else:
                    # All recovery attempts failed - get fallback message using handler (testable)
                    fallback_msg = self._get_recovery_fallback_message_with_handler(stream_ctx)
                    # Record outcome for Q-learning (fallback = partial failure)
                    self._record_intelligent_outcome(
                        success=False,
                        quality_score=0.3,  # Low quality since model didn't provide useful content
                        user_satisfied=False,
                        completed=False,
                    )
                    yield self._generate_content_chunk_with_handler(
                        fallback_msg, is_final=True
                    )
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
                # Update quality score for Q-learning outcome recording
                if quality_result:
                    new_score = quality_result.get("quality_score", stream_ctx.last_quality_score)
                    stream_ctx.update_quality_score(new_score)

            # Check for loop warning using handler delegation (testable)
            unified_loop_warning = self.unified_tracker.check_loop_warning()
            loop_warning_chunk = self._handle_loop_warning_with_handler(
                stream_ctx, unified_loop_warning
            )
            if loop_warning_chunk:
                logger.warning(f"UnifiedTaskTracker loop warning: {unified_loop_warning}")
                yield loop_warning_chunk
            else:
                # PRIMARY: Check UnifiedTaskTracker for stop decision (single source of truth)
                unified_should_force, unified_hint = self.unified_tracker.should_force_action()
                if unified_should_force and not stream_ctx.force_completion:
                    stream_ctx.force_completion = True
                    logger.info(
                        f"UnifiedTaskTracker forcing action: {unified_hint}, "
                        f"metrics={self.unified_tracker.get_metrics()}"
                    )

                logger.debug(f"After streaming pass, tool_calls = {tool_calls}")

                if not tool_calls:
                    # CRITICAL FIX: Yield content to UI immediately when there are no tool calls
                    # This ensures the user sees the model's response even if the loop continues
                    # for intent classification and action decisions
                    # Save content for intent classification before yielding
                    content_for_intent = full_content or ""
                    if full_content:
                        sanitized = self._sanitize_response(full_content)
                        if sanitized:
                            logger.debug(f"Yielding content to UI: {len(sanitized)} chars")
                            yield StreamChunk(content=sanitized)
                            # Track accumulated content - use context method for consistency
                            stream_ctx.accumulate_content(sanitized)
                            logger.debug(
                                f"Total accumulated content: {stream_ctx.total_accumulated_chars} chars"
                            )
                            # Clear full_content to prevent duplicate output later
                            full_content = ""

                    # Check if model intended to continue but didn't make a tool call
                    # Use semantic intent classification to determine continuation action
                    # NOTE: Use the LAST portion of the response for intent classification
                    # because asking_input patterns like "Would you like me to..." typically
                    # appear at the END of a long response
                    intent_text = content_for_intent
                    if len(intent_text) > 500:
                        intent_text = intent_text[-500:]
                    intent_result = self.intent_classifier.classify_intent_sync(intent_text)

                    logger.debug(
                        f"Intent classification: {intent_result.intent.name} "
                        f"(confidence={intent_result.confidence:.3f}, "
                        f"text_len={len(intent_text)}, "
                        f"top_matches={intent_result.top_matches[:3]})"
                    )

                    # Initialize tracking variables
                    if not hasattr(self, "_continuation_prompts"):
                        self._continuation_prompts = 0
                    if not hasattr(self, "_asking_input_prompts"):
                        self._asking_input_prompts = 0
                    if not hasattr(self, "_consecutive_blocked_attempts"):
                        self._consecutive_blocked_attempts = 0

                    # Check for response loop using UnifiedTaskTracker
                    # (detects when model keeps responding with similar text without tool calls)
                    is_repeated_response = self.unified_tracker.check_response_loop(
                        full_content or ""
                    )

                    # Use helper to determine what action to take
                    one_shot_mode = getattr(self.settings, "one_shot_mode", False)
                    action_result = self._determine_continuation_action(
                        intent_result=intent_result,
                        is_analysis_task=stream_ctx.is_analysis_task,
                        is_action_task=stream_ctx.is_action_task,
                        content_length=content_length,
                        full_content=full_content,
                        continuation_prompts=self._continuation_prompts,
                        asking_input_prompts=self._asking_input_prompts,
                        one_shot_mode=one_shot_mode,
                        mentioned_tools=mentioned_tools_detected,  # Pass hallucinated tool mentions
                    )

                    # Apply state updates from action result
                    if "continuation_prompts" in action_result.get("updates", {}):
                        self._continuation_prompts = action_result["updates"][
                            "continuation_prompts"
                        ]
                    if "asking_input_prompts" in action_result.get("updates", {}):
                        self._asking_input_prompts = action_result["updates"][
                            "asking_input_prompts"
                        ]
                    if action_result.get("set_final_summary_requested"):
                        self._final_summary_requested = True

                    action = action_result["action"]

                    # Override: If repeated response detected, force completion to prevent loop
                    if is_repeated_response and action in ("prompt_tool_call", "request_summary"):
                        action = "finish"
                        logger.info(
                            f"Continuation action: {action} - "
                            "Overriding to finish due to repeated response"
                        )
                    else:
                        logger.info(f"Continuation action: {action} - {action_result['reason']}")

                    skip_rest = False

                    # Handle action: continue_asking_input
                    if action == "continue_asking_input":
                        self.add_message("user", action_result["message"])
                        skip_rest = True

                    # Handle action: return_to_user
                    elif action == "return_to_user":
                        # Yield the accumulated content before returning
                        if full_content:
                            sanitized = self._sanitize_response(full_content)
                            if sanitized:
                                yield StreamChunk(content=sanitized)
                        yield StreamChunk(content="", is_final=True)
                        return

                    # Handle action: prompt_tool_call
                    # NOTE: Use "user" role instead of "system" because many models
                    # (especially Qwen, Ollama local models) don't handle mid-conversation
                    # system messages well - they expect system messages only at the start.
                    # Using "user" role ensures the continuation prompt is processed correctly.
                    elif action == "prompt_tool_call":
                        self.add_message("user", action_result["message"])
                        self.unified_tracker.increment_turn()
                        skip_rest = True

                    # Handle action: request_summary
                    elif action == "request_summary":
                        self.add_message("user", action_result["message"])
                        skip_rest = True

                    # Handle action: request_completion
                    elif action == "request_completion":
                        self.add_message("user", action_result["message"])
                        skip_rest = True

                    # Handle action: force_tool_execution (for hallucinated tool mentions)
                    # Uses handler delegation for testable attempt tracking and message injection
                    elif action == "force_tool_execution":
                        mentioned_tools = action_result.get("mentioned_tools", [])
                        force_message = action_result.get("message")
                        self._handle_force_tool_execution_with_handler(
                            stream_ctx, mentioned_tools, force_message
                        )
                        skip_rest = True

                    if skip_rest:
                        pass
                    else:
                        # Handle action: finish - No more tool calls requested
                        # Yield the accumulated content to the UI (was missing!)
                        if full_content:
                            sanitized = self._sanitize_response(full_content)
                            if sanitized:
                                yield StreamChunk(content=sanitized)

                        # Display performance metrics using handler delegation
                        elapsed_time = time.time() - stream_ctx.start_time
                        metrics_line = self._format_completion_metrics_with_handler(
                            stream_ctx, elapsed_time
                        )
                        yield self._generate_metrics_chunk_with_handler(metrics_line)
                        # Record outcome for Q-learning (normal completion = success)
                        self._record_intelligent_outcome(
                            success=True,
                            quality_score=stream_ctx.last_quality_score,
                            user_satisfied=True,
                            completed=True,
                        )
                        yield StreamChunk(content="", is_final=True)
                        return

                # Tool execution section - runs regardless of loop warning
                logger.debug(
                    f"Entering tool execution: tool_calls={len(tool_calls) if tool_calls else 0}, "
                    f"tool_calls_used={self.tool_calls_used}/{self.tool_budget}"
                )

                # Sync tool tracking to context for handler methods
                stream_ctx.tool_calls_used = self.tool_calls_used
                stream_ctx.tool_budget = self.tool_budget

                remaining = stream_ctx.get_remaining_budget()

                # Warn when approaching budget limit - uses handler delegation
                budget_warning = self._check_tool_budget_with_handler(stream_ctx)
                if budget_warning:
                    yield budget_warning

                if remaining <= 0:
                    # Use handler delegation for budget exhausted chunks (testable)
                    for chunk in self._get_budget_exhausted_chunks_with_handler(stream_ctx):
                        yield chunk
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
                            sanitized = self._sanitize_response(response.content)
                            if sanitized:
                                yield StreamChunk(content=sanitized + "\n")
                    except Exception as e:
                        logger.warning(f"Failed to generate final summary: {e}")
                        # Use handler delegation for budget error chunk (testable)
                        yield self._generate_budget_error_chunk_with_handler()

                    # Finalize and display performance metrics using handler delegation
                    final_metrics = self._finalize_stream_metrics()
                    elapsed_time = (
                        final_metrics.total_duration if final_metrics else time.time() - stream_ctx.start_time
                    )
                    ttft = final_metrics.time_to_first_token if final_metrics else None
                    metrics_line = self._format_budget_exhausted_metrics_with_handler(
                        stream_ctx, elapsed_time, ttft
                    )
                    # Record outcome for Q-learning (budget reached = partial success)
                    self._record_intelligent_outcome(
                        success=True,  # We provided a summary
                        quality_score=stream_ctx.last_quality_score,
                        user_satisfied=True,
                        completed=True,
                    )
                    yield StreamChunk(
                        content=f"\n{metrics_line}\n",
                        is_final=True,
                    )
                    return

                # Force final response after too many consecutive tool calls without output
                # This prevents endless tool call loops
                # Sync unique_resources to context for progress check
                stream_ctx.unique_resources = unique_resources

                # Check progress using handler delegation - will set force_completion if stuck
                self._check_progress_with_handler(stream_ctx)

                # Force completion if too many low-output iterations or research calls
                # Use handler delegation for message generation (testable)
                force_chunk = self._handle_force_completion_with_handler(stream_ctx)
                if force_chunk:
                    yield force_chunk

                    # Force a final response by calling provider WITHOUT tools
                    try:
                        response = await self.provider.chat(
                            messages=self.messages,
                            model=self.model,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            tools=None,  # No tools - force text response
                        )
                        if response and response.content:
                            sanitized = self._sanitize_response(response.content)
                            if sanitized:
                                self.add_message("assistant", sanitized)
                                yield StreamChunk(content=sanitized)
                    except Exception as e:
                        logger.warning(f"Error forcing final response: {e}")
                        # Use handler delegation for force response error chunk (testable)
                        yield self._generate_force_response_error_chunk_with_handler()
                    return  # Exit the loop after forcing final response

                # Guard against None tool_calls (can happen when model response has no tool calls
                # but continuation logic decided to continue the loop)
                # Truncate to remaining budget using handler delegation
                tool_calls = self._truncate_tool_calls_with_handler(tool_calls or [], stream_ctx)

                # Filter out tool calls that are blocked after loop warning
                # After warning, the same signature cannot be attempted again
                # Uses handler delegation for testable blocked tool processing
                filtered_tool_calls = []
                blocked_count = 0
                for tc in tool_calls:
                    tc_name = tc.get("name", "")
                    tc_args = tc.get("arguments", {})
                    block_reason = self.unified_tracker.is_blocked_after_warning(tc_name, tc_args)
                    if block_reason:
                        # Use handler to process blocked tool (records and adds feedback)
                        chunk = self._handle_blocked_tool_with_handler(
                            stream_ctx, tc_name, tc_args, block_reason
                        )
                        yield chunk
                        blocked_count += 1
                    else:
                        filtered_tool_calls.append(tc)

                # Initialize variables that may not be set if no tool calls
                tool_name = None
                tool_results = []

                # Check if we should force completion due to excessive blocking
                # Uses handler delegation for testable threshold checking
                all_blocked = blocked_count > 0 and not filtered_tool_calls
                threshold_result = self._check_blocked_threshold_with_handler(
                    stream_ctx, all_blocked
                )
                if threshold_result:
                    chunk, should_clear = threshold_result
                    yield chunk
                    if should_clear:
                        filtered_tool_calls = []

                tool_calls = filtered_tool_calls

                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "tool")
                    tool_args = tool_call.get("arguments", {})
                    # Generate user-friendly status message with relevant context
                    status_msg = self._get_tool_status_message(tool_name, tool_args)
                    # Emit structured tool_start event using handler delegation (testable)
                    yield self._generate_tool_start_chunk_with_handler(
                        tool_name, tool_args, status_msg
                    )

                tool_results = await self._handle_tool_calls(tool_calls)
                # CRITICAL FIX: Increment tool_calls_used counter to prevent infinite loops
                self.tool_calls_used += len(tool_calls)

                # Generate tool result and preview chunks using handler delegation
                for result in tool_results:
                    tool_name = result.get("name", "tool")
                    for chunk in self._generate_tool_result_chunks_with_handler(result):
                        yield chunk

                # Use handler delegation for thinking status chunk (testable)
                yield self._generate_thinking_status_chunk_with_handler()

                # Update reminder manager state and inject consolidated reminder if needed
                # This replaces the previous per-tool-call evidence injection with smart throttling
                self.reminder_manager.update_state(
                    observed_files=set(self.observed_files) if self.observed_files else set(),
                    executed_tool=tool_name,
                    tool_calls=self.tool_calls_used,
                )

                # Get consolidated reminder (only returns content when injection is due)
                reminder = self.reminder_manager.get_consolidated_reminder()
                if reminder:
                    self.add_message("system", reminder)

                # Update context message for next iteration (uses helper method)
                stream_ctx.update_context_message(full_content or user_message)

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

            except Exception as e:
                last_error = str(e)
                if attempt < max_attempts - 1:
                    delay = min(base_delay * (2**attempt), max_delay)
                    logger.warning(
                        f"Tool '{tool_name}' raised exception (attempt {attempt + 1}/{max_attempts}): {e}. "
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
                continue

            # Validate tool name format (reject hallucinated/malformed names)
            if not self._is_valid_tool_name(tool_name):
                self.console.print(
                    f"[yellow]⚠ Skipping invalid/hallucinated tool name: {tool_name}[/]"
                )
                continue

            # Resolve legacy/alias names to canonical form before checks
            try:
                from victor.tools.decorators import resolve_tool_name

                canonical_tool_name = resolve_tool_name(tool_name)
            except Exception:
                canonical_tool_name = tool_name

            # Skip unknown tools immediately (no retries, no budget cost)
            if not self.tools.is_tool_enabled(canonical_tool_name):
                # Log original and canonical names to aid debugging in tests
                self.console.print(
                    f"[yellow]⚠ Skipping unknown or disabled tool: {tool_name} (resolved: {canonical_tool_name})[/]"
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

            # Apply adapter-based normalization for missing required parameters
            # This handles provider-specific quirks like Gemini omitting 'path' for list_directory
            normalized_args = self.tool_adapter.normalize_arguments(normalized_args, tool_name)

            # Infer operation from alias for git tool (e.g., git_log → git with operation=log)
            normalized_args = self._infer_git_operation(
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
            exec_result = await self.tool_executor.execute(
                tool_name=tool_name,
                arguments=normalized_args,
                context=context,
                skip_normalization=True,
            )
            success = exec_result.success
            error_msg = exec_result.error

            # Update counters and tracking
            self.tool_calls_used += 1
            self.executed_tools.append(tool_name)
            if tool_name == "read" and "path" in normalized_args:
                self.observed_files.append(str(normalized_args.get("path")))

            # Reset continuation prompts counter on successful tool call
            # This allows the model to get fresh continuation prompts if it pauses again
            if hasattr(self, "_continuation_prompts") and self._continuation_prompts > 0:
                logger.debug(
                    f"Resetting continuation prompts counter (was {self._continuation_prompts}) after successful tool call"
                )
                self._continuation_prompts = 0

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
        self.conversation.clear()
        self._system_added = False

        # Reset session-specific state to prevent memory leaks
        self.tool_calls_used = 0
        self.failed_tool_signatures.clear()
        self.observed_files.clear()
        self.executed_tools.clear()
        self._consecutive_blocked_attempts = 0
        self._total_blocked_attempts = 0

        # Reset conversation state machine
        if hasattr(self, "conversation_state"):
            self.conversation_state.reset()

        # Reset context reminder manager
        if hasattr(self, "reminder_manager"):
            self.reminder_manager.reset()

        # Reset metrics collector
        if hasattr(self, "_metrics_collector"):
            self._metrics_collector.reset_stats()

        # Reset optimization components for clean session
        if hasattr(self, "_context_compactor") and self._context_compactor:
            self._context_compactor.reset_statistics()

        if hasattr(self, "_sequence_tracker") and self._sequence_tracker:
            self._sequence_tracker.clear_history()

        if hasattr(self, "_usage_analytics") and self._usage_analytics:
            # End current session if active, start fresh
            if self._usage_analytics._current_session is not None:
                self._usage_analytics.end_session()
            self._usage_analytics.start_session()

        logger.debug("Conversation and session state reset (including optimization components)")

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

        Args:
            session_id: ID of the session to recover

        Returns:
            True if session was recovered successfully
        """
        if not self.memory_manager:
            logger.warning("Memory manager not enabled, cannot recover session")
            return False

        try:
            session = self.memory_manager.get_session(session_id)
            if not session:
                logger.warning("Session not found: %s", session_id)
                return False

            # Update current session
            self._memory_session_id = session_id

            # Restore messages to in-memory conversation
            self.conversation.clear()
            for msg in session.messages:
                provider_msg = msg.to_provider_format()
                self.conversation.add_message(
                    role=provider_msg["role"],
                    content=provider_msg["content"],
                )

            logger.info(
                f"Recovered session {session_id[:8]}... " f"with {len(session.messages)} messages"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to recover session {session_id}: {e}")
            return False

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

        Returns:
            Dictionary with session statistics
        """
        if not self.memory_manager or not self._memory_session_id:
            return {
                "enabled": False,
                "session_id": None,
                "message_count": len(self.messages),
            }

        try:
            session = self.memory_manager.get_session(self._memory_session_id)
            if not session:
                return {
                    "enabled": True,
                    "session_id": self._memory_session_id,
                    "error": "Session not found",
                }

            total_tokens = sum(m.token_count for m in session.messages)
            return {
                "enabled": True,
                "session_id": self._memory_session_id,
                "message_count": len(session.messages),
                "total_tokens": total_tokens,
                "max_tokens": session.max_tokens,
                "reserved_tokens": session.reserved_tokens,
                "available_tokens": session.max_tokens - session.reserved_tokens - total_tokens,
                "project_path": session.project_path,
                "provider": session.provider,
                "model": session.model,
            }
        except Exception as e:
            logger.warning(f"Failed to get session stats: {e}")
            return {"enabled": True, "session_id": self._memory_session_id, "error": str(e)}

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
        logger.info("Shutting down AgentOrchestrator...")

        # Cancel all background tasks first
        if self._background_tasks:
            logger.debug("Cancelling %d background task(s)...", len(self._background_tasks))
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()

            # Wait for all tasks to complete cancellation
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
            logger.debug("Background tasks cancelled")

        # Log final analytics
        self.usage_logger.log_event(
            "session_end",
            {
                "tool_calls_used": self.tool_calls_used,
                "total_messages": self.conversation.message_count(),
            },
        )

        # Close provider connection
        if self.provider:
            try:
                await self.provider.close()
                logger.debug("Provider connection closed")
            except Exception as e:
                logger.warning("Error closing provider: %s", str(e))

        # Stop code execution manager (cleans up Docker containers)
        if hasattr(self, "code_manager") and self.code_manager:
            try:
                self.code_manager.stop()
                logger.debug("Code execution manager stopped")
            except Exception as e:
                logger.warning(f"Error stopping code manager: {e}")

        # Close semantic selector
        if self.semantic_selector:
            try:
                await self.semantic_selector.close()
                logger.debug("Semantic selector closed")
            except Exception as e:
                logger.warning("Error closing semantic selector: %s", str(e))

        # Signal shutdown to EmbeddingService singleton
        # This prevents post-shutdown embedding operations
        try:
            from victor.embeddings.service import EmbeddingService

            if EmbeddingService._instance is not None:
                EmbeddingService._instance.shutdown()
                logger.debug("EmbeddingService shutdown signaled")
        except Exception as e:
            logger.debug(f"Error signaling EmbeddingService shutdown: {e}")

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

        Args:
            provider: Target provider name
            model: Optional specific model
            on_switch: Optional callback(provider, model) after switch
        """
        await self._provider_manager.switch_provider(provider, model)
        # Sync instance attributes
        self.provider = self._provider_manager.provider
        self.model = self._provider_manager.model
        self.provider_name = self._provider_manager.provider_name

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

    def get_enabled_tools(self) -> Set[str]:
        """Get currently enabled tool names (protocol method).

        Returns:
            Set of enabled tool names for this session
        """
        # Check for framework-set tools first
        if hasattr(self, "_enabled_tools") and self._enabled_tools:
            return self._enabled_tools
        # Fall back to all available tools
        return self.get_available_tools()

    def set_enabled_tools(self, tools: Set[str]) -> None:
        """Set which tools are enabled for this session (protocol method).

        Args:
            tools: Set of tool names to enable
        """
        self._enabled_tools = tools

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled (protocol method).

        Args:
            tool_name: Name of tool to check

        Returns:
            True if tool is enabled
        """
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

    def cancel(self) -> None:
        """Cancel any in-progress operation (protocol method).

        Alias for request_cancellation() for protocol conformance.
        """
        self.request_cancellation()

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

        # Create provider instance using registry
        provider = ProviderRegistry.create(profile.provider, **provider_settings)

        return cls(
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
