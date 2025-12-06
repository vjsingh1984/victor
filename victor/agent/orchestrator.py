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

Remaining Orchestrator Responsibilities:
- Provider/model initialization and switching
- Tool registration and MCP integration
- High-level chat flow coordination
- Configuration loading and validation

Future Extraction Candidates:
- ProviderManager: Provider initialization, switching, health checks (lines ~1101-1287)
- ToolRegistrar: Tool registration, plugins, MCP setup (lines ~1527-1775)
- MCPIntegration: MCP server setup and registry management (lines ~1601-1676)

Note: Keep orchestrator as a thin facade. New logic should go into
appropriate extracted components, not added here.
"""

import ast
import asyncio
import importlib
import inspect
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from rich.console import Console

from victor.agent.argument_normalizer import ArgumentNormalizer, NormalizationStrategy
from victor.agent.message_history import MessageHistory
from victor.agent.conversation_memory import (
    ConversationStore,
    MessageRole,
)
from victor.agent.conversation_embedding_store import (
    ConversationEmbeddingStore,
)
from victor.agent.conversation_state import ConversationStateMachine, ConversationStage
from victor.agent.debug_logger import get_debug_logger
from victor.agent.action_authorizer import ActionAuthorizer, ActionIntent
from victor.agent.loop_detector import ProgressConfig, LoopDetector, TaskType
from victor.agent.prompt_builder import SystemPromptBuilder, get_task_type_hint
from victor.agent.response_sanitizer import ResponseSanitizer
from victor.agent.search_router import SearchRouter
from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity, DEFAULT_BUDGETS
from victor.agent.context_reminder import create_reminder_manager
from victor.agent.stream_handler import StreamMetrics
from victor.agent.milestone_monitor import TaskMilestoneMonitor, TASK_CONFIGS
from victor.agent.unified_task_tracker import (
    UnifiedTaskTracker,
    TaskType as UnifiedTaskType,
)
from victor.embeddings.task_classifier import TaskType as ClassifierTaskType

# New decomposed components (facades for orchestrator responsibilities)
from victor.agent.conversation_controller import (
    ConversationController,
    ConversationConfig,
    ContextMetrics,
    CompactionStrategy,
)
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

from victor.agent.tool_selection import (
    CORE_TOOLS,
    ToolSelector,
)
from victor.agent.tool_calling import (
    ToolCallingAdapterRegistry,
    ToolCallParseResult,
)
from victor.agent.tool_executor import ToolExecutor, ValidationMode
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
from victor.tools.batch_processor_tool import set_batch_processor_config
from victor.tools.base import CostTier, ToolRegistry
from victor.tools.code_executor_tool import CodeExecutionManager
from victor.tools.code_review_tool import set_code_review_config
from victor.tools.dependency_graph import ToolDependencyGraph
from victor.tools.git_tool import set_git_provider
from victor.tools.mcp_bridge_tool import configure_mcp_client, get_mcp_tool_definitions
from victor.tools.plugin_registry import ToolPluginRegistry
from victor.tools.semantic_selector import SemanticToolSelector
from victor.tools.web_search_tool import set_web_search_provider, set_web_tool_defaults
from victor.embeddings.intent_classifier import IntentClassifier, IntentType
from victor.workflows.base import WorkflowRegistry
from victor.workflows.new_feature_workflow import NewFeatureWorkflow

logger = logging.getLogger(__name__)

# Tools with progressive parameters - different params = progress, not a loop
# Format: tool_name -> list of param names that indicate progress
PROGRESSIVE_TOOLS = {
    "read_file": ["path", "offset", "limit"],
    "code_search": ["query", "directory"],
    "semantic_code_search": ["query", "directory"],
    "list_directory": ["path", "recursive"],
    "execute_bash": ["command"],
    "git": ["operation", "files", "branch"],
    "http_request": ["url", "method"],
    "web_search": ["query"],
    "web_summarize": ["query"],
    "web_fetch": ["url"],
}


class AgentOrchestrator:
    """Orchestrates agent interactions, tool execution, and provider communication."""

    # Known provider context windows (in tokens)
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

        # Try to get context window from provider
        context_tokens = None
        if hasattr(provider, "get_context_window"):
            try:
                context_tokens = provider.get_context_window(model)
            except Exception:
                pass

        # Fall back to provider defaults
        if context_tokens is None:
            provider_name = getattr(provider, "name", "").lower()
            context_tokens = AgentOrchestrator.PROVIDER_CONTEXT_WINDOWS.get(provider_name, 100000)

        # Convert tokens to chars: ~3.5 chars per token with 80% safety margin
        max_chars = int(context_tokens * 3.5 * 0.8)
        logger.info(f"Model context: {context_tokens:,} tokens -> {max_chars:,} chars limit")
        return max_chars

    @staticmethod
    def _map_unified_to_legacy_task_type(unified_type: "UnifiedTaskType") -> "ClassifierTaskType":
        """Map UnifiedTaskTracker TaskType to task_classifier TaskType.

        Both enums have the same string values, so we look up by value.
        This maintains backward compatibility with TASK_CONFIGS.

        Args:
            unified_type: TaskType from UnifiedTaskTracker

        Returns:
            TaskType from task_classifier for TASK_CONFIGS lookup
        """
        # Map unified enum to classifier enum by matching string values
        try:
            return ClassifierTaskType(unified_type.value)
        except ValueError:
            # Fallback to GENERAL if the value doesn't exist in classifier enum
            return ClassifierTaskType.GENERAL

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
        """
        self.settings = settings
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.console = console or Console()
        self.tool_selection = tool_selection or {}
        self.thinking = thinking
        self.provider_name = (provider_name or getattr(provider, "name", "") or "").lower()
        self.tool_calling_models = getattr(settings, "tool_calling_models", {})
        self.tool_capabilities = ToolCallingMatrix(
            self.tool_calling_models,
            always_allow_providers=["openai", "anthropic", "google", "xai"],
        )

        # Initialize tool calling adapter for unified provider handling
        self.tool_adapter = ToolCallingAdapterRegistry.get_adapter(
            provider_name=self.provider_name or getattr(provider, "name", "unknown"),
            model=model,
            config={"settings": settings},
        )
        self.tool_calling_caps = self.tool_adapter.get_capabilities()
        logger.info(
            f"Tool calling adapter: {self.tool_adapter.provider_name}, "
            f"native={self.tool_calling_caps.native_tool_calls}, "
            f"format={self.tool_calling_caps.tool_call_format.value}"
        )

        # Response sanitizer for cleaning model output
        self.sanitizer = ResponseSanitizer()

        # System prompt builder for provider-specific prompts
        self.prompt_builder = SystemPromptBuilder(
            provider_name=self.provider_name,
            model=model,
            tool_adapter=self.tool_adapter,
            capabilities=self.tool_calling_caps,
        )

        # Load project context from .victor/init.md (similar to Claude Code's CLAUDE.md)
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

        # Unified progress tracker for loop detection and progress monitoring
        # Replaces scattered loop detection variables with a single source of truth
        self.progress_tracker = LoopDetector(
            config=ProgressConfig(
                tool_budget=self.tool_budget,
                max_iterations_default=getattr(settings, "max_exploration_iterations", 5),
                max_iterations_analysis=getattr(
                    settings, "max_exploration_iterations_analysis", 15
                ),
                max_iterations_action=getattr(settings, "max_exploration_iterations_action", 6),
                max_iterations_research=getattr(settings, "max_research_iterations", 6),
                repeat_threshold_default=3,
                repeat_threshold_analysis=5,
                min_content_threshold=getattr(settings, "min_content_threshold", 150),
                max_total_iterations=getattr(settings, "max_total_loop_iterations", 20),
            ),
            task_type=TaskType.DEFAULT,
        )

        # Gap implementations: Complexity classifier, action authorizer, search router
        self.task_classifier = ComplexityClassifier()
        self.intent_detector = ActionAuthorizer()
        self.search_router = SearchRouter()

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

        # Analytics
        from victor.config.settings import get_project_paths

        analytics_log_file = get_project_paths().global_logs_dir / "usage.jsonl"
        self.usage_logger = UsageLogger(analytics_log_file, enabled=self.settings.analytics_enabled)
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

        # Tool usage analytics
        self._tool_usage_stats: Dict[str, Dict[str, Any]] = {}
        self._tool_selection_stats: Dict[str, int] = {
            "semantic_selections": 0,
            "keyword_selections": 0,
            "fallback_selections": 0,
            "total_tools_selected": 0,
            "total_tools_executed": 0,
        }
        # Cost tracking
        self._cost_tracking: Dict[str, Any] = {
            "total_cost_weight": 0.0,
            "cost_by_tier": {tier.value: 0.0 for tier in CostTier},
            "calls_by_tier": {tier.value: 0 for tier in CostTier},
        }
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
                project_path = str(paths.project_root)
                session = self.memory_manager.create_session(
                    project_path=project_path,
                    provider=self.provider_name,
                    model=model,
                    max_tokens=max_context,
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

        # Conversation state machine for intelligent stage detection
        self.conversation_state = ConversationStateMachine()

        # Intent classifier for semantic continuation/completion detection
        # Uses embeddings instead of hardcoded phrase matching
        self.intent_classifier = IntentClassifier.get_instance()

        # Tool registry
        self.tools = ToolRegistry()
        self._register_default_tools()
        self._load_tool_configurations()  # Load tool enable/disable states from config
        self.tools.register_before_hook(self._log_tool_call)

        # Plugin system for extensible tools
        self.plugin_manager: Optional[ToolPluginRegistry] = None
        if getattr(settings, "plugin_enabled", True):
            self._initialize_plugins()

        # Argument normalizer for handling malformed tool arguments (e.g., Python vs JSON syntax)
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

        self.tool_executor = ToolExecutor(
            tool_registry=self.tools,
            argument_normalizer=self.argument_normalizer,
            tool_cache=self.tool_cache,
            max_retries=getattr(settings, "tool_retry_max_attempts", 3),
            retry_delay=getattr(settings, "tool_retry_base_delay", 1.0),
            validation_mode=validation_mode,
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

        # Initialize TaskMilestoneMonitor for goal-aware orchestration
        # DEPRECATED: Use unified_tracker instead. Kept for backward compatibility.
        self.task_tracker = TaskMilestoneMonitor()
        # Apply model-specific exploration settings to legacy task tracker
        self.task_tracker.set_model_exploration_settings(
            exploration_multiplier=self.tool_calling_caps.exploration_multiplier,
            continuation_patience=self.tool_calling_caps.continuation_patience,
        )

        # NEW: Initialize UnifiedTaskTracker (consolidates task_tracker + progress_tracker)
        # This is the single source of truth for task progress, milestones, and loop detection
        self.unified_tracker = UnifiedTaskTracker()
        # Apply model-specific exploration settings
        self.unified_tracker.set_model_exploration_settings(
            exploration_multiplier=self.tool_calling_caps.exploration_multiplier,
            continuation_patience=self.tool_calling_caps.continuation_patience,
        )

        # Initialize unified ToolSelector (handles semantic + keyword selection)
        self.tool_selector = ToolSelector(
            tools=self.tools,
            semantic_selector=self.semantic_selector,
            conversation_state=self.conversation_state,
            task_tracker=self.task_tracker,
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

        # TaskAnalyzer: Unified task analysis facade
        self._task_analyzer = get_task_analyzer()

        logger.info(
            "Orchestrator initialized with decomposed components: "
            "ConversationController, ToolPipeline, StreamingController, TaskAnalyzer"
        )

    # =====================================================================
    # Callbacks for decomposed components
    # =====================================================================

    def _on_tool_start_callback(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Callback when tool execution starts (from ToolPipeline)."""
        # Use the pipeline's calls_used as the iteration count
        iteration = self._tool_pipeline.calls_used if hasattr(self, "_tool_pipeline") else 0
        self.debug_logger.log_tool_call(tool_name, arguments, iteration)

    def _on_tool_complete_callback(self, result: ToolCallResult) -> None:
        """Callback when tool execution completes (from ToolPipeline)."""
        # Update tool usage stats
        if result.tool_name not in self._tool_usage_stats:
            self._tool_usage_stats[result.tool_name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "total_time_ms": 0.0,
            }
        stats = self._tool_usage_stats[result.tool_name]
        stats["calls"] += 1
        if result.success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
        stats["total_time_ms"] += result.execution_time_ms

    def _on_streaming_session_complete(self, session: StreamingSession) -> None:
        """Callback when streaming session completes (from StreamingController)."""
        self.usage_logger.log_event(
            "stream_completed",
            {
                "session_id": session.session_id,
                "model": session.model,
                "provider": session.provider,
                "duration": session.duration,
                "cancelled": session.cancelled,
            },
        )

    # =====================================================================
    # Component accessors for external use
    # =====================================================================

    @property
    def conversation_controller(self) -> ConversationController:
        """Get the conversation controller component."""
        return self._conversation_controller

    @property
    def tool_pipeline(self) -> ToolPipeline:
        """Get the tool pipeline component."""
        return self._tool_pipeline

    @property
    def streaming_controller(self) -> StreamingController:
        """Get the streaming controller component."""
        return self._streaming_controller

    @property
    def task_analyzer(self) -> TaskAnalyzer:
        """Get the task analyzer component."""
        return self._task_analyzer

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
        self._current_stream_metrics = StreamMetrics(start_time=time.time())
        return self._current_stream_metrics

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

            # Initialize async (fire and forget for faster startup)
            asyncio.create_task(self._conversation_embedding_store.initialize())

            logger.info(
                "[AgentOrchestrator] ConversationEmbeddingStore configured. "
                "Message embeddings will sync to LanceDB."
            )
        except Exception as e:
            logger.warning(f"Failed to initialize ConversationEmbeddingStore: {e}")
            self._conversation_embedding_store = None

    def _record_first_token(self) -> None:
        """Record the time of first token received."""
        if hasattr(self, "_current_stream_metrics") and self._current_stream_metrics:
            if self._current_stream_metrics.first_token_time is None:
                self._current_stream_metrics.first_token_time = time.time()

    def _finalize_stream_metrics(self) -> Optional[StreamMetrics]:
        """Finalize stream metrics at end of streaming session."""
        if hasattr(self, "_current_stream_metrics") and self._current_stream_metrics:
            self._current_stream_metrics.end_time = time.time()
            metrics = self._current_stream_metrics

            # Log stream metrics
            self.usage_logger.log_event(
                "stream_completed",
                {
                    "ttft": metrics.time_to_first_token,
                    "total_duration": metrics.total_duration,
                    "tokens_per_second": metrics.tokens_per_second,
                    "total_chunks": metrics.total_chunks,
                },
            )

            # Record to streaming metrics collector if available
            if self.streaming_metrics_collector:
                try:
                    from victor.analytics.streaming_metrics import StreamMetrics as AnalyticsMetrics
                    import uuid

                    # Estimate total tokens from content length (roughly 4 chars per token)
                    estimated_tokens = metrics.total_content_length // 4

                    # Convert to analytics format and record
                    analytics_metrics = AnalyticsMetrics(
                        request_id=str(uuid.uuid4()),
                        start_time=metrics.start_time,
                        first_token_time=metrics.first_token_time,
                        last_token_time=metrics.end_time,
                        total_chunks=metrics.total_chunks,
                        total_tokens=estimated_tokens,
                        model=self.model,
                        provider=self.provider_name,
                    )
                    self.streaming_metrics_collector.record_metrics(analytics_metrics)
                except Exception as e:
                    logger.debug(f"Failed to record to metrics collector: {e}")

            return metrics
        return None

    def get_last_stream_metrics(self) -> Optional[StreamMetrics]:
        """Get metrics from the last streaming session."""
        return getattr(self, "_current_stream_metrics", None)

    def get_streaming_metrics_summary(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive streaming metrics summary.

        Returns:
            Dictionary with aggregated metrics or None if metrics disabled.
        """
        if not self.streaming_metrics_collector:
            return None

        summary = self.streaming_metrics_collector.get_summary()
        # Convert MetricsSummary dataclass to dict if needed
        if hasattr(summary, "__dict__"):
            return vars(summary)
        return summary  # type: ignore[return-value]

    def get_streaming_metrics_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent streaming metrics history.

        Args:
            limit: Maximum number of recent metrics to return

        Returns:
            List of recent metrics dictionaries
        """
        if not self.streaming_metrics_collector:
            return []

        metrics_list = self.streaming_metrics_collector.get_recent_metrics(count=limit)
        # Convert StreamMetrics to dictionaries
        return [vars(m) if hasattr(m, "__dict__") else m for m in metrics_list]  # type: ignore[misc]

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
        if method == "semantic":
            self._tool_selection_stats["semantic_selections"] += 1
        elif method == "keyword":
            self._tool_selection_stats["keyword_selections"] += 1
        elif method == "fallback":
            self._tool_selection_stats["fallback_selections"] += 1

        self._tool_selection_stats["total_tools_selected"] += num_tools
        self.usage_logger.log_event("tool_selection", {"method": method, "tool_count": num_tools})

        logger.debug(
            f"Tool selection: method={method}, num_tools={num_tools}, "
            f"stats={self._tool_selection_stats}"
        )

    def _record_tool_execution(self, tool_name: str, success: bool, elapsed_ms: float) -> None:
        """Record tool execution statistics.

        Args:
            tool_name: Name of the tool executed
            success: Whether execution succeeded
            elapsed_ms: Execution time in milliseconds
        """
        # Get tool cost tier
        tool_cost = self.tools.get_tool_cost(tool_name)
        cost_tier = tool_cost if tool_cost else CostTier.FREE
        cost_weight = cost_tier.weight

        if tool_name not in self._tool_usage_stats:
            self._tool_usage_stats[tool_name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_time_ms": 0.0,
                "avg_time_ms": 0.0,
                "min_time_ms": float("inf"),
                "max_time_ms": 0.0,
                "cost_tier": cost_tier.value,
                "total_cost_weight": 0.0,
            }

        stats = self._tool_usage_stats[tool_name]
        stats["total_calls"] += 1
        stats["successful_calls"] += 1 if success else 0
        stats["failed_calls"] += 0 if success else 1
        stats["total_time_ms"] += elapsed_ms
        stats["avg_time_ms"] = stats["total_time_ms"] / stats["total_calls"]
        stats["min_time_ms"] = min(stats["min_time_ms"], elapsed_ms)
        stats["max_time_ms"] = max(stats["max_time_ms"], elapsed_ms)
        stats["total_cost_weight"] += cost_weight

        # Update global cost tracking
        self._cost_tracking["total_cost_weight"] += cost_weight
        self._cost_tracking["cost_by_tier"][cost_tier.value] += cost_weight
        self._cost_tracking["calls_by_tier"][cost_tier.value] += 1

        self._tool_selection_stats["total_tools_executed"] += 1

        logger.debug(
            f"Tool executed: {tool_name} "
            f"(success={success}, time={elapsed_ms:.1f}ms, "
            f"total_calls={stats['total_calls']}, "
            f"success_rate={stats['successful_calls']/stats['total_calls']*100:.1f}%)"
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
        return {
            "selection_stats": self._tool_selection_stats,
            "tool_stats": self._tool_usage_stats,
            "cost_tracking": self._cost_tracking,
            "top_tools_by_usage": sorted(
                [(name, stats["total_calls"]) for name, stats in self._tool_usage_stats.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
            "top_tools_by_time": sorted(
                [(name, stats["total_time_ms"]) for name, stats in self._tool_usage_stats.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
            "top_tools_by_cost": sorted(
                [
                    (name, stats.get("total_cost_weight", 0.0))
                    for name, stats in self._tool_usage_stats.items()
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
            "conversation_state": self.conversation_state.get_state_summary(),
        }

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

            # Update core attributes
            old_provider_name = self.provider_name
            old_model = self.model

            self.provider = new_provider
            self.model = new_model
            self.provider_name = provider_name.lower()

            # Reinitialize tool calling adapter for the new provider/model
            self.tool_adapter = ToolCallingAdapterRegistry.get_adapter(
                provider_name=self.provider_name,
                model=new_model,
                config={"settings": self.settings},
            )
            self.tool_calling_caps = self.tool_adapter.get_capabilities()

            # Apply model-specific exploration settings to legacy task tracker
            self.task_tracker.set_model_exploration_settings(
                exploration_multiplier=self.tool_calling_caps.exploration_multiplier,
                continuation_patience=self.tool_calling_caps.continuation_patience,
            )

            # Apply model-specific exploration settings to unified tracker (primary)
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

            # Update tool budget based on new adapter's recommendation
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

            return True

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
        try:
            old_model = self.model
            self.model = model

            # Reinitialize tool calling adapter for the new model
            self.tool_adapter = ToolCallingAdapterRegistry.get_adapter(
                provider_name=self.provider_name,
                model=model,
                config={"settings": self.settings},
            )
            self.tool_calling_caps = self.tool_adapter.get_capabilities()

            # Apply model-specific exploration settings to legacy task tracker
            self.task_tracker.set_model_exploration_settings(
                exploration_multiplier=self.tool_calling_caps.exploration_multiplier,
                continuation_patience=self.tool_calling_caps.continuation_patience,
            )

            # Apply model-specific exploration settings to unified tracker (primary)
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

            return True

        except Exception as e:
            logger.error(f"Failed to switch model to {model}: {e}")
            return False

    def get_current_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider and model.

        Returns:
            Dictionary with provider/model info and capabilities
        """
        return {
            "provider": self.provider_name,
            "model": self.model,
            "supports_tools": self.provider.supports_tools() if self.provider else False,
            "native_tool_calls": self.tool_calling_caps.native_tool_calls,
            "streaming_tool_calls": self.tool_calling_caps.streaming_tool_calls,
            "parallel_tool_calls": self.tool_calling_caps.parallel_tool_calls,
            "thinking_mode": self.tool_calling_caps.thinking_mode,
            "tool_budget": self.tool_budget,
            "tool_calls_used": self.tool_calls_used,
        }

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

    def _is_cloud_provider(self) -> bool:
        """Check if the current provider is a cloud-based API with robust tool calling."""
        return self.prompt_builder.is_cloud_provider()

    def _is_local_provider(self) -> bool:
        """Check if the current provider is a local model (Ollama, LMStudio, vLLM)."""
        return self.prompt_builder.is_local_provider()

    def _build_system_prompt_with_adapter(self) -> str:
        """Build system prompt using the tool calling adapter."""
        return self.prompt_builder.build()

    def _build_system_prompt_for_provider(self) -> str:
        """Build an appropriate system prompt based on the provider type."""
        return self.prompt_builder._build_for_provider()

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

    def _extract_file_structure(self, content: str, file_path: str) -> str:
        """Extract a structural summary for very large files.

        For Python files, extracts class and function definitions.
        For other files, shows line count and sample lines.

        Args:
            content: Full file content
            file_path: Path to the file

        Returns:
            Structural summary string
        """
        lines = content.split("\n")
        num_lines = len(lines)

        # Detect file type
        ext = Path(file_path).suffix.lower()

        summary_parts = [f"FILE STRUCTURE: {file_path}"]
        summary_parts.append(f"Total lines: {num_lines}")

        if ext in (".py", ".pyi"):
            # Extract Python structure
            classes = []
            functions = []
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("class ") and ":" in stripped:
                    class_name = stripped[6:].split("(")[0].split(":")[0].strip()
                    classes.append(f"  Line {i+1}: class {class_name}")
                elif stripped.startswith("def ") and "(" in stripped:
                    func_name = stripped[4:].split("(")[0].strip()
                    # Only top-level functions (no leading whitespace)
                    if not line.startswith(" ") and not line.startswith("\t"):
                        functions.append(f"  Line {i+1}: def {func_name}()")

            if classes:
                summary_parts.append(f"\nClasses ({len(classes)}):")
                summary_parts.extend(classes[:20])  # Max 20 classes
                if len(classes) > 20:
                    summary_parts.append(f"  ... and {len(classes) - 20} more")

            if functions:
                summary_parts.append(f"\nFunctions ({len(functions)}):")
                summary_parts.extend(functions[:30])  # Max 30 functions
                if len(functions) > 30:
                    summary_parts.append(f"  ... and {len(functions) - 30} more")

        elif ext in (".js", ".ts", ".jsx", ".tsx"):
            # Extract JS/TS structure
            exports = []
            functions = []
            for i, line in enumerate(lines):
                stripped = line.strip()
                if "function " in stripped or "const " in stripped or "export " in stripped:
                    if len(stripped) < 100:  # Skip very long lines
                        if stripped.startswith("export "):
                            exports.append(f"  Line {i+1}: {stripped[:60]}...")
                        elif "function " in stripped:
                            functions.append(f"  Line {i+1}: {stripped[:60]}...")

            if exports:
                summary_parts.append(f"\nExports ({len(exports)}):")
                summary_parts.extend(exports[:20])
            if functions:
                summary_parts.append(f"\nFunctions ({len(functions)}):")
                summary_parts.extend(functions[:20])

        # Add first and last few lines as sample
        summary_parts.append("\n--- FIRST 30 LINES ---")
        summary_parts.extend(lines[:30])
        summary_parts.append("\n--- LAST 20 LINES ---")
        summary_parts.extend(lines[-20:])

        return "\n".join(summary_parts)

    def _format_tool_output(self, tool_name: str, args: Dict[str, Any], output: Any) -> str:
        """Format tool output with clear boundaries to prevent model hallucination.

        Uses structured markers that models recognize as authoritative tool output.
        This prevents the model from ignoring or fabricating tool results.

        Args:
            tool_name: Name of the tool that was executed
            args: Arguments passed to the tool
            output: Raw output from the tool

        Returns:
            Formatted string with clear TOOL_OUTPUT boundaries
        """
        # Maximum output size to prevent context overflow (reduced from 50K to 15K)
        max_output_chars = getattr(self.settings, "max_tool_output_chars", 15000)

        output_str = str(output) if output is not None else ""
        original_len = len(output_str)
        truncated = False

        if original_len > max_output_chars:
            truncated = True
            output_str = output_str[:max_output_chars]

        # Special formatting for file reading tools - make content unmistakably clear
        if tool_name == "read_file":
            file_path = args.get("path", "unknown")

            # For very large files (>50KB), show structure instead of raw content
            structure_threshold = getattr(self.settings, "file_structure_threshold", 50000)
            if original_len > structure_threshold:
                # Show file structure summary for very large files
                file_content = str(output) if output is not None else ""
                structure_summary = self._extract_file_structure(file_content, file_path)
                return f"""<TOOL_OUTPUT tool="{tool_name}" path="{file_path}">
═══ FILE IS VERY LARGE ({original_len:,} chars / {len(file_content.splitlines())} lines) ═══
{structure_summary}
═══ END OF FILE STRUCTURE ═══
</TOOL_OUTPUT>

NOTE: This file is very large. Showing structure summary instead of full content.
To see specific sections, use read_file with offset/limit parameters or code_search to find specific code."""

            header = f"═══ ACTUAL FILE CONTENT: {file_path} ═══"
            footer = f"═══ END OF FILE: {file_path} ═══"
            if truncated:
                footer = f"═══ END OF FILE (TRUNCATED: showing {max_output_chars:,} of {original_len:,} chars): {file_path} ═══"

            return f"""<TOOL_OUTPUT tool="{tool_name}" path="{file_path}">
{header}
{output_str}
{footer}
</TOOL_OUTPUT>

IMPORTANT: The content above between the ═══ markers is the EXACT content of the file.
You MUST use this actual content in your analysis. Do NOT fabricate or imagine different content."""

        elif tool_name == "list_directory":
            dir_path = args.get("path", ".")
            return f"""<TOOL_OUTPUT tool="{tool_name}" path="{dir_path}">
═══ ACTUAL DIRECTORY LISTING: {dir_path} ═══
{output_str}
═══ END OF DIRECTORY LISTING ═══
</TOOL_OUTPUT>

Use only the files/directories listed above. Do not invent files that are not shown."""

        elif tool_name in ("code_search", "semantic_code_search"):
            query = args.get("query", args.get("pattern", ""))
            return f"""<TOOL_OUTPUT tool="{tool_name}" query="{query}">
═══ SEARCH RESULTS ═══
{output_str}
═══ END OF SEARCH RESULTS ═══
</TOOL_OUTPUT>

These are the actual search results. Reference only the files and matches shown above."""

        elif tool_name == "execute_bash":
            command = args.get("command", "")
            return f"""<TOOL_OUTPUT tool="{tool_name}" command="{command}">
═══ COMMAND OUTPUT ═══
{output_str}
═══ END OF COMMAND OUTPUT ═══
</TOOL_OUTPUT>"""

        else:
            # Generic tool output format
            truncation_note = " [OUTPUT TRUNCATED]" if truncated else ""
            return f"""<TOOL_OUTPUT tool="{tool_name}">
{output_str}{truncation_note}
</TOOL_OUTPUT>"""

    def _register_default_workflows(self) -> None:
        """Register default workflows."""
        self.workflow_registry.register(NewFeatureWorkflow())

    def _register_default_tools(self) -> None:
        """Dynamically discovers and registers all tools from the victor.tools directory."""
        # --- Pre-registration setup ---
        # Some tools have functions that need to be called before registration to set
        # providers or configurations.

        # Set git provider
        set_git_provider(self.provider, self.model)

        # Set batch processor config
        set_batch_processor_config(max_workers=4)

        # Set code review config
        set_code_review_config(max_complexity=10)

        # Set web search provider and defaults (if not in air-gapped mode)
        if not self.settings.airgapped_mode:
            set_web_search_provider(self.provider, self.model)
            try:
                tool_config = self.settings.load_tool_config()
                web_cfg = tool_config.get("web_tools", {}) or tool_config.get("web", {}) or {}
                set_web_tool_defaults(
                    fetch_top=web_cfg.get("summarize_fetch_top"),
                    fetch_pool=web_cfg.get("summarize_fetch_pool"),
                    max_content_length=web_cfg.get("summarize_max_content_length"),
                )
            except Exception as exc:
                logger.warning(f"Failed to apply web tool defaults: {exc}")

        # --- Dynamic Tool Discovery and Registration ---
        tools_dir = os.path.join(os.path.dirname(__file__), "..", "tools")
        excluded_files = {"__init__.py", "base.py", "decorators.py", "semantic_selector.py"}
        registered_tools_count = 0

        # Import BaseTool for isinstance checks
        from victor.tools.base import BaseTool as BaseToolClass

        for filename in os.listdir(tools_dir):
            if filename.endswith(".py") and filename not in excluded_files:
                module_name = f"victor.tools.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for _name, obj in inspect.getmembers(module):
                        # Register @tool decorated functions
                        if inspect.isfunction(obj) and getattr(obj, "_is_tool", False):
                            self.tools.register(obj)
                            registered_tools_count += 1
                        # Register BaseTool class instances (class-based tools)
                        elif (
                            inspect.isclass(obj)
                            and issubclass(obj, BaseToolClass)
                            and obj is not BaseToolClass
                            and hasattr(obj, "name")
                        ):
                            try:
                                tool_instance = obj()
                                self.tools.register(tool_instance)
                                registered_tools_count += 1
                            except Exception as e:
                                logger.debug(f"Skipped registering {_name}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to load tools from {module_name}: {e}")

        logger.debug(f"Dynamically registered {registered_tools_count} tools from victor.tools/")

        # --- Post-registration setup for special tools (like MCP) ---
        # Register MCP tools if configured (supports both legacy and new registry approach)
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
        """Initialize and load tool plugins from configured directories."""
        try:
            from victor.config.settings import get_project_paths

            # Use centralized path for plugins directory
            plugin_dirs = [get_project_paths().global_plugins_dir]
            plugin_config = getattr(self.settings, "plugin_config", {})
            disabled_plugins = set(getattr(self.settings, "plugin_disabled", []))

            self.plugin_manager = ToolPluginRegistry(
                plugin_dirs=plugin_dirs,
                config=plugin_config,
            )

            # Disable specified plugins
            for plugin_name in disabled_plugins:
                self.plugin_manager.disable_plugin(plugin_name)

            # Discover and load plugins from directories
            loaded_count = self.plugin_manager.discover_and_load()

            # Load plugins from packages
            for package_name in getattr(self.settings, "plugin_packages", []):
                plugin = self.plugin_manager.load_plugin_from_package(package_name)
                if plugin:
                    self.plugin_manager.register_plugin(plugin)

            # Register plugin tools with our tool registry
            if loaded_count > 0 or self.plugin_manager.loaded_plugins:
                tool_count = self.plugin_manager.register_tools(self.tools)
                logger.info(
                    f"Plugins loaded: {len(self.plugin_manager.loaded_plugins)} plugins, "
                    f"{tool_count} tools"
                )

        except Exception as e:
            logger.warning(f"Failed to initialize plugin system: {e}")
            self.plugin_manager = None

    def _plan_tools(
        self, goals: List[str], available_inputs: Optional[List[str]] = None
    ) -> List[ToolDefinition]:
        """Plan a sequence of tools to satisfy goals using the dependency graph."""
        if not goals or not self.tool_graph:
            return []

        available = available_inputs or []
        plan_names = self.tool_graph.plan(goals, available)
        tool_defs: List[ToolDefinition] = []
        for name in plan_names:
            tool = self.tools.get(name)
            if tool and self.tools.is_tool_enabled(name):
                tool_defs.append(
                    ToolDefinition(
                        name=tool.name, description=tool.description, parameters=tool.parameters
                    )
                )
        return tool_defs

    def _goal_hints_for_message(self, user_message: str) -> List[str]:
        """Infer planning goals from the user request."""
        text = user_message.lower()
        goals: List[str] = []
        if any(kw in text for kw in ["summarize", "summary", "analyze", "overview"]):
            goals.append("summary")
        if any(kw in text for kw in ["review", "code review", "audit"]):
            goals.append("summary")
        if any(kw in text for kw in ["doc", "documentation", "readme"]):
            goals.append("documentation")
        if any(kw in text for kw in ["security", "vulnerability", "secret", "scan"]):
            goals.append("security_report")
        if any(kw in text for kw in ["complexity", "metrics", "maintainability", "technical debt"]):
            goals.append("metrics_report")
        return goals

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

                # Check if core tools are included (use centralized CORE_TOOLS from tool_selection)
                missing_core = CORE_TOOLS - set(enabled_tools)
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
                disabled_core = CORE_TOOLS & set(disabled_tools)
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
                        if tool_name in CORE_TOOLS:
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
                self.add_message("assistant", fallback_content)
                final_response = CompletionResponse(
                    content=fallback_content,
                    role="assistant",
                    tool_calls=None,
                )

        return final_response

    async def stream_chat(self, user_message: str) -> AsyncIterator[StreamChunk]:
        """Stream a chat response.

        Args:
            user_message: User's message

        Yields:
            StreamChunk objects with incremental response

        Note:
            Stream metrics (TTFT, throughput) are available via get_last_stream_metrics()
            after the stream completes.

        The stream can be cancelled by calling request_cancellation(). When cancelled,
        the stream will yield a final chunk indicating cancellation and stop.
        """
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

        # Reset legacy trackers (kept for backward compatibility during transition)
        self.progress_tracker.reset()
        self.task_tracker.reset()

        # Reset context reminder manager for new conversation turn
        self.reminder_manager.reset()

        # Local aliases for frequently-used values
        max_total_iterations = self.unified_tracker.config.get(
            "max_total_iterations", 50
        )
        total_iterations = 0
        force_completion = False

        # Add user message to history
        self.add_message("user", user_message)

        # Detect task type using unified tracker (single source of truth)
        unified_task_type = self.unified_tracker.detect_task_type(user_message)
        # Map to legacy task type for backward compatibility with TASK_CONFIGS
        task_type = self._map_unified_to_legacy_task_type(unified_task_type)
        logger.info(f"Task type detected: {unified_task_type.value}")

        # Get exploration iterations from TASK_CONFIGS for this task type
        task_config = TASK_CONFIGS.get(task_type)
        max_exploration_iterations = task_config.max_exploration_iterations if task_config else 8

        # Inject task-specific prompt hint for better guidance
        task_hint = get_task_type_hint(task_type.value)
        if task_hint:
            self.add_message("system", task_hint.strip())
            logger.debug(f"Injected task hint for task type: {task_type.value}")

        # Gap 1: Classify task complexity and adjust tool budget
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

        # Gap 4: Detect intent and inject prompt guard for non-write tasks
        intent_result = self.intent_detector.detect(user_message)
        if intent_result.intent in (ActionIntent.DISPLAY_ONLY, ActionIntent.READ_ONLY):
            if intent_result.prompt_guard:
                self.add_message("system", intent_result.prompt_guard.strip())
                logger.info(f"Intent: {intent_result.intent.value}, injected prompt guard")
        elif intent_result.intent == ActionIntent.WRITE_ALLOWED:
            logger.info("Intent: write_allowed, no prompt guard needed")

        # Get tool definitions
        # Iteratively stream → run tools → stream follow-up until no tool calls or budget exhausted
        context_msg = user_message

        # Detect action-oriented tasks (create, execute, run) - should allow more exploration before action
        action_keywords = ["create", "generate", "write", "execute", "run", "make", "build"]
        is_action_task = any(keyword in user_message.lower() for keyword in action_keywords)

        # Detect analysis/exploration tasks that need extensive reading
        analysis_keywords = [
            "analyze",
            "analysis",
            "review",
            "examine",
            "understand",
            "explore",
            "audit",
            "inspect",
            "investigate",
            "survey",
            "comprehensive",
            "thorough",
            "entire codebase",
            "all files",
            "full analysis",
            "deep dive",
            # Question patterns that require exploration
            "what are the",
            "how does",
            "how do",
            "explain the",
            "describe the",
            "key components",
            "architecture",
            "structure",
        ]
        is_analysis_task = any(keyword in user_message.lower() for keyword in analysis_keywords)

        # SYNCHRONIZE task types between task_tracker and progress_tracker
        # Map fine-grained TaskMilestoneMonitor task types to LoopDetector coarse types
        from victor.embeddings.task_classifier import TaskType as MilestoneTaskType
        milestone_to_loop_map = {
            MilestoneTaskType.EDIT: TaskType.ACTION,
            MilestoneTaskType.CREATE: TaskType.ACTION,
            MilestoneTaskType.CREATE_SIMPLE: TaskType.DEFAULT,
            MilestoneTaskType.SEARCH: TaskType.ANALYSIS,
            MilestoneTaskType.ANALYZE: TaskType.ANALYSIS,
            MilestoneTaskType.ANALYSIS_DEEP: TaskType.ANALYSIS,
            MilestoneTaskType.DESIGN: TaskType.DEFAULT,
            MilestoneTaskType.GENERAL: TaskType.DEFAULT,
            MilestoneTaskType.ACTION: TaskType.ACTION,
        }

        # Use milestone monitor's task type as primary source, with keyword override for analysis
        loop_task_type = milestone_to_loop_map.get(task_type, TaskType.DEFAULT)

        # Override: if keywords indicate analysis, use ANALYSIS even if milestone says otherwise
        # This handles compound prompts like "What are the components? Update the code"
        if is_analysis_task:
            loop_task_type = TaskType.ANALYSIS
            # For compound analysis+edit tasks, unified_tracker handles exploration limits
            if task_type in (MilestoneTaskType.EDIT, MilestoneTaskType.CREATE):
                logger.info(
                    f"Compound task detected (analysis+{task_type.value}): "
                    f"unified_tracker will use appropriate exploration limits"
                )
        elif is_action_task:
            loop_task_type = TaskType.ACTION

        logger.info(
            f"Task type classification: loop_type={loop_task_type.value}, "
            f"milestone={task_type.value}, is_analysis={is_analysis_task}, is_action={is_action_task}"
        )

        # For analysis tasks: increase temperature to reduce repetition/getting stuck
        if is_analysis_task:
            # Bump temperature for analysis tasks (helps prevent getting stuck)
            analysis_temp = min(self.temperature + 0.2, 1.0)
            logger.info(
                f"Analysis task: increasing temperature {self.temperature:.1f} -> {analysis_temp:.1f}"
            )
            self.temperature = analysis_temp

            # Add simplified analysis guidance - focus on one module at a time
            self.add_message(
                "system",
                "ANALYSIS APPROACH: Work through the codebase one module at a time. "
                "For each module: 1) List its files, 2) Read 2-3 key files, 3) Note observations. "
                "After examining 3-4 modules, provide your summary. Keep responses concise.",
            )

        # Add guidance for analysis tasks - encourage thorough exploration
        if is_analysis_task:
            # Increase tool budget for analysis tasks
            original_budget = self.tool_budget
            self.tool_budget = max(
                self.tool_budget, 200
            )  # Allow at least 200 tool calls for analysis
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

        # Add guidance for action-oriented tasks
        if is_action_task:
            logger.info(
                f"Detected action-oriented task - allowing up to {max_exploration_iterations} exploration iterations"
            )

            # Detect if this is specifically about executing/running scripts
            execution_keywords = ["execute", "run"]
            needs_execution = any(keyword in user_message.lower() for keyword in execution_keywords)

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

        goals = self._goal_hints_for_message(user_message)

        # Log all limits for debugging
        logger.info(
            f"Stream chat limits: "
            f"tool_budget={self.tool_budget}, "
            f"max_total_iterations={max_total_iterations}, "
            f"max_exploration_iterations={max_exploration_iterations}, "
            f"is_analysis_task={is_analysis_task}, "
            f"is_action_task={is_action_task}"
        )

        # Reset debug logger for new conversation turn
        self.debug_logger.reset()

        while True:
            # Check for cancellation request
            if self._check_cancellation():
                logger.info("Stream cancelled by user request")
                self._is_streaming = False
                yield StreamChunk(
                    content="\n\n[Cancelled by user]\n",
                    is_final=True,
                )
                return

            # Check hard iteration limit
            total_iterations += 1
            unique_resources = self.unified_tracker.unique_resources
            logger.debug(
                f"Iteration {total_iterations}/{max_total_iterations}: "
                f"tool_calls_used={self.tool_calls_used}/{self.tool_budget}, "
                f"unique_resources={len(unique_resources)}, "
                f"force_completion={force_completion}"
            )

            # Use debug logger for incremental tracking
            self.debug_logger.log_iteration_start(
                total_iterations,
                tool_calls=self.tool_calls_used,
                files_read=len(unique_resources),
            )
            self.debug_logger.log_limits(
                tool_budget=self.tool_budget,
                tool_calls_used=self.tool_calls_used,
                max_iterations=max_total_iterations,
                current_iteration=total_iterations,
                is_analysis_task=is_analysis_task,
            )

            # Check for context overflow (using model-aware limit)
            max_context = self._get_max_context_chars()
            if self._check_context_overflow(max_context):
                # Try smart compaction first before forcing completion
                logger.warning("Context overflow detected. Attempting smart compaction...")
                removed = self._conversation_controller.smart_compact_history(
                    current_query=user_message
                )
                if removed > 0:
                    logger.info(f"Smart compaction removed {removed} messages")
                    yield StreamChunk(
                        content=f"\n[context] Compacted history ({removed} messages) to continue.\n"
                    )
                    # Inject context reminder about compacted content
                    self._conversation_controller.inject_compaction_context()

                # Check if still overflowing after compaction
                if self._check_context_overflow(max_context):
                    logger.warning("Still overflowing after compaction. Forcing completion.")
                    yield StreamChunk(
                        content="\n[tool] ⚠ Context size limit reached. Providing summary.\n"
                    )
                    # Force completion due to context overflow
                    # Use temporary messages to avoid polluting conversation history
                    # Use thinking disable prefix if model supports it (e.g., Qwen3 /no_think)
                    completion_prompt = self._get_thinking_disabled_prompt(
                        "Context limit reached. Summarize in 2-3 sentences."
                    )

                    # Take only recent context (last 8 messages) for the completion request
                    recent_messages = self.messages[-8:] if len(self.messages) > 8 else self.messages[:]
                    completion_messages = recent_messages + [Message(role="user", content=completion_prompt)]

                    try:
                        response = await self.provider.chat(
                            messages=completion_messages,
                            model=self.model,
                            temperature=self.temperature,
                            max_tokens=min(self.max_tokens, 1024),  # Limit output for overflow
                            tools=None,
                        )
                        if response and response.content:
                            sanitized = self._sanitize_response(response.content)
                            if sanitized:
                                self.add_message("assistant", sanitized)
                                yield StreamChunk(content=sanitized)
                    except Exception as e:
                        logger.warning(f"Final response after context overflow failed: {e}")
                    yield StreamChunk(content="", is_final=True)
                    return

            if total_iterations > max_total_iterations:
                logger.warning(
                    f"Hard iteration limit reached ({max_total_iterations}). Forcing completion."
                )
                yield StreamChunk(
                    content=f"\n[tool] ⚠ Maximum iterations ({max_total_iterations}) reached. Providing summary.\n"
                )
                # Force a final response without tools
                # Use temporary messages to avoid polluting conversation history
                # Use thinking disable prefix if model supports it (e.g., Qwen3 /no_think)
                iteration_prompt = self._get_thinking_disabled_prompt(
                    "Max iterations reached. Summarize key findings in 3-4 sentences. "
                    "Do NOT attempt any more tool calls."
                )

                # Take only recent context (last 10 messages) for the completion request
                recent_messages = self.messages[-10:] if len(self.messages) > 10 else self.messages[:]
                completion_messages = recent_messages + [Message(role="user", content=iteration_prompt)]

                # Make one final call without tools
                try:
                    response = await self.provider.chat(
                        messages=completion_messages,
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=min(self.max_tokens, 1024),  # Limit output for forced completion
                        tools=None,  # No tools - force text response
                    )
                    if response and response.content:
                        sanitized = self._sanitize_response(response.content)
                        if sanitized:
                            self.add_message("assistant", sanitized)
                            yield StreamChunk(content=sanitized)
                except Exception as e:
                    logger.warning(f"Final response generation failed: {e}")
                    yield StreamChunk(
                        content="Unable to generate final summary due to iteration limit.\n"
                    )
                yield StreamChunk(content="", is_final=True)
                break

            # Select tools for this pass
            tools = None
            provider_supports_tools = self.provider.supports_tools()
            tooling_allowed = provider_supports_tools and self._model_supports_tool_calls()

            if tooling_allowed:
                # Get planned tools if we have inferred goals
                planned_tools = None
                if goals:
                    available_inputs = ["query"]
                    if self.observed_files:
                        available_inputs.append("file_contents")
                    planned_tools = self._plan_tools(goals, available_inputs=available_inputs)

                # Use unified ToolSelector for tool selection
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

            # Prepare optional thinking parameter for providers that support it (Anthropic)
            provider_kwargs = {}
            if self.thinking:
                # Anthropic extended thinking format
                provider_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}

            full_content = ""
            tool_calls = None
            garbage_detected = False
            consecutive_garbage_chunks = 0
            max_garbage_chunks = 3  # Stop after 3 consecutive garbage chunks

            async for chunk in self.provider.stream(
                messages=self.messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=tools,
                **provider_kwargs,
            ):
                # Check for garbage content (local model confusion)
                if chunk.content and self._is_garbage_content(chunk.content):
                    consecutive_garbage_chunks += 1
                    if consecutive_garbage_chunks >= max_garbage_chunks:
                        if not garbage_detected:
                            garbage_detected = True
                            logger.warning(
                                f"Garbage content detected after {len(full_content)} chars - "
                                "stopping stream early"
                            )
                        # Don't yield garbage content, don't accumulate it
                        continue
                else:
                    consecutive_garbage_chunks = 0

                full_content += chunk.content
                # Track stream metrics
                stream_metrics.total_chunks += 1
                # Estimate tokens (rough: ~4 chars per token)
                if chunk.content:
                    # Record TTFT on first content chunk
                    self._record_first_token()
                    total_tokens += len(chunk.content) / 4
                    stream_metrics.total_content_length += len(chunk.content)
                if chunk.tool_calls:
                    logger.debug(f"Received tool_calls in chunk: {chunk.tool_calls}")
                    tool_calls = chunk.tool_calls
                    stream_metrics.tool_calls_count += len(chunk.tool_calls)

                # Capture usage from final chunks (provider-reported tokens)
                if chunk.usage:
                    for key in cumulative_usage:
                        cumulative_usage[key] += chunk.usage.get(key, 0)
                    logger.debug(
                        f"Chunk usage: in={chunk.usage.get('prompt_tokens', 0)} "
                        f"out={chunk.usage.get('completion_tokens', 0)} "
                        f"cache_read={chunk.usage.get('cache_read_input_tokens', 0)}"
                    )

                yield chunk

            # If garbage was detected, force completion on next iteration
            if garbage_detected and not tool_calls:
                force_completion = True
                logger.info("Setting force_completion due to garbage detection")

            # Use unified adapter-based tool call parsing with fallbacks
            if not tool_calls and full_content:
                parse_result = self._parse_tool_calls_with_adapter(full_content, tool_calls)
                if parse_result.tool_calls:
                    # Convert ToolCall objects to dicts for compatibility
                    tool_calls = [tc.to_dict() for tc in parse_result.tool_calls]
                    full_content = parse_result.remaining_content

            # Ensure tool_calls is a list of dicts to avoid type errors from malformed provider output
            if tool_calls:
                normalized_tool_calls = [tc for tc in tool_calls if isinstance(tc, dict)]
                if len(normalized_tool_calls) != len(tool_calls):
                    logger.warning(f"Dropped non-dict tool_calls: {tool_calls}")
                tool_calls = normalized_tool_calls or None

            # Filter out invalid/hallucinated tool names early (adapter already validates, but double-check for enabled)
            if tool_calls:
                valid_tool_calls = []
                for tc in tool_calls:
                    name = tc.get("name", "")
                    if self.tools.is_tool_enabled(name):
                        valid_tool_calls.append(tc)
                    else:
                        logger.debug(f"Filtered out disabled tool: {name}")
                if len(valid_tool_calls) != len(tool_calls):
                    logger.warning(
                        f"Filtered {len(tool_calls) - len(valid_tool_calls)} invalid tool calls"
                    )
                tool_calls = valid_tool_calls or None

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
            elif not tool_calls:
                # No content and no tool calls; attempt aggressive recovery
                logger.warning("Model returned empty response - attempting aggressive recovery")

                # Check if model has thinking disable prefix (e.g., Qwen3 /no_think)
                thinking_prefix = getattr(self.tool_calling_caps, "thinking_disable_prefix", None)
                if thinking_prefix:
                    logger.debug(f"Using thinking disable prefix '{thinking_prefix}' for recovery")

                # Build recovery prompts - _get_thinking_disabled_prompt adds prefix if available
                # Use simpler prompts and lower temps for models with thinking mode
                has_thinking_mode = getattr(self.tool_calling_caps, "thinking_mode", False)
                if has_thinking_mode:
                    # Simpler prompts and lower temps for thinking models
                    recovery_prompts = [
                        (
                            self._get_thinking_disabled_prompt(
                                "Respond in 2-3 sentences: What files did you read and what did you find?"
                            ),
                            min(self.temperature + 0.1, 0.7),
                        ),
                        (
                            self._get_thinking_disabled_prompt(
                                "List 3 bullet points about the code you examined."
                            ),
                            min(self.temperature + 0.2, 0.8),
                        ),
                        (
                            self._get_thinking_disabled_prompt(
                                "One sentence answer: What is the main thing you learned?"
                            ),
                            min(self.temperature + 0.3, 0.9),
                        ),
                    ]
                else:
                    # Standard recovery prompts
                    recovery_prompts = [
                        (
                            "Summarize your findings so far. What files did you examine? "
                            "What patterns or issues did you notice? Keep it brief.",
                            min(self.temperature + 0.2, 1.0),
                        ),
                        (
                            "Based on the code you've seen, list 3-5 observations or suggestions.",
                            min(self.temperature + 0.3, 1.0),
                        ),
                        (
                            "What did you learn from the files? One paragraph summary.",
                            min(self.temperature + 0.4, 1.0),
                        ),
                    ]

                recovery_success = False
                for attempt, (prompt, temp) in enumerate(recovery_prompts, 1):
                    logger.info(f"Recovery attempt {attempt}/3 with temp={temp:.1f}")

                    # Create temporary message list to avoid polluting conversation history
                    # Include only recent context (last 5 exchanges) to reduce token load
                    recent_messages = self.messages[-10:] if len(self.messages) > 10 else self.messages[:]
                    recovery_messages = recent_messages + [Message(role="user", content=prompt)]

                    try:
                        response = await self.provider.chat(
                            messages=recovery_messages,
                            model=self.model,
                            temperature=temp,
                            max_tokens=min(self.max_tokens, 1024),  # Limit output for recovery
                            tools=None,  # Force text response
                        )
                        if response and response.content:
                            logger.debug(
                                f"Recovery attempt {attempt}: got {len(response.content)} chars"
                            )
                            sanitized = self._sanitize_response(response.content)
                            if sanitized and len(sanitized) > 20:
                                self.add_message("assistant", sanitized)
                                yield StreamChunk(content=sanitized, is_final=True)
                                recovery_success = True
                                break
                            elif response.content and len(response.content) > 20:
                                # Use raw if sanitization failed but content exists
                                self.add_message("assistant", response.content)
                                yield StreamChunk(content=response.content, is_final=True)
                                recovery_success = True
                                break
                        else:
                            logger.debug(f"Recovery attempt {attempt}: empty response")
                    except Exception as exc:
                        logger.warning(f"Recovery attempt {attempt} failed: {exc}")

                if recovery_success:
                    return

                # All recovery attempts failed - provide helpful error
                if is_analysis_task and self.tool_calls_used > 0:
                    # Generate a minimal summary from what we know
                    unique_resources = self.unified_tracker.unique_resources
                    files_examined = list(unique_resources)[:10]
                    fallback_msg = (
                        f"\n\n**Analysis Summary** (auto-generated)\n\n"
                        f"Examined {len(unique_resources)} files including:\n"
                        + "\n".join(f"- {f}" for f in files_examined)
                        + "\n\nThe model was unable to provide detailed analysis. "
                        "Try with a simpler query like 'analyze victor/agent/' or use a different model."
                    )
                else:
                    fallback_msg = "No tool calls were returned and the model provided no content. Please retry or simplify the request."
                yield StreamChunk(content=fallback_msg, is_final=True)
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

            # Check for loop warning using UnifiedTaskTracker (primary)
            # This gives the model a chance to correct behavior before we force stop
            unified_loop_warning = self.unified_tracker.check_loop_warning()
            if unified_loop_warning and not force_completion:
                logger.warning(f"UnifiedTaskTracker loop warning: {unified_loop_warning}")
                yield StreamChunk(
                    content=f"\n[loop] ⚠ Warning: Approaching loop limit - {unified_loop_warning}\n"
                )
                # Inject system message to warn the model
                self.add_message(
                    "system",
                    "WARNING: You are about to hit loop detection. You have been performing "
                    "the same operation repeatedly (e.g., writing the same file, making the same call). "
                    "Please do something DIFFERENT now:\n"
                    "- If you're writing a file repeatedly, STOP and move to a different task\n"
                    "- If you're stuck, provide your current progress and ask for clarification\n"
                    "- If you've completed the task, provide a summary and finish\n\n"
                    "Continuing the same operation will force the conversation to end.",
                )
                # Continue to give the model one more chance
                continue

            # PRIMARY: Check UnifiedTaskTracker for stop decision (single source of truth)
            unified_should_force, unified_hint = self.unified_tracker.should_force_action()
            if unified_should_force and not force_completion:
                force_completion = True
                logger.info(
                    f"UnifiedTaskTracker forcing action: {unified_hint}, "
                    f"metrics={self.unified_tracker.get_metrics()}"
                )

            # LEGACY (kept for transition validation - can be removed after validation)
            # Check if legacy progress tracker also recommends stopping
            legacy_stop_reason = self.progress_tracker.should_stop()
            if legacy_stop_reason.should_stop:
                logger.debug(f"Legacy LoopDetector also recommends stop: {legacy_stop_reason.reason}")

            # LEGACY: Check legacy task-aware force action
            legacy_should_force, legacy_force_hint = self.task_tracker.should_force_action()
            if legacy_should_force:
                logger.debug(f"Legacy TaskTracker also recommends force: {legacy_force_hint}")

            logger.debug(f"After streaming pass, tool_calls = {tool_calls}")

            if not tool_calls:
                # Check if model intended to continue but didn't make a tool call
                # (common with local models that sometimes forget to actually call the tool)
                # Use semantic intent classification instead of hardcoded phrase matching
                # NOTE: Use the LAST portion of the response for intent classification
                # because asking_input patterns like "Would you like me to..." typically
                # appear at the END of a long response, and would be diluted if we use full_content
                intent_text = full_content or ""
                if len(intent_text) > 500:
                    # Use last 500 chars for intent detection (captures ending questions)
                    intent_text = intent_text[-500:]
                intent_result = self.intent_classifier.classify_intent_sync(intent_text)

                # Only consider continuation if semantically matches continuation intent
                # and does NOT match completion intent
                intends_to_continue = intent_result.intent == IntentType.CONTINUATION
                is_completion = intent_result.intent == IntentType.COMPLETION
                is_asking_input = intent_result.intent == IntentType.ASKING_INPUT

                logger.debug(
                    f"Intent classification: {intent_result.intent.name} "
                    f"(confidence={intent_result.confidence:.3f}, "
                    f"text_len={len(intent_text)}, "
                    f"top_matches={intent_result.top_matches[:3]})"
                )

                # Track continuation prompts to avoid infinite loops
                if not hasattr(self, "_continuation_prompts"):
                    self._continuation_prompts = 0

                # Track asking input prompts separately
                if not hasattr(self, "_asking_input_prompts"):
                    self._asking_input_prompts = 0

                # Handle model asking for user input (e.g., "Would you like me to...")
                # Behavior depends on one_shot_mode:
                # - one_shot_mode=True: auto-continue with "yes" response
                # - one_shot_mode=False (interactive): return to user to let them choose
                max_asking_input_prompts = 3  # Prevent infinite loops
                one_shot_mode = getattr(self.settings, "one_shot_mode", False)

                if is_asking_input:
                    if one_shot_mode and self._asking_input_prompts < max_asking_input_prompts:
                        # One-shot mode: auto-continue with recommended approach
                        self._asking_input_prompts += 1
                        logger.info(
                            f"Model asking for user input (one-shot) - auto-continuing "
                            f"(attempt {self._asking_input_prompts}/{max_asking_input_prompts})"
                        )
                        # Inject a "yes, continue" response as if the user said yes
                        self.add_message(
                            "user",
                            "Yes, please continue with the implementation. "
                            "Proceed with the most sensible approach based on what you've learned.",
                        )
                        # Continue the loop with the user's affirmative response
                        continue
                    elif not one_shot_mode:
                        # Interactive mode: let user see the question and respond
                        logger.info("Model asking for user input (interactive) - returning to user")
                        # Don't add any message - let the user respond
                        # Break out to yield final chunk and let user see the question
                        break

                # If the model seems to want to continue, prompt it to make a tool call
                # Allow more continuation prompts for complex tasks, but enable for ALL tasks
                # when the model explicitly indicates continuation intent
                requires_continuation_support = is_analysis_task or is_action_task or intends_to_continue
                max_continuation_prompts = 10 if is_analysis_task else (5 if is_action_task else 3)
                budget_threshold = (
                    self.tool_budget // 4 if requires_continuation_support else self.tool_budget // 2
                )
                max_iterations = self.unified_tracker.config.get("max_total_iterations", 50)
                iteration_threshold = (
                    max_iterations * 3 // 4 if requires_continuation_support else max_iterations // 2
                )

                # Also continue if intent is NEUTRAL and response looks incomplete
                # (short content without structured output)
                # Note: Require substantial content (>200 chars) with a structural marker
                # to consider it "complete" - a lone "2. " doesn't count as structured output
                has_substantial_structured_content = content_length > 200 and any(
                    marker in (full_content or "")
                    for marker in ["## ", "**Summary", "**Strengths", "**Weaknesses"]
                )
                content_looks_incomplete = content_length < 500 and not has_substantial_structured_content
                should_prompt_continuation = intends_to_continue or (
                    intent_result.intent == IntentType.NEUTRAL
                    and content_looks_incomplete
                    and not is_completion
                )

                if (
                    requires_continuation_support
                    and should_prompt_continuation
                    and self.tool_calls_used < budget_threshold
                    and self.unified_tracker.iterations < iteration_threshold
                    and self._continuation_prompts < max_continuation_prompts
                ):
                    self._continuation_prompts += 1
                    logger.info(
                        f"Model indicated continuation intent without tool call - prompting to continue "
                        f"(attempt {self._continuation_prompts}/{max_continuation_prompts})"
                    )
                    self.add_message(
                        "system",
                        "You said you would examine more files but did not call any tool. "
                        "Either:\n"
                        "1. Call list_directory to explore a directory, OR\n"
                        "2. Call read_file to read a specific file, OR\n"
                        "3. Provide your analysis NOW if you have enough information.\n\n"
                        "Make a tool call or provide your summary.",
                    )
                    # Track this turn without tool call for productivity ratio calculation
                    self.unified_tracker.increment_turn()
                    # Continue the loop to give the model another chance
                    continue

                # If we hit the continuation limit, ask for a summary instead
                if (
                    requires_continuation_support
                    and self._continuation_prompts >= max_continuation_prompts
                    and self.tool_calls_used > 0
                ):
                    logger.info(
                        f"Max continuation prompts ({max_continuation_prompts}) reached, requesting summary"
                    )
                    self.add_message(
                        "system",
                        "Please complete the task NOW based on what you have done so far. "
                        "Provide a summary of your progress and any remaining steps.",
                    )
                    # One more iteration to get the summary
                    self._continuation_prompts = 99  # Prevent further prompting
                    continue

                # For tasks with tool calls but incomplete output, request completion
                if (
                    requires_continuation_support
                    and self.tool_calls_used > 0
                    and content_looks_incomplete
                    and not is_completion
                    and not hasattr(self, "_final_summary_requested")
                ):
                    self._final_summary_requested = True
                    logger.info(
                        "Analysis task exiting with incomplete output - requesting final summary"
                    )
                    self.add_message(
                        "system",
                        "You have examined several files. Please provide a complete summary "
                        "of your analysis including:\n"
                        "1. **Strengths** - What the codebase does well\n"
                        "2. **Weaknesses** - Areas that need improvement\n"
                        "3. **Recommendations** - Specific suggestions for improvement\n\n"
                        "Provide your analysis NOW.",
                    )
                    continue

                # No more tool calls requested; finish
                # Display performance metrics
                elapsed_time = time.time() - start_time

                # Build token usage summary
                # Use actual provider-reported usage if available, else use estimate
                if cumulative_usage["total_tokens"] > 0:
                    # Provider-reported tokens (accurate)
                    input_tokens = cumulative_usage["prompt_tokens"]
                    output_tokens = cumulative_usage["completion_tokens"]
                    display_tokens = cumulative_usage["total_tokens"]
                    cache_read = cumulative_usage.get("cache_read_input_tokens", 0)
                    cache_create = cumulative_usage.get("cache_creation_input_tokens", 0)

                    # Build metrics line
                    tokens_per_second = output_tokens / elapsed_time if elapsed_time > 0 else 0
                    metrics_parts = [
                        f"📊 in={input_tokens:,}",
                        f"out={output_tokens:,}",
                    ]
                    if cache_read > 0:
                        metrics_parts.append(f"cached={cache_read:,}")
                    if cache_create > 0:
                        metrics_parts.append(f"cache_new={cache_create:,}")
                    metrics_parts.extend([
                        f"| {elapsed_time:.1f}s",
                        f"| {tokens_per_second:.1f} tok/s",
                    ])
                    metrics_line = " ".join(metrics_parts)
                else:
                    # Fallback to estimate
                    tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
                    metrics_line = (
                        f"📊 ~{total_tokens:.0f} tokens (est.) | "
                        f"{elapsed_time:.1f}s | {tokens_per_second:.1f} tok/s"
                    )

                yield StreamChunk(content=f"\n\n{metrics_line}\n")
                yield StreamChunk(content="", is_final=True)
                break

            remaining = max(0, self.tool_budget - self.tool_calls_used)

            # Warn when approaching budget limit
            warning_threshold = getattr(self.settings, "tool_call_budget_warning_threshold", 250)
            if self.tool_calls_used >= warning_threshold and remaining > 0:
                yield StreamChunk(
                    content=f"[tool] ⚠ Approaching tool budget limit: {self.tool_calls_used}/{self.tool_budget} calls used\n"
                )

            if remaining <= 0:
                yield StreamChunk(
                    content=f"[tool] ⚠ Tool budget reached ({self.tool_budget}); skipping tool calls.\n"
                )
                # Try to generate a final summary before exiting
                yield StreamChunk(content="Generating final summary...\n")
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
                    yield StreamChunk(content="Unable to generate summary due to budget limit.\n")

                # Finalize and display performance metrics
                final_metrics = self._finalize_stream_metrics()
                elapsed_time = (
                    final_metrics.total_duration if final_metrics else time.time() - start_time
                )
                tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
                ttft_info = ""
                if final_metrics and final_metrics.time_to_first_token:
                    ttft_info = f" | TTFT: {final_metrics.time_to_first_token:.2f}s"
                yield StreamChunk(
                    content=f"\n📊 {total_tokens:.0f} tokens | {elapsed_time:.1f}s | {tokens_per_second:.1f} tok/s{ttft_info}\n",
                    is_final=True,
                )
                break

            # Force final response after too many consecutive tool calls without output
            # This prevents endless tool call loops
            # For analysis/action tasks, allow significantly more tool calls
            base_max_consecutive = 8
            if is_analysis_task:
                base_max_consecutive = 50  # Analysis needs many tool calls to explore codebase
            elif is_action_task:
                base_max_consecutive = 30  # Action tasks (web search, multi-step) need flexibility
            max_consecutive_tool_calls = getattr(
                self.settings, "max_consecutive_tool_calls", base_max_consecutive
            )
            if self.tool_calls_used >= max_consecutive_tool_calls and not force_completion:
                # Check if we've been making progress (reading new files)
                # For analysis/action tasks, use a more lenient progress threshold
                # Action tasks may do web searches, directory listings, bash commands - not just file reads
                requires_lenient_progress = is_analysis_task or is_action_task
                progress_threshold = (
                    self.tool_calls_used // 4 if requires_lenient_progress else self.tool_calls_used // 2
                )
                if len(unique_resources) < progress_threshold:
                    # Not making good progress - force completion
                    logger.warning(
                        f"Forcing completion: {self.tool_calls_used} tool calls but only "
                        f"{len(unique_resources)} unique resources (threshold: {progress_threshold})"
                    )
                    force_completion = True

            # Force completion if too many low-output iterations or research calls
            if force_completion:
                # Check stop reason from unified tracker to determine message type
                stop_decision = self.unified_tracker.should_stop()
                is_research_loop = (
                    stop_decision.reason.value == "loop_detected"
                    and "research" in stop_decision.hint.lower()
                )

                if is_research_loop:
                    yield StreamChunk(
                        content="[tool] ⚠ Research loop detected - forcing synthesis\n"
                    )
                    self.add_message(
                        "system",
                        "You have performed multiple consecutive research/web searches. "
                        "STOP searching now. Instead, SYNTHESIZE and ANALYZE the information you've already gathered. "
                        "Provide your FINAL ANSWER based on the search results you have collected. "
                        "Answer all parts of the user's question comprehensively.",
                    )
                else:
                    yield StreamChunk(
                        content="⚠️ Reached exploration limit - summarizing findings...\n"
                    )
                    self.add_message(
                        "system",
                        "You have made multiple tool calls without providing substantial analysis. "
                        "STOP using tools now. Instead, provide your FINAL COMPREHENSIVE ANSWER based on "
                        "the information you have already gathered. Answer all parts of the user's question.",
                    )

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
                    yield StreamChunk(
                        content="Unable to generate final summary. Please try a simpler query."
                    )
                break  # Exit the loop after forcing final response

            tool_calls = tool_calls[:remaining]

            # Filter out tool calls that are blocked after loop warning
            # After warning, the same signature cannot be attempted again
            filtered_tool_calls = []
            for tc in tool_calls:
                tc_name = tc.get("name", "")
                tc_args = tc.get("arguments", {})
                block_reason = self.unified_tracker.is_blocked_after_warning(tc_name, tc_args)
                if block_reason:
                    yield StreamChunk(
                        content=f"\n[loop] ⛔ {block_reason}\n"
                    )
                    # Inject message to guide model to different approach
                    self.add_message(
                        "system",
                        f"BLOCKED: {block_reason}\n"
                        "You MUST try a different approach. Do NOT repeat the same operation.\n"
                        "Either: 1) Use a different tool, 2) Use different parameters, or "
                        "3) Provide your final response without further tool calls.",
                    )
                else:
                    filtered_tool_calls.append(tc)

            if not filtered_tool_calls and tool_calls:
                # All tool calls were blocked - continue loop for model to respond
                continue

            tool_calls = filtered_tool_calls

            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "tool")
                tool_args = tool_call.get("arguments", {})
                # Clean user-friendly message without internal [tool] prefix
                # Show relevant context for observability (command, path, file, etc.)
                if tool_name == "execute_bash" and "command" in tool_args:
                    cmd = tool_args["command"]
                    # Truncate long commands for readability
                    cmd_display = cmd[:80] + "..." if len(cmd) > 80 else cmd
                    yield StreamChunk(
                        content="",
                        metadata={"status": f"🔧 Running {tool_name}: `{cmd_display}`"},
                    )
                elif tool_name == "list_directory":
                    path = tool_args.get("path", ".")
                    yield StreamChunk(
                        content="",
                        metadata={"status": f"🔧 Listing directory: {path}"},
                    )
                elif tool_name == "read_file":
                    path = tool_args.get("path", "file")
                    yield StreamChunk(
                        content="",
                        metadata={"status": f"🔧 Reading file: {path}"},
                    )
                elif tool_name == "edit_files":
                    files = tool_args.get("files", [])
                    if files and isinstance(files, list):
                        paths = [f.get("path", "?") for f in files[:3]]
                        path_display = ", ".join(paths)
                        if len(files) > 3:
                            path_display += f" (+{len(files) - 3} more)"
                        yield StreamChunk(
                            content="",
                            metadata={"status": f"🔧 Editing: {path_display}"},
                        )
                    else:
                        yield StreamChunk(
                            content="",
                            metadata={"status": f"🔧 Running {tool_name}..."},
                        )
                elif tool_name == "write_file":
                    path = tool_args.get("path", "file")
                    yield StreamChunk(
                        content="",
                        metadata={"status": f"🔧 Writing file: {path}"},
                    )
                elif tool_name == "code_search":
                    query = tool_args.get("query", "")
                    query_display = query[:50] + "..." if len(query) > 50 else query
                    yield StreamChunk(
                        content="",
                        metadata={"status": f"🔧 Searching: {query_display}"},
                    )
                else:
                    yield StreamChunk(
                        content="",
                        metadata={"status": f"🔧 Running {tool_name}..."},
                    )

            tool_results = await self._handle_tool_calls(tool_calls)
            for result in tool_results:
                tool_name = result.get("name", "tool")
                elapsed = result.get("elapsed", 0.0)
                if result.get("success"):
                    yield StreamChunk(
                        content="",
                        metadata={"status": f"✓ {tool_name} ({elapsed:.1f}s)"},
                    )
                    # Show content preview for write/edit operations (like Claude Code)
                    tool_args = result.get("args", {})
                    if tool_name == "write_file" and tool_args.get("content"):
                        content = tool_args["content"]
                        lines = content.split("\n")
                        preview_lines = 8  # Show first N lines
                        if len(lines) > preview_lines:
                            preview = "\n".join(lines[:preview_lines])
                            preview += f"\n... ({len(lines) - preview_lines} more lines)"
                        else:
                            preview = content
                        # Format as a code block preview
                        yield StreamChunk(
                            content="",
                            metadata={"file_preview": preview, "path": tool_args.get("path", "")},
                        )
                    elif tool_name == "edit_files" and tool_args.get("files"):
                        # Show edit operations summary
                        files = tool_args.get("files", [])
                        for file_edit in files[:3]:  # Show first 3 files
                            path = file_edit.get("path", "")
                            edits = file_edit.get("edits", [])
                            for edit in edits[:2]:  # Show first 2 edits per file
                                old_str = edit.get("old_string", "")[:50]
                                new_str = edit.get("new_string", "")[:50]
                                if old_str and new_str:
                                    yield StreamChunk(
                                        content="",
                                        metadata={
                                            "edit_preview": f"- {old_str}...\n+ {new_str}...",
                                            "path": path,
                                        },
                                    )
                else:
                    yield StreamChunk(
                        content="",
                        metadata={"status": f"✗ {tool_name} failed"},
                    )

            yield StreamChunk(content="", metadata={"status": "💭 Thinking..."})

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

            context_msg = full_content or user_message

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

            # Skip unknown tools immediately (no retries, no budget cost)
            if not self.tools.is_tool_enabled(tool_name):
                self.console.print(f"[yellow]⚠ Skipping unknown or disabled tool: {tool_name}[/]")
                continue

            if self.tool_calls_used >= self.tool_budget:
                self.console.print(
                    f"[yellow]⚠ Tool budget reached ({self.tool_budget}); skipping remaining tool calls.[/]"
                )
                break

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
            exec_result = await self.tool_executor.execute(
                tool_name=tool_name,
                arguments=normalized_args,
                context=context,
            )
            success = exec_result.success
            error_msg = exec_result.error

            # Update counters and tracking
            self.tool_calls_used += 1
            self.executed_tools.append(tool_name)
            if tool_name == "read_file" and "path" in normalized_args:
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

            # Record tool execution analytics
            self._record_tool_execution(tool_name, success, elapsed_ms)

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

                # For semantic failures (tool ran but returned success=False), pass the
                # error details back to the model so it can understand and retry
                if success and not semantic_success:
                    # Tool executed but had semantic failure - give model actionable feedback
                    error_output = output if isinstance(output, dict) else {"error": error_display}
                    formatted_error = self._format_tool_output(
                        tool_name, normalized_args, error_output
                    )
                    self.add_message("user", formatted_error)

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
        """
        self.conversation.clear()
        self._system_added = False

        # Reset session-specific state to prevent memory leaks
        self.tool_calls_used = 0
        self.failed_tool_signatures.clear()
        self.observed_files.clear()
        self.executed_tools.clear()

        # Reset conversation state machine
        if hasattr(self, "conversation_state"):
            self.conversation_state.reset()

        # Reset context reminder manager
        if hasattr(self, "reminder_manager"):
            self.reminder_manager.reset()

        logger.debug("Conversation and session state reset")

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
                logger.warning(f"Session not found: {session_id}")
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
            max_tokens: Override max tokens for this retrieval

        Returns:
            List of messages in provider format
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
            logger.debug(f"Cancelling {len(self._background_tasks)} background task(s)...")
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
                logger.warning(f"Error closing provider: {e}")

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
                logger.warning(f"Error closing semantic selector: {e}")

        logger.info("AgentOrchestrator shutdown complete")

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
            raise ValueError(f"Profile not found: {profile_name}")

        # Get provider settings
        provider_settings = settings.get_provider_settings(profile.provider)

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
        )
