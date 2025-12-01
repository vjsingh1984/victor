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

"""Agent orchestrator for managing conversations and tool execution."""

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
from victor.agent.conversation import ConversationManager
from victor.agent.conversation_state import ConversationStateMachine, ConversationStage
from victor.agent.prompt_builder import SystemPromptBuilder
from victor.agent.response_sanitizer import ResponseSanitizer
from victor.agent.stream_handler import StreamMetrics
from victor.agent.tool_selection import (
    CORE_TOOLS,
    ToolSelector,
)
from victor.agent.tool_calling import (
    ToolCallingAdapterRegistry,
    ToolCallParseResult,
)
from victor.agent.tool_executor import ToolExecutor
from victor.analytics.logger import UsageLogger
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
from victor.tools.plugin_manager import ToolPluginManager
from victor.tools.semantic_selector import SemanticToolSelector
from victor.tools.web_search_tool import set_web_search_provider, set_web_tool_defaults
from victor.workflows.base import WorkflowRegistry
from victor.workflows.new_feature_workflow import NewFeatureWorkflow

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrates agent interactions, tool execution, and provider communication."""

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

        # Load project context from .victor.md (similar to Claude Code's CLAUDE.md)
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
        default_budget = self.tool_calling_caps.recommended_tool_budget
        self.tool_budget = getattr(settings, "tool_call_budget", default_budget)
        self.tool_calls_used = 0
        self.observed_files: List[str] = []
        self.executed_tools: List[str] = []
        self.failed_tool_signatures: set[tuple[str, str]] = set()
        self._tool_capability_warned = False

        # Analytics
        log_file = Path(self.settings.analytics_log_file).expanduser()
        self.usage_logger = UsageLogger(log_file, enabled=self.settings.analytics_enabled)
        self.usage_logger.log_event(
            "session_start", {"model": self.model, "provider": self.provider.__class__.__name__}
        )

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

            cache_dir = Path(
                getattr(self.settings, "tool_cache_dir", "~/.victor/cache")
            ).expanduser()
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

        # Conversation history (using ConversationManager for better encapsulation)
        self.conversation = ConversationManager(
            system_prompt=self._system_prompt,
            max_history_messages=getattr(settings, "max_conversation_history", 100),
        )

        # Conversation state machine for intelligent stage detection
        self.conversation_state = ConversationStateMachine()

        # Tool registry
        self.tools = ToolRegistry()
        self._register_default_tools()
        self._load_tool_configurations()  # Load tool enable/disable states from config
        self.tools.register_before_hook(self._log_tool_call)

        # Plugin system for extensible tools
        self.plugin_manager: Optional[ToolPluginManager] = None
        if getattr(settings, "plugin_enabled", True):
            self._initialize_plugins()

        # Argument normalizer for handling malformed tool arguments (e.g., Python vs JSON syntax)
        provider_name = provider.__class__.__name__ if provider else "unknown"
        self.argument_normalizer = ArgumentNormalizer(provider_name=provider_name)

        # Tool executor for centralized tool execution with retry, caching, and metrics
        self.tool_executor = ToolExecutor(
            tool_registry=self.tools,
            argument_normalizer=self.argument_normalizer,
            tool_cache=self.tool_cache,
            max_retries=getattr(settings, "tool_retry_max_attempts", 3),
            retry_delay=getattr(settings, "tool_retry_base_delay", 1.0),
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

        # Initialize unified ToolSelector (handles semantic + keyword selection)
        self.tool_selector = ToolSelector(
            tools=self.tools,
            semantic_selector=self.semantic_selector,
            conversation_state=self.conversation_state,
            model=self.model,
            provider_name=self.provider_name,
            tool_selection_config=self.tool_selection,
            fallback_max_tools=getattr(settings, "fallback_max_tools", 8),
            on_selection_recorded=self._record_tool_selection,
        )

    @property
    def messages(self) -> List[Message]:
        """Get conversation messages (backward compatibility property).

        Returns:
            List of messages in conversation history
        """
        return self.conversation.messages

    def _init_stream_metrics(self) -> StreamMetrics:
        """Initialize fresh stream metrics for a new streaming session."""
        self._current_stream_metrics = StreamMetrics(start_time=time.time())
        return self._current_stream_metrics

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
            return metrics
        return None

    def get_last_stream_metrics(self) -> Optional[StreamMetrics]:
        """Get metrics from the last streaming session."""
        return getattr(self, "_current_stream_metrics", None)

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

    def start_embedding_preload(self) -> None:
        """Start background embedding preload task.

        Should be called after orchestrator initialization to avoid blocking
        the main thread. Safe to call multiple times (no-op if already started).
        """
        if not self.use_semantic_selection or self._embedding_preload_task:
            return

        import asyncio

        try:
            # Create background task for embedding preload
            loop = asyncio.get_event_loop()
            self._embedding_preload_task = loop.create_task(self._preload_embeddings())
            logger.info("Started background task for embedding preload")
        except RuntimeError:
            # No event loop available yet, will load on first query
            logger.debug("No event loop available for background preload, will load on first query")

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

    def _log_tool_call(self, name: str, kwargs: dict) -> None:
        """A hook that logs information before a tool is called."""
        self.console.print(f"[dim]Attempting to call tool '{name}' with arguments: {kwargs}[/dim]")

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

        for filename in os.listdir(tools_dir):
            if filename.endswith(".py") and filename not in excluded_files:
                module_name = f"victor.tools.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for _name, obj in inspect.getmembers(module):
                        if inspect.isfunction(obj) and getattr(obj, "_is_tool", False):
                            self.tools.register(obj)
                            logger.debug(
                                f"Dynamically registered tool: {obj.__name__} from {module_name}"
                            )
                except Exception as e:
                    logger.warning(f"Failed to load tools from {module_name}: {e}")

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
                asyncio.create_task(self._start_mcp_registry())
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
                asyncio.create_task(mcp_client.connect(cmd_parts))
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
            self.tool_graph.add_tool(
                "plan_files",
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
            plugin_dirs = [
                Path(d).expanduser()
                for d in getattr(self.settings, "plugin_dirs", ["~/.victor/plugins"])
            ]
            plugin_config = getattr(self.settings, "plugin_config", {})
            disabled_plugins = set(getattr(self.settings, "plugin_disabled", []))

            self.plugin_manager = ToolPluginManager(
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
        if role == "user":
            self.usage_logger.log_event("user_prompt", {"content": content})
        elif role == "assistant":
            self.usage_logger.log_event("assistant_response", {"content": content})

    def _ensure_system_message(self) -> None:
        """Ensure the system prompt is included once at the start of the conversation."""
        self.conversation.ensure_system_prompt()
        self._system_added = True

    async def chat(self, user_message: str) -> CompletionResponse:
        """Send a chat message and get response.

        Args:
            user_message: User's message

        Returns:
            CompletionResponse from the model
        """
        self._ensure_system_message()
        # Add user message to history
        self.add_message("user", user_message)

        # Get tool definitions if provider supports them
        # Intelligently select relevant tools based on the user's message
        tools = None
        if self.provider.supports_tools():
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

        # Get response from provider
        # Prepare optional thinking parameter for providers that support it (Anthropic)
        provider_kwargs = {}
        if self.thinking:
            # Anthropic extended thinking format
            provider_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}

        response = await self.provider.chat(
            messages=self.messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=tools,
            **provider_kwargs,
        )

        # Add assistant response to history
        self.add_message("assistant", response.content)

        # Handle tool calls if present
        if response.tool_calls:
            await self._handle_tool_calls(response.tool_calls)

        return response

    async def stream_chat(self, user_message: str) -> AsyncIterator[StreamChunk]:
        """Stream a chat response.

        Args:
            user_message: User's message

        Yields:
            StreamChunk objects with incremental response

        Note:
            Stream metrics (TTFT, throughput) are available via get_last_stream_metrics()
            after the stream completes.
        """
        # Track performance metrics using StreamMetrics
        stream_metrics = self._init_stream_metrics()
        start_time = stream_metrics.start_time
        total_tokens: float = 0

        self._ensure_system_message()
        self.tool_calls_used = 0
        self.observed_files = []
        self.executed_tools = []
        self.failed_tool_signatures = set()
        # Add user message to history
        self.add_message("user", user_message)

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
        ]
        is_analysis_task = any(keyword in user_message.lower() for keyword in analysis_keywords)

        # Track exploration without output (prevents endless exploration loops)
        consecutive_low_output_iterations = 0

        # Use configurable limits from settings, adjusted by adapter capabilities
        # Local models with strict prompting needs get lower limits
        needs_strict = self.tool_calling_caps.requires_strict_prompting

        # Base limits - reduced for models needing strict prompting
        base_analysis_limit = 10 if needs_strict else 15
        base_action_limit = 4 if needs_strict else 6
        base_default_limit = 3 if needs_strict else 5

        if is_analysis_task:
            max_low_output_iterations = getattr(
                self.settings, "max_exploration_iterations_analysis", base_analysis_limit
            )
            logger.info(
                f"Detected analysis task - allowing up to {max_low_output_iterations} exploration iterations"
            )
        elif is_action_task:
            max_low_output_iterations = getattr(
                self.settings, "max_exploration_iterations_action", base_action_limit
            )
        else:
            max_low_output_iterations = getattr(
                self.settings, "max_exploration_iterations", base_default_limit
            )

        # Hard limit on total loop iterations to prevent runaway loops
        # Lower for models without native tool calling
        base_total_limit = 15 if needs_strict else 20
        max_total_iterations = getattr(self.settings, "max_total_loop_iterations", base_total_limit)
        total_iterations = 0

        min_content_threshold = getattr(self.settings, "min_content_threshold", 150)
        force_completion = False

        # Track unique files read and tool calls for smarter loop detection
        unique_files_read: set = set()
        recent_tool_calls: list = []  # Track last N tool calls for loop detection
        max_recent_calls = 10  # Window for detecting repeated calls

        # Track consecutive research calls (prevents endless research loops)
        consecutive_research_calls = 0
        max_research_iterations = getattr(self.settings, "max_research_iterations", 6)

        # Add guidance for action-oriented tasks
        if is_action_task:
            logger.info(
                f"Detected action-oriented task - allowing up to {max_low_output_iterations} exploration iterations"
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

        while True:
            # Check hard iteration limit
            total_iterations += 1
            if total_iterations > max_total_iterations:
                logger.warning(
                    f"Hard iteration limit reached ({max_total_iterations}). Forcing completion."
                )
                yield StreamChunk(
                    content=f"\n[tool] ⚠ Maximum iterations ({max_total_iterations}) reached. Providing summary.\n"
                )
                # Force a final response without tools
                self.add_message(
                    "system",
                    "CRITICAL: You have reached the maximum number of iterations. "
                    "STOP ALL TOOL CALLS immediately. Provide a brief summary of what you found "
                    "based on the information gathered so far. Do NOT attempt any more tool calls.",
                )
                # Make one final call without tools
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
                # No content and no tool calls; attempt a non-stream fallback for a textual answer
                try:
                    response = await self.provider.chat(
                        messages=self.messages,
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        tools=None,
                    )
                    if response and response.content:
                        self.add_message("assistant", response.content)
                        yield StreamChunk(content=response.content, is_final=True)
                        return
                except Exception as exc:
                    logger.warning(f"Non-stream fallback failed: {exc}")
                # If fallback also failed, surface a minimal message
                fallback_msg = "No tool calls were returned and the model provided no content. Please retry or simplify the request."
                yield StreamChunk(content=fallback_msg, is_final=True)
                return

            # Track exploration without substantial output
            # Distinguish between exploration tools (read-only), research tools (web), and action tools (write/execute)
            exploration_tools = {
                "list_directory",
                "read_file",
                "plan_files",
                "code_search",
                "find_symbol",
                "find_references",
            }
            research_tools = {"web_search", "web_fetch"}
            action_tools = {
                "write_file",
                "execute_bash",
                "edit_files",
                "run_tests",
                "git_suggest_commit",
            }

            # Check if this iteration used only exploration tools or research tools
            tool_names = [tc.get("name") for tc in (tool_calls or [])]
            only_exploration = (
                all(name in exploration_tools for name in tool_names) if tool_names else False
            )
            has_research = (
                any(name in research_tools for name in tool_names) if tool_names else False
            )

            # Track unique files being read for progress detection
            for tc in tool_calls or []:
                if tc.get("name") == "read_file":
                    file_path = tc.get("arguments", {}).get("path", "")
                    if file_path:
                        unique_files_read.add(file_path)
                elif tc.get("name") == "list_directory":
                    dir_path = tc.get("arguments", {}).get("path", "")
                    if dir_path:
                        unique_files_read.add(f"dir:{dir_path}")

            # Track recent tool calls for loop detection
            call_signature = tuple(
                sorted(
                    (tc.get("name", ""), str(tc.get("arguments", {}))) for tc in (tool_calls or [])
                )
            )
            recent_tool_calls.append(call_signature)
            if len(recent_tool_calls) > max_recent_calls:
                recent_tool_calls.pop(0)

            # Detect true loops: same tool call signature repeated 3+ times in recent history
            # Also detect semantic loops: same tool names with slightly different args
            is_true_loop = False
            if len(recent_tool_calls) >= 3:
                last_call = recent_tool_calls[-1]
                repeat_count = sum(1 for c in recent_tool_calls[-5:] if c == last_call)
                if repeat_count >= 3:
                    is_true_loop = True
                    logger.warning(
                        f"True loop detected: same tool call repeated {repeat_count} times"
                    )

                # Detect semantic loops: same tools called repeatedly
                if not is_true_loop and len(recent_tool_calls) >= 4:
                    # Extract just tool names from signatures
                    recent_tool_names = []
                    for sig in recent_tool_calls[-6:]:
                        if sig:
                            names = [name for name, _ in sig]
                            recent_tool_names.append(tuple(sorted(names)))

                    # Check if the same set of tools keeps being called
                    if len(recent_tool_names) >= 4:
                        last_tools = recent_tool_names[-1]
                        same_tools_count = sum(1 for t in recent_tool_names[-4:] if t == last_tools)
                        if same_tools_count >= 3:
                            is_true_loop = True
                            logger.warning(
                                f"Semantic loop detected: same tools ({last_tools}) called {same_tools_count} times"
                            )

            content_length = len(full_content.strip())

            # Smart exploration tracking:
            # - For analysis tasks: only force completion on true loops, not just iteration count
            # - For other tasks: use iteration count but track progress
            if is_analysis_task:
                # For analysis tasks, only force completion on true loops
                if is_true_loop:
                    force_completion = True
                    logger.warning("Forcing completion due to detected loop in analysis task")
                # Log progress for analysis tasks
                if tool_calls and only_exploration:
                    logger.debug(
                        f"Analysis exploration: {len(unique_files_read)} unique files examined"
                    )
            elif content_length < min_content_threshold and tool_calls and only_exploration:
                # For non-analysis tasks, check if we're making progress (reading new files)
                progress_made = len(unique_files_read) > consecutive_low_output_iterations
                if not progress_made or is_true_loop:
                    consecutive_low_output_iterations += 1
                    logger.debug(
                        f"Low output exploration iteration {consecutive_low_output_iterations}/{max_low_output_iterations} (content length: {content_length}, tools: {tool_names}, unique files: {len(unique_files_read)})"
                    )

                    if consecutive_low_output_iterations >= max_low_output_iterations:
                        force_completion = True
                        logger.warning(
                            f"Forcing completion after {consecutive_low_output_iterations} exploration iterations with minimal output"
                        )
                else:
                    logger.debug(
                        f"Progress detected: {len(unique_files_read)} unique files read - not counting as low-output iteration"
                    )
            else:
                # Reset counter when substantial output is produced OR action tools are used
                if content_length >= min_content_threshold or any(
                    name in action_tools for name in tool_names
                ):
                    consecutive_low_output_iterations = 0

            # Track consecutive research calls (prevents endless web search loops)
            if has_research:
                consecutive_research_calls += 1
                logger.debug(
                    f"Research iteration {consecutive_research_calls}/{max_research_iterations} (tools: {tool_names})"
                )

                if consecutive_research_calls >= max_research_iterations:
                    force_completion = True
                    logger.warning(
                        f"Forcing synthesis after {consecutive_research_calls} consecutive research calls"
                    )
            else:
                # Reset research counter when non-research tools are used
                consecutive_research_calls = 0

            logger.debug(f"After streaming pass, tool_calls = {tool_calls}")

            if not tool_calls:
                # No more tool calls requested; finish
                # Display performance metrics
                elapsed_time = time.time() - start_time
                tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
                yield StreamChunk(
                    content=f"\n\n📊 {total_tokens:.0f} tokens | {elapsed_time:.1f}s | {tokens_per_second:.1f} tok/s\n"
                )
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
            max_consecutive_tool_calls = getattr(self.settings, "max_consecutive_tool_calls", 8)
            if self.tool_calls_used >= max_consecutive_tool_calls and not force_completion:
                # Check if we've been making progress (reading new files)
                if len(unique_files_read) < self.tool_calls_used // 2:
                    # Not making good progress - force completion
                    logger.warning(
                        f"Forcing completion: {self.tool_calls_used} tool calls but only "
                        f"{len(unique_files_read)} unique files read"
                    )
                    force_completion = True

            # Force completion if too many low-output iterations or research calls
            if force_completion:
                if consecutive_research_calls >= max_research_iterations:
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
                    context_msg = "Synthesize search results and provide final answer now"
                else:
                    yield StreamChunk(
                        content="[tool] ⚠ Exploration loop detected - forcing final summary\n"
                    )
                    self.add_message(
                        "system",
                        "You have made multiple tool calls without providing substantial analysis. "
                        "STOP using tools now. Instead, provide your FINAL COMPREHENSIVE ANSWER based on "
                        "the information you have already gathered. Answer all parts of the user's question.",
                    )
                    context_msg = "Provide final comprehensive answer now"

                # Clear tool calls to force final response
                tool_calls = []
                # Continue to next iteration which will generate final response without tools
                continue

            tool_calls = tool_calls[:remaining]
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "tool")
                yield StreamChunk(content=f"[tool] … {tool_name} started\n")

            tool_results = await self._handle_tool_calls(tool_calls)
            for result in tool_results:
                tool_name = result.get("name", "tool")
                elapsed = result.get("elapsed", 0.0)
                status = "ok" if result.get("success") else "failed"
                yield StreamChunk(content=f"[tool] ✓ {tool_name} {status} ({elapsed:.1f}s)\n")

            yield StreamChunk(content="Generating final response...\n")

            if self.observed_files:
                evidence_note = "Evidence files read: " + ", ".join(
                    sorted(set(self.observed_files))
                )
                self.add_message(
                    "system",
                    evidence_note + " Only cite these or tool outputs; do not invent paths.",
                )
            else:
                self.add_message(
                    "system",
                    "No files read yet. Avoid file-specific claims; suggest using tools if needed.",
                )

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
                                    "plan_files",
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

            self.console.print(f"\n[bold cyan]Executing tool:[/] {tool_name}")

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

            # Calculate execution time
            elapsed_ms = (time.monotonic() - start) * 1000  # Convert to milliseconds

            # Record tool execution analytics
            self._record_tool_execution(tool_name, success, elapsed_ms)

            # Update conversation state machine for stage detection
            self.conversation_state.record_tool_execution(tool_name, normalized_args)

            # ToolExecutionResult stores actual output in .result field
            output = exec_result.result if success else None
            # Only set error_display for failures; keep None for successes
            error_display = None if success else (error_msg or "Unknown error")

            self.usage_logger.log_event(
                "tool_result",
                {
                    "tool_name": tool_name,
                    "success": success,
                    "result": output,
                    "error": error_display,
                },
            )

            if success:
                self.console.print(
                    f"[green]✓ Tool executed successfully[/] [dim]({elapsed_ms:.0f}ms)[/dim]"
                )
                # Add tool result to conversation
                self.add_message(
                    "user",
                    f"Tool '{tool_name}' result: {output}",
                )
                results.append(
                    {
                        "name": tool_name,
                        "success": True,
                        "elapsed": time.monotonic() - start,
                    }
                )
            else:
                self.failed_tool_signatures.add(signature)
                # error_display was already set above from exec_result.error
                self.console.print(
                    f"[red]✗ Tool execution failed: {error_display}[/] [dim]({elapsed_ms:.0f}ms)[/dim]"
                )
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
        """Clear conversation history."""
        self.conversation.clear()
        self._system_added = False

    async def shutdown(self) -> None:
        """Clean up resources and shutdown gracefully.

        Should be called when the orchestrator is no longer needed.
        Cleans up:
        - Provider connections
        - Code execution manager (Docker containers)
        - Semantic selector resources
        - HTTP clients
        """
        logger.info("Shutting down AgentOrchestrator...")

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
