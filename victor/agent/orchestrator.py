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
import re
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from rich.console import Console

from victor.agent.argument_normalizer import ArgumentNormalizer, NormalizationStrategy
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
from victor.tools.base import ToolRegistry
from victor.tools.code_executor_tool import CodeExecutionManager
from victor.tools.code_review_tool import set_code_review_config
from victor.tools.dependency_graph import ToolDependencyGraph
from victor.tools.git_tool import set_git_provider
from victor.tools.mcp_bridge_tool import configure_mcp_client, get_mcp_tool_definitions
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

        # Load project context from .victor.md (similar to Claude Code's CLAUDE.md)
        self.project_context = ProjectContext()
        self.project_context.load()

        # Build system prompt with project context
        base_system_prompt = (
            "You are a code analyst for this repository. Follow this loop:\n"
            "1) Plan briefly, then call list_directory/read_file to inspect real files (not imagined). "
            "2) If the user names multiple files, read ALL of them before proposing or concluding. "
            "3) Keep issuing tool calls iteratively until the task is complete or the tool budget is exhausted—do not stop after the first tool call. "
            "4) If the user asks to modify or apply changes, propose a concise plan and then execute write_file/edit_files to apply the edits and show diffs. "
            "4) Do NOT invent tool outputs—only cite results from tools you actually called. "
            "5) When asked for web info, use web_search/web_summarize; when asked about repo code, use read_file/analyze_docs/code_review.\n"
            "Always report findings with file paths. Avoid generic advice. Stop once you hit the tool budget."
        )

        # Inject project context if available
        if self.project_context.content:
            self._system_prompt = (
                base_system_prompt + "\n\n" + self.project_context.get_system_prompt_addition()
            )
            logger.info(f"Loaded project context from {self.project_context.context_file}")
        else:
            self._system_prompt = base_system_prompt

        self._system_added = False
        self.tool_budget = getattr(settings, "tool_call_budget", 20)
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
        # Minimal dependency graph
        self.tool_graph = ToolDependencyGraph()
        self._register_default_tool_dependencies()

        # Stateful managers
        self.code_manager = CodeExecutionManager()
        self.code_manager.start()

        # Workflow registry
        self.workflow_registry = WorkflowRegistry()
        self._register_default_workflows()

        # Conversation history
        self.messages: List[Message] = []

        # Tool registry
        self.tools = ToolRegistry()
        self._register_default_tools()
        self._load_tool_configurations()  # Load tool enable/disable states from config
        self.tools.register_before_hook(self._log_tool_call)

        # Argument normalizer for handling malformed tool arguments (e.g., Python vs JSON syntax)
        provider_name = provider.__class__.__name__ if provider else "unknown"
        self.argument_normalizer = ArgumentNormalizer(provider_name=provider_name)

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
            # Initialize embeddings asynchronously in background to avoid blocking first query
            self._embeddings_initialized = False
            self._embedding_preload_task: Optional[asyncio.Task[None]] = None

    def shutdown(self) -> None:
        """Gracefully shut down stateful managers."""
        self.console.print("[dim]Shutting down code execution environment...[/dim]")
        self.code_manager.stop()

    async def _preload_embeddings(self) -> None:
        """Preload tool embeddings in background to avoid blocking first query.

        This is called asynchronously during initialization if semantic tool
        selection is enabled. Errors are logged but don't crash the app.
        """
        if not self.semantic_selector or self._embeddings_initialized:
            return

        try:
            logger.info("Starting background embedding preload...")
            await self.semantic_selector.initialize_tool_embeddings(self.tools)
            self._embeddings_initialized = True
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
        if tool_name not in self._tool_usage_stats:
            self._tool_usage_stats[tool_name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_time_ms": 0.0,
                "avg_time_ms": 0.0,
                "min_time_ms": float("inf"),
                "max_time_ms": 0.0,
            }

        stats = self._tool_usage_stats[tool_name]
        stats["total_calls"] += 1
        stats["successful_calls"] += 1 if success else 0
        stats["failed_calls"] += 0 if success else 1
        stats["total_time_ms"] += elapsed_ms
        stats["avg_time_ms"] = stats["total_time_ms"] / stats["total_calls"]
        stats["min_time_ms"] = min(stats["min_time_ms"], elapsed_ms)
        stats["max_time_ms"] = max(stats["max_time_ms"], elapsed_ms)

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
            - Overall metrics
        """
        return {
            "selection_stats": self._tool_selection_stats,
            "tool_stats": self._tool_usage_stats,
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
        }

    def _parse_json_tool_call_from_content(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """Best-effort parser for JSON tool calls embedded in text content."""
        try:
            if "name" not in content or "{" not in content:
                return None
            match = re.search(r"({.*})", content, re.DOTALL)
            if not match:
                return None
            blob = match.group(1).strip()
            parsed = json.loads(blob)
            calls = parsed if isinstance(parsed, list) else [parsed]
            normalized: List[Dict[str, Any]] = []
            for call in calls:
                if not isinstance(call, dict):
                    continue
                name = call.get("name")
                if not name:
                    continue
                args = call.get("arguments") or call.get("parameters") or {}
                normalized.append(
                    {"id": f"json_call_{len(normalized)+1}", "name": name, "arguments": args}
                )
            return normalized or None
        except Exception:
            return None

    def _parse_xmlish_tool_call_from_content(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """Handle simple XML/HTML-like wrappers that contain JSON."""
        try:
            matches = re.findall(
                r"<function[^>]*>(.*?)</function>", content, re.DOTALL | re.IGNORECASE
            )
            if not matches:
                return None
            normalized: List[Dict[str, Any]] = []
            for body in matches:
                # Attempt to find JSON inside the function block
                json_match = re.search(r"({.*})", body, re.DOTALL)
                if not json_match:
                    continue
                parsed = json.loads(json_match.group(1))
                calls = parsed if isinstance(parsed, list) else [parsed]
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    name = call.get("name")
                    if not name:
                        continue
                    args = call.get("arguments") or call.get("parameters") or {}
                    normalized.append(
                        {"id": f"xml_call_{len(normalized)+1}", "name": name, "arguments": args}
                    )
            return normalized or None
        except Exception:
            return None

    def _strip_markup(self, text: str) -> str:
        """Remove simple XML/HTML-like tags to salvage plain text."""
        if not text:
            return text
        cleaned = re.sub(r"<[^>]+>", " ", text)
        return " ".join(cleaned.split())

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
        # Register MCP tools if configured
        if getattr(self.settings, "use_mcp_tools", False):
            mcp_command = getattr(self.settings, "mcp_command", None)
            if mcp_command:
                try:
                    from victor.mcp.client import MCPClient

                    mcp_client = MCPClient()
                    cmd_parts = mcp_command.split()
                    asyncio.create_task(mcp_client.connect(cmd_parts))
                    configure_mcp_client(
                        mcp_client, prefix=getattr(self.settings, "mcp_prefix", "mcp")
                    )
                except Exception as exc:
                    logger.warning(f"Failed to start MCP client: {exc}")

            for mcp_tool in get_mcp_tool_definitions():
                self.tools.register_dict(mcp_tool)

    def _register_default_tool_dependencies(self) -> None:
        """Register minimal tool input/output specs for planning."""
        try:
            self.tool_graph.add_tool("code_search", inputs=["query"], outputs=["file_candidates"])
            self.tool_graph.add_tool(
                "semantic_code_search", inputs=["query"], outputs=["file_candidates"]
            )
            self.tool_graph.add_tool("plan_files", inputs=["query"], outputs=["file_candidates"])
            self.tool_graph.add_tool(
                "read_file", inputs=["file_candidates"], outputs=["file_contents"]
            )
            self.tool_graph.add_tool("analyze_docs", inputs=["file_contents"], outputs=["summary"])
            self.tool_graph.add_tool("code_review", inputs=["file_contents"], outputs=["summary"])
            self.tool_graph.add_tool(
                "generate_docs", inputs=["file_contents"], outputs=["documentation"]
            )
            self.tool_graph.add_tool(
                "security_scan", inputs=["file_contents"], outputs=["security_report"]
            )
            self.tool_graph.add_tool(
                "analyze_metrics", inputs=["file_contents"], outputs=["metrics_report"]
            )
        except Exception as exc:
            logger.debug(f"Failed to register tool dependencies: {exc}")

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

            # Core tools that should generally remain enabled
            core_tools = {"read_file", "write_file", "list_directory", "execute_bash"}

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

                # Check if core tools are included
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

    def _get_adaptive_threshold(self, user_message: str) -> tuple[float, int]:
        """Calculate adaptive similarity threshold and max_tools based on context.

        Adapts based on:
        1. Model size (from config or detected from model name)
        2. Query specificity (vague queries need stricter thresholds)
        3. Conversation depth (deeper conversations can be more permissive)

        Args:
            user_message: The user's input message

        Returns:
            Tuple of (similarity_threshold, max_tools)
        """
        # Factor 1: Model size - Check configuration first, then fall back to detection
        if self.tool_selection and "base_threshold" in self.tool_selection:
            # Use configured values
            base_threshold = self.tool_selection.get("base_threshold", 0.18)
            base_max_tools = self.tool_selection.get("base_max_tools", 10)
            logger.debug(
                f"Using configured tool selection: threshold={base_threshold:.2f}, "
                f"max_tools={base_max_tools}"
            )
        else:
            # Fall back to model name pattern detection (backwards compatibility)
            model_lower = self.model.lower()

            # Detect model size from common naming patterns
            if any(size in model_lower for size in [":0.5b", ":1b", ":1.5b", ":3b"]):
                # Tiny models (0.5B-3B): Very strict
                base_threshold = 0.35
                base_max_tools = 5
            elif any(size in model_lower for size in [":7b", ":8b"]):
                # Small models (7B-8B): Strict
                base_threshold = 0.25
                base_max_tools = 7
            elif any(size in model_lower for size in [":13b", ":14b", ":15b"]):
                # Medium models (13B-15B): Moderate
                base_threshold = 0.20
                base_max_tools = 10
            elif any(size in model_lower for size in [":30b", ":32b", ":34b", ":70b", ":72b"]):
                # Large models (30B+): Permissive
                base_threshold = 0.15
                base_max_tools = 12
            else:
                # Unknown size or cloud models (Claude, GPT): Moderate-permissive
                base_threshold = 0.18
                base_max_tools = 10

            logger.debug(
                f"Detected tool selection from model name '{model_lower}': "
                f"threshold={base_threshold:.2f}, max_tools={base_max_tools}"
            )

        # Factor 2: Query specificity
        word_count = len(user_message.split())

        if word_count < 5:
            # Very vague query ("help me", "fix this") → stricter
            base_threshold += 0.10
            base_max_tools = max(5, base_max_tools - 2)
        elif word_count < 10:
            # Somewhat vague query → slightly stricter
            base_threshold += 0.05
        elif word_count > 20:
            # Detailed query → looser (more context = better matching)
            base_threshold -= 0.05
            base_max_tools = min(15, base_max_tools + 2)

        # Factor 3: Conversation depth
        conversation_depth = len(self.messages)

        if conversation_depth > 15:
            # Deep conversation → looser (lots of context available)
            base_threshold -= 0.05
            base_max_tools = min(15, base_max_tools + 1)
        elif conversation_depth > 8:
            # Moderate conversation → slightly looser
            base_threshold -= 0.03

        # Clamp values to reasonable ranges
        threshold = max(0.10, min(0.40, base_threshold))
        max_tools = max(5, min(15, base_max_tools))

        logger.debug(
            f"Adaptive threshold: {threshold:.2f}, max_tools: {max_tools} "
            f"(model: {self.model}, words: {word_count}, depth: {conversation_depth})"
        )

        return threshold, max_tools

    async def _select_relevant_tools_semantic(self, user_message: str) -> List[ToolDefinition]:
        """Select tools using embedding-based semantic similarity.

        Args:
            user_message: The user's input message

        Returns:
            List of relevant ToolDefinition objects based on semantic similarity
        """
        if not self.semantic_selector:
            # Fallback to keyword-based if semantic not initialized
            return self._select_relevant_tools_keywords(user_message)

        # Initialize embeddings on first call
        if not self._embeddings_initialized:
            logger.info("Initializing tool embeddings (one-time operation)...")
            await self.semantic_selector.initialize_tool_embeddings(self.tools)
            self._embeddings_initialized = True

        # Get adaptive threshold and max_tools based on model size, query, and context
        threshold, max_tools = self._get_adaptive_threshold(user_message)

        # Select tools with context awareness (Phase 2 enhancement)
        # Pass conversation history for better tool selection across multi-turn tasks
        # Convert Message objects to dicts for semantic selector
        conversation_dicts = [msg.model_dump() for msg in self.messages] if self.messages else None
        tools = await self.semantic_selector.select_relevant_tools_with_context(
            user_message=user_message,
            tools=self.tools,
            conversation_history=conversation_dicts,  # Phase 2: Add context
            max_tools=max_tools,
            similarity_threshold=threshold,
        )

        # Blend with keyword-selected tools to avoid missing obvious categories (e.g., web search)
        keyword_tools = self._select_relevant_tools_keywords(user_message)
        if keyword_tools:
            existing = {t.name for t in tools}
            tools.extend([t for t in keyword_tools if t.name not in existing])

        # If the user explicitly mentions searching the web, ensure web tools are present
        message_lower = user_message.lower()
        if any(
            kw in message_lower for kw in ["search", "web", "online", "lookup", "http", "https"]
        ):
            must_have = {"web_search", "web_summarize", "web_fetch"}
            existing = {t.name for t in tools}
            for tool in self.tools.list_tools():
                if tool.name in must_have and tool.name not in existing:
                    tools.append(
                        ToolDefinition(
                            name=tool.name, description=tool.description, parameters=tool.parameters
                        )
                    )

        # Deduplicate in case of overlaps
        dedup = {}
        for t in tools:
            dedup[t.name] = t
        tools = list(dedup.values())

        logger.info(
            f"Semantic+keyword tools selected ({len(tools)}): {', '.join(t.name for t in tools)}"
        )

        # Smart Fallback: If 0 tools selected, use core tools + keyword fallback
        # This prevents overwhelming small models while still providing useful tools
        if not tools:
            logger.warning(
                "Semantic selection returned 0 tools. "
                "Using smart fallback: core tools + keyword matching."
            )
            # Core tools that are almost always useful
            core_tool_names = {
                "read_file",
                "write_file",
                "list_directory",
                "execute_bash",
                "edit_files",
            }

            # Get keyword-based tools as fallback
            keyword_tools = self._select_relevant_tools_keywords(user_message)

            # Combine core tools with keyword-selected tools
            all_tools_map = {tool.name: tool for tool in self.tools.list_tools()}
            tools = []

            # Add core tools first
            for tool_name in core_tool_names:
                if tool_name in all_tools_map:
                    tool = all_tools_map[tool_name]
                    tools.append(
                        ToolDefinition(
                            name=tool.name, description=tool.description, parameters=tool.parameters
                        )
                    )

            # Add keyword-selected tools (avoiding duplicates)
            existing_names = {t.name for t in tools}
            for keyword_tool in keyword_tools:
                if keyword_tool.name not in existing_names:
                    tools.append(keyword_tool)

            logger.info(
                f"Smart fallback selected {len(tools)} tools: "
                f"{', '.join(t.name for t in tools)}"
            )

            # Record fallback selection analytics
            self._record_tool_selection("fallback", len(tools))
        else:
            # Record semantic selection analytics
            self._record_tool_selection("semantic", len(tools))

        return tools

    def _select_relevant_tools_keywords(self, user_message: str) -> List[ToolDefinition]:
        """Intelligently select relevant tools based on the user's message.

        For small models (<7B), limit to essential tools to prevent overwhelming.
        For larger models, send all relevant tools.

        Args:
            user_message: The user's input message

        Returns:
            List of relevant ToolDefinition objects
        """
        all_tools = list(self.tools.list_tools())

        # Core tools that are almost always useful (filesystem, bash, editor)
        core_tool_names = {
            "read_file",
            "write_file",
            "list_directory",
            "execute_bash",
            "edit_files",
        }

        # Categorize tools by use case
        tool_categories = {
            "git": ["git", "git_suggest_commit", "git_create_pr"],
            "testing": ["testing_generate", "testing_run", "testing_coverage"],
            "refactor": [
                "refactor_extract_function",
                "refactor_inline_variable",
                "refactor_organize_imports",
            ],
            "security": ["security_scan"],
            "docs": ["generate_docs", "analyze_docs"],
            "review": ["code_review"],
            "web": ["web_search", "web_fetch", "web_summarize"],
            "docker": ["docker"],
            "metrics": ["analyze_metrics"],
            "batch": ["batch"],
            "cicd": ["cicd"],
            "scaffold": ["scaffold"],
            "plan": ["plan_files"],
            "search": ["code_search"],
        }

        # Keyword matching for tool selection
        message_lower = user_message.lower()
        selected_categories = set()
        planned_tools: List[ToolDefinition] = []
        goals = self._goal_hints_for_message(user_message)
        if goals:
            planned_tools = self._plan_tools(goals, available_inputs=["query"])

        # Match keywords to categories
        if any(kw in message_lower for kw in ["git", "commit", "branch", "merge", "repository"]):
            selected_categories.add("git")
        if any(kw in message_lower for kw in ["test", "pytest", "unittest", "coverage"]):
            selected_categories.add("testing")
        if any(kw in message_lower for kw in ["refactor", "rename", "extract", "reorganize"]):
            selected_categories.add("refactor")
        if any(kw in message_lower for kw in ["security", "vulnerability", "secret", "scan"]):
            selected_categories.add("security")
        if any(kw in message_lower for kw in ["document", "docstring", "readme", "api doc"]):
            selected_categories.add("docs")
        if any(
            kw in message_lower for kw in ["review", "analyze code", "check code", "code quality"]
        ):
            selected_categories.add("review")
        if any(
            kw in message_lower
            for kw in [
                "search web",
                "search the web",
                "look up",
                "find online",
                "search for",
                "web search",
                "online search",
            ]
        ):
            selected_categories.add("web")
        if any(kw in message_lower for kw in ["docker", "container", "image"]):
            selected_categories.add("docker")
        if any(
            kw in message_lower
            for kw in ["complexity", "metrics", "maintainability", "technical debt"]
        ):
            selected_categories.add("metrics")
        if any(
            kw in message_lower
            for kw in ["batch", "bulk", "multiple files", "search files", "replace across"]
        ):
            selected_categories.add("batch")
        if any(
            kw in message_lower
            for kw in [
                "ci/cd",
                "cicd",
                "pipeline",
                "github actions",
                "gitlab ci",
                "circleci",
                "workflow",
            ]
        ):
            selected_categories.add("cicd")
        if any(
            kw in message_lower
            for kw in ["scaffold", "template", "boilerplate", "new project", "create project"]
        ):
            selected_categories.add("scaffold")
        if any(
            kw in message_lower for kw in ["plan", "which files", "pick files", "where to start"]
        ):
            selected_categories.add("plan")
        if any(
            kw in message_lower
            for kw in ["search code", "code search", "find file", "locate code", "where is"]
        ):
            selected_categories.add("search")

        # Build selected tool names
        selected_tool_names = core_tool_names.copy()
        for category in selected_categories:
            selected_tool_names.update(tool_categories.get(category, []))

        # For small models, limit total tools
        is_small_model = False
        if self.provider.name == "ollama":
            model_lower = self.model.lower()
            small_model_indicators = [":0.5b", ":1.5b", ":3b"]
            is_small_model = any(indicator in model_lower for indicator in small_model_indicators)

        # Filter tools
        selected_tools = planned_tools.copy()
        existing_names = {t.name for t in selected_tools}
        for tool in all_tools:
            if tool.name in selected_tool_names and tool.name not in existing_names:
                selected_tools.append(
                    ToolDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters,
                    )
                )
                existing_names.add(tool.name)

        # For small models, limit to max 10 tools (core + most relevant)
        if is_small_model and len(selected_tools) > 10:
            # Prioritize core tools, then others
            core_tools = [t for t in selected_tools if t.name in core_tool_names]
            other_tools = [t for t in selected_tools if t.name not in core_tool_names]
            selected_tools = core_tools + other_tools[: max(0, 10 - len(core_tools))]

        tool_names = [t.name for t in selected_tools]
        logger.info(
            f"Selected {len(selected_tools)} tools for prompt (small_model={is_small_model}): {', '.join(tool_names)}"
        )

        return selected_tools

    def _prioritize_tools_stage(
        self, user_message: str, tools: Optional[List[ToolDefinition]], stage: str
    ) -> Optional[List[ToolDefinition]]:
        """Stage-aware pruning of tool list to keep it focused per step."""
        if not tools:
            return tools

        # Stage definitions
        planning_tools = {"plan_files", "code_search", "list_directory"}
        reading_tools = {"read_file", "analyze_docs", "code_review"}
        web_tools = {"web_search", "web_summarize", "web_fetch"}
        core = {"write_file", "edit_files"}

        message_lower = user_message.lower()
        needs_web = any(kw in message_lower for kw in ["http", "https", "web", "online", "search"])

        if stage == "initial":
            keep = planning_tools | (web_tools if needs_web else set()) | core
        elif stage == "post_plan":
            keep = reading_tools | planning_tools | core | (web_tools if needs_web else set())
        else:  # post_read
            keep = reading_tools | core | (web_tools if needs_web else set())

        pruned = [t for t in tools if t.name in keep]
        if pruned:
            return pruned

        # If pruning eliminated everything, fall back to a minimal safe set to avoid broadcasting all tools
        core_fallback = {"read_file", "write_file", "list_directory", "execute_bash", "edit_files"}
        fallback_tools = [t for t in tools if t.name in core_fallback]

        if fallback_tools:
            return fallback_tools

        # As a last resort, return a small prefix to prevent broadcasting the entire registry
        fallback_limit = getattr(self.settings, "fallback_max_tools", 8)
        return tools[:fallback_limit]

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        self.messages.append(Message(role=role, content=content))
        if role == "user":
            self.usage_logger.log_event("user_prompt", {"content": content})
        elif role == "assistant":
            self.usage_logger.log_event("assistant_response", {"content": content})

    def _ensure_system_message(self) -> None:
        """Ensure the system prompt is included once at the start of the conversation."""
        if not self._system_added:
            self.messages.append(Message(role="system", content=self._system_prompt))
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
            if self.use_semantic_selection:
                tools = await self._select_relevant_tools_semantic(user_message)
            else:
                tools = self._select_relevant_tools_keywords(user_message)
            tools = self._prioritize_tools_stage(user_message, tools, stage="initial")

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
        """
        # Track performance metrics
        start_time = time.time()
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
        first_pass = True
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

        # Use configurable limits from settings
        # Analysis tasks get much higher limits (or unlimited) since they're expected to explore extensively
        if is_analysis_task:
            max_low_output_iterations = getattr(
                self.settings, "max_exploration_iterations_analysis", 50
            )
            logger.info(
                f"Detected analysis task - allowing up to {max_low_output_iterations} exploration iterations"
            )
        elif is_action_task:
            max_low_output_iterations = getattr(
                self.settings, "max_exploration_iterations_action", 12
            )
        else:
            max_low_output_iterations = getattr(self.settings, "max_exploration_iterations", 8)

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
            # Select tools for this pass
            tools = None
            provider_supports_tools = self.provider.supports_tools()
            tooling_allowed = provider_supports_tools and self._model_supports_tool_calls()

            if tooling_allowed:
                if self.use_semantic_selection:
                    tools = await self._select_relevant_tools_semantic(context_msg)
                else:
                    tools = self._select_relevant_tools_keywords(context_msg)
                # If we have inferred goals and no files read yet, prepend planned chain
                if goals:
                    available_inputs = ["query"]
                    # If files already observed, skip search and use file contents as available
                    if self.observed_files:
                        available_inputs.append("file_contents")
                    planned = self._plan_tools(goals, available_inputs=available_inputs)
                    if planned:
                        # Deduplicate while preserving order: planned first, then rest
                        existing = {t.name for t in planned}
                        remainder = [t for t in tools if t.name not in existing] if tools else []
                        tools = planned + remainder
                stage = (
                    "initial"
                    if first_pass
                    else ("post_read" if self.observed_files else "post_plan")
                )
                tools = self._prioritize_tools_stage(context_msg, tools, stage=stage)

            # Prepare optional thinking parameter for providers that support it (Anthropic)
            provider_kwargs = {}
            if self.thinking:
                # Anthropic extended thinking format
                provider_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}

            full_content = ""
            tool_calls = None
            async for chunk in self.provider.stream(
                messages=self.messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=tools,
                **provider_kwargs,
            ):
                full_content += chunk.content
                # Estimate tokens (rough: ~4 chars per token)
                if chunk.content:
                    total_tokens += len(chunk.content) / 4
                if chunk.tool_calls:
                    logger.debug(f"Received tool_calls in chunk: {chunk.tool_calls}")
                    tool_calls = chunk.tool_calls
                yield chunk

            # Fallback: parse JSON tool calls from accumulated content
            if (
                not tool_calls
                and full_content
                and hasattr(self.provider, "_parse_json_tool_call_from_content")
            ):
                parsed_tool_calls = self.provider._parse_json_tool_call_from_content(full_content)
                if parsed_tool_calls:
                    logger.debug("Parsed tool call from accumulated streaming content (fallback)")
                    tool_calls = parsed_tool_calls
                    full_content = ""

            # Provider-agnostic JSON tool-call fallback
            if not tool_calls and full_content:
                parsed_tool_calls = self._parse_json_tool_call_from_content(full_content)
                if parsed_tool_calls:
                    logger.debug(
                        "Parsed tool call from accumulated streaming content (generic fallback)"
                    )
                    tool_calls = parsed_tool_calls
                    full_content = ""
                else:
                    # Try XML-ish wrapper containing JSON
                    parsed_tool_calls = self._parse_xmlish_tool_call_from_content(full_content)
                    if parsed_tool_calls:
                        logger.debug(
                            "Parsed tool call from XML-ish streaming content (generic fallback)"
                        )
                        tool_calls = parsed_tool_calls
                        full_content = ""

            # Ensure tool_calls is a list of dicts to avoid type errors from malformed provider output
            if tool_calls:
                normalized_tool_calls = [tc for tc in tool_calls if isinstance(tc, dict)]
                if len(normalized_tool_calls) != len(tool_calls):
                    logger.warning(f"Dropped non-dict tool_calls: {tool_calls}")
                tool_calls = normalized_tool_calls or None

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
                plain_text = self._strip_markup(full_content)
                self.add_message("assistant", plain_text or full_content)
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
            is_true_loop = False
            if len(recent_tool_calls) >= 3:
                last_call = recent_tool_calls[-1]
                repeat_count = sum(1 for c in recent_tool_calls[-5:] if c == last_call)
                if repeat_count >= 3:
                    is_true_loop = True
                    logger.warning(
                        f"True loop detected: same tool call repeated {repeat_count} times"
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
                yield StreamChunk(
                    content="No tools executed; cannot provide evidence-based answer.\n"
                )
                # Display performance metrics
                elapsed_time = time.time() - start_time
                tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
                yield StreamChunk(
                    content=f"\n📊 {total_tokens:.0f} tokens | {elapsed_time:.1f}s | {tokens_per_second:.1f} tok/s\n",
                    is_final=True,
                )
                break

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
                first_pass = False
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

            first_pass = False
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

            # Execute tool with retry logic (using normalized arguments)
            result, success, error_msg = await self._execute_tool_with_retry(
                tool_name, normalized_args, context
            )

            # Update counters and tracking
            self.tool_calls_used += 1
            self.executed_tools.append(tool_name)
            if tool_name == "read_file" and "path" in normalized_args:
                self.observed_files.append(str(normalized_args.get("path")))

            # Calculate execution time
            elapsed_ms = (time.monotonic() - start) * 1000  # Convert to milliseconds

            # Record tool execution analytics
            self._record_tool_execution(tool_name, success, elapsed_ms)

            output = result.output if success and result else None
            error_display = error_msg or (result.error if result else "Unknown error")

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
                error_display = error_msg or (result.error if result else "Unknown error")
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
        self.messages.clear()

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
