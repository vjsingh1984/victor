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

import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from rich.console import Console
from rich.markdown import Markdown

from victor.config.settings import Settings

logger = logging.getLogger(__name__)
from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.registry import ProviderRegistry
from victor.tools.base import ToolRegistry
from victor.tools.bash import execute_bash
from victor.tools.code_executor_tool import (
    CodeExecutionManager,
    execute_python_in_sandbox,
    upload_files_to_sandbox,
)
from victor.tools.filesystem import list_directory, read_file, write_file
from victor.tools.file_editor_tool import edit_files
from victor.tools.git_tool import (
    git,
    git_suggest_commit,
    git_create_pr,
    git_analyze_conflicts,
    set_git_provider,
)
from victor.tools.batch_processor_tool import batch, set_batch_processor_config
from victor.tools.cicd_tool import cicd
from victor.tools.scaffold_tool import scaffold
from victor.tools.docker_tool import docker
from victor.tools.metrics_tool import analyze_metrics
from victor.tools.security_scanner_tool import security_scan
from victor.tools.documentation_tool import generate_docs, analyze_docs
from victor.tools.code_review_tool import code_review, set_code_review_config
from victor.tools.refactor_tool import (
    refactor_extract_function,
    refactor_inline_variable,
    refactor_organize_imports,
)
from victor.tools.testing_tool import run_tests
from victor.tools.web_search_tool import web_search, web_fetch, web_summarize, set_web_search_provider, set_web_tool_defaults
from victor.tools.plan_tool import plan_files
from victor.tools.code_search_tool import code_search
from victor.tools.mcp_bridge_tool import mcp_call, configure_mcp_client, get_mcp_tool_definitions
from victor.tools.workflow_tool import run_workflow
from victor.tools.code_intelligence_tool import find_symbol, find_references, rename_symbol
from victor.tools.semantic_selector import SemanticToolSelector
from victor.workflows.base import WorkflowRegistry
from victor.workflows.new_feature_workflow import NewFeatureWorkflow


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
    ):
        """Initialize orchestrator.

        Args:
            settings: The application settings.
            provider: LLM provider instance
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            console: Rich console for output
        """
        self.settings = settings
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.console = console or Console()
        self._system_prompt = (
            "You are a code analyst for this repository. Follow this loop:\n"
            "1) Plan briefly, then call list_directory/read_file to inspect real files (not imagined). "
            "2) If the user names multiple files, read ALL of them before proposing or concluding. "
            "3) Keep issuing tool calls iteratively until the task is complete or the tool budget is exhausted—do not stop after the first tool call. "
            "4) If the user asks to modify or apply changes, propose a concise plan and then execute write_file/edit_files to apply the edits and show diffs. "
            "4) Do NOT invent tool outputs—only cite results from tools you actually called. "
            "5) When asked for web info, use web_search/web_summarize; when asked about repo code, use read_file/analyze_docs/code_review.\n"
            "Always report findings with file paths. Avoid generic advice. Stop once you hit the tool budget."
        )
        self._system_added = False
        self.tool_budget = getattr(settings, "tool_call_budget", 20)
        self.tool_calls_used = 0
        self.observed_files: List[str] = []
        self.executed_tools: List[str] = []

        # Tool usage analytics
        self._tool_usage_stats: Dict[str, Dict[str, Any]] = {}
        self._tool_selection_stats: Dict[str, int] = {
            "semantic_selections": 0,
            "keyword_selections": 0,
            "fallback_selections": 0,
            "total_tools_selected": 0,
            "total_tools_executed": 0,
        }

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
        self.tools.register_before_hook(self._log_tool_call)

        # Semantic tool selector (optional, configured via settings)
        self.use_semantic_selection = getattr(settings, 'use_semantic_tool_selection', False)
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
            self._embedding_preload_task = None

    def shutdown(self):
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
                "min_time_ms": float('inf'),
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
                reverse=True
            )[:10],
            "top_tools_by_time": sorted(
                [(name, stats["total_time_ms"]) for name, stats in self._tool_usage_stats.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10],
        }

    def _log_tool_call(self, name: str, kwargs: dict) -> None:
        """A hook that logs information before a tool is called."""
        self.console.print(f"[dim]Attempting to call tool '{name}' with arguments: {kwargs}[/dim]")

    def _register_default_workflows(self) -> None:
        """Register default workflows."""
        self.workflow_registry.register(NewFeatureWorkflow())

    def _register_default_tools(self) -> None:
        """Register default tools."""
        self.tools.register(run_workflow)
        self.tools.register(execute_python_in_sandbox)
        self.tools.register(upload_files_to_sandbox)
        self.tools.register(read_file)
        self.tools.register(write_file)
        self.tools.register(list_directory)
        self.tools.register(execute_bash)
        self.tools.register(run_tests)
        self.tools.register(find_symbol)
        self.tools.register(find_references)
        self.tools.register(rename_symbol)

        # Register file editor tool (consolidated)
        self.tools.register(edit_files)

        # Set git provider and register git tools
        set_git_provider(self.provider, self.model)
        # Register git tool (consolidated)
        self.tools.register(git)
        # Register AI-powered git tools (keep separate)
        self.tools.register(git_suggest_commit)
        self.tools.register(git_create_pr)
        self.tools.register(git_analyze_conflicts)

        # Register batch processor tool (consolidated)
        set_batch_processor_config(max_workers=4)
        self.tools.register(batch)

        # Register CI/CD tool (consolidated)
        self.tools.register(cicd)

        # Register scaffold tool (consolidated)
        self.tools.register(scaffold)

        # Register Docker tool (consolidated)
        self.tools.register(docker)

        # Register metrics tool (consolidated)
        self.tools.register(analyze_metrics)

        # Register security scanner tool (consolidated)
        self.tools.register(security_scan)

        # Register documentation tools (consolidated)
        self.tools.register(generate_docs)
        self.tools.register(analyze_docs)

        # Register code review tools
        set_code_review_config(max_complexity=10)
        # Register code review tool (consolidated)
        self.tools.register(code_review)

        # Register refactor tools
        # Note: rename_symbol is in code_intelligence_tool, not here (avoid duplicate)
        self.tools.register(refactor_extract_function)
        self.tools.register(refactor_inline_variable)
        self.tools.register(refactor_organize_imports)

        # Only register network-dependent tools if not in air-gapped mode
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
        self.tools.register(web_search)
        self.tools.register(web_fetch)
        self.tools.register(web_summarize)
        self.tools.register(plan_files)
        self.tools.register(code_search)
        self.tools.register(mcp_call)

        # Register MCP tools if configured
        if getattr(self.settings, "use_mcp_tools", False):
            if getattr(self.settings, "mcp_command", None):
                try:
                    from victor.mcp.client import MCPClient

                    mcp_client = MCPClient()
                    cmd_parts = self.settings.mcp_command.split()
                    asyncio.create_task(mcp_client.connect(cmd_parts))
                    configure_mcp_client(mcp_client, prefix=getattr(self.settings, "mcp_prefix", "mcp"))
                except Exception as exc:
                    logger.warning(f"Failed to start MCP client: {exc}")

            for mcp_tool in get_mcp_tool_definitions():
                self.tools.register_dict(mcp_tool)


    def _should_use_tools(self) -> bool:
        """Always return True - tool selection is handled by _select_relevant_tools()."""
        return True

    def _get_adaptive_threshold(self, user_message: str) -> tuple[float, int]:
        """Calculate adaptive similarity threshold and max_tools based on context.

        Adapts based on:
        1. Model size (smaller models need stricter filtering)
        2. Query specificity (vague queries need stricter thresholds)
        3. Conversation depth (deeper conversations can be more permissive)

        Args:
            user_message: The user's input message

        Returns:
            Tuple of (similarity_threshold, max_tools)
        """
        # Factor 1: Model size
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
            f"(model_size: {model_lower}, words: {word_count}, depth: {conversation_depth})"
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
        tools = await self.semantic_selector.select_relevant_tools_with_context(
            user_message=user_message,
            tools=self.tools,
            conversation_history=self.conversation_history,  # Phase 2: Add context
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
        if any(kw in message_lower for kw in ["search", "web", "online", "lookup", "http", "https"]):
            must_have = {"web_search", "web_summarize", "web_fetch"}
            existing = {t.name for t in tools}
            for tool in self.tools.list_tools():
                if tool.name in must_have and tool.name not in existing:
                    tools.append(ToolDefinition(name=tool.name, description=tool.description, parameters=tool.parameters))

        # Deduplicate in case of overlaps
        dedup = {}
        for t in tools:
            dedup[t.name] = t
        tools = list(dedup.values())

        logger.info(f"Semantic+keyword tools selected ({len(tools)}): {', '.join(t.name for t in tools)}")

        # Smart Fallback: If 0 tools selected, use core tools + keyword fallback
        # This prevents overwhelming small models while still providing useful tools
        if not tools:
            logger.warning(
                "Semantic selection returned 0 tools. "
                "Using smart fallback: core tools + keyword matching."
            )
            # Core tools that are almost always useful
            core_tool_names = {
                "read_file", "write_file", "list_directory",
                "execute_bash", "edit_files"
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
                    tools.append(ToolDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters
                    ))

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
            "read_file", "write_file", "list_directory",
            "execute_bash",
            "edit_files"
        }

        # Categorize tools by use case
        tool_categories = {
            "git": ["git", "git_suggest_commit", "git_create_pr"],
            "testing": ["testing_generate", "testing_run", "testing_coverage"],
            "refactor": ["refactor_extract_function", "refactor_inline_variable", "refactor_organize_imports"],
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
        if any(kw in message_lower for kw in ["review", "analyze code", "check code", "code quality"]):
            selected_categories.add("review")
        if any(kw in message_lower for kw in ["search web", "search the web", "look up", "find online", "search for", "web search", "online search"]):
            selected_categories.add("web")
        if any(kw in message_lower for kw in ["docker", "container", "image"]):
            selected_categories.add("docker")
        if any(kw in message_lower for kw in ["complexity", "metrics", "maintainability", "technical debt"]):
            selected_categories.add("metrics")
        if any(kw in message_lower for kw in ["batch", "bulk", "multiple files", "search files", "replace across"]):
            selected_categories.add("batch")
        if any(kw in message_lower for kw in ["ci/cd", "cicd", "pipeline", "github actions", "gitlab ci", "circleci", "workflow"]):
            selected_categories.add("cicd")
        if any(kw in message_lower for kw in ["scaffold", "template", "boilerplate", "new project", "create project"]):
            selected_categories.add("scaffold")
        if any(kw in message_lower for kw in ["plan", "which files", "pick files", "where to start"]):
            selected_categories.add("plan")
        if any(kw in message_lower for kw in ["search code", "code search", "find file", "locate code", "where is"]):
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
        selected_tools = []
        for tool in all_tools:
            if tool.name in selected_tool_names:
                selected_tools.append(ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                ))

        # For small models, limit to max 10 tools (core + most relevant)
        if is_small_model and len(selected_tools) > 10:
            # Prioritize core tools, then others
            core_tools = [t for t in selected_tools if t.name in core_tool_names]
            other_tools = [t for t in selected_tools if t.name not in core_tool_names]
            selected_tools = core_tools + other_tools[: max(0, 10 - len(core_tools))]

        tool_names = [t.name for t in selected_tools]
        logger.info(f"Selected {len(selected_tools)} tools for prompt (small_model={is_small_model}): {', '.join(tool_names)}")

        return selected_tools

    def _prioritize_tools_stage(self, user_message: str, tools: Optional[List[ToolDefinition]], stage: str) -> Optional[List[ToolDefinition]]:
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
        # If pruning eliminated everything, fall back to original list
        return pruned or tools

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        self.messages.append(Message(role=role, content=content))

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
        response = await self.provider.chat(
            messages=self.messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=tools,
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
        self._ensure_system_message()
        self.tool_calls_used = 0
        self.observed_files = []
        self.executed_tools = []
        # Add user message to history
        self.add_message("user", user_message)

        # Get tool definitions
        # Iteratively stream → run tools → stream follow-up until no tool calls or budget exhausted
        first_pass = True
        context_msg = user_message
        while True:
            # Select tools for this pass
            tools = None
            if self.provider.supports_tools():
                if self.use_semantic_selection:
                    tools = await self._select_relevant_tools_semantic(context_msg)
                else:
                    tools = self._select_relevant_tools_keywords(context_msg)
                stage = "initial" if first_pass else ("post_read" if self.observed_files else "post_plan")
                tools = self._prioritize_tools_stage(context_msg, tools, stage=stage)

            full_content = ""
            tool_calls = None
            async for chunk in self.provider.stream(
                messages=self.messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=tools,
            ):
                full_content += chunk.content
                if chunk.tool_calls:
                    logger.debug(f"Received tool_calls in chunk: {chunk.tool_calls}")
                    tool_calls = chunk.tool_calls
                yield chunk

            # Fallback: parse JSON tool calls from accumulated content
            if not tool_calls and full_content and hasattr(self.provider, "_parse_json_tool_call_from_content"):
                parsed_tool_calls = self.provider._parse_json_tool_call_from_content(full_content)
                if parsed_tool_calls:
                    logger.debug("Parsed tool call from accumulated streaming content (fallback)")
                    tool_calls = parsed_tool_calls
                    full_content = ""

            if full_content:
                self.add_message("assistant", full_content)

            logger.debug(f"After streaming pass, tool_calls = {tool_calls}")

            if not tool_calls:
                # No more tool calls requested; finish
                yield StreamChunk(content="", is_final=True)
                break

            remaining = max(0, self.tool_budget - self.tool_calls_used)
            if remaining <= 0:
                yield StreamChunk(content=f"[tool] ⚠ Tool budget reached ({self.tool_budget}); skipping tool calls.\n")
                yield StreamChunk(content="No tools executed; cannot provide evidence-based answer.\n", is_final=True)
                break

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
                evidence_note = "Evidence files read: " + ", ".join(sorted(set(self.observed_files)))
                self.add_message("system", evidence_note + " Only cite these or tool outputs; do not invent paths.")
            else:
                self.add_message("system", "No files read yet. Avoid file-specific claims; suggest using tools if needed.")

            first_pass = False
            context_msg = full_content or user_message

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
                self.console.print(f"[yellow]⚠ Skipping invalid tool call (not a dict): {tool_call}[/]")
                continue

            tool_name = tool_call.get("name")
            if not tool_name:
                self.console.print(f"[yellow]⚠ Skipping tool call without name: {tool_call}[/]")
                continue

            if self.tool_calls_used >= self.tool_budget:
                self.console.print(f"[yellow]⚠ Tool budget reached ({self.tool_budget}); skipping remaining tool calls.[/]")
                break

            tool_args = tool_call.get("arguments", {})

            self.console.print(f"\n[bold cyan]Executing tool:[/] {tool_name}")

            start = time.monotonic()
            try:
                # Create context for the tool
                context = {
                    "code_manager": self.code_manager,
                    "provider": self.provider,
                    "model": self.model,
                    "tool_registry": self.tools,
                    "workflow_registry": self.workflow_registry,
                }
                
                # Execute tool
                result = await self.tools.execute(tool_name, context=context, **tool_args)
                self.tool_calls_used += 1
                self.executed_tools.append(tool_name)
                if tool_name == "read_file" and "path" in tool_args:
                    self.observed_files.append(str(tool_args.get("path")))

                # Calculate execution time
                elapsed_ms = (time.monotonic() - start) * 1000  # Convert to milliseconds

                # Record tool execution analytics
                self._record_tool_execution(tool_name, result.success, elapsed_ms)

                if result.success:
                    self.console.print(f"[green]✓ Tool executed successfully[/] [dim]({elapsed_ms:.0f}ms)[/dim]")
                    # Add tool result to conversation
                    self.add_message(
                        "user",
                        f"Tool '{tool_name}' result: {result.output}",
                    )
                    results.append(
                        {
                            "name": tool_name,
                            "success": True,
                            "elapsed": time.monotonic() - start,
                        }
                    )
                else:
                    self.console.print(f"[red]✗ Tool execution failed: {result.error}[/] [dim]({elapsed_ms:.0f}ms)[/dim]")
                    results.append(
                        {
                            "name": tool_name,
                            "success": False,
                            "elapsed": time.monotonic() - start,
                        }
                    )
            except Exception as e:
                elapsed_ms = (time.monotonic() - start) * 1000
                self.console.print(f"[red]✗ Tool execution error: {e}[/] [dim]({elapsed_ms:.0f}ms)[/dim]")

                # Record failed execution analytics
                self._record_tool_execution(tool_name or "unknown", False, elapsed_ms)

                results.append(
                    {
                        "name": tool_name or "unknown",
                        "success": False,
                        "elapsed": time.monotonic() - start,
                        "error": str(e),
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
    ) -> "AgentOrchestrator":
        """Create orchestrator from settings.

        Args:
            settings: Application settings
            profile_name: Profile to use

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
        )
