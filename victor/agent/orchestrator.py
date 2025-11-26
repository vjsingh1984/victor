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
from victor.tools.web_search_tool import web_search, web_fetch, web_summarize, set_web_search_provider
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
            # Initialize embeddings asynchronously (will be done in first call)
            self._embeddings_initialized = False

    def shutdown(self):
        """Gracefully shut down stateful managers."""
        self.console.print("[dim]Shutting down code execution environment...[/dim]")
        self.code_manager.stop()

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
            self.tools.register(web_search)
            self.tools.register(web_fetch)
            self.tools.register(web_summarize)


    def _should_use_tools(self) -> bool:
        """Always return True - tool selection is handled by _select_relevant_tools()."""
        return True

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

        # Select tools semantically (using defaults: max_tools=5, threshold=0.15)
        tools = await self.semantic_selector.select_relevant_tools(
            user_message=user_message,
            tools=self.tools,
        )

        # Fallback: If 0 tools selected, provide ALL tools to the model
        # Since we reduced to 31 tools specifically for this fallback,
        # the model can handle choosing from the full set
        if not tools:
            logger.info(
                "Semantic selection returned 0 tools. "
                f"Falling back to ALL {len(self.tools.list_tools())} tools."
            )
            tools = [
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters
                )
                for tool in self.tools.list_tools()
            ]

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
        if any(kw in message_lower for kw in ["search web", "look up", "find online", "search for"]):
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

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        self.messages.append(Message(role=role, content=content))

    async def chat(self, user_message: str) -> CompletionResponse:
        """Send a chat message and get response.

        Args:
            user_message: User's message

        Returns:
            CompletionResponse from the model
        """
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
        # Add user message to history
        self.add_message("user", user_message)

        # Get tool definitions
        # Intelligently select relevant tools based on the user's message
        tools = None
        if self.provider.supports_tools():
            if self.use_semantic_selection:
                tools = await self._select_relevant_tools_semantic(user_message)
            else:
                tools = self._select_relevant_tools_keywords(user_message)

        # Stream response
        full_content = ""
        async for chunk in self.provider.stream(
            messages=self.messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=tools,
        ):
            full_content += chunk.content
            yield chunk

        # Add to history
        if full_content:
            self.add_message("assistant", full_content)

    async def _handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """Handle tool calls from the model.

        Args:
            tool_calls: List of tool call requests
        """
        if not tool_calls:
            return

        for tool_call in tool_calls:
            # Validate tool call structure
            if not isinstance(tool_call, dict):
                self.console.print(f"[yellow]⚠ Skipping invalid tool call (not a dict): {tool_call}[/]")
                continue

            tool_name = tool_call.get("name")
            if not tool_name:
                self.console.print(f"[yellow]⚠ Skipping tool call without name: {tool_call}[/]")
                continue

            tool_args = tool_call.get("arguments", {})

            self.console.print(f"\n[bold cyan]Executing tool:[/] {tool_name}")

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

                if result.success:
                    self.console.print(f"[green]✓ Tool executed successfully[/]")
                    # Add tool result to conversation
                    self.add_message(
                        "user",
                        f"Tool '{tool_name}' result: {result.output}",
                    )
                else:
                    self.console.print(f"[red]✗ Tool execution failed: {result.error}[/]")
            except Exception as e:
                self.console.print(f"[red]✗ Tool execution error: {e}[/]")

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
