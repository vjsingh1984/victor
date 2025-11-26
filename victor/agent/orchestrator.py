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

from typing import Any, AsyncIterator, Dict, List, Optional

from rich.console import Console
from rich.markdown import Markdown

from victor.config.settings import Settings
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
from victor.tools.file_editor_tool import (
    file_editor_start_transaction,
    file_editor_add_create,
    file_editor_add_modify,
    file_editor_add_delete,
    file_editor_add_rename,
    file_editor_preview,
    file_editor_commit,
    file_editor_rollback,
    file_editor_abort,
    file_editor_status,
)
from victor.tools.git_tool import (
    git_status,
    git_diff,
    git_stage,
    git_commit,
    git_log,
    git_branch,
    git_suggest_commit,
    git_create_pr,
    git_analyze_conflicts,
    set_git_provider,
)
from victor.tools.batch_processor_tool import (
    batch_search,
    batch_replace,
    batch_analyze,
    batch_list_files,
    batch_transform,
    set_batch_processor_config,
)
from victor.tools.cicd_tool import (
    cicd_generate,
    cicd_validate,
    cicd_list_templates,
    cicd_create_workflow,
)
from victor.tools.scaffold_tool import (
    scaffold_create,
    scaffold_list_templates,
    scaffold_add_file,
    scaffold_init_git,
)
from victor.tools.docker_tool import (
    docker_ps,
    docker_images,
    docker_pull,
    docker_run,
    docker_stop,
    docker_start,
    docker_restart,
    docker_rm,
    docker_rmi,
    docker_logs,
    docker_stats,
    docker_inspect,
    docker_networks,
    docker_volumes,
    docker_exec,
)
from victor.tools.metrics_tool import (
    metrics_complexity,
    metrics_maintainability,
    metrics_debt,
    metrics_profile,
    metrics_analyze,
    metrics_report,
)
from victor.tools.security_scanner_tool import (
    security_scan_secrets,
    security_scan_dependencies,
    security_scan_config,
    security_scan_all,
    security_check_file,
)
from victor.tools.documentation_tool import (
    docs_generate_docstrings,
    docs_generate_api,
    docs_generate_readme,
    docs_add_type_hints,
    docs_analyze_coverage,
)
from victor.tools.code_review_tool import (
    code_review_file,
    code_review_directory,
    code_review_security,
    code_review_complexity,
    code_review_best_practices,
    set_code_review_config,
)
from victor.tools.refactor_tool import (
    refactor_rename_symbol,
    refactor_extract_function,
    refactor_inline_variable,
    refactor_organize_imports,
)
from victor.tools.testing_tool import run_tests
from victor.tools.web_search_tool import web_search, web_fetch, web_summarize, set_web_search_provider
from victor.tools.workflow_tool import run_workflow
from victor.tools.code_intelligence_tool import find_symbol, find_references, rename_symbol
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
        self.tools.register(file_editor_start_transaction)
        self.tools.register(file_editor_add_create)
        self.tools.register(file_editor_add_modify)
        self.tools.register(file_editor_add_delete)
        self.tools.register(file_editor_add_rename)
        self.tools.register(file_editor_preview)
        self.tools.register(file_editor_commit)
        self.tools.register(file_editor_rollback)
        self.tools.register(file_editor_abort)
        self.tools.register(file_editor_status)

        # Set git provider and register git tools
        set_git_provider(self.provider, self.model)
        self.tools.register(git_status)
        self.tools.register(git_diff)
        self.tools.register(git_stage)
        self.tools.register(git_commit)
        self.tools.register(git_log)
        self.tools.register(git_branch)
        self.tools.register(git_suggest_commit)
        self.tools.register(git_create_pr)
        self.tools.register(git_analyze_conflicts)

        # Register batch processor tools
        set_batch_processor_config(max_workers=4)
        self.tools.register(batch_search)
        self.tools.register(batch_replace)
        self.tools.register(batch_analyze)
        self.tools.register(batch_list_files)
        self.tools.register(batch_transform)

        # Register CI/CD tools
        self.tools.register(cicd_generate)
        self.tools.register(cicd_validate)
        self.tools.register(cicd_list_templates)
        self.tools.register(cicd_create_workflow)

        # Register scaffold tools
        self.tools.register(scaffold_create)
        self.tools.register(scaffold_list_templates)
        self.tools.register(scaffold_add_file)
        self.tools.register(scaffold_init_git)

        # Register Docker tools
        self.tools.register(docker_ps)
        self.tools.register(docker_images)
        self.tools.register(docker_pull)
        self.tools.register(docker_run)
        self.tools.register(docker_stop)
        self.tools.register(docker_start)
        self.tools.register(docker_restart)
        self.tools.register(docker_rm)
        self.tools.register(docker_rmi)
        self.tools.register(docker_logs)
        self.tools.register(docker_stats)
        self.tools.register(docker_inspect)
        self.tools.register(docker_networks)
        self.tools.register(docker_volumes)
        self.tools.register(docker_exec)

        # Register metrics tools
        self.tools.register(metrics_complexity)
        self.tools.register(metrics_maintainability)
        self.tools.register(metrics_debt)
        self.tools.register(metrics_profile)
        self.tools.register(metrics_analyze)
        self.tools.register(metrics_report)

        # Register security scanner tools
        self.tools.register(security_scan_secrets)
        self.tools.register(security_scan_dependencies)
        self.tools.register(security_scan_config)
        self.tools.register(security_scan_all)
        self.tools.register(security_check_file)

        # Register documentation tools
        self.tools.register(docs_generate_docstrings)
        self.tools.register(docs_generate_api)
        self.tools.register(docs_generate_readme)
        self.tools.register(docs_add_type_hints)
        self.tools.register(docs_analyze_coverage)

        # Register code review tools
        set_code_review_config(max_complexity=10)
        self.tools.register(code_review_file)
        self.tools.register(code_review_directory)
        self.tools.register(code_review_security)
        self.tools.register(code_review_complexity)
        self.tools.register(code_review_best_practices)

        # Register refactor tools
        self.tools.register(refactor_rename_symbol)
        self.tools.register(refactor_extract_function)
        self.tools.register(refactor_inline_variable)
        self.tools.register(refactor_organize_imports)

        # Only register network-dependent tools if not in air-gapped mode
        if not self.settings.airgapped_mode:
            set_web_search_provider(self.provider, self.model)
            self.tools.register(web_search)
            self.tools.register(web_fetch)
            self.tools.register(web_summarize)
        

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
        tools = None
        if self.provider.supports_tools():
            tools = [
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                )
                for tool in self.tools.list_tools()
            ]

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
        tools = None
        if self.provider.supports_tools():
            tools = [
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                )
                for tool in self.tools.list_tools()
            ]

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
