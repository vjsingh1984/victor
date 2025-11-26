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
from victor.tools.file_editor_tool import FileEditorTool
from victor.tools.git_tool import GitTool
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
        self.tools.register(FileEditorTool())
        self.tools.register(GitTool(provider=self.provider, model=self.model))

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
