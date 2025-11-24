"""Agent orchestrator for managing conversations and tool execution."""

from typing import Any, AsyncIterator, Dict, List, Optional

from rich.console import Console
from rich.markdown import Markdown

from codingagent.config.settings import Settings
from codingagent.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    StreamChunk,
    ToolDefinition,
)
from codingagent.providers.ollama import OllamaProvider
from codingagent.tools.base import ToolRegistry
from codingagent.tools.bash import BashTool
from codingagent.tools.filesystem import ListDirectoryTool, ReadFileTool, WriteFileTool


class AgentOrchestrator:
    """Orchestrates agent interactions, tool execution, and provider communication."""

    def __init__(
        self,
        provider: BaseProvider,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        console: Optional[Console] = None,
    ):
        """Initialize orchestrator.

        Args:
            provider: LLM provider instance
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            console: Rich console for output
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.console = console or Console()

        # Conversation history
        self.messages: List[Message] = []

        # Tool registry
        self.tools = ToolRegistry()
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register default tools."""
        self.tools.register(ReadFileTool())
        self.tools.register(WriteFileTool())
        self.tools.register(ListDirectoryTool())
        self.tools.register(BashTool(timeout=60))

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
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})

            self.console.print(f"\n[bold cyan]Executing tool:[/] {tool_name}")
            self.console.print(f"[dim]Arguments:[/] {tool_args}")

            # Execute tool
            result = await self.tools.execute(tool_name, **tool_args)

            if result.success:
                self.console.print(f"[green]✓ Tool executed successfully[/]")
                # Add tool result to conversation
                self.add_message(
                    "user",
                    f"Tool '{tool_name}' result: {result.output}",
                )
            else:
                self.console.print(f"[red]✗ Tool execution failed: {result.error}[/]")

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

        # Create provider instance
        # For now, only Ollama is implemented
        if profile.provider == "ollama":
            provider = OllamaProvider(**provider_settings)
        else:
            raise ValueError(f"Provider not yet implemented: {profile.provider}")

        return cls(
            provider=provider,
            model=profile.model,
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
        )
