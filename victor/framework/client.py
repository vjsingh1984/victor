"""VictorClient — unified facade making CLI/TUI/Web first-class framework clients.

This is the SINGLE entry point that the UI layer (CLI, TUI, Web server) uses
to interact with the framework. It wraps Agent.create() and exposes a clean
boundary so the CLI never reaches into internals.

Fixes Issue #1: CLI bypasses Agent API entirely
Fixes Issue #3: ServiceContainer DI bootstrapped into app lifecycle
Fixes Issue #4: Protocol boundary between UI and framework
Fixes Issue #5: Unified typed config replaces scattered settings mutation

Usage in CLI (victor/ui/commands/chat.py):
    from victor.framework.client import VictorClient
    from victor.framework.config_models import VictorConfig

    config = VictorConfig.from_cli_args(provider="anthropic", model="claude-3")
    client = VictorClient(config)
    result = await client.chat("Hello")
    # OR streaming:
    async for event in client.stream("Hello"):
        print(event)

Usage in TUI:
    config = VictorConfig.from_settings(settings)
    client = VictorClient(config)
    session = await client.create_session()

Architecture:
    CLI/TUI/Web  →  VictorClient  →  Agent  →  Orchestrator  →  Providers/Tools
                         ↓
               ServiceContainer (bootstrapped once)
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.framework.config_models import VictorConfig
    from victor.framework.agent import Agent
    from victor.core.container import ServiceContainer

logger = logging.getLogger(__name__)


class _ChatResult:
    """Concrete ChatResultProtocol implementation."""

    def __init__(
        self,
        content: str,
        *,
        success: bool = True,
        tool_calls: int = 0,
        iterations: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._content = content
        self._success = success
        self._tool_calls = tool_calls
        self._iterations = iterations
        self._metadata = metadata or {}

    @property
    def content(self) -> str:
        return self._content

    @property
    def success(self) -> bool:
        return self._success

    @property
    def tool_calls(self) -> int:
        return self._tool_calls

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata


class _StreamEvent:
    """Concrete StreamEventProtocol implementation."""

    def __init__(
        self,
        event_type: str,
        *,
        content: Optional[str] = None,
        tool_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._event_type = event_type
        self._content = content
        self._tool_name = tool_name
        self._metadata = metadata or {}

    @property
    def event_type(self) -> str:
        return self._event_type

    @property
    def content(self) -> Optional[str]:
        return self._content

    @property
    def tool_name(self) -> Optional[str]:
        return self._tool_name

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata


class VictorClient:
    """Unified client facade — the ONLY interface the UI layer uses.

    This class makes the CLI a first-class client of the framework by:
    1. Going through Agent.create() instead of bypassing it
    2. Bootstrapping the DI container once at initialization
    3. Accepting typed VictorConfig instead of raw Settings mutation
    4. Providing chat(), stream(), create_session(), run_workflow()

    Example:
        from victor.framework.client import VictorClient
        from victor.framework.config_models import VictorConfig

        config = VictorConfig.from_cli_args(provider="anthropic")
        async with VictorClient(config) as client:
            result = await client.chat("Write a hello world")
            print(result.content)
    """

    def __init__(
        self,
        config: "VictorConfig",
        *,
        container: Optional["ServiceContainer"] = None,
    ) -> None:
        """Initialize client with typed config and optional DI container.

        Args:
            config: Unified typed configuration (replaces raw Settings mutation)
            container: Optional pre-built DI container. If None, one is
                       bootstrapped lazily on first use.
        """
        self._config = config
        self._container = container
        self._agent: Optional["Agent"] = None
        self._initialized = False

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    async def _ensure_initialized(self) -> "Agent":
        """Lazily initialize agent via Agent.create() — the framework's own API."""
        if self._agent is not None:
            return self._agent

        from victor.framework.agent import Agent
        from victor.framework.config_models import VictorConfig

        config = self._config

        # Bootstrap DI container if not provided
        if self._container is None:
            self._container = self._bootstrap_container()

        # Use the framework's own Agent.create() — NOT bypass it
        agent_kwargs = config.agent.to_agent_create_kwargs()
        self._agent = await Agent.create(**agent_kwargs)
        self._initialized = True

        logger.info(
            "VictorClient initialized (provider=%s, model=%s)",
            config.agent.provider,
            config.agent.model or "default",
        )
        return self._agent

    def _bootstrap_container(self) -> "ServiceContainer":
        """Bootstrap the DI container with core services.

        This is the FIX for Issue #3: ServiceContainer now participates
        in the application lifecycle instead of being orphaned.
        """
        try:
            from victor.core.container import ServiceContainer

            container = ServiceContainer()

            # Register typed config as singleton
            container.register_instance(type(self._config), self._config)

            # Bootstrap core services (lazy — they resolve on first use)
            try:
                from victor.core.bootstrap import bootstrap_container

                bootstrap_container(container)
            except ImportError:
                logger.debug("bootstrap_container not available, using minimal DI")

            return container
        except Exception as e:
            logger.warning(f"DI container bootstrap failed, continuing without: {e}")
            from victor.core.container import ServiceContainer

            return ServiceContainer()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API — the UI layer uses ONLY these methods
    # ─────────────────────────────────────────────────────────────────────────

    async def chat(
        self,
        message: str,
        *,
        stream: bool = False,
    ) -> _ChatResult:
        """Send a single message and get a response.

        Args:
            message: User's message
            stream: If True, use streaming internally but return final result

        Returns:
            ChatResult with response content and metadata
        """
        agent = await self._ensure_initialized()
        result = await agent.run(message)

        return _ChatResult(
            content=result.content if hasattr(result, "content") else str(result),
            success=True,
            metadata={
                "provider": self._config.agent.provider,
                "model": self._config.agent.model,
            },
        )

    async def stream(
        self,
        message: str,
    ) -> AsyncIterator[_StreamEvent]:
        """Send a message and yield streaming events.

        Args:
            message: User's message

        Yields:
            StreamEvent instances (content, thinking, tool_call, etc.)
        """
        from victor.framework.events import EventType

        agent = await self._ensure_initialized()

        async for event in agent.stream(message):
            if event.type == EventType.CONTENT:
                yield _StreamEvent(
                    "content",
                    content=event.content,
                    metadata=getattr(event, "metadata", {}),
                )
            elif event.type == EventType.THINKING:
                yield _StreamEvent(
                    "thinking",
                    content=event.content,
                    metadata=getattr(event, "metadata", {}),
                )
            elif event.type == EventType.TOOL_CALL:
                yield _StreamEvent(
                    "tool_call",
                    tool_name=getattr(event, "tool_name", None),
                    content=str(getattr(event, "arguments", "")),
                    metadata=getattr(event, "metadata", {}),
                )
            elif event.type == EventType.TOOL_RESULT:
                yield _StreamEvent(
                    "tool_result",
                    tool_name=getattr(event, "tool_name", None),
                    content=getattr(event, "content", None),
                    metadata=getattr(event, "metadata", {}),
                )
            elif event.type == EventType.ERROR:
                yield _StreamEvent(
                    "error",
                    content=getattr(event, "content", str(event)),
                    metadata=getattr(event, "metadata", {}),
                )
            else:
                yield _StreamEvent(
                    getattr(event.type, "value", str(event.type)),
                    content=getattr(event, "content", None),
                    metadata={"raw_event": event},
                )

    async def create_session(self) -> "_ChatSession":
        """Create an interactive multi-turn chat session.

        Returns:
            ChatSession for multi-turn conversation
        """
        agent = await self._ensure_initialized()
        return _ChatSession(agent, self._config)

    async def run_workflow(
        self,
        workflow_name: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a named workflow via the Agent's workflow API.

        Args:
            workflow_name: Registered workflow name
            inputs: Workflow input parameters

        Returns:
            Workflow execution result
        """
        agent = await self._ensure_initialized()
        return await agent.run_workflow(workflow_name, inputs=inputs or {})

    def get_available_workflows(self) -> List[str]:
        """List available workflow names."""
        if self._agent:
            return self._agent.get_available_workflows()
        return []

    def get_available_verticals(self) -> List[str]:
        """List available vertical names."""
        try:
            from victor.framework.shim import list_verticals

            return list_verticals()
        except Exception:
            return []

    def get_available_providers(self) -> List[str]:
        """List available provider names."""
        try:
            from victor.providers import list_available_providers

            return list_available_providers()
        except Exception:
            return ["anthropic", "openai", "ollama", "google"]

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle Management
    # ─────────────────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Clean up agent and container resources."""
        if self._agent is not None:
            await self._agent.close()
            self._agent = None

        if self._container is not None:
            try:
                await self._container.adispose()
            except Exception as e:
                logger.debug(f"Container dispose error: {e}")
            self._container = None

        self._initialized = False

    async def __aenter__(self) -> "VictorClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "pending"
        return (
            f"VictorClient(provider={self._config.agent.provider}, "
            f"model={self._config.agent.model}, status={status})"
        )


class _ChatSession:
    """Interactive multi-turn chat session."""

    def __init__(self, agent: "Agent", config: "VictorConfig") -> None:
        self._agent = agent
        self._config = config
        self._history: List[Dict[str, Any]] = []

    async def send(self, message: str) -> _ChatResult:
        """Send a message in the session."""
        result = await self._agent.run(message)
        self._history.append({"role": "user", "content": message})

        response_content = result.content if hasattr(result, "content") else str(result)
        self._history.append({"role": "assistant", "content": response_content})

        return _ChatResult(
            content=response_content,
            success=True,
            metadata={"provider": self._config.agent.provider},
        )

    async def stream(self, message: str) -> AsyncIterator[_StreamEvent]:
        """Stream a response in the session."""
        from victor.framework.events import EventType

        collected = []
        async for event in self._agent.stream(message):
            if event.type == EventType.CONTENT and event.content:
                collected.append(event.content)
            yield _StreamEvent(
                getattr(event.type, "value", str(event.type)),
                content=getattr(event, "content", None),
                tool_name=getattr(event, "tool_name", None),
                metadata=getattr(event, "metadata", {}),
            )

        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": "".join(collected)})

    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return list(self._history)

    async def close(self) -> None:
        """Close the session."""
        pass


__all__ = [
    "VictorClient",
    "_ChatResult",
    "_StreamEvent",
    "_ChatSession",
]
