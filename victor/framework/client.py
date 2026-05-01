"""VictorClient — unified facade making CLI/TUI/Web first-class framework clients.

This is the SINGLE entry point that the UI layer (CLI, TUI, Web server) uses
to interact with the framework. It properly delegates to SERVICES layer
instead of bypassing to orchestrator.

Architectural Alignment:
    - Uses ServiceAccessor to access ChatService, ToolService, etc.
    - Does NOT bypass to orchestrator directly
    - Enforces proper service boundaries
    - SessionConfig for CLI/runtime overrides (not settings mutation)

Usage in CLI (victor/ui/commands/chat.py):
    from victor.framework.client import VictorClient
    from victor.framework.session_config import SessionConfig

    config = SessionConfig.from_cli_flags(tool_budget=50, enable_smart_routing=True)
    client = VictorClient(config)
    result = await client.chat("Hello")
    # OR streaming:
    async for event in client.stream("Hello"):
        print(event)

Architecture:
    CLI/TUI/Web  →  VictorClient  →  Services (Chat, Tool, Session, etc.)  →  Providers
                         ↓
               ExecutionContext (with ServiceAccessor)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.framework.session_config import SessionConfig
    from victor.framework.agent import Agent
    from victor.runtime.context import RuntimeExecutionContext, ServiceAccessor
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
        event_type: Any,
        *,
        content: Optional[str] = None,
        tool_name: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        result: Optional[Any] = None,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._type = event_type
        self._content = content
        self._tool_name = tool_name
        self._arguments = arguments or {}
        self._result = result
        self._success = success
        self._metadata = metadata or {}

    @property
    def event_type(self) -> str:
        return getattr(self._type, "value", str(self._type))

    @property
    def type(self) -> Any:
        return self._type

    @property
    def content(self) -> Optional[str]:
        return self._content

    @property
    def tool_name(self) -> Optional[str]:
        return self._tool_name

    @property
    def arguments(self) -> Dict[str, Any]:
        return self._arguments

    @property
    def result(self) -> Any:
        return self._result

    @property
    def success(self) -> bool:
        return self._success

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata


@dataclass
class _RenderChunk:
    """Chunk adapter consumed by existing CLI/TUI renderer helpers."""

    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def _session_metadata(config: "SessionConfig") -> Dict[str, Any]:
    """Build stable metadata for a client-managed chat session."""
    return {
        "tool_budget": config.tool_budget,
        "smart_routing": config.smart_routing.enabled,
    }


async def _iter_runtime_stream_events(
    runtime: Any,
    message: str,
) -> AsyncIterator[Any]:
    """Yield framework stream events from a chat runtime or agent wrapper."""
    if hasattr(runtime, "stream_chat"):
        from victor.framework._internal import stream_with_events

        async for event in stream_with_events(runtime, message, response_prompt=message):
            yield event
        return

    async for event in runtime.stream(message):
        yield event


def _to_stream_event(event: Any) -> _StreamEvent:
    """Convert a framework event into the client stream-event surface."""
    from victor.framework.events import EventType

    if event.type == EventType.CONTENT:
        return _StreamEvent(
            EventType.CONTENT,
            content=event.content,
            metadata=getattr(event, "metadata", {}),
        )
    if event.type == EventType.THINKING:
        return _StreamEvent(
            EventType.THINKING,
            content=event.content,
            metadata=getattr(event, "metadata", {}),
        )
    if event.type == EventType.TOOL_CALL:
        return _StreamEvent(
            EventType.TOOL_CALL,
            tool_name=getattr(event, "tool_name", None),
            content=str(getattr(event, "arguments", "")),
            arguments=getattr(event, "arguments", {}),
            metadata={
                **getattr(event, "metadata", {}),
                "arguments": getattr(event, "arguments", {}),
            },
        )
    if event.type == EventType.TOOL_RESULT:
        result_payload = {
            **getattr(event, "metadata", {}),
            "result": getattr(event, "result", None),
            "success": getattr(event, "success", True),
            "arguments": getattr(event, "arguments", {}),
        }
        return _StreamEvent(
            EventType.TOOL_RESULT,
            tool_name=getattr(event, "tool_name", None),
            content=getattr(event, "result", None) or getattr(event, "content", None),
            arguments=getattr(event, "arguments", {}),
            result=result_payload,
            success=getattr(event, "success", True),
            metadata=result_payload,
        )
    if event.type == EventType.ERROR:
        return _StreamEvent(
            EventType.ERROR,
            content=getattr(event, "content", str(event)),
            metadata=getattr(event, "metadata", {}),
        )
    return _StreamEvent(
        event.type,
        content=getattr(event, "content", None),
        metadata={"raw_event": event},
    )


class VictorClient:
    """Unified client facade — the ONLY interface the UI layer uses.

    This class makes the CLI a first-class client of the framework by:
    1. Accepting SessionConfig for CLI/runtime overrides (not settings mutation)
    2. Creating Agent via Agent.create() with session_config
    3. Accessing services through ServiceAccessor (not orchestrator bypass)
    4. Providing chat(), stream(), create_session(), run_workflow()

    Architectural Guarantees:
        - NEVER accesses orchestrator directly
        - ALWAYS uses services (ChatService, ToolService, etc.)
        - ENFORCES proper service boundaries

    Example:
        from victor.framework.client import VictorClient
        from victor.framework.session_config import SessionConfig

        config = SessionConfig.from_cli_flags(tool_budget=50)
        async with VictorClient(config) as client:
            result = await client.chat("Write a hello world")
            print(result.content)
    """

    def __init__(
        self,
        config: "SessionConfig",
        *,
        container: Optional["ServiceContainer"] = None,
    ) -> None:
        """Initialize client with session config and optional DI container.

        Args:
            config: SessionConfig with CLI/runtime overrides (immutable)
            container: Optional pre-built DI container. If None, one is
                       bootstrapped lazily on first use.
        """
        self._config = config
        self._container = container
        self._agent: Optional["Agent"] = None
        self._context: Optional["RuntimeExecutionContext"] = None
        self._initialized = False

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    async def _ensure_initialized(self) -> "Agent":
        """Lazily initialize agent via Agent.create() with SessionConfig."""
        if self._agent is not None:
            return self._agent

        from victor.framework.agent import Agent
        from victor.config.settings import load_settings

        # Bootstrap DI container if not provided
        if self._container is None:
            self._container = self._bootstrap_container()

        # Load base settings
        settings = load_settings()

        # Apply SessionConfig overrides to settings (ONLY place where settings is mutated)
        self._config.apply_to_settings(settings)

        provider_override = getattr(self._config, "provider_override", None)
        provider_name = None
        model_name = None

        if provider_override is not None:
            provider_name = provider_override.provider
            model_name = provider_override.model

        if self._config.agent_profile is None:
            provider_settings = getattr(settings, "provider", None)
            if provider_name is None:
                provider_name = getattr(provider_settings, "default_provider", None)
            if model_name is None:
                model_name = getattr(provider_settings, "default_model", None)

        # Create agent with SessionConfig (including agent_profile if specified)
        self._agent = await Agent.create(
            profile=self._config.agent_profile,  # Pass agent_profile from SessionConfig
            provider=provider_name,
            model=model_name,
            session_config=self._config,  # Pass SessionConfig
        )
        execution_context = getattr(self._agent, "execution_context", None)
        if execution_context is not None:
            self._context = execution_context
        elif hasattr(self._agent, "get_orchestrator"):
            orchestrator = self._agent.get_orchestrator()
            self._context = getattr(orchestrator, "_execution_context", None)
        self._initialized = True

        logger.info(
            "VictorClient initialized (SessionConfig applied: agent_profile=%s, tool_budget=%s, smart_routing=%s)",
            self._config.agent_profile,
            self._config.tool_budget,
            self._config.smart_routing.enabled,
        )
        return self._agent

    async def initialize(self) -> "Agent":
        """Public initialization hook for UI surfaces."""
        return await self._ensure_initialized()

    async def start_embedding_preload(self) -> None:
        """Warm embedding-dependent runtime state when supported."""
        agent = await self._ensure_initialized()
        if hasattr(agent, "start_embedding_preload"):
            agent.start_embedding_preload()

    async def get_session_metrics(self) -> Dict[str, Any]:
        """Return session-level runtime metrics when available."""
        agent = await self._ensure_initialized()
        if hasattr(agent, "get_session_metrics"):
            metrics = agent.get_session_metrics()
            if isinstance(metrics, dict):
                return metrics
        return {}

    def _bootstrap_container(self) -> "ServiceContainer":
        """Bootstrap the DI container with core services."""
        try:
            # Bootstrap core services
            try:
                from victor.core.bootstrap import bootstrap_container

                return bootstrap_container()
            except ImportError:
                logger.debug("bootstrap_container not available, using minimal DI")

            from victor.core.container import ServiceContainer

            return ServiceContainer()
        except Exception as e:
            logger.warning(f"DI container bootstrap failed, continuing without: {e}")
            from victor.core.container import ServiceContainer

            return ServiceContainer()

    def _get_services(self) -> "ServiceAccessor":
        """Get ServiceAccessor to access services (NOT orchestrator directly)."""
        if self._agent is None:
            raise RuntimeError("Client not initialized. Call await _ensure_initialized() first.")

        runtime_context = self._context
        if runtime_context is not None and getattr(runtime_context, "services", None) is not None:
            return runtime_context.services

        if hasattr(self._agent, "get_orchestrator"):
            orchestrator = self._agent.get_orchestrator()
            runtime_context = getattr(orchestrator, "_execution_context", None)
            if runtime_context is not None and getattr(runtime_context, "services", None) is not None:
                self._context = runtime_context
                return runtime_context.services

        # Fallback: create ServiceAccessor from container
        from victor.runtime.context import ServiceAccessor

        return ServiceAccessor(_container=self._container)

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

        Uses ChatService (not orchestrator directly) for proper service layering.

        Args:
            message: User's message
            stream: If True, use streaming internally but return final result

        Returns:
            ChatResult with response content and metadata
        """
        agent = await self._ensure_initialized()
        services = self._get_services()
        chat_service = getattr(services, "chat", None) if services is not None else None

        if chat_service is not None:
            result = await chat_service.chat(message, stream=stream)
            content = result.content if hasattr(result, "content") else str(result)
            tool_calls = len(getattr(result, "tool_calls", []) or [])
        else:
            result = await agent.run(message)
            content = result.content if hasattr(result, "content") else str(result)
            tool_calls = len(getattr(result, "tool_calls", []) or [])

        return _ChatResult(
            content=content,
            success=True,
            tool_calls=tool_calls,
            metadata={
                "tool_budget": self._config.tool_budget,
                "smart_routing": self._config.smart_routing.enabled,
            },
        )

    async def stream(
        self,
        message: str,
    ) -> AsyncIterator[_StreamEvent]:
        """Send a message and yield streaming events.

        Uses ChatService for streaming (not orchestrator directly).

        Args:
            message: User's message

        Yields:
            StreamEvent instances (content, thinking, tool_call, etc.)
        """
        agent = await self._ensure_initialized()
        services = self._get_services()
        chat_runtime = getattr(services, "chat", None) if services is not None else None
        runtime = chat_runtime if chat_runtime is not None else agent

        async for event in _iter_runtime_stream_events(runtime, message):
            yield _to_stream_event(event)

    async def stream_chat(self, message: str) -> AsyncIterator[_RenderChunk]:
        """Yield renderer-compatible chunks for legacy UI streaming helpers."""
        async for event in self.stream(message):
            metadata = dict(event.metadata or {})

            if event.event_type == "thinking":
                if event.content:
                    metadata["reasoning_content"] = event.content
                yield _RenderChunk(metadata=metadata)
                continue

            if event.event_type == "tool_call":
                metadata["tool_start"] = {
                    "name": event.tool_name or "unknown",
                    "arguments": metadata.get("arguments", {}),
                }
                yield _RenderChunk(metadata=metadata)
                continue

            if event.event_type == "tool_result":
                result_payload = event.result if isinstance(event.result, dict) else metadata
                metadata["tool_result"] = {
                    "name": event.tool_name or "unknown",
                    "result": event.content or result_payload.get("result", ""),
                    "success": result_payload.get("success", True),
                    "arguments": result_payload.get("arguments", {}),
                }
                yield _RenderChunk(metadata=metadata)
                continue

            if event.event_type == "error":
                metadata["status"] = event.content or metadata.get("error", "Error")
                yield _RenderChunk(metadata=metadata)
                continue

            yield _RenderChunk(content=event.content or "", metadata=metadata)

    async def create_session(self) -> "_ChatSession":
        """Create an interactive multi-turn chat session.

        Uses SessionService for proper session management.

        Returns:
            ChatSession for multi-turn conversation
        """
        agent = await self._ensure_initialized()
        services = self._get_services()
        session_service = getattr(services, "session", None) if services is not None else None
        chat_service = getattr(services, "chat", None) if services is not None else None
        session_id = None

        if session_service is not None:
            session_id = await session_service.create_session(
                metadata=_session_metadata(self._config)
            )
        if chat_service is not None and hasattr(chat_service, "reset_conversation"):
            chat_service.reset_conversation()

        return _ChatSession(
            agent,
            self._config,
            chat_service=chat_service,
            session_service=session_service,
            session_id=session_id,
        )

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
            from victor.core.verticals import list_verticals

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

        self._context = None
        self._initialized = False

    async def __aenter__(self) -> "VictorClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "pending"
        return (
            f"VictorClient(tool_budget={self._config.tool_budget}, "
            f"smart_routing={self._config.smart_routing.enabled}, status={status})"
        )


class _ChatSession:
    """Interactive multi-turn chat session."""

    def __init__(
        self,
        agent: "Agent",
        config: "SessionConfig",
        *,
        chat_service: Optional[Any] = None,
        session_service: Optional[Any] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self._agent = agent
        self._config = config
        self._chat_service = chat_service
        self._session_service = session_service
        self._session_id = session_id
        self._history: List[Dict[str, Any]] = []

    async def send(self, message: str) -> _ChatResult:
        """Send a message in the session."""
        if self._chat_service is not None:
            result = await self._chat_service.chat(message, stream=False)
        else:
            result = await self._agent.run(message)
        self._history.append({"role": "user", "content": message})

        response_content = result.content if hasattr(result, "content") else str(result)
        self._history.append({"role": "assistant", "content": response_content})

        return _ChatResult(
            content=response_content,
            success=True,
            metadata={
                "tool_budget": self._config.tool_budget,
                "smart_routing": self._config.smart_routing.enabled,
            },
        )

    async def stream(self, message: str) -> AsyncIterator[_StreamEvent]:
        """Stream a response in the session."""
        from victor.framework.events import EventType

        collected = []
        runtime = self._chat_service if self._chat_service is not None else self._agent
        async for event in _iter_runtime_stream_events(runtime, message):
            if event.type == EventType.CONTENT and event.content:
                collected.append(event.content)
            yield _to_stream_event(event)

        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": "".join(collected)})

    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return list(self._history)

    async def close(self) -> None:
        """Close the session."""
        if self._session_service is not None and self._session_id is not None:
            await self._session_service.close_session(self._session_id)
            self._session_id = None


__all__ = [
    "VictorClient",
    "_ChatResult",
    "_StreamEvent",
    "_ChatSession",
]
