"""Agent - Simplified entry point for Victor agents.

This is the main class users interact with. It wraps AgentOrchestrator
and provides a clean, simple API for common use cases.

Example:
    # Simple use case
    agent = await Agent.create(provider="anthropic")
    result = await agent.run("Write a hello world function")
    print(result.content)

    # Streaming
    async for event in agent.stream("Refactor this"):
        if event.type == EventType.CONTENT:
            print(event.content, end="")

    # Escape hatch to full power
    orchestrator = agent.get_orchestrator()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, List, Optional, Type, Union

from victor.framework.config import AgentConfig
from victor.framework.errors import AgentError, CancellationError, ProviderError
from victor.framework.events import Event, EventType
from victor.framework.state import State, StateObserver
from victor.framework.task import TaskResult
from victor.framework.tools import ToolSet, ToolsInput

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.framework.agent_components import AgentSession
    from victor.framework.cqrs_bridge import CQRSBridge, FrameworkEventAdapter
    from victor.observability import EventBus, ObservabilityIntegration
    from victor.verticals.base import VerticalBase, VerticalConfig


class Agent:
    """Simplified interface for creating and using Victor agents.

    The Agent class provides a "golden path" API that covers 90% of use cases
    with minimal configuration. For advanced use cases, access the underlying
    AgentOrchestrator via get_orchestrator().

    Attributes:
        state: Observable agent state (stage, tool usage, files)

    Example - Simple usage:
        agent = await Agent.create(provider="anthropic", model="claude-sonnet-4-20250514")
        result = await agent.run("Write a function to parse JSON")
        print(result.content)

    Example - With tools:
        agent = await Agent.create(
            provider="anthropic",
            tools=["filesystem", "git"]
        )
        result = await agent.run("Create a new feature branch and add a README")

    Example - Streaming with events:
        async for event in agent.stream("Refactor this file"):
            if event.type == EventType.THINKING:
                print(f"Thinking: {event.content}")
            elif event.type == EventType.TOOL_CALL:
                print(f"Tool: {event.tool_name}({event.arguments})")
            elif event.type == EventType.CONTENT:
                print(event.content, end="")

    Example - State observation:
        agent.on_state_change(lambda old, new: print(f"{old} -> {new}"))

    Example - Escape hatch to full power:
        orchestrator = agent.get_orchestrator()
        # Access all internal components
    """

    def __init__(
        self,
        orchestrator: "AgentOrchestrator",
        provider: str = "anthropic",
        model: Optional[str] = None,
        vertical: Optional[Type["VerticalBase"]] = None,
        vertical_config: Optional["VerticalConfig"] = None,
    ) -> None:
        """Initialize Agent with orchestrator. Use Agent.create() instead.

        Args:
            orchestrator: AgentOrchestrator instance
            provider: Provider name for reference
            model: Model name for reference
            vertical: Optional vertical class used to create this agent
            vertical_config: Optional vertical configuration applied
        """
        self._orchestrator = orchestrator
        self._provider = provider
        self._model = model
        self._vertical = vertical
        self._vertical_config = vertical_config
        self._state = State(orchestrator)
        self._state_observers: List[StateObserver] = []
        # CQRS integration (optional)
        self._cqrs_bridge: Optional["CQRSBridge"] = None
        self._cqrs_session_id: Optional[str] = None
        self._cqrs_adapter: Optional["FrameworkEventAdapter"] = None

    @classmethod
    async def create(
        cls,
        provider: str = "anthropic",
        model: Optional[str] = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: ToolsInput = None,
        thinking: bool = False,
        airgapped: bool = False,
        profile: Optional[str] = None,
        workspace: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        vertical: Optional[Type["VerticalBase"]] = None,
        enable_observability: bool = True,
        session_id: Optional[str] = None,
    ) -> "Agent":
        """Create a new Agent instance.

        This is the primary way to create an Agent. For most use cases,
        you only need to specify the provider.

        Args:
            provider: LLM provider name (anthropic, openai, ollama, google, etc.)
            model: Model identifier. If None, uses provider default.
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            tools: Tool configuration - ToolSet, list of category names, or None
            thinking: Enable extended thinking mode (Claude only)
            airgapped: Disable network-dependent tools
            profile: Profile name from ~/.victor/profiles.yaml
            workspace: Working directory for file operations
            config: Advanced configuration (overrides other options)
            vertical: Optional vertical class (e.g., CodingAssistant, ResearchAssistant).
                When provided, the vertical's configuration (tools, system_prompt,
                stages) is automatically applied.
            enable_observability: Auto-initialize ObservabilityIntegration for
                unified event handling. Defaults to True.
            session_id: Optional session ID for event correlation.

        Returns:
            Configured Agent instance

        Raises:
            ProviderError: If provider initialization fails
            AgentError: If configuration is invalid

        Example:
            # Simple
            agent = await Agent.create()

            # With provider
            agent = await Agent.create(provider="openai", model="gpt-4-turbo")

            # With tools
            agent = await Agent.create(tools=ToolSet.coding())

            # With vertical (domain-specific assistant)
            from victor.verticals import CodingAssistant
            agent = await Agent.create(vertical=CodingAssistant)

            # With observability events
            agent = await Agent.create(session_id="my-session")
            agent.subscribe_to_events("TOOL", lambda e: print(f"Tool: {e.name}"))

            # Advanced
            agent = await Agent.create(
                config=AgentConfig(tool_budget=100, enable_semantic_search=True)
            )
        """
        from victor.framework._internal import create_orchestrator_from_options

        # Extract configuration from vertical if provided
        vertical_config: Optional["VerticalConfig"] = None
        system_prompt: Optional[str] = None

        if vertical:
            vertical_config = vertical.get_config()
            # Vertical tools override explicit tools if not provided
            if tools is None:
                tools = vertical_config.tools
            # Get vertical's system prompt
            system_prompt = vertical_config.system_prompt
            # Apply provider hints from vertical
            if vertical_config.provider_hints.get("preferred_providers"):
                prefs = vertical_config.provider_hints["preferred_providers"]
                if provider == "anthropic" and "anthropic" not in prefs and prefs:
                    # Use first preferred provider if default isn't preferred
                    pass  # Keep user's choice, hints are just hints

        try:
            orchestrator = await create_orchestrator_from_options(
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                thinking=thinking,
                airgapped=airgapped,
                profile=profile,
                workspace=workspace,
                config=config,
                system_prompt=system_prompt,
                enable_observability=enable_observability,
                session_id=session_id,
            )
            return cls(
                orchestrator,
                provider=provider,
                model=model,
                vertical=vertical,
                vertical_config=vertical_config,
            )
        except Exception as e:
            if "provider" in str(e).lower() or "api" in str(e).lower():
                raise ProviderError(str(e), provider=provider) from e
            raise AgentError(str(e)) from e

    @classmethod
    def from_orchestrator(cls, orchestrator: "AgentOrchestrator") -> "Agent":
        """Create an Agent wrapper around an existing AgentOrchestrator.

        This is the escape hatch for users who have already configured
        an AgentOrchestrator and want to use the simplified API.

        Args:
            orchestrator: Existing AgentOrchestrator instance

        Returns:
            Agent wrapping the orchestrator
        """
        provider = getattr(orchestrator.provider, "name", "unknown")
        model = getattr(orchestrator, "model", None)
        return cls(orchestrator, provider=provider, model=model)

    # =========================================================================
    # Primary Methods
    # =========================================================================

    async def run(
        self,
        prompt: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskResult:
        """Run a task and return the complete result.

        This is the simplest way to use the agent - provide a prompt and
        get back the complete response including any tool results.

        Args:
            prompt: What the agent should do
            context: Optional context dict (files, variables, etc.)

        Returns:
            TaskResult with content, tool_calls, and metadata

        Example:
            result = await agent.run("Explain the authentication flow")
            print(result.content)

            # With context
            result = await agent.run(
                "Fix the bug in this code",
                context={"file": "auth.py", "error": "IndexError"}
            )
        """
        from victor.framework._internal import collect_tool_calls, format_context_message

        # Collect all events
        events: List[Event] = []
        content_parts: List[str] = []
        final_success = True
        final_error: Optional[str] = None

        try:
            async for event in self.stream(prompt, context=context):
                events.append(event)

                if event.type == EventType.CONTENT:
                    content_parts.append(event.content)
                elif event.type == EventType.ERROR:
                    final_success = False
                    final_error = event.error
                elif event.type == EventType.STREAM_END:
                    if not event.success:
                        final_success = False
                        final_error = event.error

        except CancellationError:
            final_success = False
            final_error = "Operation cancelled"
        except Exception as e:
            final_success = False
            final_error = str(e)

        return TaskResult(
            content="".join(content_parts),
            tool_calls=collect_tool_calls(events),
            success=final_success,
            error=final_error,
            metadata={
                "event_count": len(events),
                "stage": self._state.stage.value,
            },
        )

    async def stream(
        self,
        prompt: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Event]:
        """Stream events as the agent processes a task.

        This provides real-time visibility into the agent's reasoning
        and actions. Events include thinking, tool calls, content, and errors.

        Args:
            prompt: What the agent should do
            context: Optional context dict

        Yields:
            Event objects representing agent actions

        Example:
            async for event in agent.stream("Analyze this codebase"):
                if event.type == EventType.THINKING:
                    print(f"Thinking: {event.content[:50]}...")
                elif event.type == EventType.TOOL_CALL:
                    print(f"Calling {event.tool_name}")
                elif event.type == EventType.TOOL_RESULT:
                    print(f"Result: {event.result[:100]}...")
                elif event.type == EventType.CONTENT:
                    print(event.content, end="", flush=True)
        """
        from victor.framework._internal import format_context_message, stream_with_events

        # Apply context to conversation if provided
        if context:
            context_message = format_context_message(context)
            if context_message:
                # Prepend context to the prompt
                prompt = f"{context_message}\n\n{prompt}"

        # Track state for observers
        old_stage = self._state.stage

        async for event in stream_with_events(self._orchestrator, prompt):
            # Forward to CQRS if enabled
            self._forward_event_to_cqrs(event)

            # Check for state changes and notify observers
            new_stage = self._state.stage
            if new_stage != old_stage:
                old_state = State(self._orchestrator)
                old_state._orchestrator = self._orchestrator
                for observer in self._state_observers:
                    try:
                        observer(old_state, self._state)
                    except Exception:
                        pass  # Don't let observer errors break streaming
                old_stage = new_stage

            yield event

    def chat(self, prompt: str) -> "ChatSession":
        """Start an interactive chat session.

        Returns a ChatSession that maintains conversation context
        across multiple turns.

        Args:
            prompt: Initial message

        Returns:
            ChatSession for multi-turn conversation

        Example:
            session = agent.chat("Let's refactor the auth module")
            response = await session.send("First, show me the current code")
            response = await session.send("Now extract the validation logic")
        """
        return ChatSession(self, prompt)

    # =========================================================================
    # State and Observation
    # =========================================================================

    @property
    def state(self) -> State:
        """Get current agent state.

        Returns:
            State object with stage, tool_calls_used, files_observed, etc.
        """
        return self._state

    @property
    def vertical(self) -> Optional[Type["VerticalBase"]]:
        """Get the vertical class used to create this agent.

        Returns:
            Vertical class or None if not created from a vertical.
        """
        return self._vertical

    @property
    def vertical_config(self) -> Optional["VerticalConfig"]:
        """Get the vertical configuration applied to this agent.

        Returns:
            VerticalConfig or None if not created from a vertical.
        """
        return self._vertical_config

    @property
    def vertical_name(self) -> Optional[str]:
        """Get the name of the vertical used to create this agent.

        Returns:
            Vertical name string or None.
        """
        if self._vertical:
            return self._vertical.name
        return None

    def on_state_change(
        self,
        callback: StateObserver,
    ) -> Callable[[], None]:
        """Register a callback for state changes.

        Args:
            callback: Function called with (old_state, new_state)

        Returns:
            Unsubscribe function

        Example:
            def log_state(old, new):
                print(f"State: {old.stage} -> {new.stage}")

            unsubscribe = agent.on_state_change(log_state)
            # Later: unsubscribe()
        """
        self._state_observers.append(callback)

        def unsubscribe() -> None:
            if callback in self._state_observers:
                self._state_observers.remove(callback)

        return unsubscribe

    # =========================================================================
    # Configuration
    # =========================================================================

    async def switch_model(
        self,
        provider: str,
        model: str,
    ) -> None:
        """Switch to a different model.

        Args:
            provider: New provider name
            model: New model identifier
        """
        if hasattr(self._orchestrator, "provider_manager"):
            await self._orchestrator.provider_manager.switch_provider(provider, model)
        self._provider = provider
        self._model = model

    def set_tools(self, tools: Union[ToolSet, List[str]]) -> None:
        """Update available tools.

        Args:
            tools: ToolSet or list of category names
        """
        from victor.framework._internal import configure_tools

        configure_tools(self._orchestrator, tools)

    # =========================================================================
    # Escape Hatch
    # =========================================================================

    def get_orchestrator(self) -> "AgentOrchestrator":
        """Get the underlying AgentOrchestrator for advanced usage.

        This provides access to all internal components when the
        simplified API is insufficient.

        Returns:
            AgentOrchestrator instance

        Example:
            orchestrator = agent.get_orchestrator()

            # Access decomposed components
            controller = orchestrator.conversation_controller
            pipeline = orchestrator.tool_pipeline

            # Access internal state
            metrics = orchestrator.streaming_controller.get_session_history()
        """
        return self._orchestrator

    # =========================================================================
    # Observability
    # =========================================================================

    @property
    def event_bus(self) -> Optional["EventBus"]:
        """Get the EventBus for subscribing to agent events.

        The EventBus provides access to all agent events including:
        - Tool execution (start/end)
        - State transitions
        - Model requests/responses
        - Errors

        Returns:
            EventBus instance, or None if observability is disabled

        Example:
            from victor.observability import EventCategory

            def on_tool_event(event):
                print(f"Tool: {event.name} - {event.data}")

            agent.event_bus.subscribe(EventCategory.TOOL, on_tool_event)
        """
        observability = getattr(self._orchestrator, "observability", None)
        if observability:
            return observability.event_bus
        return None

    @property
    def observability(self) -> Optional["ObservabilityIntegration"]:
        """Get the ObservabilityIntegration for advanced event handling.

        Returns:
            ObservabilityIntegration instance, or None if disabled
        """
        return getattr(self._orchestrator, "observability", None)

    def subscribe_to_events(
        self,
        category: str,
        handler: Callable[[Any], None],
    ) -> Optional[Callable[[], None]]:
        """Subscribe to events of a specific category.

        Convenience method for subscribing to EventBus events without
        directly importing observability types.

        Args:
            category: Event category name ("TOOL", "STATE", "MODEL", "ERROR", "LIFECYCLE")
            handler: Callback function receiving VictorEvent

        Returns:
            Unsubscribe function, or None if observability is disabled

        Example:
            def log_tools(event):
                print(f"Tool called: {event.name}")

            unsubscribe = agent.subscribe_to_events("TOOL", log_tools)
            # Later: unsubscribe()
        """
        event_bus = self.event_bus
        if not event_bus:
            return None

        from victor.observability import EventCategory

        category_enum = EventCategory[category.upper()]
        return event_bus.subscribe(category_enum, handler)

    # =========================================================================
    # CQRS Integration
    # =========================================================================

    async def enable_cqrs(
        self,
        session_id: Optional[str] = None,
        enable_event_sourcing: bool = True,
    ) -> "CQRSBridge":
        """Enable CQRS architecture for this agent.

        This connects the agent to the CQRS subsystem, enabling:
        - Event sourcing for all agent operations
        - Command/Query separation
        - Session projections for read models
        - Event replay and audit

        Args:
            session_id: Optional session ID for correlation.
            enable_event_sourcing: Whether to source events.

        Returns:
            CQRSBridge instance for direct access.

        Example:
            bridge = await agent.enable_cqrs()

            # Events are now automatically sourced
            async for event in agent.stream("Analyze code"):
                print(event.content)

            # Query session history
            history = await bridge.get_conversation_history(bridge._adapters.keys()[0])
        """
        from victor.framework.cqrs_bridge import CQRSBridge

        if not self._cqrs_bridge:
            self._cqrs_bridge = await CQRSBridge.create(
                enable_event_sourcing=enable_event_sourcing,
                enable_observability=self.event_bus is not None,
            )

        # Connect this agent
        self._cqrs_session_id = self._cqrs_bridge.connect_agent(
            self,
            session_id=session_id,
        )

        return self._cqrs_bridge

    @property
    def cqrs_bridge(self) -> Optional["CQRSBridge"]:
        """Get the CQRS bridge if enabled.

        Returns:
            CQRSBridge instance or None.
        """
        return self._cqrs_bridge

    @property
    def cqrs_session_id(self) -> Optional[str]:
        """Get the CQRS session ID if enabled.

        Returns:
            Session ID or None.
        """
        return self._cqrs_session_id

    async def cqrs_get_session(self) -> Dict[str, Any]:
        """Get current session via CQRS query.

        Requires CQRS to be enabled via enable_cqrs().

        Returns:
            Session details.

        Raises:
            AgentError: If CQRS is not enabled.
        """
        if not self._cqrs_bridge or not self._cqrs_session_id:
            raise AgentError("CQRS not enabled. Call enable_cqrs() first.")

        return await self._cqrs_bridge.get_session(self._cqrs_session_id)

    async def cqrs_get_history(self, limit: int = 100) -> Dict[str, Any]:
        """Get conversation history via CQRS query.

        Requires CQRS to be enabled via enable_cqrs().

        Args:
            limit: Maximum messages to retrieve.

        Returns:
            Conversation history.

        Raises:
            AgentError: If CQRS is not enabled.
        """
        if not self._cqrs_bridge or not self._cqrs_session_id:
            raise AgentError("CQRS not enabled. Call enable_cqrs() first.")

        return await self._cqrs_bridge.get_conversation_history(
            self._cqrs_session_id,
            limit=limit,
        )

    async def cqrs_get_metrics(self) -> Dict[str, Any]:
        """Get session metrics via CQRS query.

        Requires CQRS to be enabled via enable_cqrs().

        Returns:
            Session metrics.

        Raises:
            AgentError: If CQRS is not enabled.
        """
        if not self._cqrs_bridge or not self._cqrs_session_id:
            raise AgentError("CQRS not enabled. Call enable_cqrs() first.")

        return await self._cqrs_bridge.get_metrics(self._cqrs_session_id)

    def _forward_event_to_cqrs(self, event: Event) -> None:
        """Forward a framework event to CQRS subsystem.

        Called internally during streaming to source events.

        Args:
            event: Framework Event to forward.
        """
        if self._cqrs_adapter:
            self._cqrs_adapter.forward(event)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def reset(self) -> None:
        """Reset conversation history and state."""
        self._orchestrator.reset_conversation()
        self._state.reset()

    async def close(self) -> None:
        """Clean up resources."""
        # Clean up CQRS bridge
        if self._cqrs_bridge:
            if self._cqrs_session_id:
                self._cqrs_bridge.disconnect_agent(self._cqrs_session_id)
            self._cqrs_bridge.close()
            self._cqrs_bridge = None
            self._cqrs_session_id = None
            self._cqrs_adapter = None

        # Clean up orchestrator
        if hasattr(self._orchestrator, "close"):
            await self._orchestrator.close()

    async def __aenter__(self) -> "Agent":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        return f"Agent(provider={self._provider}, model={self._model}, state={self._state})"


class ChatSession:
    """Multi-turn chat session with maintained context.

    Use this for conversations that span multiple exchanges.
    The session maintains conversation history automatically.

    Note: ChatSession is a simplified interface that delegates to AgentSession
    for full functionality. For advanced features like lifecycle hooks, metrics,
    and pause/resume support, use AgentSession directly.

    Example:
        session = agent.chat("Let's analyze the codebase")
        r1 = await session.send("What's the main entry point?")
        r2 = await session.send("Now explain that function")
        print(session.history)

    Advanced usage (via AgentSession):
        from victor.framework.agent_components import AgentSession, SessionLifecycleHooks

        hooks = SessionLifecycleHooks(
            on_start=lambda s: print(f"Session started"),
            on_close=lambda s, m: print(f"Completed {m.total_turns} turns"),
        )
        session = AgentSession(agent, "Let's analyze", hooks=hooks)
    """

    def __init__(self, agent: Agent, initial_prompt: str) -> None:
        """Initialize chat session.

        Args:
            agent: Agent instance
            initial_prompt: First message in the conversation
        """
        # Delegate to AgentSession for actual implementation
        from victor.framework.agent_components import AgentSession

        self._delegate = AgentSession(agent, initial_prompt)

    async def send(self, message: str) -> TaskResult:
        """Send a message in the conversation.

        Args:
            message: User message

        Returns:
            TaskResult with response
        """
        return await self._delegate.send(message)

    async def stream(self, message: str) -> AsyncIterator[Event]:
        """Stream a response in the conversation.

        Args:
            message: User message

        Yields:
            Event objects
        """
        async for event in self._delegate.stream(message):
            yield event

    @property
    def turn_count(self) -> int:
        """Get number of turns in the conversation."""
        return self._delegate.turn_count

    @property
    def history(self) -> List[Dict[str, str]]:
        """Get conversation history.

        Returns:
            List of message dictionaries with role and content
        """
        return self._delegate.history

    # Expose the underlying AgentSession for advanced usage
    def get_session(self) -> "AgentSession":
        """Get the underlying AgentSession for advanced features.

        Use this to access additional capabilities like:
        - Lifecycle hooks
        - Metrics tracking
        - Pause/resume/close
        - Session state management

        Returns:
            The underlying AgentSession instance
        """
        from victor.framework.agent_components import AgentSession

        return self._delegate
