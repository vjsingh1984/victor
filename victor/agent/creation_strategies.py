"""Agent creation strategies following the Strategy pattern.

This module provides different strategies for creating agents, allowing
the CLI, TUI, and one-shot modes to use consistent initialization logic
while maintaining their unique requirements.

Design Pattern: Strategy
- Context: Agent creation across different modes (CLI, TUI, one-shot)
- Strategy Interface: AgentCreationStrategy
- Concrete Strategies: FrameworkStrategy, LegacyStrategy

SOLID Principles Applied:
- Single Responsibility: Each strategy handles one creation method
- Open/Closed: New strategies can be added without modifying existing code
- Liskov Substitution: All strategies are interchangeable
- Interface Segregation: Focused interfaces for specific needs
- Dependency Inversion: High-level modules depend on abstractions, not concrete implementations
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    # Use protocol for type hint to avoid circular dependency (DIP compliance)
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class AgentCreationContext:
    """Context for agent creation.

    Encapsulates all parameters needed for agent creation,
    replacing scattered individual parameters.
    """

    settings: "Settings"
    profile: str
    provider: Optional[str] = None
    model: Optional[str] = None
    thinking: bool = False
    vertical: Optional[Any] = None
    enable_observability: bool = True
    session_id: Optional[str] = None
    tool_budget: Optional[int] = None
    max_iterations: Optional[int] = None
    mode: Optional[str] = None
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "profile": self.profile,
            "provider": self.provider,
            "model": self.model,
            "thinking": self.thinking,
            "vertical": self.vertical.__name__ if self.vertical else None,
            "enable_observability": self.enable_observability,
            "session_id": self.session_id,
            "tool_budget": self.tool_budget,
            "max_iterations": self.max_iterations,
            "mode": self.mode,
            "metadata": self.metadata,
        }


class AgentCreationStrategy(ABC):
    """Strategy interface for agent creation.

    Concrete implementations define how to create an agent
    with different initialization approaches (framework vs legacy).

    This follows the Strategy pattern and Open/Closed principle:
    - New creation methods can be added by creating new strategies
    - Existing code doesn't need to change
    """

    @abstractmethod
    async def create_agent(self, context: AgentCreationContext) -> "AgentOrchestrator":
        """Create an agent orchestrator.

        Args:
            context: AgentCreationContext with all parameters

        Returns:
            Configured AgentOrchestrator instance
        """
        pass

    @abstractmethod
    def supports_observability(self) -> bool:
        """Whether this strategy supports observability features."""
        pass

    @abstractmethod
    def supports_verticals(self) -> bool:
        """Whether this strategy supports vertical integration."""
        pass


class FrameworkStrategy(AgentCreationStrategy):
    """Agent creation using FrameworkShim with full feature support.

    This is the recommended strategy that provides:
    - Observability integration
    - Vertical system integration
    - Enhanced prompt building with contributors
    - Event tracking
    """

    def __init__(self) -> None:
        self._shim: Optional[Any] = None

    async def create_agent(self, context: AgentCreationContext) -> "AgentOrchestrator":
        """Create agent using FrameworkShim."""
        from victor.framework.shim import FrameworkShim

        self._shim = FrameworkShim(
            context.settings,
            profile_name=context.profile,
            thinking=context.thinking,
            vertical=context.vertical,
            enable_observability=context.enable_observability,
            session_id=context.session_id,
        )

        agent = await self._shim.create_orchestrator()

        # Emit session start event
        self._shim.emit_session_start(
            {
                "mode": context.metadata.get("mode", "unknown"),
                "vertical": context.vertical.__name__ if context.vertical else None,
                "thinking": context.thinking,
            }
        )

        # Apply overrides
        self._apply_overrides(agent, context)

        return agent  # type: ignore[return-value]

    def supports_observability(self) -> bool:
        return True

    def supports_verticals(self) -> bool:
        return True

    def _apply_overrides(self, agent: OrchestratorProtocol, context: AgentCreationContext) -> None:
        """Apply budget, iteration, and mode overrides to agent."""
        if context.tool_budget is not None and hasattr(agent, 'unified_tracker') and agent.unified_tracker is not None:
            agent.unified_tracker.set_tool_budget(context.tool_budget, user_override=True)

        if context.max_iterations is not None and hasattr(agent, 'unified_tracker') and agent.unified_tracker is not None:
            agent.unified_tracker.set_max_iterations(context.max_iterations, user_override=True)

        if context.mode:
            from victor.agent.mode_controller import AgentMode, get_mode_controller

            controller = get_mode_controller()
            try:
                controller.switch_mode(AgentMode(context.mode))
            except Exception as e:
                logger.warning(f"Failed to switch mode to {context.mode}: {e}")


class LegacyStrategy(AgentCreationStrategy):
    """Agent creation using direct AgentOrchestrator initialization.

    This is a legacy strategy for backward compatibility and troubleshooting.
    It bypasses framework features and uses direct orchestrator creation.

    Use only when:
    - Troubleshooting framework issues
    - Need to bypass observability
    - Testing without vertical integration
    """

    async def create_agent(self, context: AgentCreationContext) -> "AgentOrchestrator":
        """Create agent using direct AgentOrchestrator.from_settings()."""
        from victor.agent.orchestrator import AgentOrchestrator

        agent = await AgentOrchestrator.from_settings(
            context.settings,
            context.profile,
            thinking=context.thinking,
        )

        # Apply overrides (same as framework strategy)
        self._apply_overrides(agent, context)

        return agent

    def supports_observability(self) -> bool:
        return False

    def supports_verticals(self) -> bool:
        return False

    def _apply_overrides(self, agent: "AgentOrchestrator", context: AgentCreationContext) -> None:
        """Apply budget, iteration, and mode overrides to agent."""
        if context.tool_budget is not None:
            if agent.unified_tracker is not None:
                agent.unified_tracker.set_tool_budget(context.tool_budget, user_override=True)

        if context.max_iterations is not None:
            if agent.unified_tracker is not None:
                agent.unified_tracker.set_max_iterations(context.max_iterations, user_override=True)

        if context.mode:
            from victor.agent.mode_controller import AgentMode, get_mode_controller

            controller = get_mode_controller()
            try:
                controller.switch_mode(AgentMode(context.mode))
            except Exception as e:
                logger.warning(f"Failed to switch mode to {context.mode}: {e}")


class AgentCreationFactory:
    """Factory for creating agents using appropriate strategies.

    This follows the Factory pattern combined with Strategy pattern:
    - Factory: Decides which strategy to use
    - Strategy: Implements the creation logic

    Usage:
        factory = AgentCreationFactory()

        # Automatic strategy selection
        context = AgentCreationContext(...)
        agent = await factory.create_agent(context)

        # Explicit strategy selection
        agent = await factory.create_agent(context, strategy=FrameworkStrategy())
    """

    def __init__(self, default_strategy: Optional[AgentCreationStrategy] = None):
        """Initialize factory with default strategy.

        Args:
            default_strategy: Default strategy to use. Defaults to FrameworkStrategy.
        """
        self._default_strategy = default_strategy or FrameworkStrategy()

    async def create_agent(
        self,
        context: AgentCreationContext,
        strategy: Optional[AgentCreationStrategy] = None,
    ) -> "AgentOrchestrator":
        """Create an agent using the specified or default strategy.

        Args:
            context: AgentCreationContext with all parameters
            strategy: Optional strategy override. If not provided, uses default.

        Returns:
            Configured AgentOrchestrator instance
        """
        creation_strategy = strategy or self._default_strategy

        logger.info(
            f"Creating agent using {creation_strategy.__class__.__name__} "
            f"(observability: {creation_strategy.supports_observability()}, "
            f"verticals: {creation_strategy.supports_verticals()})"
        )

        return await creation_strategy.create_agent(context)

    @staticmethod
    def create_context_from_cli_args(
        settings: "Settings",
        profile: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        mode: Optional[str] = None,
        tool_budget: Optional[int] = None,
        max_iterations: Optional[int] = None,
        vertical: Optional[str] = None,
        enable_observability: bool = True,
        session_id: Optional[str] = None,
        **metadata: Any,
    ) -> AgentCreationContext:
        """Create AgentCreationContext from CLI arguments.

        This is a convenience method that bridges the gap between
        CLI argument parsing and the context object.

        Args:
            settings: Settings instance
            profile: Profile name
            provider: Optional provider override
            model: Optional model override
            thinking: Enable thinking mode
            mode: Agent mode (build, plan, explore)
            tool_budget: Tool call budget
            max_iterations: Maximum iterations
            vertical: Vertical name
            enable_observability: Enable observability features
            session_id: Session ID for tracking
            **metadata: Additional metadata

        Returns:
            AgentCreationContext instance
        """
        from victor.core.verticals import get_vertical

        vertical_class = get_vertical(vertical) if vertical else None

        return AgentCreationContext(
            settings=settings,
            profile=profile,
            provider=provider,
            model=model,
            thinking=thinking,
            vertical=vertical_class,
            enable_observability=enable_observability,
            session_id=session_id,
            tool_budget=tool_budget,
            max_iterations=max_iterations,
            mode=mode,
            metadata=metadata,
        )


__all__ = [
    "AgentCreationStrategy",
    "FrameworkStrategy",
    "LegacyStrategy",
    "AgentCreationFactory",
    "AgentCreationContext",
]
