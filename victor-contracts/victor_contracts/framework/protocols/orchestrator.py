"""Orchestrator protocol definitions.

These protocols define how verticals interact with the agent orchestrator.
"""

from __future__ import annotations

from typing import (
    Protocol,
    runtime_checkable,
    Dict,
    Any,
    List,
    Optional,
    Callable,
    Awaitable,
)


@runtime_checkable
class Orchestrator(Protocol):
    """Protocol for agent orchestration.

    Orchestrators manage the execution flow of agents, including
    tool calls, LLM interactions, and result processing.
    """

    async def run(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a single-turn agent interaction.

        Args:
            prompt: User's input prompt
            context: Additional context for execution

        Returns:
            Execution result dictionary
        """
        ...

    async def stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Callable[[], Awaitable[Dict[str, Any]]]:
        """Run a streaming agent interaction.

        Args:
            prompt: User's input prompt
            context: Additional context for execution

        Returns:
            Async callable that yields streaming chunks
        """
        ...

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a multi-turn chat interaction.

        Args:
            messages: List of message dictionaries
            context: Additional context for execution

        Returns:
            Execution result dictionary
        """
        ...

    def get_available_tools(self) -> List[str]:
        """Return list of available tool names.

        Returns:
            List of tool names
        """
        ...

    def get_state(self) -> Dict[str, Any]:
        """Return current orchestrator state.

        Returns:
            State dictionary
        """
        ...
