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

"""Sub-agent base classes for hierarchical task delegation.

This module provides the core infrastructure for sub-agents - specialized,
isolated instances of AgentOrchestrator with constrained scopes. Sub-agents
enable hierarchical task delegation and parallel execution.

Design Principles:
- Context Isolation: Each sub-agent has independent message history
- Resource Constraints: Limited tool access and budget
- Role Specialization: Specific roles (researcher, planner, executor, etc.)
- Backward Compatible: Existing code works without sub-agents

Example Usage:
    from victor.agent.subagents.base import SubAgent, SubAgentRole, SubAgentConfig

    # Create sub-agent configuration
    config = SubAgentConfig(
        role=SubAgentRole.RESEARCHER,
        task="Research authentication patterns in codebase",
        allowed_tools=["read", "search", "code_search"],
        tool_budget=15,
        context_limit=50000,
    )

    # Create and execute sub-agent
    subagent = SubAgent(config, parent_orchestrator)
    result = await subagent.execute()

    print(f"Success: {result.success}")
    print(f"Summary: {result.summary}")
    print(f"Tool calls used: {result.tool_calls_used}")
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Union

from victor.agent.subagents.protocols import SubAgentContext, SubAgentContextAdapter

if TYPE_CHECKING:
    from victor.agent.presentation import PresentationProtocol

# Import from canonical location to avoid circular dependencies
from victor.protocols.team import IAgent
from victor.teams.types import AgentMessage

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.core.container import ServiceContainer
    from victor.providers.base import StreamChunk

logger = logging.getLogger(__name__)


class SubAgentRole(Enum):
    """Role specialization for sub-agents.

    Each role has specific capabilities and constraints:

    - RESEARCHER: Read-only exploration (read, search, code_search, web_search)
    - PLANNER: Task breakdown and planning (read, ls, search, plan_files)
    - EXECUTOR: Code changes and execution (read, write, edit, shell, test, git)
    - REVIEWER: Quality checks and testing (read, search, test, git_diff, shell)
    - TESTER: Test writing and running (read, write to tests/, test, shell)
    """

    RESEARCHER = "researcher"
    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    TESTER = "tester"


@dataclass
class SubAgentConfig:
    """Configuration for a sub-agent instance.

    Defines the constraints and capabilities for a sub-agent. All fields
    except role and task have sensible defaults based on the role.

    Attributes:
        role: Sub-agent role specialization
        task: Task description for the sub-agent to execute
        allowed_tools: List of tool names the sub-agent can use
        tool_budget: Maximum number of tool calls allowed
        context_limit: Maximum context size in characters
        can_spawn_subagents: Whether this sub-agent can spawn child sub-agents
        working_directory: Optional working directory override
        timeout_seconds: Maximum execution time in seconds
        system_prompt_override: Optional custom system prompt (overrides role default)
        disable_embeddings: Disable codebase embeddings for this sub-agent (workflow service mode)
    """

    role: SubAgentRole
    task: str
    allowed_tools: List[str]
    tool_budget: int
    context_limit: int
    can_spawn_subagents: bool = False
    working_directory: Optional[str] = None
    timeout_seconds: int = 300
    system_prompt_override: Optional[str] = None
    disable_embeddings: bool = False


@dataclass
class SubAgentResult:
    """Result from sub-agent execution.

    Contains both the outcome (success/failure) and detailed metrics about
    the execution. The summary provides a concise overview while details
    contain the full response and tool calls.

    Attributes:
        success: Whether the sub-agent completed successfully
        summary: Brief summary of the result (first 500 chars)
        details: Full details including response and tool calls
        tool_calls_used: Number of tool calls made
        context_size: Final context size in characters
        duration_seconds: Execution time in seconds
        error: Error message if execution failed
    """

    success: bool
    summary: str
    details: Dict[str, Any]
    tool_calls_used: int
    context_size: int
    duration_seconds: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "success": self.success,
            "summary": self.summary,
            "details": self.details,
            "tool_calls_used": self.tool_calls_used,
            "context_size": self.context_size,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


class SubAgent(IAgent):
    """Represents a spawned sub-agent instance.

    A sub-agent is a wrapper around AgentOrchestrator with:
    - Constrained tool access (only allowed_tools)
    - Independent context (isolated message history)
    - Limited budget (max tool calls)
    - Role-specific system prompt

    Sub-agents execute a single task and return a structured result.
    They cannot interact directly with the user - all results are
    returned to the parent orchestrator.

    Attributes:
        config: Configuration defining constraints and capabilities
        parent: Parent orchestrator that spawned this sub-agent
        orchestrator: Constrained orchestrator instance for execution

    Example:
        config = SubAgentConfig(
            role=SubAgentRole.PLANNER,
            task="Create implementation plan for JWT auth",
            allowed_tools=["read", "ls", "search", "plan_files"],
            tool_budget=10,
            context_limit=30000,
        )

        subagent = SubAgent(config, parent_orchestrator)
        result = await subagent.execute()
    """

    def __init__(
        self,
        config: SubAgentConfig,
        parent: Union["AgentOrchestrator", SubAgentContext],
        presentation: Optional["PresentationProtocol"] = None,
    ):
        """Initialize sub-agent with configuration and parent context.

        Args:
            config: Sub-agent configuration
            parent: Parent context providing settings, provider, model, and tools.
                Can be either a full AgentOrchestrator (for backward compatibility)
                or any object implementing the SubAgentContext protocol.
            presentation: Optional presentation adapter for icons (creates default if None)

        Note:
            If a full AgentOrchestrator is passed, it will be automatically
            adapted to SubAgentContext for ISP compliance. This maintains
            backward compatibility with existing code while enabling cleaner
            testing through protocol-based dependency injection.
        """
        self.config = config
        self._id = uuid.uuid4().hex[:12]

        # Lazy init for backward compatibility
        if presentation is None:
            from victor.agent.presentation import create_presentation_adapter

            self._presentation = create_presentation_adapter()
        else:
            self._presentation = presentation

        # Auto-adapt full orchestrator to SubAgentContext for ISP compliance
        # Check if it's already a SubAgentContext (including adapters)
        if isinstance(parent, SubAgentContext):
            self._context: SubAgentContext = parent
        else:
            # Wrap the orchestrator with the adapter
            self._context = SubAgentContextAdapter(parent)

        # Keep reference to parent for backward compatibility
        # (some code may access self.parent directly)
        self.parent = parent
        self.orchestrator: Optional["AgentOrchestrator"] = None

        logger.info(
            f"Created {config.role.value} sub-agent: {config.task[:50]}... "
            f"(budget={config.tool_budget}, tools={len(config.allowed_tools)})"
        )

    @property
    def id(self) -> str:
        """Unique identifier for this agent."""
        return self._id

    @property
    def role(self):
        """Role of this agent (SubAgentRole)."""
        return self.config.role

    @property
    def persona(self):
        """Persona of this agent (None for SubAgent)."""
        return None

    async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
        """Execute a task using this sub-agent.

        Note: The SubAgent is configured with a specific task at creation.
        This method executes that configured task, ignoring the passed task
        parameter. The context parameter is also ignored as SubAgent has
        its own isolated context.

        Returns:
            String summary of execution outcome.
        """
        # Execute the configured task (ignore passed task parameter)
        result = await self.execute()
        return result.summary if result.summary else ""

    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Receive a message (SubAgents don't support direct messaging).

        SubAgents are isolated execution units that don't participate in
        direct agent-to-agent communication. This method exists for protocol
        compatibility.

        Returns:
            None, as SubAgents don't respond to messages
        """
        return None

    def _create_constrained_orchestrator(self) -> "AgentOrchestrator":
        """Create orchestrator with role-specific constraints.

        Creates a new orchestrator instance with:
        - Limited tool access (only allowed_tools)
        - Constrained budget and context
        - Role-specific system prompt
        - Shared provider and settings from parent context
        - disable_embeddings flag for workflow service mode

        Returns:
            Constrained orchestrator instance
        """
        from copy import deepcopy
        from victor.agent.orchestrator import AgentOrchestrator

        # Copy settings from parent context but apply constraints
        settings = deepcopy(self._context.settings)
        settings.tool_budget = self.config.tool_budget
        settings.max_context_chars = self.config.context_limit

        # Create new orchestrator instance with same provider
        # Use the actual provider object (not just the name) for proper initialization
        orchestrator = AgentOrchestrator(
            settings=settings,
            provider=self._context.provider,
            model=self._context.model,
            temperature=self._context.temperature,
            provider_name=self._context.provider_name,
            # Note: We'll share the parent's DI container for now
            # In production, we might want isolated scoped containers
        )

        # Set disable_embeddings flag for workflow service mode
        if self.config.disable_embeddings:
            gear_icon = self._presentation.icon("gear", with_color=False)
            logger.debug(f"   {gear_icon}  Setting disable_embeddings=True for {self.config.role.value} sub-agent")
            if hasattr(orchestrator, '_session_state_manager'):
                orchestrator._session_state_manager.execution_state.disable_embeddings = True

        # Set role-specific system prompt
        system_prompt = self._get_role_prompt()
        orchestrator.set_system_prompt(system_prompt)

        # Register only allowed tools
        self._configure_allowed_tools(orchestrator)

        logger.debug(
            f"Created constrained orchestrator for {self.config.role.value}: "
            f"{len(self.config.allowed_tools)} tools, budget={self.config.tool_budget}"
        )

        return orchestrator

    def _configure_allowed_tools(self, orchestrator: "AgentOrchestrator") -> None:
        """Configure orchestrator with only allowed tools.

        Args:
            orchestrator: Orchestrator to configure
        """
        # Clear existing tool registrations
        orchestrator.tool_registry.clear()

        # Register only allowed tools from parent context
        for tool_name in self.config.allowed_tools:
            tool = self._context.tool_registry.get(tool_name)
            if tool:
                orchestrator.tool_registry.register(tool)
            else:
                logger.warning(
                    f"Allowed tool '{tool_name}' not found in parent registry "
                    f"for {self.config.role.value} sub-agent"
                )

    def _get_role_prompt(self) -> str:
        """Get system prompt for this role.

        If a custom system_prompt_override is provided in config, use that.
        Otherwise, use the role-specific default prompt.

        Returns:
            System prompt string
        """
        if self.config.system_prompt_override:
            return self.config.system_prompt_override

        # Import here to avoid circular dependency
        from victor.agent.subagents.prompts import get_role_prompt

        return get_role_prompt(self.config.role)

    async def _execute_with_retry(self) -> Any:
        """Execute orchestrator.chat() with exponential backoff retry.

        Implements retry logic with exponential backoff for handling transient errors
        like rate limits, timeouts, and connection issues.

        Returns:
            Response from orchestrator.chat()

        Raises:
            Exception: If all retries are exhausted
        """
        from victor.core.errors import (
            ProviderConnectionError,
            ProviderError,
            ProviderRateLimitError,
            ProviderTimeoutError,
        )

        max_attempts = 3
        base_delay = 1.0  # Start with 1 second
        max_delay = 60.0  # Cap at 60 seconds

        last_exception = None

        refresh_icon = self._presentation.icon("refresh", with_color=False)
        success_icon = self._presentation.icon("success", with_color=False)
        error_icon = self._presentation.icon("error", with_color=False)

        logger.debug(f"   {refresh_icon} Starting execution with retry logic (max {max_attempts} attempts)")

        for attempt in range(1, max_attempts + 1):
            try:
                if attempt > 1:
                    logger.debug(f"   {refresh_icon} Retry attempt {attempt}/{max_attempts}...")

                # Try to execute the chat
                response = await self.orchestrator.chat(self.config.task)

                if attempt > 1:
                    logger.debug(f"   {success_icon} Retry attempt {attempt}/{max_attempts} succeeded!")

                return response

            except (
                ProviderRateLimitError,
                ProviderTimeoutError,
                ProviderConnectionError,
                ProviderError,  # Generic provider errors (e.g., server disconnects)
                ConnectionError,
                TimeoutError,
                OSError,  # Network-level errors
            ) as e:
                last_exception = e

                if attempt >= max_attempts:
                    # Final attempt failed, will raise after loop
                    logger.warning(
                        f"Sub-agent {self.config.role.value}: All {max_attempts} attempts failed. Last error: {type(e).__name__}: {e}"
                    )
                    break

                # Calculate exponential backoff delay: 2^(attempt-1) * base_delay
                # This is similar to Fibonacci: 1, 2, 4, 8, 16, 32...
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)

                logger.info(
                    f"Sub-agent {self.config.role.value}: Attempt {attempt}/{max_attempts} failed with "
                    f"{type(e).__name__}. Retrying in {delay:.1f}s..."
                )

                await asyncio.sleep(delay)

            except Exception as e:
                # Non-retriable error, raise immediately
                logger.error(f"Sub-agent {self.config.role.value}: Non-retriable error: {type(e).__name__}: {e}")
                raise

        # All retries exhausted
        logger.error(f"   {error_icon} All retry attempts exhausted for {self.config.role.value}")
        raise last_exception

    async def execute(self) -> SubAgentResult:
        """Execute the sub-agent task.

        Runs the task in the constrained orchestrator and returns a
        structured result. Handles exceptions gracefully and always
        returns a SubAgentResult (never raises).

        Returns:
            SubAgentResult with execution outcome and metrics
        """
        start_time = time.time()

        try:
            # Create constrained orchestrator lazily
            if self.orchestrator is None:
                self.orchestrator = self._create_constrained_orchestrator()

            logger.info(f"Executing {self.config.role.value} sub-agent: {self.config.task[:50]}...")

            # Run the task with retry on rate limits
            response = await self._execute_with_retry()

            # Extract metrics
            tool_calls_used = getattr(self.orchestrator, "tool_calls_used", 0)
            context_size = len(str(self.orchestrator.get_messages()))

            # Create success result
            result = SubAgentResult(
                success=True,
                summary=response.content[:500] if response.content else "",
                details={
                    "full_response": response.content,
                    "tool_calls": getattr(response, "tool_calls", []) or [],
                    "role": self.config.role.value,
                },
                tool_calls_used=tool_calls_used,
                context_size=context_size,
                duration_seconds=time.time() - start_time,
            )

            logger.info(
                f"{self.config.role.value} sub-agent completed: "
                f"{tool_calls_used}/{self.config.tool_budget} tool calls, "
                f"{result.duration_seconds:.1f}s"
            )

            return result

        except Exception as e:
            # Create error result
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(
                f"{self.config.role.value} sub-agent failed: {error_msg}",
                exc_info=True,
            )

            tool_calls_used = 0
            context_size = 0
            if self.orchestrator:
                tool_calls_used = getattr(self.orchestrator, "tool_calls_used", 0)
                try:
                    context_size = len(str(self.orchestrator.get_messages()))
                except Exception:
                    pass

            return SubAgentResult(
                success=False,
                summary=f"Sub-agent failed: {error_msg[:450]}",
                details={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "role": self.config.role.value,
                },
                tool_calls_used=tool_calls_used,
                context_size=context_size,
                duration_seconds=time.time() - start_time,
                error=error_msg,
            )

    async def stream_execute(self) -> AsyncIterator["StreamChunk"]:
        """Execute sub-agent task with streaming output.

        Like execute() but yields chunks as content is generated.
        Uses orchestrator.stream_chat() instead of .chat().

        Yields:
            StreamChunk with incremental content and tool calls

        Example:
            subagent = SubAgent(config, parent_orchestrator)
            async for chunk in subagent.stream_execute():
                print(chunk.content, end="", flush=True)
                if chunk.is_final:
                    print(f"\\nCompleted: {chunk.metadata}")
        """
        from victor.providers.base import StreamChunk

        start_time = time.time()

        try:
            # Create constrained orchestrator lazily
            if self.orchestrator is None:
                self.orchestrator = self._create_constrained_orchestrator()

            logger.info(
                f"Stream executing {self.config.role.value} sub-agent: "
                f"{self.config.task[:50]}..."
            )

            # Stream the task using orchestrator.stream_chat()
            async for chunk in self.orchestrator.stream_chat(self.config.task):
                yield chunk

                # If this was the final chunk from the orchestrator, we'll add metadata
                if chunk.is_final:
                    # Extract metrics
                    tool_calls_used = getattr(self.orchestrator, "tool_calls_used", 0)
                    context_size = len(str(self.orchestrator.get_messages()))
                    duration = time.time() - start_time

                    # Enhance the final chunk with execution metadata
                    enhanced_metadata = chunk.metadata.copy() if chunk.metadata else {}
                    enhanced_metadata.update(
                        {
                            "tool_calls_used": tool_calls_used,
                            "context_size": context_size,
                            "duration_seconds": duration,
                            "role": self.config.role.value,
                            "success": True,
                        }
                    )

                    # Yield a final chunk with metadata (if not already included)
                    yield StreamChunk(
                        content="",
                        is_final=True,
                        metadata=enhanced_metadata,
                    )

                    logger.info(
                        f"{self.config.role.value} sub-agent stream completed: "
                        f"{tool_calls_used}/{self.config.tool_budget} tool calls, "
                        f"{duration:.1f}s"
                    )
                    return

        except Exception as e:
            # Create error chunk
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(
                f"{self.config.role.value} sub-agent stream failed: {error_msg}",
                exc_info=True,
            )

            tool_calls_used = 0
            context_size = 0
            if self.orchestrator:
                tool_calls_used = getattr(self.orchestrator, "tool_calls_used", 0)
                try:
                    context_size = len(str(self.orchestrator.get_messages()))
                except Exception:
                    pass

            # Yield error chunk with is_final=True
            yield StreamChunk(
                content="",
                is_final=True,
                metadata={
                    "error": error_msg,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "tool_calls_used": tool_calls_used,
                    "context_size": context_size,
                    "duration_seconds": time.time() - start_time,
                    "role": self.config.role.value,
                    "success": False,
                },
            )


__all__ = [
    "SubAgentRole",
    "SubAgentConfig",
    "SubAgentResult",
    "SubAgent",
]
