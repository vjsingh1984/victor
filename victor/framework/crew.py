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

"""Crew - CrewAI-compatible multi-agent orchestration.

This module provides CrewAI-compatible abstractions for building agent crews
with rich personas, task dependencies, and delegation capabilities.

Design Principles (SOLID):
    - Single Responsibility: Agent handles persona, Task handles work item
    - Open/Closed: Extensible via protocols and callbacks
    - Liskov Substitution: All agents implement CrewAgentProtocol
    - Interface Segregation: Small, focused protocols
    - Dependency Inversion: Depends on protocols, not concrete teams

Key Concepts:
    - CrewAgent: Agent with role, goal, backstory, and tools
    - CrewTask: Work item with description, expected output, and dependencies
    - Crew: Orchestrates agents executing tasks
    - Process: Sequential or Hierarchical execution

Example:
    from victor.framework.crew import Crew, CrewAgent, CrewTask, Process

    # Define agents with rich personas
    researcher = CrewAgent(
        role="Senior Research Analyst",
        goal="Find and analyze technical patterns",
        backstory="You are an expert at finding patterns in codebases...",
        tools=["code_search", "semantic_code_search"],
        allow_delegation=True,
    )

    developer = CrewAgent(
        role="Senior Software Developer",
        goal="Implement high-quality code",
        backstory="You write clean, maintainable code following best practices...",
        tools=["write_file", "edit_file", "run_bash"],
    )

    # Define tasks with dependencies
    research_task = CrewTask(
        description="Research authentication patterns in the codebase",
        agent=researcher,
        expected_output="List of auth patterns with file locations",
    )

    impl_task = CrewTask(
        description="Implement OAuth2 authentication",
        agent=developer,
        expected_output="Working OAuth2 implementation",
        dependencies=[research_task],
    )

    # Create and run crew
    crew = Crew(
        agents=[researcher, developer],
        tasks=[research_task, impl_task],
        process=Process.SEQUENTIAL,
    )

    result = await crew.kickoff()
    print(result.output)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Type,
    Union,
    runtime_checkable,
)

from victor.framework.teams import (
    AgentTeam,
    MemberResult,
    TeamConfig,
    TeamEvent,
    TeamEventType,
    TeamFormation,
    TeamMember,
    TeamMemberSpec,
    TeamResult,
)

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.framework.agent import Agent

logger = logging.getLogger(__name__)


class Process(str, Enum):
    """Execution process type for crews.

    Similar to CrewAI's process types.
    """

    SEQUENTIAL = "sequential"
    """Tasks execute one after another in order."""

    HIERARCHICAL = "hierarchical"
    """Manager agent delegates tasks to workers."""

    PARALLEL = "parallel"
    """Independent tasks execute in parallel."""


class VerbosityLevel(int, Enum):
    """Verbosity level for crew output."""

    QUIET = 0
    """Minimal output."""

    NORMAL = 1
    """Standard progress information."""

    VERBOSE = 2
    """Detailed execution logs."""


@runtime_checkable
class CrewAgentProtocol(Protocol):
    """Protocol for crew agents."""

    @property
    def role(self) -> str:
        """Agent's role in the crew."""
        ...

    @property
    def goal(self) -> str:
        """Agent's goal/objective."""
        ...

    @property
    def backstory(self) -> str:
        """Agent's background story."""
        ...


@dataclass
class CrewAgent:
    """An agent with rich persona for crew execution.

    CrewAgent represents an AI agent with a defined role, goal, backstory,
    and capabilities. Similar to CrewAI's Agent class.

    Attributes:
        role: The agent's role (e.g., "Senior Research Analyst")
        goal: What the agent is trying to achieve
        backstory: Background story defining the agent's personality
        tools: List of tool names the agent can use
        allow_delegation: Whether agent can delegate to others
        verbose: Whether to show detailed output
        cache: Whether to cache tool results
        memory: Whether to use long-term memory
        max_iterations: Maximum iterations per task
        max_tool_calls: Maximum tool calls per task

    Example:
        researcher = CrewAgent(
            role="Senior Research Analyst",
            goal="Find and analyze code patterns",
            backstory="You are an expert at understanding complex codebases. "
                      "You have years of experience identifying patterns...",
            tools=["code_search", "semantic_code_search", "read_file"],
            allow_delegation=True,
        )
    """

    role: str
    goal: str
    backstory: str = ""
    tools: List[str] = field(default_factory=list)
    allow_delegation: bool = False
    verbose: bool = True
    cache: bool = True
    memory: bool = False
    max_iterations: int = 25
    max_tool_calls: int = 25
    _id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def id(self) -> str:
        """Get agent ID."""
        return self._id

    def to_system_prompt(self) -> str:
        """Generate system prompt from agent persona.

        Returns:
            System prompt string incorporating role, goal, and backstory
        """
        lines = [
            f"# Role: {self.role}",
            "",
            f"## Goal",
            self.goal,
            "",
        ]

        if self.backstory:
            lines.extend([
                "## Background",
                self.backstory,
                "",
            ])

        if self.allow_delegation:
            lines.extend([
                "## Delegation",
                "You can delegate tasks to other team members when appropriate.",
                "",
            ])

        return "\n".join(lines)

    def to_team_member_spec(self, priority: int = 0) -> TeamMemberSpec:
        """Convert to TeamMemberSpec for team execution.

        Args:
            priority: Execution priority (lower = earlier)

        Returns:
            TeamMemberSpec for team infrastructure
        """
        # Map role to base role type
        role_lower = self.role.lower()
        if any(kw in role_lower for kw in ["research", "analyst", "analyst"]):
            base_role = "researcher"
        elif any(kw in role_lower for kw in ["develop", "engineer", "code", "impl"]):
            base_role = "executor"
        elif any(kw in role_lower for kw in ["review", "qa", "test", "critic"]):
            base_role = "reviewer"
        elif any(kw in role_lower for kw in ["plan", "architect", "design"]):
            base_role = "planner"
        else:
            base_role = "executor"

        return TeamMemberSpec(
            role=base_role,
            goal=f"{self.goal}\n\n{self.to_system_prompt()}",
            name=self.role,
            tool_budget=self.max_tool_calls,
            priority=priority,
        )


@dataclass
class CrewTask:
    """A task to be executed by a crew agent.

    CrewTask represents a unit of work with a description, expected output,
    and optional dependencies on other tasks.

    Attributes:
        description: What needs to be done
        agent: The agent assigned to this task
        expected_output: Description of expected result
        dependencies: Tasks that must complete before this one
        context: Additional context from other tasks
        tools: Task-specific tools (overrides agent tools)
        output_file: Optional file to write output to
        async_execution: Whether to execute asynchronously
        callback: Function called on completion

    Example:
        research_task = CrewTask(
            description="Research authentication patterns in the codebase",
            agent=researcher,
            expected_output="A detailed list of authentication patterns "
                          "with file locations and code snippets",
        )

        impl_task = CrewTask(
            description="Implement OAuth2 authentication",
            agent=developer,
            expected_output="Working OAuth2 implementation with tests",
            dependencies=[research_task],
        )
    """

    description: str
    agent: CrewAgent
    expected_output: str = ""
    dependencies: List["CrewTask"] = field(default_factory=list)
    context: Optional[str] = None
    tools: Optional[List[str]] = None
    output_file: Optional[str] = None
    async_execution: bool = False
    callback: Optional[Callable[[Any], None]] = None
    _id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    _output: Optional[str] = None

    @property
    def id(self) -> str:
        """Get task ID."""
        return self._id

    @property
    def output(self) -> Optional[str]:
        """Get task output (after execution)."""
        return self._output

    def build_context(self) -> str:
        """Build context string from dependencies.

        Returns:
            Combined context from dependent tasks
        """
        parts = []

        if self.context:
            parts.append(f"## Additional Context\n{self.context}")

        for dep in self.dependencies:
            if dep.output:
                parts.append(
                    f"## From {dep.agent.role}\n"
                    f"Task: {dep.description[:100]}...\n"
                    f"Result: {dep.output}"
                )

        return "\n\n".join(parts)

    def build_full_description(self) -> str:
        """Build full task description with context.

        Returns:
            Complete task description
        """
        lines = [
            "# Task",
            self.description,
            "",
        ]

        if self.expected_output:
            lines.extend([
                "## Expected Output",
                self.expected_output,
                "",
            ])

        context = self.build_context()
        if context:
            lines.append(context)

        return "\n".join(lines)


@dataclass
class CrewOutput:
    """Output from crew execution.

    Attributes:
        output: Final combined output
        task_outputs: Output from each task
        success: Whether all tasks succeeded
        duration: Total execution time
        token_usage: Token usage statistics
    """

    output: str
    task_outputs: Dict[str, str]
    success: bool
    duration: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output": self.output,
            "task_outputs": self.task_outputs,
            "success": self.success,
            "duration": self.duration,
            "token_usage": self.token_usage,
        }


class Crew:
    """Orchestrates a crew of agents executing tasks.

    Crew provides a CrewAI-compatible interface for multi-agent orchestration,
    leveraging Victor's teams infrastructure while providing familiar abstractions.

    Attributes:
        agents: List of crew agents
        tasks: List of tasks to execute
        process: Execution process type
        verbose: Verbosity level
        manager_agent: Optional manager for hierarchical process

    Example:
        crew = Crew(
            agents=[researcher, developer, reviewer],
            tasks=[research_task, impl_task, review_task],
            process=Process.SEQUENTIAL,
            verbose=True,
        )

        result = await crew.kickoff()
        print(result.output)
    """

    def __init__(
        self,
        agents: List[CrewAgent],
        tasks: List[CrewTask],
        process: Process = Process.SEQUENTIAL,
        verbose: Union[bool, VerbosityLevel] = True,
        manager_agent: Optional[CrewAgent] = None,
        max_rpm: Optional[int] = None,
        memory: bool = False,
        share_crew_context: bool = True,
        function_calling_llm: Optional[str] = None,
        cache: bool = True,
    ):
        """Initialize a crew.

        Args:
            agents: List of agents in the crew
            tasks: List of tasks to execute
            process: Execution process (sequential, hierarchical, parallel)
            verbose: Whether to show detailed output
            manager_agent: Manager agent for hierarchical process
            max_rpm: Maximum requests per minute (rate limiting)
            memory: Whether to use long-term memory
            share_crew_context: Whether agents share context
            function_calling_llm: Preferred LLM for function calling
            cache: Whether to cache tool results
        """
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self.verbose = VerbosityLevel.VERBOSE if verbose else VerbosityLevel.QUIET
        self.manager_agent = manager_agent
        self.max_rpm = max_rpm
        self.memory = memory
        self.share_crew_context = share_crew_context
        self.function_calling_llm = function_calling_llm
        self.cache = cache

        self._orchestrator: Optional["AgentOrchestrator"] = None
        self._team: Optional[AgentTeam] = None

    def _validate(self) -> List[str]:
        """Validate crew configuration.

        Returns:
            List of validation errors
        """
        errors = []

        if not self.agents:
            errors.append("Crew must have at least one agent")

        if not self.tasks:
            errors.append("Crew must have at least one task")

        # Check all task agents are in the crew
        agent_ids = {a.id for a in self.agents}
        for task in self.tasks:
            if task.agent.id not in agent_ids:
                errors.append(
                    f"Task '{task.description[:30]}...' assigned to "
                    f"agent '{task.agent.role}' not in crew"
                )

        # Check for circular dependencies
        if self._has_circular_deps():
            errors.append("Task dependencies contain a cycle")

        # Hierarchical requires manager
        if self.process == Process.HIERARCHICAL and not self.manager_agent:
            errors.append("Hierarchical process requires a manager_agent")

        return errors

    def _has_circular_deps(self) -> bool:
        """Check for circular task dependencies."""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def has_cycle(task: CrewTask) -> bool:
            visited.add(task.id)
            rec_stack.add(task.id)

            for dep in task.dependencies:
                if dep.id not in visited:
                    if has_cycle(dep):
                        return True
                elif dep.id in rec_stack:
                    return True

            rec_stack.remove(task.id)
            return False

        for task in self.tasks:
            if task.id not in visited:
                if has_cycle(task):
                    return True

        return False

    def _get_execution_order(self) -> List[CrewTask]:
        """Get topological order for task execution.

        Returns:
            Tasks in dependency-respecting order
        """
        # Build dependency graph
        in_degree: Dict[str, int] = {t.id: 0 for t in self.tasks}
        graph: Dict[str, List[str]] = {t.id: [] for t in self.tasks}
        task_map = {t.id: t for t in self.tasks}

        for task in self.tasks:
            for dep in task.dependencies:
                graph[dep.id].append(task.id)
                in_degree[task.id] += 1

        # Topological sort (Kahn's algorithm)
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(task_map[current])

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    async def kickoff(
        self,
        orchestrator: Optional["AgentOrchestrator"] = None,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> CrewOutput:
        """Start crew execution.

        Args:
            orchestrator: AgentOrchestrator instance (required)
            inputs: Optional input variables for tasks

        Returns:
            CrewOutput with results
        """
        # Validate
        errors = self._validate()
        if errors:
            raise ValueError(f"Invalid crew: {'; '.join(errors)}")

        if orchestrator:
            self._orchestrator = orchestrator
        if not self._orchestrator:
            raise ValueError("Orchestrator required for crew execution")

        start_time = time.time()
        task_outputs: Dict[str, str] = {}

        # Get execution order
        ordered_tasks = self._get_execution_order()

        if self.verbose >= VerbosityLevel.NORMAL:
            logger.info(
                f"Starting crew with {len(self.agents)} agents, "
                f"{len(self.tasks)} tasks ({self.process.value} process)"
            )

        # Execute based on process type
        if self.process == Process.SEQUENTIAL:
            success = await self._execute_sequential(ordered_tasks, task_outputs, inputs)
        elif self.process == Process.PARALLEL:
            success = await self._execute_parallel(ordered_tasks, task_outputs, inputs)
        elif self.process == Process.HIERARCHICAL:
            success = await self._execute_hierarchical(ordered_tasks, task_outputs, inputs)
        else:
            success = await self._execute_sequential(ordered_tasks, task_outputs, inputs)

        # Combine outputs
        final_output = "\n\n---\n\n".join(
            f"## {task.agent.role}\n{task.output or 'No output'}"
            for task in ordered_tasks
            if task.output
        )

        # Emit RL event
        self._emit_crew_completed_event(
            success=success,
            duration=time.time() - start_time,
            task_count=len(self.tasks),
        )

        return CrewOutput(
            output=final_output,
            task_outputs=task_outputs,
            success=success,
            duration=time.time() - start_time,
        )

    async def _execute_sequential(
        self,
        tasks: List[CrewTask],
        outputs: Dict[str, str],
        inputs: Optional[Dict[str, Any]],
    ) -> bool:
        """Execute tasks sequentially.

        Args:
            tasks: Tasks in execution order
            outputs: Dict to store outputs
            inputs: Input variables

        Returns:
            Success status
        """
        for task in tasks:
            if self.verbose >= VerbosityLevel.VERBOSE:
                logger.info(f"Executing task: {task.description[:50]}...")

            result = await self._execute_task(task, inputs)
            task._output = result
            outputs[task.id] = result

            if task.callback:
                task.callback(result)

        return True

    async def _execute_parallel(
        self,
        tasks: List[CrewTask],
        outputs: Dict[str, str],
        inputs: Optional[Dict[str, Any]],
    ) -> bool:
        """Execute independent tasks in parallel.

        Args:
            tasks: Tasks in execution order
            outputs: Dict to store outputs
            inputs: Input variables

        Returns:
            Success status
        """
        # Group tasks by dependency level
        levels: List[List[CrewTask]] = []
        remaining = set(t.id for t in tasks)
        task_map = {t.id: t for t in tasks}
        completed: Set[str] = set()

        while remaining:
            # Find tasks with all deps satisfied
            ready = [
                task_map[tid]
                for tid in remaining
                if all(d.id in completed for d in task_map[tid].dependencies)
            ]

            if not ready:
                # Deadlock - execute remaining sequentially
                for tid in remaining:
                    ready.append(task_map[tid])
                    break

            levels.append(ready)
            for task in ready:
                remaining.remove(task.id)
                completed.add(task.id)

        # Execute each level in parallel
        for level in levels:
            async def execute_task_wrapper(task: CrewTask) -> tuple:
                result = await self._execute_task(task, inputs)
                return task.id, result

            results = await asyncio.gather(
                *[execute_task_wrapper(t) for t in level],
                return_exceptions=True,
            )

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed: {result}")
                    continue
                tid, output = result
                task_map[tid]._output = output
                outputs[tid] = output

        return True

    async def _execute_hierarchical(
        self,
        tasks: List[CrewTask],
        outputs: Dict[str, str],
        inputs: Optional[Dict[str, Any]],
    ) -> bool:
        """Execute tasks with manager delegation.

        Args:
            tasks: Tasks in execution order
            outputs: Dict to store outputs
            inputs: Input variables

        Returns:
            Success status
        """
        if not self.manager_agent:
            return await self._execute_sequential(tasks, outputs, inputs)

        # Create team with manager
        team = await AgentTeam.create(
            orchestrator=self._orchestrator,
            name="Hierarchical Crew",
            goal=f"Execute {len(tasks)} tasks with manager coordination",
            members=[
                self.manager_agent.to_team_member_spec(priority=0),
                *[a.to_team_member_spec(priority=i + 1) for i, a in enumerate(self.agents)],
            ],
            formation=TeamFormation.HIERARCHICAL,
        )

        # Update manager spec to be manager
        team._config.members[0] = TeamMember(
            id=team._config.members[0].id,
            role=team._config.members[0].role,
            name=team._config.members[0].name,
            goal=team._config.members[0].goal,
            tool_budget=team._config.members[0].tool_budget,
            is_manager=True,
            priority=0,
        )

        result = await team.run()

        if result.success:
            # Extract outputs from member results
            for mid, mresult in result.member_results.items():
                outputs[mid] = mresult.summary or ""

        return result.success

    async def _execute_task(
        self,
        task: CrewTask,
        inputs: Optional[Dict[str, Any]],
    ) -> str:
        """Execute a single task.

        Args:
            task: Task to execute
            inputs: Input variables

        Returns:
            Task output string
        """
        # Build full description with context
        description = task.build_full_description()

        # Substitute input variables
        if inputs:
            for key, value in inputs.items():
                description = description.replace(f"{{{key}}}", str(value))

        # Create team with single member
        team = await AgentTeam.create(
            orchestrator=self._orchestrator,
            name=f"Task: {task.description[:30]}...",
            goal=description,
            members=[task.agent.to_team_member_spec()],
            formation=TeamFormation.SEQUENTIAL,
            total_tool_budget=task.agent.max_tool_calls,
        )

        result = await team.run()

        if result.success and result.member_results:
            # Get first member result
            member_result = next(iter(result.member_results.values()))
            return member_result.summary or ""

        return ""

    def _emit_crew_completed_event(
        self,
        success: bool,
        duration: float,
        task_count: int,
    ) -> None:
        """Emit RL event for crew completion."""
        try:
            from victor.agent.rl.hooks import get_rl_hooks, RLEvent, RLEventType

            hooks = get_rl_hooks()
            if hooks is None:
                return

            quality = 0.8 if success else 0.2
            if success and duration < 60:
                quality += 0.1
            if success and task_count > 3:
                quality += 0.05

            event = RLEvent(
                type=RLEventType.TEAM_COMPLETED,
                team_id=f"crew_{id(self)}",
                team_formation=self.process.value,
                success=success,
                quality_score=min(1.0, quality),
                metadata={
                    "duration_seconds": duration,
                    "task_count": task_count,
                    "agent_count": len(self.agents),
                    "process": self.process.value,
                },
            )
            hooks.emit(event)

        except Exception as e:
            logger.debug(f"Crew event emission failed: {e}")

    async def kickoff_async(
        self,
        orchestrator: Optional["AgentOrchestrator"] = None,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> CrewOutput:
        """Alias for kickoff() for CrewAI compatibility."""
        return await self.kickoff(orchestrator, inputs)

    def __repr__(self) -> str:
        return (
            f"Crew(agents={len(self.agents)}, tasks={len(self.tasks)}, "
            f"process={self.process.value})"
        )


# Convenience factory functions
def create_agent(
    role: str,
    goal: str,
    backstory: str = "",
    **kwargs: Any,
) -> CrewAgent:
    """Create a new crew agent.

    Args:
        role: Agent's role
        goal: Agent's goal
        backstory: Agent's background story
        **kwargs: Additional agent attributes

    Returns:
        CrewAgent instance
    """
    return CrewAgent(role=role, goal=goal, backstory=backstory, **kwargs)


def create_task(
    description: str,
    agent: CrewAgent,
    expected_output: str = "",
    **kwargs: Any,
) -> CrewTask:
    """Create a new crew task.

    Args:
        description: Task description
        agent: Agent to execute the task
        expected_output: Expected output description
        **kwargs: Additional task attributes

    Returns:
        CrewTask instance
    """
    return CrewTask(
        description=description,
        agent=agent,
        expected_output=expected_output,
        **kwargs,
    )


__all__ = [
    # Core classes
    "Crew",
    "CrewAgent",
    "CrewTask",
    "CrewOutput",
    # Enums
    "Process",
    "VerbosityLevel",
    # Protocols
    "CrewAgentProtocol",
    # Factory functions
    "create_agent",
    "create_task",
]
