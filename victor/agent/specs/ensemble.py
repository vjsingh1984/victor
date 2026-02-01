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

"""Ensemble orchestration patterns.

Provides composable patterns for coordinating multiple agents:
- Pipeline: Sequential execution (A -> B -> C)
- Parallel: Concurrent execution (A | B | C)
- Hierarchical: Manager delegates to workers

Inspired by:
- Prefect/Airflow task DAGs
- Kubernetes Pod/Deployment patterns
- MapReduce paradigms
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from collections.abc import Callable

from victor.agent.specs.models import AgentSpec

logger = logging.getLogger(__name__)


class EnsembleType(str, Enum):
    """Types of ensemble coordination."""

    PIPELINE = "pipeline"  # Sequential: A -> B -> C
    PARALLEL = "parallel"  # Concurrent: A | B | C
    HIERARCHICAL = "hierarchical"  # Manager delegates to workers
    ROUTER = "router"  # Route to appropriate agent
    CONSENSUS = "consensus"  # Multiple agents vote/agree


class ExecutionStatus(str, Enum):
    """Status of ensemble execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class AgentResult:
    """Result from a single agent execution."""

    agent_name: str
    status: ExecutionStatus
    output: Any = None
    error: Optional[str] = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    tool_calls: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def success(self) -> bool:
        return self.status == ExecutionStatus.COMPLETED


@dataclass
class EnsembleResult:
    """Result from ensemble execution."""

    ensemble_type: EnsembleType
    status: ExecutionStatus
    agent_results: list[AgentResult] = field(default_factory=list)
    final_output: Any = None
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == ExecutionStatus.COMPLETED

    def aggregate_stats(self) -> None:
        """Aggregate stats from agent results."""
        self.total_tokens = sum(r.tokens_used for r in self.agent_results)
        self.total_cost_usd = sum(r.cost_usd for r in self.agent_results)
        if self.agent_results:
            self.total_duration_ms = max(r.duration_ms for r in self.agent_results)


class Ensemble(ABC):
    """Base class for ensemble patterns.

    Ensembles coordinate multiple agents to complete complex tasks.
    They handle:
    - Agent scheduling (sequential, parallel, conditional)
    - Context passing between agents
    - Error handling and recovery
    - Result aggregation
    """

    def __init__(
        self,
        agents: list[AgentSpec],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize ensemble.

        Args:
            agents: Agent specifications
            name: Ensemble name
            description: Ensemble description
        """
        self.agents = agents
        self.name = name or self.ensemble_type.value
        self.description = description or ""

    @property
    @abstractmethod
    def ensemble_type(self) -> EnsembleType:
        """Type of this ensemble."""
        ...

    @abstractmethod
    async def execute(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
        orchestrator: Optional[Any] = None,
    ) -> EnsembleResult:
        """Execute the ensemble on a task.

        Args:
            task: Task description
            context: Initial context
            orchestrator: AgentOrchestrator instance

        Returns:
            EnsembleResult with outputs from all agents
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize ensemble to dictionary."""
        return {
            "type": self.ensemble_type.value,
            "name": self.name,
            "description": self.description,
            "agents": [a.to_dict() for a in self.agents],
        }


class Pipeline(Ensemble):
    """Sequential pipeline execution.

    Executes agents in order, passing output from each
    to the next as context.

    Example:
        pipeline = Pipeline([researcher, coder, reviewer])
        # researcher -> coder -> reviewer
    """

    def __init__(
        self,
        agents: list[AgentSpec],
        name: Optional[str] = None,
        continue_on_error: bool = False,
    ):
        """Initialize pipeline.

        Args:
            agents: Agents to execute in sequence
            name: Pipeline name
            continue_on_error: Continue if an agent fails
        """
        super().__init__(agents, name)
        self.continue_on_error = continue_on_error

    @property
    def ensemble_type(self) -> EnsembleType:
        return EnsembleType.PIPELINE

    async def execute(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
        orchestrator: Optional[Any] = None,
    ) -> EnsembleResult:
        """Execute pipeline sequentially."""
        result = EnsembleResult(
            ensemble_type=self.ensemble_type,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )

        current_context = context or {}
        current_context["original_task"] = task

        for agent in self.agents:
            agent_result = await self._execute_agent(agent, task, current_context, orchestrator)
            result.agent_results.append(agent_result)

            if not agent_result.success:
                if not self.continue_on_error:
                    result.status = ExecutionStatus.FAILED
                    result.completed_at = datetime.now(timezone.utc)
                    result.aggregate_stats()
                    return result

            # Pass output to next agent
            current_context = {
                **current_context,
                f"{agent.name}_output": agent_result.output,
                "previous_output": agent_result.output,
            }

        result.status = ExecutionStatus.COMPLETED
        result.completed_at = datetime.now(timezone.utc)
        result.final_output = current_context.get("previous_output")
        result.aggregate_stats()

        return result

    async def _execute_agent(
        self,
        agent: AgentSpec,
        task: str,
        context: dict[str, Any],
        orchestrator: Optional[Any],
    ) -> AgentResult:
        """Execute a single agent."""
        start_time = datetime.now(timezone.utc)

        try:
            if orchestrator:
                # Use real orchestrator
                output = await self._run_with_orchestrator(agent, task, context, orchestrator)
            else:
                # Mock execution for testing
                output = f"[{agent.name}] Completed: {task}"

            return AgentResult(
                agent_name=agent.name,
                status=ExecutionStatus.COMPLETED,
                output=output,
                started_at=start_time,
                completed_at=datetime.now(timezone.utc),
                duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
            )

        except Exception as e:
            logger.error(f"Agent {agent.name} failed: {e}")
            return AgentResult(
                agent_name=agent.name,
                status=ExecutionStatus.FAILED,
                error=str(e),
                started_at=start_time,
                completed_at=datetime.now(timezone.utc),
            )

    async def _run_with_orchestrator(
        self,
        agent: AgentSpec,
        task: str,
        context: dict[str, Any],
        orchestrator: Any,
    ) -> Any:
        """Run agent using the orchestrator."""
        # Build prompt with context
        prompt_parts = [f"Task: {task}"]

        if "previous_output" in context:
            prompt_parts.append(f"\nPrevious agent output:\n{context['previous_output']}")

        if agent.system_prompt:
            prompt_parts.insert(0, agent.system_prompt)

        _prompt = "\n\n".join(prompt_parts)

        # NOTE: Ensemble requires orchestrator integration for parallel agent execution
        # Deferred: Ensemble feature pending multi-agent coordination refactor
        # For now, return mock output
        return f"[{agent.name}] Output for: {task}"


class Parallel(Ensemble):
    """Parallel execution pattern.

    Executes all agents concurrently and aggregates results.

    Example:
        parallel = Parallel([
            security_checker,
            code_reviewer,
            test_runner,
        ])
        # All run at the same time
    """

    def __init__(
        self,
        agents: list[AgentSpec],
        name: Optional[str] = None,
        require_all: bool = True,
        aggregator: Optional[Callable[[list[AgentResult]], Any]] = None,
    ):
        """Initialize parallel ensemble.

        Args:
            agents: Agents to execute in parallel
            name: Ensemble name
            require_all: Fail if any agent fails
            aggregator: Function to aggregate results
        """
        super().__init__(agents, name)
        self.require_all = require_all
        self.aggregator = aggregator or self._default_aggregator

    @property
    def ensemble_type(self) -> EnsembleType:
        return EnsembleType.PARALLEL

    async def execute(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
        orchestrator: Optional[Any] = None,
    ) -> EnsembleResult:
        """Execute all agents in parallel."""
        result = EnsembleResult(
            ensemble_type=self.ensemble_type,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )

        # Execute all agents concurrently
        tasks = [
            self._execute_agent(agent, task, context or {}, orchestrator) for agent in self.agents
        ]
        agent_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, res in enumerate(agent_results):
            if isinstance(res, Exception):
                result.agent_results.append(
                    AgentResult(
                        agent_name=self.agents[i].name,
                        status=ExecutionStatus.FAILED,
                        error=str(res),
                    )
                )
            elif isinstance(res, AgentResult):
                result.agent_results.append(res)
            else:
                # Unexpected type, convert to error result
                result.agent_results.append(
                    AgentResult(
                        agent_name=self.agents[i].name,
                        status=ExecutionStatus.FAILED,
                        error=f"Unexpected result type: {type(res)}",
                    )
                )

        # Check for failures
        failed = [r for r in result.agent_results if not r.success]
        if failed and self.require_all:
            result.status = ExecutionStatus.FAILED
        else:
            result.status = ExecutionStatus.COMPLETED

        result.completed_at = datetime.now(timezone.utc)
        result.final_output = self.aggregator(result.agent_results)
        result.aggregate_stats()

        return result

    async def _execute_agent(
        self,
        agent: AgentSpec,
        task: str,
        context: dict[str, Any],
        orchestrator: Optional[Any],
    ) -> AgentResult:
        """Execute a single agent."""
        start_time = datetime.now(timezone.utc)

        try:
            # Mock execution for now
            output = f"[{agent.name}] Parallel result: {task}"

            return AgentResult(
                agent_name=agent.name,
                status=ExecutionStatus.COMPLETED,
                output=output,
                started_at=start_time,
                completed_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            return AgentResult(
                agent_name=agent.name,
                status=ExecutionStatus.FAILED,
                error=str(e),
                started_at=start_time,
                completed_at=datetime.now(timezone.utc),
            )

    def _default_aggregator(self, results: list[AgentResult]) -> dict[str, Any]:
        """Default result aggregation."""
        return {r.agent_name: r.output for r in results if r.success}


class Hierarchical(Ensemble):
    """Hierarchical delegation pattern.

    A manager agent delegates subtasks to worker agents.

    Example:
        hierarchical = Hierarchical(
            manager=architect_agent,
            workers=[frontend_dev, backend_dev, db_admin],
        )
    """

    def __init__(
        self,
        manager: AgentSpec,
        workers: list[AgentSpec],
        name: Optional[str] = None,
        max_delegations: int = 10,
    ):
        """Initialize hierarchical ensemble.

        Args:
            manager: Manager agent that delegates
            workers: Worker agents that execute subtasks
            name: Ensemble name
            max_delegations: Maximum delegation count
        """
        super().__init__([manager] + workers, name)
        self.manager = manager
        self.workers = workers
        self.max_delegations = max_delegations

    @property
    def ensemble_type(self) -> EnsembleType:
        return EnsembleType.HIERARCHICAL

    async def execute(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
        orchestrator: Optional[Any] = None,
    ) -> EnsembleResult:
        """Execute with hierarchical delegation."""
        result = EnsembleResult(
            ensemble_type=self.ensemble_type,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )

        # Manager analyzes task and creates delegation plan
        manager_result = await self._run_manager(task, context or {}, orchestrator)
        result.agent_results.append(manager_result)

        if not manager_result.success:
            result.status = ExecutionStatus.FAILED
            result.completed_at = datetime.now(timezone.utc)
            return result

        # Execute worker tasks (mock for now)
        for worker in self.workers:
            worker_result = AgentResult(
                agent_name=worker.name,
                status=ExecutionStatus.COMPLETED,
                output=f"[{worker.name}] Completed delegated task",
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )
            result.agent_results.append(worker_result)

        result.status = ExecutionStatus.COMPLETED
        result.completed_at = datetime.now(timezone.utc)
        result.final_output = "Hierarchical task completed"
        result.aggregate_stats()

        return result

    async def _run_manager(
        self,
        task: str,
        context: dict[str, Any],
        orchestrator: Optional[Any],
    ) -> AgentResult:
        """Run the manager agent."""
        start_time = datetime.now(timezone.utc)

        # Mock manager execution
        return AgentResult(
            agent_name=self.manager.name,
            status=ExecutionStatus.COMPLETED,
            output=f"[{self.manager.name}] Delegated task to workers",
            started_at=start_time,
            completed_at=datetime.now(timezone.utc),
        )


def create_pipeline(
    agents: list[AgentSpec | str],
    name: Optional[str] = None,
) -> Pipeline:
    """Factory to create a pipeline.

    Args:
        agents: Agent specs or preset names
        name: Pipeline name

    Returns:
        Configured Pipeline
    """
    from victor.agent.specs.presets import get_preset_agent

    resolved = []
    for agent in agents:
        if isinstance(agent, str):
            resolved.append(get_preset_agent(agent))
        else:
            resolved.append(agent)

    return Pipeline(resolved, name=name)


def create_parallel(
    agents: list[AgentSpec | str],
    name: Optional[str] = None,
) -> Parallel:
    """Factory to create a parallel ensemble.

    Args:
        agents: Agent specs or preset names
        name: Ensemble name

    Returns:
        Configured Parallel
    """
    from victor.agent.specs.presets import get_preset_agent

    resolved = []
    for agent in agents:
        if isinstance(agent, str):
            resolved.append(get_preset_agent(agent))
        else:
            resolved.append(agent)

    return Parallel(resolved, name=name)
