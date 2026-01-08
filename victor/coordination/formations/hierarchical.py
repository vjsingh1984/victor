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

"""Hierarchical formation strategy.

Manager agent delegates to worker agents,
then synthesizes results.
"""

import logging
from typing import Any, Dict, List, Optional

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.teams.types import AgentMessage, MemberResult, MessageType

logger = logging.getLogger(__name__)


class HierarchicalFormation(BaseFormationStrategy):
    """Execute with manager-worker delegation pattern.

    In hierarchical formation:
    - First agent (manager) delegates tasks to workers
    - Workers execute in parallel
    - Manager synthesizes worker results
    - Requires manager role in first position

    Use case: Complex task decomposition, supervised execution
    """

    def get_required_roles(self) -> Optional[List[str]]:
        """Hierarchical formation requires manager role."""
        return ["manager", "coordinator", "lead"]

    async def execute(
        self,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> List[MemberResult]:
        """Execute with manager-worker pattern."""
        if len(agents) < 2:
            raise ValueError(
                "Hierarchical formation requires at least 2 agents (manager + workers)"
            )

        # Check for explicit manager in context first
        explicit_manager_id = context.shared_state.get("explicit_manager_id")

        # Auto-detect manager: find agent with DELEGATE capability or role attribute
        manager = None
        workers = []

        for i, agent in enumerate(agents):
            # If explicit manager is set, use it
            if explicit_manager_id and agent.id == explicit_manager_id:
                manager = agent
                continue

            # Otherwise check for manager by capability
            is_manager = False
            if hasattr(agent, "role") and hasattr(agent.role, "capabilities"):
                from victor.framework.agent_protocols import AgentCapability

                if AgentCapability.DELEGATE in agent.role.capabilities:
                    is_manager = True

            # Check if agent has _role attribute with manager in name
            if not is_manager and hasattr(agent, "_role") and hasattr(agent._role, "name"):
                role_name = agent._role.name.lower()
                if "manager" in role_name or "lead" in role_name or "coordinator" in role_name:
                    is_manager = True

            if is_manager:
                manager = agent
            else:
                workers.append(agent)

        # If no manager found by capability, use first agent as fallback
        if manager is None:
            logger.warning(
                "HierarchicalFormation: no manager detected by capability, using first agent as manager"
            )
            manager = agents[0]
            workers = agents[1:]
        else:
            if explicit_manager_id:
                logger.info(f"HierarchicalFormation: using explicit manager={manager.id}")
            else:
                logger.info(f"HierarchicalFormation: auto-detected manager={manager.id}")

        logger.debug(
            f"HierarchicalFormation: manager={manager.id}, workers={[w.id for w in workers]}"
        )

        # Phase 1: Manager plans and delegates (executes first!)
        manager_result = await manager.execute(task, context)

        # Create results list with manager first, followed by workers in original order
        results = []
        results.append(manager_result)

        # Check if manager created delegation tasks
        if not manager_result.success or not manager_result.metadata.get("delegated_tasks"):
            logger.info(
                "HierarchicalFormation: manager did not delegate tasks, executing all workers with original task"
            )
            # Fallback: Execute all workers with the original task
            worker_tasks = [
                self._execute_worker(worker, task, context, i) for i, worker in enumerate(workers)
            ]

            # Execute workers in parallel
            import asyncio

            worker_results = await asyncio.gather(*worker_tasks, return_exceptions=True)

            # Process worker results
            for i, result in enumerate(worker_results):
                if isinstance(result, Exception):
                    logger.error(f"HierarchicalFormation: worker {workers[i].id} failed: {result}")
                    results.append(
                        MemberResult(
                            member_id=workers[i].id,
                            success=False,
                            output="",
                            error=str(result),
                            metadata={"index": i + 1, "role": "worker"},
                        )
                    )
                else:
                    results.append(result)

            return results

        # Phase 2: Workers execute delegated tasks in parallel
        delegated_tasks = manager_result.metadata["delegated_tasks"]

        if len(delegated_tasks) != len(workers):
            logger.warning(
                f"HierarchicalFormation: task count mismatch: "
                f"{len(delegated_tasks)} tasks vs {len(workers)} workers"
            )

        worker_tasks = []
        for i, worker in enumerate(workers):
            if i < len(delegated_tasks):
                worker_task = delegated_tasks[i]
                worker_tasks.append(self._execute_worker(worker, worker_task, context, i))

        # Execute workers in parallel
        import asyncio

        worker_results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        # Process worker results
        for i, result in enumerate(worker_results):
            if isinstance(result, Exception):
                logger.error(f"HierarchicalFormation: worker {workers[i].id} failed: {result}")
                results.append(
                    MemberResult(
                        member_id=workers[i].id,
                        success=False,
                        output="",
                        error=str(result),
                        metadata={"index": i + 1, "role": "worker"},
                    )
                )
            else:
                results.append(result)

        # Phase 3: Manager synthesizes results
        synthesis_task = AgentMessage(
            message_type=MessageType.RESULT,
            sender_id="system",
            recipient_id=manager.id,
            content={
                "task": "Synthesize worker results",
                "worker_results": [
                    {
                        "agent_id": r.member_id,
                        "success": r.success,
                        "content": r.output,
                    }
                    for r in results[1:]
                ],
            },
        )

        synthesis_result = await manager.execute(synthesis_task, context)
        results[0] = synthesis_result  # Replace with final synthesis

        return results

    async def _execute_worker(
        self,
        worker: Any,
        task: AgentMessage,
        context: TeamContext,
        index: int,
    ) -> MemberResult:
        """Execute a single worker."""
        logger.debug(f"HierarchicalFormation: executing worker {index+1}: {worker.id}")
        return await worker.execute(task, context)

    def validate_context(self, context: TeamContext) -> bool:
        """Hierarchical formation requires delegation support."""
        return context is not None and hasattr(context, "shared_state")

    def supports_early_termination(self) -> bool:
        """Hierarchical formation requires all workers to complete."""
        return False
