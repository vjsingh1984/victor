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

"""Parallel formation strategy.

Agents execute simultaneously with independent contexts.
All agents receive the same task and work independently.
"""

import asyncio
import logging
from typing import Any, Dict, List, cast

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.teams.types import AgentMessage, MemberResult

logger = logging.getLogger(__name__)


class ParallelFormation(BaseFormationStrategy):
    """Execute all agents in parallel with independent contexts.

    In parallel formation:
    - All agents receive the same task simultaneously
    - Each agent works independently
    - Results are collected and combined at the end
    - No inter-agent communication during execution

    Use case: Diverse perspectives, redundancy, independent analysis
    """

    async def execute(
        self,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> List[MemberResult]:
        """Execute all agents in parallel."""
        # Create independent tasks for each agent
        tasks = [self._execute_agent(agent, task, context, i) for i, agent in enumerate(agents)]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, handling any exceptions
        processed_results: List[MemberResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"ParallelFormation: agent {agents[i].id} failed: {result}")
                processed_results.append(
                    MemberResult(
                        member_id=agents[i].id,
                        success=False,
                        output="",
                        error=str(result),
                        metadata={"index": i},
                    )
                )
            else:
                processed_results.append(cast(MemberResult, result))

        return processed_results

    async def _execute_agent(
        self,
        agent: Any,
        task: AgentMessage,
        context: TeamContext,
        index: int,
    ) -> MemberResult:
        """Execute a single agent."""
        logger.debug(f"ParallelFormation: executing agent {index+1}: {agent.id}")
        return cast(MemberResult, await agent.execute(task, context))

    def validate_context(self, context: TeamContext) -> bool:
        """Parallel formation requires minimal context."""
        return context is not None

    def supports_early_termination(self) -> bool:
        """Parallel formation waits for all agents to complete."""
        return False
