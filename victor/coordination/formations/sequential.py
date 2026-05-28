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

"""Sequential formation strategy.

Agents execute one after another, with context chaining:
output of agent N becomes input for agent N+1.
"""

import logging
from typing import Any, Dict, List

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.teams.types import AgentMessage, MemberResult, MessageType

logger = logging.getLogger(__name__)


class SequentialFormation(BaseFormationStrategy):
    """Execute agents sequentially with context accumulation.

    In sequential formation:
    - All agents receive the same task
    - Agent 1 executes with initial context
    - Agent 2 executes with Agent 1's output in context
    - Agent N executes with all previous outputs in context
    - Context accumulates through shared_state

    Use case: Multi-perspective analysis, sequential review
    """

    async def execute(
        self,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> List[MemberResult]:
        """Execute agents sequentially with context chaining."""
        results = []
        previous_output = None
        previous_agent_id = None

        for i, agent in enumerate(agents):
            logger.debug(
                f"SequentialFormation: executing agent {i+1}/{len(agents)}: {agent.id}"
            )

            # Add previous output and agent to context
            if previous_output:
                context.shared_state["previous_output"] = previous_output
            if previous_agent_id is not None:
                context.shared_state["previous_agent"] = previous_agent_id

            # Create task for this agent with previous_agent in data
            agent_task = task
            if previous_agent_id is not None:
                # Create a copy of task data with previous_agent
                task_data = dict(task.data)
                task_data["previous_agent"] = previous_agent_id
                agent_task = AgentMessage(
                    sender_id=task.sender_id,
                    content=task.content,
                    message_type=task.message_type,
                    recipient_id=agent.id,
                    data=task_data,
                    timestamp=task.timestamp,
                    reply_to=task.reply_to,
                    priority=task.priority,
                )

            # Execute agent with task
            try:
                result = await agent.execute(agent_task, context)
                results.append(result)

                # Store output and agent ID for next agent's context
                if result.success and result.output:
                    previous_output = result.output
                    previous_agent_id = agent.id

            except Exception as e:
                logger.error(f"SequentialFormation: agent {agent.id} failed: {e}")
                results.append(
                    MemberResult(
                        member_id=agent.id,
                        success=False,
                        output="",
                        error=str(e),
                        metadata={"index": i},
                    )
                )
                # Continue with next agent even if one fails

        return results

    def validate_context(self, context: TeamContext) -> bool:
        """Sequential formation requires minimal context."""
        return context is not None

    def supports_early_termination(self) -> bool:
        """Sequential formation can terminate on first failure."""
        return True
