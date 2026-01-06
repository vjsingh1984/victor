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
    """Execute agents sequentially with context chaining.

    In sequential formation:
    - Agent 1 receives initial task
    - Agent 2 receives Agent 1's result as context
    - Agent N receives Agent N-1's result as context
    - Context accumulates through the chain

    Use case: Stepwise refinement, multi-stage analysis
    """

    async def execute(
        self,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> List[MemberResult]:
        """Execute agents sequentially with context chaining."""
        results = []
        current_task = task

        for i, agent in enumerate(agents):
            logger.debug(
                f"SequentialFormation: executing agent {i+1}/{len(agents)}: {agent.id}"
            )

            # Execute agent with current task
            try:
                result = await agent.execute(current_task, context)
                results.append(result)

                # Chain context for next agent
                if result.success and result.output:
                    # Create task for next agent with previous result
                    current_task = AgentMessage(
                        message_type=MessageType.TASK,
                        sender_id="system",
                        content=result.output,
                        data={
                            "previous_agent": agent.id,
                            "previous_result": result.metadata,
                        },
                    )

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
