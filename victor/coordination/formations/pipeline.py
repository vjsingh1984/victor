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

"""Pipeline formation strategy.

Output of each agent feeds directly into the next agent.
Each agent transforms the data, creating a processing pipeline.
"""

import logging
from typing import Any, Dict, List

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.teams.types import AgentMessage, MemberResult, MessageType

logger = logging.getLogger(__name__)


class PipelineFormation(BaseFormationStrategy):
    """Execute agents as a processing pipeline.

    In pipeline formation:
    - Agent 1 processes initial data
    - Agent 2 transforms Agent 1's output
    - Agent N transforms Agent N-1's output
    - Each agent must produce output for next agent
    - Similar to Unix pipes: agent1 | agent2 | agent3

    Use case: Multi-stage data processing, transformation pipelines
    """

    async def execute(
        self,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> List[MemberResult]:
        """Execute agents as a pipeline."""
        results = []
        current_data = task.content

        for i, agent in enumerate(agents):
            logger.debug(
                f"PipelineFormation: stage {i+1}/{len(agents)}: {agent.id}"
            )

            # Create task for this stage
            stage_task = AgentMessage(
                message_message_type=MessageType.TASK,
                sender="system",
                receiver=agent.id,
                content=current_data,
                metadata={"stage": i, "pipeline": True},
            )

            try:
                result = await agent.execute(stage_task, context)
                results.append(result)

                if not result.success:
                    logger.error(f"PipelineFormation: stage {i} failed, stopping pipeline")
                    # Stop pipeline on failure
                    break

                # Update current_data for next stage
                if result.content is not None:
                    current_data = result.content
                else:
                    logger.warning(f"PipelineFormation: stage {i} produced no output")
                    break

            except Exception as e:
                logger.error(f"PipelineFormation: stage {i} failed: {e}")
                results.append(
                    MemberResult(
                        agent_id=agent.id,
                        success=False,
                        content=None,
                        error=str(e),
                        metadata={"stage": i, "pipeline": True},
                    )
                )
                # Stop pipeline on exception
                break

        return results

    def validate_context(self, context: TeamContext) -> bool:
        """Pipeline formation requires minimal context."""
        return context is not None

    def supports_early_termination(self) -> bool:
        """Pipeline formation terminates on first failure."""
        return True
