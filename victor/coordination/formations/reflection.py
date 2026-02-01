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

"""Reflection formation for iterative refinement with feedback.

This module provides ReflectionFormation, which implements the reflection
loop pattern: generator → critic → refinement → (repeat).

Useful for:
- Code review and refinement
- Iterative problem solving
- Quality improvement through feedback

SOLID Principles:
- SRP: Reflection logic only
- OCP: Extensible via max_iterations parameter
- LSP: Substitutable with other formations
- DIP: Depends on TeamContext and BaseFormationStrategy abstractions

Usage:
    from victor.coordination.formations.reflection import ReflectionFormation
    from victor.coordination.formations.base import TeamContext

    # Create formation with max iterations
    formation = ReflectionFormation(max_iterations=3)

    # Create context with generator and critic agents
    context = TeamContext("team-1", "reflection")
    context.set("generator", generator_agent)
    context.set("critic", critic_agent)

    # Execute with reflection loop
    results = await formation.execute(agents, context, task)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.teams.types import AgentMessage, MemberResult

logger = logging.getLogger(__name__)


class ReflectionFormation(BaseFormationStrategy):
    """Implements reflection loop pattern for iterative refinement.

    Formation Pattern:
        Generator Agent → Critic Agent → Refinement → (repeat)

    The generator produces a solution, the critic provides feedback,
    and the generator refines based on the feedback. This continues
    until the critic is satisfied or max iterations is reached.

    Useful for:
    - Code review and iterative improvement
    - Problem refinement through feedback
    - Quality assurance cycles

    SOLID: SRP (reflection logic only), OCP (extensible via max_iterations)

    Attributes:
        max_iterations: Maximum number of refinement cycles
        satisfaction_keywords: Keywords that indicate critic satisfaction

    Example:
        >>> formation = ReflectionFormation(max_iterations=3)
        >>> context = TeamContext("team-1", "reflection")
        >>> context.set("generator", generator_agent)
        >>> context.set("critic", critic_agent)
        >>>
        >>> results = await formation.execute(
        ...     agents=[generator_agent, critic_agent],
        ...     context=context,
        ...     task=AgentMessage("Refactor this code")
        ... )
        >>>
        >>> # Results include iteration count and final feedback
        >>> print(f"Iterations: {results[0].metadata['iterations']}")
        >>> print(f"Final feedback: {results[0].metadata['final_feedback']}")
    """

    def __init__(
        self,
        max_iterations: int = 3,
        satisfaction_keywords: Optional[list[str]] = None,
    ):
        """Initialize the reflection formation.

        Args:
            max_iterations: Maximum number of refinement cycles (default: 3)
            satisfaction_keywords: Keywords that indicate critic satisfaction
                (default: ["good", "satisfactory", "acceptable", "approved", "excellent"])
        """
        self.max_iterations = max_iterations
        self.satisfaction_keywords = satisfaction_keywords or [
            "good",
            "satisfactory",
            "acceptable",
            "approved",
            "excellent",
            "perfect",
            "great",
        ]

    async def execute(
        self,
        agents: list[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> list[MemberResult]:
        """Execute reflection loop: generator → critic → refine → repeat.

        Args:
            agents: List of agents (should contain generator and critic)
            context: Team context with agent references
            task: Task message to process

        Returns:
            List of MemberResult with metadata:
                - result: Final generated result
                - iterations: Number of iterations performed
                - final_feedback: Critic's final feedback
                - satisfied: Whether critic was satisfied
        """
        # Get agents from context
        generator = context.get("generator")
        critic = context.get("critic")

        if not generator or not critic:
            logger.error("ReflectionFormation requires 'generator' and 'critic' agents in context")
            return [
                MemberResult(
                    member_id="reflection_formation",
                    success=False,
                    output="",
                    error="Missing generator or critic agent in context",
                )
            ]

        result = None
        feedback = None

        # Reflection loop
        for iteration in range(self.max_iterations):
            logger.info(f"Reflection iteration {iteration + 1}/{self.max_iterations}")

            # Generate solution
            try:
                generator_response = await generator.execute(
                    task.content, context=context.shared_state
                )
                result = generator_response
            except Exception as e:
                logger.error(f"Generator failed in iteration {iteration + 1}: {e}")
                return [
                    MemberResult(
                        member_id=generator.id,
                        success=False,
                        output=str(result) if result else "",
                        error=f"Generator failed: {str(e)}",
                    )
                ]

            # Critique the solution
            critique_prompt = (
                f"Critique this solution:\n\n{result}\n\nProvide feedback for improvement."
            )
            try:
                critique_response = await critic.execute(
                    critique_prompt, context=context.shared_state
                )
                feedback = critique_response
            except Exception as e:
                logger.warning(f"Critic failed in iteration {iteration + 1}: {e}")
                feedback = f"Critique unavailable: {str(e)}"

            # Check if satisfied
            if self._is_satisfied(feedback):
                logger.info(f"Critic satisfied after {iteration + 1} iterations")
                break

            # Refine with feedback for next iteration
            from victor.teams.types import MessageType

            task = AgentMessage(
                sender_id="reflection_formation",
                content=f"Refine this solution based on feedback:\n\n{result}\n\nFeedback: {feedback}",
                message_type=MessageType.TASK,
            )

        # Return final result with metadata
        return [
            MemberResult(
                member_id="reflection_formation",
                success=True,
                output=str(result) if result else "",
                metadata={
                    "iterations": iteration + 1,
                    "final_feedback": feedback,
                    "satisfied": self._is_satisfied(feedback),
                    "formation": "reflection",
                },
            )
        ]

    def validate_context(self, context: TeamContext) -> bool:
        """Validate that context has required generator and critic.

        Args:
            context: Team context to validate

        Returns:
            True if context has 'generator' and 'critic' keys
        """
        generator = context.get("generator")
        critic = context.get("critic")

        has_generator = generator is not None
        has_critic = critic is not None

        if not has_generator:
            logger.warning("ReflectionFormation context missing 'generator' agent")

        if not has_critic:
            logger.warning("ReflectionFormation context missing 'critic' agent")

        return has_generator and has_critic

    def get_required_roles(self) -> Optional[list[str]]:
        """Get required roles for reflection formation.

        Returns:
            List of required role names: ['generator', 'critic']
        """
        return ["generator", "critic"]

    def supports_early_termination(self) -> bool:
        """Check if formation supports early termination.

        Reflection formation can terminate early if critic is satisfied.

        Returns:
            True (supports early termination)
        """
        return True

    def _is_satisfied(self, feedback: Optional[str]) -> bool:
        """Check if critic feedback indicates satisfaction.

        Args:
            feedback: Critic's feedback text

        Returns:
            True if feedback contains satisfaction keywords
        """
        if not feedback:
            return False

        feedback_lower = feedback.lower()

        # Check for satisfaction keywords
        for keyword in self.satisfaction_keywords:
            if keyword in feedback_lower:
                return True

        return False


__all__ = ["ReflectionFormation"]
