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

"""Example coordinator demonstrating state-passed architecture.

This module provides a reference implementation of a coordinator using
the state-passed architecture pattern. It demonstrates how to:

1. Receive ContextSnapshot instead of orchestrator reference
2. Return CoordinatorResult with StateTransitions
3. Avoid direct state mutation
4. Make coordinators testable and deterministic

This is a TEMPLATE/EXAMPLE for refactoring existing coordinators.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.agent.coordinators.state_context import (
    ContextSnapshot,
    CoordinatorResult,
    StateTransition,
    TransitionBatch,
    TransitionType,
)
from victor.framework.task import TaskComplexity

if TYPE_CHECKING:
    from victor.providers.base import Message, ToolCall

logger = logging.getLogger(__name__)


class ExampleStatePassedCoordinator:
    """Example coordinator using state-passed architecture.

    This coordinator demonstrates the pure function pattern where:
    - Input: Immutable ContextSnapshot
    - Output: CoordinatorResult with StateTransitions
    - No side effects during execution

    Example usage:
        coordinator = ExampleStatePassedCoordinator()
        snapshot = create_snapshot(orchestrator)
        result = await coordinator.analyze(snapshot, "What files exist?")
        # Apply transitions to orchestrator
        await applier.apply_batch(result.transitions)
    """

    def __init__(
        self,
        complexity_threshold: TaskComplexity = TaskComplexity.MEDIUM,
    ):
        """Initialize the coordinator.

        Args:
            complexity_threshold: Minimum complexity to trigger special handling
        """
        self._complexity_threshold = complexity_threshold

    async def analyze(
        self,
        context: ContextSnapshot,
        user_message: str,
    ) -> CoordinatorResult:
        """Analyze user message and determine next actions.

        This is a pure function that:
        1. Reads from the immutable context snapshot
        2. Decides what should happen next
        3. Returns state transitions expressing those decisions

        Args:
            context: Immutable snapshot of current state
            user_message: User's message

        Returns:
            CoordinatorResult with transitions to apply
        """
        # Example 1: Simple message logging
        # No orchestration access needed - just return transitions
        return self._handle_simple_query(context, user_message)

    def _handle_simple_query(
        self,
        context: ContextSnapshot,
        user_message: str,
    ) -> CoordinatorResult:
        """Handle a simple query that doesn't require tool execution.

        This demonstrates reading from context and returning transitions.

        Args:
            context: Immutable context snapshot
            user_message: User's message

        Returns:
            CoordinatorResult with message transition
        """
        # Read from context (no direct orchestrator access)
        message_count = context.message_count
        stage = context.conversation_stage

        # Make decisions based on context
        reasoning = f"Processing simple query (message #{message_count + 1}, stage={stage})"

        # Create transition batch
        batch = TransitionBatch()

        # Example: Update state to track query type
        batch.update_state("last_query_type", "simple", scope="conversation")

        # Example: Set capability flag
        batch.add(StateTransition(
            transition_type=TransitionType.UPDATE_CAPABILITY,
            data={"capability": "simple_query_handled", "value": True},
        ))

        return CoordinatorResult(
            transitions=batch,
            reasoning=reasoning,
            confidence=0.9,
            should_continue=True,
        )

    async def analyze_with_tool_execution(
        self,
        context: ContextSnapshot,
        user_message: str,
    ) -> CoordinatorResult:
        """Analyze and potentially request tool execution.

        This demonstrates requesting tool execution through transitions.

        Args:
            context: Immutable context snapshot
            user_message: User's message

        Returns:
            CoordinatorResult with potential tool execution transition
        """
        # Analyze message content (pure computation)
        needs_file_info = "file" in user_message.lower() or "directory" in user_message.lower()

        if not needs_file_info:
            # No tool needed, just acknowledge
            return CoordinatorResult.no_op(
                reasoning="No file information requested, skipping tool execution"
            )

        # Check context for existing information
        has_observed_files = len(context.observed_files) > 0

        if has_observed_files:
            # We already have file information
            reasoning = f"Using existing file info from {len(context.observed_files)} observed files"

            return CoordinatorResult(
                transitions=TransitionBatch(),
                reasoning=reasoning,
                confidence=0.7,
            )

        # Request tool execution through transition
        reasoning = "Need to list files to answer query about file/directory structure"

        batch = TransitionBatch()
        batch.execute_tool(
            tool_name="list_directory",
            arguments={"path": "."}
        )

        return CoordinatorResult(
            transitions=batch,
            reasoning=reasoning,
            confidence=0.95,
            metadata={
                "tool_requested": "list_directory",
                "tool_reason": "file_structure_inquiry",
            }
        )

    async def analyze_with_multi_step_plan(
        self,
        context: ContextSnapshot,
        user_message: str,
    ) -> CoordinatorResult:
        """Analyze and create a multi-step plan.

        This demonstrates creating complex state transitions.

        Args:
            context: Immutable context snapshot
            user_message: User's message

        Returns:
            CoordinatorResult with multi-step transitions
        """
        # Check if this looks like a complex task
        complexity_indicators = [
            "analyze",
            "implement",
            "refactor",
            "multiple",
            "several",
            "each",
        ]

        is_complex = any(indicator in user_message.lower() for indicator in complexity_indicators)

        if not is_complex:
            return CoordinatorResult.no_op(
                reasoning="Query appears simple, no planning needed"
            )

        # Create a multi-step plan
        batch = TransitionBatch()

        # Step 1: Update conversation stage
        batch.add(StateTransition(
            transition_type=TransitionType.UPDATE_STAGE,
            data={"stage": "planning"},
        ))

        # Step 2: Store the plan in state
        plan_steps = [
            "Analyze current state",
            "Identify necessary changes",
            "Implement changes",
            "Verify results",
        ]

        batch.update_state("plan_steps", plan_steps, scope="conversation")
        batch.update_state("current_step", 0, scope="conversation")

        # Step 3: Set capability to track planning mode
        batch.add(StateTransition(
            transition_type=TransitionType.UPDATE_CAPABILITY,
            data={"capability": "planning_mode", "value": True},
        ))

        reasoning = f"Created {len(plan_steps)}-step plan for complex task"

        return CoordinatorResult(
            transitions=batch,
            reasoning=reasoning,
            confidence=0.85,
            metadata={
                "plan_step_count": len(plan_steps),
                "complexity": "high",
            }
        )

    def get_analysis_summary(self, context: ContextSnapshot) -> Dict[str, Any]:
        """Get analysis summary from context (demonstrates context reading).

        This is a synchronous helper showing how to read various
        context properties without orchestrator access.

        Args:
            context: Immutable context snapshot

        Returns:
            Summary dictionary
        """
        return {
            "message_count": context.message_count,
            "conversation_stage": context.conversation_stage,
            "is_complete": context.is_complete,
            "has_observed_files": len(context.observed_files) > 0,
            "capabilities": list(context.capabilities.keys()),
            "model": context.model,
            "provider": context.provider,
            "conversation_state_keys": list(context.conversation_state.keys()),
            "session_state_keys": list(context.session_state.keys()),
        }


class OrchestratorIntegrationExample:
    """Example showing how to integrate state-passed coordinator with orchestrator.

    This demonstrates the pattern for using state-passed coordinators
    within the orchestrator's execution flow.
    """

    @staticmethod
    async def example_usage(orchestrator: Any, user_message: str) -> None:
        """Example of using state-passed coordinator in orchestrator.

        Args:
            orchestrator: AgentOrchestrator instance
            user_message: User's message to process
        """
        from victor.agent.coordinators.state_context import (
            create_snapshot,
            TransitionApplier,
        )

        # 1. Create coordinator
        coordinator = ExampleStatePassedCoordinator()

        # 2. Create snapshot from orchestrator
        snapshot = create_snapshot(orchestrator)

        # 3. Call coordinator with snapshot (pure function)
        result = await coordinator.analyze(snapshot, user_message)

        # 4. Apply transitions to orchestrator
        applier = TransitionApplier(orchestrator)
        await applier.apply_batch(result.transitions)

        # 5. Use result metadata if needed
        if result.reasoning:
            logger.info(f"Coordinator reasoning: {result.reasoning}")

        if not result.should_continue:
            logger.info("Coordinator requested to stop processing")

        if result.handoff_to:
            logger.info(f"Coordinator requested handoff to: {result.handoff_to}")


# Testing example
class ExampleStatePassedCoordinatorTest:
    """Example test showing how to test state-passed coordinators.

    This demonstrates the benefit of state-passed architecture:
    coordinators can be tested without mocking the full orchestrator.
    """

    @staticmethod
    def create_test_snapshot(
        messages: Optional[List] = None,
        conversation_stage: str = "initial",
    ) -> ContextSnapshot:
        """Create a test context snapshot.

        Args:
            messages: Optional message list
            conversation_stage: Conversation stage

        Returns:
            Test context snapshot
        """
        from victor.config.settings import Settings

        return ContextSnapshot(
            messages=tuple(messages or []),
            session_id="test-session",
            conversation_stage=conversation_stage,
            settings=Settings(),
            model="test-model",
            provider="test-provider",
            max_tokens=4096,
            temperature=0.7,
            conversation_state={},
            session_state={},
            observed_files=(),
            capabilities={},
        )

    @staticmethod
    async def test_simple_query():
        """Test simple query handling."""
        coordinator = ExampleStatePassedCoordinator()
        snapshot = ExampleStatePassedCoordinatorTest.create_test_snapshot()

        result = await coordinator.analyze(snapshot, "What files exist?")

        # Assertions (no orchestrator mock needed!)
        assert result.confidence > 0.5
        assert result.should_continue is True
        assert not result.transitions.is_empty()  # Should have transitions
        assert result.reasoning is not None

        print("✓ Test passed: simple_query")

    @staticmethod
    async def test_tool_execution_request():
        """Test tool execution through transitions."""
        coordinator = ExampleStatePassedCoordinator()
        snapshot = ExampleStatePassedCoordinatorTest.create_test_snapshot()

        result = await coordinator.analyze_with_tool_execution(
            snapshot,
            "List all files in the project"
        )

        # Verify tool execution was requested
        assert not result.transitions.is_empty()

        # Find the tool execution transition
        tool_transitions = [
            t for t in result.transitions.transitions
            if t.transition_type == TransitionType.EXECUTE_TOOL
        ]

        assert len(tool_transitions) > 0
        assert tool_transitions[0].data["tool_name"] == "list_directory"

        print("✓ Test passed: tool_execution_request")

    @staticmethod
    async def test_context_reading():
        """Test reading from context snapshot."""
        coordinator = ExampleStatePassedCoordinator()
        snapshot = ExampleStatePassedCoordinatorTest.create_test_snapshot(
            conversation_stage="planning"
        )

        summary = coordinator.get_analysis_summary(snapshot)

        # Verify context was read correctly
        assert summary["conversation_stage"] == "planning"
        assert summary["message_count"] == 0
        assert summary["is_complete"] is False

        print("✓ Test passed: context_reading")


# Run example tests if executed directly
if __name__ == "__main__":
    import asyncio

    async def main():
        """Run example tests."""
        await ExampleStatePassedCoordinatorTest.test_simple_query()
        await ExampleStatePassedCoordinatorTest.test_tool_execution_request()
        await ExampleStatePassedCoordinatorTest.test_context_reading()
        print("\n✓ All tests passed!")

    asyncio.run(main())
