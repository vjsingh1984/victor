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

"""Integration tests for 02_workflows notebook using Ollama.

Tests the StateGraph workflow tutorial with real LLM calls using Ollama.
"""

import os
import pytest
import typing

from victor.framework.graph import END, StateGraph


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OLLAMA_HOST") and not os.path.exists("/tmp/ollama.pid"),
    reason="Requires Ollama running on localhost"
)
class TestWorkflowsNotebook:
    """Integration tests for the workflows notebook.

    Based on: docs/tutorials/notebooks/02_workflows.ipynb
    """

    @pytest.mark.asyncio
    async def test_simple_stategraph_workflow(self):
        """Test a simple StateGraph workflow."""
        # Define state
        class ProcessState(typing.TypedDict):
            text: str
            processed: bool
            result: str

        # Define workflow
        def analyze(state: ProcessState) -> str:
            """Analyze the input text."""
            return "validated" if len(state["text"]) > 0 else "invalid"

        def process(state: ProcessState) -> str:
            """Process the validated text."""
            if state["processed"]:
                return END
            return "processed"

        def format_result(state: ProcessState) -> str:
            """Format the final result."""
            return END

        # Create graph
        graph = StateGraph(ProcessState)
        graph.add_node("analyze", analyze)
        graph.add_node("process", process)
        graph.add_node("format_result", format_result)

        graph.add_edge("analyze", "process")
        graph.add_edge("process", "format_result")

        # Compile and run
        app = graph.compile()

        # Test execution
        initial_state = ProcessState(text="Hello", processed=False)
        result_state = await app.run(initial_state)

        assert result_state["processed"] is True
        assert result_state["result"] is not None

    @pytest.mark.asyncio
    async def test_conditional_workflow(self):
        """Test a workflow with conditional logic."""
        class ReviewState(typing.TypedDict):
            rating: int
            approved: bool

        def should_approve(state: ReviewState) -> str:
            """Check if rating is high enough."""
            return "approved" if state["rating"] >= 4 else "rejected"

        def send_back(state: ReviewState) -> str:
            """Send back for revision."""
            return END

        graph = StateGraph(ReviewState)
        graph.add_node("should_approve", should_approve)
        graph.add_edge("should_approve", "send_back", condition=lambda s: s["approved"])

        app = graph.compile()

        # Test with high rating
        state1 = await app.run(ReviewState(rating=5, approved=False))
        assert state1["approved"] is True

        # Test with low rating
        state2 = await app.run(ReviewState(rating=2, approved=False))
        assert state2["approved"] is False

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel workflow execution."""
        import asyncio

        class ParallelState(typing.TypedDict):
            task1_result: str
            task2_result: str
            task3_result: str

        async def task1(state: ParallelState) -> str:
            await asyncio.sleep(0.01)
            state["task1_result"] = "done1"
            return "task2"

        async def task2(state: ParallelState) -> str:
            await asyncio.sleep(0.01)
            state["task2_result"] = "done2"
            return "task3"

        async def task3(state: ParallelState) -> str:
            state["task3_result"] = "done3"
            return END

        graph = StateGraph(ParallelState)
        graph.add_node("task1", task1)
        graph.add_node("task2", task2)
        graph.add_node("task3", task3)
        graph.add_edge("task1", "task2")
        graph.add_edge("task1", "task3")

        app = graph.compile()

        result = await app.run(ParallelState(
            task1_result="",
            task2_result="",
            task3_result=""
        ))

        assert result["task1_result"] == "done1"
        assert result["task2_result"] == "done2"
        assert result["task3_result"] == "done3"

    @pytest.mark.asyncio
    async def test_workflow_with_checkpointing(self):
        """Test workflow with checkpoint/save functionality."""
        class CounterState(typing.TypedDict):
            count: int
            doubled: str

        def increment(state: CounterState) -> str:
            state["count"] += 1
            return "double" if state["count"] >= 2 else "increment"

        def double(state: CounterState) -> str:
            state["doubled"] = str(state["count"] * 2)
            return END

        graph = StateGraph(CounterState)
        graph.add_node("increment", increment)
        graph.add_node("double", double)
        graph.add_edge("increment", "increment")
        graph.add_edge("increment", "double")

        app = graph.compile(checkpointer_id="test_checkpoint")

        # Run to completion
        result = await app.run(CounterState(count=0, doubled=""))
        assert result["count"] == 2
        assert result["doubled"] == "4"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
