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

"""High-level BenchmarkAgent using framework APIs.

This module provides a high-level agent for benchmark evaluation that:
- Uses Agent.create() instead of raw orchestrator
- Leverages framework observability for metrics collection
- Integrates with the BenchmarkVertical for domain-specific configuration
- Provides a clean API for benchmark harness integration

This replaces the low-level VictorAgentAdapter with a framework-aligned approach.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from victor.benchmark.assistant import BenchmarkVertical
from victor.benchmark.task_bridge import (
    benchmark_task_to_framework_task,
    build_benchmark_prompt,
)
from victor.evaluation.protocol import BenchmarkTask

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkAgentConfig:
    """Configuration for BenchmarkAgent."""

    provider: str = "anthropic"
    model: Optional[str] = None
    timeout_per_task: float = 300.0
    timeout_per_turn: float = 60.0
    max_turns: int = 15
    tool_budget: int = 30
    temperature: float = 0.2
    enable_thinking: bool = True
    enable_observability: bool = True
    # Workflow execution options
    use_workflow: bool = False  # If True, use YAML workflow instead of direct Agent.run()
    workflow_name: Optional[str] = None  # Workflow to use (auto-detected if None)


@dataclass
class ExecutionTrace:
    """Trace of benchmark task execution.

    Collects metrics during execution for reporting.
    """

    task_id: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    # Token usage
    tokens_input: int = 0
    tokens_output: int = 0

    # Execution metrics
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    turns: int = 0

    # Results
    generated_code: str = ""
    generated_patch: Optional[str] = None
    error: Optional[str] = None
    success: bool = False

    @property
    def tokens_used(self) -> int:
        return self.tokens_input + self.tokens_output

    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tokens_used": self.tokens_used,
            "tool_calls": len(self.tool_calls),
            "turns": self.turns,
            "generated_code": self.generated_code[:500] if self.generated_code else "",
            "error": self.error,
            "success": self.success,
        }


class BenchmarkAgent:
    """High-level agent for benchmark task execution.

    Uses the framework's Agent API instead of raw orchestrator,
    providing proper integration with vertical system, observability,
    and resource management.

    Example:
        agent = await BenchmarkAgent.create(provider="anthropic")
        trace = await agent.execute_task(benchmark_task, workspace_path)
        print(f"Tokens used: {trace.tokens_used}")
        print(f"Tool calls: {len(trace.tool_calls)}")
    """

    def __init__(
        self,
        agent: Any,  # victor.framework.Agent
        config: BenchmarkAgentConfig,
    ):
        """Initialize with a framework Agent instance.

        Use BenchmarkAgent.create() instead of direct construction.
        """
        self._agent = agent
        self._config = config
        self._current_trace: Optional[ExecutionTrace] = None
        self._event_unsubscribe: Optional[Callable[..., Any]] = None

    @classmethod
    async def create(
        cls,
        provider: str = "anthropic",
        model: Optional[str] = None,
        config: Optional[BenchmarkAgentConfig] = None,
        profile: Optional[str] = None,
    ) -> "BenchmarkAgent":
        """Create a BenchmarkAgent using framework Agent.create().

        This is the recommended way to create a BenchmarkAgent.
        It properly initializes the framework Agent with BenchmarkVertical.

        Args:
            provider: LLM provider name
            model: Optional model override
            config: Optional configuration
            profile: Optional Victor profile name

        Returns:
            Configured BenchmarkAgent instance
        """
        from victor.framework.agent import Agent

        config = config or BenchmarkAgentConfig(provider=provider, model=model)

        # Create framework agent with BenchmarkVertical
        agent = await Agent.create(
            provider=provider,
            model=model,
            vertical=BenchmarkVertical,
            temperature=config.temperature,
            thinking=config.enable_thinking,
            enable_observability=config.enable_observability,
            profile=profile,
        )

        return cls(agent, config)

    @classmethod
    async def from_profile(
        cls,
        profile: str = "default",
        model_override: Optional[str] = None,
        config: Optional[BenchmarkAgentConfig] = None,
    ) -> "BenchmarkAgent":
        """Create BenchmarkAgent from a Victor profile.

        Args:
            profile: Profile name from ~/.victor/config.yaml
            model_override: Optional model to override profile setting
            config: Optional configuration

        Returns:
            Configured BenchmarkAgent instance
        """
        from victor.framework.agent import Agent

        config = config or BenchmarkAgentConfig()

        # Create from profile - Agent.create handles profile loading
        agent = await Agent.create(
            profile=profile,
            model=model_override,
            vertical=BenchmarkVertical,
            temperature=config.temperature,
            thinking=config.enable_thinking,
            enable_observability=config.enable_observability,
        )

        return cls(agent, config)

    async def execute_task(
        self,
        task: BenchmarkTask,
        workspace_path: Optional[Path] = None,
        use_workflow: Optional[bool] = None,
        workflow_name: Optional[str] = None,
    ) -> ExecutionTrace:
        """Execute a benchmark task and return execution trace.

        This is the main entry point for benchmark evaluation.
        It converts the BenchmarkTask to a framework Task and executes it
        using either Agent.run() or a YAML workflow.

        Args:
            task: The benchmark task to execute
            workspace_path: Optional workspace directory
            use_workflow: Override config.use_workflow for this task
            workflow_name: Override config.workflow_name for this task

        Returns:
            ExecutionTrace with metrics and generated code
        """
        # Determine execution mode
        should_use_workflow = (
            use_workflow if use_workflow is not None else self._config.use_workflow
        )
        wf_name = workflow_name or self._config.workflow_name

        if should_use_workflow:
            return await self._execute_with_workflow(task, workspace_path, wf_name)
        else:
            return await self._execute_direct(task, workspace_path)

    async def _execute_direct(
        self,
        task: BenchmarkTask,
        workspace_path: Optional[Path] = None,
    ) -> ExecutionTrace:
        """Execute task using direct Agent.run() call.

        This is the original execution path, providing maximum flexibility
        but not using the structured YAML workflow.
        """
        trace = ExecutionTrace(task_id=task.task_id)
        self._current_trace = trace

        try:
            # Subscribe to events for metrics collection
            self._subscribe_to_events(trace)

            # Set workspace if provided
            if workspace_path:
                orchestrator = self._agent.get_orchestrator()
                if hasattr(orchestrator, "set_workspace"):
                    orchestrator.set_workspace(workspace_path)

            # Convert benchmark task to framework task
            framework_task = benchmark_task_to_framework_task(task)

            # Build enriched prompt
            prompt = build_benchmark_prompt(task, str(workspace_path) if workspace_path else None)

            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    self._agent.run(prompt, context=framework_task.context),
                    timeout=self._config.timeout_per_task,
                )

                # Extract results
                trace.generated_code = result.content or ""
                trace.success = result.success
                trace.error = result.error

                # Extract tool calls from result
                if result.tool_calls:
                    trace.tool_calls = result.tool_calls

                # Extract token usage from metadata
                if result.metadata:
                    trace.tokens_input = result.metadata.get("tokens_input", 0)
                    trace.tokens_output = result.metadata.get("tokens_output", 0)
                    trace.turns = result.metadata.get("turns", 0)

            except asyncio.TimeoutError:
                trace.error = f"Task timed out after {self._config.timeout_per_task}s"
                trace.success = False
                logger.warning(f"Task {task.task_id} timed out")

        except Exception as e:
            trace.error = str(e)
            trace.success = False
            logger.exception(f"Error executing task {task.task_id}")

        finally:
            trace.end_time = time.time()
            self._unsubscribe_from_events()
            self._current_trace = None

        return trace

    async def _execute_with_workflow(
        self,
        task: BenchmarkTask,
        workspace_path: Optional[Path] = None,
        workflow_name: Optional[str] = None,
    ) -> ExecutionTrace:
        """Execute task using YAML workflow system.

        Uses BenchmarkWorkflowProvider to execute structured workflows
        with proper stage management, conditions, and transforms.

        Args:
            task: The benchmark task to execute
            workspace_path: Optional workspace directory
            workflow_name: Specific workflow to use (auto-detected if None)

        Returns:
            ExecutionTrace with metrics and generated code
        """
        from victor.benchmark.workflows import BenchmarkWorkflowProvider
        from victor.workflows.streaming import WorkflowEventType

        trace = ExecutionTrace(task_id=task.task_id)
        self._current_trace = trace

        try:
            # Subscribe to events for metrics collection
            self._subscribe_to_events(trace)

            # Set workspace if provided
            orchestrator = self._agent.get_orchestrator()
            if workspace_path and hasattr(orchestrator, "set_workspace"):
                orchestrator.set_workspace(workspace_path)

            # Get workflow provider
            provider = BenchmarkWorkflowProvider()

            # Auto-detect workflow if not specified
            if not workflow_name:
                workflow_name = self._detect_workflow_for_task(task, provider)

            logger.info(f"Executing task {task.task_id} with workflow: {workflow_name}")

            # Build workflow context from task
            context = self._build_workflow_context(task, workspace_path)

            # Execute workflow with streaming for progress tracking
            # Uses the new UnifiedWorkflowCompiler-based streaming API
            final_context: Dict[str, Any] = {}
            try:

                async def execute_workflow() -> None:
                    nonlocal final_context
                    # Use new unified compiler streaming API
                    async for node_id, state in provider.stream_compiled_workflow(
                        workflow_name, context
                    ):
                        # Each yield represents a completed node
                        logger.debug(f"Completed node: {node_id}")
                        trace.turns += 1
                        # Track tool calls from state if available
                        if "_tool_calls" in state:
                            for tool_call in state.get("_tool_calls", []):
                                trace.tool_calls.append(
                                    {
                                        "name": tool_call.get("name", "unknown"),
                                        "timestamp": time.time(),
                                    }
                                )
                        # Keep latest state as final context
                        final_context = state

                await asyncio.wait_for(
                    execute_workflow(),
                    timeout=self._config.timeout_per_task,
                )

                # Extract results from workflow context
                trace.generated_code = final_context.get("solution_code", "")
                trace.generated_patch = final_context.get("generated_patch")
                trace.success = final_context.get("status") == "completed"

                # Extract metrics if available
                if "test_results" in final_context:
                    test_results = final_context["test_results"]
                    trace.success = test_results.get("pass_rate", 0) >= 0.95

            except asyncio.TimeoutError:
                trace.error = f"Workflow timed out after {self._config.timeout_per_task}s"
                trace.success = False
                logger.warning(f"Task {task.task_id} workflow timed out")

        except Exception as e:
            trace.error = str(e)
            trace.success = False
            logger.exception(f"Error executing workflow for task {task.task_id}")

        finally:
            trace.end_time = time.time()
            self._unsubscribe_from_events()
            self._current_trace = None

        return trace

    def _detect_workflow_for_task(
        self,
        task: BenchmarkTask,
        provider: Any,
    ) -> str:
        """Auto-detect appropriate workflow based on task characteristics.

        Args:
            task: The benchmark task
            provider: BenchmarkWorkflowProvider instance

        Returns:
            Workflow name to use
        """
        # Check task type hints
        task_type = getattr(task, "task_type", None) or getattr(task, "benchmark", None)

        if task_type:
            workflow = provider.get_workflow_for_task_type(str(task_type))
            if workflow:
                return str(workflow)

        # Infer from task attributes
        if hasattr(task, "repo") and task.repo:
            # SWE-bench style task
            return "swe_bench"
        elif hasattr(task, "test_code") and task.test_code:
            # Code generation with tests
            return "code_generation"
        else:
            # Default to code generation
            return "code_generation"

    def _build_workflow_context(
        self,
        task: BenchmarkTask,
        workspace_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Build workflow context from benchmark task.

        Args:
            task: The benchmark task
            workspace_path: Optional workspace directory

        Returns:
            Context dict for workflow execution
        """
        context: Dict[str, Any] = {
            "task_id": task.task_id,
            "problem_id": task.task_id,
            "task_description": task.description or task.prompt,
            "prompt": task.prompt,
            "language": getattr(task, "language", "python"),
            "workspace_path": str(workspace_path) if workspace_path else None,
            # Test configuration
            "test_command": getattr(task, "test_command", "pytest"),
            "test_timeout": min(self._config.timeout_per_task / 2, 180),
            # Iteration limits
            "max_iterations": 5,
            "fix_iterations": 0,
        }

        # SWE-bench specific fields
        if hasattr(task, "repo") and task.repo:
            context["repo"] = task.repo
            context["base_commit"] = getattr(task, "base_commit", None)
            context["issue_text"] = getattr(task, "issue_text", task.prompt)
            context["issue_summary"] = task.description or task.prompt[:200]

        # Code generation specific fields
        if hasattr(task, "test_code") and task.test_code:
            context["test_cases"] = task.test_code
            context["context_code"] = getattr(task, "context_code", "")

        return context

    def _subscribe_to_events(self, trace: ExecutionTrace) -> None:
        """Subscribe to framework events for metrics collection."""
        try:
            from victor.framework.events import EventType

            def on_event(event: Any) -> None:
                """Handle framework events."""
                if not self._current_trace:
                    return

                event_type = getattr(event, "type", None)

                if event_type == EventType.TOOL_CALL:
                    self._current_trace.tool_calls.append(
                        {
                            "name": getattr(event, "tool_name", "unknown"),
                            "timestamp": time.time(),
                        }
                    )
                elif event_type == EventType.CONTENT:
                    # Track turns based on content events
                    pass

            self._event_unsubscribe = self._agent.subscribe_to_events("*", on_event)

        except Exception as e:
            logger.debug(f"Could not subscribe to events: {e}")

    def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from framework events."""
        if self._event_unsubscribe:
            try:
                self._event_unsubscribe()
            except Exception:
                pass
            self._event_unsubscribe = None

    async def reset(self) -> None:
        """Reset agent state for next task."""
        await self._agent.reset()

    async def close(self) -> None:
        """Clean up agent resources."""
        self._unsubscribe_from_events()
        await self._agent.close()

    async def __aenter__(self) -> "BenchmarkAgent":
        """Context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Context manager exit."""
        await self.close()

    @property
    def agent(self) -> Any:
        """Access the underlying framework Agent."""
        return self._agent

    @property
    def config(self) -> BenchmarkAgentConfig:
        """Get agent configuration."""
        return self._config


async def create_benchmark_agent(
    provider: str = "anthropic",
    model: Optional[str] = None,
    profile: Optional[str] = None,
    **kwargs: Any,
) -> BenchmarkAgent:
    """Factory function for creating BenchmarkAgent.

    Convenience function that wraps BenchmarkAgent.create().

    Args:
        provider: LLM provider name
        model: Optional model name
        profile: Optional Victor profile
        **kwargs: Additional config options

    Returns:
        Configured BenchmarkAgent
    """
    config = BenchmarkAgentConfig(provider=provider, model=model, **kwargs)

    if profile:
        return await BenchmarkAgent.from_profile(profile, model, config)
    else:
        return await BenchmarkAgent.create(provider, model, config)
