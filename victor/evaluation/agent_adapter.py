# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Victor Agent Adapter for agentic benchmarks.

This module provides an adapter that connects Victor's AgentOrchestrator
to the AgenticBenchmarkRunner, enabling evaluation of Victor's agentic
capabilities on real-world coding tasks.

Features:
- Captures tool calls and file edits during execution
- Tracks multi-turn conversation metrics
- Generates patch diffs from file modifications
- Integrates with Victor's tool system

Usage:
    from victor.evaluation.agent_adapter import (
        VictorAgentAdapter,
        create_victor_agent_callback,
    )

    # Create adapter for a profile
    adapter = VictorAgentAdapter.from_profile("default", base_url="http://localhost:11434")

    # Use as callback for AgenticBenchmarkRunner
    callback = create_victor_agent_callback(adapter)
    result = await runner.run_task(task, callback, config)
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from victor.config.settings import load_settings

# Use TYPE_CHECKING to avoid circular import at runtime
# Chain: orchestrator → code_correction_middleware → evaluation → agent_adapter → orchestrator
if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
else:
    # Deferred import at runtime - only when actually needed
    AgentOrchestrator = None  # Will be imported lazily in from_profile()
from victor.evaluation.agentic_harness import (
    AgenticExecutionTrace,
    FileEdit,
    ToolCall,
)
from victor.evaluation.correction.metrics import (
    CorrectionMetricsCollector,
)
from victor.evaluation.protocol import BenchmarkTask
from victor.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


@dataclass
class AdapterConfig:
    """Configuration for the Victor agent adapter."""

    max_turns: int = 20  # Maximum conversation turns
    tool_budget: int = 50  # Maximum tool calls
    max_tool_calls: int = 50  # Alias for tool_budget (backwards compat)
    timeout_per_turn: int = 120  # Seconds per turn
    track_file_edits: bool = True
    track_diffs: bool = True
    working_dir: Optional[Path] = None

    # Correction metrics tracking
    track_corrections: bool = True  # Enable correction metrics collection
    keep_correction_attempts: bool = False  # Store individual correction records


class VictorAgentAdapter:
    """Adapter that connects Victor's orchestrator to agentic benchmarks.

    This adapter wraps the AgentOrchestrator and captures execution traces
    suitable for benchmark evaluation.
    """

    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        config: Optional[AdapterConfig] = None,
    ):
        """Initialize the adapter.

        Args:
            orchestrator: Victor's AgentOrchestrator instance
            config: Adapter configuration
        """
        self.orchestrator = orchestrator
        self.config = config or AdapterConfig()

        # Execution tracking
        self._tool_calls: List[ToolCall] = []
        self._file_edits: List[FileEdit] = []
        self._messages: List[Dict[str, str]] = []
        self._turns: int = 0
        self._file_snapshots: Dict[str, str] = {}  # path -> content before edit

        # Correction metrics collector
        self._metrics_collector: Optional[CorrectionMetricsCollector] = None
        if self.config.track_corrections:
            self._metrics_collector = CorrectionMetricsCollector(
                keep_attempts=self.config.keep_correction_attempts
            )

        # Hook into tool execution
        self._original_tool_start = orchestrator._on_tool_start_callback
        self._original_tool_complete = orchestrator._on_tool_complete_callback
        orchestrator._on_tool_start_callback = self._on_tool_start
        orchestrator._on_tool_complete_callback = self._on_tool_complete

    def _on_tool_start(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Track tool call start."""
        # Call original callback
        if self._original_tool_start:
            self._original_tool_start(tool_name, arguments)

        # Record tool call
        self._tool_calls.append(
            ToolCall(
                name=tool_name,
                arguments=arguments,
                timestamp=time.time(),
            )
        )

        # Snapshot file before edit
        if self.config.track_file_edits and tool_name in (
            "file_write",
            "file_edit",
            "edit_file",
            "patch",
        ):
            path = arguments.get("path") or arguments.get("file_path", "")
            if path and self.config.working_dir:
                full_path = self.config.working_dir / path
                if full_path.exists():
                    try:
                        self._file_snapshots[path] = full_path.read_text()
                    except Exception:
                        pass

    def _on_tool_complete(self, result: Any) -> None:
        """Track tool call completion."""
        # Call original callback
        if self._original_tool_complete:
            self._original_tool_complete(result)

        # Update last tool call with result
        if self._tool_calls:
            last_call = self._tool_calls[-1]
            if hasattr(result, "success"):
                last_call.success = result.success
                last_call.result = str(result.result) if hasattr(result, "result") else None
            elif hasattr(result, "tool_name"):
                last_call.success = getattr(result, "success", True)
                last_call.result = getattr(result, "result", None)

        # Track file edit after completion
        if self.config.track_file_edits and self._tool_calls:
            last_call = self._tool_calls[-1]
            if last_call.name in ("file_write", "file_edit", "edit_file", "patch"):
                path = last_call.arguments.get("path") or last_call.arguments.get("file_path", "")
                if path and self.config.working_dir:
                    self._capture_file_edit(path, last_call.name)

    def _capture_file_edit(self, path: str, action: str) -> None:
        """Capture file edit after tool completion."""
        if not self.config.working_dir:
            return

        full_path = self.config.working_dir / path
        before_content = self._file_snapshots.get(path, "")
        after_content = ""

        if full_path.exists():
            try:
                after_content = full_path.read_text()
            except Exception:
                return

        # Determine action type
        if not before_content and after_content:
            edit_action = "create"
        elif before_content and not after_content:
            edit_action = "delete"
        else:
            edit_action = "modify"

        # Generate diff
        diff = ""
        if self.config.track_diffs and before_content != after_content:
            try:
                import difflib

                diff_lines = difflib.unified_diff(
                    before_content.splitlines(keepends=True),
                    after_content.splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                )
                diff = "".join(diff_lines)
            except Exception:
                pass

        self._file_edits.append(
            FileEdit(
                path=path,
                action=edit_action,
                before_content=before_content,
                after_content=after_content,
                diff=diff,
            )
        )

    def reset(self) -> None:
        """Reset tracking state for new task."""
        self._tool_calls = []
        self._file_edits = []
        self._messages = []
        self._turns = 0
        self._file_snapshots = {}
        self.orchestrator.reset_conversation()

        # Reset metrics collector for new task
        if self.config.track_corrections:
            self._metrics_collector = CorrectionMetricsCollector(
                keep_attempts=self.config.keep_correction_attempts
            )

    async def execute_task(
        self,
        task: BenchmarkTask,
        workspace_dir: Path,
    ) -> AgenticExecutionTrace:
        """Execute an agentic task and return execution trace.

        Args:
            task: The benchmark task to execute
            workspace_dir: Directory where task files are located

        Returns:
            AgenticExecutionTrace with tool calls, file edits, and messages
        """
        self.reset()
        self.config.working_dir = workspace_dir

        trace = AgenticExecutionTrace(
            task_id=task.task_id,
            start_time=time.time(),
        )

        # Build task prompt
        prompt = self._build_task_prompt(task, workspace_dir)

        try:
            # Execute agent loop
            complete = False
            while not complete and self._turns < self.config.max_turns:
                self._turns += 1

                # Add user message
                self._messages.append(
                    {"role": "user", "content": prompt if self._turns == 1 else "Continue."}
                )

                # Get agent response
                try:
                    response = await asyncio.wait_for(
                        self.orchestrator.chat(prompt if self._turns == 1 else "Continue."),
                        timeout=self.config.timeout_per_turn,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Turn {self._turns} timed out")
                    break

                assistant_content = response.content if response else ""
                self._messages.append({"role": "assistant", "content": assistant_content})

                # Check for completion signals
                complete = self._is_task_complete(assistant_content)

                # Check tool budget
                if len(self._tool_calls) >= self.config.tool_budget:
                    logger.warning(f"Tool budget exhausted: {len(self._tool_calls)}")
                    break

                # Update prompt for continuation
                prompt = "Continue."

        except Exception as e:
            logger.error(f"Task execution error: {e}")
            trace.validation_errors["execution"] = str(e)

        # Populate trace
        trace.end_time = time.time()
        trace.turns = self._turns
        trace.messages = self._messages.copy()
        trace.tool_calls = self._tool_calls.copy()
        trace.file_edits = self._file_edits.copy()

        # Generate combined patch from file edits
        trace.generated_patch = self._generate_combined_patch()

        # Populate correction metrics if tracking is enabled
        if self._metrics_collector:
            trace.correction_metrics = self._metrics_collector.metrics.to_dict()

        return trace

    def _build_task_prompt(self, task: BenchmarkTask, workspace_dir: Path) -> str:
        """Build the initial prompt for the task."""
        parts = []

        # Task description
        parts.append(f"# Task: {task.task_id}")
        parts.append("")

        if task.prompt:
            parts.append("## Problem Statement")
            parts.append(task.prompt)
            parts.append("")

        # Working directory
        parts.append("## Working Directory")
        parts.append(f"You are working in: {workspace_dir}")
        parts.append("")

        # Context code if provided
        if task.context_code:
            parts.append("## Context Code")
            parts.append("```python")
            parts.append(task.context_code)
            parts.append("```")
            parts.append("")

        # Instructions
        parts.append("## Instructions")
        parts.append("1. Analyze the problem and explore relevant files")
        parts.append("2. Implement the required changes")
        parts.append("3. Test your changes if applicable")
        parts.append("4. Say 'TASK COMPLETE' when finished")
        parts.append("")

        return "\n".join(parts)

    def _is_task_complete(self, response: str) -> bool:
        """Check if agent signals task completion."""
        completion_phrases = [
            "task complete",
            "task completed",
            "task is complete",
            "i have completed",
            "the task has been completed",
            "changes have been applied",
            "implementation complete",
        ]
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in completion_phrases)

    def _generate_combined_patch(self) -> str:
        """Generate combined unified diff from all file edits."""
        if not self._file_edits:
            return ""

        patches = []
        for edit in self._file_edits:
            if edit.diff:
                patches.append(edit.diff)

        return "\n".join(patches)

    @classmethod
    def from_profile(
        cls,
        profile: str = "default",
        base_url: Optional[str] = None,
        model_override: Optional[str] = None,
        timeout: int = 120,
        config: Optional[AdapterConfig] = None,
    ) -> "VictorAgentAdapter":
        """Create adapter from a Victor profile.

        Args:
            profile: Profile name from profiles.yaml
            base_url: Override base URL (e.g., for specific Ollama host)
            model_override: Override model from profile
            timeout: Request timeout
            config: Adapter configuration

        Returns:
            VictorAgentAdapter instance
        """
        settings = load_settings()
        profiles = settings.load_profiles()

        if profile not in profiles:
            raise ValueError(f"Profile '{profile}' not found. Available: {list(profiles.keys())}")

        profile_config = profiles[profile]

        # Get provider settings (ProfileConfig is a Pydantic model, use attribute access)
        provider_name = profile_config.provider
        model = model_override or profile_config.model

        # Handle base URL override
        if base_url:
            if provider_name == "ollama":
                os.environ["OLLAMA_HOST"] = base_url
            elif provider_name == "lmstudio":
                os.environ["LMSTUDIO_ENDPOINTS"] = base_url

        # Create provider using ProviderRegistry (class methods)
        # Note: api_key and base_url may be extra fields from profiles.yaml (ProfileConfig allows extra="allow")
        provider = ProviderRegistry.create(
            provider_name,
            settings=settings,
            api_key=getattr(profile_config, "api_key", None),
            base_url=base_url or getattr(profile_config, "base_url", None),
            timeout=timeout,
        )

        # Create orchestrator
        orchestrator = AgentOrchestrator(
            settings=settings,
            provider=provider,
            model=model,
            temperature=profile_config.temperature,
            max_tokens=profile_config.max_tokens,
            provider_name=provider_name,
        )

        return cls(orchestrator, config)


def create_victor_agent_callback(
    adapter: VictorAgentAdapter,
) -> Callable[[BenchmarkTask, Path], AgenticExecutionTrace]:
    """Create an agent callback for AgenticBenchmarkRunner.

    Args:
        adapter: VictorAgentAdapter instance

    Returns:
        Async callback function for benchmark runner
    """

    async def callback(task: BenchmarkTask, workspace_dir: Path) -> AgenticExecutionTrace:
        return await adapter.execute_task(task, workspace_dir)

    return callback


# Convenience function for quick setup
async def run_agentic_task(
    task: BenchmarkTask,
    workspace_dir: Path,
    profile: str = "default",
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> AgenticExecutionTrace:
    """Run a single agentic task with Victor.

    Args:
        task: Benchmark task to execute
        workspace_dir: Working directory for task
        profile: Victor profile name
        base_url: Override base URL
        model: Override model

    Returns:
        AgenticExecutionTrace with results
    """
    adapter = VictorAgentAdapter.from_profile(
        profile=profile,
        base_url=base_url,
        model_override=model,
    )
    return await adapter.execute_task(task, workspace_dir)
