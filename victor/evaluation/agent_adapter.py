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
from victor.agent.task_completion import TaskCompletionDetector
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
    """Configuration for the Victor agent adapter.

    Defaults are based on ACTION complexity from ComplexityBudget,
    which is appropriate for benchmark tasks that require multiple tool calls.
    """

    # Defaults from ComplexityBudget(ACTION) - single source of truth
    # ACTION: tool_budget=50, max_turns=30, max_continuation_requests=15, timeout_seconds=600
    max_turns: int = 30  # Maximum conversation turns (ACTION complexity)
    tool_budget: int = 50  # Maximum tool calls (ACTION complexity)
    max_tool_calls: int = 50  # Alias for tool_budget (backwards compat)
    total_timeout: int = 600  # Total task timeout in seconds (ACTION complexity)
    min_turn_timeout: int = 180  # Minimum per-turn timeout (P2: Timeout Fix)
    track_file_edits: bool = True
    track_diffs: bool = True
    working_dir: Optional[Path] = None

    # Correction metrics tracking
    track_corrections: bool = True  # Enable correction metrics collection
    keep_correction_attempts: bool = False  # Store individual correction records

    @property
    def timeout_per_turn(self) -> int:
        """Calculate per-turn timeout with minimum enforcement (P2: Timeout Fix).

        Uses SafeTimeoutPolicy to ensure at least min_turn_timeout seconds per turn,
        even for slow models like DeepSeek.
        """
        from victor.evaluation.timeout_calculator import SafeTimeoutPolicy

        policy = SafeTimeoutPolicy(min_turn_timeout=self.min_turn_timeout)
        return policy.calculate(self.total_timeout, self.max_turns)


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

        # Task completion detector (uses framework's detection, not gaming code)
        self._completion_detector = TaskCompletionDetector()

        # Hook into tool execution - must update BOTH orchestrator AND ToolPipeline
        # The ToolPipeline receives a copy of the callback at init time, so we need
        # to update it directly for our hooks to receive tool call events
        self._original_tool_start = orchestrator._on_tool_start_callback
        self._original_tool_complete = orchestrator._on_tool_complete_callback
        orchestrator._on_tool_start_callback = self._on_tool_start
        orchestrator._on_tool_complete_callback = self._on_tool_complete

        # CRITICAL: Also update ToolPipeline's callbacks directly
        # Without this, tool calls are not tracked in evaluation traces
        if hasattr(orchestrator, "_tool_pipeline") and orchestrator._tool_pipeline:
            orchestrator._tool_pipeline.on_tool_start = self._on_tool_start
            orchestrator._tool_pipeline.on_tool_complete = self._on_tool_complete

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

            # Record tool result for completion detection (framework integration)
            result_dict = {
                "success": getattr(result, "success", True),
                "path": last_call.arguments.get("path")
                or last_call.arguments.get("file_path"),
            }
            self._completion_detector.record_tool_result(last_call.name, result_dict)

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

        # Reset token usage for fresh tracking per task
        if hasattr(self.orchestrator, "reset_token_usage"):
            self.orchestrator.reset_token_usage()

        # Reset metrics collector for new task
        if self.config.track_corrections:
            self._metrics_collector = CorrectionMetricsCollector(
                keep_attempts=self.config.keep_correction_attempts
            )

        # Reset completion detector for new task
        self._completion_detector.reset()

    async def execute_task(
        self,
        task: BenchmarkTask,
        workspace_dir: Path,
    ) -> AgenticExecutionTrace:
        """Execute an agentic task and return execution trace.

        Uses the framework's PromptEnrichmentService and TaskCompletionDetector
        rather than benchmark-specific gaming code. This tests Victor as users
        would actually use it.

        Args:
            task: The benchmark task to execute
            workspace_dir: Directory where task files are located

        Returns:
            AgenticExecutionTrace with tool calls, file edits, and messages
        """
        self.reset()
        self.config.working_dir = workspace_dir

        # CRITICAL: Set workspace BEFORE any orchestrator operations
        # This ensures tools like file read/write, grep, etc. operate on the benchmark
        # repo rather than Victor's own codebase. Uses framework method for proper
        # encapsulation of project context updates.
        self.orchestrator.set_workspace(workspace_dir)

        trace = AgenticExecutionTrace(
            task_id=task.task_id,
            start_time=time.time(),
        )

        # Inject task context into vertical context for framework enrichment
        # This is the proper architecture: hints flow through the enrichment pipeline
        self._inject_task_context(task, workspace_dir)

        # Analyze intent for completion detection (framework feature)
        task_description = task.issue_text or task.prompt
        self._completion_detector.analyze_intent(task_description)

        # Determine task complexity with fallback chain:
        # 1. Explicit override from task (highest priority)
        # 2. Inference from task description
        # 3. Conservative default (MEDIUM)
        complexity_value = self._determine_complexity(task, task_description)
        self._completion_detector.configure_for_complexity(complexity_value)

        # Use task's issue_text or prompt directly - let framework handle enrichment
        prompt = task_description

        try:
            # Execute agent loop
            complete = False
            while not complete and self._turns < self.config.max_turns:
                self._turns += 1

                # Add user message
                current_message = prompt if self._turns == 1 else "Continue."
                self._messages.append({"role": "user", "content": current_message})

                # Get agent response
                try:
                    response = await asyncio.wait_for(
                        self.orchestrator.chat(current_message),
                        timeout=self.config.timeout_per_turn,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Turn {self._turns} timed out")
                    break

                assistant_content = response.content if response else ""
                self._messages.append({"role": "assistant", "content": assistant_content})

                # Use framework's completion detection (not gaming code)
                self._completion_detector.analyze_response(assistant_content)
                complete = self._completion_detector.should_stop()

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

        # Capture token usage from orchestrator (P1: Token Tracking Fix)
        if hasattr(self.orchestrator, "get_token_usage"):
            trace.token_usage = self.orchestrator.get_token_usage()

        return trace

    def _determine_complexity(self, task: BenchmarkTask, task_description: str) -> str:
        """Determine task complexity using fallback chain.

        Priority order:
        1. Explicit override from task.complexity_override (caller knows best)
        2. Inference from task description using TaskComplexityService
        3. Conservative default: "medium"

        This design:
        - Keeps override optional (minimal config)
        - Allows max control when needed
        - Framework doesn't fight what caller wants

        Args:
            task: The benchmark task (may have complexity_override)
            task_description: Text to classify if no override

        Returns:
            Complexity level string: simple, medium, complex, generation, action, analysis
        """
        # Priority 1: Explicit override (caller knows best)
        if task.complexity_override:
            logger.info(
                f"Using explicit complexity override: {task.complexity_override}"
            )
            return task.complexity_override

        # Priority 2: Inference from task description
        try:
            from victor.framework.task.complexity import TaskComplexityService

            service = TaskComplexityService()
            classification = service.classify(task_description)

            # Only use inference if confidence is reasonable
            if classification.confidence >= 0.5:
                logger.info(
                    f"Task classified as {classification.complexity.value} "
                    f"(budget: {classification.tool_budget}, confidence: {classification.confidence:.2f})"
                )
                return classification.complexity.value
            else:
                logger.info(
                    f"Low confidence classification ({classification.confidence:.2f}), "
                    f"using conservative default"
                )
        except Exception as e:
            logger.warning(f"Complexity inference failed: {e}")

        # Priority 3: Conservative default
        logger.info("Using conservative default complexity: medium")
        return "medium"

    def _inject_task_context(self, task: BenchmarkTask, workspace_dir: Path) -> None:
        """Inject task context into orchestrator's system prompt for enrichment.

        This replaces the gaming approach of _build_task_prompt() with proper
        framework integration. Context is appended to the system prompt using
        the orchestrator's native append_to_system_prompt() method.

        Args:
            task: The benchmark task
            workspace_dir: Working directory for the task
        """
        context_sections = []

        # Add working directory context
        context_sections.append(
            f"## Working Directory\nYou are working in: {workspace_dir}"
        )

        # Add repository context if available
        if task.repo:
            repo_name = task.repo.replace("https://github.com/", "").replace(".git", "")
            context_sections.append(f"**Repository:** {repo_name}")

        # Add hints through the system prompt (framework integration)
        # These become part of the system context, benefiting all verticals
        if task.hints:
            hints_section = "## Hints\n" + "\n".join(f"- {hint}" for hint in task.hints)
            context_sections.append(hints_section)

        # Add context code if provided
        if task.context_code:
            code_section = f"## Context Code\n```python\n{task.context_code}\n```"
            context_sections.append(code_section)

        # Append all context to the orchestrator's system prompt
        if context_sections:
            combined_context = "\n\n".join(context_sections)
            self.orchestrator.append_to_system_prompt(combined_context)
            logger.debug(f"Injected task context: {len(context_sections)} sections")

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
        # Only pass base_url if explicitly set, otherwise let provider use its default
        provider_kwargs = {
            "settings": settings,
            "timeout": timeout,
        }

        # Get API key from profile, environment variable, or keyring
        api_key = getattr(profile_config, "api_key", None)
        # Resolve ${ENV_VAR} references
        if api_key and api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var)

        # If still no API key, try keyring
        if not api_key:
            try:
                from victor.config.api_keys import get_api_key

                api_key = get_api_key(provider_name)
            except ImportError:
                pass

        if api_key:
            provider_kwargs["api_key"] = api_key

        effective_base_url = base_url or getattr(profile_config, "base_url", None)
        if effective_base_url:
            provider_kwargs["base_url"] = effective_base_url

        provider = ProviderRegistry.create(provider_name, **provider_kwargs)

        # Create orchestrator - lazy import to break circular dependency
        # Chain: orchestrator → code_correction_middleware → evaluation → agent_adapter → orchestrator
        from victor.agent.orchestrator import AgentOrchestrator as Orchestrator

        orchestrator = Orchestrator(
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
