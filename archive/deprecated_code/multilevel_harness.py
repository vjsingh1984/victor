# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Multi-level benchmark harness for measuring coding assistant efficiency.

This module provides three levels of benchmark execution to help pinpoint
where issues occur in the coding assistant pipeline:

1. CLI Level (full stack): Runs tasks through actual `victor chat --no-tui`
   subprocess, testing the complete end-to-end experience.

2. Orchestrator Level (agent + tools): Uses AgentOrchestrator directly with
   tools, testing the agent loop and tool execution.

3. Provider Level (raw model): Tests raw LLM capability without tools,
   measuring pure model performance.

Metrics tracked at each level:
- Pass@1 rate
- Turns to completion (multi-turn interactions)
- Tool calls used
- Token efficiency (input/output/total)
- Time to completion
"""

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable, Awaitable

from victor.evaluation.protocol import (
    BenchmarkTask,
    EvaluationConfig,
    TaskResult,
    TaskStatus,
)
from victor.agent.complexity_classifier import (
    ComplexityClassifier,
    TaskComplexity,
)
from victor.agent.prompt_corpus_registry import (
    PromptCorpusRegistry,
)

logger = logging.getLogger(__name__)


class BenchmarkLevel(Enum):
    """Benchmark execution level."""

    PROVIDER = "provider"  # Raw LLM capability
    ORCHESTRATOR = "orchestrator"  # Agent + tools
    CLI = "cli"  # Full stack via subprocess


@dataclass
class LevelMetrics:
    """Metrics specific to multi-level benchmarking."""

    # Pass metrics
    passed: int = 0
    failed: int = 0
    errors: int = 0
    timeouts: int = 0

    # Efficiency metrics
    total_turns: int = 0
    total_tool_calls: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_time_seconds: float = 0.0

    # Per-task details
    task_metrics: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_tasks(self) -> int:
        """Total tasks attempted."""
        return self.passed + self.failed + self.errors + self.timeouts

    @property
    def pass_rate(self) -> float:
        """Pass rate as a decimal."""
        if self.total_tasks == 0:
            return 0.0
        return self.passed / self.total_tasks

    @property
    def avg_turns(self) -> float:
        """Average turns per task."""
        if self.total_tasks == 0:
            return 0.0
        return self.total_turns / self.total_tasks

    @property
    def avg_tool_calls(self) -> float:
        """Average tool calls per task."""
        if self.total_tasks == 0:
            return 0.0
        return self.total_tool_calls / self.total_tasks

    @property
    def avg_tokens(self) -> float:
        """Average total tokens per task."""
        if self.total_tasks == 0:
            return 0.0
        return (self.total_tokens_input + self.total_tokens_output) / self.total_tasks

    @property
    def avg_time(self) -> float:
        """Average time per task in seconds."""
        if self.total_tasks == 0:
            return 0.0
        return self.total_time_seconds / self.total_tasks

    def to_dict(self) -> dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "summary": {
                "total_tasks": self.total_tasks,
                "passed": self.passed,
                "failed": self.failed,
                "errors": self.errors,
                "timeouts": self.timeouts,
                "pass_rate": round(self.pass_rate, 4),
            },
            "efficiency": {
                "total_turns": self.total_turns,
                "avg_turns": round(self.avg_turns, 2),
                "total_tool_calls": self.total_tool_calls,
                "avg_tool_calls": round(self.avg_tool_calls, 2),
                "total_tokens_input": self.total_tokens_input,
                "total_tokens_output": self.total_tokens_output,
                "avg_tokens": round(self.avg_tokens, 2),
                "total_time_seconds": round(self.total_time_seconds, 2),
                "avg_time_seconds": round(self.avg_time, 2),
            },
            "tasks": self.task_metrics,
        }


@dataclass
class TaskExecutionResult:
    """Result from executing a single task at any level."""

    task_id: str
    success: bool
    generated_code: str = ""
    error_message: str = ""

    # Efficiency metrics
    turns: int = 1
    tool_calls: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    duration_seconds: float = 0.0

    # Test results
    tests_passed: int = 0
    tests_total: int = 0

    def to_task_result(self) -> TaskResult:
        """Convert to standard TaskResult."""
        return TaskResult(
            task_id=self.task_id,
            status=TaskStatus.PASSED if self.success else TaskStatus.FAILED,
            generated_code=self.generated_code,
            error_message=self.error_message if not self.success else None,
            tests_passed=self.tests_passed,
            tests_total=self.tests_total,
            duration_seconds=self.duration_seconds,
            tokens_input=self.tokens_input,
            tokens_output=self.tokens_output,
            tool_calls=self.tool_calls,
            turns=self.turns,
        )


class BaseLevelRunner(ABC):
    """Abstract base class for level-specific runners."""

    @property
    @abstractmethod
    def level(self) -> BenchmarkLevel:
        """The benchmark level this runner handles."""
        ...

    @abstractmethod
    async def execute_task(
        self,
        task: BenchmarkTask,
        config: EvaluationConfig,
    ) -> TaskExecutionResult:
        """Execute a single task at this level.

        Args:
            task: The benchmark task
            config: Evaluation configuration

        Returns:
            TaskExecutionResult with metrics
        """
        ...


class ProviderLevelRunner(BaseLevelRunner):
    """Provider-level runner - tests raw LLM capability.

    This is the simplest level, using direct provider calls without
    any agent loop or tool execution.
    """

    def __init__(
        self,
        provider: Any,
        model_name: str,
        timeout: int = 120,
    ):
        """Initialize provider runner.

        Args:
            provider: LLM provider instance
            model_name: Model name to use
            timeout: Request timeout in seconds
        """
        self._provider = provider
        self._model_name = model_name
        self._timeout = timeout

    @property
    def level(self) -> BenchmarkLevel:
        return BenchmarkLevel.PROVIDER

    async def execute_task(
        self,
        task: BenchmarkTask,
        config: EvaluationConfig,
    ) -> TaskExecutionResult:
        """Execute task using direct provider call."""
        from victor.providers.base import Message

        start_time = time.time()

        prompt = f"""You are an expert Python programmer. Complete the following function.
Your response must contain ONLY the complete Python code, starting with any necessary imports.
Do not include any explanations, markdown formatting, or code blocks - just the raw Python code.

{task.prompt}

Complete the function above. Write the implementation that passes all the test cases.
"""

        messages = [Message(role="user", content=prompt)]

        try:
            response = await asyncio.wait_for(
                self._provider.chat(
                    messages=messages,
                    model=self._model_name,
                    temperature=0.2,
                    max_tokens=1024,
                ),
                timeout=self._timeout,
            )

            generated_code = response.content.strip()

            # Clean up markdown code blocks if present
            generated_code = self._clean_code(generated_code)

            duration = time.time() - start_time

            return TaskExecutionResult(
                task_id=task.task_id,
                success=True,  # Will be validated later
                generated_code=generated_code,
                turns=1,
                tool_calls=0,
                tokens_input=getattr(response, "input_tokens", 0),
                tokens_output=getattr(response, "output_tokens", 0),
                duration_seconds=duration,
            )

        except asyncio.TimeoutError:
            return TaskExecutionResult(
                task_id=task.task_id,
                success=False,
                error_message="Timeout",
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            return TaskExecutionResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                duration_seconds=time.time() - start_time,
            )

    def _clean_code(self, code: str) -> str:
        """Clean up markdown code blocks."""
        if "```python" in code:
            match = re.search(r"```python\n(.*?)```", code, re.DOTALL)
            if match:
                return match.group(1).strip()
        elif "```" in code:
            match = re.search(r"```\n(.*?)```", code, re.DOTALL)
            if match:
                return match.group(1).strip()
        return code


class OrchestratorLevelRunner(BaseLevelRunner):
    """Orchestrator-level runner - tests agent + tools.

    Uses AgentOrchestrator with tools enabled, tracking multi-turn
    interactions and tool usage metrics.

    For GENERATION tasks (like HumanEval), bypasses tools and uses
    direct provider calls to avoid tool hallucination issues.
    """

    def __init__(
        self,
        orchestrator: Any,  # AgentOrchestrator
        timeout: int = 300,
        provider: Optional[Any] = None,  # BaseProvider for fallback
        model_name: Optional[str] = None,  # Model name for direct provider calls
    ):
        """Initialize orchestrator runner.

        Args:
            orchestrator: AgentOrchestrator instance
            timeout: Total timeout for task completion
            provider: Optional provider for tools-free generation tasks
            model_name: Model name for direct provider calls (required if provider is set)
        """
        self._orchestrator = orchestrator
        self._timeout = timeout
        self._provider = provider
        self._model_name = model_name or "default"
        self._classifier = ComplexityClassifier()
        self._prompt_registry: Optional[PromptCorpusRegistry] = None

    @property
    def level(self) -> BenchmarkLevel:
        return BenchmarkLevel.ORCHESTRATOR

    async def execute_task(
        self,
        task: BenchmarkTask,
        config: EvaluationConfig,
    ) -> TaskExecutionResult:
        """Execute task using orchestrator with tools."""
        start_time = time.time()

        # Simple user prompt - let orchestrator use its own generic system prompt
        # This simulates how a real user would ask Victor to complete a coding task
        prompt = f"""Complete this Python function:

{task.prompt}

Implement the function to pass the test cases shown in the docstring."""

        # Classify the task to determine if it's a generation task
        classification = self._classifier.classify(prompt)
        is_generation = classification.complexity == TaskComplexity.GENERATION

        turns = 0
        tool_calls = 0
        tokens_input = 0
        tokens_output = 0
        generated_code = ""
        accumulated_content = ""  # Accumulate all streamed content

        try:
            # For GENERATION tasks with a provider available, use direct provider call
            # This avoids tool hallucination issues where models try to read/write files
            if is_generation and self._provider:
                return await self._execute_generation_task(
                    task, prompt, start_time
                )

            # Reset conversation for clean state (if method exists)
            if hasattr(self._orchestrator, 'clear_history'):
                self._orchestrator.clear_history()
            elif hasattr(self._orchestrator, 'reset'):
                self._orchestrator.reset()
            elif hasattr(self._orchestrator, '_conversation_controller'):
                # Clear through conversation controller
                if hasattr(self._orchestrator._conversation_controller, 'clear'):
                    self._orchestrator._conversation_controller.clear()

            # Run the orchestrator loop
            async for response in self._orchestrator.stream_chat(prompt):
                turns += 1

                # Track metrics from response
                if hasattr(response, "tool_calls_made"):
                    tool_calls += response.tool_calls_made
                if hasattr(response, "input_tokens"):
                    tokens_input += response.input_tokens
                if hasattr(response, "output_tokens"):
                    tokens_output += response.output_tokens

                # Accumulate content from each chunk
                if hasattr(response, "content") and response.content:
                    accumulated_content += response.content

                # Check if we've exceeded timeout
                if time.time() - start_time > self._timeout:
                    break

            # Extract code from full accumulated content
            if accumulated_content:
                generated_code = self._extract_code(accumulated_content)

            duration = time.time() - start_time

            return TaskExecutionResult(
                task_id=task.task_id,
                success=bool(generated_code),
                generated_code=generated_code,
                turns=max(1, turns),
                tool_calls=tool_calls,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                duration_seconds=duration,
            )

        except asyncio.TimeoutError:
            return TaskExecutionResult(
                task_id=task.task_id,
                success=False,
                error_message="Timeout",
                turns=turns,
                tool_calls=tool_calls,
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            return TaskExecutionResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                turns=turns,
                tool_calls=tool_calls,
                duration_seconds=time.time() - start_time,
            )

    def _get_prompt_registry(self) -> PromptCorpusRegistry:
        """Get or initialize the prompt corpus registry (lazy init)."""
        if self._prompt_registry is None:
            self._prompt_registry = PromptCorpusRegistry()
        return self._prompt_registry

    async def _execute_generation_task(
        self,
        task: BenchmarkTask,
        prompt: str,
        start_time: float,
    ) -> TaskExecutionResult:
        """Execute a generation task using direct provider call (no tools).

        This method handles code generation tasks like HumanEval by bypassing
        the orchestrator's tool infrastructure, preventing tool hallucination.

        Uses the PromptCorpusRegistry to select category-specific system prompts
        based on embedding similarity to the corpus of benchmark prompts.
        """
        from victor.providers.base import Message

        # Use corpus-based enriched prompt for category-specific guidance
        registry = self._get_prompt_registry()
        enriched = registry.build_prompt(prompt)

        # Log the matched category for debugging
        logger.debug(
            f"Task {task.task_id}: matched category {enriched.category.value}, "
            f"hints: {enriched.hints}"
        )

        messages = [
            Message(role="system", content=enriched.system_prompt),
            Message(role="user", content=prompt),
        ]

        tokens_input = 0
        tokens_output = 0

        try:
            # Use non-streaming chat for simplicity (no tools)
            response = await asyncio.wait_for(
                self._provider.chat(
                    messages=messages,
                    model=self._model_name,
                    temperature=0.2,
                    max_tokens=1024,
                    tools=None,  # No tools for generation tasks
                ),
                timeout=self._timeout,
            )

            generated_content = response.content.strip()

            # Extract token usage if available
            if hasattr(response, "usage") and response.usage:
                tokens_input = response.usage.get("prompt_tokens", 0)
                tokens_output = response.usage.get("completion_tokens", 0)

            duration = time.time() - start_time
            generated_code = self._extract_code(generated_content)

            return TaskExecutionResult(
                task_id=task.task_id,
                success=bool(generated_code),
                generated_code=generated_code,
                turns=1,  # Single turn for generation
                tool_calls=0,  # No tools used
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                duration_seconds=duration,
            )

        except asyncio.TimeoutError:
            return TaskExecutionResult(
                task_id=task.task_id,
                success=False,
                error_message="Timeout",
                turns=1,
                tool_calls=0,
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            return TaskExecutionResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                turns=1,
                tool_calls=0,
                duration_seconds=time.time() - start_time,
            )

    def _extract_code(self, content: str) -> str:
        """Extract Python code from response content."""
        # Try to find code block
        if "```python" in content:
            match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
            if match:
                return match.group(1).strip()
        elif "```" in content:
            match = re.search(r"```\n(.*?)```", content, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Look for def statements
        if "def " in content:
            # Find the function definition
            lines = content.split("\n")
            code_lines = []
            in_function = False
            indent = 0

            for line in lines:
                if line.strip().startswith("def "):
                    in_function = True
                    indent = len(line) - len(line.lstrip())
                    code_lines.append(line)
                elif in_function:
                    if line.strip() == "":
                        code_lines.append(line)
                    elif len(line) - len(line.lstrip()) <= indent and line.strip():
                        # End of function
                        break
                    else:
                        code_lines.append(line)

            if code_lines:
                return "\n".join(code_lines)

        return content


class CLILevelRunner(BaseLevelRunner):
    """CLI-level runner - tests full stack via subprocess.

    Runs tasks through `victor chat --no-tui` subprocess, testing
    the complete end-to-end experience including CLI parsing,
    profile loading, provider initialization, and agent execution.

    Uses the --code-only flag for clean code extraction in automation mode.
    """

    def __init__(
        self,
        profile: str = "default",
        timeout: int = 300,
        victor_path: Optional[str] = None,
        use_json_mode: bool = False,
    ):
        """Initialize CLI runner.

        Args:
            profile: Profile name from profiles.yaml
            timeout: Total timeout for CLI execution
            victor_path: Optional path to victor CLI (defaults to 'victor')
            use_json_mode: If True, use --json for structured output with metrics.
                          If False (default), use --code-only for simpler extraction.
        """
        self._profile = profile
        self._timeout = timeout
        self._victor_path = victor_path or "python -m victor.ui.cli"
        self._use_json_mode = use_json_mode

    @property
    def level(self) -> BenchmarkLevel:
        return BenchmarkLevel.CLI

    async def execute_task(
        self,
        task: BenchmarkTask,
        config: EvaluationConfig,
    ) -> TaskExecutionResult:
        """Execute task via CLI subprocess.

        Uses --code-only flag for clean code extraction without Rich formatting.
        Alternatively uses --json for structured output with metrics.
        """
        start_time = time.time()

        # Build prompt for code generation
        prompt_lines = task.prompt.strip().split("\n")
        compact_prompt = " ".join(line.strip() for line in prompt_lines if line.strip())

        # Enhanced prompt for benchmark automation - explicit instructions for clean code output
        # NOTE: This prompt ONLY affects benchmark harness, NOT interactive user experience
        full_prompt = (
            f"IMPORTANT: This is a code-only task. DO NOT call any tools. "
            f"DO NOT use markdown code blocks. Return ONLY the raw Python function. "
            f"Complete this Python function: {compact_prompt} "
            f"Output ONLY the function definition starting with 'def'. No explanations."
        )

        try:
            # Build CLI command with automation flags
            cmd = [
                "python", "-m", "victor.ui.cli",
                "chat",
                "--profile", self._profile,
                "--no-stream",  # Don't stream for cleaner output
                "--quiet",  # Suppress status messages
                "--log-level", "ERROR",  # Suppress logging noise
            ]

            # Choose output mode
            if self._use_json_mode:
                cmd.append("--json")
            else:
                cmd.append("--code-only")

            # Add the prompt as argument
            cmd.append(full_prompt)

            # Run victor CLI
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                return TaskExecutionResult(
                    task_id=task.task_id,
                    success=False,
                    error_message="CLI timeout",
                    duration_seconds=time.time() - start_time,
                )

            duration = time.time() - start_time
            output = stdout.decode()

            # Parse output based on mode
            if self._use_json_mode:
                return self._parse_json_output(task.task_id, output, duration)
            else:
                # Code-only mode: output IS the code
                generated_code = output.strip()
                return TaskExecutionResult(
                    task_id=task.task_id,
                    success=bool(generated_code),
                    generated_code=generated_code,
                    turns=1,
                    tool_calls=0,
                    tokens_input=0,
                    tokens_output=0,
                    duration_seconds=duration,
                )

        except Exception as e:
            return TaskExecutionResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                duration_seconds=time.time() - start_time,
            )

    def _parse_json_output(
        self,
        task_id: str,
        output: str,
        duration: float,
    ) -> TaskExecutionResult:
        """Parse JSON mode output for structured results with metrics."""
        try:
            data = json.loads(output)
            content = data.get("content", "")
            metrics = data.get("metrics", {})
            usage = metrics.get("usage", {})

            # Extract code from content (may still have markdown)
            generated_code = self._extract_code_from_content(content)

            return TaskExecutionResult(
                task_id=task_id,
                success=bool(generated_code),
                generated_code=generated_code,
                turns=1,
                tool_calls=0,
                tokens_input=usage.get("prompt_tokens", 0),
                tokens_output=usage.get("completion_tokens", 0),
                duration_seconds=duration,
            )
        except json.JSONDecodeError:
            # Fallback: treat as plain text
            return TaskExecutionResult(
                task_id=task_id,
                success=bool(output.strip()),
                generated_code=output.strip(),
                turns=1,
                duration_seconds=duration,
            )

    def _extract_code_from_content(self, content: str) -> str:
        """Extract code from JSON content field."""
        # Try markdown code blocks
        if "```python" in content:
            match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
            if match:
                return match.group(1).strip()
        elif "```" in content:
            match = re.search(r"```\n(.*?)```", content, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Look for def statements
        if "def " in content:
            return self._extract_function_from_text(content)

        return content.strip()

    def _extract_function_from_text(self, text: str) -> str:
        """Extract Python function definitions from raw text."""
        lines = text.split("\n")
        code_lines = []
        in_function = False
        func_indent = 0

        for line in lines:
            # Skip empty lines before function
            if not in_function and not line.strip():
                continue

            # Look for function definition
            if "def " in line and "(" in line:
                in_function = True
                # Get the indentation of def
                func_indent = len(line) - len(line.lstrip())
                code_lines = [line]  # Reset to capture latest function
                continue

            if in_function:
                if not line.strip():
                    code_lines.append(line)
                elif line.lstrip().startswith("#"):
                    code_lines.append(line)
                else:
                    curr_indent = len(line) - len(line.lstrip())
                    # End of function if we're back to func_indent or less
                    # (and it's a non-empty line)
                    if curr_indent <= func_indent and line.strip() and not line.strip().startswith("def "):
                        break
                    code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines)

        return ""

    def _parse_metrics(self, output: str) -> dict[str, int]:
        """Parse metrics from CLI output."""
        metrics = {"turns": 1, "tool_calls": 0, "tokens_input": 0, "tokens_output": 0}

        # Look for metrics in output (Victor typically logs these)
        if "tool calls:" in output.lower():
            match = re.search(r"tool calls:\s*(\d+)", output.lower())
            if match:
                metrics["tool_calls"] = int(match.group(1))

        if "tokens:" in output.lower():
            match = re.search(r"(\d+)\s*input.*?(\d+)\s*output", output.lower())
            if match:
                metrics["tokens_input"] = int(match.group(1))
                metrics["tokens_output"] = int(match.group(2))

        return metrics


class MultiLevelBenchmark:
    """Orchestrates benchmarks across multiple levels.

    Runs the same tasks at different levels and compares results
    to identify where issues occur in the pipeline.
    """

    def __init__(
        self,
        levels: Optional[list[BenchmarkLevel]] = None,
    ):
        """Initialize multi-level benchmark.

        Args:
            levels: List of levels to run (defaults to all)
        """
        self.levels = levels or [
            BenchmarkLevel.PROVIDER,
            BenchmarkLevel.ORCHESTRATOR,
            BenchmarkLevel.CLI,
        ]
        self._runners: dict[BenchmarkLevel, BaseLevelRunner] = {}
        self._results: dict[BenchmarkLevel, LevelMetrics] = {}

    def register_runner(
        self,
        runner: BaseLevelRunner,
    ) -> None:
        """Register a level runner.

        Args:
            runner: The runner to register
        """
        self._runners[runner.level] = runner

    async def run_benchmark(
        self,
        tasks: list[BenchmarkTask],
        test_callback: Callable[
            [BenchmarkTask, str], Awaitable[tuple[bool, int, int]]
        ],
        config: EvaluationConfig,
        progress_callback: Optional[
            Callable[[BenchmarkLevel, int, int, TaskExecutionResult], None]
        ] = None,
    ) -> dict[BenchmarkLevel, LevelMetrics]:
        """Run benchmark across all registered levels.

        Args:
            tasks: List of benchmark tasks
            test_callback: Async callback to test generated code.
                          Signature: (task, code) -> (passed, tests_passed, tests_total)
            config: Evaluation configuration
            progress_callback: Optional callback for progress updates.
                              Signature: (level, task_idx, total, result) -> None

        Returns:
            Dict mapping levels to their metrics
        """
        self._results = {}

        for level in self.levels:
            if level not in self._runners:
                logger.warning(f"No runner registered for level: {level.value}")
                continue

            runner = self._runners[level]
            metrics = LevelMetrics()

            logger.info(f"Running {level.value} level benchmark...")

            for i, task in enumerate(tasks):
                try:
                    # Execute task at this level
                    result = await runner.execute_task(task, config)

                    # Test the generated code
                    if result.generated_code:
                        passed, tests_passed, tests_total = await test_callback(
                            task, result.generated_code
                        )
                        result.success = passed
                        result.tests_passed = tests_passed
                        result.tests_total = tests_total

                    # Update metrics
                    if result.success:
                        metrics.passed += 1
                    elif result.error_message == "Timeout":
                        metrics.timeouts += 1
                    elif result.error_message:
                        metrics.errors += 1
                    else:
                        metrics.failed += 1

                    metrics.total_turns += result.turns
                    metrics.total_tool_calls += result.tool_calls
                    metrics.total_tokens_input += result.tokens_input
                    metrics.total_tokens_output += result.tokens_output
                    metrics.total_time_seconds += result.duration_seconds

                    metrics.task_metrics.append({
                        "task_id": result.task_id,
                        "success": result.success,
                        "turns": result.turns,
                        "tool_calls": result.tool_calls,
                        "tokens": result.tokens_input + result.tokens_output,
                        "duration": round(result.duration_seconds, 2),
                        "tests_passed": result.tests_passed,
                        "tests_total": result.tests_total,
                    })

                    # Progress callback
                    if progress_callback:
                        progress_callback(level, i, len(tasks), result)

                except Exception as e:
                    logger.error(f"Error running task {task.task_id}: {e}")
                    metrics.errors += 1

            self._results[level] = metrics

        return self._results

    def compare_levels(self) -> dict[str, Any]:
        """Compare results across levels.

        Returns:
            Comparison report showing where issues occur
        """
        if not self._results:
            return {"error": "No results to compare"}

        comparison = {
            "levels": {},
            "analysis": {
                "pass_rate_diff": {},
                "efficiency_impact": {},
                "bottleneck": None,
            },
        }

        # Extract metrics per level
        for level, metrics in self._results.items():
            comparison["levels"][level.value] = {
                "pass_rate": round(metrics.pass_rate, 4),
                "avg_turns": round(metrics.avg_turns, 2),
                "avg_tool_calls": round(metrics.avg_tool_calls, 2),
                "avg_tokens": round(metrics.avg_tokens, 2),
                "avg_time": round(metrics.avg_time, 2),
            }

        # Analyze differences
        levels = list(self._results.keys())
        if len(levels) >= 2:
            for i in range(1, len(levels)):
                prev_level = levels[i - 1]
                curr_level = levels[i]

                prev_metrics = self._results[prev_level]
                curr_metrics = self._results[curr_level]

                diff = curr_metrics.pass_rate - prev_metrics.pass_rate
                comparison["analysis"]["pass_rate_diff"][
                    f"{prev_level.value}_to_{curr_level.value}"
                ] = round(diff, 4)

                # Check for significant degradation
                if diff < -0.1:  # More than 10% drop
                    comparison["analysis"]["bottleneck"] = (
                        f"Significant degradation from {prev_level.value} to "
                        f"{curr_level.value}: {diff:.1%} pass rate drop"
                    )

        return comparison

    def generate_report(self) -> str:
        """Generate human-readable comparison report."""
        if not self._results:
            return "No results to report"

        lines = []
        lines.append("=" * 70)
        lines.append("       MULTI-LEVEL BENCHMARK COMPARISON REPORT")
        lines.append("=" * 70)
        lines.append("")

        for level, metrics in self._results.items():
            lines.append(f"LEVEL: {level.value.upper()}")
            lines.append("-" * 40)
            lines.append(f"  Pass Rate:      {metrics.pass_rate:.1%}")
            lines.append(f"  Passed/Total:   {metrics.passed}/{metrics.total_tasks}")
            lines.append(f"  Errors:         {metrics.errors}")
            lines.append(f"  Timeouts:       {metrics.timeouts}")
            lines.append(f"  Avg Turns:      {metrics.avg_turns:.1f}")
            lines.append(f"  Avg Tool Calls: {metrics.avg_tool_calls:.1f}")
            lines.append(f"  Avg Tokens:     {metrics.avg_tokens:.0f}")
            lines.append(f"  Avg Time:       {metrics.avg_time:.1f}s")
            lines.append("")

        # Analysis
        comparison = self.compare_levels()
        if comparison.get("analysis", {}).get("bottleneck"):
            lines.append("ANALYSIS")
            lines.append("-" * 40)
            lines.append(f"  {comparison['analysis']['bottleneck']}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


def create_provider_runner(
    profile: str,
    base_url: Optional[str] = None,
    model_override: Optional[str] = None,
    timeout: int = 120,
) -> ProviderLevelRunner:
    """Create a provider-level runner from profile.

    Args:
        profile: Profile name from profiles.yaml
        base_url: Override base URL
        model_override: Override model name
        timeout: Request timeout

    Returns:
        ProviderLevelRunner instance
    """
    from victor.config.settings import load_settings
    from victor.providers.registry import ProviderRegistry

    settings = load_settings()
    profiles = settings.load_profiles()

    if profile not in profiles:
        raise ValueError(f"Profile '{profile}' not found")

    profile_config = profiles[profile]
    model_name = model_override or profile_config.model
    provider_settings = settings.get_provider_settings(profile_config.provider)

    if base_url:
        provider_settings["base_url"] = base_url

    provider = ProviderRegistry.create(
        profile_config.provider,
        model=model_name,
        **provider_settings,
    )

    return ProviderLevelRunner(provider, model_name, timeout)


def create_orchestrator_runner(
    profile: str,
    base_url: Optional[str] = None,
    model_override: Optional[str] = None,
    timeout: int = 300,
) -> OrchestratorLevelRunner:
    """Create an orchestrator-level runner from profile.

    Args:
        profile: Profile name from profiles.yaml
        base_url: Override base URL
        model_override: Override model name
        timeout: Total timeout

    Returns:
        OrchestratorLevelRunner instance
    """
    from victor.config.settings import load_settings
    from victor.providers.registry import ProviderRegistry
    from victor.agent.orchestrator import AgentOrchestrator

    settings = load_settings()
    profiles = settings.load_profiles()

    if profile not in profiles:
        raise ValueError(f"Profile '{profile}' not found")

    profile_config = profiles[profile]
    model_name = model_override or profile_config.model
    provider_settings = settings.get_provider_settings(profile_config.provider)

    if base_url:
        provider_settings["base_url"] = base_url

    provider = ProviderRegistry.create(
        profile_config.provider,
        model=model_name,
        **provider_settings,
    )

    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=provider,
        model=model_name,
    )

    # Pass provider and model_name for tools-free generation task fallback
    return OrchestratorLevelRunner(
        orchestrator, timeout, provider=provider, model_name=model_name
    )


def create_cli_runner(
    profile: str = "default",
    timeout: int = 300,
) -> CLILevelRunner:
    """Create a CLI-level runner.

    Args:
        profile: Profile name
        timeout: CLI timeout

    Returns:
        CLILevelRunner instance
    """
    return CLILevelRunner(profile=profile, timeout=timeout)
