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

"""Evaluation harness for running benchmarks.

Provides infrastructure for evaluating coding agents against
standardized benchmarks like SWE-bench, HumanEval, etc.
"""

import asyncio
import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from victor.evaluation.protocol import (
    BenchmarkTask,
    BenchmarkType,
    EvaluationConfig,
    EvaluationResult,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class BenchmarkRunner(Protocol):
    """Protocol for benchmark runners."""

    @property
    def benchmark_type(self) -> BenchmarkType:
        """The type of benchmark this runner handles."""
        ...

    async def load_tasks(
        self,
        config: EvaluationConfig,
    ) -> list[BenchmarkTask]:
        """Load benchmark tasks."""
        ...

    async def run_task(
        self,
        task: BenchmarkTask,
        agent_output: str,
        config: EvaluationConfig,
    ) -> TaskResult:
        """Run a single task and evaluate the result."""
        ...


class BaseBenchmarkRunner(ABC):
    """Abstract base class for benchmark runners."""

    @property
    @abstractmethod
    def benchmark_type(self) -> BenchmarkType:
        """The type of benchmark this runner handles."""
        ...

    @abstractmethod
    async def load_tasks(
        self,
        config: EvaluationConfig,
    ) -> list[BenchmarkTask]:
        """Load benchmark tasks."""
        ...

    @abstractmethod
    async def run_task(
        self,
        task: BenchmarkTask,
        agent_output: str,
        config: EvaluationConfig,
    ) -> TaskResult:
        """Run a single task and evaluate the result."""
        ...

    def _filter_tasks(
        self,
        tasks: list[BenchmarkTask],
        config: EvaluationConfig,
    ) -> list[BenchmarkTask]:
        """Filter tasks based on config."""
        filtered = tasks

        if config.task_ids:
            filtered = [t for t in filtered if t.task_id in config.task_ids]

        if config.languages:
            filtered = [t for t in filtered if t.language in config.languages]

        if config.categories:
            filtered = [t for t in filtered if t.category in config.categories]

        if config.difficulties:
            filtered = [t for t in filtered if t.difficulty in config.difficulties]

        if config.max_tasks:
            filtered = filtered[: config.max_tasks]

        return filtered


class TaskEnvironment:
    """Isolated environment for running benchmark tasks."""

    def __init__(
        self,
        task: BenchmarkTask,
        workspace_dir: Optional[Path] = None,
        use_docker: bool = False,
        docker_image: str = "python:3.11",
    ):
        """Initialize the task environment.

        Args:
            task: The benchmark task
            workspace_dir: Base directory for workspaces
            use_docker: Whether to use Docker isolation
            docker_image: Docker image to use
        """
        self.task = task
        self.workspace_dir = workspace_dir or Path(tempfile.gettempdir())
        self.use_docker = use_docker
        self.docker_image = docker_image
        self._temp_dir: Optional[Path] = None

    async def setup(self) -> Path:
        """Set up the task environment.

        Returns:
            Path to the workspace directory
        """
        # Create temporary directory (sanitize task_id to be valid for directory names)
        safe_task_id = self.task.task_id.replace("/", "_").replace("\\", "_")
        self._temp_dir = Path(
            tempfile.mkdtemp(
                prefix=f"eval_{safe_task_id}_",
                dir=self.workspace_dir,
            )
        )

        # Clone repo if specified (for SWE-bench)
        if self.task.repo:
            await self._clone_repo()

        # Write context code
        if self.task.context_code:
            code_file = self._temp_dir / "solution.py"
            code_file.write_text(self.task.context_code)

        # Write test code
        if self.task.test_code:
            test_file = self._temp_dir / "test_solution.py"
            test_file.write_text(self.task.test_code)

        return self._temp_dir

    async def _clone_repo(self) -> None:
        """Clone the task's repository.

        For SWE-bench tasks, we need to clone the full repo (or enough history)
        to be able to checkout the specific base_commit.
        """
        if not self.task.repo or not self._temp_dir:
            return

        repo_dir = self._temp_dir / "repo"

        # Clone with enough history to access base_commit
        # Using --depth 100 as compromise between speed and history
        clone_cmd = ["git", "clone", "--depth", "100", self.task.repo, str(repo_dir)]

        try:
            logger.info(f"Cloning {self.task.repo}...")
            result = await asyncio.create_subprocess_exec(
                *clone_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                # Try full clone if shallow clone failed
                logger.warning(f"Shallow clone failed, trying full clone: {stderr.decode()}")
                clone_cmd = ["git", "clone", self.task.repo, str(repo_dir)]
                result = await asyncio.create_subprocess_exec(
                    *clone_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await result.communicate()

            # Checkout specific commit if provided
            if self.task.base_commit and repo_dir.exists():
                # Fetch the specific commit if not available
                fetch_cmd = ["git", "fetch", "--depth", "1", "origin", self.task.base_commit]
                result = await asyncio.create_subprocess_exec(
                    *fetch_cmd,
                    cwd=repo_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await result.communicate()

                # Checkout the commit
                checkout_cmd = ["git", "checkout", self.task.base_commit]
                result = await asyncio.create_subprocess_exec(
                    *checkout_cmd,
                    cwd=repo_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await result.communicate()
                if result.returncode != 0:
                    logger.warning(f"Failed to checkout {self.task.base_commit}: {stderr.decode()}")
                else:
                    logger.info(f"Checked out {self.task.base_commit}")

        except Exception as e:
            logger.warning(f"Failed to clone repo: {e}")

    async def apply_patch(self, patch: str) -> bool:
        """Apply a patch to the repository.

        Args:
            patch: The patch content

        Returns:
            True if patch applied successfully
        """
        if not self._temp_dir:
            return False

        patch_file = self._temp_dir / "solution.patch"
        patch_file.write_text(patch)

        try:
            result = await asyncio.create_subprocess_exec(
                "git",
                "apply",
                str(patch_file),
                cwd=(
                    self._temp_dir / "repo"
                    if (self._temp_dir / "repo").exists()
                    else self._temp_dir
                ),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await result.wait()
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Failed to apply patch: {e}")
            return False

    async def run_tests(self, timeout: int = 300) -> tuple[int, int, str, str]:
        """Run tests in the environment.

        Args:
            timeout: Timeout in seconds

        Returns:
            Tuple of (passed, total, stdout, stderr)
        """
        if not self._temp_dir:
            return 0, 0, "", "Environment not set up"

        test_dir = self._temp_dir / "repo" if (self._temp_dir / "repo").exists() else self._temp_dir

        if self.use_docker:
            return await self._run_tests_docker(test_dir, timeout)
        else:
            return await self._run_tests_local(test_dir, timeout)

    async def _run_tests_local(
        self,
        test_dir: Path,
        timeout: int,
    ) -> tuple[int, int, str, str]:
        """Run tests locally."""
        try:
            # Detect test framework
            if (test_dir / "pytest.ini").exists() or (test_dir / "setup.py").exists():
                cmd = ["python", "-m", "pytest", "-v", "--tb=short"]
            else:
                cmd = ["python", "-m", "unittest", "discover", "-v"]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=test_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    result.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                result.kill()
                return 0, 0, "", "Timeout"

            stdout_str = stdout.decode()
            stderr_str = stderr.decode()

            # Parse test results
            passed, total = self._parse_test_output(stdout_str + stderr_str)

            return passed, total, stdout_str, stderr_str

        except Exception as e:
            return 0, 0, "", str(e)

    async def _run_tests_docker(
        self,
        test_dir: Path,
        timeout: int,
    ) -> tuple[int, int, str, str]:
        """Run tests in Docker container."""
        try:
            cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{test_dir}:/workspace",
                "-w",
                "/workspace",
                "--network",
                "none",
                self.docker_image,
                "python",
                "-m",
                "pytest",
                "-v",
                "--tb=short",
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    result.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                # Kill container
                await asyncio.create_subprocess_exec(
                    "docker",
                    "kill",
                    str(result.pid),
                )
                return 0, 0, "", "Timeout"

            stdout_str = stdout.decode()
            stderr_str = stderr.decode()
            passed, total = self._parse_test_output(stdout_str + stderr_str)

            return passed, total, stdout_str, stderr_str

        except Exception as e:
            return 0, 0, "", str(e)

    def _parse_test_output(self, output: str) -> tuple[int, int]:
        """Parse test output to extract pass/fail counts."""
        import re

        # pytest format: "5 passed, 2 failed"
        pytest_match = re.search(r"(\d+) passed", output)
        pytest_failed = re.search(r"(\d+) failed", output)

        if pytest_match:
            passed = int(pytest_match.group(1))
            failed = int(pytest_failed.group(1)) if pytest_failed else 0
            return passed, passed + failed

        # unittest format: "Ran X tests"
        unittest_match = re.search(r"Ran (\d+) test", output)
        if unittest_match:
            total = int(unittest_match.group(1))
            failures = re.search(r"failures=(\d+)", output)
            errors = re.search(r"errors=(\d+)", output)
            failed = (int(failures.group(1)) if failures else 0) + (
                int(errors.group(1)) if errors else 0
            )
            return total - failed, total

        return 0, 0

    async def cleanup(self) -> None:
        """Clean up the task environment."""
        if self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup: {e}")


class EvaluationHarness:
    """Main harness for running evaluations."""

    def __init__(
        self,
        runners: Optional[dict[BenchmarkType, BaseBenchmarkRunner]] = None,
    ):
        """Initialize the harness.

        Args:
            runners: Dict mapping benchmark types to runners
        """
        self._runners = runners or {}
        try:
            from victor.config.secure_paths import get_victor_dir

            self._results_dir = get_victor_dir() / "evaluations"
        except ImportError:
            self._results_dir = Path.home() / ".victor" / "evaluations"
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def register_runner(self, runner: BaseBenchmarkRunner) -> None:
        """Register a benchmark runner.

        Args:
            runner: The runner to register
        """
        self._runners[runner.benchmark_type] = runner

    def get_runner(self, benchmark_type: BenchmarkType) -> Optional[BaseBenchmarkRunner]:
        """Get a runner by benchmark type.

        Args:
            benchmark_type: The benchmark type

        Returns:
            The runner or None
        """
        return self._runners.get(benchmark_type)

    async def run_evaluation(
        self,
        config: EvaluationConfig,
        agent_callback: Any,  # Callable that takes task and returns output
        progress_callback: Optional[Any] = None,  # Callable(task_idx, total, TaskResult)
        retry_callback: Optional[Any] = None,  # Callable for self-correction retries
    ) -> EvaluationResult:
        """Run a complete evaluation.

        Args:
            config: Evaluation configuration
            agent_callback: Async function that runs the agent on a task
            progress_callback: Optional callback called after each task completes.
                              Signature: (task_index: int, total_tasks: int, result: TaskResult) -> None
            retry_callback: Optional callback for self-correction retries.
                           Signature: (task, previous_code, feedback_prompt) -> str
                           Required if config.enable_self_correction is True.

        Returns:
            EvaluationResult with all task results
        """
        runner = self.get_runner(config.benchmark)
        if runner is None:
            raise ValueError(f"No runner for benchmark: {config.benchmark}")

        # Warn if self-correction enabled but no retry_callback
        if config.enable_self_correction and retry_callback is None:
            logger.warning(
                "Self-correction enabled but no retry_callback provided. "
                "Auto-fix will work but LLM retries will be skipped."
            )

        # Create metrics collector for self-correction tracking
        metrics_collector = None
        if config.enable_self_correction:
            from victor.evaluation.correction import CorrectionMetricsCollector

            metrics_collector = CorrectionMetricsCollector()

        result = EvaluationResult(config=config)
        result.start_time = datetime.now()

        # Load tasks
        tasks = await runner.load_tasks(config)
        logger.info(f"Loaded {len(tasks)} tasks for {config.benchmark.value}")

        # Run tasks
        if config.parallel_tasks > 1:
            results = await self._run_parallel(
                tasks,
                runner,
                agent_callback,
                config,
                progress_callback,
                retry_callback,
                metrics_collector,
            )
        else:
            results = await self._run_sequential(
                tasks,
                runner,
                agent_callback,
                config,
                progress_callback,
                retry_callback,
                metrics_collector,
            )

        result.task_results = results
        result.end_time = datetime.now()

        # Add correction metrics to result
        if metrics_collector:
            result.correction_metrics = metrics_collector.metrics.to_dict()
            logger.info(
                f"Self-correction metrics: {metrics_collector.metrics.correction_success_rate:.1%} success rate"
            )

        # Save results
        self._save_results(result)

        return result

    async def _run_sequential(
        self,
        tasks: list[BenchmarkTask],
        runner: BaseBenchmarkRunner,
        agent_callback: Any,
        config: EvaluationConfig,
        progress_callback: Optional[Any] = None,
        retry_callback: Optional[Any] = None,
        metrics_collector: Optional[Any] = None,
    ) -> list[TaskResult]:
        """Run tasks sequentially."""
        results = []

        for i, task in enumerate(tasks):
            logger.info(f"Running task {i + 1}/{len(tasks)}: {task.task_id}")

            try:
                task_result = await self._run_single_task(
                    task, runner, agent_callback, config, retry_callback, metrics_collector
                )
                results.append(task_result)

                status_str = "PASS" if task_result.is_success else "FAIL"
                logger.info(f"  Result: {status_str}")

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(i, len(tasks), task_result)

            except Exception as e:
                logger.error(f"  Error: {e}")
                error_result = TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.ERROR,
                    error_message=str(e),
                )
                results.append(error_result)

                # Call progress callback for errors too
                if progress_callback:
                    progress_callback(i, len(tasks), error_result)

        return results

    async def _run_parallel(
        self,
        tasks: list[BenchmarkTask],
        runner: BaseBenchmarkRunner,
        agent_callback: Any,
        config: EvaluationConfig,
        progress_callback: Optional[Any] = None,
        retry_callback: Optional[Any] = None,
        metrics_collector: Optional[Any] = None,
    ) -> list[TaskResult]:
        """Run tasks in parallel."""
        semaphore = asyncio.Semaphore(config.parallel_tasks)
        completed_count = 0
        lock = asyncio.Lock()

        async def run_with_semaphore(idx: int, task: BenchmarkTask) -> TaskResult:
            nonlocal completed_count
            async with semaphore:
                try:
                    result = await self._run_single_task(
                        task, runner, agent_callback, config, retry_callback, metrics_collector
                    )
                except Exception as e:
                    result = TaskResult(
                        task_id=task.task_id,
                        status=TaskStatus.ERROR,
                        error_message=str(e),
                    )

                # Call progress callback with lock to ensure ordered output
                if progress_callback:
                    async with lock:
                        completed_count += 1
                        progress_callback(completed_count - 1, len(tasks), result)

                return result

        results = await asyncio.gather(
            *[run_with_semaphore(i, task) for i, task in enumerate(tasks)]
        )
        return list(results)

    async def _run_single_task(
        self,
        task: BenchmarkTask,
        runner: BaseBenchmarkRunner,
        agent_callback: Any,
        config: EvaluationConfig,
        retry_callback: Any = None,
        metrics_collector: Any = None,
    ) -> TaskResult:
        """Run a single task with optional self-correction.

        Args:
            task: The benchmark task
            runner: The benchmark runner
            agent_callback: Callback to generate initial output
            config: Evaluation configuration
            retry_callback: Optional callback for self-correction retries.
                           Signature: (task, previous_code, feedback_prompt) -> str
            metrics_collector: Optional CorrectionMetricsCollector for tracking metrics
        """
        start_time = datetime.now()
        task_result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            start_time=start_time,
        )

        try:
            # Run agent with timeout
            try:
                agent_output = await asyncio.wait_for(
                    agent_callback(task),
                    timeout=config.timeout_per_task,
                )
            except asyncio.TimeoutError:
                task_result.status = TaskStatus.TIMEOUT
                task_result.error_message = "Agent timeout"
                # Check if callback stored partial data before cancellation
                partial_data = getattr(agent_callback, "_partial_data", None)
                if partial_data:
                    task_result.tokens_input = partial_data.get("tokens_input", 0)
                    task_result.tokens_output = partial_data.get("tokens_output", 0)
                    task_result.tokens_used = partial_data.get("tokens_used", 0)
                    task_result.tool_calls = partial_data.get("tool_calls", 0)
                    task_result.turns = partial_data.get("turns", 0)
                    task_result.generated_code = partial_data.get("code", "")
                    logger.info(
                        f"Task timed out - partial metrics recovered: "
                        f"tool_calls={task_result.tool_calls}, turns={task_result.turns}"
                    )
                else:
                    logger.info("Task timed out - no partial metrics available")
                return task_result

            # Handle dict return type for token tracking (P1 fix)
            # Callbacks can return either:
            # - str: Just the generated code (legacy)
            # - dict: Code plus metrics {code, tokens_input, tokens_output, tokens_used, tool_calls, turns}
            if isinstance(agent_output, dict):
                task_result.tokens_input = agent_output.get("tokens_input", 0)
                task_result.tokens_output = agent_output.get("tokens_output", 0)
                task_result.tokens_used = agent_output.get("tokens_used", 0)
                task_result.tool_calls = agent_output.get("tool_calls", 0)
                task_result.turns = agent_output.get("turns", 0)
                agent_output = agent_output.get("code", "")

            # Self-correction loop (if enabled)
            if config.enable_self_correction:
                agent_output, task_result = await self._run_with_self_correction(
                    task=task,
                    runner=runner,
                    agent_output=agent_output,
                    config=config,
                    retry_callback=retry_callback,
                    task_result=task_result,
                    metrics_collector=metrics_collector,
                )
            else:
                # Standard flow without self-correction
                task_result.generated_code = agent_output

                # Analyze code quality
                try:
                    from victor.evaluation.analyzers import get_code_quality_analyzer

                    analyzer = get_code_quality_analyzer()
                    code_quality = await analyzer.analyze(
                        agent_output,
                        language=task.language,
                    )
                    task_result.code_quality = code_quality
                except Exception as e:
                    logger.warning(f"Code quality analysis failed: {e}")

                # Evaluate result
                eval_result = await runner.run_task(task, agent_output, config)

                # Merge results
                task_result.status = eval_result.status
                task_result.tests_passed = eval_result.tests_passed
                task_result.tests_failed = eval_result.tests_failed
                task_result.tests_total = eval_result.tests_total
                task_result.stdout = eval_result.stdout
                task_result.stderr = eval_result.stderr

                # Calculate completion score
                task_result.completion_score = task_result.calculate_completion_score()

        except Exception as e:
            task_result.status = TaskStatus.ERROR
            task_result.error_message = str(e)
            import traceback

            task_result.traceback = traceback.format_exc()

        task_result.end_time = datetime.now()
        task_result.duration_seconds = (task_result.end_time - start_time).total_seconds()

        return task_result

    async def _run_with_self_correction(
        self,
        task: BenchmarkTask,
        runner: BaseBenchmarkRunner,
        agent_output: str,
        config: EvaluationConfig,
        retry_callback: Any,
        task_result: TaskResult,
        metrics_collector: Any = None,
    ) -> tuple[str, TaskResult]:
        """Run task with self-correction loop.

        This implements generic iterative refinement:
        1. Validate code (syntax, imports)
        2. Auto-fix common issues
        3. Run tests
        4. If failed, generate feedback and retry

        This is NOT task-specific - it works for any code generation.

        Args:
            metrics_collector: Optional CorrectionMetricsCollector for tracking metrics
        """
        from victor.evaluation.correction import (
            create_self_corrector,
            detect_language,
        )

        corrector = create_self_corrector(
            max_iterations=config.self_correction_max_iterations,
            auto_fix=config.auto_fix_imports,
        )

        # Detect language for metrics
        lang = detect_language(agent_output, filename=f"solution.{task.language}")

        best_output = agent_output
        best_result = None

        for iteration in range(config.self_correction_max_iterations):
            task_result.attempts = iteration + 1

            # Step 1: Validate and auto-fix
            fixed_code, validation = corrector.validate_and_fix(agent_output)
            agent_output = fixed_code

            # Record validation in metrics
            if metrics_collector:
                metrics_collector.record_validation(lang, validation)

            # Step 2: Analyze code quality
            code_quality = None
            try:
                from victor.evaluation.analyzers import get_code_quality_analyzer

                analyzer = get_code_quality_analyzer()
                code_quality = await analyzer.analyze(
                    agent_output,
                    language=task.language,
                )
            except Exception as e:
                logger.warning(f"Code quality analysis failed: {e}")

            # Step 3: Run tests
            eval_result = await runner.run_task(task, agent_output, config)

            # Track correction attempt in metrics
            if metrics_collector:
                with metrics_collector.track_correction(
                    task_id=task.task_id,
                    language=lang,
                    iteration=iteration + 1,
                    validation_before=validation,
                    test_passed_before=eval_result.tests_passed,
                    test_total=eval_result.tests_total,
                ) as tracker:
                    is_success = eval_result.status == TaskStatus.PASSED
                    tracker.set_result(
                        success=is_success,
                        validation_after=validation,
                        test_passed_after=eval_result.tests_passed,
                        auto_fixed=fixed_code != agent_output,
                    )

            # Track best result
            if best_result is None or eval_result.tests_passed > best_result.tests_passed:
                best_output = agent_output
                best_result = eval_result

            # Check if passed
            if eval_result.status == TaskStatus.PASSED:
                task_result.generated_code = agent_output
                task_result.code_quality = code_quality
                task_result.status = eval_result.status
                task_result.tests_passed = eval_result.tests_passed
                task_result.tests_failed = eval_result.tests_failed
                task_result.tests_total = eval_result.tests_total
                task_result.stdout = eval_result.stdout
                task_result.stderr = eval_result.stderr
                task_result.successful_attempts = 1
                task_result.completion_score = task_result.calculate_completion_score()
                logger.info(f"  Self-correction: PASSED on iteration {iteration + 1}")
                return agent_output, task_result

            # Step 4: Generate feedback for retry
            if iteration < config.self_correction_max_iterations - 1 and retry_callback:
                feedback = corrector.generate_feedback(
                    code=agent_output,
                    validation=validation,
                    test_stdout=eval_result.stdout,
                    test_stderr=eval_result.stderr,
                    test_passed=eval_result.tests_passed,
                    test_total=eval_result.tests_total,
                )

                if feedback.has_issues:
                    retry_prompt = corrector.build_retry_prompt(
                        original_prompt=task.prompt,
                        previous_code=agent_output,
                        feedback=feedback,
                        iteration=iteration + 1,
                    )

                    logger.info(
                        f"  Self-correction: Retry {iteration + 2}/{config.self_correction_max_iterations}"
                    )

                    try:
                        agent_output = await asyncio.wait_for(
                            retry_callback(task, agent_output, retry_prompt),
                            timeout=config.timeout_per_task,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("  Self-correction: Retry timeout")
                        break
                    except Exception as e:
                        logger.warning(f"  Self-correction: Retry failed: {e}")
                        break

        # Use best result
        task_result.generated_code = best_output
        task_result.code_quality = code_quality
        if best_result:
            task_result.status = best_result.status
            task_result.tests_passed = best_result.tests_passed
            task_result.tests_failed = best_result.tests_failed
            task_result.tests_total = best_result.tests_total
            task_result.stdout = best_result.stdout
            task_result.stderr = best_result.stderr
        task_result.completion_score = task_result.calculate_completion_score()

        return best_output, task_result

    def _save_results(self, result: EvaluationResult) -> Path:
        """Save evaluation results to disk."""
        import json

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{result.config.benchmark.value}_{timestamp}.json"
        output_path = self._results_dir / filename

        # Serialize result
        data = {
            "config": {
                "benchmark": result.config.benchmark.value,
                "model": result.config.model,
                "max_tasks": result.config.max_tasks,
                "timeout_per_task": result.config.timeout_per_task,
            },
            "summary": result.get_metrics(),
            "start_time": result.start_time.isoformat() if result.start_time else None,
            "end_time": result.end_time.isoformat() if result.end_time else None,
            "tasks": [
                {
                    "task_id": r.task_id,
                    "status": r.status.value,
                    "tests_passed": r.tests_passed,
                    "tests_total": r.tests_total,
                    "duration_seconds": r.duration_seconds,
                    "tokens_used": r.tokens_used,
                    "tokens_input": r.tokens_input,
                    "tokens_output": r.tokens_output,
                    "tool_calls": r.tool_calls,
                    "turns": r.turns,
                    "completion_score": r.completion_score,
                    "code_quality": (
                        {
                            "syntax_valid": r.code_quality.syntax_valid,
                            "lint_errors": r.code_quality.lint_errors,
                            "lint_warnings": r.code_quality.lint_warnings,
                            "style_score": r.code_quality.style_score,
                            "cyclomatic_complexity": r.code_quality.cyclomatic_complexity,
                            "maintainability_index": r.code_quality.maintainability_index,
                            "type_coverage": r.code_quality.type_coverage,
                            "overall_score": r.code_quality.get_overall_score(),
                        }
                        if r.code_quality
                        else None
                    ),
                    "error_message": r.error_message,
                }
                for r in result.task_results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to: {output_path}")
        return output_path

    def load_results(self, path: Path) -> dict:
        """Load evaluation results from disk."""
        import json

        with open(path) as f:
            return json.load(f)

    def generate_report(
        self,
        result: EvaluationResult,
        format: str = "text",
    ) -> str:
        """Generate an evaluation report.

        Args:
            result: Evaluation result
            format: Report format (text, markdown, json)

        Returns:
            Report string
        """
        if format == "markdown":
            return self._generate_markdown_report(result)
        elif format == "json":
            import json

            return json.dumps(result.get_metrics(), indent=2)
        else:
            return self._generate_text_report(result)

    def _generate_text_report(self, result: EvaluationResult) -> str:
        """Generate text report."""
        lines = []

        lines.append("=" * 70)
        lines.append("EVALUATION REPORT")
        lines.append("=" * 70)
        lines.append("")

        lines.append("CONFIGURATION")
        lines.append("-" * 40)
        lines.append(f"Benchmark: {result.config.benchmark.value}")
        lines.append(f"Model: {result.config.model}")
        lines.append("")

        metrics = result.get_metrics()
        lines.append("RESULTS")
        lines.append("-" * 40)
        lines.append(f"Total Tasks:    {metrics['total_tasks']}")
        lines.append(f"Passed:         {metrics['passed']} ({metrics['pass_rate']:.1%})")
        lines.append(f"Failed:         {metrics['failed']}")
        lines.append(f"Errors:         {metrics['errors']}")
        lines.append(f"Timeouts:       {metrics['timeouts']}")
        lines.append("")

        lines.append("RESOURCES")
        lines.append("-" * 40)
        lines.append(f"Total Duration: {metrics['duration_seconds']:.1f}s")
        lines.append(f"Total Tokens:   {metrics['total_tokens']}")
        lines.append(f"Avg Tokens/Task: {metrics['avg_tokens_per_task']:.0f}")
        lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def _generate_markdown_report(self, result: EvaluationResult) -> str:
        """Generate markdown report."""
        metrics = result.get_metrics()

        lines = []
        lines.append("# Evaluation Report")
        lines.append("")
        lines.append("## Configuration")
        lines.append("")
        lines.append(f"- **Benchmark:** {result.config.benchmark.value}")
        lines.append(f"- **Model:** {result.config.model}")
        lines.append("")

        lines.append("## Results")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Tasks | {metrics['total_tasks']} |")
        lines.append(f"| Pass Rate | {metrics['pass_rate']:.1%} |")
        lines.append(f"| Passed | {metrics['passed']} |")
        lines.append(f"| Failed | {metrics['failed']} |")
        lines.append(f"| Errors | {metrics['errors']} |")
        lines.append(f"| Timeouts | {metrics['timeouts']} |")
        lines.append("")

        lines.append("## Resource Usage")
        lines.append("")
        lines.append(f"- **Duration:** {metrics['duration_seconds']:.1f}s")
        lines.append(f"- **Tokens:** {metrics['total_tokens']}")
        lines.append(f"- **Tool Calls:** {metrics['total_tool_calls']}")
        lines.append("")

        return "\n".join(lines)


# Global harness instance
_harness: Optional[EvaluationHarness] = None


def get_harness() -> EvaluationHarness:
    """Get the global evaluation harness."""
    global _harness
    if _harness is None:
        _harness = EvaluationHarness()
    return _harness
