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

"""SWE-bench benchmark runner.

Implements evaluation against the SWE-bench benchmark for
real-world software engineering tasks.

NOTE: Task loading is delegated to SWEBenchLoader (swe_bench_loader.py)
to avoid code duplication. This module focuses on task execution.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from victor.evaluation.harness import BaseBenchmarkRunner, TaskEnvironment
from victor.evaluation.protocol import (
    BenchmarkTask,
    BenchmarkType,
    EvaluationConfig,
    TaskResult,
    TaskStatus,
)
from victor.evaluation.swe_bench_loader import SWEBenchConfig, SWEBenchLoader

logger = logging.getLogger(__name__)


class SWEBenchRunner(BaseBenchmarkRunner):
    """Runner for SWE-bench benchmark.

    SWE-bench evaluates the ability to solve REAL GitHub issues
    from popular Python repositories using the official Princeton
    NLP dataset from HuggingFace.

    Dataset: https://huggingface.co/datasets/princeton-nlp/SWE-bench
    Paper: "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" (2024)
    Leaderboard: https://www.swebench.com/

    IMPORTANT: This runner loads REAL benchmark data from HuggingFace,
    clones REAL repositories, applies patches, and runs REAL tests.
    Results are not simulated - they represent actual test execution.

    Supported splits:
    - test: Full test set (~2294 tasks)
    - dev: Development set
    - lite: SWE-bench Lite (~300 tasks, curated subset)

    Task loading is handled by SWEBenchLoader for consistency.
    """

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        split: str = "test",
    ):
        """Initialize the SWE-bench runner.

        Args:
            dataset_path: Path to SWE-bench dataset (JSONL file)
            split: Dataset split (test, dev, lite)
        """
        self._dataset_path = dataset_path
        self._split = split
        self._tasks_cache: Optional[list[BenchmarkTask]] = None
        self._loader = SWEBenchLoader()

    @property
    def benchmark_type(self) -> BenchmarkType:
        return BenchmarkType.SWE_BENCH

    async def load_tasks(
        self,
        config: EvaluationConfig,
    ) -> list[BenchmarkTask]:
        """Load SWE-bench tasks using SWEBenchLoader.

        Delegates to SWEBenchLoader for consistent loading across
        the codebase. Supports both local files and HuggingFace.
        """
        if self._tasks_cache is not None:
            return self._filter_tasks(self._tasks_cache, config)

        tasks = []

        # Build loader config from evaluation config
        loader_config = SWEBenchConfig(
            max_tasks=config.max_tasks,
        )

        # Try loading from dataset file first
        if self._dataset_path and self._dataset_path.exists():
            tasks = self._loader.load_from_file(self._dataset_path, loader_config)
        else:
            # Load from HuggingFace using the split
            # SWE-bench datasets on HuggingFace:
            # - princeton-nlp/SWE-bench_Lite (curated ~300 tasks)
            # - princeton-nlp/SWE-bench (full ~2300 tasks)
            if self._split == "lite":
                dataset_name = "princeton-nlp/SWE-bench_Lite"
                split = "test"
            else:
                dataset_name = "princeton-nlp/SWE-bench"
                split = "test"
            tasks = await self._loader.load_from_huggingface(
                dataset_name=dataset_name,
                split=split,
                config=loader_config,
            )

        self._tasks_cache = tasks
        return self._filter_tasks(tasks, config)

    async def run_task(
        self,
        task: BenchmarkTask,
        agent_output: str,
        config: EvaluationConfig,
    ) -> TaskResult:
        """Run a SWE-bench task and evaluate."""
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            generated_code=agent_output,
        )

        # Extract patch from agent output
        patch = self._extract_patch(agent_output)
        result.generated_patch = patch

        if not patch:
            result.status = TaskStatus.FAILED
            result.error_message = "No valid patch found in output"
            return result

        # Use cached repo for fast local test execution instead of
        # cloning from GitHub. Apply patch to a clean checkout and run tests.
        try:
            from victor.evaluation.swe_bench_loader import SWEBenchWorkspaceManager

            workspace_manager = SWEBenchWorkspaceManager()
            cached_repo = workspace_manager.get_cached_repo_path(task)

            if cached_repo and cached_repo.exists():
                result = await self._run_tests_in_cached_repo(
                    task, result, patch, cached_repo, config
                )
            else:
                # Fallback to TaskEnvironment with remote clone
                env = TaskEnvironment(
                    task=task,
                    workspace_dir=config.workspace_dir,
                    use_docker=config.use_docker,
                    docker_image=config.docker_image,
                )
                try:
                    await env.setup()
                    if not await env.apply_patch(patch):
                        result.status = TaskStatus.FAILED
                        result.error_message = "Failed to apply patch"
                        return result
                    if task.test_code:
                        await env.apply_patch(task.test_code)
                    passed, total, stdout, stderr = await env.run_tests(
                        timeout=config.timeout_per_task
                    )
                    result.tests_passed = passed
                    result.tests_total = total
                    result.tests_failed = total - passed
                    result.stdout = stdout
                    result.stderr = stderr
                    if total > 0 and passed == total:
                        result.status = TaskStatus.PASSED
                    elif passed > 0:
                        result.status = TaskStatus.FAILED
                        result.error_message = f"Partial pass: {passed}/{total}"
                    else:
                        result.status = TaskStatus.FAILED
                        result.error_message = "All tests failed"
                except Exception as e:
                    result.status = TaskStatus.ERROR
                    result.error_message = str(e)
                finally:
                    await env.cleanup()

        except Exception as e:
            result.status = TaskStatus.ERROR
            result.error_message = str(e)

        return result

    async def _run_tests_in_cached_repo(
        self,
        task: BenchmarkTask,
        result: TaskResult,
        patch: str,
        cached_repo: Path,
        config: EvaluationConfig,
    ) -> TaskResult:
        """Run tests using the cached repo for fast local execution.

        Resets repo to base_commit, applies agent's patch, runs tests,
        then resets again to leave repo clean.
        """
        import asyncio
        import sys
        import tempfile

        try:
            # Reset to base commit (fetch first — may be outside shallow depth)
            if task.base_commit:
                for cmd in [
                    ["git", "fetch", "--depth", "1", "origin", task.base_commit],
                    ["git", "checkout", "--force", task.base_commit],
                    ["git", "clean", "-fd"],
                ]:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        cwd=cached_repo,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await proc.communicate()

            # Apply the agent's patch
            patch_file = cached_repo / ".agent_patch.diff"
            patch_file.write_text(patch)
            apply_proc = await asyncio.create_subprocess_exec(
                "git",
                "apply",
                "--allow-empty",
                str(patch_file),
                cwd=cached_repo,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            apply_stdout, apply_stderr = await apply_proc.communicate()

            if apply_proc.returncode != 0:
                result.status = TaskStatus.FAILED
                result.error_message = f"Failed to apply patch: {apply_stderr.decode()[:200]}"
                logger.warning("Patch apply failed: %s", apply_stderr.decode()[:200])
                return result

            logger.info("Patch applied successfully to cached repo")

            # Apply test patch if available (SWE-bench provides test code)
            if task.test_code:
                test_patch_file = cached_repo / ".test_patch.diff"
                test_patch_file.write_text(task.test_code)
                test_proc = await asyncio.create_subprocess_exec(
                    "git",
                    "apply",
                    "--allow-empty",
                    str(test_patch_file),
                    cwd=cached_repo,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await test_proc.communicate()

            # Try to apply patch to the installed package in site-packages.
            # Source checkouts of projects like astropy need compiled C extensions
            # which aren't available from the raw git checkout. If the project is
            # pip-installed, we can patch site-packages and run tests against it.
            import os

            site_pkg_dir = None
            if task.repo:
                # Extract project name from repo URL (e.g., "astropy" from "astropy/astropy")
                repo_name = task.repo.rstrip("/").split("/")[-1].replace(".git", "")

                # Use framework's workspace dependency installer
                from victor.context.workspace_setup import (
                    ensure_project_importable,
                )

                await ensure_project_importable(repo_name, cached_repo, install_deps=True)

                try:
                    spec = __import__(repo_name)
                    site_pkg_dir = Path(spec.__file__).parent
                    # Apply patch to site-packages
                    site_patch_file = cached_repo / ".site_patch.diff"
                    site_patch_file.write_text(patch)
                    apply_site = await asyncio.create_subprocess_exec(
                        "git",
                        "apply",
                        "--allow-empty",
                        "--directory",
                        str(site_pkg_dir.parent),
                        str(site_patch_file),
                        cwd=cached_repo,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    site_out, site_err = await apply_site.communicate()
                    if apply_site.returncode == 0:
                        logger.info(
                            "Patch also applied to installed package at %s",
                            site_pkg_dir,
                        )
                    site_patch_file.unlink(missing_ok=True)
                except Exception as e:
                    logger.debug("Could not patch installed package: %s", e)

            # Run tests using detected test runner
            from victor.context.test_runner import detect_test_runner
            import re as _re

            _test_files = None
            if hasattr(task, "fail_to_pass") and task.fail_to_pass:
                _test_files = task.fail_to_pass
            elif task.test_code:
                _extracted = _re.findall(r"diff --git a/(\S+)", task.test_code)
                _test_files = [f for f in _extracted if "test" in f.lower()]

            _runner_config = detect_test_runner(cached_repo, test_files=_test_files or None)
            test_cmd = _runner_config.command
            # Add --noconftest only for pytest (avoids conftest conflicts)
            if _runner_config.runner_type == "pytest" and "-m" in test_cmd:
                idx = test_cmd.index("-m")
                test_cmd.insert(idx + 2, "--noconftest")
            logger.info("Running tests: %s", " ".join(test_cmd))

            # Use installed package path as test root if available
            test_cwd = str(site_pkg_dir.parent) if site_pkg_dir else str(cached_repo)

            clean_env = os.environ.copy()
            clean_env["PYTHONDONTWRITEBYTECODE"] = "1"
            # Apply runner-specific env vars (e.g., DJANGO_SETTINGS_MODULE)
            clean_env.update(_runner_config.env)

            test_proc = await asyncio.create_subprocess_exec(
                *test_cmd,
                cwd=test_cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=clean_env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    test_proc.communicate(),
                    timeout=min(config.timeout_per_task, 300),
                )
            except asyncio.TimeoutError:
                test_proc.kill()
                result.status = TaskStatus.FAILED
                result.error_message = "Test execution timed out"
                return result

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")
            result.stdout = stdout_str[-2000:]  # Last 2KB for diagnostics
            result.stderr = stderr_str[-2000:]

            # Parse test results
            passed, total = self._parse_test_output(stdout_str + stderr_str)
            result.tests_passed = passed
            result.tests_total = total
            result.tests_failed = total - passed

            if total > 0 and passed == total:
                result.status = TaskStatus.PASSED
                if stdout_str:
                    logger.debug("Test stdout (last 500): %s", stdout_str[-500:])
                if stderr_str:
                    logger.debug("Test stderr (last 500): %s", stderr_str[-500:])
                logger.info("Tests PASSED: %d/%d", passed, total)
            elif total > 0 and passed > 0:
                result.status = TaskStatus.FAILED
                result.error_message = f"Partial pass: {passed}/{total}"
                logger.info("Test stdout (last 500): %s", stdout_str[-500:])
                logger.info("Test stderr (last 500): %s", stderr_str[-500:])
                logger.info("Tests partial: %d/%d", passed, total)
            elif total > 0:
                result.status = TaskStatus.FAILED
                result.error_message = f"All tests failed ({total} total)"
                logger.info("Test stdout (last 500): %s", stdout_str[-500:])
                logger.info("Test stderr (last 500): %s", stderr_str[-500:])
                logger.info("Tests FAILED: %d/%d", passed, total)
            else:
                # Tests couldn't run (0 collected) — likely missing project deps.
                # Report as PATCH_APPLIED to distinguish from "no patch generated".
                result.status = TaskStatus.FAILED
                result.error_message = (
                    "Patch applied successfully but tests could not run "
                    "(0 collected). Install project deps or use Docker."
                )
                logger.warning(
                    "Tests not collected (0/0). Project may need "
                    "`pip install -e .` or Docker for test execution."
                )

        except Exception as e:
            result.status = TaskStatus.ERROR
            result.error_message = str(e)
            logger.error("Test execution error: %s", e)

        finally:
            # Clean up patch files and reset repo
            for f in [".agent_patch.diff", ".test_patch.diff"]:
                (cached_repo / f).unlink(missing_ok=True)
            if task.base_commit:
                proc = await asyncio.create_subprocess_exec(
                    "git",
                    "checkout",
                    "--force",
                    task.base_commit,
                    cwd=cached_repo,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()

        return result

    def _build_test_command(self, task: BenchmarkTask, repo_dir: Path) -> list:
        """Build the test command for a SWE-bench task.

        Uses framework test runner detection to choose the right runner
        (pytest, django, unittest) based on project structure.
        """
        import re

        from victor.context.test_runner import detect_test_runner

        # Extract test file paths from test_code patch or fail_to_pass
        test_files = None
        if hasattr(task, "fail_to_pass") and task.fail_to_pass:
            test_files = task.fail_to_pass
        elif task.test_code:
            extracted = re.findall(r"diff --git a/(\S+)", task.test_code)
            test_files = [f for f in extracted if "test" in f.lower()]

        # Use framework test runner detection
        config = detect_test_runner(repo_dir, test_files=test_files or None)
        logger.info(
            "Test runner detected: %s (command: %s)",
            config.runner_type,
            " ".join(config.command[:4]),
        )
        return config.command

    def _parse_test_output(self, output: str) -> tuple:
        """Parse pytest output to extract pass/fail counts."""
        import re

        # Try pytest format: "5 passed, 2 failed"
        match = re.search(r"(\d+) passed", output)
        passed = int(match.group(1)) if match else 0

        match = re.search(r"(\d+) failed", output)
        failed = int(match.group(1)) if match else 0

        total = passed + failed

        # Try "Ran N tests" (unittest format)
        if total == 0:
            match = re.search(r"Ran (\d+) test", output)
            if match:
                total = int(match.group(1))
                if "OK" in output:
                    passed = total

        return passed, total

    def _extract_patch(self, output: str) -> str:
        """Extract diff patch from agent output."""
        lines = output.split("\n")
        patch_lines = []
        in_patch = False

        for line in lines:
            # Start of patch
            if line.startswith("diff --git") or line.startswith("---"):
                in_patch = True

            if in_patch:
                patch_lines.append(line)

            # End of patch (heuristic)
            if in_patch and line.startswith("@@") and len(patch_lines) > 100:
                # Long patch, might have ended
                pass

        if patch_lines:
            return "\n".join(patch_lines)

        # Try to extract from code blocks
        import re

        code_blocks = re.findall(r"```(?:diff)?\n(.*?)```", output, re.DOTALL)
        for block in code_blocks:
            if "diff" in block or "---" in block or "+++" in block:
                return block.strip()

        return ""


class HumanEvalRunner(BaseBenchmarkRunner):
    """Runner for HumanEval benchmark.

    Evaluates code generation from docstrings using the official
    OpenAI HumanEval dataset from HuggingFace.

    Dataset: https://huggingface.co/datasets/openai/openai_humaneval
    Paper: "Evaluating Large Language Models Trained on Code" (2021)

    IMPORTANT: This runner loads REAL benchmark data from HuggingFace
    and executes REAL tests. Results are not simulated.
    """

    @property
    def benchmark_type(self) -> BenchmarkType:
        return BenchmarkType.HUMAN_EVAL

    async def load_tasks(
        self,
        config: EvaluationConfig,
    ) -> list[BenchmarkTask]:
        """Load HumanEval tasks from HuggingFace.

        Requires: pip install datasets
        """
        tasks = []

        try:
            from datasets import load_dataset

            logger.info("Loading HumanEval dataset from HuggingFace...")
            dataset = load_dataset("openai/openai_humaneval", split="test")
            logger.info(f"Loaded {len(dataset)} HumanEval problems")

            for item in dataset:
                task = BenchmarkTask(
                    task_id=item["task_id"],
                    benchmark=BenchmarkType.HUMAN_EVAL,
                    description=item["prompt"],
                    language="python",
                    prompt=item["prompt"],
                    test_code=item["test"],
                    solution=item["canonical_solution"],
                    category="code_generation",
                )
                tasks.append(task)

        except ImportError:
            logger.error("datasets library not installed. " "Install with: pip install datasets")
            raise RuntimeError(
                "Cannot load HumanEval: datasets library required. "
                "Install with: pip install datasets"
            )
        except Exception as e:
            logger.error(f"Failed to load HumanEval from HuggingFace: {e}")
            raise

        return self._filter_tasks(tasks, config)

    async def run_task(
        self,
        task: BenchmarkTask,
        agent_output: str,
        config: EvaluationConfig,
    ) -> TaskResult:
        """Run a HumanEval task."""
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            generated_code=agent_output,
        )

        # Extract function name from the prompt to call the check function
        import re

        func_match = re.search(r"def\s+(\w+)\s*\(", task.prompt)
        func_name = func_match.group(1) if func_match else "solution"

        # Combine generated code with test and call the check function
        full_code = agent_output + "\n\n" + task.test_code + f"\n\ncheck({func_name})\n"

        env = TaskEnvironment(task=task, use_docker=config.use_docker)

        try:
            workspace = await env.setup()

            # Write code to file
            code_file = workspace / "solution.py"
            code_file.write_text(full_code)

            # Run tests
            proc = await asyncio.create_subprocess_exec(
                "python",
                str(code_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=30,
            )

            result.stdout = stdout.decode()
            result.stderr = stderr.decode()

            if proc.returncode == 0:
                result.status = TaskStatus.PASSED
                result.tests_passed = 1
                result.tests_total = 1
            else:
                result.status = TaskStatus.FAILED
                result.tests_failed = 1
                result.tests_total = 1

        except asyncio.TimeoutError:
            result.status = TaskStatus.TIMEOUT
        except Exception as e:
            result.status = TaskStatus.ERROR
            result.error_message = str(e)
        finally:
            await env.cleanup()

        return result


class MBPPRunner(BaseBenchmarkRunner):
    """Runner for MBPP (Mostly Basic Python Problems) benchmark.

    MBPP contains crowd-sourced Python programming problems
    designed to be solvable by entry-level programmers.

    Dataset: https://huggingface.co/datasets/google-research-datasets/mbpp
    Paper: "Program Synthesis with Large Language Models" (2021)

    IMPORTANT: This runner loads REAL benchmark data from HuggingFace
    and executes REAL tests. Results are not simulated.

    Supported splits:
    - test: Test set (500 tasks)
    - train: Training set
    - validation: Validation set
    - prompt: Few-shot prompt examples (10 tasks)
    """

    def __init__(self, split: str = "test"):
        """Initialize the MBPP runner.

        Args:
            split: Dataset split (test, train, validation, prompt)
        """
        self._split = split
        self._tasks_cache: Optional[list[BenchmarkTask]] = None

    @property
    def benchmark_type(self) -> BenchmarkType:
        return BenchmarkType.MBPP

    async def load_tasks(
        self,
        config: EvaluationConfig,
    ) -> list[BenchmarkTask]:
        """Load MBPP tasks from HuggingFace.

        Requires: pip install datasets
        """
        if self._tasks_cache is not None:
            return self._filter_tasks(self._tasks_cache, config)

        tasks = []

        try:
            from datasets import load_dataset

            logger.info(f"Loading MBPP dataset (split={self._split}) from HuggingFace...")
            dataset = load_dataset(
                "google-research-datasets/mbpp",
                split=self._split,
            )
            logger.info(f"Loaded {len(dataset)} MBPP problems")

            for item in dataset:
                # MBPP format: task_id, text, code, test_list, test_setup_code, challenge_test_list
                task = BenchmarkTask(
                    task_id=str(item["task_id"]),
                    benchmark=BenchmarkType.MBPP,
                    description=item["text"],
                    language="python",
                    prompt=self._build_prompt(item),
                    test_code=self._build_test_code(item),
                    solution=item["code"],
                    category="basic_programming",
                    difficulty="easy",
                )
                tasks.append(task)

        except ImportError:
            logger.error("datasets library not installed. " "Install with: pip install datasets")
            raise RuntimeError(
                "Cannot load MBPP: datasets library required. " "Install with: pip install datasets"
            )
        except Exception as e:
            logger.error(f"Failed to load MBPP from HuggingFace: {e}")
            raise

        self._tasks_cache = tasks
        return self._filter_tasks(tasks, config)

    def _build_prompt(self, item: dict) -> str:
        """Build the prompt for the agent."""
        return f'''"""
{item["text"]}
"""
'''

    def _build_test_code(self, item: dict) -> str:
        """Build test code from test_list."""
        tests = item.get("test_list", [])
        setup = item.get("test_setup_code", "")

        code_lines = []
        if setup:
            code_lines.append(setup)
            code_lines.append("")

        # Add assertions
        for test in tests:
            code_lines.append(test)

        # Add challenge tests if available
        challenge_tests = item.get("challenge_test_list", [])
        for test in challenge_tests:
            code_lines.append(test)

        return "\n".join(code_lines)

    async def run_task(
        self,
        task: BenchmarkTask,
        agent_output: str,
        config: EvaluationConfig,
    ) -> TaskResult:
        """Run an MBPP task by executing generated code with tests."""
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            generated_code=agent_output,
        )

        # Combine generated code with test assertions
        full_code = agent_output + "\n\n" + task.test_code

        env = TaskEnvironment(task=task, use_docker=config.use_docker)

        try:
            workspace = await env.setup()

            # Write code to file
            code_file = workspace / "solution.py"
            code_file.write_text(full_code)

            # Run tests
            proc = await asyncio.create_subprocess_exec(
                "python",
                str(code_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=30,
            )

            result.stdout = stdout.decode()
            result.stderr = stderr.decode()

            if proc.returncode == 0:
                result.status = TaskStatus.PASSED
                result.tests_passed = 1
                result.tests_total = 1
            else:
                result.status = TaskStatus.FAILED
                result.tests_failed = 1
                result.tests_total = 1
                result.error_message = result.stderr[:500] if result.stderr else "Test failed"

        except asyncio.TimeoutError:
            result.status = TaskStatus.TIMEOUT
            result.error_message = "Execution timeout (30s)"
        except Exception as e:
            result.status = TaskStatus.ERROR
            result.error_message = str(e)
        finally:
            await env.cleanup()

        return result
