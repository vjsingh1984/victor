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
"""

import json
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
    """

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        split: str = "test",
    ):
        """Initialize the SWE-bench runner.

        Args:
            dataset_path: Path to SWE-bench dataset (JSON file)
            split: Dataset split (test, dev, lite)
        """
        self._dataset_path = dataset_path
        self._split = split
        self._tasks_cache: Optional[list[BenchmarkTask]] = None

    @property
    def benchmark_type(self) -> BenchmarkType:
        return BenchmarkType.SWE_BENCH

    async def load_tasks(
        self,
        config: EvaluationConfig,
    ) -> list[BenchmarkTask]:
        """Load SWE-bench tasks."""
        if self._tasks_cache is not None:
            return self._filter_tasks(self._tasks_cache, config)

        tasks = []

        # Try loading from dataset file
        if self._dataset_path and self._dataset_path.exists():
            tasks = self._load_from_file(self._dataset_path)
        else:
            # Try loading from Hugging Face datasets
            tasks = await self._load_from_hf()

        self._tasks_cache = tasks
        return self._filter_tasks(tasks, config)

    def _load_from_file(self, path: Path) -> list[BenchmarkTask]:
        """Load tasks from JSON file."""
        tasks = []

        try:
            with open(path) as f:
                data = json.load(f)

            for item in data:
                task = BenchmarkTask(
                    task_id=item.get("instance_id", ""),
                    benchmark=BenchmarkType.SWE_BENCH,
                    description=item.get("problem_statement", ""),
                    language="python",
                    prompt=self._build_prompt(item),
                    repo=item.get("repo", ""),
                    base_commit=item.get("base_commit", ""),
                    issue_text=item.get("problem_statement", ""),
                    hints=item.get("hints_text", "").split("\n") if item.get("hints_text") else [],
                    patch=item.get("patch", ""),
                    test_code=item.get("test_patch", ""),
                    difficulty=self._determine_difficulty(item),
                    category=item.get("repo", "").split("/")[-1] if item.get("repo") else "",
                )
                tasks.append(task)

        except Exception as e:
            logger.error(f"Failed to load SWE-bench from file: {e}")

        return tasks

    async def _load_from_hf(self) -> list[BenchmarkTask]:
        """Load tasks from Hugging Face datasets."""
        tasks = []

        try:
            from datasets import load_dataset

            # Load SWE-bench lite (smaller subset)
            if self._split == "lite":
                dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
            else:
                dataset = load_dataset("princeton-nlp/SWE-bench", split=self._split)

            for item in dataset:
                task = BenchmarkTask(
                    task_id=item.get("instance_id", ""),
                    benchmark=BenchmarkType.SWE_BENCH,
                    description=item.get("problem_statement", ""),
                    language="python",
                    prompt=self._build_prompt(item),
                    repo=item.get("repo", ""),
                    base_commit=item.get("base_commit", ""),
                    issue_text=item.get("problem_statement", ""),
                    patch=item.get("patch", ""),
                    test_code=item.get("test_patch", ""),
                    difficulty=self._determine_difficulty(item),
                    category=item.get("repo", "").split("/")[-1] if item.get("repo") else "",
                )
                tasks.append(task)

        except ImportError:
            logger.warning("datasets library not installed, cannot load from HF")
        except Exception as e:
            logger.error(f"Failed to load from Hugging Face: {e}")

        return tasks

    def _build_prompt(self, item: dict) -> str:
        """Build the prompt for the agent."""
        repo = item.get("repo", "")
        issue = item.get("problem_statement", "")
        hints = item.get("hints_text", "")

        prompt = f"""You are working on the repository: {repo}

## Issue Description
{issue}

## Instructions
Please analyze this issue and provide a solution. Your response should be a unified diff patch that can be applied to fix the issue.

The patch should:
1. Fix the described issue
2. Not introduce any new bugs
3. Follow the existing code style
4. Include minimal changes needed to fix the issue
"""

        if hints:
            prompt += f"\n## Hints\n{hints}\n"

        prompt += "\nProvide your solution as a diff patch."

        return prompt

    def _determine_difficulty(self, item: dict) -> str:
        """Determine task difficulty based on patch size."""
        patch = item.get("patch", "")
        lines = len(patch.split("\n"))

        if lines < 10:
            return "easy"
        elif lines < 50:
            return "medium"
        else:
            return "hard"

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

        # Set up environment
        env = TaskEnvironment(
            task=task,
            workspace_dir=config.workspace_dir,
            use_docker=config.use_docker,
            docker_image=config.docker_image,
        )

        try:
            await env.setup()

            # Apply patch
            if not await env.apply_patch(patch):
                result.status = TaskStatus.FAILED
                result.error_message = "Failed to apply patch"
                return result

            # Apply test patch if available
            if task.test_code:
                await env.apply_patch(task.test_code)

            # Run tests
            passed, total, stdout, stderr = await env.run_tests(
                timeout=config.timeout_per_task
            )

            result.tests_passed = passed
            result.tests_total = total
            result.tests_failed = total - passed
            result.stdout = stdout
            result.stderr = stderr

            # Determine status
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

        return result

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
            logger.error(
                "datasets library not installed. "
                "Install with: pip install datasets"
            )
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

        # Combine generated code with test
        full_code = agent_output + "\n\n" + task.test_code

        env = TaskEnvironment(task=task, use_docker=config.use_docker)

        try:
            workspace = await env.setup()

            # Write code to file
            code_file = workspace / "solution.py"
            code_file.write_text(full_code)

            # Run tests
            import asyncio
            proc = await asyncio.create_subprocess_exec(
                "python", str(code_file),
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
            logger.error(
                "datasets library not installed. "
                "Install with: pip install datasets"
            )
            raise RuntimeError(
                "Cannot load MBPP: datasets library required. "
                "Install with: pip install datasets"
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
            import asyncio as aio
            proc = await aio.create_subprocess_exec(
                "python", str(code_file),
                stdout=aio.subprocess.PIPE,
                stderr=aio.subprocess.PIPE,
            )

            stdout, stderr = await aio.wait_for(
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
