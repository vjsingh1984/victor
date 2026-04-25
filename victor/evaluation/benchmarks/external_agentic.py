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

"""Local-manifest adapters for external agentic benchmarks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from victor.evaluation.harness import BaseBenchmarkRunner, TaskEnvironment
from victor.evaluation.protocol import (
    BenchmarkFailureCategory,
    BenchmarkTask,
    BenchmarkType,
    EvaluationConfig,
    TaskResult,
    TaskStatus,
    is_external_agentic_benchmark,
)

logger = logging.getLogger(__name__)


class ExternalAgenticBenchmarkRunner(BaseBenchmarkRunner):
    """Runner for local JSON/JSONL external agentic benchmark manifests."""

    def __init__(self, benchmark_type: BenchmarkType, dataset_path: Path):
        if not is_external_agentic_benchmark(benchmark_type):
            raise ValueError(f"{benchmark_type.value} is not an external agentic benchmark")
        self._benchmark_type = benchmark_type
        self._dataset_path = dataset_path
        self._tasks_cache: list[BenchmarkTask] | None = None

    @property
    def benchmark_type(self) -> BenchmarkType:
        return self._benchmark_type

    async def load_tasks(self, config: EvaluationConfig) -> list[BenchmarkTask]:
        """Load tasks from a local dataset manifest."""
        if self._tasks_cache is None:
            self._tasks_cache = self._load_from_file(self._dataset_path)
        return self._filter_tasks(self._tasks_cache, config)

    async def run_task(
        self,
        task: BenchmarkTask,
        agent_output: str,
        config: EvaluationConfig,
    ) -> TaskResult:
        """Evaluate an external benchmark task against local tests or patches."""
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            generated_code=agent_output,
        )
        env = TaskEnvironment(
            task=task,
            workspace_dir=config.workspace_dir,
            use_docker=config.use_docker and bool(task.repo),
            docker_image=config.docker_image,
        )

        try:
            workspace = await env.setup()
            if self._looks_like_patch(agent_output):
                result.generated_patch = agent_output
                if not await env.apply_patch(agent_output):
                    result.status = TaskStatus.FAILED
                    result.error_message = "Failed to apply generated patch"
                    result.failure_category = BenchmarkFailureCategory.PATCH_APPLICATION
                    return result
            elif agent_output.strip():
                target_dir = workspace / "repo" if (workspace / "repo").exists() else workspace
                (target_dir / "solution.py").write_text(agent_output)
            else:
                result.status = TaskStatus.FAILED
                result.error_message = "No generated output"
                result.failure_category = BenchmarkFailureCategory.TASK_COMPLETION
                return result

            if not task.test_code:
                result.status = TaskStatus.PASSED
                return result

            if not task.repo and not (workspace / "pytest.ini").exists():
                (workspace / "pytest.ini").write_text("[pytest]\npython_files = test_*.py\n")

            passed, total, stdout, stderr = await env.run_tests(
                timeout=task.timeout_seconds or config.timeout_per_task
            )
            result.tests_passed = passed
            result.tests_total = total
            result.tests_failed = max(total - passed, 0)
            result.stdout = stdout
            result.stderr = stderr

            if stderr == "Timeout":
                result.status = TaskStatus.TIMEOUT
                result.error_message = "Test execution timed out"
                result.failure_category = BenchmarkFailureCategory.TIMEOUT
            elif total > 0 and passed == total:
                result.status = TaskStatus.PASSED
            elif total > 0:
                result.status = TaskStatus.FAILED
                result.error_message = (
                    f"Partial pass: {passed}/{total}" if passed else "All tests failed"
                )
                result.failure_category = BenchmarkFailureCategory.TEST_FAILURE
            elif stderr:
                result.status = TaskStatus.ERROR
                result.error_message = stderr[:500]
                result.failure_category = BenchmarkFailureCategory.ENVIRONMENT_ERROR
            else:
                result.status = TaskStatus.PASSED

            return result
        except Exception as exc:
            result.status = TaskStatus.ERROR
            result.error_message = str(exc)
            result.failure_category = BenchmarkFailureCategory.EXECUTION_ERROR
            return result
        finally:
            await env.cleanup()

    def _load_from_file(self, dataset_path: Path) -> list[BenchmarkTask]:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found: {dataset_path}")
        records = self._read_records(dataset_path)
        return [self._record_to_task(record, index) for index, record in enumerate(records, start=1)]

    def _read_records(self, dataset_path: Path) -> list[dict[str, Any]]:
        if dataset_path.suffix == ".jsonl":
            records = []
            for line in dataset_path.read_text().splitlines():
                if not line.strip():
                    continue
                records.append(json.loads(line))
            return records

        payload = json.loads(dataset_path.read_text())
        if isinstance(payload, dict):
            payload = payload.get("tasks", [])
        if not isinstance(payload, list):
            raise ValueError("Benchmark dataset payload must be a list or {'tasks': [...]} object")
        return payload

    def _record_to_task(self, record: dict[str, Any], index: int) -> BenchmarkTask:
        description = str(
            record.get("description")
            or record.get("instruction")
            or record.get("prompt")
            or record.get("issue_text")
            or ""
        ).strip()
        prompt = str(record.get("prompt") or record.get("instruction") or description).strip()
        if not prompt:
            raise ValueError(f"Task #{index} is missing a prompt or description")

        return BenchmarkTask(
            task_id=str(record.get("task_id") or record.get("id") or f"{self._benchmark_type.value}-{index}"),
            benchmark=self._benchmark_type,
            description=description or prompt,
            language=str(record.get("language") or "python"),
            prompt=prompt,
            context_code=str(record.get("context_code") or record.get("context") or ""),
            test_code=str(record.get("test_code") or record.get("tests") or ""),
            repo=record.get("repo"),
            base_commit=record.get("base_commit"),
            issue_text=record.get("issue_text"),
            hints=self._as_list(record.get("hints")),
            solution=record.get("solution"),
            patch=record.get("patch"),
            difficulty=str(record.get("difficulty") or "medium"),
            category=str(record.get("category") or "external_agentic"),
            tags=self._as_list(record.get("tags")),
            timeout_seconds=int(record.get("timeout_seconds") or 300),
            complexity_override=record.get("complexity_override"),
            task_type_hint=record.get("task_type_hint"),
        )

    @staticmethod
    def _as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]

    @staticmethod
    def _looks_like_patch(output: str) -> bool:
        stripped = output.lstrip()
        return stripped.startswith("diff --git") or stripped.startswith("--- ") or "@@" in output
