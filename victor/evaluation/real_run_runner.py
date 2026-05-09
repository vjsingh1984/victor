"""Real-run benchmark runner for live ChatService sessions against benchmark corpora.

Conforms to BenchmarkRunner protocol so EvaluationHarness can execute it without
modification. Unlike fixture-based runners this runner drives actual agent sessions and
produces FrameworkResult artifacts that can be fed into save_stable_run_publication_bundle().

Example::

    config = RealRunConfig(
        framework=Framework.VICTOR,
        model="claude-opus-4-7",
        benchmark=BenchmarkType.ISSUE_FIX,
        max_tasks=10,
        output_dir=Path("/tmp/bench"),
    )
    runner = RealRunBenchmarkRunner(config)
    eval_result, framework_result = await runner.execute_real_run(eval_config)
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

try:
    from victor.core import get_container
    from victor.agent.services.protocols import ChatServiceProtocol
    from victor.evaluation.benchmarks.framework_comparison import (
        FrameworkResult,
        compute_metrics_from_result,
        save_stable_run_publication_bundle,
    )
    from victor.evaluation.harness import EvaluationHarness
except Exception:  # pragma: no cover - import isolation for minimal evaluation installs
    get_container = None  # type: ignore[assignment]
    ChatServiceProtocol = None  # type: ignore[assignment]
    FrameworkResult = None  # type: ignore[assignment]
    compute_metrics_from_result = None  # type: ignore[assignment]
    save_stable_run_publication_bundle = None  # type: ignore[assignment]
    EvaluationHarness = None  # type: ignore[assignment]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RealRunConfig:
    """Runtime parameters for a live benchmark execution."""

    framework: Any  # Framework enum value; imported lazily to avoid circular deps
    model: str
    benchmark: Any  # BenchmarkType enum value
    max_tasks: Optional[int] = None
    timeout_per_task: int = 300
    parallel_tasks: int = 1
    output_dir: Optional[Path] = None


# =============================================================================
# Runner
# =============================================================================


class RealRunBenchmarkRunner:
    """Drives a live ChatService session against a benchmark corpus.

    Design contract:
    - Does NOT import AgentOrchestrator; uses ChatService via DI container.
    - Does NOT modify EvaluationHarness, BenchmarkRunner protocol, or
      BaseBenchmarkRunner.
    - Falls back gracefully when ChatService or EvaluationHarness are unavailable
      (import-time isolation).
    """

    def __init__(self, config: RealRunConfig) -> None:
        self._config = config

    async def execute_real_run(
        self,
        eval_config: Any,
        *,
        resume: bool = False,
    ) -> tuple[Any, Any]:
        """Run the benchmark and return (EvaluationResult, FrameworkResult).

        Args:
            eval_config: EvaluationConfig instance.
            resume: Forward to EvaluationHarness to resume from checkpoint.

        Returns:
            Tuple of (EvaluationResult, FrameworkResult).
        """
        if EvaluationHarness is None or FrameworkResult is None or compute_metrics_from_result is None:
            raise RuntimeError("Real-run benchmark dependencies are unavailable")

        harness = EvaluationHarness()
        agent_callback = self._make_agent_callback()

        eval_result = await harness.run_evaluation(
            eval_config,
            agent_callback=agent_callback,
            resume=resume,
        )

        metrics = compute_metrics_from_result(eval_result)
        framework_result = FrameworkResult(
            framework=self._config.framework,
            benchmark=self._config.benchmark,
            model=self._config.model,
            metrics=metrics,
            config={
                "source": "real_run",
                "timeout_per_task": self._config.timeout_per_task,
                "parallel_tasks": self._config.parallel_tasks,
                "max_tasks": self._config.max_tasks,
            },
            task_results=[],
        )

        if self._config.output_dir is not None:
            self._maybe_save_bundle(framework_result)

        return eval_result, framework_result

    def _make_agent_callback(self) -> Callable[[Any], Awaitable[str]]:
        """Return an async callable that submits a BenchmarkTask to a live ChatService session."""
        timeout = self._config.timeout_per_task

        async def _run_task(task: Any) -> str:
            try:
                if get_container is None or ChatServiceProtocol is None:
                    raise RuntimeError("ChatService dependencies are unavailable")
                chat_service = get_container().get(ChatServiceProtocol)
            except Exception as exc:
                logger.warning("RealRunBenchmarkRunner: ChatService unavailable: %s", exc)
                return ""

            prompt = getattr(task, "prompt", None) or str(task)
            try:
                return await asyncio.wait_for(
                    self._invoke_chat_service(chat_service, prompt),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "RealRunBenchmarkRunner: task timed out after %ss (task_id=%s)",
                    timeout,
                    getattr(task, "task_id", "?"),
                )
                return ""
            except Exception as exc:
                logger.warning("RealRunBenchmarkRunner: task failed: %s", exc)
                return ""

        return _run_task

    async def _invoke_chat_service(self, chat_service: Any, prompt: str) -> str:
        """Submit a single prompt to ChatService and collect the text response."""
        chunks: list[str] = []
        try:
            async for event in chat_service.stream_response(prompt):
                content = getattr(event, "content", None) or getattr(event, "text", None)
                if content:
                    chunks.append(str(content))
        except Exception:
            # Fallback: try synchronous-style chat if stream not available
            result = await chat_service.chat(prompt)
            return str(result)
        return "".join(chunks)

    def _maybe_save_bundle(self, framework_result: Any) -> None:
        """Attempt to persist the framework result as a publication bundle."""
        try:
            output_dir = self._config.output_dir
            assert output_dir is not None
            if save_stable_run_publication_bundle is None:
                raise RuntimeError("stable-run publication bundler is unavailable")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Serialise framework_result to a temp JSON file for the bundle loader
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, dir=output_dir
            ) as fh:
                def _default(obj: Any) -> Any:
                    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                        return dataclasses.asdict(obj)
                    if hasattr(obj, "value"):
                        return obj.value
                    if isinstance(obj, (datetime.datetime, datetime.date)):
                        return obj.isoformat()
                    return str(obj)

                json.dump(dataclasses.asdict(framework_result), fh, default=_default)
                tmp_path = Path(fh.name)

            save_stable_run_publication_bundle(
                output_path=output_dir,
                result_paths=[tmp_path],
                benchmark=getattr(self._config.benchmark, "value", str(self._config.benchmark)),
            )
            tmp_path.unlink(missing_ok=True)
            logger.info("RealRunBenchmarkRunner: publication bundle saved to %s", output_dir)
        except Exception as exc:
            logger.warning("RealRunBenchmarkRunner: bundle save failed: %s", exc)
