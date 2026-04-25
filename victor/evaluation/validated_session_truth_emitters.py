from __future__ import annotations

# Copyright 2026 Vijaykumar Singh <singhvijay@users.noreply.github.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Registry-backed emitters for validated session-truth feedback artifacts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

from victor.evaluation.protocol import (
    BenchmarkType,
    EvaluationConfig,
    EvaluationResult,
    TaskResult,
    is_browser_task_benchmark,
)
from victor.evaluation.runtime_feedback import (
    build_browser_validated_session_feedback_payload,
    build_deep_research_validated_session_feedback_payload,
)


@dataclass(frozen=True)
class ValidatedSessionTruthArtifact:
    """Serialized validated session-truth artifact ready to persist."""

    path: Path
    record: dict[str, Any]


class ValidatedSessionTruthEmitter(Protocol):
    """Strategy interface for benchmark-family validated session-truth emission."""

    def supports(self, benchmark: BenchmarkType) -> bool:
        """Return whether the emitter can handle the benchmark family."""

    def build_artifact(
        self,
        task_result: TaskResult,
        *,
        config: EvaluationConfig,
        evaluation_result: EvaluationResult,
        summary: Mapping[str, Any],
        results_dir: Path,
        source_result_path: Path,
    ) -> Optional[ValidatedSessionTruthArtifact]:
        """Build a persisted validated session-truth artifact when evidence is strong enough."""


def _safe_task_id(task_id: str) -> str:
    return str(task_id).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _feedback_artifact_path(
    *,
    results_dir: Path,
    benchmark: BenchmarkType,
    task_id: str,
    source_result_path: Path,
) -> Path:
    safe_task_id = _safe_task_id(task_id)
    return (
        results_dir
        / f"eval_session_{benchmark.value}_{safe_task_id}_{source_result_path.stem}.json"
    )


def _session_feedback_input(
    task_result: TaskResult,
    *,
    config: EvaluationConfig,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": task_result.task_id,
        "status": task_result.status.value,
        "completion_score": task_result.completion_score,
        "failure_category": (
            task_result.failure_category.value if task_result.failure_category is not None else None
        ),
        "failure_details": dict(task_result.failure_details),
        "benchmark": config.benchmark.value,
        "model": config.model,
        "dataset_metadata": dict(config.dataset_metadata),
        "total_tasks": summary.get("total_tasks"),
        "passed_tasks": summary.get("passed"),
        "failed_tasks": summary.get("failed"),
    }


def _artifact_record(
    task_result: TaskResult,
    *,
    config: EvaluationConfig,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": task_result.task_id,
        "benchmark": config.benchmark.value,
        "model": config.model,
        "runtime_evaluation_feedback": dict(payload),
        "status": task_result.status.value,
        "completion_score": task_result.completion_score,
        "failure_category": (
            task_result.failure_category.value if task_result.failure_category is not None else None
        ),
        "failure_details": dict(task_result.failure_details),
    }


class BrowserValidatedSessionTruthEmitter:
    """Validated session-truth emitter for browser-task benchmark families."""

    def supports(self, benchmark: BenchmarkType) -> bool:
        return is_browser_task_benchmark(benchmark)

    def build_artifact(
        self,
        task_result: TaskResult,
        *,
        config: EvaluationConfig,
        evaluation_result: EvaluationResult,
        summary: Mapping[str, Any],
        results_dir: Path,
        source_result_path: Path,
    ) -> Optional[ValidatedSessionTruthArtifact]:
        del evaluation_result  # Reserved for richer browser-task emitters.
        artifact_path = _feedback_artifact_path(
            results_dir=results_dir,
            benchmark=config.benchmark,
            task_id=task_result.task_id,
            source_result_path=source_result_path,
        )
        payload = build_browser_validated_session_feedback_payload(
            _session_feedback_input(task_result, config=config, summary=summary),
            source_result_path=artifact_path,
            metadata={"source_evaluation_path": str(source_result_path)},
        )
        if payload is None:
            return None
        return ValidatedSessionTruthArtifact(
            path=artifact_path,
            record=_artifact_record(task_result, config=config, payload=payload),
        )


class DeepResearchValidatedSessionTruthEmitter:
    """Validated session-truth emitter for DR3-style deep-research benchmarks."""

    def supports(self, benchmark: BenchmarkType) -> bool:
        return benchmark == BenchmarkType.DR3_EVAL

    def build_artifact(
        self,
        task_result: TaskResult,
        *,
        config: EvaluationConfig,
        evaluation_result: EvaluationResult,
        summary: Mapping[str, Any],
        results_dir: Path,
        source_result_path: Path,
    ) -> Optional[ValidatedSessionTruthArtifact]:
        del evaluation_result  # Reserved for richer research-task emitters.
        artifact_path = _feedback_artifact_path(
            results_dir=results_dir,
            benchmark=config.benchmark,
            task_id=task_result.task_id,
            source_result_path=source_result_path,
        )
        payload = build_deep_research_validated_session_feedback_payload(
            _session_feedback_input(task_result, config=config, summary=summary),
            source_result_path=artifact_path,
            metadata={"source_evaluation_path": str(source_result_path)},
        )
        if payload is None:
            return None
        return ValidatedSessionTruthArtifact(
            path=artifact_path,
            record=_artifact_record(task_result, config=config, payload=payload),
        )


class ValidatedSessionTruthEmitterRegistry:
    """Registry for benchmark-family validated session-truth emitters."""

    def __init__(self, emitters: Optional[list[ValidatedSessionTruthEmitter]] = None):
        self._emitters = list(emitters or [])

    def register(self, emitter: ValidatedSessionTruthEmitter) -> None:
        self._emitters.append(emitter)

    def resolve(self, benchmark: BenchmarkType) -> Optional[ValidatedSessionTruthEmitter]:
        for emitter in self._emitters:
            if emitter.supports(benchmark):
                return emitter
        return None


def create_default_validated_session_truth_emitter_registry() -> (
    ValidatedSessionTruthEmitterRegistry
):
    """Return the canonical emitter registry used by the evaluation harness."""
    return ValidatedSessionTruthEmitterRegistry(
        [
            BrowserValidatedSessionTruthEmitter(),
            DeepResearchValidatedSessionTruthEmitter(),
        ]
    )
