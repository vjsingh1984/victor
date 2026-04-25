from __future__ import annotations

# Copyright 2026 Vijaykumar Singh <singhvijay@users.noreply.github.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Registry-backed emitters for validated session-truth feedback artifacts."""

from dataclasses import dataclass, field
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
    build_swe_bench_validated_session_feedback_payload,
)


@dataclass(frozen=True)
class ValidatedSessionTruthArtifact:
    """Serialized validated session-truth artifact ready to persist."""

    path: Path
    record: dict[str, Any]


@dataclass(frozen=True)
class ValidatedSessionTruthEmissionContext:
    """Canonical emission context shared by all benchmark-family emitters."""

    benchmark: BenchmarkType
    results_dir: Path
    task_id: str
    source_result_path: Optional[Path] = None
    task_result: Optional[TaskResult] = None
    config: Optional[EvaluationConfig] = None
    evaluation_result: Optional[EvaluationResult] = None
    summary: Mapping[str, Any] = field(default_factory=dict)
    validation_result: Any = None
    score: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ValidatedSessionTruthEmitter(Protocol):
    """Strategy interface for benchmark-family validated session-truth emission."""

    def supports(self, benchmark: BenchmarkType) -> bool:
        """Return whether the emitter can handle the benchmark family."""

    def build_artifact(
        self,
        context: ValidatedSessionTruthEmissionContext,
    ) -> Optional[ValidatedSessionTruthArtifact]:
        """Build a persisted validated session-truth artifact when evidence is strong enough."""


def _safe_task_id(task_id: str) -> str:
    return str(task_id).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _feedback_artifact_path(
    *,
    results_dir: Path,
    benchmark: BenchmarkType,
    task_id: str,
    source_result_path: Optional[Path],
) -> Path:
    safe_task_id = _safe_task_id(task_id)
    source_stem = source_result_path.stem if source_result_path is not None else "session"
    return results_dir / f"eval_session_{benchmark.value}_{safe_task_id}_{source_stem}.json"


def _session_feedback_input(
    context: ValidatedSessionTruthEmissionContext,
) -> Optional[dict[str, Any]]:
    task_result = context.task_result
    config = context.config
    if task_result is None or config is None:
        return None
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
        "total_tasks": context.summary.get("total_tasks"),
        "passed_tasks": context.summary.get("passed"),
        "failed_tasks": context.summary.get("failed"),
    }


def _artifact_record_from_task_result(
    context: ValidatedSessionTruthEmissionContext,
    *,
    payload: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    task_result = context.task_result
    config = context.config
    if task_result is None or config is None:
        return None
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
        context: ValidatedSessionTruthEmissionContext,
    ) -> Optional[ValidatedSessionTruthArtifact]:
        payload_input = _session_feedback_input(context)
        if payload_input is None:
            return None

        artifact_path = _feedback_artifact_path(
            results_dir=context.results_dir,
            benchmark=context.benchmark,
            task_id=context.task_id,
            source_result_path=context.source_result_path,
        )
        payload = build_browser_validated_session_feedback_payload(
            payload_input,
            source_result_path=artifact_path,
            metadata={"source_evaluation_path": str(context.source_result_path), **context.metadata},
        )
        if payload is None:
            return None
        record = _artifact_record_from_task_result(context, payload=payload)
        if record is None:
            return None
        return ValidatedSessionTruthArtifact(path=artifact_path, record=record)


class DeepResearchValidatedSessionTruthEmitter:
    """Validated session-truth emitter for DR3-style deep-research benchmarks."""

    def supports(self, benchmark: BenchmarkType) -> bool:
        return benchmark == BenchmarkType.DR3_EVAL

    def build_artifact(
        self,
        context: ValidatedSessionTruthEmissionContext,
    ) -> Optional[ValidatedSessionTruthArtifact]:
        payload_input = _session_feedback_input(context)
        if payload_input is None:
            return None

        artifact_path = _feedback_artifact_path(
            results_dir=context.results_dir,
            benchmark=context.benchmark,
            task_id=context.task_id,
            source_result_path=context.source_result_path,
        )
        payload = build_deep_research_validated_session_feedback_payload(
            payload_input,
            source_result_path=artifact_path,
            metadata={"source_evaluation_path": str(context.source_result_path), **context.metadata},
        )
        if payload is None:
            return None
        record = _artifact_record_from_task_result(context, payload=payload)
        if record is None:
            return None
        return ValidatedSessionTruthArtifact(path=artifact_path, record=record)


class SWEBenchValidatedSessionTruthEmitter:
    """Validated session-truth emitter for objective SWE-bench validation output."""

    def supports(self, benchmark: BenchmarkType) -> bool:
        return benchmark == BenchmarkType.SWE_BENCH

    def build_artifact(
        self,
        context: ValidatedSessionTruthEmissionContext,
    ) -> Optional[ValidatedSessionTruthArtifact]:
        validation_result = context.validation_result
        if validation_result is None:
            return None

        artifact_path = context.results_dir / f"eval_session_{context.task_id}.json"
        payload = build_swe_bench_validated_session_feedback_payload(
            validation_result,
            score=context.score,
            source_result_path=artifact_path,
            metadata=dict(context.metadata),
        )
        if payload is None:
            return None

        return ValidatedSessionTruthArtifact(
            path=artifact_path,
            record={
                "instance_id": context.task_id,
                "repo": getattr(getattr(validation_result, "baseline", None), "repo", None),
                "runtime_evaluation_feedback": payload,
                "validation_result": (
                    validation_result.to_dict()
                    if hasattr(validation_result, "to_dict")
                    else {
                        "success": getattr(validation_result, "success", False),
                        "partial_success": getattr(validation_result, "partial_success", False),
                        "score": getattr(validation_result, "score", 0.0),
                    }
                ),
                "score": (
                    context.score.to_dict()
                    if context.score is not None and hasattr(context.score, "to_dict")
                    else None
                ),
            },
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
    """Return the canonical emitter registry used by evaluation runtimes."""
    return ValidatedSessionTruthEmitterRegistry(
        [
            BrowserValidatedSessionTruthEmitter(),
            DeepResearchValidatedSessionTruthEmitter(),
            SWEBenchValidatedSessionTruthEmitter(),
        ]
    )
