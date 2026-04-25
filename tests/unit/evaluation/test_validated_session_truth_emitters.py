# Copyright 2026 Vijaykumar Singh <singhvijay@users.noreply.github.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for registry-backed validated session-truth emitters."""

from victor.evaluation.protocol import (
    BenchmarkFailureCategory,
    BenchmarkType,
    EvaluationConfig,
    EvaluationResult,
    TaskResult,
    TaskStatus,
)
from victor.evaluation.validated_session_truth_emitters import (
    ValidatedSessionTruthEmitterRegistry,
    create_default_validated_session_truth_emitter_registry,
)


def test_default_registry_resolves_browser_benchmark():
    registry = create_default_validated_session_truth_emitter_registry()

    emitter = registry.resolve(BenchmarkType.GUIDE)

    assert emitter is not None
    assert emitter.supports(BenchmarkType.CLAW_BENCH) is True


def test_default_registry_resolves_deep_research_benchmark():
    registry = create_default_validated_session_truth_emitter_registry()

    emitter = registry.resolve(BenchmarkType.DR3_EVAL)

    assert emitter is not None
    assert emitter.supports(BenchmarkType.DR3_EVAL) is True


def test_registry_returns_none_for_unsupported_benchmark():
    registry = ValidatedSessionTruthEmitterRegistry()

    assert registry.resolve(BenchmarkType.HUMAN_EVAL) is None


def test_browser_emitter_builds_artifact_with_canonical_payload(tmp_path):
    registry = create_default_validated_session_truth_emitter_registry()
    emitter = registry.resolve(BenchmarkType.GUIDE)
    assert emitter is not None

    config = EvaluationConfig(
        benchmark=BenchmarkType.GUIDE,
        model="test",
        dataset_metadata={"source_name": "GUIDE"},
    )
    task_result = TaskResult(
        task_id="guide-1",
        status=TaskStatus.FAILED,
        completion_score=0.45,
        failure_category=BenchmarkFailureCategory.TASK_COMPLETION,
        failure_details={
            "action_coverage": 0.5,
            "answer_coverage": 0.35,
            "matched_actions": ["open_url"],
            "missing_actions": ["click"],
            "matched_answer_phrases": [],
            "missing_answer_phrases": ["settings"],
            "forbidden_action_hits": [],
        },
    )
    evaluation_result = EvaluationResult(config=config, task_results=[task_result])

    artifact = emitter.build_artifact(
        task_result,
        config=config,
        evaluation_result=evaluation_result,
        summary={"total_tasks": 1, "passed": 0, "failed": 1},
        results_dir=tmp_path,
        source_result_path=tmp_path / "eval_guide_20260425_010101.json",
    )

    assert artifact is not None
    assert artifact.path.name.startswith("eval_session_guide_guide-1_")
    assert (
        artifact.record["runtime_evaluation_feedback"]["metadata"]["truth_validation_mode"]
        == "browser_posthoc_validation"
    )
    assert artifact.record["runtime_evaluation_feedback"]["metadata"]["scope"]["vertical"] == (
        "browser"
    )


def test_deep_research_emitter_returns_none_without_posthoc_evidence(tmp_path):
    registry = create_default_validated_session_truth_emitter_registry()
    emitter = registry.resolve(BenchmarkType.DR3_EVAL)
    assert emitter is not None

    config = EvaluationConfig(benchmark=BenchmarkType.DR3_EVAL, model="test")
    task_result = TaskResult(task_id="dr3-empty", status=TaskStatus.FAILED, completion_score=0.0)
    evaluation_result = EvaluationResult(config=config, task_results=[task_result])

    artifact = emitter.build_artifact(
        task_result,
        config=config,
        evaluation_result=evaluation_result,
        summary={"total_tasks": 1, "passed": 0, "failed": 1},
        results_dir=tmp_path,
        source_result_path=tmp_path / "eval_dr3_eval_20260425_010101.json",
    )

    assert artifact is None
