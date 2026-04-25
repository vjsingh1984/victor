# Copyright 2026 Vijaykumar Singh <singhvijay@users.noreply.github.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Persistence and derivation helpers for runtime calibration from benchmark truth."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from victor.framework.runtime_evaluation_policy import RuntimeEvaluationFeedback

RUNTIME_EVALUATION_FEEDBACK_FILENAME = "runtime_evaluation_feedback.json"


def get_runtime_evaluation_feedback_path(base_dir: Optional[Path] = None) -> Path:
    """Return the canonical persisted runtime-feedback file path."""
    if base_dir is not None:
        return Path(base_dir) / RUNTIME_EVALUATION_FEEDBACK_FILENAME

    try:
        from victor.config.secure_paths import get_victor_dir

        return get_victor_dir() / "evaluations" / RUNTIME_EVALUATION_FEEDBACK_FILENAME
    except ImportError:
        return Path.home() / ".victor" / "evaluations" / RUNTIME_EVALUATION_FEEDBACK_FILENAME


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _extract_task_entries(
    result_or_payload: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Normalize evaluation result objects and saved payloads into plain records."""
    if hasattr(result_or_payload, "task_results") and hasattr(result_or_payload, "get_metrics"):
        tasks: list[dict[str, Any]] = []
        for task in result_or_payload.task_results:
            assessment = task.get_confidence_assessment()
            tasks.append(
                {
                    "status": task.status.value,
                    "confidence_assessment": assessment.to_dict(),
                }
            )
        config = {
            "benchmark": result_or_payload.config.benchmark.value,
            "model": result_or_payload.config.model,
            "dataset_metadata": dict(result_or_payload.config.dataset_metadata),
        }
        return tasks, result_or_payload.get_metrics(), config

    payload = dict(result_or_payload or {})
    tasks = list(payload.get("tasks") or payload.get("task_results") or [])
    summary = dict(payload.get("summary") or {})
    config = dict(payload.get("config") or {})
    return tasks, summary, config


def derive_runtime_evaluation_feedback(result_or_payload: Any) -> RuntimeEvaluationFeedback:
    """Derive runtime calibration feedback from evaluation results or saved payloads."""
    tasks, summary, config = _extract_task_entries(result_or_payload)
    passed_confidences: list[float] = []
    failed_confidences: list[float] = []
    passed_evidence: list[float] = []

    for task in tasks:
        assessment_payload = task.get("confidence_assessment") or {}
        if not assessment_payload:
            continue
        confidence = float(assessment_payload.get("confidence_score", 0.0) or 0.0)
        evidence = float(assessment_payload.get("evidence_score", 0.0) or 0.0)
        status = str(task.get("status") or "").lower()
        if status == "passed":
            passed_confidences.append(confidence)
            passed_evidence.append(evidence)
        else:
            failed_confidences.append(confidence)

    overconfidence_rate = float(summary.get("overconfidence_rate", 0.0) or 0.0)
    underconfidence_rate = float(summary.get("underconfidence_rate", 0.0) or 0.0)
    truth_alignment_rate = float(summary.get("truth_alignment_rate", 0.0) or 0.0)

    threshold = 0.8
    threshold += min(0.15, overconfidence_rate * 0.25)
    threshold -= min(0.15, underconfidence_rate * 0.20)

    if passed_confidences:
        threshold = min(threshold, (sum(passed_confidences) / len(passed_confidences)) - 0.05)
    if failed_confidences:
        threshold = max(threshold, (sum(failed_confidences) / len(failed_confidences)) + 0.05)
    if truth_alignment_rate < 0.75:
        threshold += min(0.08, (0.75 - truth_alignment_rate) * 0.20)
    threshold = _clamp(threshold, 0.55, 0.92)

    progress_threshold = _clamp(threshold - 0.15, 0.35, threshold)
    if underconfidence_rate > overconfidence_rate:
        progress_threshold = _clamp(
            progress_threshold - min(0.05, (underconfidence_rate - overconfidence_rate) * 0.10),
            0.35,
            threshold,
        )
    elif overconfidence_rate > underconfidence_rate:
        progress_threshold = _clamp(
            progress_threshold + min(0.05, (overconfidence_rate - underconfidence_rate) * 0.10),
            0.35,
            threshold,
        )

    evidence_threshold = threshold + 0.05
    if passed_evidence:
        evidence_threshold = max(
            evidence_threshold, (sum(passed_evidence) / len(passed_evidence)) - 0.02
        )
    if overconfidence_rate > 0:
        evidence_threshold += min(0.05, overconfidence_rate * 0.10)
    evidence_threshold = _clamp(evidence_threshold, 0.55, 0.95)

    return RuntimeEvaluationFeedback(
        completion_threshold=round(threshold, 4),
        enhanced_progress_threshold=round(progress_threshold, 4),
        minimum_supported_evidence_score=round(evidence_threshold, 4),
        metadata={
            "source": "benchmark_truth_feedback",
            "benchmark": config.get("benchmark"),
            "model": config.get("model"),
            "dataset_metadata": dict(config.get("dataset_metadata") or {}),
            "truth_alignment_rate": round(truth_alignment_rate, 4),
            "overconfidence_rate": round(overconfidence_rate, 4),
            "underconfidence_rate": round(underconfidence_rate, 4),
            "task_count": len(tasks),
        },
    )


def save_runtime_evaluation_feedback(
    feedback: RuntimeEvaluationFeedback,
    *,
    path: Optional[Path] = None,
    source_result_path: Optional[Path] = None,
) -> Path:
    """Persist runtime calibration feedback to the canonical evaluation-feedback file."""
    target_path = Path(path) if path is not None else get_runtime_evaluation_feedback_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = feedback.to_dict()
    payload["metadata"] = {
        **dict(feedback.metadata),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "source_result_path": str(source_result_path) if source_result_path is not None else None,
    }
    target_path.write_text(json.dumps(payload, indent=2))
    return target_path


def load_runtime_evaluation_feedback(
    path: Optional[Path] = None,
) -> Optional[RuntimeEvaluationFeedback]:
    """Load persisted runtime calibration feedback when available."""
    target_path = Path(path) if path is not None else get_runtime_evaluation_feedback_path()
    if not target_path.exists():
        return None
    payload = json.loads(target_path.read_text())
    return RuntimeEvaluationFeedback.from_dict(payload)
