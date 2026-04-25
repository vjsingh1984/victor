# Copyright 2026 Vijaykumar Singh <singhvijay@users.noreply.github.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Persistence and derivation helpers for runtime calibration from validated evaluation truth."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from victor.framework.runtime_evaluation_policy import RuntimeEvaluationFeedback

RUNTIME_EVALUATION_FEEDBACK_FILENAME = "runtime_evaluation_feedback.json"
VALIDATED_RUNTIME_FEEDBACK_SOURCES = {
    "benchmark_truth_feedback",
    "validated_session_truth_feedback",
    "validated_evaluation_truth_feedback",
}
AGGREGATED_RUNTIME_FEEDBACK_SOURCE = "validated_evaluation_truth_aggregate"
FRESHNESS_HALF_LIFE_DAYS = 14.0


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


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat()


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
            "validated_evaluation_truth": True,
            "truth_validation_mode": "benchmark",
            "benchmark": config.get("benchmark"),
            "model": config.get("model"),
            "dataset_metadata": dict(config.get("dataset_metadata") or {}),
            "truth_alignment_rate": round(truth_alignment_rate, 4),
            "overconfidence_rate": round(overconfidence_rate, 4),
            "underconfidence_rate": round(underconfidence_rate, 4),
            "task_count": len(tasks),
        },
    )


def build_runtime_evaluation_feedback_payload(
    feedback: RuntimeEvaluationFeedback,
    *,
    source_result_path: Optional[Path] = None,
    saved_at: Optional[datetime] = None,
) -> dict[str, Any]:
    """Serialize runtime feedback with normalized validation metadata."""
    payload = feedback.to_dict()
    metadata = dict(payload.get("metadata") or {})
    source = metadata.get("source")
    validated = bool(metadata.get("validated_evaluation_truth")) or (
        source in VALIDATED_RUNTIME_FEEDBACK_SOURCES
    )
    metadata["validated_evaluation_truth"] = validated
    if metadata.get("truth_validation_mode") is None:
        if source == "benchmark_truth_feedback":
            metadata["truth_validation_mode"] = "benchmark"
        elif validated:
            metadata["truth_validation_mode"] = "posthoc_validated"
    resolved_saved_at = (
        saved_at or _parse_timestamp(metadata.get("saved_at")) or datetime.now(timezone.utc)
    )
    metadata["saved_at"] = _format_timestamp(resolved_saved_at)
    if source_result_path is not None:
        metadata["source_result_path"] = str(source_result_path)
    else:
        metadata["source_result_path"] = metadata.get("source_result_path")
    payload["metadata"] = metadata
    return payload


def _extract_feedback_payload(value: Any) -> Optional[dict[str, Any]]:
    """Normalize direct feedback payloads and evaluation result JSON records."""
    if not isinstance(value, Mapping):
        return None
    if "runtime_evaluation_feedback" in value:
        nested = value.get("runtime_evaluation_feedback")
        return dict(nested) if isinstance(nested, Mapping) else None
    if any(
        key in value
        for key in (
            "completion_threshold",
            "enhanced_progress_threshold",
            "minimum_supported_evidence_score",
        )
    ):
        return dict(value)
    return None


def _load_feedback_payload_file(path: Path) -> Optional[dict[str, Any]]:
    """Load a normalized feedback payload from a result or direct-feedback file."""
    try:
        raw_payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    payload = _extract_feedback_payload(raw_payload)
    if payload is None:
        return None
    feedback = RuntimeEvaluationFeedback.from_dict(payload)
    source_result_path = payload.get("metadata", {}).get("source_result_path")
    if source_result_path is None and path.name.startswith("eval_"):
        source_result_path = path
    saved_at = _parse_timestamp(payload.get("metadata", {}).get("saved_at"))
    if saved_at is None:
        try:
            saved_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            saved_at = None
    return build_runtime_evaluation_feedback_payload(
        feedback,
        source_result_path=Path(source_result_path) if source_result_path else None,
        saved_at=saved_at,
    )


def _is_validated_feedback_payload(payload: Mapping[str, Any]) -> bool:
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("validated_evaluation_truth") is True:
        return True
    return metadata.get("source") in VALIDATED_RUNTIME_FEEDBACK_SOURCES


def _resolve_feedback_directory(target_path: Path) -> Optional[Path]:
    if target_path.exists() and target_path.is_dir():
        return target_path
    if target_path.name == RUNTIME_EVALUATION_FEEDBACK_FILENAME:
        return target_path.parent
    return None


def _load_feedback_payloads_from_directory(base_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for result_path in sorted(base_dir.glob("eval_*.json")):
        payload = _load_feedback_payload_file(result_path)
        if payload is not None:
            payloads.append(payload)
    if payloads:
        return payloads

    canonical_path = base_dir / RUNTIME_EVALUATION_FEEDBACK_FILENAME
    if canonical_path.exists():
        payload = _load_feedback_payload_file(canonical_path)
        if payload is not None:
            payloads.append(payload)
    return payloads


def _feedback_weight(payload: Mapping[str, Any], reference_time: datetime) -> float:
    metadata = dict(payload.get("metadata") or {})
    saved_at = _parse_timestamp(metadata.get("saved_at")) or reference_time
    age_days = max(0.0, (reference_time - saved_at).total_seconds() / 86400.0)
    recency_weight = math.pow(0.5, age_days / FRESHNESS_HALF_LIFE_DAYS)
    truth_alignment = _clamp(
        float(metadata.get("truth_alignment_rate", 0.75) or 0.75),
        0.25,
        1.0,
    )
    task_count = max(1.0, float(metadata.get("task_count", 1.0) or 1.0))
    coverage_weight = min(3.0, 0.75 + (math.sqrt(task_count) / 4.0))
    return max(0.05, recency_weight * truth_alignment * coverage_weight)


def _select_metadata(values: list[Any], *, fallback: Any = None) -> Any:
    filtered = [value for value in values if value not in (None, {}, [])]
    if not filtered:
        return fallback
    unique = []
    for value in filtered:
        if value not in unique:
            unique.append(value)
    return unique[0] if len(unique) == 1 else fallback


def _aggregate_feedback_payloads(
    payloads: list[dict[str, Any]],
) -> Optional[RuntimeEvaluationFeedback]:
    if not payloads:
        return None

    validated_payloads = [
        payload for payload in payloads if _is_validated_feedback_payload(payload)
    ]
    excluded_count = len(payloads) - len(validated_payloads)
    if not validated_payloads:
        return None

    saved_at_values = [
        _parse_timestamp(dict(payload.get("metadata") or {}).get("saved_at"))
        for payload in validated_payloads
    ]
    resolved_saved_at_values = [value for value in saved_at_values if value is not None]
    reference_time = max(resolved_saved_at_values, default=datetime.now(timezone.utc))
    weights = [_feedback_weight(payload, reference_time) for payload in validated_payloads]

    def weighted_average(field_name: str) -> Optional[float]:
        weighted_pairs = [
            (float(payload[field_name]), weight)
            for payload, weight in zip(validated_payloads, weights)
            if payload.get(field_name) is not None
        ]
        if not weighted_pairs:
            return None
        numerator = sum(value * weight for value, weight in weighted_pairs)
        denominator = sum(weight for _, weight in weighted_pairs)
        return round(numerator / max(denominator, 1e-9), 4)

    metadata_list = [dict(payload.get("metadata") or {}) for payload in validated_payloads]
    freshest_payload = max(
        validated_payloads,
        key=lambda payload: _parse_timestamp(dict(payload.get("metadata") or {}).get("saved_at"))
        or datetime.min.replace(tzinfo=timezone.utc),
    )
    freshest_metadata = dict(freshest_payload.get("metadata") or {})
    benchmarks = [metadata.get("benchmark") for metadata in metadata_list]
    models = [metadata.get("model") for metadata in metadata_list]
    datasets = [metadata.get("dataset_metadata") for metadata in metadata_list]
    task_counts = [int(metadata.get("task_count", 0) or 0) for metadata in metadata_list]
    freshest_saved_at = max(resolved_saved_at_values, default=None)
    oldest_saved_at = min(resolved_saved_at_values, default=None)

    return RuntimeEvaluationFeedback(
        completion_threshold=weighted_average("completion_threshold"),
        enhanced_progress_threshold=weighted_average("enhanced_progress_threshold"),
        minimum_supported_evidence_score=weighted_average("minimum_supported_evidence_score"),
        metadata={
            "source": AGGREGATED_RUNTIME_FEEDBACK_SOURCE,
            "validated_evaluation_truth": True,
            "truth_validation_mode": "aggregated_validated_evaluation_truth",
            "aggregation_strategy": "recency_reliability_weighted",
            "aggregated_artifact_count": len(validated_payloads),
            "excluded_artifact_count": excluded_count,
            "validation_sources": sorted(
                {
                    metadata.get("source")
                    for metadata in metadata_list
                    if metadata.get("source") is not None
                }
            ),
            "benchmark": _select_metadata(benchmarks, fallback=freshest_metadata.get("benchmark")),
            "model": _select_metadata(models, fallback=freshest_metadata.get("model")),
            "dataset_metadata": _select_metadata(
                datasets,
                fallback=dict(freshest_metadata.get("dataset_metadata") or {}),
            ),
            "truth_alignment_rate": round(
                sum(
                    _clamp(float(metadata.get("truth_alignment_rate", 0.75) or 0.75), 0.0, 1.0)
                    * weight
                    for metadata, weight in zip(metadata_list, weights)
                )
                / max(sum(weights), 1e-9),
                4,
            ),
            "task_count": sum(task_counts),
            "freshest_saved_at": _format_timestamp(freshest_saved_at),
            "oldest_saved_at": _format_timestamp(oldest_saved_at),
            "saved_at": _format_timestamp(freshest_saved_at or reference_time),
            "source_result_path": freshest_metadata.get("source_result_path"),
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
    payload = build_runtime_evaluation_feedback_payload(
        feedback,
        source_result_path=source_result_path,
    )
    target_path.write_text(json.dumps(payload, indent=2))
    return target_path


def refresh_runtime_evaluation_feedback_aggregate(base_dir: Path) -> Optional[Path]:
    """Refresh the canonical aggregate from validated evaluation-truth artifacts."""
    aggregate = load_runtime_evaluation_feedback(
        Path(base_dir) / RUNTIME_EVALUATION_FEEDBACK_FILENAME
    )
    if aggregate is None:
        return None
    canonical_path = get_runtime_evaluation_feedback_path(base_dir)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.write_text(
        json.dumps(
            build_runtime_evaluation_feedback_payload(
                aggregate,
                source_result_path=(
                    Path(aggregate.metadata["source_result_path"])
                    if aggregate.metadata.get("source_result_path")
                    else None
                ),
                saved_at=_parse_timestamp(aggregate.metadata.get("saved_at")),
            ),
            indent=2,
        )
    )
    return canonical_path


def load_runtime_evaluation_feedback(
    path: Optional[Path] = None,
) -> Optional[RuntimeEvaluationFeedback]:
    """Load persisted runtime calibration feedback when available."""
    target_path = Path(path) if path is not None else get_runtime_evaluation_feedback_path()
    feedback_dir = _resolve_feedback_directory(target_path)
    if feedback_dir is not None:
        aggregate = _aggregate_feedback_payloads(
            _load_feedback_payloads_from_directory(feedback_dir)
        )
        if aggregate is not None:
            return aggregate
        if target_path.exists():
            payload = _load_feedback_payload_file(target_path)
            if payload is not None:
                return RuntimeEvaluationFeedback.from_dict(payload)
        return None

    if not target_path.exists():
        return None
    payload = _load_feedback_payload_file(target_path)
    if payload is None:
        return None
    return RuntimeEvaluationFeedback.from_dict(payload)
