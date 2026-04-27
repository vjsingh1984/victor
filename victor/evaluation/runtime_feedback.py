# Copyright 2026 Vijaykumar Singh <singhvijay@users.noreply.github.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Persistence and derivation helpers for runtime calibration from validated evaluation truth."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
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
VALIDATED_RUNTIME_FEEDBACK_SOURCE_TRUST = {
    "benchmark_truth_feedback": 1.0,
    "validated_evaluation_truth_feedback": 0.95,
    "validated_session_truth_feedback": 0.85,
}
SCOPE_FIELD_WEIGHTS = {
    "project": 4.0,
    "model": 3.5,
    "task_type": 2.5,
    "provider": 2.0,
    "benchmark": 1.5,
    "vertical": 1.0,
    "workflow": 0.75,
}


@dataclass(frozen=True)
class RuntimeEvaluationFeedbackScope:
    """Explicit scope schema for validated evaluation-truth feedback."""

    project: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    task_type: Optional[str] = None
    benchmark: Optional[str] = None
    vertical: Optional[str] = None
    workflow: Optional[str] = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the canonical scope schema."""
        return {
            "project": self.project,
            "provider": self.provider,
            "model": self.model,
            "task_type": self.task_type,
            "benchmark": self.benchmark,
            "vertical": self.vertical,
            "workflow": self.workflow,
            "tags": list(self.tags),
        }

    def is_empty(self) -> bool:
        """Return whether the scope carries any discriminating fields."""
        return not any(
            (
                self.project,
                self.provider,
                self.model,
                self.task_type,
                self.benchmark,
                self.vertical,
                self.workflow,
                self.tags,
            )
        )

    @classmethod
    def from_value(cls, value: Any) -> "RuntimeEvaluationFeedbackScope":
        """Normalize mappings and existing scope objects into the typed schema."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return cls()
        tags_value = value.get("tags") or ()
        if isinstance(tags_value, str):
            tags = (tags_value,)
        else:
            tags = tuple(str(tag) for tag in tags_value if str(tag).strip())
        return cls(
            project=_coerce_optional_text(value.get("project")),
            provider=_coerce_optional_text(value.get("provider")),
            model=_coerce_optional_text(value.get("model")),
            task_type=_coerce_optional_text(value.get("task_type")),
            benchmark=_coerce_optional_text(value.get("benchmark")),
            vertical=_coerce_optional_text(value.get("vertical")),
            workflow=_coerce_optional_text(value.get("workflow")),
            tags=tags,
        )


def get_runtime_evaluation_feedback_path(base_dir: Optional[Path] = None) -> Path:
    """Return the canonical persisted runtime-feedback file path."""
    if base_dir is not None:
        return Path(base_dir) / RUNTIME_EVALUATION_FEEDBACK_FILENAME

    try:
        from victor.config.secure_paths import get_victor_dir

        return get_victor_dir() / "evaluations" / RUNTIME_EVALUATION_FEEDBACK_FILENAME
    except ImportError:
        return Path.home() / ".victor" / "evaluations" / RUNTIME_EVALUATION_FEEDBACK_FILENAME


def _coerce_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalized_scope_token(value: Any) -> Optional[str]:
    text = _coerce_optional_text(value)
    return text.lower() if text is not None else None


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
            "provider": result_or_payload.config.provider,
            "prompt_candidate_hash": result_or_payload.config.prompt_candidate_hash,
            "section_name": result_or_payload.config.prompt_section_name,
            "prompt_section_name": result_or_payload.config.prompt_section_name,
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
            "provider": config.get("provider"),
            "model": config.get("model"),
            "prompt_candidate_hash": config.get("prompt_candidate_hash"),
            "section_name": config.get("section_name") or config.get("prompt_section_name"),
            "scope": RuntimeEvaluationFeedbackScope(
                benchmark=_coerce_optional_text(config.get("benchmark")),
                provider=_coerce_optional_text(config.get("provider")),
                model=_coerce_optional_text(config.get("model")),
            ).to_dict(),
            "dataset_metadata": dict(config.get("dataset_metadata") or {}),
            "truth_alignment_rate": round(truth_alignment_rate, 4),
            "overconfidence_rate": round(overconfidence_rate, 4),
            "underconfidence_rate": round(underconfidence_rate, 4),
            "task_count": len(tasks),
            "topology_feedback_coverage": round(
                float(summary.get("topology_feedback_coverage", 0.0) or 0.0), 4
            ),
            "avg_topology_reward": round(
                float(summary.get("avg_topology_reward", 0.0) or 0.0), 4
            ),
            "avg_topology_confidence": round(
                float(summary.get("avg_topology_confidence", 0.0) or 0.0), 4
            ),
            "topology_actions": dict(summary.get("topology_actions") or {}),
            "topology_execution_modes": dict(summary.get("topology_execution_modes") or {}),
        },
    )


def runtime_evaluation_feedback_scope_from_context(
    context: Optional[Mapping[str, Any]],
) -> RuntimeEvaluationFeedbackScope:
    """Build the canonical scope schema from runtime context/config mappings."""
    if not isinstance(context, Mapping):
        return RuntimeEvaluationFeedbackScope()
    if "scope" in context:
        explicit_scope = RuntimeEvaluationFeedbackScope.from_value(context.get("scope"))
        if not explicit_scope.is_empty():
            return explicit_scope
    return RuntimeEvaluationFeedbackScope(
        project=_coerce_optional_text(
            context.get("project")
            or context.get("project_name")
            or context.get("repository")
            or context.get("repo")
            or context.get("workspace_name")
        ),
        provider=_coerce_optional_text(context.get("provider") or context.get("provider_name")),
        model=_coerce_optional_text(context.get("model") or context.get("model_name")),
        task_type=_coerce_optional_text(context.get("task_type")),
        benchmark=_coerce_optional_text(context.get("benchmark")),
        vertical=_coerce_optional_text(context.get("vertical")),
        workflow=_coerce_optional_text(context.get("workflow") or context.get("workflow_name")),
        tags=tuple(str(tag) for tag in (context.get("tags") or ()) if str(tag).strip()),
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
    scope = RuntimeEvaluationFeedbackScope.from_value(metadata.get("scope"))
    if scope.is_empty():
        scope = runtime_evaluation_feedback_scope_from_context(metadata)
    metadata["scope"] = scope.to_dict()
    if metadata.get("benchmark") is None:
        metadata["benchmark"] = scope.benchmark
    if metadata.get("model") is None:
        metadata["model"] = scope.model
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


def build_validated_session_feedback_payload(
    feedback: RuntimeEvaluationFeedback,
    *,
    scope: RuntimeEvaluationFeedbackScope,
    validation_label: str = "posthoc_validated",
    metadata: Optional[Mapping[str, Any]] = None,
    source: str = "validated_session_truth_feedback",
    source_result_path: Optional[Path] = None,
    saved_at: Optional[datetime] = None,
) -> dict[str, Any]:
    """Build an explicit validated-session truth payload using the canonical schema."""
    resolved_scope = RuntimeEvaluationFeedbackScope.from_value(scope)
    payload_metadata = dict(metadata or {})
    payload_metadata.update(
        {
            "source": source,
            "validated_evaluation_truth": True,
            "truth_validation_mode": validation_label,
            "scope": resolved_scope.to_dict(),
        }
    )
    return build_runtime_evaluation_feedback_payload(
        RuntimeEvaluationFeedback(
            completion_threshold=feedback.completion_threshold,
            enhanced_progress_threshold=feedback.enhanced_progress_threshold,
            minimum_supported_evidence_score=feedback.minimum_supported_evidence_score,
            metadata=payload_metadata,
        ),
        source_result_path=source_result_path,
        saved_at=saved_at,
    )


def build_swe_bench_validated_session_feedback_payload(
    validation_result: Any,
    *,
    score: Optional[Any] = None,
    source_result_path: Optional[Path] = None,
    saved_at: Optional[datetime] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Build validated session-truth payload from objective SWE-bench validation."""
    baseline = getattr(validation_result, "baseline", None)
    if baseline is None:
        return None

    metadata_payload = dict(metadata or {})
    provider = _coerce_optional_text(metadata_payload.get("provider"))
    model = _coerce_optional_text(metadata_payload.get("model"))
    prompt_candidate_hash = _coerce_optional_text(metadata_payload.get("prompt_candidate_hash"))
    section_name = _coerce_optional_text(
        metadata_payload.get("section_name") or metadata_payload.get("prompt_section_name")
    )

    baseline_status = str(getattr(getattr(baseline, "status", None), "value", "") or "").lower()
    fail_to_pass = list(getattr(baseline, "fail_to_pass", []) or [])
    pass_to_pass = list(getattr(baseline, "pass_to_pass", []) or [])
    total_fail_to_pass = len(fail_to_pass)
    total_pass_to_pass = len(pass_to_pass)
    total_validated_tests = total_fail_to_pass + total_pass_to_pass
    if baseline_status != "valid" or total_validated_tests == 0:
        return None

    fixed_tests = list(getattr(validation_result, "fail_to_pass_fixed", []) or [])
    broken_tests = list(getattr(validation_result, "pass_to_pass_broken", []) or [])
    fail_to_pass_score = (
        float(getattr(score, "fail_to_pass_score", 0.0) or 0.0)
        if score is not None
        else len(fixed_tests) / max(1, total_fail_to_pass)
    )
    pass_to_pass_score = (
        float(getattr(score, "pass_to_pass_score", 1.0) or 1.0)
        if score is not None
        else (
            1.0 - (len(broken_tests) / max(1, total_pass_to_pass))
            if total_pass_to_pass > 0
            else 1.0
        )
    )
    overall_score = (
        float(getattr(score, "overall_score", getattr(validation_result, "score", 0.0)) or 0.0)
        if score is not None
        else float(getattr(validation_result, "score", 0.0) or 0.0)
    )
    resolved = bool(getattr(score, "resolved", getattr(validation_result, "success", False)))
    partial = bool(getattr(score, "partial", getattr(validation_result, "partial_success", False)))
    unresolved_gap = _clamp(1.0 - overall_score, 0.0, 1.0)
    regression_rate = len(broken_tests) / max(1, total_pass_to_pass)

    completion_threshold = _clamp(
        0.72 + (unresolved_gap * 0.16) + (regression_rate * 0.12) - (0.05 if resolved else 0.0),
        0.58,
        0.92,
    )
    progress_threshold = _clamp(
        completion_threshold - 0.18 - (0.03 if partial and not resolved else 0.0),
        0.35,
        completion_threshold,
    )
    evidence_threshold = _clamp(
        0.76 + (unresolved_gap * 0.10) + (regression_rate * 0.14),
        0.65,
        0.95,
    )
    truth_alignment_rate = _clamp(0.55 + (overall_score * 0.4) - (regression_rate * 0.1), 0.4, 0.99)

    post_change_results = getattr(validation_result, "post_change_results", None)
    payload_metadata = dict(metadata_payload)
    payload_metadata.update(
        {
            "benchmark": "swe_bench",
            "provider": provider,
            "model": model,
            "prompt_candidate_hash": prompt_candidate_hash,
            "section_name": section_name,
            "prompt_section_name": section_name,
            "vertical": "coding",
            "truth_alignment_rate": round(truth_alignment_rate, 4),
            "task_count": total_validated_tests,
            "validation_summary": {
                "success": bool(getattr(validation_result, "success", False)),
                "partial_success": bool(getattr(validation_result, "partial_success", False)),
                "fail_to_pass_total": total_fail_to_pass,
                "fail_to_pass_fixed": len(fixed_tests),
                "pass_to_pass_total": total_pass_to_pass,
                "pass_to_pass_broken": len(broken_tests),
                "post_change_total": int(getattr(post_change_results, "total", 0) or 0),
                "post_change_passed": int(getattr(post_change_results, "passed", 0) or 0),
            },
            "score_summary": {
                "resolved": resolved,
                "partial": partial,
                "fail_to_pass_score": round(fail_to_pass_score, 4),
                "pass_to_pass_score": round(pass_to_pass_score, 4),
                "overall_score": round(overall_score, 4),
            },
        }
    )

    return build_validated_session_feedback_payload(
        RuntimeEvaluationFeedback(
            completion_threshold=round(completion_threshold, 4),
            enhanced_progress_threshold=round(progress_threshold, 4),
            minimum_supported_evidence_score=round(evidence_threshold, 4),
        ),
        scope=RuntimeEvaluationFeedbackScope(
            project=_coerce_optional_text(getattr(baseline, "repo", None)),
            provider=provider,
            model=model,
            task_type="edit",
            benchmark="swe_bench",
            vertical="coding",
            workflow="evaluation_orchestrator",
            tags=("agentic", "coding", "validated-session"),
        ),
        validation_label="swe_bench_posthoc_validation",
        metadata=payload_metadata,
        source_result_path=source_result_path,
        saved_at=saved_at,
    )


def _extract_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _extract_mapping(value: Any, key: str) -> dict[str, Any]:
    raw_value = _extract_value(value, key, {})
    return dict(raw_value) if isinstance(raw_value, Mapping) else {}


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]
    return [str(item).strip() for item in items if str(item).strip()]


def _normalized_status(value: Any) -> str:
    raw_status = _extract_value(value, "status", "")
    return str(getattr(raw_status, "value", raw_status or "")).strip().lower()


def _validated_requirement_count(*items: list[str]) -> int:
    return max(1, sum(len(item) for item in items))


def _sanitize_runtime_feedback_tag(tag: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "-", tag.strip().lower()).strip("-")


def _build_coverage_validated_session_feedback_payload(
    evaluation_result: Any,
    *,
    validation_label: str,
    benchmark_default: str,
    vertical: str,
    task_type: str,
    workflow: str,
    primary_coverage_key: str,
    secondary_coverage_key: str,
    primary_matched_key: str,
    primary_missing_key: str,
    secondary_matched_key: str,
    secondary_missing_key: str,
    blocked_hits_key: str,
    primary_weight: float,
    secondary_weight: float,
    source_result_path: Optional[Path] = None,
    saved_at: Optional[datetime] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Build validated session-truth payloads for coverage-based evaluators."""
    failure_details = _extract_mapping(evaluation_result, "failure_details")
    completion_score = _coerce_float(_extract_value(evaluation_result, "completion_score"))
    primary_coverage = _coerce_float(failure_details.get(primary_coverage_key))
    secondary_coverage = _coerce_float(failure_details.get(secondary_coverage_key))

    if (
        primary_coverage is None
        and secondary_coverage is None
        and (completion_score is None or completion_score <= 0.0)
    ):
        return None

    if primary_coverage is None:
        primary_coverage = completion_score or 0.0
    if secondary_coverage is None:
        secondary_coverage = completion_score or 0.0

    combined_completion = (
        completion_score
        if completion_score is not None and completion_score > 0.0
        else (primary_coverage * primary_weight) + (secondary_coverage * secondary_weight)
    )
    combined_completion = _clamp(combined_completion, 0.0, 1.0)
    primary_gap = _clamp(1.0 - primary_coverage, 0.0, 1.0)
    secondary_gap = _clamp(1.0 - secondary_coverage, 0.0, 1.0)
    blocked_hits = _coerce_text_list(failure_details.get(blocked_hits_key))
    blocked_penalty = min(0.18, len(blocked_hits) * 0.08)
    status = _normalized_status(evaluation_result)
    passed = status == "passed"

    completion_threshold = _clamp(
        0.58
        + ((1.0 - combined_completion) * 0.18)
        + (primary_gap * 0.11)
        + (secondary_gap * 0.08)
        + blocked_penalty
        - (0.06 if passed else 0.0),
        0.5,
        0.92,
    )
    progress_threshold = _clamp(
        completion_threshold - 0.17 - (0.03 if not passed and combined_completion < 0.6 else 0.0),
        0.35,
        completion_threshold,
    )
    evidence_threshold = _clamp(
        0.68 + (primary_gap * 0.08) + (secondary_gap * 0.06) + blocked_penalty,
        0.6,
        0.95,
    )

    matched_primary = _coerce_text_list(failure_details.get(primary_matched_key))
    missing_primary = _coerce_text_list(failure_details.get(primary_missing_key))
    matched_secondary = _coerce_text_list(failure_details.get(secondary_matched_key))
    missing_secondary = _coerce_text_list(failure_details.get(secondary_missing_key))
    task_count = _validated_requirement_count(
        matched_primary,
        missing_primary,
        matched_secondary,
        missing_secondary,
    )

    total_tasks = _coerce_int(_extract_value(evaluation_result, "total_tasks"))
    passed_tasks = _coerce_int(_extract_value(evaluation_result, "passed_tasks"))
    failed_tasks = _coerce_int(_extract_value(evaluation_result, "failed_tasks"))
    if total_tasks and passed_tasks is not None:
        truth_alignment_rate = _clamp(
            (passed_tasks / max(1, total_tasks)) - (blocked_penalty * 0.15),
            0.4,
            0.99,
        )
    else:
        truth_alignment_rate = _clamp(
            0.58 + (combined_completion * 0.32) - (blocked_penalty * 0.25),
            0.4,
            0.99,
        )

    dataset_metadata = _extract_value(evaluation_result, "dataset_metadata")
    if not isinstance(dataset_metadata, Mapping):
        dataset_metadata = {}
    benchmark = (
        _coerce_optional_text(_extract_value(evaluation_result, "benchmark")) or benchmark_default
    )
    provider = _coerce_optional_text(_extract_value(evaluation_result, "provider"))
    model = _coerce_optional_text(_extract_value(evaluation_result, "model"))
    prompt_candidate_hash = _coerce_optional_text(
        _extract_value(evaluation_result, "prompt_candidate_hash")
    )
    section_name = _coerce_optional_text(
        _extract_value(evaluation_result, "section_name")
        or _extract_value(evaluation_result, "prompt_section_name")
    )
    project = _coerce_optional_text(_extract_value(evaluation_result, "project"))
    task_id = _coerce_optional_text(_extract_value(evaluation_result, "task_id"))
    failure_category = _extract_value(evaluation_result, "failure_category")
    failure_category_value = _coerce_optional_text(
        getattr(failure_category, "value", failure_category)
    )
    raw_tags = _coerce_text_list(_extract_value(evaluation_result, "tags"))
    scope_tags = tuple(
        tag
        for tag in dict.fromkeys(
            [
                "agentic",
                vertical,
                "validated-session",
                *(
                    _sanitize_runtime_feedback_tag(tag)
                    for tag in raw_tags
                    if _sanitize_runtime_feedback_tag(tag)
                ),
            ]
        )
        if tag
    )

    payload_metadata = dict(metadata or {})
    payload_metadata.update(
        {
            "benchmark": benchmark,
            "provider": provider,
            "model": model,
            "prompt_candidate_hash": prompt_candidate_hash,
            "section_name": section_name,
            "dataset_metadata": dict(dataset_metadata),
            "truth_alignment_rate": round(truth_alignment_rate, 4),
            "task_count": task_count,
            "task_id": task_id,
            "status": status or None,
            "failure_category": failure_category_value,
            "validation_summary": {
                primary_coverage_key: round(primary_coverage, 4),
                secondary_coverage_key: round(secondary_coverage, 4),
                primary_matched_key: matched_primary,
                primary_missing_key: missing_primary,
                secondary_matched_key: matched_secondary,
                secondary_missing_key: missing_secondary,
                blocked_hits_key: blocked_hits,
                "completion_score": round(combined_completion, 4),
                "status": status or None,
            },
        }
    )
    if total_tasks is not None:
        payload_metadata["suite_summary"] = {
            "total_tasks": total_tasks,
            "passed_tasks": passed_tasks or 0,
            "failed_tasks": failed_tasks or 0,
        }

    return build_validated_session_feedback_payload(
        RuntimeEvaluationFeedback(
            completion_threshold=round(completion_threshold, 4),
            enhanced_progress_threshold=round(progress_threshold, 4),
            minimum_supported_evidence_score=round(evidence_threshold, 4),
        ),
        scope=RuntimeEvaluationFeedbackScope(
            project=project,
            provider=provider,
            model=model,
            task_type=task_type,
            benchmark=benchmark,
            vertical=vertical,
            workflow=workflow,
            tags=scope_tags,
        ),
        validation_label=validation_label,
        metadata=payload_metadata,
        source_result_path=source_result_path,
        saved_at=saved_at,
    )


def build_browser_validated_session_feedback_payload(
    evaluation_result: Any,
    *,
    source_result_path: Optional[Path] = None,
    saved_at: Optional[datetime] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Build validated session-truth payloads from browser-task post-hoc validation."""
    return _build_coverage_validated_session_feedback_payload(
        evaluation_result,
        validation_label="browser_posthoc_validation",
        benchmark_default="browser_task",
        vertical="browser",
        task_type="interaction",
        workflow="evaluation_harness",
        primary_coverage_key="action_coverage",
        secondary_coverage_key="answer_coverage",
        primary_matched_key="matched_actions",
        primary_missing_key="missing_actions",
        secondary_matched_key="matched_answer_phrases",
        secondary_missing_key="missing_answer_phrases",
        blocked_hits_key="forbidden_action_hits",
        primary_weight=0.65,
        secondary_weight=0.35,
        source_result_path=source_result_path,
        saved_at=saved_at,
        metadata=metadata,
    )


def build_deep_research_validated_session_feedback_payload(
    evaluation_result: Any,
    *,
    source_result_path: Optional[Path] = None,
    saved_at: Optional[datetime] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Build validated session-truth payloads from deep-research post-hoc validation."""
    return _build_coverage_validated_session_feedback_payload(
        evaluation_result,
        validation_label="deep_research_posthoc_validation",
        benchmark_default="dr3_eval",
        vertical="research",
        task_type="analysis",
        workflow="evaluation_harness",
        primary_coverage_key="claim_coverage",
        secondary_coverage_key="citation_coverage",
        primary_matched_key="matched_claims",
        primary_missing_key="missing_claims",
        secondary_matched_key="matched_citations",
        secondary_missing_key="missing_citations",
        blocked_hits_key="forbidden_claim_hits",
        primary_weight=0.6,
        secondary_weight=0.4,
        source_result_path=source_result_path,
        saved_at=saved_at,
        metadata=metadata,
    )


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


def _source_trust_weight(source: Optional[str]) -> float:
    return VALIDATED_RUNTIME_FEEDBACK_SOURCE_TRUST.get(str(source or ""), 0.75)


def _scope_similarity_weight(
    candidate_scope: Any,
    target_scope: Any,
) -> float:
    selected_scope = RuntimeEvaluationFeedbackScope.from_value(target_scope)
    if selected_scope.is_empty():
        return 1.0

    artifact_scope = RuntimeEvaluationFeedbackScope.from_value(candidate_scope)
    considered_weight = 0.0
    score = 0.0

    for field_name, field_weight in SCOPE_FIELD_WEIGHTS.items():
        selected_value = _normalized_scope_token(getattr(selected_scope, field_name))
        if selected_value is None:
            continue
        considered_weight += field_weight
        artifact_value = _normalized_scope_token(getattr(artifact_scope, field_name))
        if artifact_value is None:
            score += field_weight * 0.45
        elif artifact_value == selected_value:
            score += field_weight
        elif field_name in {"project", "model", "task_type"}:
            score += field_weight * 0.05
        else:
            score += field_weight * 0.20

    selected_tags = {
        _normalized_scope_token(tag)
        for tag in selected_scope.tags
        if _normalized_scope_token(tag) is not None
    }
    if selected_tags:
        considered_weight += 1.0
        artifact_tags = {
            _normalized_scope_token(tag)
            for tag in artifact_scope.tags
            if _normalized_scope_token(tag) is not None
        }
        overlap = len(selected_tags & artifact_tags) / max(1, len(selected_tags))
        score += max(0.1, overlap)

    if considered_weight == 0:
        return 1.0
    return _clamp(0.2 + (score / considered_weight) * 1.25, 0.1, 1.5)


def _feedback_weight(
    payload: Mapping[str, Any],
    reference_time: datetime,
    *,
    target_scope: Optional[RuntimeEvaluationFeedbackScope] = None,
) -> float:
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
    trust_weight = _source_trust_weight(metadata.get("source"))
    scope_weight = _scope_similarity_weight(metadata.get("scope"), target_scope)
    return max(
        0.05, recency_weight * truth_alignment * coverage_weight * trust_weight * scope_weight
    )


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
    *,
    scope: Optional[Any] = None,
) -> Optional[RuntimeEvaluationFeedback]:
    if not payloads:
        return None

    validated_payloads = [
        payload for payload in payloads if _is_validated_feedback_payload(payload)
    ]
    excluded_count = len(payloads) - len(validated_payloads)
    if not validated_payloads:
        return None

    selected_scope = RuntimeEvaluationFeedbackScope.from_value(scope)
    saved_at_values = [
        _parse_timestamp(dict(payload.get("metadata") or {}).get("saved_at"))
        for payload in validated_payloads
    ]
    resolved_saved_at_values = [value for value in saved_at_values if value is not None]
    reference_time = max(resolved_saved_at_values, default=datetime.now(timezone.utc))
    weights = [
        _feedback_weight(payload, reference_time, target_scope=selected_scope)
        for payload in validated_payloads
    ]

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
    scope_scores = [
        _scope_similarity_weight(metadata.get("scope"), selected_scope)
        for metadata in metadata_list
    ]
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
            "scope_selection_strategy": (
                "scoped_relevance_recency_reliability_weighted"
                if not selected_scope.is_empty()
                else "recency_reliability_weighted"
            ),
            "scope_target": None if selected_scope.is_empty() else selected_scope.to_dict(),
            "best_scope_match_score": round(max(scope_scores, default=1.0), 4),
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
    *,
    scope: Optional[Any] = None,
) -> Optional[RuntimeEvaluationFeedback]:
    """Load persisted runtime calibration feedback when available."""
    target_path = Path(path) if path is not None else get_runtime_evaluation_feedback_path()
    feedback_dir = _resolve_feedback_directory(target_path)
    if feedback_dir is not None:
        aggregate = _aggregate_feedback_payloads(
            _load_feedback_payloads_from_directory(feedback_dir),
            scope=scope,
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
