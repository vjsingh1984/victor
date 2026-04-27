from __future__ import annotations

"""Team/worktree feedback summarization helpers for evaluation artifacts."""

from collections import Counter
from typing import Any, Iterable, Mapping, Optional


def _coerce_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(getattr(value, "value", value)).strip()
    return text or None


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return dict(value) if isinstance(value, Mapping) else {}


def _extract_mapping(value: Any, key: str) -> dict[str, Any]:
    return _coerce_mapping(_extract_value(value, key))


def _extract_sequence(value: Any, key: str) -> list[Any]:
    raw_value = _extract_value(value, key, [])
    if raw_value is None:
        return []
    if isinstance(raw_value, (list, tuple)):
        return list(raw_value)
    return []


def _normalize_path_map(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, tuple[str, ...]] = {}
    for member_id, paths in value.items():
        key = _coerce_optional_text(member_id)
        if key is None:
            continue
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, (list, tuple)):
            continue
        normalized[key] = tuple(
            text
            for text in (_coerce_optional_text(path) for path in paths)
            if text is not None
        )
    return normalized


def _extract_team_summary_mapping(value: Any) -> Optional[dict[str, Any]]:
    if isinstance(value, Mapping):
        summary = value.get("team_feedback_summary")
        if isinstance(summary, Mapping):
            return dict(summary)
    else:
        summary = getattr(value, "team_feedback_summary", None)
        if isinstance(summary, Mapping):
            return dict(summary)
    for container_name in ("metadata", "failure_details", "completion_signals"):
        container = _extract_value(value, container_name)
        if isinstance(container, Mapping):
            summary = container.get("team_feedback_summary")
            if isinstance(summary, Mapping):
                return dict(summary)
    trace = _extract_value(value, "trace")
    if trace is not None and trace is not value:
        return _extract_team_summary_mapping(trace)
    return None


def extract_team_feedback_artifacts(value: Any) -> dict[str, dict[str, Any]]:
    """Extract normalized team/worktree artifacts from task, trace, or payload objects."""
    artifacts: dict[str, dict[str, Any]] = {}
    for key in (
        "worktree_plan",
        "worktree_session",
        "merge_analysis",
        "merge_orchestration",
        "worktree_cleanup",
    ):
        mapping = _extract_mapping(value, key)
        if mapping:
            artifacts[key] = mapping

    for container_name in ("metadata", "failure_details", "completion_signals"):
        container = _extract_value(value, container_name)
        if not isinstance(container, Mapping):
            continue
        for key in (
            "worktree_plan",
            "worktree_session",
            "merge_analysis",
            "merge_orchestration",
            "worktree_cleanup",
        ):
            if key in artifacts:
                continue
            mapping = _extract_mapping(container, key)
            if mapping:
                artifacts[key] = mapping

    trace = _extract_value(value, "trace")
    if trace is not None and trace is not value:
        for key, mapping in extract_team_feedback_artifacts(trace).items():
            artifacts.setdefault(key, mapping)
    return artifacts


def summarize_team_feedback(value: Any) -> Optional[dict[str, Any]]:
    """Return a task-level team/worktree summary suitable for benchmark artifacts."""
    existing_summary = _extract_team_summary_mapping(value)
    if existing_summary:
        return existing_summary

    artifacts = extract_team_feedback_artifacts(value)
    if not artifacts:
        return None

    plan = artifacts.get("worktree_plan", {})
    session = artifacts.get("worktree_session", {})
    merge_analysis = artifacts.get("merge_analysis", {})
    merge_orchestration = artifacts.get("merge_orchestration", {})
    cleanup = artifacts.get("worktree_cleanup", {})

    plan_assignments = _extract_sequence(plan, "assignments")
    session_assignments = _extract_sequence(session, "assignments")
    assignments = plan_assignments or session_assignments
    assignment_count = len(assignments)
    scoped_member_count = sum(1 for assignment in assignments if _extract_sequence(assignment, "claimed_paths"))
    readonly_shared_path_count = len(_extract_sequence(plan, "shared_readonly_paths"))
    materialized_assignment_count = sum(
        1
        for assignment in session_assignments
        if bool(_extract_value(assignment, "materialized"))
    )
    merge_risk_level = _coerce_optional_text(
        _extract_value(value, "merge_risk_level") or merge_analysis.get("risk_level")
    )
    member_changed_files = _normalize_path_map(merge_analysis.get("member_changed_files"))
    out_of_scope_writes = _normalize_path_map(merge_analysis.get("out_of_scope_writes"))
    readonly_violations = _normalize_path_map(merge_analysis.get("readonly_violations"))
    merge_order = (
        _extract_sequence(merge_orchestration, "recommended_merge_order")
        or _extract_sequence(merge_analysis, "recommended_merge_order")
        or _extract_sequence(plan, "merge_order")
    )
    cleanup_removed = _extract_sequence(cleanup, "removed")
    cleanup_errors = _extract_sequence(cleanup, "errors")
    cleanup_skipped = _extract_sequence(cleanup, "skipped")
    changed_file_count = sum(len(paths) for paths in member_changed_files.values())
    members_with_changes = sum(1 for paths in member_changed_files.values() if paths)
    out_of_scope_count = sum(len(paths) for paths in out_of_scope_writes.values())
    readonly_violation_count = sum(len(paths) for paths in readonly_violations.values())

    return {
        "has_worktree_plan": bool(plan),
        "has_worktree_session": bool(session),
        "has_merge_analysis": bool(merge_analysis),
        "has_merge_orchestration": bool(merge_orchestration),
        "has_worktree_cleanup": bool(cleanup),
        "team_name": _coerce_optional_text(plan.get("team_name")),
        "formation": _coerce_optional_text(plan.get("formation")),
        "assignment_count": assignment_count,
        "scoped_member_count": scoped_member_count,
        "readonly_shared_path_count": readonly_shared_path_count,
        "materialized": bool(session.get("materialized", False)),
        "dry_run": bool(session.get("dry_run", False)),
        "materialized_assignment_count": materialized_assignment_count,
        "merge_risk_level": merge_risk_level,
        "merge_conflict_count": _coerce_int(merge_analysis.get("conflict_count")) or 0,
        "merge_overlap_count": len(_extract_sequence(merge_analysis, "overlapping_files")),
        "out_of_scope_write_count": out_of_scope_count,
        "readonly_violation_count": readonly_violation_count,
        "members_with_changes": members_with_changes,
        "changed_file_count": changed_file_count,
        "cleanup_removed_count": len(cleanup_removed),
        "cleanup_error_count": len(cleanup_errors),
        "cleanup_skipped_count": len(cleanup_skipped),
        "merge_order_length": len(merge_order),
        "merge_order": list(merge_order),
        "member_changed_files": {member_id: list(paths) for member_id, paths in member_changed_files.items()},
        "out_of_scope_writes": {member_id: list(paths) for member_id, paths in out_of_scope_writes.items()},
        "readonly_violations": {member_id: list(paths) for member_id, paths in readonly_violations.items()},
    }


def aggregate_team_feedback(
    values: Iterable[Any],
    *,
    total_tasks: Optional[int] = None,
) -> dict[str, Any]:
    """Aggregate team/worktree summaries across benchmark task results."""
    summaries = [summary for value in values if (summary := summarize_team_feedback(value))]
    if not summaries:
        return {
            "tasks_with_team_feedback": 0,
            "team_feedback_coverage": 0.0,
            "team_formations": {},
            "team_merge_risk_levels": {},
            "team_worktree_plan_count": 0,
            "team_worktree_materialized_count": 0,
            "team_worktree_dry_run_count": 0,
            "team_low_risk_task_count": 0,
            "team_medium_risk_task_count": 0,
            "team_high_risk_task_count": 0,
            "team_merge_conflict_task_count": 0,
            "team_merge_conflict_count": 0,
            "team_merge_overlap_task_count": 0,
            "team_out_of_scope_write_task_count": 0,
            "team_out_of_scope_write_count": 0,
            "team_readonly_violation_task_count": 0,
            "team_readonly_violation_count": 0,
            "team_cleanup_task_count": 0,
            "team_cleanup_error_task_count": 0,
            "team_cleanup_error_count": 0,
            "avg_team_assignments": 0.0,
            "avg_team_scoped_members": 0.0,
            "avg_team_members_with_changes": 0.0,
            "avg_team_changed_file_count": 0.0,
            "team_materialized_assignment_total": 0,
        }

    task_count = total_tasks if total_tasks is not None else len(summaries)
    formations = Counter(summary["formation"] for summary in summaries if summary.get("formation"))
    risk_levels = Counter(
        summary["merge_risk_level"] for summary in summaries if summary.get("merge_risk_level")
    )
    plan_count = sum(1 for summary in summaries if summary.get("has_worktree_plan"))
    materialized_count = sum(1 for summary in summaries if summary.get("materialized"))
    dry_run_count = sum(1 for summary in summaries if summary.get("dry_run"))
    merge_conflict_task_count = sum(
        1 for summary in summaries if int(summary.get("merge_conflict_count", 0) or 0) > 0
    )
    merge_overlap_task_count = sum(
        1 for summary in summaries if int(summary.get("merge_overlap_count", 0) or 0) > 0
    )
    out_of_scope_task_count = sum(
        1 for summary in summaries if int(summary.get("out_of_scope_write_count", 0) or 0) > 0
    )
    readonly_violation_task_count = sum(
        1 for summary in summaries if int(summary.get("readonly_violation_count", 0) or 0) > 0
    )
    cleanup_task_count = sum(1 for summary in summaries if summary.get("has_worktree_cleanup"))
    cleanup_error_task_count = sum(
        1 for summary in summaries if int(summary.get("cleanup_error_count", 0) or 0) > 0
    )
    summary_count = max(1, len(summaries))

    return {
        "tasks_with_team_feedback": len(summaries),
        "team_feedback_coverage": round(len(summaries) / max(1, task_count), 4),
        "team_formations": dict(formations),
        "team_merge_risk_levels": dict(risk_levels),
        "team_worktree_plan_count": plan_count,
        "team_worktree_materialized_count": materialized_count,
        "team_worktree_dry_run_count": dry_run_count,
        "team_low_risk_task_count": int(risk_levels.get("low", 0) or 0),
        "team_medium_risk_task_count": int(risk_levels.get("medium", 0) or 0),
        "team_high_risk_task_count": int(risk_levels.get("high", 0) or 0),
        "team_merge_conflict_task_count": merge_conflict_task_count,
        "team_merge_conflict_count": sum(
            int(summary.get("merge_conflict_count", 0) or 0) for summary in summaries
        ),
        "team_merge_overlap_task_count": merge_overlap_task_count,
        "team_out_of_scope_write_task_count": out_of_scope_task_count,
        "team_out_of_scope_write_count": sum(
            int(summary.get("out_of_scope_write_count", 0) or 0) for summary in summaries
        ),
        "team_readonly_violation_task_count": readonly_violation_task_count,
        "team_readonly_violation_count": sum(
            int(summary.get("readonly_violation_count", 0) or 0) for summary in summaries
        ),
        "team_cleanup_task_count": cleanup_task_count,
        "team_cleanup_error_task_count": cleanup_error_task_count,
        "team_cleanup_error_count": sum(
            int(summary.get("cleanup_error_count", 0) or 0) for summary in summaries
        ),
        "avg_team_assignments": round(
            sum(int(summary.get("assignment_count", 0) or 0) for summary in summaries) / summary_count,
            4,
        ),
        "avg_team_scoped_members": round(
            sum(int(summary.get("scoped_member_count", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "avg_team_members_with_changes": round(
            sum(int(summary.get("members_with_changes", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "avg_team_changed_file_count": round(
            sum(int(summary.get("changed_file_count", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "team_materialized_assignment_total": sum(
            int(summary.get("materialized_assignment_count", 0) or 0) for summary in summaries
        ),
    }


__all__ = [
    "aggregate_team_feedback",
    "extract_team_feedback_artifacts",
    "summarize_team_feedback",
]
