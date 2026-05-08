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
            text for text in (_coerce_optional_text(path) for path in paths) if text is not None
        )
    return normalized


def _derive_delegate_approval_contract(
    *,
    next_action: Optional[str],
    merge_next_action: Optional[str],
    delegate_follow_up_contract: Mapping[str, Any],
    delegate_reentry_contract: Mapping[str, Any],
    fix_validation_queue: list[Any],
    review_queue: list[Any],
    merge_order: list[Any],
) -> dict[str, Any]:
    resolved_action = next_action or merge_next_action or "inspect"
    retry_member_ids = _extract_sequence(delegate_reentry_contract, "retry_member_ids")
    review_member_ids = [
        member_id
        for member_id in (
            _coerce_optional_text(_extract_value(item, "member_id")) for item in review_queue
        )
        if member_id is not None
    ]
    validation_member_ids = [
        member_id
        for member_id in (
            _coerce_optional_text(_extract_value(item, "member_id")) for item in fix_validation_queue
        )
        if member_id is not None
    ]
    if resolved_action == "fix_validation":
        target_member_ids = retry_member_ids or validation_member_ids
        resume_ready = bool(delegate_reentry_contract) and bool(target_member_ids)
        return {
            "required": not resume_ready,
            "reason": "validation_failed",
            "recommended_action": "retry" if resume_ready else "approve_retry",
            "resume_ready": resume_ready,
            "auto_retry_eligible": resume_ready,
            "merge_executed": False,
            "target_member_ids": target_member_ids,
        }
    if resolved_action == "review":
        target_member_ids = retry_member_ids or review_member_ids
        return {
            "required": True,
            "reason": "review_required",
            "recommended_action": "review_then_retry",
            "resume_ready": bool(delegate_reentry_contract),
            "auto_retry_eligible": False,
            "merge_executed": False,
            "target_member_ids": target_member_ids,
        }
    if resolved_action == "merge":
        return {
            "required": True,
            "reason": "merge_ready",
            "recommended_action": "approve_merge",
            "resume_ready": False,
            "auto_retry_eligible": False,
            "merge_executed": False,
            "target_member_ids": list(merge_order),
        }
    target_member_ids = retry_member_ids or list(dict.fromkeys([*validation_member_ids, *review_member_ids]))
    return {
        "required": True,
        "reason": "inspect_required",
        "recommended_action": "inspect_worktrees",
        "resume_ready": bool(delegate_reentry_contract),
        "auto_retry_eligible": False,
        "merge_executed": False,
        "target_member_ids": target_member_ids,
    }


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
        "worker_return_contracts",
        "merge_review_contract",
        "delegate_follow_up_contract",
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
            "worker_return_contracts",
            "merge_review_contract",
            "delegate_follow_up_contract",
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
    worker_return_contracts = _coerce_mapping(artifacts.get("worker_return_contracts"))
    merge_review_contract = _coerce_mapping(artifacts.get("merge_review_contract"))
    delegate_follow_up_contract = _coerce_mapping(artifacts.get("delegate_follow_up_contract"))

    plan_assignments = _extract_sequence(plan, "assignments")
    session_assignments = _extract_sequence(session, "assignments")
    assignments = plan_assignments or session_assignments
    assignment_count = len(assignments)
    scoped_member_count = sum(
        1 for assignment in assignments if _extract_sequence(assignment, "claimed_paths")
    )
    readonly_shared_path_count = len(_extract_sequence(plan, "shared_readonly_paths"))
    materialized_assignment_count = sum(
        1 for assignment in session_assignments if bool(_extract_value(assignment, "materialized"))
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
    worker_contract_count = len(worker_return_contracts)
    worker_validation_count = sum(
        1
        for contract in worker_return_contracts.values()
        if _extract_mapping(contract, "validation_run")
    )
    worker_high_risk_count = sum(
        1
        for contract in worker_return_contracts.values()
        if _coerce_optional_text(_extract_mapping(contract, "merge_risk").get("level")) == "high"
    )
    worker_medium_risk_count = sum(
        1
        for contract in worker_return_contracts.values()
        if _coerce_optional_text(_extract_mapping(contract, "merge_risk").get("level"))
        == "medium"
    )
    review_required_members = _extract_sequence(merge_review_contract, "review_required_members")
    blocking_issues = _extract_sequence(merge_review_contract, "blocking_issues")
    merge_next_action = _coerce_optional_text(_extract_value(merge_review_contract, "next_action"))
    delegate_follow_up_next_action = _coerce_optional_text(
        _extract_value(delegate_follow_up_contract, "next_action")
    )
    if delegate_follow_up_next_action is None:
        delegate_follow_up_next_action = merge_next_action
    fix_validation_queue = _extract_sequence(delegate_follow_up_contract, "fix_validation_queue")
    review_queue = _extract_sequence(delegate_follow_up_contract, "review_queue")
    delegate_reentry_contract = _extract_mapping(delegate_follow_up_contract, "reentry_contract")
    delegate_approval_contract = _extract_mapping(delegate_follow_up_contract, "approval_contract")
    delegate_reentry_next_action = _coerce_optional_text(
        _extract_value(delegate_reentry_contract, "next_action")
    )
    if delegate_reentry_next_action is None:
        delegate_reentry_next_action = delegate_follow_up_next_action
    delegate_reentry_member_ids = _extract_sequence(delegate_reentry_contract, "retry_member_ids")
    delegate_reentry_resume_worktree_paths = _normalize_path_map(
        delegate_reentry_contract.get("resume_worktree_paths")
    )
    if merge_next_action is None:
        if bool(merge_review_contract.get("merge_ready", False)):
            merge_next_action = "merge"
        elif _extract_sequence(merge_review_contract, "validation_failed_members"):
            merge_next_action = "fix_validation"
        elif review_required_members or blocking_issues:
            merge_next_action = "review"
        else:
            merge_next_action = "inspect"
    if not delegate_approval_contract and delegate_follow_up_contract:
        delegate_approval_contract = _derive_delegate_approval_contract(
            next_action=delegate_follow_up_next_action,
            merge_next_action=merge_next_action,
            delegate_follow_up_contract=delegate_follow_up_contract,
            delegate_reentry_contract=delegate_reentry_contract,
            fix_validation_queue=fix_validation_queue,
            review_queue=review_queue,
            merge_order=list(merge_order),
        )
    delegate_approval_required = bool(delegate_approval_contract.get("required", False))
    delegate_approval_reason = _coerce_optional_text(delegate_approval_contract.get("reason"))
    delegate_approval_action = _coerce_optional_text(
        delegate_approval_contract.get("recommended_action")
    )
    delegate_approval_target_ids = _extract_sequence(delegate_approval_contract, "target_member_ids")
    delegate_approval_resume_ready = bool(delegate_approval_contract.get("resume_ready", False))
    delegate_auto_retry_eligible = bool(
        delegate_approval_contract.get("auto_retry_eligible", False)
    )

    return {
        "has_worktree_plan": bool(plan),
        "has_worktree_session": bool(session),
        "has_merge_analysis": bool(merge_analysis),
        "has_merge_orchestration": bool(merge_orchestration),
        "has_worktree_cleanup": bool(cleanup),
        "has_worker_return_contracts": bool(worker_return_contracts),
        "has_merge_review_contract": bool(merge_review_contract),
        "has_delegate_follow_up_contract": bool(delegate_follow_up_contract),
        "has_delegate_reentry_contract": bool(delegate_reentry_contract),
        "has_delegate_approval_contract": bool(delegate_approval_contract),
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
        "worker_contract_count": worker_contract_count,
        "worker_validation_count": worker_validation_count,
        "worker_high_risk_count": worker_high_risk_count,
        "worker_medium_risk_count": worker_medium_risk_count,
        "merge_ready": bool(merge_review_contract.get("merge_ready", False)),
        "review_required": bool(merge_review_contract.get("review_required", False)),
        "merge_next_action": merge_next_action,
        "delegate_follow_up_next_action": delegate_follow_up_next_action,
        "delegate_follow_up_preserve_worktrees": bool(
            delegate_follow_up_contract.get("preserve_worktrees", False)
        ),
        "delegate_approval_required": delegate_approval_required,
        "delegate_approval_reason": delegate_approval_reason,
        "delegate_approval_action": delegate_approval_action,
        "delegate_auto_retry_eligible": delegate_auto_retry_eligible,
        "delegate_resume_ready": delegate_approval_resume_ready,
        "delegate_approval_target_count": len(delegate_approval_target_ids),
        "delegate_reentry_next_action": delegate_reentry_next_action,
        "delegate_reentry_member_count": len(delegate_reentry_member_ids),
        "delegate_reentry_resume_worktree_count": len(delegate_reentry_resume_worktree_paths),
        "fix_validation_queue_count": len(fix_validation_queue),
        "review_queue_count": len(review_queue),
        "review_required_member_count": len(review_required_members),
        "merge_blocker_count": len(blocking_issues),
        "member_changed_files": {
            member_id: list(paths) for member_id, paths in member_changed_files.items()
        },
        "out_of_scope_writes": {
            member_id: list(paths) for member_id, paths in out_of_scope_writes.items()
        },
        "readonly_violations": {
            member_id: list(paths) for member_id, paths in readonly_violations.items()
        },
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
            "team_worker_contract_task_count": 0,
            "team_worker_contract_count": 0,
            "team_worker_validation_count": 0,
            "team_worker_high_risk_count": 0,
            "team_worker_medium_risk_count": 0,
            "team_merge_review_contract_task_count": 0,
            "team_merge_ready_task_count": 0,
            "team_merge_ready_rate": 0.0,
            "team_merge_next_actions": {},
            "team_delegate_follow_up_task_count": 0,
            "team_delegate_follow_up_actions": {},
            "team_delegate_approval_task_count": 0,
            "team_delegate_approval_required_task_count": 0,
            "team_delegate_auto_retry_eligible_task_count": 0,
            "team_delegate_approval_actions": {},
            "team_delegate_approval_reasons": {},
            "team_delegate_reentry_task_count": 0,
            "team_delegate_reentry_actions": {},
            "team_preserved_worktree_task_count": 0,
            "team_review_required_task_count": 0,
            "team_review_required_rate": 0.0,
            "team_merge_blocker_count": 0,
            "avg_team_materialized_assignments": 0.0,
            "avg_worker_validations_per_task": 0.0,
            "avg_team_merge_blockers": 0.0,
            "avg_fix_validation_queue_length": 0.0,
            "avg_review_queue_length": 0.0,
            "avg_delegate_approval_target_count": 0.0,
            "avg_delegate_reentry_member_count": 0.0,
            "avg_delegate_reentry_resume_worktree_count": 0.0,
            "avg_changed_files_per_materialized_assignment": 0.0,
        }

    task_count = total_tasks if total_tasks is not None else len(summaries)
    formations = Counter(summary["formation"] for summary in summaries if summary.get("formation"))
    risk_levels = Counter(
        summary["merge_risk_level"] for summary in summaries if summary.get("merge_risk_level")
    )
    next_actions = Counter(
        summary["merge_next_action"] for summary in summaries if summary.get("merge_next_action")
    )
    follow_up_actions = Counter(
        summary["delegate_follow_up_next_action"]
        for summary in summaries
        if summary.get("delegate_follow_up_next_action")
    )
    approval_actions = Counter(
        summary["delegate_approval_action"]
        for summary in summaries
        if summary.get("delegate_approval_action")
    )
    approval_reasons = Counter(
        summary["delegate_approval_reason"]
        for summary in summaries
        if summary.get("delegate_approval_reason")
    )
    reentry_actions = Counter(
        summary["delegate_reentry_next_action"]
        for summary in summaries
        if summary.get("has_delegate_reentry_contract") and summary.get("delegate_reentry_next_action")
    )
    plan_count = sum(1 for summary in summaries if summary.get("has_worktree_plan"))
    materialized_count = sum(1 for summary in summaries if summary.get("materialized"))
    dry_run_count = sum(1 for summary in summaries if summary.get("dry_run"))
    worker_contract_task_count = sum(
        1 for summary in summaries if bool(summary.get("has_worker_return_contracts"))
    )
    merge_review_contract_task_count = sum(
        1 for summary in summaries if bool(summary.get("has_merge_review_contract"))
    )
    delegate_follow_up_task_count = sum(
        1 for summary in summaries if bool(summary.get("has_delegate_follow_up_contract"))
    )
    delegate_approval_task_count = sum(
        1 for summary in summaries if bool(summary.get("has_delegate_approval_contract"))
    )
    delegate_reentry_task_count = sum(
        1 for summary in summaries if bool(summary.get("has_delegate_reentry_contract"))
    )
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
    preserved_worktree_task_count = sum(
        1 for summary in summaries if bool(summary.get("delegate_follow_up_preserve_worktrees"))
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
            sum(int(summary.get("assignment_count", 0) or 0) for summary in summaries)
            / summary_count,
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
        "team_worker_contract_task_count": worker_contract_task_count,
        "team_worker_contract_count": sum(
            int(summary.get("worker_contract_count", 0) or 0) for summary in summaries
        ),
        "team_worker_validation_count": sum(
            int(summary.get("worker_validation_count", 0) or 0) for summary in summaries
        ),
        "team_worker_high_risk_count": sum(
            int(summary.get("worker_high_risk_count", 0) or 0) for summary in summaries
        ),
        "team_worker_medium_risk_count": sum(
            int(summary.get("worker_medium_risk_count", 0) or 0) for summary in summaries
        ),
        "team_merge_review_contract_task_count": merge_review_contract_task_count,
        "team_merge_ready_task_count": sum(
            1 for summary in summaries if bool(summary.get("merge_ready"))
        ),
        "team_merge_ready_rate": round(
            sum(1 for summary in summaries if bool(summary.get("merge_ready")))
            / max(1, merge_review_contract_task_count),
            4,
        ),
        "team_merge_next_actions": dict(next_actions),
        "team_delegate_follow_up_task_count": delegate_follow_up_task_count,
        "team_delegate_follow_up_actions": dict(follow_up_actions),
        "team_delegate_approval_task_count": delegate_approval_task_count,
        "team_delegate_approval_required_task_count": sum(
            1 for summary in summaries if bool(summary.get("delegate_approval_required"))
        ),
        "team_delegate_auto_retry_eligible_task_count": sum(
            1 for summary in summaries if bool(summary.get("delegate_auto_retry_eligible"))
        ),
        "team_delegate_approval_actions": dict(approval_actions),
        "team_delegate_approval_reasons": dict(approval_reasons),
        "team_delegate_reentry_task_count": delegate_reentry_task_count,
        "team_delegate_reentry_actions": dict(reentry_actions),
        "team_preserved_worktree_task_count": preserved_worktree_task_count,
        "team_review_required_task_count": sum(
            1 for summary in summaries if bool(summary.get("review_required"))
        ),
        "team_review_required_rate": round(
            sum(1 for summary in summaries if bool(summary.get("review_required")))
            / max(1, merge_review_contract_task_count),
            4,
        ),
        "team_merge_blocker_count": sum(
            int(summary.get("merge_blocker_count", 0) or 0) for summary in summaries
        ),
        "avg_team_materialized_assignments": round(
            sum(int(summary.get("materialized_assignment_count", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "avg_worker_validations_per_task": round(
            sum(int(summary.get("worker_validation_count", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "avg_team_merge_blockers": round(
            sum(int(summary.get("merge_blocker_count", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "avg_fix_validation_queue_length": round(
            sum(int(summary.get("fix_validation_queue_count", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "avg_review_queue_length": round(
            sum(int(summary.get("review_queue_count", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "avg_delegate_approval_target_count": round(
            sum(int(summary.get("delegate_approval_target_count", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "avg_delegate_reentry_member_count": round(
            sum(int(summary.get("delegate_reentry_member_count", 0) or 0) for summary in summaries)
            / summary_count,
            4,
        ),
        "avg_delegate_reentry_resume_worktree_count": round(
            sum(
                int(summary.get("delegate_reentry_resume_worktree_count", 0) or 0)
                for summary in summaries
            )
            / summary_count,
            4,
        ),
        "avg_changed_files_per_materialized_assignment": round(
            (
                sum(int(summary.get("changed_file_count", 0) or 0) for summary in summaries)
                / max(
                    1,
                    sum(
                        int(summary.get("materialized_assignment_count", 0) or 0)
                        for summary in summaries
                    ),
                )
            ),
            4,
        ),
    }


__all__ = [
    "aggregate_team_feedback",
    "extract_team_feedback_artifacts",
    "summarize_team_feedback",
]
