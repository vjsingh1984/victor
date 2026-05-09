"""Workspace isolation service for team execution.

This service owns the effectful workspace/worktree runtime seam for teams so
``UnifiedTeamCoordinator`` can remain focused on formation coordination.
Existing public payloads intentionally keep the ``worktree_*`` names for
backward compatibility with delegate follow-up contracts and reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from victor.teams.merge_analyzer import MergeAnalyzer
from victor.teams.types import MemberResult, TeamFormation
from victor.teams.worktree_runtime import (
    GitWorktreeRuntime,
    WorktreeExecutionPlan,
    WorktreeIsolationPlanner,
    WorktreeMaterializationSession,
    WorktreeRuntimeError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkspaceIsolationPolicy:
    """Resolved workspace-isolation policy for one team execution."""

    mode: Optional[str] = None
    worktree_isolation: bool = False
    materialize_worktrees: bool = False
    dry_run_worktrees: bool = False
    auto_merge_worktrees: Optional[bool] = None
    allow_risky_worktree_merge: bool = False
    preserve_merge_workspace: bool = False
    cleanup_worktrees: Optional[bool] = None

    @property
    def should_materialize(self) -> bool:
        return self.materialize_worktrees or self.dry_run_worktrees

    @classmethod
    def from_context(cls, context: Mapping[str, Any]) -> "WorkspaceIsolationPolicy":
        mode = cls._resolve_context_mode(context)
        worktree_isolation = cls._coerce_context_flag(
            context,
            "worktree_isolation",
            default=False,
        )
        if "materialize_worktrees" in context:
            materialize_worktrees = cls._coerce_context_flag(
                context,
                "materialize_worktrees",
                default=False,
            )
        else:
            materialize_worktrees = bool(mode == "delegate" and worktree_isolation)

        auto_merge_worktrees = (
            cls._coerce_context_flag(context, "auto_merge_worktrees", default=False)
            if "auto_merge_worktrees" in context
            else None
        )
        cleanup_worktrees = (
            cls._coerce_context_flag(context, "cleanup_worktrees", default=True)
            if "cleanup_worktrees" in context
            else None
        )
        return cls(
            mode=mode,
            worktree_isolation=worktree_isolation,
            materialize_worktrees=materialize_worktrees,
            dry_run_worktrees=cls._coerce_context_flag(
                context,
                "dry_run_worktrees",
                default=False,
            ),
            auto_merge_worktrees=auto_merge_worktrees,
            allow_risky_worktree_merge=cls._coerce_context_flag(
                context,
                "allow_risky_worktree_merge",
                default=False,
            ),
            preserve_merge_workspace=cls._coerce_context_flag(
                context,
                "preserve_merge_workspace",
                default=False,
            ),
            cleanup_worktrees=cleanup_worktrees,
        )

    @staticmethod
    def _coerce_context_flag(
        context: Mapping[str, Any],
        key: str,
        *,
        default: bool = False,
    ) -> bool:
        raw_value = context.get(key)
        if raw_value is None:
            return default
        if isinstance(raw_value, bool):
            return raw_value
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def _resolve_context_mode(cls, context: Mapping[str, Any]) -> Optional[str]:
        for key in ("mode", "current_mode", "active_mode"):
            raw_value = context.get(key)
            if raw_value is None:
                continue
            value = str(raw_value).strip().lower()
            if value:
                return value
        return None


@dataclass(frozen=True)
class WorkspaceIsolationDiagnostic:
    """Actionable diagnostic emitted by workspace isolation operations."""

    operation: str
    reason: str
    message: str
    severity: str = "warning"
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_exception(
        cls,
        operation: str,
        exc: Exception,
        *,
        default_reason: str,
        severity: str = "warning",
    ) -> "WorkspaceIsolationDiagnostic":
        raw_reason = getattr(exc, "reason", None)
        reason = str(raw_reason).strip() if raw_reason else default_reason
        raw_details = getattr(exc, "details", None)
        details = dict(raw_details) if isinstance(raw_details, Mapping) else {}
        details.setdefault("exception_type", type(exc).__name__)
        return cls(
            operation=operation,
            reason=reason,
            message=str(exc),
            severity=severity,
            details=details,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "reason": self.reason,
            "message": self.message,
            "severity": self.severity,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class WorkspaceMaterializationOutcome:
    """Materialization result plus diagnostics for report/follow-up surfaces."""

    session: Optional[WorktreeMaterializationSession] = None
    diagnostics: tuple[WorkspaceIsolationDiagnostic, ...] = ()

    def diagnostics_payload(self) -> List[Dict[str, Any]]:
        return [diagnostic.to_dict() for diagnostic in self.diagnostics]


class WorkspaceIsolationService:
    """Coordinate planning, materialization, merge, and cleanup for team workspaces."""

    def __init__(
        self,
        *,
        planner: Optional[Any] = None,
        runtime: Optional[Any] = None,
        merge_analyzer: Optional[Any] = None,
    ) -> None:
        self._planner = planner or WorktreeIsolationPlanner()
        self._runtime = runtime or GitWorktreeRuntime()
        self._merge_analyzer = merge_analyzer or MergeAnalyzer()

    def resolve_policy(self, context: Mapping[str, Any]) -> WorkspaceIsolationPolicy:
        return WorkspaceIsolationPolicy.from_context(context)

    def plan(
        self,
        members: List[Any],
        *,
        context: Dict[str, Any],
        formation: TeamFormation,
    ) -> Optional[WorktreeExecutionPlan]:
        planner = getattr(self._planner, "plan", None)
        if not callable(planner):
            return None
        try:
            return planner(members, context=context, formation=formation)
        except Exception as exc:
            logger.debug("Workspace isolation planning failed; continuing without isolation: %s", exc)
            return None

    def materialize(
        self,
        plan: Optional[WorktreeExecutionPlan],
        *,
        context: Dict[str, Any],
    ) -> Optional[WorktreeMaterializationSession]:
        return self.materialize_with_diagnostics(plan, context=context).session

    def materialize_with_diagnostics(
        self,
        plan: Optional[WorktreeExecutionPlan],
        *,
        context: Dict[str, Any],
    ) -> WorkspaceMaterializationOutcome:
        if plan is None:
            return WorkspaceMaterializationOutcome()
        policy = self.resolve_policy(context)
        if not policy.should_materialize:
            return WorkspaceMaterializationOutcome()
        runtime = getattr(self._runtime, "materialize", None)
        if not callable(runtime):
            diagnostic = WorkspaceIsolationDiagnostic(
                operation="materialize",
                reason="runtime_unavailable",
                message="Workspace runtime does not support materialization.",
                details={
                    "team_name": plan.team_name,
                    "member_count": len(plan.assignments),
                },
            )
            return WorkspaceMaterializationOutcome(diagnostics=(diagnostic,))
        try:
            return WorkspaceMaterializationOutcome(
                session=runtime(plan, dry_run=policy.dry_run_worktrees)
            )
        except Exception as exc:
            logger.debug(
                "Workspace isolation materialization failed; continuing with planned paths: %s",
                exc,
            )
            diagnostic = WorkspaceIsolationDiagnostic.from_exception(
                "materialize",
                exc,
                default_reason="materialization_failed",
            )
            return WorkspaceMaterializationOutcome(diagnostics=(diagnostic,))

    def analyze_merge(
        self,
        member_results: Dict[str, MemberResult],
        *,
        worktree_plan: Optional[WorktreeExecutionPlan],
    ) -> Optional[Any]:
        analyzer = getattr(self._merge_analyzer, "analyze", None)
        if not callable(analyzer):
            return None
        try:
            return analyzer(member_results, worktree_plan=worktree_plan)
        except Exception as exc:
            logger.debug("Workspace merge analysis failed; continuing without metadata: %s", exc)
            return None

    def inject_changed_files(
        self,
        member_results: Dict[str, MemberResult],
        *,
        worktree_session: Optional[WorktreeMaterializationSession],
    ) -> None:
        if worktree_session is None:
            return
        collector = getattr(self._runtime, "collect_changed_files", None)
        if not callable(collector):
            return
        for member_id, result in list(member_results.items()):
            metadata = dict(result.metadata or {})
            if any(
                metadata.get(key) for key in ("changed_files", "files_touched", "modified_files")
            ):
                continue
            try:
                changed_files = list(collector(worktree_session, member_id))
            except Exception as exc:
                logger.debug("Failed to collect changed files for %s: %s", member_id, exc)
                continue
            if not changed_files:
                continue
            metadata["changed_files"] = changed_files
            member_results[member_id] = MemberResult(
                member_id=result.member_id,
                success=result.success,
                output=result.output,
                error=result.error,
                metadata=metadata,
                tool_calls_used=result.tool_calls_used,
                duration_seconds=result.duration_seconds,
                discoveries=list(result.discoveries),
            )

    def build_merge_orchestration(
        self,
        worktree_session: WorktreeMaterializationSession,
        *,
        merge_analysis: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        builder = getattr(self._runtime, "build_merge_orchestration", None)
        if not callable(builder):
            return None
        try:
            return builder(worktree_session, merge_analysis=merge_analysis)
        except Exception as exc:
            logger.debug("Workspace merge orchestration build failed: %s", exc)
            return None

    def build_merge_review_contract(
        self,
        worker_return_contracts: Mapping[str, Mapping[str, Any]],
        *,
        merge_analysis: Optional[Any],
        merge_orchestration: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the deterministic review/merge decision contract for delegated work."""
        if hasattr(merge_analysis, "to_dict"):
            merge_payload = merge_analysis.to_dict()
        elif isinstance(merge_analysis, Mapping):
            merge_payload = dict(merge_analysis)
        else:
            merge_payload = {}

        orchestration_payload = dict(merge_orchestration or {})
        review_required_members: list[str] = []
        validation_failed_members: list[str] = []
        blocking_issues: list[Dict[str, Any]] = []

        def add_review_member(member_id: Optional[str]) -> None:
            normalized = self._coerce_optional_text(member_id)
            if normalized is None or normalized in review_required_members:
                return
            review_required_members.append(normalized)

        for member_id, contract in worker_return_contracts.items():
            normalized_member_id = self._coerce_optional_text(member_id)
            if normalized_member_id is None:
                continue
            validation_run = (
                dict(contract.get("validation_run") or {})
                if isinstance(contract.get("validation_run"), Mapping)
                else {}
            )
            validation_status = self._coerce_optional_text(validation_run.get("status"))
            normalized_status = validation_status.lower() if validation_status is not None else None
            if normalized_status not in (None, "passed", "pass", "success", "succeeded", "ok"):
                validation_failed_members.append(normalized_member_id)
                add_review_member(normalized_member_id)
                issue: Dict[str, Any] = {
                    "type": "validation_failed",
                    "member_id": normalized_member_id,
                }
                if validation_status is not None:
                    issue["status"] = validation_status
                summary = self._coerce_optional_text(validation_run.get("summary"))
                if summary is not None:
                    issue["summary"] = summary
                command = self._coerce_optional_text(validation_run.get("command"))
                if command is not None:
                    issue["command"] = command
                blocking_issues.append(issue)

            merge_risk = (
                dict(contract.get("merge_risk") or {})
                if isinstance(contract.get("merge_risk"), Mapping)
                else {}
            )
            risk_level = self._coerce_optional_text(merge_risk.get("level")) or "low"
            if risk_level in {"medium", "high"}:
                add_review_member(normalized_member_id)
                blocking_issues.append(
                    {
                        "type": f"merge_risk_{risk_level}",
                        "member_id": normalized_member_id,
                        "reasons": list(merge_risk.get("reasons") or []),
                    }
                )

        recommended_merge_order = (
            list(orchestration_payload.get("recommended_merge_order") or [])
            or list(merge_payload.get("recommended_merge_order") or [])
            or list(worker_return_contracts.keys())
        )
        merge_risk_level = self._coerce_optional_text(
            orchestration_payload.get("merge_risk_level") or merge_payload.get("risk_level")
        )
        if "merge_execution_eligible" in orchestration_payload:
            merge_execution_eligible = bool(orchestration_payload.get("merge_execution_eligible"))
        else:
            merge_execution_eligible = merge_risk_level in (None, "low")
        merge_ready = bool(merge_execution_eligible and not blocking_issues)
        if (
            not merge_ready
            and not review_required_members
            and (not merge_execution_eligible or merge_risk_level not in (None, "low"))
        ):
            for member_id in recommended_merge_order:
                add_review_member(member_id)
        next_action = self._resolve_merge_review_next_action(
            merge_ready=merge_ready,
            validation_failed_members=validation_failed_members,
            review_required_members=review_required_members,
        )

        return {
            "merge_ready": merge_ready,
            "review_required": bool(review_required_members) or not merge_ready,
            "recommended_merge_order": recommended_merge_order,
            "review_required_members": review_required_members,
            "validation_failed_members": validation_failed_members,
            "blocking_issues": blocking_issues,
            "merge_risk_level": merge_risk_level,
            "merge_execution_eligible": merge_execution_eligible,
            "recommended_mode": orchestration_payload.get("recommended_mode"),
            "next_action": next_action,
        }

    def should_execute_merge(
        self,
        context: Dict[str, Any],
        *,
        merge_orchestration: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        policy = self.resolve_policy(context)
        if policy.auto_merge_worktrees is not None:
            return policy.auto_merge_worktrees
        if policy.mode != "delegate":
            return False
        if not policy.worktree_isolation:
            return False
        orchestration_payload = dict(merge_orchestration or {})
        return bool(orchestration_payload.get("merge_execution_eligible"))

    def execute_merge(
        self,
        worktree_session: WorktreeMaterializationSession,
        *,
        merge_analysis: Optional[Dict[str, Any]] = None,
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        executor = getattr(self._runtime, "execute_merge_orchestration", None)
        if not callable(executor):
            return None
        policy = self.resolve_policy(context)
        try:
            return executor(
                worktree_session,
                merge_analysis=merge_analysis,
                allow_risky=policy.allow_risky_worktree_merge,
                preserve_artifacts=policy.preserve_merge_workspace,
            )
        except Exception as exc:
            logger.debug("Workspace merge orchestration execution failed: %s", exc)
            diagnostic = WorkspaceIsolationDiagnostic.from_exception(
                "merge_execute",
                exc,
                default_reason="merge_execution_failed",
                severity="error",
            )
            blocked_reason = (
                diagnostic.reason
                if isinstance(exc, WorktreeRuntimeError)
                else "merge_execution_failed"
            )
            return {
                "status": "error",
                "executed": False,
                "blocked_reason": blocked_reason,
                "error": diagnostic.message,
                "error_details": dict(diagnostic.details),
                "diagnostics": [diagnostic.to_dict()],
            }

    def should_cleanup(
        self,
        context: Dict[str, Any],
        *,
        result_dict: Optional[Dict[str, Any]] = None,
    ) -> bool:
        policy = self.resolve_policy(context)
        if policy.cleanup_worktrees is not None:
            return policy.cleanup_worktrees
        follow_up_contract = (
            dict(result_dict.get("delegate_follow_up_contract") or {})
            if isinstance(result_dict, Mapping)
            else {}
        )
        if bool(follow_up_contract.get("preserve_worktrees")):
            return False
        return True

    def preserved_cleanup_summary(
        self,
        worktree_session: WorktreeMaterializationSession,
        *,
        reason: str,
    ) -> Dict[str, Any]:
        skipped: list[str] = []
        assignments = getattr(worktree_session, "assignments", [])
        for assignment in list(assignments or []):
            path = self._coerce_optional_text(getattr(assignment, "worktree_path", None))
            if path is not None:
                skipped.append(path)
        return {
            "removed": [],
            "skipped": skipped,
            "errors": [],
            "reason": reason,
        }

    def cleanup(self, worktree_session: WorktreeMaterializationSession) -> Dict[str, Any]:
        cleaner = getattr(self._runtime, "cleanup", None)
        if not callable(cleaner):
            return {"removed": [], "skipped": [], "errors": []}
        try:
            return cleaner(worktree_session, force=True)
        except Exception as exc:
            logger.debug("Workspace cleanup failed: %s", exc)
            return {"removed": [], "skipped": [], "errors": [str(exc)]}

    @staticmethod
    def _coerce_optional_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _resolve_merge_review_next_action(
        *,
        merge_ready: bool,
        validation_failed_members: Sequence[str],
        review_required_members: Sequence[str],
    ) -> str:
        if merge_ready:
            return "merge"
        if validation_failed_members:
            return "fix_validation"
        if review_required_members:
            return "review"
        return "inspect"


__all__ = [
    "WorkspaceIsolationDiagnostic",
    "WorkspaceIsolationPolicy",
    "WorkspaceIsolationService",
    "WorkspaceMaterializationOutcome",
]
