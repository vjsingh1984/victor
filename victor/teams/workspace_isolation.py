"""Workspace isolation service for team execution.

This service owns the effectful workspace/worktree runtime seam for teams so
``UnifiedTeamCoordinator`` can remain focused on formation coordination.
Existing public payloads intentionally keep the ``worktree_*`` names for
backward compatibility with delegate follow-up contracts and reports.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from victor.teams.merge_analyzer import MergeAnalyzer
from victor.teams.types import MemberResult, TeamFormation
from victor.teams.worktree_runtime import (
    GitWorktreeRuntime,
    WorktreeExecutionPlan,
    WorktreeIsolationPlanner,
    WorktreeMaterializationSession,
)

logger = logging.getLogger(__name__)


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
        if plan is None:
            return None
        if "materialize_worktrees" in context:
            materialize = self._coerce_context_flag(
                context,
                "materialize_worktrees",
                default=False,
            )
        else:
            materialize = bool(
                self._resolve_context_mode(context) == "delegate"
                and self._coerce_context_flag(context, "worktree_isolation", default=False)
            )
        dry_run = self._coerce_context_flag(context, "dry_run_worktrees", default=False)
        if not materialize and not dry_run:
            return None
        runtime = getattr(self._runtime, "materialize", None)
        if not callable(runtime):
            return None
        try:
            return runtime(plan, dry_run=dry_run)
        except Exception as exc:
            logger.debug(
                "Workspace isolation materialization failed; continuing with planned paths: %s",
                exc,
            )
            return None

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

    def should_execute_merge(
        self,
        context: Dict[str, Any],
        *,
        merge_orchestration: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        if "auto_merge_worktrees" in context:
            return self._coerce_context_flag(context, "auto_merge_worktrees", default=False)
        if self._resolve_context_mode(context) != "delegate":
            return False
        if not self._coerce_context_flag(context, "worktree_isolation", default=False):
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
        try:
            return executor(
                worktree_session,
                merge_analysis=merge_analysis,
                allow_risky=self._coerce_context_flag(
                    context,
                    "allow_risky_worktree_merge",
                    default=False,
                ),
                preserve_artifacts=self._coerce_context_flag(
                    context,
                    "preserve_merge_workspace",
                    default=False,
                ),
            )
        except Exception as exc:
            logger.debug("Workspace merge orchestration execution failed: %s", exc)
            return {
                "status": "error",
                "executed": False,
                "blocked_reason": "merge_execution_failed",
                "error": str(exc),
            }

    def should_cleanup(
        self,
        context: Dict[str, Any],
        *,
        result_dict: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if "cleanup_worktrees" in context:
            return self._coerce_context_flag(context, "cleanup_worktrees", default=True)
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
    def _coerce_context_flag(
        context: Dict[str, Any],
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
    def _resolve_context_mode(cls, context: Dict[str, Any]) -> Optional[str]:
        for key in ("mode", "current_mode", "active_mode"):
            raw_value = context.get(key)
            if raw_value is None:
                continue
            value = str(raw_value).strip().lower()
            if value:
                return value
        return None

    @staticmethod
    def _coerce_optional_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


__all__ = ["WorkspaceIsolationService"]
