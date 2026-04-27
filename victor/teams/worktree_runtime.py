# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Worktree-isolation planning helpers for team execution.

This module does not materialize git worktrees directly. It provides the
stable planning contract and per-member context overlays that the runtime can
apply before shell-backed worktree creation is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Optional, Sequence

from victor.teams.types import TeamFormation


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _slug(value: Any) -> str:
    text = _normalize_text(value) or "member"
    normalized = re.sub(r"[^a-zA-Z0-9._/-]+", "-", text).strip("-./")
    normalized = normalized.replace("/", "-").replace("\\", "-")
    return normalized or "member"


def _normalize_paths(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = [values]
    normalized: list[str] = []
    for value in list(values or []):
        text = _normalize_text(value)
        if text is None:
            continue
        normalized.append(text.rstrip("/"))
    return tuple(dict.fromkeys(normalized))


def _normalize_scope_map(raw_map: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw_map, dict):
        return {}
    normalized: dict[str, tuple[str, ...]] = {}
    for member_id, scopes in raw_map.items():
        key = _normalize_text(member_id)
        if key is None:
            continue
        normalized[key] = _normalize_paths(scopes)
    return normalized


def _path_for_member(parent_dir: str, team_name: str, member_id: str) -> str:
    return str(Path(parent_dir) / f"{_slug(team_name)}-{_slug(member_id)}")


@dataclass(frozen=True)
class WorktreeAssignment:
    """One member's isolated workspace assignment."""

    member_id: str
    branch_name: str
    worktree_name: str
    worktree_path: str
    claimed_paths: tuple[str, ...] = ()
    readonly_paths: tuple[str, ...] = ()
    merge_priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "member_id": self.member_id,
            "branch_name": self.branch_name,
            "worktree_name": self.worktree_name,
            "worktree_path": self.worktree_path,
            "claimed_paths": list(self.claimed_paths),
            "readonly_paths": list(self.readonly_paths),
            "merge_priority": self.merge_priority,
            "metadata": dict(self.metadata),
        }

    def to_context_overrides(self) -> Dict[str, Any]:
        return {
            "isolation_mode": "worktree",
            "workspace_root": self.worktree_path,
            "worktree_path": self.worktree_path,
            "branch_name": self.branch_name,
            "claimed_paths": list(self.claimed_paths),
            "readonly_paths": list(self.readonly_paths),
            "worktree_assignment": self.to_dict(),
        }


@dataclass(frozen=True)
class WorktreeExecutionPlan:
    """Execution-scoped isolation plan for a team run."""

    team_name: str
    repo_root: str
    parent_dir: str
    base_ref: str
    branch_prefix: str
    formation: TeamFormation
    assignments: tuple[WorktreeAssignment, ...]
    merge_order: tuple[str, ...]
    shared_readonly_paths: tuple[str, ...] = ()
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def assignment_for(self, member_id: str) -> Optional[WorktreeAssignment]:
        normalized = _normalize_text(member_id)
        if normalized is None:
            return None
        for assignment in self.assignments:
            if assignment.member_id == normalized:
                return assignment
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team_name": self.team_name,
            "repo_root": self.repo_root,
            "parent_dir": self.parent_dir,
            "base_ref": self.base_ref,
            "branch_prefix": self.branch_prefix,
            "formation": self.formation.value,
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "merge_order": list(self.merge_order),
            "shared_readonly_paths": list(self.shared_readonly_paths),
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
        }


class WorktreeIsolationPlanner:
    """Plan isolated per-member workspaces for team execution."""

    def plan(
        self,
        members: Sequence[Any],
        *,
        context: Dict[str, Any],
        formation: TeamFormation,
    ) -> Optional[WorktreeExecutionPlan]:
        if not self._is_enabled(context) or not members:
            return None

        repo_root = _normalize_text(
            context.get("repo_root")
            or context.get("workspace_root")
            or context.get("worktree_root")
        )
        if repo_root is None:
            return None

        team_name = _normalize_text(context.get("team_name")) or "UnifiedTeam"
        parent_dir = _normalize_text(context.get("worktree_parent"))
        if parent_dir is None:
            parent_dir = str(Path(repo_root) / ".victor" / "team_worktrees")
        branch_prefix = _normalize_text(context.get("branch_prefix")) or f"victor/{_slug(team_name)}"
        base_ref = _normalize_text(context.get("base_ref")) or "HEAD"
        shared_readonly_paths = _normalize_paths(
            context.get("shared_readonly_paths") or context.get("readonly_paths")
        )
        scope_map = _normalize_scope_map(
            context.get("member_write_scopes") or context.get("member_scopes")
        )
        explicit_merge_order = [
            member_id
            for member_id in (_normalize_paths(context.get("member_merge_order")) or ())
            if member_id
        ]

        assignments: list[WorktreeAssignment] = []
        for index, member in enumerate(members):
            member_id = _normalize_text(getattr(member, "id", None)) or f"member-{index + 1}"
            claimed_paths = scope_map.get(member_id, ())
            merge_priority = self._resolve_merge_priority(
                member=member,
                index=index,
                formation=formation,
                explicit_merge_order=explicit_merge_order,
            )
            assignments.append(
                WorktreeAssignment(
                    member_id=member_id,
                    branch_name=f"{branch_prefix}/{_slug(member_id)}-{index + 1}",
                    worktree_name=f"{_slug(team_name)}-{_slug(member_id)}",
                    worktree_path=_path_for_member(parent_dir, team_name, member_id),
                    claimed_paths=claimed_paths,
                    readonly_paths=shared_readonly_paths,
                    merge_priority=merge_priority,
                    metadata={
                        "member_index": index,
                        "formation": formation.value,
                        "is_manager": bool(getattr(member, "is_manager", False)),
                    },
                )
            )

        merge_order = tuple(
            assignment.member_id
            for assignment in sorted(assignments, key=lambda item: (item.merge_priority, item.member_id))
        )
        return WorktreeExecutionPlan(
            team_name=team_name,
            repo_root=repo_root,
            parent_dir=parent_dir,
            base_ref=base_ref,
            branch_prefix=branch_prefix,
            formation=formation,
            assignments=tuple(assignments),
            merge_order=merge_order,
            shared_readonly_paths=shared_readonly_paths,
            rationale="worktree isolation enabled by runtime context",
            metadata={
                "member_count": len(assignments),
                "scoped_members": sum(1 for assignment in assignments if assignment.claimed_paths),
            },
        )

    @staticmethod
    def _is_enabled(context: Dict[str, Any]) -> bool:
        raw_value = context.get("worktree_isolation")
        if isinstance(raw_value, bool):
            return raw_value
        if raw_value is not None:
            text = str(raw_value).strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False
        return _normalize_text(context.get("isolation_mode")) == "worktree"

    @staticmethod
    def _resolve_merge_priority(
        *,
        member: Any,
        index: int,
        formation: TeamFormation,
        explicit_merge_order: Sequence[str],
    ) -> int:
        member_id = _normalize_text(getattr(member, "id", None)) or f"member-{index + 1}"
        if member_id in explicit_merge_order:
            return explicit_merge_order.index(member_id)
        if formation == TeamFormation.HIERARCHICAL and bool(getattr(member, "is_manager", False)):
            return len(explicit_merge_order) + 1000
        return len(explicit_merge_order) + index


__all__ = [
    "WorktreeAssignment",
    "WorktreeExecutionPlan",
    "WorktreeIsolationPlanner",
]
