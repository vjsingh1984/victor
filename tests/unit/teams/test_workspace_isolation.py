from types import SimpleNamespace

from victor.teams.types import MemberResult, TeamFormation
from victor.teams.workspace_isolation import WorkspaceIsolationService
from victor.teams.worktree_runtime import (
    MaterializedWorktreeAssignment,
    WorktreeAssignment,
    WorktreeExecutionPlan,
    WorktreeMaterializationSession,
)


def _plan() -> WorktreeExecutionPlan:
    assignment = WorktreeAssignment(
        member_id="worker",
        branch_name="victor/team/worker-1",
        worktree_name="team-worker",
        worktree_path="/tmp/team-worker",
    )
    return WorktreeExecutionPlan(
        team_name="team",
        repo_root="/repo",
        parent_dir="/tmp",
        base_ref="HEAD",
        branch_prefix="victor/team",
        formation=TeamFormation.PARALLEL,
        assignments=(assignment,),
        merge_order=("worker",),
    )


def _session() -> WorktreeMaterializationSession:
    plan = _plan()
    return WorktreeMaterializationSession(
        plan=plan,
        assignments=(
            MaterializedWorktreeAssignment(
                assignment=plan.assignments[0],
                materialized=True,
                cleanup_required=True,
            ),
        ),
        materialized=True,
    )


def test_workspace_isolation_service_plans_and_materializes_delegate_context():
    plan = _plan()
    session = _session()
    planner = SimpleNamespace(plan=lambda members, *, context, formation: plan)

    class Runtime:
        def __init__(self):
            self.materialize_calls = []

        def materialize(self, received_plan, *, dry_run=False):
            self.materialize_calls.append((received_plan, dry_run))
            return session

    runtime = Runtime()
    service = WorkspaceIsolationService(planner=planner, runtime=runtime, merge_analyzer=None)

    resolved_plan = service.plan(
        [SimpleNamespace(id="worker")],
        context={"worktree_isolation": True, "repo_root": "/repo"},
        formation=TeamFormation.PARALLEL,
    )
    resolved_session = service.materialize(
        resolved_plan,
        context={"mode": "delegate", "worktree_isolation": True},
    )

    assert resolved_plan is plan
    assert resolved_session is session
    assert runtime.materialize_calls == [(plan, False)]


def test_workspace_isolation_service_injects_changed_files_without_overwriting_metadata():
    session = _session()

    class Runtime:
        def collect_changed_files(self, received_session, member_id):
            assert received_session is session
            assert member_id == "worker"
            return ("src/service.py",)

    service = WorkspaceIsolationService(runtime=Runtime())
    member_results = {
        "worker": MemberResult(
            member_id="worker",
            success=True,
            output="done",
            metadata={},
        ),
        "reported": MemberResult(
            member_id="reported",
            success=True,
            output="done",
            metadata={"changed_files": ["already.py"]},
        ),
    }

    service.inject_changed_files(member_results, worktree_session=session)

    assert member_results["worker"].metadata["changed_files"] == ["src/service.py"]
    assert member_results["reported"].metadata["changed_files"] == ["already.py"]


def test_workspace_isolation_service_preserves_cleanup_for_follow_up_contracts():
    session = _session()
    service = WorkspaceIsolationService()

    should_cleanup = service.should_cleanup(
        {},
        result_dict={"delegate_follow_up_contract": {"preserve_worktrees": True}},
    )
    summary = service.preserved_cleanup_summary(session, reason="preserved_for_follow_up")

    assert should_cleanup is False
    assert summary == {
        "removed": [],
        "skipped": ["/tmp/team-worker"],
        "errors": [],
        "reason": "preserved_for_follow_up",
    }
