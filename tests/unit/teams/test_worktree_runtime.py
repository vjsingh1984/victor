from types import SimpleNamespace

from victor.teams.types import TeamFormation
from victor.teams.worktree_runtime import WorktreeIsolationPlanner


def test_worktree_planner_builds_assignments_and_manager_last_merge_order():
    planner = WorktreeIsolationPlanner()
    members = [
        SimpleNamespace(id="lead", is_manager=True),
        SimpleNamespace(id="researcher", is_manager=False),
        SimpleNamespace(id="executor", is_manager=False),
    ]

    plan = planner.plan(
        members,
        context={
            "team_name": "Feature Team",
            "worktree_isolation": True,
            "repo_root": "/repo/project",
            "member_write_scopes": {
                "lead": ["src/orchestration"],
                "researcher": ["docs/notes"],
                "executor": ["src/auth"],
            },
            "shared_readonly_paths": ["README.md", "docs/reference"],
        },
        formation=TeamFormation.HIERARCHICAL,
    )

    assert plan is not None
    assert plan.repo_root == "/repo/project"
    assert plan.assignment_for("executor") is not None
    assert plan.assignment_for("executor").claimed_paths == ("src/auth",)
    assert plan.assignment_for("researcher").readonly_paths == ("README.md", "docs/reference")
    assert plan.merge_order[-1] == "lead"
    assert plan.assignment_for("lead").to_context_overrides()["isolation_mode"] == "worktree"
