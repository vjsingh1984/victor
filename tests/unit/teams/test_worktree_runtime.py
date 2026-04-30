from pathlib import Path
import subprocess
from types import SimpleNamespace

from victor.teams.types import TeamFormation
from victor.teams.worktree_runtime import GitWorktreeRuntime, WorktreeIsolationPlanner


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


def _init_git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "team-runtime@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Team Runtime"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    (path / "README.md").write_text("seed\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True)


def _write_and_commit(worktree_path: Path, relative_path: str, content: str, message: str) -> None:
    target = worktree_path / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    subprocess.run(["git", "add", relative_path], cwd=worktree_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=worktree_path, check=True, capture_output=True)


def test_git_worktree_runtime_materializes_collects_changes_and_cleans_up(tmp_path):
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)

    planner = WorktreeIsolationPlanner()
    plan = planner.plan(
        [SimpleNamespace(id="worker", is_manager=False)],
        context={
            "team_name": "Feature Team",
            "worktree_isolation": True,
            "repo_root": str(repo_root),
            "worktree_parent": str(tmp_path / "worktrees"),
            "member_write_scopes": {"worker": ["src/auth"]},
        },
        formation=TeamFormation.PARALLEL,
    )
    assert plan is not None

    runtime = GitWorktreeRuntime()
    session = runtime.materialize(plan)

    assert session.materialized is True
    assignment = session.assignment_for("worker")
    assert assignment is not None
    assert Path(assignment.worktree_path).exists()

    changed_file = Path(assignment.worktree_path) / "src" / "auth" / "service.py"
    changed_file.parent.mkdir(parents=True, exist_ok=True)
    changed_file.write_text("print('ok')\n")

    changed_files = runtime.collect_changed_files(session, "worker")
    assert changed_files == ("src/auth/service.py",)

    orchestration = runtime.build_merge_orchestration(
        session,
        merge_analysis={"risk_level": "low", "recommended_merge_order": ["worker"]},
    )
    assert orchestration["branches"]["worker"] == assignment.branch_name

    cleanup = runtime.cleanup(session)
    assert cleanup["errors"] == []
    assert not Path(assignment.worktree_path).exists()


def test_git_worktree_runtime_executes_guarded_merge_for_disjoint_commits(tmp_path):
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)

    planner = WorktreeIsolationPlanner()
    plan = planner.plan(
        [
            SimpleNamespace(id="worker_a", is_manager=False),
            SimpleNamespace(id="worker_b", is_manager=False),
        ],
        context={
            "team_name": "Feature Team",
            "worktree_isolation": True,
            "repo_root": str(repo_root),
            "worktree_parent": str(tmp_path / "worktrees"),
            "member_write_scopes": {
                "worker_a": ["src/auth"],
                "worker_b": ["tests/auth"],
            },
        },
        formation=TeamFormation.PARALLEL,
    )
    assert plan is not None

    runtime = GitWorktreeRuntime()
    session = runtime.materialize(plan)
    assignment_a = session.assignment_for("worker_a")
    assignment_b = session.assignment_for("worker_b")
    assert assignment_a is not None
    assert assignment_b is not None

    _write_and_commit(
        Path(assignment_a.worktree_path),
        "src/auth/service.py",
        "print('a')\n",
        "worker a change",
    )
    _write_and_commit(
        Path(assignment_b.worktree_path),
        "tests/auth/test_service.py",
        "print('b')\n",
        "worker b change",
    )

    execution = runtime.execute_merge_orchestration(
        session,
        merge_analysis={"risk_level": "low", "recommended_merge_order": ["worker_a", "worker_b"]},
    )

    assert execution["status"] == "success"
    assert execution["executed"] is True
    assert execution["merged_members"] == ["worker_a", "worker_b"]
    assert execution["cleanup"]["branch_deleted"] is True

    runtime.cleanup(session)


def test_git_worktree_runtime_blocks_risky_merge_without_override(tmp_path):
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)

    planner = WorktreeIsolationPlanner()
    plan = planner.plan(
        [SimpleNamespace(id="worker", is_manager=False)],
        context={
            "team_name": "Feature Team",
            "worktree_isolation": True,
            "repo_root": str(repo_root),
            "worktree_parent": str(tmp_path / "worktrees"),
            "member_write_scopes": {"worker": ["src/auth"]},
        },
        formation=TeamFormation.PARALLEL,
    )
    assert plan is not None

    runtime = GitWorktreeRuntime()
    session = runtime.materialize(plan)

    execution = runtime.execute_merge_orchestration(
        session,
        merge_analysis={"risk_level": "high", "recommended_merge_order": ["worker"]},
    )

    assert execution["status"] == "blocked"
    assert execution["executed"] is False
    assert execution["blocked_reason"] == "merge_risk_high"

    runtime.cleanup(session)
