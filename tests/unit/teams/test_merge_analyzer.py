from victor.teams.merge_analyzer import MergeAnalyzer, MergeRiskLevel
from victor.teams.types import MemberResult, TeamFormation
from victor.teams.worktree_runtime import WorktreeAssignment, WorktreeExecutionPlan


def _build_plan() -> WorktreeExecutionPlan:
    return WorktreeExecutionPlan(
        team_name="feature_team",
        repo_root="/repo/project",
        parent_dir="/repo/project/.victor/team_worktrees",
        base_ref="HEAD",
        branch_prefix="victor/feature-team",
        formation=TeamFormation.PARALLEL,
        assignments=(
            WorktreeAssignment(
                member_id="m1",
                branch_name="victor/feature-team/m1-1",
                worktree_name="feature-team-m1",
                worktree_path="/repo/project/.victor/team_worktrees/feature-team-m1",
                claimed_paths=("src/auth",),
                readonly_paths=("docs",),
                merge_priority=0,
            ),
            WorktreeAssignment(
                member_id="m2",
                branch_name="victor/feature-team/m2-2",
                worktree_name="feature-team-m2",
                worktree_path="/repo/project/.victor/team_worktrees/feature-team-m2",
                claimed_paths=("tests/auth",),
                readonly_paths=("docs",),
                merge_priority=1,
            ),
        ),
        merge_order=("m1", "m2"),
        shared_readonly_paths=("docs",),
    )


def test_merge_analyzer_reports_low_risk_for_disjoint_changes():
    analyzer = MergeAnalyzer()
    plan = _build_plan()
    member_results = {
        "m1": MemberResult(
            member_id="m1",
            success=True,
            output="done",
            metadata={"changed_files": ["src/auth/service.py"]},
        ),
        "m2": MemberResult(
            member_id="m2",
            success=True,
            output="done",
            metadata={"changed_files": ["tests/auth/test_service.py"]},
        ),
    }

    analysis = analyzer.analyze(member_results, worktree_plan=plan)

    assert analysis.risk_level == MergeRiskLevel.LOW
    assert analysis.conflict_count == 0
    assert analysis.recommended_merge_order == ("m1", "m2")


def test_merge_analyzer_reports_high_risk_for_overlap_and_readonly_violation():
    analyzer = MergeAnalyzer()
    plan = _build_plan()
    member_results = {
        "m1": MemberResult(
            member_id="m1",
            success=True,
            output="done",
            metadata={"changed_files": ["src/auth/service.py", "docs/architecture.md"]},
        ),
        "m2": MemberResult(
            member_id="m2",
            success=True,
            output="done",
            metadata={"changed_files": ["src/auth/service.py", "src/auth/helpers.py"]},
        ),
    }

    analysis = analyzer.analyze(member_results, worktree_plan=plan)

    assert analysis.risk_level == MergeRiskLevel.HIGH
    assert analysis.conflict_count >= 2
    assert analysis.overlapping_files[0].path == "src/auth/service.py"
    assert analysis.readonly_violations["m1"] == ("docs/architecture.md",)
    assert analysis.out_of_scope_writes["m2"] == (
        "src/auth/service.py",
        "src/auth/helpers.py",
    )
