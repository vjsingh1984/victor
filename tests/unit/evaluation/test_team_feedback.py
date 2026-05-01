from __future__ import annotations

from victor.evaluation.team_feedback import (
    aggregate_team_feedback,
    extract_team_feedback_artifacts,
    summarize_team_feedback,
)


def test_extract_team_feedback_artifacts_reads_nested_metadata():
    payload = {
        "metadata": {
            "worktree_plan": {
                "team_name": "feature_team",
                "formation": "parallel",
                "assignments": [{"member_id": "planner", "claimed_paths": ["src/auth"]}],
            },
            "merge_analysis": {
                "risk_level": "low",
                "conflict_count": 0,
                "member_changed_files": {"planner": ["src/auth/service.py"]},
            },
        }
    }

    artifacts = extract_team_feedback_artifacts(payload)

    assert artifacts["worktree_plan"]["team_name"] == "feature_team"
    assert artifacts["merge_analysis"]["risk_level"] == "low"


def test_summarize_team_feedback_returns_task_level_summary():
    summary = summarize_team_feedback(
        {
            "metadata": {
                "worktree_plan": {
                    "team_name": "feature_team",
                    "formation": "parallel",
                    "shared_readonly_paths": ["docs"],
                    "merge_order": ["planner", "tester"],
                    "assignments": [
                        {"member_id": "planner", "claimed_paths": ["src/auth"]},
                        {"member_id": "tester", "claimed_paths": ["tests/auth"]},
                    ],
                },
                "worktree_session": {
                    "materialized": True,
                    "dry_run": False,
                    "assignments": [
                        {"member_id": "planner", "materialized": True},
                        {"member_id": "tester", "materialized": True},
                    ],
                },
                "merge_analysis": {
                    "risk_level": "medium",
                    "conflict_count": 1,
                    "overlapping_files": [{"path": "src/auth/service.py"}],
                    "member_changed_files": {
                        "planner": ["src/auth/service.py"],
                        "tester": ["tests/auth/test_service.py"],
                    },
                    "out_of_scope_writes": {"tester": ["src/auth/service.py"]},
                    "readonly_violations": {},
                },
                "worktree_cleanup": {
                    "removed": ["/tmp/feature-team-planner", "/tmp/feature-team-tester"],
                    "errors": [],
                    "skipped": [],
                },
            }
        }
    )

    assert summary is not None
    assert summary["team_name"] == "feature_team"
    assert summary["formation"] == "parallel"
    assert summary["assignment_count"] == 2
    assert summary["scoped_member_count"] == 2
    assert summary["materialized"] is True
    assert summary["materialized_assignment_count"] == 2
    assert summary["merge_risk_level"] == "medium"
    assert summary["merge_conflict_count"] == 1
    assert summary["merge_overlap_count"] == 1
    assert summary["out_of_scope_write_count"] == 1
    assert summary["members_with_changes"] == 2
    assert summary["changed_file_count"] == 2
    assert summary["cleanup_removed_count"] == 2
    assert summary["cleanup_error_count"] == 0
    assert summary["merge_order"] == ["planner", "tester"]


def test_aggregate_team_feedback_rolls_up_materialization_and_risk():
    metrics = aggregate_team_feedback(
        [
            {
                "metadata": {
                    "worktree_plan": {
                        "formation": "parallel",
                        "assignments": [
                            {"member_id": "planner", "claimed_paths": ["src/auth"]},
                            {"member_id": "tester", "claimed_paths": ["tests/auth"]},
                        ],
                    },
                    "worktree_session": {
                        "materialized": True,
                        "assignments": [
                            {"member_id": "planner", "materialized": True},
                            {"member_id": "tester", "materialized": True},
                        ],
                    },
                    "merge_analysis": {
                        "risk_level": "low",
                        "conflict_count": 0,
                        "member_changed_files": {
                            "planner": ["src/auth/service.py"],
                            "tester": ["tests/auth/test_service.py"],
                        },
                    },
                    "worktree_cleanup": {"removed": ["/tmp/feature-team-planner"], "errors": []},
                }
            },
            {
                "metadata": {
                    "worktree_plan": {
                        "formation": "parallel",
                        "assignments": [
                            {"member_id": "planner", "claimed_paths": ["src/auth"]},
                            {"member_id": "reviewer", "claimed_paths": ["src/auth"]},
                        ],
                    },
                    "worktree_session": {
                        "materialized": True,
                        "dry_run": False,
                        "assignments": [
                            {"member_id": "planner", "materialized": True},
                            {"member_id": "reviewer", "materialized": True},
                        ],
                    },
                    "merge_analysis": {
                        "risk_level": "high",
                        "conflict_count": 2,
                        "overlapping_files": [{"path": "src/auth/service.py"}],
                        "member_changed_files": {
                            "planner": ["src/auth/service.py"],
                            "reviewer": ["src/auth/service.py"],
                        },
                        "readonly_violations": {"reviewer": ["docs/guide.md"]},
                    },
                    "worktree_cleanup": {"removed": [], "errors": ["cleanup failed"]},
                }
            },
            {"status": "passed"},
        ],
        total_tasks=3,
    )

    assert metrics["tasks_with_team_feedback"] == 2
    assert metrics["team_feedback_coverage"] == 0.6667
    assert metrics["team_formations"] == {"parallel": 2}
    assert metrics["team_merge_risk_levels"] == {"low": 1, "high": 1}
    assert metrics["team_worktree_plan_count"] == 2
    assert metrics["team_worktree_materialized_count"] == 2
    assert metrics["team_low_risk_task_count"] == 1
    assert metrics["team_high_risk_task_count"] == 1
    assert metrics["team_merge_conflict_task_count"] == 1
    assert metrics["team_merge_conflict_count"] == 2
    assert metrics["team_merge_overlap_task_count"] == 1
    assert metrics["team_readonly_violation_task_count"] == 1
    assert metrics["team_readonly_violation_count"] == 1
    assert metrics["team_cleanup_task_count"] == 2
    assert metrics["team_cleanup_error_task_count"] == 1
    assert metrics["team_cleanup_error_count"] == 1
    assert metrics["avg_team_assignments"] == 2.0
    assert metrics["avg_team_members_with_changes"] == 2.0
    assert metrics["team_materialized_assignment_total"] == 4
