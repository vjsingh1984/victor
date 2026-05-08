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
            "worker_return_contracts": {
                "planner": {
                    "task_summary": "Patched auth service",
                    "changed_files": ["src/auth/service.py"],
                }
            },
        }
    }

    artifacts = extract_team_feedback_artifacts(payload)

    assert artifacts["worktree_plan"]["team_name"] == "feature_team"
    assert artifacts["merge_analysis"]["risk_level"] == "low"
    assert artifacts["worker_return_contracts"]["planner"]["task_summary"] == "Patched auth service"


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
                "worker_return_contracts": {
                    "planner": {
                        "task_summary": "Patched auth service",
                        "changed_files": ["src/auth/service.py"],
                        "validation_run": {"status": "passed", "summary": "1 passed"},
                        "merge_risk": {"level": "low", "reasons": []},
                    },
                    "tester": {
                        "task_summary": "Validated auth tests",
                        "changed_files": ["tests/auth/test_service.py"],
                        "validation_run": {"status": "failed", "summary": "1 failed"},
                        "merge_risk": {"level": "medium", "reasons": ["out_of_scope_writes"]},
                    },
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
    assert summary["has_worker_return_contracts"] is True
    assert summary["worker_contract_count"] == 2
    assert summary["worker_validation_count"] == 2
    assert summary["worker_medium_risk_count"] == 1


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
                    "worker_return_contracts": {
                        "planner": {
                            "task_summary": "Patched auth service",
                            "validation_run": {"status": "passed"},
                            "merge_risk": {"level": "low"},
                        },
                        "tester": {
                            "task_summary": "Validated auth tests",
                            "validation_run": {"status": "passed"},
                            "merge_risk": {"level": "low"},
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
                    "worker_return_contracts": {
                        "planner": {
                            "task_summary": "Patched auth service",
                            "validation_run": {"status": "passed"},
                            "merge_risk": {"level": "high"},
                        },
                        "reviewer": {
                            "task_summary": "Reviewed auth patch",
                            "validation_run": {"status": "failed"},
                            "merge_risk": {"level": "high"},
                        },
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
    assert metrics["team_worker_contract_task_count"] == 2
    assert metrics["team_worker_contract_count"] == 4
    assert metrics["team_worker_validation_count"] == 4
    assert metrics["team_worker_high_risk_count"] == 2
