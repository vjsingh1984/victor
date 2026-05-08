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
            "merge_review_contract": {
                "merge_ready": True,
                "review_required": False,
                "recommended_merge_order": ["planner"],
            },
            "delegate_follow_up_contract": {
                "next_action": "merge",
                "preserve_worktrees": False,
                "fix_validation_queue": [],
                "review_queue": [],
            },
        }
    }

    artifacts = extract_team_feedback_artifacts(payload)

    assert artifacts["worktree_plan"]["team_name"] == "feature_team"
    assert artifacts["merge_analysis"]["risk_level"] == "low"
    assert artifacts["worker_return_contracts"]["planner"]["task_summary"] == "Patched auth service"
    assert artifacts["merge_review_contract"]["merge_ready"] is True
    assert artifacts["delegate_follow_up_contract"]["next_action"] == "merge"


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
                "merge_review_contract": {
                    "merge_ready": False,
                    "review_required": True,
                    "recommended_merge_order": ["planner", "tester"],
                    "review_required_members": ["tester"],
                    "validation_failed_members": ["tester"],
                    "blocking_issues": [
                        {"type": "validation_failed", "member_id": "tester"},
                        {"type": "merge_risk_medium", "member_id": "tester"},
                    ],
                },
                "delegate_follow_up_contract": {
                    "next_action": "fix_validation",
                    "preserve_worktrees": True,
                    "fix_validation_queue": [
                        {
                            "member_id": "tester",
                            "validation_command": "python -m pytest tests/unit/auth/test_service.py",
                        }
                    ],
                    "review_queue": [
                        {
                            "member_id": "tester",
                            "merge_risk_level": "medium",
                        }
                    ],
                    "reentry_contract": {
                        "mode": "delegate",
                        "next_action": "fix_validation",
                        "retry_member_ids": ["tester"],
                        "resume_worktree_paths": {"tester": "/tmp/feature-team-tester"},
                        "retry_tasks_by_member": {
                            "tester": (
                                "Fix the failing validation run for tester. Re-run "
                                "`python -m pytest tests/unit/auth/test_service.py`."
                            )
                        },
                    },
                    "approval_contract": {
                        "required": False,
                        "reason": "validation_failed",
                        "recommended_action": "retry",
                        "recommended_mode": "manual_review",
                        "resume_ready": True,
                        "auto_retry_eligible": True,
                        "merge_executed": False,
                        "target_member_ids": ["tester"],
                        "summary": (
                            "Resume preserved worktrees to fix failing validation for: tester."
                        ),
                        "next_steps": [
                            {
                                "step": "resume_delegate_retry",
                                "instruction": (
                                    "Resume preserved worktrees to fix failing validation for: "
                                    "tester."
                                ),
                                "target_member_ids": ["tester"],
                                "requires_approval": False,
                            }
                        ],
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
    assert summary["has_merge_review_contract"] is True
    assert summary["has_delegate_follow_up_contract"] is True
    assert summary["merge_ready"] is False
    assert summary["review_required"] is True
    assert summary["merge_next_action"] == "fix_validation"
    assert summary["delegate_follow_up_next_action"] == "fix_validation"
    assert summary["delegate_follow_up_preserve_worktrees"] is True
    assert summary["fix_validation_queue_count"] == 1
    assert summary["review_queue_count"] == 1
    assert summary["has_delegate_reentry_contract"] is True
    assert summary["has_delegate_approval_contract"] is True
    assert summary["delegate_reentry_next_action"] == "fix_validation"
    assert summary["delegate_reentry_member_count"] == 1
    assert summary["delegate_reentry_resume_worktree_count"] == 1
    assert summary["delegate_approval_required"] is False
    assert summary["delegate_approval_reason"] == "validation_failed"
    assert summary["delegate_approval_action"] == "retry"
    assert summary["delegate_auto_retry_eligible"] is True
    assert summary["delegate_resume_ready"] is True
    assert summary["delegate_approval_target_count"] == 1
    assert summary["delegate_approval_has_resume_context"] is True
    assert summary["delegate_approval_task_brief_count"] == 1
    assert summary["delegate_approval_step_count"] == 1
    assert summary["delegate_approval_primary_step"] == "resume_delegate_retry"
    assert summary["review_required_member_count"] == 1
    assert summary["merge_blocker_count"] == 2


def test_summarize_team_feedback_derives_approval_steps_for_legacy_artifacts():
    summary = summarize_team_feedback(
        {
            "metadata": {
                "delegate_follow_up_contract": {
                    "next_action": "fix_validation",
                    "preserve_worktrees": True,
                    "fix_validation_queue": [{"member_id": "tester"}],
                    "review_queue": [],
                    "reentry_contract": {
                        "mode": "delegate",
                        "next_action": "fix_validation",
                        "retry_member_ids": ["tester"],
                        "resume_worktree_paths": {"tester": "/tmp/feature-team-tester"},
                        "retry_tasks_by_member": {"tester": "Fix the failing validation run."},
                    },
                    "approval_contract": {
                        "required": False,
                        "reason": "validation_failed",
                        "recommended_action": "retry",
                        "resume_ready": True,
                        "auto_retry_eligible": True,
                        "merge_executed": False,
                        "target_member_ids": ["tester"],
                        "summary": (
                            "Resume preserved worktrees to fix failing validation for: tester."
                        ),
                    },
                }
            }
        }
    )

    assert summary is not None
    assert summary["delegate_approval_step_count"] == 1
    assert summary["delegate_approval_primary_step"] == "resume_delegate_retry"


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
                    "merge_review_contract": {
                        "merge_ready": True,
                        "review_required": False,
                        "recommended_merge_order": ["planner", "tester"],
                        "review_required_members": [],
                        "blocking_issues": [],
                    },
                    "delegate_follow_up_contract": {
                        "next_action": "merge",
                        "preserve_worktrees": False,
                        "fix_validation_queue": [],
                        "review_queue": [],
                        "reentry_contract": {
                            "mode": "delegate",
                            "next_action": "merge",
                            "retry_member_ids": [],
                            "resume_worktree_paths": {},
                        },
                        "approval_contract": {
                            "required": True,
                            "reason": "merge_ready",
                            "recommended_action": "approve_merge",
                            "recommended_mode": "auto_apply_safe",
                            "resume_ready": False,
                            "auto_retry_eligible": False,
                            "merge_executed": False,
                            "target_member_ids": ["planner", "tester"],
                            "summary": (
                                "Review and approve merge execution for: planner, tester."
                            ),
                            "next_steps": [
                                {
                                    "step": "approve_merge_execution",
                                    "instruction": (
                                        "Review and approve merge execution for: planner, tester."
                                    ),
                                    "target_member_ids": ["planner", "tester"],
                                    "requires_approval": True,
                                }
                            ],
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
                    "merge_review_contract": {
                        "merge_ready": False,
                        "review_required": True,
                        "recommended_merge_order": ["planner", "reviewer"],
                        "review_required_members": ["planner", "reviewer"],
                        "validation_failed_members": ["reviewer"],
                        "blocking_issues": [
                            {"type": "validation_failed", "member_id": "reviewer"},
                            {"type": "merge_risk_high", "member_id": "planner"},
                            {"type": "merge_risk_high", "member_id": "reviewer"},
                        ],
                    },
                    "delegate_follow_up_contract": {
                        "next_action": "fix_validation",
                        "preserve_worktrees": True,
                        "fix_validation_queue": [{"member_id": "reviewer"}],
                        "review_queue": [{"member_id": "planner"}, {"member_id": "reviewer"}],
                        "reentry_contract": {
                            "mode": "delegate",
                            "next_action": "fix_validation",
                            "retry_member_ids": ["reviewer"],
                            "resume_worktree_paths": {
                                "planner": "/tmp/feature-team-planner",
                                "reviewer": "/tmp/feature-team-reviewer",
                            },
                            "retry_tasks_by_member": {
                                "reviewer": "Fix the failing validation run for reviewer."
                            },
                        },
                        "approval_contract": {
                            "required": False,
                            "reason": "validation_failed",
                            "recommended_action": "retry",
                            "recommended_mode": "manual_review",
                            "resume_ready": True,
                            "auto_retry_eligible": True,
                            "merge_executed": False,
                            "target_member_ids": ["reviewer"],
                            "summary": (
                                "Resume preserved worktrees to fix failing validation for: reviewer."
                            ),
                            "next_steps": [
                                {
                                    "step": "resume_delegate_retry",
                                    "instruction": (
                                        "Resume preserved worktrees to fix failing validation for: "
                                        "reviewer."
                                    ),
                                    "target_member_ids": ["reviewer"],
                                    "requires_approval": False,
                                }
                            ],
                        },
                    },
                    "worktree_cleanup": {
                        "removed": [],
                        "errors": [],
                        "skipped": ["/tmp/feature-team-planner", "/tmp/feature-team-reviewer"],
                        "reason": "preserved_for_follow_up",
                    },
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
    assert metrics["team_cleanup_error_task_count"] == 0
    assert metrics["team_cleanup_error_count"] == 0
    assert metrics["avg_team_assignments"] == 2.0
    assert metrics["avg_team_members_with_changes"] == 2.0
    assert metrics["team_materialized_assignment_total"] == 4
    assert metrics["team_worker_contract_task_count"] == 2
    assert metrics["team_worker_contract_count"] == 4
    assert metrics["team_worker_validation_count"] == 4
    assert metrics["team_worker_high_risk_count"] == 2
    assert metrics["team_merge_review_contract_task_count"] == 2
    assert metrics["team_merge_ready_task_count"] == 1
    assert metrics["team_merge_ready_rate"] == 0.5
    assert metrics["team_merge_next_actions"] == {"merge": 1, "fix_validation": 1}
    assert metrics["team_delegate_follow_up_task_count"] == 2
    assert metrics["team_delegate_follow_up_actions"] == {"merge": 1, "fix_validation": 1}
    assert metrics["team_delegate_approval_task_count"] == 2
    assert metrics["team_delegate_approval_required_task_count"] == 1
    assert metrics["team_delegate_auto_retry_eligible_task_count"] == 1
    assert metrics["team_delegate_approval_actions"] == {"approve_merge": 1, "retry": 1}
    assert metrics["team_delegate_approval_reasons"] == {
        "merge_ready": 1,
        "validation_failed": 1,
    }
    assert metrics["team_delegate_approval_primary_steps"] == {
        "approve_merge_execution": 1,
        "resume_delegate_retry": 1,
    }
    assert metrics["team_delegate_resume_context_task_count"] == 1
    assert metrics["team_preserved_worktree_task_count"] == 1
    assert metrics["team_delegate_reentry_task_count"] == 2
    assert metrics["team_delegate_reentry_actions"] == {"merge": 1, "fix_validation": 1}
    assert metrics["team_review_required_task_count"] == 1
    assert metrics["team_review_required_rate"] == 0.5
    assert metrics["team_merge_blocker_count"] == 3
    assert metrics["avg_team_materialized_assignments"] == 2.0
    assert metrics["avg_worker_validations_per_task"] == 2.0
    assert metrics["avg_team_merge_blockers"] == 1.5
    assert metrics["avg_fix_validation_queue_length"] == 0.5
    assert metrics["avg_review_queue_length"] == 1.0
    assert metrics["avg_delegate_approval_target_count"] == 1.5
    assert metrics["avg_delegate_approval_task_brief_count"] == 0.5
    assert metrics["avg_delegate_approval_step_count"] == 1.0
    assert metrics["avg_delegate_reentry_member_count"] == 0.5
    assert metrics["avg_delegate_reentry_resume_worktree_count"] == 1.0
    assert metrics["avg_changed_files_per_materialized_assignment"] == 1.0
