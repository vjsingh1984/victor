from __future__ import annotations

import json

import pytest

from victor.ui.delegate_follow_up import (
    DelegateFollowUpContractError,
    build_delegate_follow_up_suggestions,
    load_delegate_follow_up_contract_file,
)


def test_build_delegate_follow_up_suggestions_uses_next_steps() -> None:
    contract = {
        "next_steps": [
            {
                "step_id": "review_worktrees",
                "step": "review_worktrees",
                "instruction": "Review merge risks before retrying preserved worktrees for: m1.",
            },
            {
                "step_id": "resume_delegate_retry",
                "step": "resume_delegate_retry",
                "instruction": "Resume preserved worktrees after review for: m1.",
            },
        ]
    }

    suggestions = build_delegate_follow_up_suggestions(
        workflow_path="workflows/delegate-resume.yaml",
        contract_path="delegate-follow-up.json",
        contract=contract,
    )

    assert suggestions == [
        {
            "command": (
                "/delegate-follow-up workflows/delegate-resume.yaml "
                "delegate-follow-up.json review_worktrees"
            ),
            "description": "review_worktrees: Review merge risks before retrying preserved worktrees for: m1.",
            "step_id": "review_worktrees",
            "requires_approval": None,
        },
        {
            "command": (
                "/delegate-follow-up workflows/delegate-resume.yaml "
                "delegate-follow-up.json resume_delegate_retry"
            ),
            "description": "resume_delegate_retry: Resume preserved worktrees after review for: m1.",
            "step_id": "resume_delegate_retry",
            "requires_approval": None,
        },
    ]


def test_build_delegate_follow_up_suggestions_falls_back_to_approval_contract() -> None:
    contract = {
        "approval_contract": {
            "next_steps": [
                {
                    "step_id": "approve_merge_execution",
                    "instruction": "Review and approve merge execution for: m1.",
                    "requires_approval": True,
                }
            ]
        }
    }

    suggestions = build_delegate_follow_up_suggestions(
        workflow_path="workflows/delegate-resume.yaml",
        contract_path="delegate-follow-up.json",
        contract=contract,
    )

    assert suggestions[0]["step_id"] == "approve_merge_execution"
    assert suggestions[0]["requires_approval"] is True
    assert "approve_merge_execution" in suggestions[0]["command"]
    assert "Review and approve merge execution" in suggestions[0]["description"]


def test_build_delegate_follow_up_suggestions_rejects_contract_without_steps() -> None:
    with pytest.raises(DelegateFollowUpContractError):
        build_delegate_follow_up_suggestions(
            workflow_path="workflows/delegate-resume.yaml",
            contract_path="delegate-follow-up.json",
            contract={"next_action": "inspect"},
        )


def test_load_delegate_follow_up_contract_file_rejects_non_object(tmp_path) -> None:
    contract_path = tmp_path / "delegate-follow-up.json"
    contract_path.write_text(json.dumps(["not", "object"]))

    with pytest.raises(DelegateFollowUpContractError):
        load_delegate_follow_up_contract_file(contract_path)
