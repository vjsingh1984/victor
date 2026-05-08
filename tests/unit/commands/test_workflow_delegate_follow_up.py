"""Tests for workflow CLI delegate follow-up resume support."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import typer

import victor.ui.commands.workflow as workflow_cmd


def _workflow() -> SimpleNamespace:
    return SimpleNamespace(
        name="delegate-resume",
        description="delegate resume workflow",
        start_node="resume",
        nodes={},
        metadata={},
    )


def test_load_delegate_follow_up_contract_file_requires_object(tmp_path):
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(["not", "an", "object"]))

    with pytest.raises(typer.Exit):
        workflow_cmd._load_delegate_follow_up_contract_file(contract_path)


def test_run_workflow_injects_delegate_follow_up_contract_into_graph_state(tmp_path):
    contract = {
        "primary_step_id": "resume_delegate_retry",
        "next_steps": [{"step_id": "resume_delegate_retry", "step": "retry failed tests"}],
    }
    contract_path = tmp_path / "delegate-follow-up.json"
    contract_path.write_text(json.dumps(contract))
    workflow = _workflow()
    coro = object()
    mock_async = Mock(return_value=coro)

    with (
        patch.object(
            workflow_cmd,
            "_load_workflow_file",
            return_value={"delegate-resume": workflow},
        ),
        patch.object(workflow_cmd, "_display_workflow_info"),
        patch.object(workflow_cmd.console, "print"),
        patch.object(workflow_cmd, "_execute_workflow_async", mock_async),
        patch.object(workflow_cmd, "run_sync", return_value=None) as mock_run_sync,
    ):
        workflow_cmd.run_workflow(
            "workflow.yaml",
            context='{"existing": "state"}',
            context_file=None,
            delegate_follow_up_contract=str(contract_path),
            delegate_next_step_id="resume_delegate_retry",
            workflow_name=None,
            profile=None,
            dry_run=False,
            log_level=None,
        )

    expected_context = {
        "existing": "state",
        "delegate_follow_up_contract": contract,
        "delegate_next_step_id": "resume_delegate_retry",
    }
    mock_async.assert_called_once_with(workflow, expected_context, None)
    mock_run_sync.assert_called_once_with(coro)


def test_run_workflow_rejects_delegate_step_without_contract():
    workflow = _workflow()

    with (
        patch.object(
            workflow_cmd,
            "_load_workflow_file",
            return_value={"delegate-resume": workflow},
        ),
        patch.object(workflow_cmd.console, "print"),
        pytest.raises(typer.Exit),
    ):
        workflow_cmd.run_workflow(
            "workflow.yaml",
            context=None,
            context_file=None,
            delegate_follow_up_contract=None,
            delegate_next_step_id="resume_delegate_retry",
            workflow_name=None,
            profile=None,
            dry_run=False,
            log_level=None,
        )
