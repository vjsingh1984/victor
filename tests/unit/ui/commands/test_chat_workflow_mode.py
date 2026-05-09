from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer

from victor.ui.commands.chat import run_workflow_mode
from victor.workflows.definition import TransformNode, WorkflowDefinition


def _build_workflow(name: str = "sample") -> WorkflowDefinition:
    return WorkflowDefinition(
        name=name,
        nodes={
            "start": TransformNode(
                id="start",
                name="Start",
                transform=lambda ctx: ctx,
            )
        },
        start_node="start",
    )


@pytest.mark.asyncio
async def test_run_workflow_mode_validate_only_uses_canonical_compile_helper(tmp_path) -> None:
    workflow = _build_workflow()
    compiler = MagicMock(name="compiler")
    workflow_path = tmp_path / "sample.yaml"
    workflow_path.write_text("workflows: {}\n")

    with (
        patch("victor.ui.commands.chat.setup_logging"),
        patch(
            "victor.ui.commands.chat.load_workflow_from_file",
            return_value={workflow.name: workflow},
        ),
        patch(
            "victor.ui.commands.chat._create_compile_only_compiler",
            return_value=compiler,
        ) as create_compiler,
        patch("victor.ui.commands.chat.console.print"),
    ):
        await run_workflow_mode(
            str(workflow_path),
            validate_only=True,
        )

    create_compiler.assert_called_once_with()
    compiler.compile_definition.assert_called_once_with(workflow)


@pytest.mark.asyncio
async def test_run_workflow_mode_validate_only_exits_on_canonical_compile_failure(tmp_path) -> None:
    workflow = _build_workflow()
    compiler = MagicMock(name="compiler")
    compiler.compile_definition.side_effect = ValueError("boom")
    workflow_path = tmp_path / "sample.yaml"
    workflow_path.write_text("workflows: {}\n")

    with (
        patch("victor.ui.commands.chat.setup_logging"),
        patch(
            "victor.ui.commands.chat.load_workflow_from_file",
            return_value={workflow.name: workflow},
        ),
        patch(
            "victor.ui.commands.chat._create_compile_only_compiler",
            return_value=compiler,
        ),
        patch("victor.ui.commands.chat.console.print"),
        pytest.raises(typer.Exit) as exc_info,
    ):
        await run_workflow_mode(
            str(workflow_path),
            validate_only=True,
        )

    assert exc_info.value.exit_code == 1


@pytest.mark.asyncio
async def test_run_workflow_mode_injects_delegate_follow_up_contract(tmp_path) -> None:
    workflow = _build_workflow("delegate-resume")
    compiler = MagicMock(name="compiler")
    workflow_path = tmp_path / "delegate-resume.yaml"
    workflow_path.write_text("workflows: {}\n")
    contract = {
        "primary_step_id": "resume_delegate_retry",
        "next_steps": [{"step_id": "resume_delegate_retry", "step": "retry failed tests"}],
    }
    contract_path = tmp_path / "delegate-follow-up.json"
    contract_path.write_text(json.dumps(contract))
    result = SimpleNamespace(
        success=True,
        duration_seconds=0.1,
        nodes_executed=["resume"],
        iterations=1,
        state={},
        error=None,
    )
    executor = MagicMock()
    executor.execute = AsyncMock(return_value=result)

    with (
        patch("victor.ui.commands.chat.setup_logging"),
        patch(
            "victor.ui.commands.chat.load_workflow_from_file",
            return_value={workflow.name: workflow},
        ),
        patch(
            "victor.ui.commands.chat._create_compile_only_compiler",
            return_value=compiler,
        ),
        patch("victor.ui.commands.chat.StateGraphExecutor", return_value=executor),
        patch("victor.ui.commands.chat.console.print"),
    ):
        await run_workflow_mode(
            str(workflow_path),
            delegate_follow_up_contract=str(contract_path),
            delegate_next_step_id="resume_delegate_retry",
        )

    executor.execute.assert_awaited_once_with(
        workflow,
        {
            "delegate_follow_up_contract": contract,
            "delegate_next_step_id": "resume_delegate_retry",
        },
    )


@pytest.mark.asyncio
async def test_run_workflow_mode_rejects_delegate_step_without_contract(tmp_path) -> None:
    workflow = _build_workflow("delegate-resume")
    compiler = MagicMock(name="compiler")
    workflow_path = tmp_path / "delegate-resume.yaml"
    workflow_path.write_text("workflows: {}\n")

    with (
        patch("victor.ui.commands.chat.setup_logging"),
        patch(
            "victor.ui.commands.chat.load_workflow_from_file",
            return_value={workflow.name: workflow},
        ),
        patch(
            "victor.ui.commands.chat._create_compile_only_compiler",
            return_value=compiler,
        ),
        patch("victor.ui.commands.chat.console.print"),
        pytest.raises(typer.Exit) as exc_info,
    ):
        await run_workflow_mode(
            str(workflow_path),
            delegate_next_step_id="resume_delegate_retry",
        )

    assert exc_info.value.exit_code == 1
