from __future__ import annotations

from unittest.mock import MagicMock, patch

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
