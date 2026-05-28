from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from victor.ui.slash.commands.delegate_follow_up import DelegateFollowUpCommand
from victor.ui.slash.protocol import CommandContext


def _ctx(args: list[str]) -> CommandContext:
    return CommandContext(
        console=Console(file=io.StringIO()),
        settings=MagicMock(),
        agent=None,
        args=args,
    )


def _ctx_with_buffer(args: list[str]) -> tuple[CommandContext, io.StringIO]:
    output = io.StringIO()
    return (
        CommandContext(
            console=Console(file=output),
            settings=MagicMock(),
            agent=None,
            args=args,
        ),
        output,
    )


def test_delegate_follow_up_command_metadata() -> None:
    cmd = DelegateFollowUpCommand()
    meta = cmd.metadata

    assert meta.name == "delegate-follow-up"
    assert meta.aliases == ["dfu"]
    assert meta.category == "workflow"
    assert meta.requires_agent is False
    assert meta.is_async is True


@pytest.mark.asyncio
async def test_delegate_follow_up_command_routes_through_workflow_mode() -> None:
    cmd = DelegateFollowUpCommand()

    with patch(
        "victor.ui.slash.commands.delegate_follow_up.run_workflow_mode",
        new=AsyncMock(),
    ) as run_workflow_mode:
        await cmd.execute(
            _ctx(
                [
                    "workflows/delegate-resume.yaml",
                    "delegate-follow-up.json",
                    "resume_delegate_retry",
                ]
            )
        )

    run_workflow_mode.assert_awaited_once_with(
        workflow_path="workflows/delegate-resume.yaml",
        delegate_follow_up_contract="delegate-follow-up.json",
        delegate_next_step_id="resume_delegate_retry",
    )


@pytest.mark.asyncio
async def test_delegate_follow_up_command_without_step_uses_primary_contract_step() -> (
    None
):
    cmd = DelegateFollowUpCommand()

    with patch(
        "victor.ui.slash.commands.delegate_follow_up.run_workflow_mode",
        new=AsyncMock(),
    ) as run_workflow_mode:
        await cmd.execute(
            _ctx(["workflows/delegate-resume.yaml", "delegate-follow-up.json"])
        )

    run_workflow_mode.assert_awaited_once_with(
        workflow_path="workflows/delegate-resume.yaml",
        delegate_follow_up_contract="delegate-follow-up.json",
        delegate_next_step_id=None,
    )


@pytest.mark.asyncio
async def test_delegate_follow_up_command_shows_help_when_args_missing() -> None:
    cmd = DelegateFollowUpCommand()

    with patch(
        "victor.ui.slash.commands.delegate_follow_up.run_workflow_mode",
        new=AsyncMock(),
    ) as run_workflow_mode:
        await cmd.execute(_ctx(["workflows/delegate-resume.yaml"]))

    run_workflow_mode.assert_not_called()


@pytest.mark.asyncio
async def test_delegate_follow_up_list_prints_selectable_step_commands(
    tmp_path,
) -> None:
    contract_path = tmp_path / "delegate-follow-up.json"
    contract_path.write_text(
        json.dumps(
            {
                "next_steps": [
                    {
                        "step_id": "review_worktrees",
                        "instruction": "Review merge risks before retrying preserved worktrees.",
                    },
                    {
                        "step_id": "resume_delegate_retry",
                        "instruction": "Resume preserved worktrees after review.",
                    },
                ]
            }
        )
    )
    ctx, output = _ctx_with_buffer(
        ["list", "workflows/delegate-resume.yaml", str(contract_path)]
    )
    cmd = DelegateFollowUpCommand()

    with patch(
        "victor.ui.slash.commands.delegate_follow_up.run_workflow_mode",
        new=AsyncMock(),
    ) as run_workflow_mode:
        await cmd.execute(ctx)

    run_workflow_mode.assert_not_called()
    rendered = output.getvalue()
    assert "review_worktrees" in rendered
    assert "resume_delegate_retry" in rendered
    assert "/delegate-follow-up workflows/delegate-resume.yaml" in rendered
