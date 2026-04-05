"""Focused tests for workflow HITL handler async boundaries."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from victor.workflows.hitl import (
    DefaultHITLHandler,
    HITLNodeType,
    HITLRequest,
    HITLStatus,
)


def _request(hitl_type: HITLNodeType) -> HITLRequest:
    return HITLRequest(
        request_id="req-123",
        node_id="approve",
        hitl_type=hitl_type,
        prompt="Need approval",
        context={"file": "main.py"},
    )


@pytest.mark.asyncio
async def test_default_hitl_handler_uses_to_thread_for_approval_prompt() -> None:
    handler = DefaultHITLHandler()

    async def call_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with (
        patch("builtins.input", return_value="y") as mock_input,
        patch(
            "victor.workflows.hitl.asyncio.to_thread", side_effect=call_to_thread
        ) as mock_to_thread,
    ):
        response = await handler.request_human_input(_request(HITLNodeType.APPROVAL))

    assert response.status is HITLStatus.APPROVED
    assert response.approved is True
    mock_to_thread.assert_awaited_once()
    called = mock_to_thread.await_args
    assert called.args[0] is mock_input
    assert called.args[1] == "Approve? [y/n]: "


@pytest.mark.asyncio
async def test_default_hitl_handler_uses_to_thread_for_freeform_input() -> None:
    handler = DefaultHITLHandler()

    async def call_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with (
        patch("builtins.input", return_value=" ship it ") as mock_input,
        patch(
            "victor.workflows.hitl.asyncio.to_thread", side_effect=call_to_thread
        ) as mock_to_thread,
    ):
        response = await handler.request_human_input(_request(HITLNodeType.INPUT))

    assert response.status is HITLStatus.APPROVED
    assert response.value == "ship it"
    mock_to_thread.assert_awaited_once()
    called = mock_to_thread.await_args
    assert called.args[0] is mock_input
    assert called.args[1] == "Your input: "
