from unittest.mock import MagicMock

from victor.runtime.context import ExecutionContext


def test_with_prompt_orchestrator_creates_typed_runtime_binding() -> None:
    ctx = ExecutionContext.create(MagicMock(), MagicMock(), session_id="ctx")
    prompt_orchestrator = object()

    new_ctx = ctx.with_prompt_orchestrator(prompt_orchestrator)

    assert new_ctx.prompt_orchestrator is prompt_orchestrator
    assert new_ctx.metadata["prompt_orchestrator"] is prompt_orchestrator
    assert ctx.prompt_orchestrator is None


def test_with_prompt_orchestrator_is_idempotent_for_same_binding() -> None:
    prompt_orchestrator = object()
    ctx = ExecutionContext.create(MagicMock(), MagicMock()).with_prompt_orchestrator(
        prompt_orchestrator
    )

    same_ctx = ctx.with_prompt_orchestrator(prompt_orchestrator)

    assert same_ctx is ctx
