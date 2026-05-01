"""Shared framework message execution helpers.

This module centralizes the framework-first execution contract for a single
message turn:
- prepare prompt/context payloads
- resolve the canonical service-backed chat runtime
- normalize runtime responses into TaskResult
- expose a consistent streaming-event iterator

The goal is to keep framework entry points such as Agent and VictorClient
aligned on one execution shape while the lower-level agent/services runtime
continues to own the actual orchestration work.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, AsyncIterator, Mapping, Optional

from victor.core.errors import CancellationError
from victor.framework._internal import format_context_message
from victor.framework.task import DirectResponseOutputState, TaskResult
from victor.providers.base import CompletionResponse
from victor.runtime.chat_runtime import resolve_chat_runtime, resolve_chat_service
from victor.runtime.context import (
    ResolvedRuntimeServices,
    resolve_execution_context,
    resolve_runtime_services,
)


@dataclass(frozen=True)
class PreparedMessage:
    """Canonical prompt payload for a single framework message turn."""

    runtime_message: str
    response_message: str


def prepare_message(
    user_message: str,
    context: Optional[Mapping[str, Any]] = None,
) -> PreparedMessage:
    """Prepare runtime and response messages from user input plus optional context."""
    runtime_message = user_message

    if context:
        context_message = format_context_message(dict(context))
        if context_message:
            runtime_message = f"{context_message}\n\n{user_message}"

    return PreparedMessage(
        runtime_message=runtime_message,
        response_message=user_message,
    )


def _coerce_completion_response(
    result: Any,
    *,
    default_model: Optional[str],
) -> CompletionResponse:
    """Normalize runtime chat results into CompletionResponse."""
    if isinstance(result, CompletionResponse):
        return result

    return CompletionResponse(
        content=str(result),
        model=default_model,
        tool_calls=[],
        usage=None,
        stop_reason="stop",
    )


def _resolve_stage_value(orchestrator: Any) -> str:
    """Best-effort stage value extraction for TaskResult metadata."""
    get_stage = getattr(orchestrator, "get_stage", None)
    if callable(get_stage):
        try:
            stage = get_stage()
            return getattr(stage, "value", str(stage))
        except Exception:
            pass

    stage = getattr(orchestrator, "current_stage", None)
    if stage is not None:
        return getattr(stage, "value", str(stage))

    return "unknown"


def _supports_keyword_argument(callable_obj: Any, argument: str) -> bool:
    """Return whether a callable accepts a given keyword or arbitrary kwargs."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True

    return argument in signature.parameters


async def _invoke_chat(
    runtime: Any,
    message: str,
    *,
    stream: bool = False,
    forward_stream_option: bool = False,
) -> Any:
    """Invoke the runtime chat entrypoint with compatibility-aware kwargs."""
    chat_callable = runtime.chat

    if forward_stream_option and _supports_keyword_argument(chat_callable, "stream"):
        return await chat_callable(message, stream=stream)

    return await chat_callable(message)


async def execute_message(
    *,
    orchestrator: Any,
    execution_context: Any = None,
    user_message: str,
    context: Optional[Mapping[str, Any]] = None,
    stream: bool = False,
    forward_stream_option: bool = False,
) -> TaskResult:
    """Execute a single message turn via the canonical service-backed runtime."""
    prepared = prepare_message(user_message, context)

    try:
        chat_runtime = resolve_chat_runtime(orchestrator, execution_context)
        output_state = DirectResponseOutputState(prepared.response_message)
        response = _coerce_completion_response(
            await _invoke_chat(
                chat_runtime,
                prepared.runtime_message,
                stream=stream,
                forward_stream_option=forward_stream_option,
            ),
            default_model=getattr(orchestrator, "model", "unknown"),
        )
        response_content = output_state.normalize_final_response(response.content or "")

        return TaskResult(
            content=response_content,
            tool_calls=response.tool_calls or [],
            success=True,
            error=None,
            metadata={
                "stage": _resolve_stage_value(orchestrator),
                "model": response.model,
                "usage": response.usage,
                "stop_reason": response.stop_reason,
            },
        )

    except CancellationError:
        return TaskResult(
            content="",
            tool_calls=[],
            success=False,
            error="Operation cancelled",
            metadata={"stage": _resolve_stage_value(orchestrator)},
        )
    except Exception as exc:
        return TaskResult(
            content="",
            tool_calls=[],
            success=False,
            error=str(exc),
            metadata={"stage": _resolve_stage_value(orchestrator)},
        )


async def stream_message_events(
    *,
    orchestrator: Any,
    execution_context: Any = None,
    user_message: str,
    context: Optional[Mapping[str, Any]] = None,
) -> AsyncIterator[Any]:
    """Yield framework stream events for a single message turn."""
    from victor.framework._internal import stream_with_events

    prepared = prepare_message(user_message, context)
    chat_runtime = resolve_chat_runtime(orchestrator, execution_context)

    async for event in stream_with_events(
        chat_runtime,
        prepared.runtime_message,
        response_prompt=prepared.response_message,
    ):
        yield event


async def iter_runtime_stream_events(
    runtime: Any,
    message: str,
) -> AsyncIterator[Any]:
    """Yield framework stream events from either a chat runtime or agent wrapper."""
    from victor.framework._internal import stream_with_events

    if hasattr(runtime, "stream_chat"):
        async for event in stream_with_events(runtime, message, response_prompt=message):
            yield event
        return

    async for event in runtime.stream(message):
        yield event
