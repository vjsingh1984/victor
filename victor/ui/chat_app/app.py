# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chainlit chat surface for Victor.

Run via ``victor ui`` (which shells out to ``chainlit run`` on this file). This module is
imported by Chainlit's runner, not by the framework, so ``chainlit`` is a hard import here
and the dependency is gated behind the optional ``chat-ui`` extra.

Architecture: the UI layer talks to the framework **only** through ``VictorClient`` (per the
UI-layer mandate in CLAUDE.md) — no orchestrator, no settings mutation. It streams
``client.stream()`` in-process and renders Victor's event model with the Chainlit-free
mapping in :mod:`victor.ui.chat_app.event_mapping`.
"""

from __future__ import annotations

import logging
import os

import chainlit as cl

from victor.framework.client import VictorClient
from victor.framework.session_config import SessionConfig
from victor.ui.chat_app.approval import chainlit_approval_handler
from victor.ui.chat_app.event_mapping import RenderKind, map_event
from victor.ui.rendering.markdown_presenters import tool_call_summary, tool_result_markdown

logger = logging.getLogger(__name__)

_CLIENT_KEY = "victor_client"

# Agent profile (from ~/.victor/profiles.yaml) selected via `victor ui --profile`,
# passed through the chainlit subprocess as an env var.
_PROFILE_ENV = "VICTOR_UI_PROFILE"

# Tools that mutate state or run code require explicit approval in the web UI, where a
# user is one click away. Read-only tools stay friction-free.
_ASK_ON_TOOLS = [
    "bash",
    "shell",
    "run_shell",
    "execute_command",
    "exec_command",
    "code_execution",
    "write_file",
    "edit_file",
    "delete_file",
    "git_commit",
    "git_push",
]


def _build_client() -> VictorClient:
    """Construct a VictorClient with human-in-the-loop approval for mutating tools.

    The agent profile (provider/model/etc.) comes from ``victor ui --profile`` via the
    ``VICTOR_UI_PROFILE`` env var; ``None`` uses the default profile. Richer per-session
    controls can be surfaced later via Chainlit ``ChatSettings``.
    """
    profile = os.environ.get(_PROFILE_ENV) or None
    config = SessionConfig.from_cli_flags(
        agent_profile=profile,
        tool_preview=True,
        tool_approval_enabled=True,
        ask_on_tools=_ASK_ON_TOOLS,
        ask_fallback="deny",
    )
    return VictorClient(config)


@cl.on_chat_start
async def on_chat_start() -> None:
    """Create a per-session VictorClient and greet the user."""
    # Register the approval handler before any stream triggers agent/middleware build,
    # while the container is still mutable. The framework owns the container access.
    try:
        from victor.framework.policies import register_policy_approval_handler

        register_policy_approval_handler(chainlit_approval_handler)
    except Exception:  # approval is best-effort; chat still works without it
        logger.debug("Approval handler registration skipped", exc_info=True)

    cl.user_session.set(_CLIENT_KEY, _build_client())
    await cl.Message(
        content="**Victor** is ready. Ask me to write code, search the repo, run tools — "
        "I'll stream my reasoning and tool calls as I work."
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Stream a Victor turn, rendering tokens, reasoning, and tool calls."""
    client: VictorClient | None = cl.user_session.get(_CLIENT_KEY)
    if client is None:  # session expired / restarted
        client = _build_client()
        cl.user_session.set(_CLIENT_KEY, client)

    answer = cl.Message(content="")
    await answer.send()

    # P1.4 first-token feedback: a transient "Thinking…" message so the user never stares at
    # a blank bubble; removed the instant any real output (token/reasoning/tool) arrives.
    thinking = cl.Message(content="🔄 _Thinking…_")
    await thinking.send()
    state = {"thinking_cleared": False}

    async def _clear_thinking() -> None:
        if not state["thinking_cleared"]:
            state["thinking_cleared"] = True
            try:
                await thinking.remove()
            except Exception:
                logger.debug("thinking placeholder removal failed", exc_info=True)

    # P1.2 correlate tool steps by call_id (fallback to name) so parallel/duplicate calls
    # don't overwrite each other's arguments.
    pending: dict[str, dict] = {}
    # P1.3 one accumulating reasoning step per turn (not a new box per chunk).
    reasoning_step: cl.Step | None = None
    reasoning_text = ""

    try:
        async for event in client.stream(message.content):
            action = map_event(event)

            if action.kind is RenderKind.TOKEN:
                await _clear_thinking()
                await answer.stream_token(action.text)

            elif action.kind is RenderKind.THINKING:
                await _clear_thinking()
                if action.text:
                    reasoning_text += action.text
                    if reasoning_step is None:
                        reasoning_step = cl.Step(name="reasoning", type="llm")
                        await reasoning_step.__aenter__()
                    reasoning_step.output = reasoning_text
                    await reasoning_step.update()

            elif action.kind is RenderKind.TOOL_START:
                await _clear_thinking()
                key = action.call_id or action.tool_name or "tool"
                pending[key] = {
                    "tool_name": action.tool_name or "tool",
                    "arguments": action.metadata.get("arguments", {}),
                }

            elif action.kind is RenderKind.TOOL_END:
                await _clear_thinking()
                key = action.call_id or action.tool_name or "tool"
                info = pending.pop(key, {})
                tool_name = info.get("tool_name") or action.tool_name or "tool"
                args = info.get("arguments", {})
                async with cl.Step(name=tool_call_summary(tool_name, args), type="tool") as step:
                    step.input = args
                    # P2.5 rich, CLI-parity result preview (diffs/file reads/search/shell).
                    step.output = tool_result_markdown(
                        tool_name, args, action.text, success=action.success
                    )
                    if not action.success:
                        step.is_error = True

            elif action.kind is RenderKind.ERROR:
                await _clear_thinking()
                await answer.stream_token(f"\n\n⚠️ {action.text}")

    except Exception as exc:  # surface failures in-chat instead of a blank message
        logger.exception("Victor chat turn failed")
        await answer.stream_token(f"\n\n⚠️ Victor hit an error: {exc}")
    finally:
        await _clear_thinking()
        if reasoning_step is not None:
            try:
                await reasoning_step.__aexit__(None, None, None)
            except Exception:
                logger.debug("reasoning step finalize failed", exc_info=True)

    await answer.update()


@cl.on_chat_end
async def on_chat_end() -> None:
    """Release the session's VictorClient resources."""
    client: VictorClient | None = cl.user_session.get(_CLIENT_KEY)
    if client is not None:
        try:
            await client.close()
        except Exception:  # best-effort cleanup
            logger.debug("VictorClient close failed", exc_info=True)
