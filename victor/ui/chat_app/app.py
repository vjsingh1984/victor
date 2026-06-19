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

import chainlit as cl

from victor.framework.client import VictorClient
from victor.framework.session_config import SessionConfig
from victor.ui.chat_app.approval import register_chat_ui_approval_handler
from victor.ui.chat_app.event_mapping import RenderKind, map_event

logger = logging.getLogger(__name__)

_CLIENT_KEY = "victor_client"

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

    Overrides are intentionally minimal here; richer per-session controls (provider, model,
    tool budget) can be surfaced later via Chainlit ``ChatSettings``.
    """
    config = SessionConfig.from_cli_flags(
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
    # while the global container is still mutable.
    try:
        from victor.core import get_container

        register_chat_ui_approval_handler(get_container())
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

    # Arguments captured at tool_call time, replayed when the matching result arrives.
    pending_args: dict[str, dict] = {}

    try:
        async for event in client.stream(message.content):
            action = map_event(event)

            if action.kind is RenderKind.TOKEN:
                await answer.stream_token(action.text)

            elif action.kind is RenderKind.THINKING:
                if action.text:
                    async with cl.Step(name="reasoning", type="llm") as step:
                        step.output = action.text

            elif action.kind is RenderKind.TOOL_START:
                pending_args[action.tool_name or "tool"] = action.metadata.get("arguments", {})

            elif action.kind is RenderKind.TOOL_END:
                name = action.tool_name or "tool"
                async with cl.Step(name=name, type="tool") as step:
                    step.input = pending_args.pop(name, {})
                    step.output = action.text
                    if not action.success:
                        step.is_error = True

            elif action.kind is RenderKind.ERROR:
                await answer.stream_token(f"\n\n⚠️ {action.text}")

    except Exception as exc:  # surface failures in-chat instead of a blank message
        logger.exception("Victor chat turn failed")
        await answer.stream_token(f"\n\n⚠️ Victor hit an error: {exc}")

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
