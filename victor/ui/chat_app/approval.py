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

"""Human-in-the-loop tool approval for the Chainlit chat UI.

When ``SessionConfig.tool_approval.enabled`` is set, the governance policy engine routes
the configured tools through the ASK path and calls a ``PolicyApprovalHandler`` registered
in the DI container. This module provides that handler for the web UI: it renders a
Chainlit ``AskActionMessage`` (Approve / Reject) and maps the click back to an
``ApprovalStatus``.

The decision mapping (:func:`decision_from_action`) is Chainlit-free so it can be unit
tested without the optional ``chat-ui`` extra installed; only :func:`chainlit_approval_handler`
imports ``chainlit`` (lazily, inside the coroutine).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from victor.framework.hitl import ApprovalRequest, ApprovalStatus

logger = logging.getLogger(__name__)

_RESPONDER = "chat_ui_user"

# The handler returns the framework's ApprovalHandler tuple:
# (status, response_message, responder_identity).
ApprovalResult = Tuple[ApprovalStatus, Optional[str], Optional[str]]

_APPROVE_VALUE = "approve"
_REJECT_VALUE = "reject"


def decision_from_action(action: Optional[Any]) -> ApprovalResult:
    """Map a Chainlit action payload (or ``None`` on timeout) to an approval result.

    ``action`` is the value returned by ``AskActionMessage.send()``: a dict-like with a
    ``"value"`` key, or ``None`` when the user did not respond before the timeout. Anything
    that isn't an explicit approve maps to a non-approval (fail-safe).
    """
    if action is None:
        return ApprovalStatus.TIMEOUT, "No response before timeout", _RESPONDER

    value = action.get("value") if hasattr(action, "get") else getattr(action, "value", None)
    if value == _APPROVE_VALUE:
        return ApprovalStatus.APPROVED, "Approved in chat UI", _RESPONDER
    return ApprovalStatus.REJECTED, "Rejected in chat UI", _RESPONDER


def _format_prompt(request: ApprovalRequest) -> str:
    """Build the approval prompt markdown from the request."""
    lines = [f"**Tool approval required** — {request.title}"]
    if request.description:
        lines.append("")
        lines.append(request.description)
    tool = (request.context or {}).get("tool_name")
    if tool:
        lines.append("")
        lines.append(f"`{tool}`")
    return "\n".join(lines)


async def chainlit_approval_handler(request: ApprovalRequest) -> ApprovalResult:
    """Prompt the user to approve/reject a tool via Chainlit; return the decision.

    Registered in the DI container as a ``PolicyApprovalHandler`` so the policy engine
    invokes it for ASK verdicts. Blocks the tool call until the user answers (or the
    request times out).
    """
    import chainlit as cl

    try:
        action = await cl.AskActionMessage(
            content=_format_prompt(request),
            actions=[
                cl.Action(name="approve", value=_APPROVE_VALUE, label="✅ Approve"),
                cl.Action(name="reject", value=_REJECT_VALUE, label="🚫 Reject"),
            ],
            timeout=request.timeout_seconds or 120,
        ).send()
    except Exception:  # elicitation failed -> fail safe (reject)
        logger.exception("Chat UI approval prompt failed; rejecting")
        return ApprovalStatus.REJECTED, "Approval prompt failed", _RESPONDER

    return decision_from_action(action)


def register_chat_ui_approval_handler(container: Any) -> bool:
    """Register :func:`chainlit_approval_handler` as the container's PolicyApprovalHandler.

    Idempotent and defensive: returns ``True`` if a handler is registered (or already
    present), ``False`` if registration was not possible (e.g. the container is frozen and
    no handler is registered yet). Must be called before the agent builds its middleware
    (i.e. before the first ``VictorClient.stream``), while the container is still mutable.
    """
    try:
        from victor.framework.policies import PolicyApprovalHandler
    except Exception:  # policy engine not available
        logger.debug("PolicyApprovalHandler unavailable; approval not registered", exc_info=True)
        return False

    try:
        if container.get_optional(PolicyApprovalHandler) is not None:
            return True  # already registered (e.g. a prior session in this process)
    except Exception:
        pass

    try:
        container.register(PolicyApprovalHandler, PolicyApprovalHandler(chainlit_approval_handler))
        return True
    except Exception:
        logger.debug("Could not register chat UI approval handler", exc_info=True)
        return False
