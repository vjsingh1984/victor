"""Shared planning-intent helpers for chat runtimes."""

from __future__ import annotations

EXPLICIT_PLANNING_MARKERS: tuple[str, ...] = (
    "/plan",
    "plan mode",
    "in plan mode",
    "use planning",
    "show a plan",
    "make a plan",
    "create a plan",
    "give me a plan",
    "checklist first",
    "first checklist",
    "show me a checklist first",
    "show me the checklist first",
    "create a checklist first",
    "creating a checklist first",
    "make a checklist first",
    "give me a checklist first",
)


def is_explicit_planning_request(user_message: str) -> bool:
    """Return whether the user explicitly asked to see planning output first."""
    message_lower = user_message.lower()
    return any(marker in message_lower for marker in EXPLICIT_PLANNING_MARKERS)
