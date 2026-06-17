from __future__ import annotations

import pytest

from victor.ui.commands.chat import (
    ADVANCED_CODING_AGENT_MODES,
    DEFAULT_CODING_AGENT_MODES,
    normalize_chat_mode,
)


def test_default_coding_agent_modes_are_narrow_and_ordered() -> None:
    assert DEFAULT_CODING_AGENT_MODES == ("build", "plan", "review", "delegate")


@pytest.mark.parametrize(
    ("raw_mode", "expected"),
    [
        (None, None),
        ("", None),
        ("  ", None),
        ("build", "build"),
        ("PLAN", "plan"),
        (" review ", "review"),
        ("delegate", "delegate"),
    ],
)
def test_normalize_chat_mode_accepts_default_coding_modes(
    raw_mode: str | None,
    expected: str | None,
) -> None:
    assert normalize_chat_mode(raw_mode) == expected


def test_normalize_chat_mode_accepts_explore_as_advanced_opt_in() -> None:
    assert ADVANCED_CODING_AGENT_MODES == ("explore",)
    assert normalize_chat_mode("EXPLORE") == "explore"


def test_normalize_chat_mode_rejects_unknown_with_default_and_advanced_guidance() -> None:
    with pytest.raises(ValueError) as exc_info:
        normalize_chat_mode("framework")

    message = str(exc_info.value)
    assert "build, plan, review, delegate" in message
    assert "Advanced opt-in modes: explore" in message
