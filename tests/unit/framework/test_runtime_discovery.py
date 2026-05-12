from __future__ import annotations

from types import SimpleNamespace

from victor.framework.runtime_discovery import (
    CANONICAL_AGENT_MODES,
    effective_runtime_config,
    list_runtime_modes,
    list_runtime_profiles,
)
from victor.framework.session_config import SessionConfig


class _Settings:
    def __init__(self) -> None:
        self.provider = SimpleNamespace(default_provider="ollama", default_model="qwen")

    def load_profiles(self):
        return {
            "default": SimpleNamespace(
                provider="ollama",
                model="qwen",
                description="local",
            ),
            "cloud": SimpleNamespace(
                provider="anthropic",
                model="claude",
                description=None,
            ),
        }


def test_runtime_modes_include_all_user_facing_modes() -> None:
    modes = [mode["name"] for mode in list_runtime_modes()]

    assert modes == list(CANONICAL_AGENT_MODES)
    assert {"build", "plan", "review", "delegate", "explore"}.issubset(set(modes))


def test_runtime_profiles_are_serializable() -> None:
    profiles = list_runtime_profiles(_Settings())  # type: ignore[arg-type]

    assert [profile.name for profile in profiles] == ["cloud", "default"]
    assert profiles[1].to_dict()["provider"] == "ollama"


def test_effective_runtime_config_uses_session_overrides() -> None:
    config = SessionConfig.from_cli_flags(
        agent_profile="cloud",
        provider="openai",
        model="gpt-4o",
        mode="review",
    )

    payload = effective_runtime_config(_Settings(), config)  # type: ignore[arg-type]

    assert payload["profile"] == "cloud"
    assert payload["provider"] == "openai"
    assert payload["model"] == "gpt-4o"
    assert payload["mode"] == "review"
    assert len(payload["profiles"]) == 2
