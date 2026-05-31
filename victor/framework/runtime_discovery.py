"""Shared runtime discovery helpers for CLI, API, and IDE clients."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from victor.config.settings import Settings, get_project_paths
from victor.framework.session_config import SessionConfig

CANONICAL_AGENT_MODES = ("build", "plan", "review", "delegate", "explore")


@dataclass(frozen=True)
class RuntimeProfileInfo:
    """Serializable profile metadata for interactive clients."""

    name: str
    provider: str
    model: str
    is_default: bool = False
    description: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return API-safe profile metadata."""
        return asdict(self)


def _read_default_profile(config_dir: Optional[Path] = None) -> str:
    config_path = config_dir or get_project_paths().global_victor_dir
    profiles_file = config_path / "profiles.yaml"
    if not profiles_file.exists():
        return "default"
    try:
        data = yaml.safe_load(profiles_file.read_text(encoding="utf-8")) or {}
    except Exception:
        return "default"
    return str(data.get("default_profile") or "default")


def list_runtime_profiles(
    settings: Optional[Settings] = None,
) -> list[RuntimeProfileInfo]:
    """List configured runtime profiles in a stable, client-friendly shape."""
    settings = settings or Settings()
    default_profile = _read_default_profile()
    profiles = settings.load_profiles()
    result: list[RuntimeProfileInfo] = []
    for name, profile in sorted(profiles.items()):
        result.append(
            RuntimeProfileInfo(
                name=name,
                provider=str(getattr(profile, "provider", "")),
                model=str(getattr(profile, "model", "")),
                is_default=name == default_profile,
                description=getattr(profile, "description", None),
            )
        )
    return result


def list_runtime_modes() -> list[dict[str, str]]:
    """Return canonical agent modes shared by every user-facing surface."""
    descriptions = {
        "build": "Implement changes with normal tool access.",
        "plan": "Analyze and produce a plan before changing files.",
        "review": "Inspect for bugs, risks, regressions, and missing tests.",
        "delegate": "Coordinate work that may be split across agents.",
        "explore": "Read-only codebase exploration and discovery.",
    }
    return [
        {"name": mode, "description": descriptions.get(mode, "")} for mode in CANONICAL_AGENT_MODES
    ]


def effective_runtime_config(
    settings: Optional[Settings] = None,
    session_config: Optional[SessionConfig] = None,
) -> dict[str, Any]:
    """Describe the effective runtime config without leaking secrets."""
    settings = settings or Settings()
    config = session_config or SessionConfig()
    provider_settings = getattr(settings, "provider", None)
    provider_override = getattr(config, "provider_override", None)
    provider = (
        getattr(provider_override, "provider", None)
        or getattr(provider_settings, "default_provider", None)
        or ""
    )
    model = (
        getattr(provider_override, "model", None)
        or getattr(provider_settings, "default_model", None)
        or ""
    )
    return {
        "profile": config.agent_profile or _read_default_profile(),
        "provider": provider,
        "model": model,
        "mode": config.mode or "build",
        "profiles": [profile.to_dict() for profile in list_runtime_profiles(settings)],
        "modes": list_runtime_modes(),
    }
