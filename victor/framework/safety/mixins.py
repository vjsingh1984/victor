"""Focused ISP-compliant mixins for vertical extensions.

Instead of one monolithic extension loader, verticals compose only
the mixins they need.
"""

from __future__ import annotations

from typing import Any, Optional


class SafetyExtensionMixin:
    """Mixin for verticals that provide safety extensions."""

    _safety_extension_cache: Optional[Any] = None

    @classmethod
    def get_safety_extension(cls) -> Optional[Any]:
        """Load and cache the safety extension."""
        if cls._safety_extension_cache is not None:
            return cls._safety_extension_cache
        ext = cls._load_safety_extension()
        cls._safety_extension_cache = ext
        return ext

    @classmethod
    def _load_safety_extension(cls) -> Optional[Any]:
        """Override to provide safety extension instance."""
        return None


class PromptExtensionMixin:
    """Mixin for verticals that contribute prompts."""

    _prompt_contributor_cache: Optional[Any] = None

    @classmethod
    def get_prompt_contributor(cls) -> Optional[Any]:
        if cls._prompt_contributor_cache is not None:
            return cls._prompt_contributor_cache
        ext = cls._load_prompt_contributor()
        cls._prompt_contributor_cache = ext
        return ext

    @classmethod
    def _load_prompt_contributor(cls) -> Optional[Any]:
        return None


class ModeConfigMixin:
    """Mixin for verticals that provide mode configuration."""

    _mode_config_cache: Optional[Any] = None

    @classmethod
    def get_mode_config(cls) -> Optional[Any]:
        if cls._mode_config_cache is not None:
            return cls._mode_config_cache
        ext = cls._load_mode_config()
        cls._mode_config_cache = ext
        return ext

    @classmethod
    def _load_mode_config(cls) -> Optional[Any]:
        return None


class RLConfigMixin:
    """Mixin for verticals that provide RL configuration."""

    _rl_config_cache: Optional[Any] = None

    @classmethod
    def get_rl_config(cls) -> Optional[Any]:
        if cls._rl_config_cache is not None:
            return cls._rl_config_cache
        ext = cls._load_rl_config()
        cls._rl_config_cache = ext
        return ext

    @classmethod
    def _load_rl_config(cls) -> Optional[Any]:
        return None


class TeamSpecMixin:
    """Mixin for verticals that provide team specifications."""

    _team_spec_cache: Optional[Any] = None

    @classmethod
    def get_team_spec(cls) -> Optional[Any]:
        if cls._team_spec_cache is not None:
            return cls._team_spec_cache
        ext = cls._load_team_spec()
        cls._team_spec_cache = ext
        return ext

    @classmethod
    def _load_team_spec(cls) -> Optional[Any]:
        return None


class WorkflowMixin:
    """Mixin for verticals that provide workflows."""

    _workflow_provider_cache: Optional[Any] = None

    @classmethod
    def get_workflow_provider(cls) -> Optional[Any]:
        if cls._workflow_provider_cache is not None:
            return cls._workflow_provider_cache
        ext = cls._load_workflow_provider()
        cls._workflow_provider_cache = ext
        return ext

    @classmethod
    def _load_workflow_provider(cls) -> Optional[Any]:
        return None
