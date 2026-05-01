# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Session configuration — immutable capture of CLI/runtime overrides.

This module provides ``SessionConfig``, a single dataclass that captures
all runtime overrides (CLI flags, environment variables, programmatic args)
in an **immutable** container.  The framework consumes it via
``Agent.create(session_config=...)`` instead of mutating ``Settings`` directly.

Design Rationale:
    - **Immutability**: CLI flags produce a ``SessionConfig`` once; the
      framework reads it but never mutates it.
    - **Traceability**: You can inspect ``session_config`` to see exactly
      what the caller overrode — no hidden ``settings.xxx = yyy`` mutations.
    - **Thread safety**: ``frozen=True`` dataclass is safe to share across
      coroutines without locking.

Migration Path:
    **OLD (deprecated):**
        settings.tool_settings.tool_output_preview_enabled = False
        settings.smart_routing_enabled = True

    **NEW:**
        from victor.framework.session_config import SessionConfig

        config = SessionConfig(
            tool_output_preview=False,
            smart_routing=True,
        )
        agent = await Agent.create(session_config=config)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

DEFAULT_PROVIDER_MODELS = {
    "anthropic": "claude-3-5-sonnet-20241022",
    "openai": "gpt-4o",
    "google": "gemini-2.0-flash-exp",
    "ollama": "qwen2.5-coder:7b",
    "lmstudio": "local-model",
    "vllm": "local-model",
    "deepseek": "deepseek-chat",
    "xai": "grok-beta",
    "zai": "glm-4.7",
    "cohere": "command-r-plus",
    "azure": "gpt-4o",
}

LOCAL_ENDPOINT_PROVIDERS = frozenset({"ollama", "lmstudio", "vllm", "mlx", "llama.cpp"})


@dataclass(frozen=True)
class CompactionConfig:
    """Compaction threshold overrides for a session.

    Attributes:
        threshold: Compaction threshold (0.1-0.95). Lower = earlier compaction.
        adaptive: Enable adaptive threshold based on conversation patterns.
        min_threshold: Minimum adaptive threshold (0.1-0.8).
        max_threshold: Maximum adaptive threshold (0.2-0.95).
    """

    threshold: Optional[float] = None
    adaptive: Optional[bool] = None
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None


@dataclass(frozen=True)
class SmartRoutingConfig:
    """Smart provider routing overrides for a session.

    Attributes:
        enabled: Enable smart routing across providers.
        profile: Routing profile (e.g. 'balanced', 'performance', 'cost').
        fallback_chain: Comma-separated fallback provider chain.
    """

    enabled: bool = False
    profile: str = "balanced"
    fallback_chain: Optional[str] = None


@dataclass(frozen=True)
class ToolOutputConfig:
    """Tool output preview/pruning overrides for a session.

    Attributes:
        preview_enabled: Show tool output previews.
        pruning_enabled: Enable tool output pruning.
        pruning_safe_only: Only prune safe (read-heavy) tool outputs.
    """

    preview_enabled: bool = True
    pruning_enabled: bool = False
    pruning_safe_only: bool = True


@dataclass(frozen=True)
class ProviderOverrideConfig:
    """Explicit provider/model/runtime override state for a session."""

    provider: Optional[str] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    auth_mode: Optional[str] = None
    timeout: Optional[int] = None
    coding_plan: bool = False

    @property
    def is_active(self) -> bool:
        """Return True when any explicit provider override is present."""
        return any(
            (
                self.provider is not None,
                self.model is not None,
                self.endpoint is not None,
                self.auth_mode is not None,
                self.timeout is not None,
                self.coding_plan,
            )
        )

    @classmethod
    def from_cli(
        cls,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        auth_mode: Optional[str] = None,
        timeout: Optional[int] = None,
        coding_plan: bool = False,
    ) -> "ProviderOverrideConfig":
        """Normalize provider override flags from CLI-friendly inputs."""
        normalized_provider = provider.lower() if provider else None
        normalized_model = model
        normalized_auth_mode = auth_mode.lower() if auth_mode else None

        if endpoint and normalized_provider and normalized_provider not in LOCAL_ENDPOINT_PROVIDERS:
            raise ValueError(
                "--endpoint is only supported for local providers "
                "(ollama, lmstudio, vllm, mlx, llama.cpp)."
            )
        if timeout is not None and timeout <= 0:
            raise ValueError("--provider-timeout must be greater than 0.")

        if normalized_provider and not normalized_model:
            normalized_model = DEFAULT_PROVIDER_MODELS.get(normalized_provider)
            if normalized_model is None:
                raise ValueError(
                    f"No default model known for provider '{normalized_provider}'. "
                    "Please specify --model."
                )

        # Preserve existing CLI behavior: auth/coding-plan only participate in the
        # explicit override path when at least one provider-selection flag is set.
        has_selection = any((normalized_provider, normalized_model, endpoint))
        if not has_selection:
            normalized_auth_mode = None
            coding_plan = False

        return cls(
            provider=normalized_provider,
            model=normalized_model,
            endpoint=endpoint,
            auth_mode=normalized_auth_mode,
            timeout=timeout,
            coding_plan=coding_plan,
        )

    def to_profile_overrides(self) -> dict[str, object]:
        """Return extra profile payload fields for AgentFactory synthesis."""
        overrides: dict[str, object] = {}
        if self.endpoint:
            overrides["base_url"] = self.endpoint
        if self.auth_mode:
            overrides["auth_mode"] = self.auth_mode
        if self.timeout is not None:
            overrides["timeout"] = self.timeout
        if self.coding_plan:
            overrides["coding_plan"] = True
        return overrides


@dataclass(frozen=True)
class SessionConfig:
    """Immutable capture of all CLI/runtime session overrides.

    This is the **single config object** that the CLI (or any client)
    produces and passes to ``Agent.create()``.  The framework reads it
    but never mutates it — eliminating scattered ``settings.xxx = yyy``
    mutations throughout the codebase.

    Attributes:
        agent_profile: Agent profile name from ~/.victor/profiles.yaml (e.g., 'zai-coding', 'default').
        tool_budget: Override tool call budget for this session.
        max_iterations: Override maximum iterations for this session.
        compaction: Compaction threshold overrides.
        smart_routing: Smart provider routing overrides.
        tool_output: Tool output preview/pruning overrides.
        planning_enabled: Enable structured planning for complex tasks.
        planning_model: Override model for planning tasks.
        mode: Initial agent mode ('build', 'plan', 'explore').
        show_reasoning: Show LLM reasoning/thinking content.
        provider_override: Explicit provider/model/endpoint override state.
        tool_preview: Shorthand to disable tool output preview.
        enable_pruning: Shorthand to enable broader tool output pruning.
        enable_smart_routing: Shorthand to enable smart routing.
        routing_profile: Shorthand for routing profile.
        fallback_chain: Shorthand for fallback provider chain.
        compaction_threshold: Shorthand for compaction threshold.
        adaptive_threshold: Shorthand for adaptive compaction toggle.
        compaction_min_threshold: Shorthand for adaptive min threshold.
        compaction_max_threshold: Shorthand for adaptive max threshold.
    """

    # Agent profile (from ~/.victor/profiles.yaml)
    agent_profile: Optional[str] = None

    # Agent behaviour
    tool_budget: Optional[int] = None
    max_iterations: Optional[int] = None
    planning_enabled: Optional[bool] = None
    planning_model: Optional[str] = None
    mode: Optional[str] = None
    show_reasoning: bool = False
    observability_logging: Optional[bool] = None
    auto_skill_enabled: Optional[bool] = None
    one_shot_mode: Optional[bool] = None

    # Composed sub-configs (for structured access)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    smart_routing: SmartRoutingConfig = field(default_factory=SmartRoutingConfig)
    tool_output: ToolOutputConfig = field(default_factory=ToolOutputConfig)
    provider_override: ProviderOverrideConfig = field(default_factory=ProviderOverrideConfig)

    # --- Convenience shorthands (populate sub-configs) ---

    def __post_init__(self) -> None:
        """Merge shorthand flags into composed sub-configs.

        Because the dataclass is frozen, we use ``object.__setattr__``
        to populate the composed configs from flat shorthand fields.
        """
        # Tool output shorthands
        if not self.tool_output.preview_enabled or not self.tool_output.pruning_enabled:
            pass  # Already set via tool_output

    @classmethod
    def from_cli_flags(
        cls,
        *,
        agent_profile: Optional[str] = None,
        tool_budget: Optional[int] = None,
        max_iterations: Optional[int] = None,
        compaction_threshold: Optional[float] = None,
        adaptive_threshold: Optional[bool] = None,
        compaction_min_threshold: Optional[float] = None,
        compaction_max_threshold: Optional[float] = None,
        enable_smart_routing: bool = False,
        routing_profile: str = "balanced",
        fallback_chain: Optional[str] = None,
        tool_preview: bool = True,
        enable_pruning: bool = False,
        planning_enabled: Optional[bool] = None,
        planning_model: Optional[str] = None,
        mode: Optional[str] = None,
        show_reasoning: bool = False,
        observability_logging: Optional[bool] = None,
        auto_skill_enabled: Optional[bool] = None,
        one_shot_mode: Optional[bool] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        auth_mode: Optional[str] = None,
        provider_timeout: Optional[int] = None,
        coding_plan: bool = False,
    ) -> "SessionConfig":
        """Create a ``SessionConfig`` from flat CLI flags.

        This is the primary factory for CLI code — collect all Typer
        options and pass them here to get an immutable config object.

        Args:
            agent_profile: Agent profile name from ~/.victor/profiles.yaml.
            tool_budget: Override tool call budget.
            max_iterations: Override max iterations.
            compaction_threshold: Compaction threshold (0.1-0.95).
            adaptive_threshold: Enable adaptive compaction.
            compaction_min_threshold: Adaptive min threshold.
            compaction_max_threshold: Adaptive max threshold.
            enable_smart_routing: Enable smart routing.
            routing_profile: Routing profile name.
            fallback_chain: Fallback provider chain.
            tool_preview: Show tool output previews.
            enable_pruning: Enable broader tool output pruning.
            planning_enabled: Enable structured planning.
            planning_model: Override model for planning.
            mode: Agent mode (build/plan/explore).
            show_reasoning: Show LLM reasoning.
            observability_logging: Enable event/observability logging for this session.
            auto_skill_enabled: Override skill auto-selection for this session.
            one_shot_mode: Override headless one-shot execution mode.
            provider: Override provider for this session.
            model: Override model for this session.
            endpoint: Override endpoint for local providers.
            auth_mode: Override provider auth mode.
            provider_timeout: Override provider request timeout in seconds.
            coding_plan: Enable provider-specific coding-plan endpoint mode.

        Returns:
            Immutable ``SessionConfig`` instance.

        Example::

            config = SessionConfig.from_cli_flags(
                tool_budget=50,
                enable_smart_routing=True,
                tool_preview=False,
            )
            agent = await Agent.create(session_config=config)
        """
        return cls(
            agent_profile=agent_profile,
            tool_budget=tool_budget,
            max_iterations=max_iterations,
            planning_enabled=planning_enabled,
            planning_model=planning_model,
            mode=mode,
            show_reasoning=show_reasoning,
            observability_logging=observability_logging,
            auto_skill_enabled=auto_skill_enabled,
            one_shot_mode=one_shot_mode,
            compaction=CompactionConfig(
                threshold=compaction_threshold,
                adaptive=adaptive_threshold,
                min_threshold=compaction_min_threshold,
                max_threshold=compaction_max_threshold,
            ),
            smart_routing=SmartRoutingConfig(
                enabled=enable_smart_routing,
                profile=routing_profile,
                fallback_chain=fallback_chain,
            ),
            tool_output=ToolOutputConfig(
                preview_enabled=tool_preview,
                pruning_enabled=enable_pruning,
                pruning_safe_only=not enable_pruning,
            ),
            provider_override=ProviderOverrideConfig.from_cli(
                provider=provider,
                model=model,
                endpoint=endpoint,
                auth_mode=auth_mode,
                timeout=provider_timeout,
                coding_plan=coding_plan,
            ),
        )

    def apply_to_settings(self, settings: object) -> None:
        """Apply session overrides to a Settings object.

        This is the **only** place where Settings mutation should happen
        from session config.  All CLI code should call this method instead
        of directly mutating ``settings.xxx = yyy``.

        Args:
            settings: Application Settings instance.
        """
        # Tool output settings
        tool_settings = getattr(settings, "tool_settings", None)
        if tool_settings is not None:
            if hasattr(tool_settings, "tool_output_preview_enabled"):
                object.__setattr__(
                    tool_settings,
                    "tool_output_preview_enabled",
                    self.tool_output.preview_enabled,
                )
            if hasattr(tool_settings, "tool_output_pruning_enabled"):
                object.__setattr__(
                    tool_settings,
                    "tool_output_pruning_enabled",
                    self.tool_output.pruning_enabled,
                )
            if hasattr(tool_settings, "tool_output_pruning_safe_only"):
                object.__setattr__(
                    tool_settings,
                    "tool_output_pruning_safe_only",
                    self.tool_output.pruning_safe_only,
                )

        observability_settings = getattr(settings, "observability", None)
        if self.observability_logging is not None:
            if observability_settings is not None and hasattr(
                observability_settings, "enable_observability_logging"
            ):
                object.__setattr__(
                    observability_settings,
                    "enable_observability_logging",
                    self.observability_logging,
                )
            if hasattr(settings, "enable_observability_logging"):
                object.__setattr__(
                    settings,
                    "enable_observability_logging",
                    self.observability_logging,
                )

        if self.auto_skill_enabled is not None and hasattr(settings, "skill_auto_select_enabled"):
            object.__setattr__(
                settings,
                "skill_auto_select_enabled",
                self.auto_skill_enabled,
            )

        if self.one_shot_mode is not None:
            automation = getattr(settings, "automation", None)
            if automation is not None and hasattr(automation, "one_shot_mode"):
                object.__setattr__(automation, "one_shot_mode", self.one_shot_mode)
            if hasattr(settings, "one_shot_mode"):
                object.__setattr__(settings, "one_shot_mode", self.one_shot_mode)

        provider_override = self.provider_override
        provider_settings = getattr(settings, "provider", None)
        if provider_settings is not None:
            if provider_override.provider and hasattr(provider_settings, "default_provider"):
                object.__setattr__(
                    provider_settings,
                    "default_provider",
                    provider_override.provider,
                )
            if provider_override.model and hasattr(provider_settings, "default_model"):
                object.__setattr__(
                    provider_settings,
                    "default_model",
                    provider_override.model,
                )
            if provider_override.timeout is not None and hasattr(provider_settings, "timeout"):
                object.__setattr__(
                    provider_settings,
                    "timeout",
                    provider_override.timeout,
                )
            if provider_override.endpoint:
                if provider_override.provider == "ollama" and hasattr(
                    provider_settings, "ollama_base_url"
                ):
                    object.__setattr__(
                        provider_settings,
                        "ollama_base_url",
                        provider_override.endpoint,
                    )
                elif provider_override.provider == "lmstudio" and hasattr(
                    provider_settings, "lmstudio_base_urls"
                ):
                    object.__setattr__(
                        provider_settings,
                        "lmstudio_base_urls",
                        [provider_override.endpoint],
                    )
                elif provider_override.provider == "vllm" and hasattr(
                    provider_settings, "vllm_base_url"
                ):
                    object.__setattr__(
                        provider_settings,
                        "vllm_base_url",
                        provider_override.endpoint,
                    )

        # Smart routing settings
        if self.smart_routing.enabled:
            if hasattr(settings, "smart_routing_enabled"):
                object.__setattr__(settings, "smart_routing_enabled", True)
            routing = getattr(settings, "routing", None)
            if routing is not None:
                if hasattr(routing, "profile"):
                    object.__setattr__(routing, "profile", self.smart_routing.profile)
                if hasattr(routing, "fallback_chain") and self.smart_routing.fallback_chain:
                    object.__setattr__(routing, "fallback_chain", self.smart_routing.fallback_chain)
