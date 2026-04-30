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
class SessionConfig:
    """Immutable capture of all CLI/runtime session overrides.

    This is the **single config object** that the CLI (or any client)
    produces and passes to ``Agent.create()``.  The framework reads it
    but never mutates it — eliminating scattered ``settings.xxx = yyy``
    mutations throughout the codebase.

    Attributes:
        tool_budget: Override tool call budget for this session.
        max_iterations: Override maximum iterations for this session.
        compaction: Compaction threshold overrides.
        smart_routing: Smart provider routing overrides.
        tool_output: Tool output preview/pruning overrides.
        planning_enabled: Enable structured planning for complex tasks.
        planning_model: Override model for planning tasks.
        mode: Initial agent mode ('build', 'plan', 'explore').
        show_reasoning: Show LLM reasoning/thinking content.
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

    # Agent behaviour
    tool_budget: Optional[int] = None
    max_iterations: Optional[int] = None
    planning_enabled: Optional[bool] = None
    planning_model: Optional[str] = None
    mode: Optional[str] = None
    show_reasoning: bool = False

    # Composed sub-configs (for structured access)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    smart_routing: SmartRoutingConfig = field(default_factory=SmartRoutingConfig)
    tool_output: ToolOutputConfig = field(default_factory=ToolOutputConfig)

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
    ) -> "SessionConfig":
        """Create a ``SessionConfig`` from flat CLI flags.

        This is the primary factory for CLI code — collect all Typer
        options and pass them here to get an immutable config object.

        Args:
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
            tool_budget=tool_budget,
            max_iterations=max_iterations,
            planning_enabled=planning_enabled,
            planning_model=planning_model,
            mode=mode,
            show_reasoning=show_reasoning,
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
