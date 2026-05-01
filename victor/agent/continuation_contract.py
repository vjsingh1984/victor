"""Typed continuation contract shared across streaming runtime layers.

This module provides the canonical continuation action enum and the
state-passed directive object used between continuation strategy,
intent classification, and continuation execution.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from victor.agent.decisions.schemas import ContinuationAction
from victor.agent.tool_call_extractor import ExtractedToolCall

# Backward-compatible alias for historical imports.
ContinuationActionType = ContinuationAction


@dataclass
class ContinuationStatePatch:
    """Typed state updates produced by continuation decisions."""

    continuation_prompts: Optional[int] = None
    asking_input_prompts: Optional[int] = None
    synthesis_nudge_count: Optional[int] = None
    cumulative_prompt_interventions: Optional[int] = None
    final_summary_requested: bool = False
    max_prompts_summary_requested: bool = False

    @classmethod
    def from_legacy(
        cls,
        updates: Optional[Mapping[str, Any]] = None,
        *,
        final_summary_requested: bool = False,
        max_prompts_summary_requested: bool = False,
    ) -> "ContinuationStatePatch":
        """Create a typed state patch from legacy dict-style fields."""

        updates_dict = dict(updates or {})
        return cls(
            continuation_prompts=updates_dict.get("continuation_prompts"),
            asking_input_prompts=updates_dict.get("asking_input_prompts"),
            synthesis_nudge_count=updates_dict.get("synthesis_nudge_count"),
            cumulative_prompt_interventions=updates_dict.get("cumulative_prompt_interventions"),
            final_summary_requested=bool(
                final_summary_requested or updates_dict.get("final_summary_requested")
            ),
            max_prompts_summary_requested=bool(
                max_prompts_summary_requested or updates_dict.get("max_prompts_summary_requested")
            ),
        )

    def to_updates_dict(self) -> dict[str, Any]:
        """Return non-empty numeric updates using the legacy shape."""

        updates: dict[str, Any] = {}
        if self.continuation_prompts is not None:
            updates["continuation_prompts"] = self.continuation_prompts
        if self.asking_input_prompts is not None:
            updates["asking_input_prompts"] = self.asking_input_prompts
        if self.synthesis_nudge_count is not None:
            updates["synthesis_nudge_count"] = self.synthesis_nudge_count
        if self.cumulative_prompt_interventions is not None:
            updates["cumulative_prompt_interventions"] = self.cumulative_prompt_interventions
        if self.final_summary_requested:
            updates["final_summary_requested"] = True
        if self.max_prompts_summary_requested:
            updates["max_prompts_summary_requested"] = True
        return updates

    def to_legacy_fields(self) -> dict[str, Any]:
        """Return compatibility fields expected by older runtime code."""

        payload: dict[str, Any] = {"updates": self.to_updates_dict()}
        if self.final_summary_requested:
            payload["set_final_summary_requested"] = True
        if self.max_prompts_summary_requested:
            payload["set_max_prompts_summary_requested"] = True
        return payload


@dataclass
class ContinuationDirective(Mapping[str, Any]):
    """Typed continuation directive with mapping compatibility helpers."""

    action: ContinuationActionType
    reason: str
    message: Optional[str] = None
    state_patch: ContinuationStatePatch = field(default_factory=ContinuationStatePatch)
    extracted_call: Optional[ExtractedToolCall] = None
    mentioned_tools: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_legacy(
        cls,
        *,
        action: Any,
        reason: str,
        message: Optional[str] = None,
        updates: Optional[Mapping[str, Any]] = None,
        extracted_call: Optional[ExtractedToolCall] = None,
        mentioned_tools: Optional[list[str]] = None,
        set_final_summary_requested: bool = False,
        set_max_prompts_summary_requested: bool = False,
    ) -> "ContinuationDirective":
        """Build a typed directive from legacy dict-style arguments."""

        return cls(
            action=coerce_continuation_action(action),
            reason=reason,
            message=message,
            state_patch=ContinuationStatePatch.from_legacy(
                updates,
                final_summary_requested=set_final_summary_requested,
                max_prompts_summary_requested=set_max_prompts_summary_requested,
            ),
            extracted_call=extracted_call,
            mentioned_tools=tuple(mentioned_tools or ()),
        )

    def to_legacy_payload(self) -> dict[str, Any]:
        """Return the legacy dict shape for compatibility callers."""

        payload: dict[str, Any] = {
            "action": self.action,
            "message": self.message,
            "reason": self.reason,
            **self.state_patch.to_legacy_fields(),
        }
        if self.extracted_call is not None:
            payload["extracted_call"] = self.extracted_call
        if self.mentioned_tools:
            payload["mentioned_tools"] = list(self.mentioned_tools)
        return payload

    def with_action(self, action: ContinuationActionType) -> "ContinuationDirective":
        """Return a copy with a different action value."""

        return ContinuationDirective(
            action=coerce_continuation_action(action),
            reason=self.reason,
            message=self.message,
            state_patch=self.state_patch,
            extracted_call=self.extracted_call,
            mentioned_tools=self.mentioned_tools,
        )

    def __getitem__(self, key: str) -> Any:
        return self.to_legacy_payload()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_legacy_payload())

    def __len__(self) -> int:
        return len(self.to_legacy_payload())

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_legacy_payload().get(key, default)


def coerce_continuation_action(
    value: Any,
    *,
    default: ContinuationActionType = ContinuationActionType.FINISH,
) -> ContinuationActionType:
    """Normalize continuation action values from strings or foreign enums."""

    candidate = value
    if isinstance(candidate, ContinuationActionType):
        return candidate
    if isinstance(candidate, Enum):
        candidate = candidate.value
    try:
        return ContinuationActionType(candidate)
    except ValueError:
        return default
