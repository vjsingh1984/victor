# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for prompt-candidate reward attribution (FEP-0017)."""

from unittest.mock import MagicMock

from victor.agent.services.prompt_optimization_reward import (
    emit_prompt_candidate_outcome,
)
from victor.agent.services.runtime_intelligence import PromptOptimizationIdentity


def test_emits_one_event_per_rewardable_identity(monkeypatch):
    """One PROMPT_CANDIDATE_USED event per served identity carrying the outcome."""
    from victor.framework.rl.hooks import RLEventType

    identities = [
        PromptOptimizationIdentity(
            provider="zai",
            prompt_candidate_hash="h1",
            section_name="GROUNDING_RULES",
            prompt_section_name="GROUNDING_RULES",
        ),
        PromptOptimizationIdentity(
            provider="zai",
            prompt_candidate_hash="h2",
            section_name="COMPLETION_GUIDANCE",
            prompt_section_name="COMPLETION_GUIDANCE",
        ),
        # No hash → not rewardable, must be skipped.
        PromptOptimizationIdentity(provider="zai", section_name="X"),
    ]

    captured: list = []
    fake_hooks = MagicMock()
    fake_hooks.emit = lambda ev: captured.append(ev)
    monkeypatch.setattr("victor.framework.rl.hooks.get_rl_hooks", lambda: fake_hooks)

    emit_prompt_candidate_outcome(
        identities,
        completion_score=0.85,
        success=True,
        provider="zai",
        model="glm-5.2",
        task_type="action",
        session_id="sess-1",
    )

    assert len(captured) == 2  # only the two with hashes
    emitted_hashes = set()
    for ev in captured:
        assert ev.type is RLEventType.PROMPT_CANDIDATE_USED
        assert ev.success is True
        assert ev.quality_score == 0.85
        assert ev.metadata["session_id"] == "sess-1"
        emitted_hashes.add(ev.metadata["prompt_candidate_hash"])
    assert emitted_hashes == {"h1", "h2"}


def test_noop_without_rewardable_identities():
    """No identities (or none with a hash) → no emission, no raise."""
    emit_prompt_candidate_outcome([], completion_score=0.5, success=True)
    emit_prompt_candidate_outcome(
        [PromptOptimizationIdentity(provider="zai", section_name="X")],
        completion_score=0.5,
        success=True,
    )
