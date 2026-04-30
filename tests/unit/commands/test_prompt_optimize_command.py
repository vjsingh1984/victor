# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for /prompt-optimize slash command inspection helpers."""

from __future__ import annotations

import io
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console

from victor.framework.rl.learners.prompt_optimizer import PromptCandidate, PromptOptimizerLearner
from victor.ui.slash.commands.prompt_optimize import PromptOptimizeCommand
from victor.ui.slash.protocol import CommandContext


def _make_context(args):
    stream = io.StringIO()
    console = Console(file=stream, width=120, force_terminal=False, color_system=None)
    return CommandContext(console=console, settings=SimpleNamespace(), agent=None, args=args), stream


def _make_learner():
    db = sqlite3.connect(":memory:")
    learner = PromptOptimizerLearner(name="prompt_optimizer", db_connection=db)
    first = PromptCandidate(
        section_name="GROUNDING_RULES",
        provider="default",
        text="first prompt",
        text_hash="hash0001aaaa",
        generation=1,
        parent_hash="base0000aaaa",
    )
    second = PromptCandidate(
        section_name="GROUNDING_RULES",
        provider="default",
        text="second prompt",
        text_hash="hash0002bbbb",
        generation=2,
        parent_hash="hash0001aaaa",
        is_active=True,
        benchmark_passed=True,
    )
    learner._candidates[learner._candidate_key("GROUNDING_RULES", "default")] = [first, second]
    return learner


def test_prompt_optimize_show_candidate_text():
    learner = _make_learner()
    command = PromptOptimizeCommand()
    ctx, stream = _make_context(
        ["--show", "GROUNDING_RULES", "--provider", "default", "--ordinal", "2"]
    )

    with patch(
        "victor.framework.rl.coordinator.get_rl_coordinator",
        return_value=SimpleNamespace(
            db_path="/tmp/victor.db",
            get_learner=lambda name: learner,
        ),
    ):
        command.execute(ctx)

    output = stream.getvalue()
    assert "second prompt" in output
    assert "GROUNDING_RULES" in output
    assert "hash0002" in output


def test_prompt_optimize_diff_candidates():
    learner = _make_learner()
    command = PromptOptimizeCommand()
    ctx, stream = _make_context(
        ["--diff", "GROUNDING_RULES", "--provider", "default", "--from", "1", "--to", "2"]
    )

    with patch(
        "victor.framework.rl.coordinator.get_rl_coordinator",
        return_value=SimpleNamespace(
            db_path="/tmp/victor.db",
            get_learner=lambda name: learner,
        ),
    ):
        command.execute(ctx)

    output = stream.getvalue()
    assert "--- GROUNDING_RULES:default:1" in output
    assert "+++ GROUNDING_RULES:default:2" in output
    assert "-first prompt" in output
    assert "+second prompt" in output
