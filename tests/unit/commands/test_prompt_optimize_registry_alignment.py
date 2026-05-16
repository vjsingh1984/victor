"""Registry-alignment tests for /prompt-optimize."""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console

from victor.agent import prompt_section_registry as registry_module
from victor.agent.prompt_section_registry import (
    SectionCategory,
    SectionDefinition,
    UnifiedSectionRegistry,
    _initialize_default_sections,
)
from victor.framework.rl.learners.prompt_optimizer import PromptCandidate
from victor.ui.slash.commands.prompt_optimize import PromptOptimizeCommand
from victor.ui.slash.protocol import CommandContext


def test_prompt_optimize_uses_registry_for_custom_evolvable_sections(monkeypatch) -> None:
    fresh_registry = UnifiedSectionRegistry()
    _initialize_default_sections(fresh_registry)
    fresh_registry.register(
        SectionDefinition(
            name="CUSTOM_REVIEW_GUIDANCE",
            aliases={"custom_review"},
            category=SectionCategory.TASK_HINTS,
            default_text="Review API drift first.",
            evolvable=True,
            required=False,
            priority=41,
        )
    )
    monkeypatch.setattr(registry_module, "_registry", fresh_registry)

    calls: list[tuple[str, str, str]] = []

    class _Learner:
        def get_evolvable_sections(self):
            return ["CUSTOM_REVIEW_GUIDANCE"]

        def evolve(self, section, current, provider="default", query=None, on_phase=None):
            calls.append((section, current, provider))
            return PromptCandidate(
                section_name=section,
                provider=provider,
                text=current + " Evolved.",
                text_hash="hash9999cccc",
                generation=1,
                parent_hash="base0000aaaa",
            )

    stream = io.StringIO()
    console = Console(file=stream, width=120, force_terminal=False, color_system=None)
    ctx = CommandContext(console=console, settings=SimpleNamespace(), agent=None, args=["CUSTOM"])

    with patch(
        "victor.framework.rl.coordinator.get_rl_coordinator",
        return_value=SimpleNamespace(
            db_path="/tmp/victor.db",
            get_learner=lambda name: _Learner(),
        ),
    ):
        PromptOptimizeCommand().execute(ctx)

    assert calls == [("CUSTOM_REVIEW_GUIDANCE", "Review API drift first.", "default")]
    assert "CUSTOM_REVIEW_GUIDANCE" in stream.getvalue()
