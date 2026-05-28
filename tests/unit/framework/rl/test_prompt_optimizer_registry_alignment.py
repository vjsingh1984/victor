"""Registry alignment tests for the prompt optimizer learner."""

from __future__ import annotations

from victor.agent.prompt_section_registry import get_section_registry
from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner


def test_prompt_optimizer_uses_registry_order_for_evolvable_sections() -> None:
    registry = get_section_registry()
    expected = [
        section.name
        for section in sorted(
            (section for section in registry.get_all() if section.evolvable),
            key=lambda section: (section.priority, section.name),
        )
    ]

    assert PromptOptimizerLearner.get_evolvable_sections() == expected


def test_prompt_optimizer_class_override_still_supported(monkeypatch) -> None:
    monkeypatch.setattr(
        PromptOptimizerLearner, "EVOLVABLE_SECTIONS", ["CUSTOM_SECTION"]
    )

    assert PromptOptimizerLearner.get_evolvable_sections() == ["CUSTOM_SECTION"]
