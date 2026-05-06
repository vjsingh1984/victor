"""Tests for the unified prompt section registry."""

from victor.agent import prompt_section_registry as registry_module
from victor.agent.prompt_section_registry import (
    UnifiedSectionRegistry,
    _initialize_default_sections,
    get_section_registry,
    register_prompt_contributor_sections,
)
from victor.core.verticals.protocols.prompt_provider import PromptSectionContribution


def test_registry_exposes_all_core_evolvable_sections() -> None:
    registry = get_section_registry()
    evolvable = registry.get_evolvable_sections()

    assert "ASI_TOOL_EFFECTIVENESS_GUIDANCE" in evolvable
    assert "GROUNDING_RULES" in evolvable
    assert "COMPLETION_GUIDANCE" in evolvable
    assert "CONCISE_MODE_GUIDANCE" in evolvable
    assert "PARALLEL_READ_GUIDANCE" in evolvable
    assert "LARGE_FILE_PAGINATION_GUIDANCE" in evolvable
    assert "GROUNDING_RULES_EXTENDED" in evolvable
    assert "FEW_SHOT_EXAMPLES" in evolvable
    assert "INIT_SYNTHESIS_RULES" in evolvable


def test_registry_resolves_aliases_for_new_prompt_sections() -> None:
    registry = get_section_registry()

    assert registry.get("parallel_reads").name == "PARALLEL_READ_GUIDANCE"
    assert registry.get("tool_output_grounding").name == "GROUNDING_RULES_EXTENDED"
    assert registry.get("synthesis_rules").name == "INIT_SYNTHESIS_RULES"


def test_runtime_registration_adds_named_contributor_sections(monkeypatch) -> None:
    fresh_registry = UnifiedSectionRegistry()
    _initialize_default_sections(fresh_registry)
    monkeypatch.setattr(registry_module, "_registry", fresh_registry)

    class _Contributor:
        def get_prompt_section_contributions(self):
            return [
                PromptSectionContribution(
                    name="CUSTOM_REVIEW_GUIDANCE",
                    text="Review for API drift first.",
                    aliases={"custom_review"},
                    category="task_hints",
                    evolvable=True,
                    required=False,
                    priority=42,
                )
            ]

        def get_priority(self) -> int:
            return 42

    register_prompt_contributor_sections([_Contributor()])

    registered = fresh_registry.get("CUSTOM_REVIEW_GUIDANCE")
    assert registered is not None
    assert registered.default_text == "Review for API drift first."
    assert registered.evolvable is True
    assert fresh_registry.get("custom_review").name == "CUSTOM_REVIEW_GUIDANCE"


def test_runtime_registration_falls_back_for_legacy_contributors(monkeypatch) -> None:
    fresh_registry = UnifiedSectionRegistry()
    _initialize_default_sections(fresh_registry)
    monkeypatch.setattr(registry_module, "_registry", fresh_registry)

    class _LegacyContributor:
        def get_system_prompt_section(self) -> str:
            return "Legacy contributor guidance."

        def get_priority(self) -> int:
            return 55

    register_prompt_contributor_sections([_LegacyContributor()])

    registered = fresh_registry.get("VERTICAL_LEGACYCONTRIBUTOR")
    assert registered is not None
    assert registered.default_text == "Legacy contributor guidance."
    assert registered.evolvable is False
